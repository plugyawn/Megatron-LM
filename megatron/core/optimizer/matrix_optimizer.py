# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron matrix optimizer integration for composable matrix update rules.

``matrix_optimizer`` selects the matrix update rule for eligible affine weights.
``matrix_input_preconditioner=feature_gram`` optionally right-preconditions that
rule with ``C_in = X.T @ X``. ``matrix_output_preconditioner=grad_gram``
optionally left-preconditions it with ``C_out = dY.T @ dY``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from dataclasses import fields, replace
from typing import Dict, List, Optional

import torch

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None

from megatron.core.matrix_update import (
    ExtraWgradFactor,
    MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
    TPUpdateMode,
    MatrixInputPreconditionerRecipe,
    MatrixOutputPreconditionerRecipe,
    MatrixPreconditionerApproximation,
    MatrixPreconditionerKind,
    MatrixPreconditionerNormalization,
    MatrixPreconditionerScope,
    configure_matrix_update_param,
    configure_model_matrix_updates,
    get_matrix_optimizer_info,
    get_matrix_shard_spec,
    matrix_update_family_from_optimizer_name,
    matrix_small_gram_side_for_spec,
    set_matrix_sidecar_owner,
    set_feature_gram_finalization_required,
    set_grad_gram_finalization_required,
    update_matrix_shard_spec,
)
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank

from . import (
    _get_megatron_optimizer_based_on_param_groups,
    _get_param_groups,
    _model_chunks_use_distributed_optimizer_buffers,
    _setup_layerwise_fallback_distopt_routing,
)
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .matrix_function_optimizer import MatrixFunctionOptimizer
from .optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params, FP32Optimizer, MegatronOptimizer
from .optimizer_config import OptimizerConfig, ParamKey

try:
    from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
        fully_shard_optimizer as megatron_fsdp_fully_shard_optimizer,
    )
except ImportError:
    megatron_fsdp_fully_shard_optimizer = None

try:
    from emerging_optimizers.matrix_tp_apply import (
        tp_allgather_logical_matrix_update,
        tp_block_local_approx,
        tp_small_gram_newton_schulz_allreduce,
    )
    from emerging_optimizers.matrix_update_rules import (
        apply_diag_newton_muon_update_,
        apply_diag_left_preconditioned_update_,
        apply_diag_right_preconditioned_update_,
        apply_diag_two_sided_preconditioned_update_,
        factorize_feature_gram,
        factorize_grad_gram,
        newton_schulz_orthogonalize,
        right_precondition_with_factorized_feature_gram,
    )
    from emerging_optimizers.utils import fp32_matmul_precision

    HAVE_EMERGING_MATRIX_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_MATRIX_OPTIMIZERS = False


logger = logging.getLogger(__name__)


def _matrix_optimizer_model_parallel_group(pg_collection: ProcessGroupCollection):
    """Return the optimizer stats group matching Megatron's standard dense path."""

    return getattr(pg_collection, "mp", getattr(pg_collection, "tp", None))


def _matrix_optimizer_tp_group(pg_collection: ProcessGroupCollection):
    return getattr(pg_collection, "tp", None)


def _matrix_optimizer_fsdp_stats_group(pg_collection: ProcessGroupCollection):
    return (
        getattr(pg_collection, "tp_dp_cp", None)
        or getattr(pg_collection, "dp_cp", None)
        or getattr(pg_collection, "dp", None)
        or _matrix_optimizer_model_parallel_group(pg_collection)
    )


def _model_chunks_use_megatron_fsdp(model_chunks: List[MegatronModule]) -> bool:
    for model_chunk in model_chunks:
        if model_chunk.__class__.__name__ == "MegatronFSDP":
            return True
        ddp_config = getattr(model_chunk, "ddp_config", None)
        if bool(getattr(ddp_config, "use_megatron_fsdp", False)):
            return True
        if hasattr(model_chunk, "parameters"):
            params = model_chunk.parameters()
        elif hasattr(model_chunk, "named_parameters"):
            params = (param for _, param in model_chunk.named_parameters())
        else:
            params = ()
        if any(getattr(param, "_megatron_fsdp_model", None) is not None for param in params):
            return True
    return False


def _set_matrix_optimizer_process_groups(
    optimizer: MegatronOptimizer,
    pg_collection: ProcessGroupCollection,
) -> None:
    """Attach optimizer-local reduction groups instead of relying on global MPU state."""

    model_parallel_group = _matrix_optimizer_model_parallel_group(pg_collection)
    tp_group = _matrix_optimizer_tp_group(pg_collection)
    for child in getattr(optimizer, "chained_optimizers", [optimizer]):
        setattr(child, "grad_stats_parallel_group", model_parallel_group)
        setattr(child, "tp_group", tp_group)


def _tp_mode_from_config(config: OptimizerConfig) -> TPUpdateMode:
    return {
        "allgather": TPUpdateMode.TP_ALLGATHER_LOGICAL_MATRIX,
        "small_gram_ns": TPUpdateMode.TP_SMALL_GRAM_NS_ALLREDUCE,
        "block_local": TPUpdateMode.TP_BLOCK_LOCAL_APPROX,
    }[config.matrix_tp_update_mode]


class MegatronFSDPOptimizer(MegatronOptimizer):
    """Megatron optimizer adapter for FSDP-hooked torch optimizers."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(optimizer, config, lambda opt, config=None: None)
        self.grad_stats_parallel_group = _matrix_optimizer_fsdp_stats_group(pg_collection)
        self.tp_group = _matrix_optimizer_tp_group(pg_collection)

    def _param_name(self, param: torch.nn.Parameter) -> str:
        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
            _matrix_optimizer_param_checkpoint_identity,
            _megatron_fsdp_model_from_optimizer_params,
            _optimizer_param_state_indices,
        )

        mfsdp_model = _megatron_fsdp_model_from_optimizer_params(self.optimizer)
        if mfsdp_model is None:
            raise RuntimeError(
                "Megatron-FSDP optimizer checkpointing requires optimizer parameters "
                "to carry MegatronFSDP ownership metadata."
            )
        param_state_indices = _optimizer_param_state_indices(self.optimizer)
        param_idx = str(param_state_indices.get(id(param), "unknown"))
        param_name = _matrix_optimizer_param_checkpoint_identity(param, mfsdp_model)
        if param_name is None:
            raise RuntimeError(
                "Megatron-FSDP optimizer checkpointing requires a stable parameter "
                f"identity for optimizer state index {param_idx}."
            )
        return param_name

    def _param_groups_to_param2group_meta(self, param_groups: list[dict]) -> dict:
        param_to_group_meta = {}
        for group in param_groups:
            group_meta = group.copy()
            del group_meta["params"]
            for param in group["params"]:
                param_to_group_meta[self._param_name(param)] = group_meta
        return param_to_group_meta

    def _param2group_meta_to_param_groups(
        self,
        param_to_group_meta: dict,
        param_groups: list[dict],
        strict: bool = True,
    ) -> list[dict]:
        new_param_groups = []
        for group in param_groups:
            new_group = {"params": []}
            for param in group["params"]:
                param_name = self._param_name(param)
                if param_name not in param_to_group_meta:
                    if strict:
                        raise ValueError(
                            f"Parameter {param_name} not found in param_to_group_meta mapping."
                        )
                    continue
                group_meta = param_to_group_meta[param_name]
                new_group_wo_params = new_group.copy()
                del new_group_wo_params["params"]
                if new_group_wo_params and new_group_wo_params != group_meta:
                    error_info = (
                        f"Parameter {param_name} and the parameters in the same group "
                        f"{new_group['params']} have different metadata. Please check "
                        "whether the checkpoint and current param_groups match. "
                        f"Parameter {param_name} has metadata {group_meta}, "
                        f"while other group metadata is {new_group}."
                    )
                    if strict:
                        raise ValueError(error_info)
                    logger.warning(error_info)
                    continue
                new_group["params"].append(param_name)
                new_group.update(group_meta)
            new_param_groups.append(new_group)
        return new_param_groups

    def _param_state_index_to_name(self) -> dict[int, str]:
        index_to_name = {}
        seen_param_ids = {}
        state_idx = 0
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                param_id = id(param)
                if param_id not in seen_param_ids:
                    seen_param_ids[param_id] = state_idx
                    index_to_name[state_idx] = self._param_name(param)
                    state_idx += 1
        return index_to_name

    def _optimizer_state_dtensor_keys(self) -> list[str]:
        local_keys = set()
        for param_state in self.optimizer.state.values():
            if not isinstance(param_state, dict):
                continue
            for state_key, state_value in param_state.items():
                if DTensor is not None and isinstance(state_value, DTensor):
                    local_keys.add(state_key)
        if torch.distributed.is_initialized():
            gathered_keys = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_keys, sorted(local_keys))
            local_keys = set()
            for rank_keys in gathered_keys:
                local_keys.update(rank_keys or [])
        return sorted(local_keys)

    def _optimizer_state_step_value(self) -> Optional[float]:
        local_step = None
        for param_state in self.optimizer.state.values():
            if not isinstance(param_state, dict) or "step" not in param_state:
                continue
            step = param_state["step"]
            if torch.is_tensor(step) and step.numel() == 1:
                local_step = float(step.detach().cpu().item())
                break
            if isinstance(step, (int, float)):
                local_step = float(step)
                break
        if torch.distributed.is_initialized():
            gathered_steps = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_steps, local_step)
            for step in gathered_steps:
                if step is not None:
                    return float(step)
        return local_step

    def _empty_dtensor_state_like_param(
        self, param: torch.Tensor, *, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        local_param = self._local_param_tensor(param)
        state_dtype = dtype if dtype is not None else local_param.dtype
        return DTensor.from_local(
            local_tensor=torch.empty(0, dtype=state_dtype, device=local_param.device),
            device_mesh=param.device_mesh,
            placements=param.placements,
            shape=param.shape,
            stride=param.stride(),
        )

    def _save_state_placeholders_for_param(
        self, param: torch.Tensor, group: dict, step_value: Optional[float]
    ) -> dict:
        if DTensor is None or not isinstance(param, DTensor):
            return {}
        placeholders = {}
        if "betas" in group:
            placeholders["exp_avg"] = self._empty_dtensor_state_like_param(
                param, dtype=self.config.exp_avg_dtype
            )
            placeholders["exp_avg_sq"] = self._empty_dtensor_state_like_param(
                param, dtype=self.config.exp_avg_sq_dtype
            )
            if group.get("amsgrad", False):
                placeholders["max_exp_avg_sq"] = self._empty_dtensor_state_like_param(
                    param, dtype=self.config.exp_avg_sq_dtype
                )
            if step_value is not None:
                local_param = self._local_param_tensor(param)
                placeholders["step"] = torch.tensor(
                    step_value, dtype=torch.float32, device=local_param.device
                )
        elif float(group.get("momentum", 0.0) or 0.0) != 0.0:
            placeholders["momentum_buffer"] = self._empty_dtensor_state_like_param(param)
        return placeholders

    def _add_empty_state_for_missing_params(
        self, packed_state: dict, step_value: Optional[float]
    ) -> None:
        if DTensor is None:
            return
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if not isinstance(param, DTensor):
                    continue
                param_name = self._param_name(param)
                param_state = packed_state.setdefault(param_name, {})
                if not isinstance(param_state, dict):
                    continue
                for state_key, placeholder in self._save_state_placeholders_for_param(
                    param, group, step_value
                ).items():
                    param_state.setdefault(state_key, placeholder)

    def _local_param_tensor(self, param: torch.Tensor) -> torch.Tensor:
        if DTensor is not None and isinstance(param, DTensor):
            return param.to_local()
        return param

    def _state_tensor_placeholder_like_param(
        self, param: torch.Tensor, *, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        local_param = self._local_param_tensor(param)
        state_dtype = dtype if dtype is not None else local_param.dtype
        local_placeholder = torch.empty_strided(
            tuple(local_param.shape),
            tuple(local_param.stride()),
            dtype=state_dtype,
            device=local_param.device,
        )
        if DTensor is not None and isinstance(param, DTensor):
            placeholder = DTensor.from_local(
                local_tensor=local_placeholder,
                device_mesh=param.device_mesh,
                placements=param.placements,
                shape=param.shape,
                stride=param.stride(),
            )
        else:
            placeholder = local_placeholder
        matrix_shard_spec = get_matrix_shard_spec(param)
        if matrix_shard_spec is not None:
            update_matrix_shard_spec(placeholder, matrix_shard_spec)
        return placeholder

    def _scalar_step_placeholder(self, param: torch.Tensor) -> torch.Tensor:
        local_param = self._local_param_tensor(param)
        return torch.tensor(0.0, dtype=torch.float32, device=local_param.device)

    def _load_state_placeholders_for_param(self, param: torch.Tensor, group: dict) -> dict:
        placeholders = {}
        if "betas" in group:
            placeholders["step"] = self._scalar_step_placeholder(param)
            placeholders["exp_avg"] = self._state_tensor_placeholder_like_param(
                param, dtype=self.config.exp_avg_dtype
            )
            placeholders["exp_avg_sq"] = self._state_tensor_placeholder_like_param(
                param, dtype=self.config.exp_avg_sq_dtype
            )
            if group.get("amsgrad", False):
                placeholders["max_exp_avg_sq"] = self._state_tensor_placeholder_like_param(
                    param, dtype=self.config.exp_avg_sq_dtype
                )
        elif float(group.get("momentum", 0.0) or 0.0) != 0.0:
            placeholders["momentum_buffer"] = self._state_tensor_placeholder_like_param(param)
        return placeholders

    def _add_load_state_placeholders(self, packed_state: dict) -> None:
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                param_state = packed_state.setdefault(self._param_name(param), {})
                if not isinstance(param_state, dict):
                    continue
                for state_name, state_value in self._load_state_placeholders_for_param(
                    param, group
                ).items():
                    param_state.setdefault(state_name, state_value)

    def prepare_grads(self) -> bool:
        return False

    def step_with_ready_grads(self) -> bool:
        self.optimizer.step()
        return True

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_loss_scale(self) -> torch.Tensor:
        params = self.get_parameters()
        if params:
            device = params[0].device
        elif torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        return torch.tensor([1.0], dtype=torch.float32, device=device)

    def reload_model_params(self, state_dict=None):
        return None

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if "param_to_group_meta" in state_dict:
            from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
                _matrix_optimizer_checkpoint_state_dict_for_validation,
                _megatron_fsdp_model_from_optimizer_params,
                _validate_matrix_optimizer_checkpoint_metadata,
            )

            state_dict = state_dict.copy()
            mfsdp_model = _megatron_fsdp_model_from_optimizer_params(self.optimizer)
            if mfsdp_model is not None:
                _validate_matrix_optimizer_checkpoint_metadata(
                    self.optimizer,
                    mfsdp_model,
                    _matrix_optimizer_checkpoint_state_dict_for_validation(
                        self.optimizer, state_dict, self._param_name
                    ),
                )
            state_dict["param_groups"] = self._param2group_meta_to_param_groups(
                state_dict["param_to_group_meta"], self.optimizer.param_groups
            )
            del state_dict["param_to_group_meta"]
        self.optimizer.load_state_dict(state_dict)

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        return True, None, None

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        state_dict = self.optimizer.state_dict()
        index_to_name = self._param_state_index_to_name()
        step_value = self._optimizer_state_step_value()
        packed_state = {}
        for param_idx, param_state in state_dict.get("state", {}).items():
            if isinstance(param_idx, int) and param_idx in index_to_name:
                packed_state[index_to_name[param_idx]] = param_state
            elif isinstance(param_idx, str) and param_idx.isdigit() and int(param_idx) in index_to_name:
                packed_state[index_to_name[int(param_idx)]] = param_state
            else:
                packed_state[param_idx] = param_state
        if is_loading:
            self._add_load_state_placeholders(packed_state)
        else:
            self._add_empty_state_for_missing_params(packed_state, step_value)

        sharded_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("state", "param_groups")
        }
        sharded_state_dict["state"] = packed_state
        sharded_state_dict["param_to_group_meta"] = self._param_groups_to_param2group_meta(
            self.optimizer.param_groups
        )

        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
            _add_matrix_optimizer_checkpoint_metadata,
            _megatron_fsdp_model_from_optimizer_params,
        )

        mfsdp_model = _megatron_fsdp_model_from_optimizer_params(self.optimizer)
        if mfsdp_model is not None:
            _add_matrix_optimizer_checkpoint_metadata(
                self.optimizer, mfsdp_model, sharded_state_dict
            )
        return sharded_state_dict


class _ChainedTorchOptimizer(torch.optim.Optimizer):
    """Torch optimizer facade that lets Megatron-FSDP wrap one owner-level step.

    Megatron-FSDP optimizer hooks are owner-wide: ``step()`` may synchronize every
    gradient for the owning FSDP module and install every optimized model weight.
    Matrix/fallback splits therefore must be chained below a single FSDP-hooked
    torch optimizer per owner rather than exposed as separate FSDP optimizer
    children.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if not optimizers:
            raise ValueError("_ChainedTorchOptimizer requires at least one child optimizer.")
        self.optimizers = optimizers
        self._child_group_counts = [len(optimizer.param_groups) for optimizer in optimizers]
        param_groups = [
            param_group
            for optimizer in optimizers
            for param_group in optimizer.param_groups
        ]
        super().__init__(param_groups, defaults={})
        self.param_groups = param_groups
        self._refresh_state_from_children()

    def _child_optimizer_kind(self, optimizer: torch.optim.Optimizer) -> str:
        if isinstance(optimizer, MatrixFunctionOptimizer):
            return "matrix_function"
        return "fallback"

    def _refresh_state_from_children(self) -> None:
        self.state = defaultdict(dict)
        for optimizer in self.optimizers:
            for param, param_state in getattr(optimizer, "state", {}).items():
                self.state[param] = param_state

    def _sync_children_from_self(self) -> None:
        cursor = 0
        for optimizer, group_count in zip(self.optimizers, self._child_group_counts):
            optimizer.param_groups = self.param_groups[cursor : cursor + group_count]
            cursor += group_count
            optimizer.state = defaultdict(dict)
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    if param in self.state:
                        optimizer.state[param] = self.state[param]

    def step(self, *args, **kwargs):
        loss = None
        for optimizer in self.optimizers:
            child_loss = optimizer.step(*args, **kwargs)
            if loss is None:
                loss = child_loss
        self._refresh_state_from_children()
        return loss

    def zero_grad(self, *args, **kwargs):
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        self._refresh_state_from_children()
        state_dict = super().state_dict()
        child_extra_states = []
        matrix_steps = []
        for optimizer in self.optimizers:
            child_state_dict = optimizer.state_dict()
            child_extra_state = {
                key: value
                for key, value in child_state_dict.items()
                if key not in ("state", "param_groups")
            }
            child_extra_state["child_kind"] = self._child_optimizer_kind(optimizer)
            if "matrix_step" in child_extra_state:
                matrix_steps.append(int(child_extra_state["matrix_step"]))
            child_extra_states.append(child_extra_state)
        state_dict["_child_optimizer_extra_state"] = child_extra_states
        if matrix_steps:
            if len(set(matrix_steps)) != 1:
                raise RuntimeError(
                    "Chained matrix optimizer children have inconsistent matrix_step values."
                )
            state_dict["matrix_step"] = matrix_steps[0]
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._sync_children_from_self()
        child_extra_states = state_dict.get("_child_optimizer_extra_state")
        if child_extra_states is None:
            child_extra_states = [{} for _ in self.optimizers]
            if "matrix_step" in state_dict:
                for child_extra_state, optimizer in zip(child_extra_states, self.optimizers):
                    optimizer_state_dict = optimizer.state_dict()
                    if "matrix_step" in optimizer_state_dict or hasattr(optimizer, "_matrix_step"):
                        child_extra_state["matrix_step"] = state_dict["matrix_step"]
                        break
        if len(child_extra_states) != len(self.optimizers):
            raise ValueError(
                "Chained optimizer checkpoint child state count does not match optimizer count: "
                f"checkpoint={len(child_extra_states)}, optimizer={len(self.optimizers)}."
            )
        for optimizer, child_extra_state in zip(self.optimizers, child_extra_states):
            child_extra_state = dict(child_extra_state)
            checkpoint_child_kind = child_extra_state.pop("child_kind", None)
            current_child_kind = self._child_optimizer_kind(optimizer)
            if checkpoint_child_kind is not None and checkpoint_child_kind != current_child_kind:
                raise ValueError(
                    "Chained optimizer checkpoint child kind does not match optimizer order: "
                    f"checkpoint={checkpoint_child_kind}, optimizer={current_child_kind}."
                )
            if not child_extra_state:
                continue
            child_state_dict = optimizer.state_dict()
            child_state_dict.update(child_extra_state)
            optimizer.load_state_dict(child_state_dict)
        self._refresh_state_from_children()


def _muon_scale_factor(size_out: int, size_in: int, config: OptimizerConfig) -> float:
    mode = config.muon_scale_mode
    if mode == "none":
        return 1.0
    if mode == "shape_scaling":
        return max(1.0, size_out / size_in) ** 0.5
    if mode == "spectral":
        return max(size_out, size_in) ** 0.5
    if mode == "unit_rms_norm":
        return (size_out / size_in) ** 0.5
    raise ValueError(f"Invalid muon_scale_mode: {mode}")


def _muon_scale_shape(
    grad: torch.Tensor,
    param: torch.nn.Parameter,
    tp_update_mode: Optional[TPUpdateMode] = None,
) -> tuple[int, int]:
    """Return the logical matrix shape used for Muon scaling."""

    matrix_shard_spec = get_matrix_shard_spec(param)
    if matrix_shard_spec is not None and (
        matrix_shard_spec.dp_shard_axis is not None
        or tp_update_mode
        in (
            TPUpdateMode.TP_ALLGATHER_LOGICAL_MATRIX,
            TPUpdateMode.TP_SMALL_GRAM_NS_ALLREDUCE,
        )
    ):
        return matrix_shard_spec.logical_shape
    return (grad.size(-2), grad.size(-1))


def _fsdp_small_gram_tp_layout(matrix_shard_spec) -> str:
    """Map a matrix-axis FSDP shard contract to the EO small-Gram layout name."""

    if matrix_shard_spec.dp_shard_axis not in (0, 1):
        raise RuntimeError(
            "Matrix FSDP small-Gram Muon requires a MatrixShardSpec with dp_shard_axis."
        )
    small_gram_side = matrix_small_gram_side_for_spec(matrix_shard_spec)
    if small_gram_side == "right":
        return "column_parallel"
    if small_gram_side == "left":
        return "row_parallel"
    raise RuntimeError(f"Unsupported MatrixShardSpec small_gram_side: {small_gram_side}")


def _fsdp_small_gram_process_group(matrix_shard_spec, pg_collection: ProcessGroupCollection):
    """Return the process group that spans all matrix-axis shards for FSDP Muon."""

    if matrix_shard_spec.tp_shard_axis is not None:
        process_group = getattr(pg_collection, "tp_dp_cp", None)
        if process_group is None:
            raise RuntimeError(
                "Matrix FSDP Muon with both TP and DP matrix sharding requires "
                "pg_collection.tp_dp_cp so the small Gram is reduced over every "
                "matrix-axis shard."
            )
        return process_group
    process_group = getattr(pg_collection, "dp_cp", None)
    if process_group is None:
        process_group = getattr(pg_collection, "dp", None)
    if process_group is None:
        raise RuntimeError(
            "Matrix FSDP Muon requires pg_collection.dp_cp or pg_collection.dp for "
            "small-Gram all-reduce."
        )
    return process_group


def _fsdp_small_gram_newton_schulz_allreduce(
    local_matrix: torch.Tensor,
    matrix_shard_spec,
    *,
    group: torch.distributed.ProcessGroup,
    config: OptimizerConfig,
) -> torch.Tensor:
    """Apply exact small-Gram Muon to a matrix-axis FSDP local shard."""

    dtensor_template = None
    local_matrix_for_update = local_matrix
    if DTensor is not None and isinstance(local_matrix, DTensor):
        dtensor_template = local_matrix
        local_matrix_for_update = local_matrix.to_local()

    if local_matrix_for_update.ndim != 2:
        raise RuntimeError(
            "Matrix FSDP Muon requires a 2D local matrix shard, got "
            f"{tuple(local_matrix_for_update.shape)}."
        )
    if tuple(local_matrix_for_update.shape) != tuple(matrix_shard_spec.local_shape):
        raise RuntimeError(
            "Matrix FSDP Muon local shard shape does not match MatrixShardSpec: "
            f"local_matrix={tuple(local_matrix_for_update.shape)}, "
            f"spec.local_shape={matrix_shard_spec.local_shape}."
        )
    tp_layout = _fsdp_small_gram_tp_layout(matrix_shard_spec)
    local_update = tp_small_gram_newton_schulz_allreduce(
        local_matrix_for_update,
        tp_layout=tp_layout,
        group=group,
        logical_shape=matrix_shard_spec.logical_shape,
        steps=config.muon_num_ns_steps,
        coefficient_type=config.muon_coefficient_type,
        use_syrk=False,
    )
    if dtensor_template is not None:
        return DTensor.from_local(
            local_update,
            device_mesh=dtensor_template.device_mesh,
            placements=dtensor_template.placements,
            run_check=False,
            shape=dtensor_template.shape,
            stride=tuple(dtensor_template.stride()),
        )
    return local_update


def _dtensor_from_local_like(local_tensor: torch.Tensor, dtensor_template) -> torch.Tensor:
    return DTensor.from_local(
        local_tensor,
        device_mesh=dtensor_template.device_mesh,
        placements=dtensor_template.placements,
        run_check=False,
        shape=dtensor_template.shape,
        stride=tuple(dtensor_template.stride()),
    )


def _copy_matrix_model_refs_to_main_params(optimizer: MegatronOptimizer) -> None:
    if not isinstance(optimizer, Float16OptimizerWithFloat16Params):
        return
    for model_group, main_group in zip(optimizer.float16_groups, optimizer.fp32_from_float16_groups):
        for model_param, main_param in zip(model_group, main_group):
            main_param._matrix_update_model_param = model_param


def _make_matrix_update_rule(config: OptimizerConfig, pg_collection: ProcessGroupCollection):
    tp_update_mode = _tp_mode_from_config(config)
    factor_cache = {}

    def orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
        return newton_schulz_orthogonalize(
            matrix,
            steps=config.muon_num_ns_steps,
            coefficient_type=config.muon_coefficient_type,
            use_syrk=False,
        )

    def get_cached_factorization(
        param: torch.nn.Parameter, gram: torch.Tensor, *, side: str, ridge: float
    ):
        generation_attr = "_feature_gram_generation" if side == "input" else "_grad_gram_generation"
        generation = getattr(param, generation_attr, None)
        cache_key = (
            side,
            generation,
            tuple(gram.shape),
            gram.dtype,
            gram.device,
            ridge,
        )
        cached = factor_cache.get((id(param), side))
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        factorizer = factorize_feature_gram if side == "input" else factorize_grad_gram
        factorization = factorizer(
            gram, ridge=ridge
        )
        factor_cache[(id(param), side)] = (cache_key, factorization)
        return factorization

    def update_rule(
        grad: torch.Tensor,
        feature_gram: Optional[torch.Tensor],
        grad_gram: Optional[torch.Tensor],
        param: torch.nn.Parameter,
    ):
        info = getattr(param, "_mcore_linear_weight_info", None)
        matrix_shard_spec = get_matrix_shard_spec(param)
        if info is None and matrix_shard_spec is None:
            raise RuntimeError("Matrix optimizer parameter is missing LinearWeightInfo.")

        direction = grad
        sidecar_preconditioner_requested = (
            config.matrix_output_preconditioner == "grad_gram"
            or config.matrix_input_preconditioner == "feature_gram"
        )
        dtensor_template = None
        if (
            sidecar_preconditioner_requested
            and DTensor is not None
            and isinstance(direction, DTensor)
        ):
            dtensor_template = direction
            direction = direction.to_local()
        if config.matrix_output_preconditioner == "grad_gram":
            if grad_gram is None:
                raise RuntimeError("grad_gram output preconditioner requested but not collected.")
            factorization = get_cached_factorization(
                param,
                grad_gram,
                side="output",
                ridge=config.matrix_output_preconditioner_ridge,
            )
            direction = factorization.right_solve(direction.mT).mT

        if config.matrix_input_preconditioner == "feature_gram":
            if feature_gram is None:
                raise RuntimeError("feature_gram input preconditioner requested but not collected.")
            factorization = get_cached_factorization(
                param,
                feature_gram,
                side="input",
                ridge=config.matrix_input_preconditioner_ridge,
            )
            direction = right_precondition_with_factorized_feature_gram(direction, factorization)

        if config.matrix_optimizer == "sgd":
            if dtensor_template is not None:
                direction = _dtensor_from_local_like(direction, dtensor_template)
            return -direction
        if config.matrix_optimizer != "muon":
            raise RuntimeError(f"Unsupported matrix optimizer: {config.matrix_optimizer}")

        preconditioned = direction.to(torch.float32)
        tp_layout = info.tp_layout if info is not None else matrix_shard_spec.tp_layout
        with fp32_matmul_precision(config.muon_fp32_matmul_prec):
            if matrix_shard_spec is not None and matrix_shard_spec.dp_shard_axis is not None:
                update = _fsdp_small_gram_newton_schulz_allreduce(
                    preconditioned,
                    matrix_shard_spec,
                    group=_fsdp_small_gram_process_group(matrix_shard_spec, pg_collection),
                    config=config,
                )
                if dtensor_template is not None and not isinstance(update, DTensor):
                    update = _dtensor_from_local_like(update, dtensor_template)
            elif tp_update_mode == TPUpdateMode.TP_ALLGATHER_LOGICAL_MATRIX:
                update = tp_allgather_logical_matrix_update(
                    preconditioned,
                    orthogonalize,
                    tp_layout=tp_layout,
                    group=pg_collection.tp,
                )
            elif tp_update_mode == TPUpdateMode.TP_SMALL_GRAM_NS_ALLREDUCE:
                update = tp_small_gram_newton_schulz_allreduce(
                    preconditioned,
                    tp_layout=tp_layout,
                    group=pg_collection.tp,
                    logical_shape=info.logical_shape,
                    steps=config.muon_num_ns_steps,
                    coefficient_type=config.muon_coefficient_type,
                    use_syrk=False,
                )
            elif tp_update_mode == TPUpdateMode.TP_BLOCK_LOCAL_APPROX:
                update = tp_block_local_approx(
                    preconditioned,
                    orthogonalize,
                    approximation_label="tp_block_local_matrix_update",
                )
            else:
                raise RuntimeError(f"Unsupported TP update mode: {tp_update_mode}")
        scale_rows, scale_cols = _muon_scale_shape(grad, param, tp_update_mode)
        scale = _muon_scale_factor(scale_rows, scale_cols, config)
        return -update * scale * config.muon_extra_scale_factor

    return update_rule


def _make_matrix_inplace_update_rule(
    config: OptimizerConfig, pg_collection: ProcessGroupCollection
):
    """Return an optional fused in-place update rule for cheap diagonal cases.

    For SGD-style matrix updates, diagonal input, output, and two-sided
    preconditioners are eligible for the EO fused in-place/Triton helpers. For
    Muon, only the diagonal input-side FEATURE_GRAM path uses the fast diagonal
    helper today: diagonal preconditioning and parameter application are
    in-place, while Polar Express/NS remains the shared Muon implementation.
    Output-side or two-sided Muon preconditioning intentionally falls back to
    the generic update rule. Full and block-diagonal Grams also fall back to the
    generic rule, where the cached factorized solve path preserves the
    supported math without implying a Triton Gram-solve implementation.
    """

    if config.matrix_optimizer not in ("sgd", "muon"):
        return None
    input_requested = config.matrix_input_preconditioner == "feature_gram"
    output_requested = config.matrix_output_preconditioner == "grad_gram"
    if not input_requested and not output_requested:
        return None
    if config.matrix_input_preconditioner not in ("none", "feature_gram"):
        return None
    if config.matrix_output_preconditioner not in ("none", "grad_gram"):
        return None
    tp_update_mode = _tp_mode_from_config(config)

    def can_use_local_newton_muon(model_param: torch.nn.Parameter) -> bool:
        if tp_update_mode == TPUpdateMode.TP_BLOCK_LOCAL_APPROX:
            return True
        info = getattr(model_param, "_mcore_linear_weight_info", None)
        if info is None:
            return False
        if info.tp_layout in ("none", "duplicated"):
            return True
        return get_pg_size(pg_collection.tp) == 1

    def inplace_update_rule(
        param: torch.nn.Parameter,
        grad: torch.Tensor,
        feature_gram: Optional[torch.Tensor],
        grad_gram: Optional[torch.Tensor],
        model_param: torch.nn.Parameter,
        lr: float,
        weight_decay: float,
        decoupled_weight_decay: bool,
    ) -> bool:
        if input_requested and (feature_gram is None or feature_gram.ndim != 1):
            return False
        if output_requested and (grad_gram is None or grad_gram.ndim != 1):
            return False
        if config.matrix_optimizer == "sgd":
            if input_requested and output_requested:
                apply_diag_two_sided_preconditioned_update_(
                    param,
                    grad,
                    grad_gram,
                    feature_gram,
                    lr=lr,
                    ridge_left=config.matrix_output_preconditioner_ridge,
                    ridge_right=config.matrix_input_preconditioner_ridge,
                    update_scale=1.0,
                    weight_decay=weight_decay,
                    decoupled_weight_decay=decoupled_weight_decay,
                )
            elif output_requested:
                apply_diag_left_preconditioned_update_(
                    param,
                    grad,
                    grad_gram,
                    lr=lr,
                    ridge=config.matrix_output_preconditioner_ridge,
                    update_scale=1.0,
                    weight_decay=weight_decay,
                    decoupled_weight_decay=decoupled_weight_decay,
                )
            else:
                apply_diag_right_preconditioned_update_(
                    param,
                    grad,
                    feature_gram,
                    lr=lr,
                    ridge=config.matrix_input_preconditioner_ridge,
                    update_scale=1.0,
                    weight_decay=weight_decay,
                    decoupled_weight_decay=decoupled_weight_decay,
                )
            return True
        if output_requested:
            return False
        if not can_use_local_newton_muon(model_param):
            return False
        apply_diag_newton_muon_update_(
            param,
            grad,
            feature_gram,
            lr=lr,
            ridge=config.matrix_input_preconditioner_ridge,
            num_ns_steps=config.muon_num_ns_steps,
            coefficient_type=config.muon_coefficient_type,
            scale_mode=config.muon_scale_mode,
            extra_scale_factor=config.muon_extra_scale_factor,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            fp32_matmul_prec=config.muon_fp32_matmul_prec,
            use_syrk=False,
        )
        return True

    return inplace_update_rule


def _configured_fsdp_matrix_params(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
) -> list[torch.nn.Parameter]:
    _validate_fsdp_matrix_sidecar_config(config)
    expected_update_family = matrix_update_family_from_optimizer_name(config.matrix_optimizer)
    configured = []
    for model_chunk in model_chunks:
        for param in model_chunk.parameters():
            info = get_matrix_optimizer_info(param)
            if info is None:
                continue
            if (
                info.owner != MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION
                or info.update_family != expected_update_family
            ):
                continue
            if get_matrix_shard_spec(param) is None:
                raise RuntimeError(
                    "Megatron-FSDP matrix optimizer parameter is missing MatrixShardSpec "
                    "metadata after FSDP wrapping."
                )
            _configure_fsdp_diag_sidecars(param, config)
            configured.append(param)
    return configured


def _validate_fsdp_matrix_sidecar_config(config: OptimizerConfig) -> None:
    """Allow only transient diagonal sidecars under Megatron-FSDP v1."""

    if config.matrix_input_preconditioner == "feature_gram":
        if config.matrix_input_preconditioner_approximation != "diag":
            raise RuntimeError(
                "Megatron-FSDP FEATURE_GRAM sidecars support diag approximation only; "
                "full and block_diag require explicit sidecar sharding/checkpoint contracts."
            )
        if config.matrix_input_preconditioner_refresh_interval != 1:
            raise RuntimeError(
                "Megatron-FSDP FEATURE_GRAM sidecars require refresh_interval=1; cached "
                "sidecar checkpoint/resume semantics are not implemented yet."
            )
        if config.matrix_input_preconditioner_ema_beta is not None:
            raise RuntimeError(
                "Megatron-FSDP FEATURE_GRAM sidecars do not support EMA state yet."
            )
    elif config.matrix_input_preconditioner != "none":
        raise RuntimeError("Megatron-FSDP matrix input sidecar must be none or feature_gram.")

    if config.matrix_output_preconditioner == "grad_gram":
        if config.matrix_output_preconditioner_approximation != "diag":
            raise RuntimeError(
                "Megatron-FSDP GRAD_GRAM sidecars support diag approximation only; "
                "full and block_diag require explicit sidecar sharding/checkpoint contracts."
            )
        if config.matrix_output_preconditioner_refresh_interval != 1:
            raise RuntimeError(
                "Megatron-FSDP GRAD_GRAM sidecars require refresh_interval=1; cached "
                "sidecar checkpoint/resume semantics are not implemented yet."
            )
        if config.matrix_output_preconditioner_ema_beta is not None:
            raise RuntimeError(
                "Megatron-FSDP GRAD_GRAM sidecars do not support EMA state yet."
            )
    elif config.matrix_output_preconditioner != "none":
        raise RuntimeError("Megatron-FSDP matrix output sidecar must be none or grad_gram.")


def _fsdp_input_diag_recipe(config: OptimizerConfig) -> MatrixInputPreconditionerRecipe:
    return MatrixInputPreconditionerRecipe(
        kind=MatrixPreconditionerKind.FEATURE_GRAM,
        approximation=MatrixPreconditionerApproximation.DIAG,
        scope=MatrixPreconditionerScope.DIAG_APPROX,
        normalization=MatrixPreconditionerNormalization(config.matrix_input_preconditioner_normalization),
        activation_dtype=config.matrix_input_preconditioner_activation_dtype,
        accumulation_dtype=config.matrix_input_preconditioner_accumulation_dtype,
        refresh_interval=config.matrix_input_preconditioner_refresh_interval,
        token_sample_size=config.matrix_input_preconditioner_token_sample_size,
        ridge=config.matrix_input_preconditioner_ridge,
        ema_beta=config.matrix_input_preconditioner_ema_beta,
        min_samples_per_feature=config.matrix_input_preconditioner_min_samples_per_feature,
        block_size=config.matrix_input_preconditioner_block_size,
    )


def _fsdp_output_diag_recipe(config: OptimizerConfig) -> MatrixOutputPreconditionerRecipe:
    return MatrixOutputPreconditionerRecipe(
        kind=MatrixPreconditionerKind.GRAD_GRAM,
        approximation=MatrixPreconditionerApproximation.DIAG,
        scope=MatrixPreconditionerScope.DIAG_APPROX,
        normalization=MatrixPreconditionerNormalization(config.matrix_output_preconditioner_normalization),
        gradient_dtype=config.matrix_output_preconditioner_gradient_dtype,
        accumulation_dtype=config.matrix_output_preconditioner_accumulation_dtype,
        refresh_interval=config.matrix_output_preconditioner_refresh_interval,
        token_sample_size=config.matrix_output_preconditioner_token_sample_size,
        ridge=config.matrix_output_preconditioner_ridge,
        ema_beta=config.matrix_output_preconditioner_ema_beta,
        min_samples_per_feature=config.matrix_output_preconditioner_min_samples_per_feature,
        block_size=config.matrix_output_preconditioner_block_size,
    )


def _configure_fsdp_diag_sidecars(param: torch.nn.Parameter, config: OptimizerConfig) -> None:
    input_recipe = None
    output_recipe = None
    factors = ExtraWgradFactor.NONE
    if config.matrix_input_preconditioner == "feature_gram":
        input_recipe = _fsdp_input_diag_recipe(config)
        factors |= ExtraWgradFactor.FEATURE_GRAM
    if config.matrix_output_preconditioner == "grad_gram":
        output_recipe = _fsdp_output_diag_recipe(config)
        factors |= ExtraWgradFactor.GRAD_GRAM
    if factors == ExtraWgradFactor.NONE:
        return
    configure_matrix_update_param(
        param,
        recipe=input_recipe,
        output_recipe=output_recipe,
        factors=factors,
    )
    set_matrix_sidecar_owner(param, param)
    orig_param = getattr(param, "orig_param", None)
    if orig_param is not None:
        set_matrix_sidecar_owner(orig_param, param)


def _megatron_fsdp_owner_for_optimizer_param(param: torch.nn.Parameter) -> object:
    owner = getattr(param, "_megatron_fsdp_model", None)
    if owner is None:
        raise RuntimeError(
            "Megatron-FSDP matrix optimizer parameters must carry _megatron_fsdp_model "
            "metadata before optimizer construction. Ensure matrix parameters were "
            "registered before FSDP wrapping and optimizer params came from the "
            "MegatronFSDP-wrapped model."
        )
    return owner


def _split_param_groups_by_megatron_fsdp_owner(
    param_groups: list[dict],
) -> "OrderedDict[object, list[dict]]":
    """Split optimizer param groups so every group set has one MegatronFSDP owner."""

    owner_to_param_groups: "OrderedDict[object, list[dict]]" = OrderedDict()
    for param_group in param_groups:
        params_by_owner: "OrderedDict[object, list[torch.nn.Parameter]]" = OrderedDict()
        for param in param_group.get("params", []):
            owner = _megatron_fsdp_owner_for_optimizer_param(param)
            params_by_owner.setdefault(owner, []).append(param)
        for owner, owner_params in params_by_owner.items():
            owner_param_group = param_group.copy()
            owner_param_group["params"] = owner_params
            owner_to_param_groups.setdefault(owner, []).append(owner_param_group)
    return owner_to_param_groups


def _matrix_fallback_optimizer_config(config: OptimizerConfig) -> OptimizerConfig:
    """Return ``config`` with all matrix-optimizer-specific options disabled.

    The fallback optimizer should see the same ordinary optimizer settings as
    the user requested, but no active matrix ownership/preconditioner options.
    Deriving the reset values from ``OptimizerConfig`` defaults avoids a second
    hand-maintained copy of every ``matrix_*`` field.
    """

    default_config = OptimizerConfig()
    matrix_defaults = {
        field.name: getattr(default_config, field.name)
        for field in fields(OptimizerConfig)
        if field.name.startswith("matrix_")
    }
    matrix_defaults["use_layer_wise_distributed_optimizer"] = False
    return replace(config, **matrix_defaults)


def get_megatron_matrix_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    matrix_config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Build a chained matrix optimizer for affine weights plus Adam/SGD fallback."""

    if not HAVE_EMERGING_MATRIX_OPTIMIZERS:
        raise RuntimeError("emerging_optimizers matrix update rules are not installed.")
    use_megatron_fsdp = _model_chunks_use_megatron_fsdp(model_chunks)
    if config.use_distributed_optimizer and not use_megatron_fsdp:
        raise RuntimeError(
            "Matrix optimizers do not support standard DistributedOptimizer until logical matrix "
            "gather/apply/scatter views exist."
        )
    if config.fp16:
        raise RuntimeError("Matrix optimizers support fp32/bf16 paths in this checkout; fp16 is unsupported.")
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    if use_megatron_fsdp:
        configured_matrix_params = _configured_fsdp_matrix_params(model_chunks, config)
    else:
        configured_matrix_params = configure_model_matrix_updates(model_chunks, config)
    if not configured_matrix_params:
        if use_megatron_fsdp:
            raise RuntimeError(
                "matrix_optimizer requested with Megatron-FSDP, but no matrix-owned parameters "
                "were registered before FSDP wrapping. Matrix-aware FSDP planning requires "
                "pre-FSDP registration via the Megatron training wrapper/buffer-routing path."
            )
        raise RuntimeError("matrix_optimizer requested, but no eligible native affine matrix parameters were found.")

    feature_gram_params = [
        param for param in configured_matrix_params if hasattr(param, "_feature_gram_recipe")
    ]
    grad_gram_params = [
        param for param in configured_matrix_params if hasattr(param, "_grad_gram_recipe")
    ]
    gram_groups = []
    if (feature_gram_params or grad_gram_params) and get_pg_size(pg_collection.dp_cp) > 1:
        gram_groups.append(pg_collection.dp_cp)
    set_feature_gram_finalization_required(
        feature_gram_params, required=bool(gram_groups)
    )
    set_grad_gram_finalization_required(
        grad_gram_params, required=bool(gram_groups)
    )

    log_single_rank(
        logger,
        logging.INFO,
        f"Setting up {config.matrix_optimizer} for {len(configured_matrix_params)} matrix parameters.",
    )

    matrix_param_ids = {id(param) for param in configured_matrix_params}
    def is_matrix_param(name: str, param: torch.nn.Parameter) -> bool:
        del name
        return id(param) in matrix_param_ids

    def is_fallback_param(name: str, param: torch.nn.Parameter) -> bool:
        del name
        return id(param) not in matrix_param_ids

    matrix_param_groups = _get_param_groups(
        model_chunks,
        config,
        matrix_config_overrides if matrix_config_overrides is not None else config_overrides,
        param_filter=is_matrix_param,
    )

    use_layer_wise = config.use_layer_wise_distributed_optimizer
    use_separate_distributed_optimizer = (
        use_layer_wise
        and _model_chunks_use_distributed_optimizer_buffers(model_chunks)
        and not use_megatron_fsdp
    )

    matrix_update_rule = _make_matrix_update_rule(config, pg_collection)
    matrix_inplace_update_rule = (
        None if use_megatron_fsdp else _make_matrix_inplace_update_rule(config, pg_collection)
    )

    def build_matrix_torch_optimizer(param_groups: list[dict]) -> MatrixFunctionOptimizer:
        if config.matrix_optimizer == "muon":
            matrix_momentum = config.muon_momentum
            matrix_nesterov = config.muon_nesterov
        elif config.matrix_optimizer == "sgd":
            matrix_momentum = config.sgd_momentum
            matrix_nesterov = False
        else:
            matrix_momentum = 0.0
            matrix_nesterov = False
        return MatrixFunctionOptimizer(
            param_groups,
            lr=config.lr,
            update_rule=matrix_update_rule,
            inplace_update_rule=matrix_inplace_update_rule,
            weight_decay=config.weight_decay,
            decoupled_weight_decay=config.decoupled_weight_decay,
            momentum=matrix_momentum,
            nesterov=matrix_nesterov,
            tp_update_mode=_tp_mode_from_config(config),
            gram_process_groups=gram_groups,
        )

    def matrix_init_state_fn(opt, config=None):
        return None

    optimizers = []
    fsdp_owner_to_torch_optimizers = defaultdict(list)
    if use_megatron_fsdp:
        if megatron_fsdp_fully_shard_optimizer is None:
            raise RuntimeError("Megatron-FSDP matrix optimizer requires Megatron-FSDP support.")
        for owner, owner_matrix_param_groups in _split_param_groups_by_megatron_fsdp_owner(
            matrix_param_groups
        ).items():
            matrix_torch_optimizer = build_matrix_torch_optimizer(owner_matrix_param_groups)
            fsdp_owner_to_torch_optimizers[owner].append(matrix_torch_optimizer)
    elif use_layer_wise:
        matrix_torch_optimizer = build_matrix_torch_optimizer(matrix_param_groups)
        matrix_optimizer = LayerWiseDistributedOptimizer(
            [matrix_torch_optimizer],
            config,
            pg_collection,
            init_state_fn_list=[matrix_init_state_fn],
            model_chunks=model_chunks,
        )
        for chained_optimizer in matrix_optimizer.chained_optimizers:
            _copy_matrix_model_refs_to_main_params(chained_optimizer)
    elif config.bf16:
        matrix_torch_optimizer = build_matrix_torch_optimizer(matrix_param_groups)
        matrix_optimizer = Float16OptimizerWithFloat16Params(
            matrix_torch_optimizer, config, None, matrix_init_state_fn
        )
        _copy_matrix_model_refs_to_main_params(matrix_optimizer)
    else:
        matrix_torch_optimizer = build_matrix_torch_optimizer(matrix_param_groups)
        matrix_optimizer = FP32Optimizer(matrix_torch_optimizer, config, matrix_init_state_fn)
    if not use_megatron_fsdp:
        _set_matrix_optimizer_process_groups(matrix_optimizer, pg_collection)
        optimizers.append(matrix_optimizer)

    fallback_config = _matrix_fallback_optimizer_config(config)
    if use_separate_distributed_optimizer:
        fallback_config = replace(fallback_config, use_distributed_optimizer=True)
    fallback_param_groups = _get_param_groups(
        model_chunks, fallback_config, config_overrides, param_filter=is_fallback_param
    )
    if any(group.get('is_expert_parallel', False) for group in fallback_param_groups):
        raise RuntimeError(
            "Matrix optimizer fallback does not yet support expert-parallel fallback "
            "parameter groups. Route expert fallback through the standard Megatron "
            "optimizer path or add explicit expert DistOpt routing before enabling this "
            "combination."
        )
    if fallback_param_groups:
        if use_megatron_fsdp:
            fallback_torch_config = replace(fallback_config, use_distributed_optimizer=False)
            for owner, owner_fallback_param_groups in _split_param_groups_by_megatron_fsdp_owner(
                fallback_param_groups
            ).items():
                fallback_torch_optimizer, _ = _get_megatron_optimizer_based_on_param_groups(
                    config=fallback_torch_config,
                    model_chunks=model_chunks,
                    param_groups=owner_fallback_param_groups,
                    model_parallel_group=_matrix_optimizer_model_parallel_group(pg_collection),
                    pg_collection=pg_collection,
                    skip_megatron_wrapping=True,
                )
                fsdp_owner_to_torch_optimizers[owner].append(fallback_torch_optimizer)
        else:
            if use_separate_distributed_optimizer:
                (
                    distopt_process_groups,
                    distopt_distributed_optimizer_instance_id,
                    distopt_per_model_buffers,
                ) = _setup_layerwise_fallback_distopt_routing(pg_collection, model_chunks)
                fallback_optimizer = _get_megatron_optimizer_based_on_param_groups(
                    config=fallback_config,
                    model_chunks=model_chunks,
                    param_groups=fallback_param_groups,
                    per_model_buffers=distopt_per_model_buffers,
                    model_parallel_group=distopt_process_groups['mp_group'],
                    data_parallel_group=distopt_process_groups['intra_dp_cp_group'],
                    data_parallel_group_gloo=distopt_process_groups['intra_dp_cp_group_gloo'],
                    data_parallel_group_idx=get_pg_rank(distopt_process_groups['mp_group']),
                    intra_dist_opt_group=distopt_process_groups['intra_dist_opt_group'],
                    distributed_optimizer_instance_id=distopt_distributed_optimizer_instance_id,
                    pg_collection=pg_collection,
                    skip_megatron_wrapping=False,
                )
            else:
                fallback_optimizer = _get_megatron_optimizer_based_on_param_groups(
                    config=fallback_config,
                    model_chunks=model_chunks,
                    param_groups=fallback_param_groups,
                    model_parallel_group=_matrix_optimizer_model_parallel_group(pg_collection),
                    pg_collection=pg_collection,
                    skip_megatron_wrapping=False,
                )
            if isinstance(fallback_optimizer, ChainedOptimizer):
                for chained_optimizer in fallback_optimizer.chained_optimizers:
                    setattr(chained_optimizer, '_chained_optimizer_config', config)
                optimizers += fallback_optimizer.chained_optimizers
            else:
                setattr(fallback_optimizer, '_chained_optimizer_config', config)
                optimizers.append(fallback_optimizer)
    if use_megatron_fsdp:
        for owner_torch_optimizers in fsdp_owner_to_torch_optimizers.values():
            owner_torch_optimizer = _ChainedTorchOptimizer(owner_torch_optimizers)
            optimizers.append(
                MegatronFSDPOptimizer(
                    megatron_fsdp_fully_shard_optimizer(
                        owner_torch_optimizer,
                        preproc_state_dict_for_dcp_ckpt=False,
                    ),
                    config,
                    pg_collection,
                )
            )
    return ChainedOptimizer(optimizers)
