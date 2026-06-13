# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron matrix optimizer integration for composable matrix update rules.

``matrix_optimizer`` selects the matrix update rule for eligible affine weights.
``matrix_input_preconditioner=feature_gram`` optionally right-preconditions that
rule with ``C_in = X.T @ X``. ``matrix_output_preconditioner=grad_gram``
optionally left-preconditions it with ``C_out = dY.T @ dY``.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import torch

from megatron.core.matrix_update import (
    TPUpdateMode,
    configure_model_matrix_updates,
    get_matrix_shard_spec,
    matrix_small_gram_side_for_spec,
    set_feature_gram_finalization_required,
    set_grad_gram_finalization_required,
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


def _muon_scale_factor(size_out: int, size_in: int, config: OptimizerConfig) -> float:
    mode = config.muon_scale_mode
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
) -> tuple[int, int]:
    """Return the logical matrix shape used for Muon scaling."""

    matrix_shard_spec = get_matrix_shard_spec(param)
    if matrix_shard_spec is not None and matrix_shard_spec.dp_shard_axis is not None:
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

    if local_matrix.ndim != 2:
        raise RuntimeError(
            f"Matrix FSDP Muon requires a 2D local matrix shard, got {tuple(local_matrix.shape)}."
        )
    if tuple(local_matrix.shape) != tuple(matrix_shard_spec.local_shape):
        raise RuntimeError(
            "Matrix FSDP Muon local shard shape does not match MatrixShardSpec: "
            f"local_matrix={tuple(local_matrix.shape)}, spec.local_shape={matrix_shard_spec.local_shape}."
        )
    tp_layout = _fsdp_small_gram_tp_layout(matrix_shard_spec)
    return tp_small_gram_newton_schulz_allreduce(
        local_matrix,
        tp_layout=tp_layout,
        group=group,
        logical_shape=matrix_shard_spec.logical_shape,
        steps=config.muon_num_ns_steps,
        coefficient_type=config.muon_coefficient_type,
        use_syrk=False,
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
        factorization = factorize_feature_gram(
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
        scale_rows, scale_cols = _muon_scale_shape(grad, param)
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
    if config.use_distributed_optimizer:
        raise RuntimeError(
            "Matrix optimizers do not support standard DistributedOptimizer until logical matrix "
            "gather/apply/scatter views exist."
        )
    if config.fp16:
        raise RuntimeError("Matrix optimizers support fp32/bf16 paths in this checkout; fp16 is unsupported.")
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    configured_matrix_params = configure_model_matrix_updates(model_chunks, config)
    if not configured_matrix_params:
        raise RuntimeError("matrix_optimizer requested, but no eligible native affine matrix parameters were found.")

    feature_gram_params = [
        param for param in configured_matrix_params if hasattr(param, "_feature_gram_recipe")
    ]
    grad_gram_params = [
        param for param in configured_matrix_params if hasattr(param, "_grad_gram_recipe")
    ]
    feature_gram_groups = []
    if (feature_gram_params or grad_gram_params) and get_pg_size(pg_collection.dp_cp) > 1:
        feature_gram_groups.append(pg_collection.dp_cp)
    set_feature_gram_finalization_required(
        feature_gram_params, required=bool(feature_gram_groups)
    )
    set_grad_gram_finalization_required(
        grad_gram_params, required=bool(feature_gram_groups)
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
        use_layer_wise and _model_chunks_use_distributed_optimizer_buffers(model_chunks)
    )

    matrix_optimizer = MatrixFunctionOptimizer(
        matrix_param_groups,
        lr=config.lr,
        update_rule=_make_matrix_update_rule(config, pg_collection),
        inplace_update_rule=_make_matrix_inplace_update_rule(config, pg_collection),
        weight_decay=config.weight_decay,
        decoupled_weight_decay=config.decoupled_weight_decay,
        tp_update_mode=_tp_mode_from_config(config),
        feature_gram_process_groups=feature_gram_groups,
    )

    def matrix_init_state_fn(opt, config=None):
        return None

    if use_layer_wise:
        matrix_optimizer = LayerWiseDistributedOptimizer(
            [matrix_optimizer],
            config,
            pg_collection,
            init_state_fn_list=[matrix_init_state_fn],
            model_chunks=model_chunks,
        )
        for chained_optimizer in matrix_optimizer.chained_optimizers:
            _copy_matrix_model_refs_to_main_params(chained_optimizer)
    elif config.bf16:
        matrix_optimizer = Float16OptimizerWithFloat16Params(
            matrix_optimizer, config, None, matrix_init_state_fn
        )
        _copy_matrix_model_refs_to_main_params(matrix_optimizer)
    else:
        matrix_optimizer = FP32Optimizer(matrix_optimizer, config, matrix_init_state_fn)
    _set_matrix_optimizer_process_groups(matrix_optimizer, pg_collection)

    fallback_config = copy.copy(config)
    fallback_config.matrix_optimizer = "none"
    fallback_config.use_layer_wise_distributed_optimizer = False
    if use_separate_distributed_optimizer:
        fallback_config.use_distributed_optimizer = True
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
        setattr(fallback_optimizer, '_chained_optimizer_config', config)
    else:
        fallback_optimizer = _get_megatron_optimizer_based_on_param_groups(
            config=fallback_config,
            model_chunks=model_chunks,
            param_groups=fallback_param_groups,
            model_parallel_group=_matrix_optimizer_model_parallel_group(pg_collection),
            pg_collection=pg_collection,
            skip_megatron_wrapping=False,
        )

    optimizers = [matrix_optimizer]
    if isinstance(fallback_optimizer, ChainedOptimizer):
        optimizers += fallback_optimizer.chained_optimizers
    else:
        optimizers.append(fallback_optimizer)
    return ChainedOptimizer(optimizers)
