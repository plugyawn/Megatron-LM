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
    set_feature_gram_finalization_required,
    set_grad_gram_finalization_required,
)
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank

from . import _get_megatron_optimizer_based_on_param_groups, _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .matrix_function_optimizer import MatrixFunctionOptimizer
from .optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params, FP32Optimizer, MegatronOptimizer
from .optimizer_config import OptimizerConfig, ParamKey

try:
    from emerging_optimizers.matrix_tp_apply import (
        tp_allgather_logical_matrix_update,
        tp_block_local_approx,
        tp_small_gram_polar_allreduce,
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


def _tp_mode_from_config(config: OptimizerConfig) -> TPUpdateMode:
    return {
        "allgather": TPUpdateMode.TP_ALLGATHER_LOGICAL_MATRIX,
        "small_gram_polar": TPUpdateMode.TP_SMALL_GRAM_POLAR_ALLREDUCE,
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
            coefficient_type=config.muon_ns_coefficients,
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
        if info is None:
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
        tp_group = pg_collection.tp
        tp_layout = info.tp_layout
        with fp32_matmul_precision(config.muon_fp32_matmul_prec):
            if tp_update_mode == TPUpdateMode.TP_ALLGATHER_LOGICAL_MATRIX:
                update = tp_allgather_logical_matrix_update(
                    preconditioned,
                    orthogonalize,
                    tp_layout=tp_layout,
                    group=tp_group,
                )
            elif tp_update_mode == TPUpdateMode.TP_SMALL_GRAM_POLAR_ALLREDUCE:
                update = tp_small_gram_polar_allreduce(
                    preconditioned, tp_layout=tp_layout, group=tp_group
                )
            elif tp_update_mode == TPUpdateMode.TP_BLOCK_LOCAL_APPROX:
                update = tp_block_local_approx(
                    preconditioned,
                    orthogonalize,
                    approximation_label="tp_block_local_matrix_update",
                )
            else:
                raise RuntimeError(f"Unsupported TP update mode: {tp_update_mode}")
        scale = _muon_scale_factor(grad.size(-2), grad.size(-1), config)
        return -update * scale * config.muon_extra_scale_factor

    return update_rule


def _make_matrix_inplace_update_rule(
    config: OptimizerConfig, pg_collection: ProcessGroupCollection
):
    """Return an optional fused in-place update rule for cheap diagonal cases.

    Only diagonal input/output preconditioners are eligible for the EO fused
    in-place/Triton helpers. Full and block-diagonal Grams intentionally fall
    back to the generic update rule, where the cached factorized solve path
    preserves the supported math without implying a Triton Gram-solve
    implementation.
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
            coefficient_type=config.muon_ns_coefficients,
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
    matrix_params = []
    fallback_params = []
    for model_chunk in model_chunks:
        for _, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in matrix_param_ids:
                matrix_params.append(param)
            else:
                fallback_params.append(param)

    try:
        for param in fallback_params:
            param.requires_grad = False
        matrix_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    finally:
        for param in fallback_params:
            param.requires_grad = True

    use_layer_wise = config.use_layer_wise_distributed_optimizer
    ddp_uses_distributed_optimizer = (
        bool(getattr(model_chunks[0], 'ddp_config', None))
        and model_chunks[0].ddp_config.use_distributed_optimizer
    )
    use_separate_distributed_optimizer = use_layer_wise and ddp_uses_distributed_optimizer

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

    previous_matrix_optimizer = config.matrix_optimizer
    try:
        for param in matrix_params:
            param.requires_grad = False
        config.matrix_optimizer = "none"
        if use_separate_distributed_optimizer:
            fallback_config = copy.copy(config)
            fallback_config.use_distributed_optimizer = True
            fallback_config.use_layer_wise_distributed_optimizer = False
            fallback_param_groups = _get_param_groups(
                model_chunks, fallback_config, config_overrides
            )
            distopt_process_groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
                pg_collection, model_chunks, use_gloo_process_groups=False
            )
            distopt_per_model_buffers = {}
            for model_chunk_idx, model_chunk in enumerate(model_chunks):
                if not hasattr(model_chunk, 'buffers'):
                    continue
                non_layer_wise_buffers = [
                    buffer
                    for buffer in model_chunk.buffers
                    if buffer.params
                    and not getattr(buffer.params[0], 'is_managed_by_layer_wise_optimizer', False)
                ]
                if non_layer_wise_buffers:
                    distopt_per_model_buffers[model_chunk_idx] = non_layer_wise_buffers
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
                distributed_optimizer_instance_id=0,
                pg_collection=pg_collection,
                skip_megatron_wrapping=False,
            )
            if hasattr(fallback_optimizer, 'config'):
                fallback_optimizer.config = config
        else:
            fallback_optimizer = get_megatron_optimizer(
                config,
                model_chunks,
                config_overrides=config_overrides,
                use_gloo_process_groups=use_gloo_process_groups,
            )
    finally:
        config.matrix_optimizer = previous_matrix_optimizer
        for param in matrix_params:
            param.requires_grad = True

    optimizers = [matrix_optimizer]
    if isinstance(fallback_optimizer, ChainedOptimizer):
        optimizers += fallback_optimizer.chained_optimizers
    else:
        optimizers.append(fallback_optimizer)
    return ChainedOptimizer(optimizers)
