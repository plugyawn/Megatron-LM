# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Matrix-update metadata and extra wgrad factor utilities.

This module intentionally keeps collection method-free: affine matrix optimizers
may request extra factors beside ``main_grad``, but optimizers do not receive raw
activations or method-specific targets.

The current ``FEATURE_GRAM`` factor is the input-side/right preconditioner
statistic ``X.T @ X``. Output-side/left factors such as ``dY.T @ dY`` are not
collected here yet.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import torch


class ExtraWgradFactor(enum.IntFlag):
    """Optional affine wgrad-side statistics."""

    NONE = 0
    FEATURE_GRAM = 1
    FEATURE_SUM = 2


class FeatureGramApproximation(enum.Enum):
    """Stored representation of ``X.T @ X``."""

    FULL = "full"
    DIAG = "diag"
    BLOCK_DIAG = "block_diag"
    SKETCH = "sketch"


class FeatureGramScope(enum.Enum):
    """Semantic scope of a collected input-side/right feature Gram."""

    GLOBAL_EXACT = "global_exact"
    TP_LOCAL_BLOCK_DIAG = "tp_local_block_diag"
    BLOCK_DIAG_APPROX = "block_diag_approx"
    DIAG_APPROX = "diag_approx"
    SKETCH_APPROX = "sketch_approx"


class FeatureGramNormalization(enum.Enum):
    """Convention used when an optimizer consumes a feature Gram."""

    SUM = "sum"
    MEAN = "mean"


class TPUpdateMode(enum.Enum):
    """Tensor-parallel matrix-function apply mode."""

    TP_ALLGATHER_LOGICAL_MATRIX = "tp_allgather_logical_matrix"
    TP_SMALL_GRAM_POLAR_ALLREDUCE = "tp_small_gram_polar_allreduce"
    TP_BLOCK_LOCAL_APPROX = "tp_block_local_approx"


@dataclass(frozen=True)
class LinearWeightInfo:
    """Static metadata for a local shard of an affine weight."""

    param: torch.nn.Parameter
    logical_shape: tuple[int, int]
    local_shape: tuple[int, int]
    tp_layout: Literal["none", "duplicated", "column_parallel", "row_parallel", "grouped_expert"]
    sequence_parallel: bool
    expert_parallel: bool
    has_bias: bool
    is_affine_weight: bool
    is_embedding: bool = False
    is_lm_head: bool = False
    role: Optional[str] = None


@dataclass
class FeatureGramRecipe:
    """Collection and consumption policy for input-side ``FEATURE_GRAM = X.T @ X``."""

    approximation: FeatureGramApproximation
    scope: FeatureGramScope
    normalization: FeatureGramNormalization
    source_dtype: Literal["bf16_saved", "fp32_cast", "fp8_dequant"]
    accumulation_dtype: torch.dtype
    refresh_interval: int
    token_sample_size: Optional[int]
    ridge: float
    ema_beta: Optional[float]
    min_samples_per_feature: Optional[float]
    block_size: int = 128


@dataclass
class MatrixApplyPlan:
    """How a matrix-shaped update is applied under DP/TP sharding."""

    dp_apply_mode: Literal[
        "layerwise_whole_param_owner",
        "dp_allgather_compute_scatter",
        "standard_distopt_forbidden",
    ]
    tp_update_mode: TPUpdateMode
    feature_gram_scope: FeatureGramScope
    requires_full_logical_gradient: bool
    requires_full_logical_weight_for_direction: bool = False
    approximation_label: Optional[str] = None


def _enum_from_string(enum_type, value):
    if isinstance(value, enum_type):
        return value
    return enum_type(value)


def feature_gram_scope_for(
    approximation: FeatureGramApproximation,
    tp_layout: str,
) -> FeatureGramScope:
    """Infer the factor scope implied by approximation and TP layout.

    Row-parallel linears own only a feature-axis shard. Until cross-TP Gram
    support exists, any local row-parallel factor is TP-local metadata rather
    than a global exact or generic diagonal approximation.
    """

    if tp_layout == "row_parallel":
        return FeatureGramScope.TP_LOCAL_BLOCK_DIAG
    if approximation == FeatureGramApproximation.BLOCK_DIAG:
        return FeatureGramScope.BLOCK_DIAG_APPROX
    if approximation == FeatureGramApproximation.DIAG:
        return FeatureGramScope.DIAG_APPROX
    if approximation == FeatureGramApproximation.SKETCH:
        return FeatureGramScope.SKETCH_APPROX
    return FeatureGramScope.GLOBAL_EXACT


def set_linear_weight_info(
    param: Optional[torch.nn.Parameter],
    *,
    logical_shape: tuple[int, int],
    tp_layout: Literal["none", "duplicated", "column_parallel", "row_parallel", "grouped_expert"],
    sequence_parallel: bool,
    expert_parallel: bool,
    has_bias: bool,
    role: Optional[str] = None,
    is_embedding: bool = False,
    is_lm_head: bool = False,
    collector: str = "native",
) -> None:
    """Attach affine matrix metadata to a weight parameter."""

    if param is None:
        return
    setattr(
        param,
        "_mcore_linear_weight_info",
        LinearWeightInfo(
            param=param,
            logical_shape=logical_shape,
            local_shape=tuple(param.shape),
            tp_layout=tp_layout,
            sequence_parallel=sequence_parallel,
            expert_parallel=expert_parallel,
            has_bias=has_bias,
            is_affine_weight=True,
            is_embedding=is_embedding,
            is_lm_head=is_lm_head,
            role=role,
        ),
    )
    setattr(param, "_feature_gram_collector", collector)


def is_matrix_update_eligible(
    param: torch.nn.Parameter,
    *,
    min_matrix_dim: int,
) -> bool:
    """Return whether a parameter is structurally eligible for matrix updates."""

    info = getattr(param, "_mcore_linear_weight_info", None)
    if info is None:
        return False
    if param.ndim != 2 or not info.is_affine_weight:
        return False
    if min(info.logical_shape) < min_matrix_dim:
        return False
    if info.is_embedding or info.is_lm_head:
        return False
    if info.expert_parallel or info.tp_layout == "grouped_expert":
        return False
    if getattr(param, "is_embedding_or_output_parameter", False):
        return False
    if getattr(param, "is_embedding_parameter", False):
        return False
    if getattr(param, "shared_embedding", False):
        return False
    return True


def _feature_dim(param: torch.nn.Parameter) -> int:
    info = getattr(param, "_mcore_linear_weight_info", None)
    if info is None:
        if param.ndim != 2:
            raise ValueError("FEATURE_GRAM requires a 2D affine weight or LinearWeightInfo.")
        return param.shape[1]
    return info.local_shape[1]


def _feature_gram_shape(param: torch.nn.Parameter, recipe: FeatureGramRecipe) -> tuple[int, ...]:
    dim = _feature_dim(param)
    if recipe.approximation == FeatureGramApproximation.FULL:
        return (dim, dim)
    if recipe.approximation == FeatureGramApproximation.DIAG:
        return (dim,)
    if recipe.approximation == FeatureGramApproximation.BLOCK_DIAG:
        num_blocks = (dim + recipe.block_size - 1) // recipe.block_size
        return (num_blocks, recipe.block_size, recipe.block_size)
    raise NotImplementedError(
        f"FEATURE_GRAM approximation {recipe.approximation.value!r} is not implemented yet. "
        "Use 'full', 'diag', or 'block_diag', or add an explicit storage format for this approximation."
    )


def _validate_feature_gram_recipe(param: torch.nn.Parameter, recipe: FeatureGramRecipe) -> None:
    if recipe.refresh_interval < 1:
        raise ValueError("matrix-feature-gram-refresh-interval must be >= 1")
    if recipe.block_size < 1:
        raise ValueError("matrix-feature-gram-block-size must be >= 1")
    if recipe.token_sample_size is not None and recipe.token_sample_size < 1:
        raise ValueError("matrix-feature-gram-token-sample-size must be >= 1 when set")
    if recipe.min_samples_per_feature is not None and recipe.token_sample_size is not None:
        feature_dim = _feature_dim(param)
        min_samples = recipe.min_samples_per_feature * feature_dim
        if (
            recipe.approximation == FeatureGramApproximation.FULL
            and recipe.token_sample_size < min_samples
        ):
            raise ValueError(
                "Full FEATURE_GRAM is rank-deficient under this token sample size; "
                "use diag/block_diag/sketch or increase matrix-feature-gram-token-sample-size."
            )


def _needs_distributed_finalization() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def _transformer_engine_feature_gram_available() -> bool:
    try:
        from transformer_engine.pytorch.module.extra_wgrad import (  # pylint: disable=unused-import
            maybe_accumulate_feature_gram,
        )
    except Exception:
        return False
    return True


def _mark_feature_gram_unfinalized(param: torch.nn.Parameter) -> None:
    param._feature_gram_finalized = not getattr(param, "_feature_gram_finalization_required", False)


def set_feature_gram_finalization_required(
    params: Iterable[torch.nn.Parameter], *, required: bool
) -> None:
    for param in params:
        param._feature_gram_finalization_required = required
        _mark_feature_gram_unfinalized(param)


def finalize_feature_gram_buffers(
    params: Iterable[torch.nn.Parameter],
    *,
    process_groups: Iterable[torch.distributed.ProcessGroup] = (),
) -> None:
    """Reduce input-side FEATURE_GRAM buffers across caller-specified groups.

    Megatron call sites must pass the DP/CP groups required by their exactness
    contract. This function deliberately does not infer those groups from the
    default world, because using the wrong group would silently change the
    feature-statistic semantics.
    """

    groups = tuple(process_groups)
    params = list(params)
    if any(getattr(param, "_feature_gram_finalization_required", False) for param in params) and not groups:
        raise RuntimeError(
            "FEATURE_GRAM finalization requires explicit process_groups for these parameters."
        )

    for param in params:
        if not hasattr(param, "main_grad_feature_gram"):
            continue
        for group in groups:
            torch.distributed.all_reduce(
                param.main_grad_feature_gram, op=torch.distributed.ReduceOp.SUM, group=group
            )
            torch.distributed.all_reduce(
                param.main_grad_feature_count, op=torch.distributed.ReduceOp.SUM, group=group
            )
            if hasattr(param, "main_grad_feature_sum"):
                torch.distributed.all_reduce(
                    param.main_grad_feature_sum, op=torch.distributed.ReduceOp.SUM, group=group
                )
        param._feature_gram_finalized = True


def allocate_feature_gram_buffers(param: torch.nn.Parameter, recipe: FeatureGramRecipe) -> None:
    """Allocate per-parameter extra wgrad buffers for a requested recipe."""

    _validate_feature_gram_recipe(param, recipe)
    gram_shape = _feature_gram_shape(param, recipe)
    gram = getattr(param, "main_grad_feature_gram", None)
    if (
        gram is None
        or tuple(gram.shape) != gram_shape
        or gram.dtype != recipe.accumulation_dtype
        or gram.device != param.device
    ):
        param.main_grad_feature_gram = torch.zeros(
            gram_shape,
            device=param.device,
            dtype=recipe.accumulation_dtype,
        )
    if not hasattr(param, "main_grad_feature_count"):
        param.main_grad_feature_count = torch.zeros((), device=param.device, dtype=torch.float64)
    if getattr(param, "_extra_wgrad_factors", ExtraWgradFactor.NONE) & ExtraWgradFactor.FEATURE_SUM:
        if not hasattr(param, "main_grad_feature_sum"):
            param.main_grad_feature_sum = torch.zeros(
                (_feature_dim(param),), device=param.device, dtype=recipe.accumulation_dtype
            )
    param._feature_gram_recipe = recipe
    param._feature_gram_scope = recipe.scope
    if not hasattr(param, "_feature_gram_generation"):
        param._feature_gram_generation = 0
    _mark_feature_gram_unfinalized(param)


def reset_feature_gram_buffers(
    param: torch.nn.Parameter, *, active: bool = True, zero: bool = True
) -> None:
    """Prepare transient factor buffers for a collection window.

    ``active=False, zero=False`` preserves the previous FEATURE_GRAM for
    refresh-cadence reuse. This is what allows matrix optimizers to cache
    Cholesky/diagonal solve factors across steps when refresh_interval > 1.
    """

    if zero and hasattr(param, "main_grad_feature_gram"):
        param.main_grad_feature_gram.zero_()
    if zero and hasattr(param, "main_grad_feature_count"):
        param.main_grad_feature_count.zero_()
    if zero and hasattr(param, "main_grad_feature_sum"):
        param.main_grad_feature_sum.zero_()
    if zero:
        param._feature_gram_generation = getattr(param, "_feature_gram_generation", 0) + 1
        _mark_feature_gram_unfinalized(param)
    param._feature_gram_active = active


def iter_matrix_update_params(modules: Iterable[torch.nn.Module]):
    """Yield parameters that carry matrix-update metadata."""

    for module in modules:
        for param in module.parameters():
            if hasattr(param, "_mcore_linear_weight_info"):
                yield param


def configure_matrix_update_param(
    param: torch.nn.Parameter,
    *,
    recipe: FeatureGramRecipe,
    factors: ExtraWgradFactor = ExtraWgradFactor.FEATURE_GRAM,
) -> None:
    """Enable requested extra wgrad factors on an eligible parameter."""

    param._extra_wgrad_factors = factors
    allocate_feature_gram_buffers(param, recipe)
    reset_feature_gram_buffers(param, active=True)


def recipe_from_optimizer_config(config, info: LinearWeightInfo) -> FeatureGramRecipe:
    """Build a feature Gram recipe from an ``OptimizerConfig``-like object."""

    approximation = _enum_from_string(FeatureGramApproximation, config.matrix_feature_gram)
    scope = feature_gram_scope_for(approximation, info.tp_layout)
    normalization = _enum_from_string(
        FeatureGramNormalization, config.matrix_feature_gram_normalization
    )
    return FeatureGramRecipe(
        approximation=approximation,
        scope=scope,
        normalization=normalization,
        source_dtype=config.matrix_feature_gram_source_dtype,
        accumulation_dtype=getattr(config, "matrix_feature_gram_accumulation_dtype", torch.float32),
        refresh_interval=config.matrix_feature_gram_refresh_interval,
        token_sample_size=config.matrix_feature_gram_token_sample_size,
        ridge=config.matrix_feature_gram_ridge,
        ema_beta=config.matrix_feature_gram_ema_beta,
        min_samples_per_feature=config.matrix_feature_gram_min_samples_per_feature,
        block_size=getattr(config, "matrix_feature_gram_block_size", 128),
    )


def configure_model_matrix_updates(modules: Iterable[torch.nn.Module], config) -> list[torch.nn.Parameter]:
    """Enable FEATURE_GRAM collection for eligible parameters.

    This function is intentionally fail-closed for backends that do not collect
    the factor in this checkout.
    """

    if getattr(config, "matrix_optimizer", "none") == "none":
        return []
    if getattr(config, "use_distributed_optimizer", False):
        raise RuntimeError(
            "Matrix optimizers do not support standard DistributedOptimizer until logical "
            "matrix gather/apply/scatter views exist."
        )
    if getattr(config, "matrix_bias_mode", "fallback") == "augmented_feature_sum":
        raise RuntimeError(
            "matrix-bias-mode=augmented_feature_sum requires FEATURE_SUM collection and "
            "augmented affine solves; use fallback."
        )

    configured: list[torch.nn.Parameter] = []
    for param in iter_matrix_update_params(modules):
        info = getattr(param, "_mcore_linear_weight_info")
        if (
            param.ndim == 2
            and info.is_affine_weight
            and min(info.logical_shape) >= config.matrix_min_dim
            and (info.expert_parallel or info.tp_layout == "grouped_expert")
        ):
            raise RuntimeError(
                "Matrix optimizers do not support grouped/expert affine weights in this "
                f"checkout (role={info.role!r}, tp_layout={info.tp_layout!r})."
            )
        if not is_matrix_update_eligible(param, min_matrix_dim=config.matrix_min_dim):
            continue
        collector = getattr(param, "_feature_gram_collector", "unknown")
        if collector == "transformer_engine":
            if not _transformer_engine_feature_gram_available():
                raise RuntimeError(
                    "FEATURE_GRAM collection for Transformer Engine linears requires a "
                    "Transformer Engine build that exposes "
                    "transformer_engine.pytorch.module.extra_wgrad.maybe_accumulate_feature_gram."
                )
        elif collector != "native":
            raise RuntimeError(
                f"FEATURE_GRAM collection for {collector!r} linears is not available in this "
                f"checkout (role={info.role!r}, tp_layout={info.tp_layout!r})."
            )
        recipe = recipe_from_optimizer_config(config, info)
        if (
            info.tp_layout == "row_parallel"
            and recipe.approximation == FeatureGramApproximation.FULL
        ):
            raise RuntimeError(
                "matrix-feature-gram=full is not supported for row-parallel weights without "
                "cross-TP FEATURE_GRAM collection; use diag/block_diag/sketch or disable the "
                f"matrix optimizer for this parameter (role={info.role!r})."
            )
        if collector == "native" and recipe.source_dtype == "fp8_dequant":
            raise RuntimeError(
                "FEATURE_GRAM source_dtype=fp8_dequant requires Transformer Engine "
                "collection at the wgrad site; native linears cannot dequantize FP8 sources."
            )
        factors = ExtraWgradFactor.FEATURE_GRAM
        configure_matrix_update_param(param, recipe=recipe, factors=factors)
        configured.append(param)
    return configured


def _cast_feature_input(inputmat: torch.Tensor, recipe: FeatureGramRecipe) -> torch.Tensor:
    if recipe.source_dtype not in ("bf16_saved", "fp32_cast", "fp8_dequant"):
        raise ValueError(f"Unsupported feature Gram source dtype: {recipe.source_dtype}")
    if recipe.source_dtype == "fp8_dequant":
        raise NotImplementedError(
            "FEATURE_GRAM source_dtype=fp8_dequant requires Transformer Engine dequantization "
            "support at the wgrad site; this native collector fails closed."
        )
    if recipe.source_dtype == "fp32_cast":
        return inputmat.to(torch.float32)
    return inputmat.to(recipe.accumulation_dtype)


def _accumulate_diag_feature_gram(gram: torch.Tensor, x: torch.Tensor) -> None:
    if gram.is_cuda and x.is_cuda:
        try:
            from emerging_optimizers.triton_kernels.feature_gram import diag_feature_gram_reduce

            diag_feature_gram_reduce(x, out=gram, accumulate=True)
            return
        except Exception:
            pass
    gram.add_(torch.sum(x * x, dim=0))


def _accumulate_block_diag_feature_gram(
    gram: torch.Tensor,
    x: torch.Tensor,
    *,
    feature_dim: int,
    block_size: int,
) -> None:
    """Accumulate padded block-diagonal ``X.T @ X`` via batched GEMM."""

    expected_blocks, expected_block, _ = gram.shape
    if expected_block != block_size:
        raise RuntimeError(
            f"FEATURE_GRAM block buffer has block size {expected_block}, expected {block_size}."
        )
    padded_dim = expected_blocks * block_size
    if x.shape[-1] != feature_dim:
        raise RuntimeError(
            f"FEATURE_GRAM block_diag expected {feature_dim} input features, got {x.shape[-1]}."
        )
    if padded_dim != feature_dim:
        x = torch.nn.functional.pad(x, (0, padded_dim - feature_dim))
    x_blocks = x.reshape(x.shape[0], expected_blocks, block_size).transpose(0, 1)
    gram.add_(torch.bmm(x_blocks.transpose(1, 2), x_blocks))


@torch.no_grad()
def maybe_accumulate_feature_gram(weight: torch.Tensor, inputmat: torch.Tensor) -> None:
    """Accumulate ``X.T @ X`` for a linear weight when requested.

    The caller must pass the same logical 2D feature matrix used for ordinary
    wgrad. This function stores raw sums and count; normalization is applied by
    ``get_feature_gram_for_optimizer``.
    """

    flags = getattr(weight, "_extra_wgrad_factors", ExtraWgradFactor.NONE)
    if not (flags & ExtraWgradFactor.FEATURE_GRAM):
        return
    if not getattr(weight, "_feature_gram_active", True):
        return
    recipe = getattr(weight, "_feature_gram_recipe", None)
    if recipe is None:
        raise RuntimeError("FEATURE_GRAM requested without a FeatureGramRecipe.")
    if inputmat.dim() != 2:
        inputmat = inputmat.reshape(-1, inputmat.shape[-1])
    x = _cast_feature_input(inputmat, recipe)
    if recipe.token_sample_size is not None:
        remaining = recipe.token_sample_size - int(weight.main_grad_feature_count.item())
        if remaining <= 0:
            return
        # Deterministic prefix sampling keeps collection graph-safe and testable.
        if x.shape[0] > remaining:
            x = x[:remaining]

    gram = weight.main_grad_feature_gram
    if recipe.approximation == FeatureGramApproximation.FULL:
        gram.add_(x.t().matmul(x))
    elif recipe.approximation == FeatureGramApproximation.DIAG:
        _accumulate_diag_feature_gram(gram, x)
    elif recipe.approximation == FeatureGramApproximation.BLOCK_DIAG:
        _accumulate_block_diag_feature_gram(
            gram,
            x,
            feature_dim=_feature_dim(weight),
            block_size=recipe.block_size,
        )
    else:
        raise NotImplementedError(
            f"FEATURE_GRAM approximation {recipe.approximation.value!r} is not implemented yet."
        )
    weight.main_grad_feature_count.add_(float(x.shape[0]))
    _mark_feature_gram_unfinalized(weight)
    if flags & ExtraWgradFactor.FEATURE_SUM:
        if not hasattr(weight, "main_grad_feature_sum"):
            raise RuntimeError("FEATURE_SUM requested without a main_grad_feature_sum buffer.")
        weight.main_grad_feature_sum.add_(x.sum(dim=0))


def get_feature_gram_for_optimizer(param: torch.nn.Parameter) -> torch.Tensor:
    """Return the feature Gram using the recipe's consumption normalization."""

    recipe = getattr(param, "_feature_gram_recipe", None)
    if recipe is None:
        raise RuntimeError("Parameter has no FeatureGramRecipe.")
    if _needs_distributed_finalization() and not getattr(param, "_feature_gram_finalized", False):
        raise RuntimeError(
            "FEATURE_GRAM has not been finalized across distributed groups; call "
            "finalize_feature_gram_buffers before optimizer consumption."
        )
    gram = param.main_grad_feature_gram
    if param.main_grad_feature_count.item() <= 0.0:
        raise RuntimeError(
            "FEATURE_GRAM has zero collected feature rows; ensure the wgrad path collected "
            "main_grad_feature_gram before optimizer consumption."
        )
    if recipe.normalization == FeatureGramNormalization.MEAN:
        count = param.main_grad_feature_count.clamp_min(1.0).to(gram.dtype)
        return gram / count
    return gram
