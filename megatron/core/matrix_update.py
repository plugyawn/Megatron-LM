# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Matrix-update metadata and extra wgrad factor utilities.

This module intentionally keeps collection method-free: affine matrix optimizers
may request extra factors beside ``main_grad``, but optimizers do not receive raw
activations or method-specific targets.

The current concrete factors are:

* ``FEATURE_GRAM = X.T @ X`` for input-side/right preconditioning.
* ``GRAD_GRAM = dY.T @ dY`` for output-side/left preconditioning.
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
    GRAD_GRAM = 4


class MatrixPreconditionerKind(enum.Enum):
    """Kind of matrix preconditioner attached to a matrix optimizer update."""

    NONE = "none"
    FEATURE_GRAM = "feature_gram"
    GRAD_GRAM = "grad_gram"


class MatrixPreconditionerApproximation(enum.Enum):
    """Stored representation of a matrix preconditioner Gram."""

    FULL = "full"
    DIAG = "diag"
    BLOCK_DIAG = "block_diag"


class MatrixPreconditionerScope(enum.Enum):
    """Semantic scope of a collected matrix preconditioner."""

    GLOBAL_EXACT = "global_exact"
    TP_LOCAL_BLOCK_DIAG = "tp_local_block_diag"
    BLOCK_DIAG_APPROX = "block_diag_approx"
    DIAG_APPROX = "diag_approx"


class MatrixPreconditionerNormalization(enum.Enum):
    """Convention used when an optimizer consumes a collected Gram."""

    SUM = "sum"
    MEAN = "mean"


class TPUpdateMode(enum.Enum):
    """Tensor-parallel matrix-function apply mode."""

    TP_ALLGATHER_LOGICAL_MATRIX = "tp_allgather_logical_matrix"
    TP_SMALL_GRAM_NS_ALLREDUCE = "tp_small_gram_ns_allreduce"
    TP_BLOCK_LOCAL_APPROX = "tp_block_local_approx"


MATRIX_OPTIMIZER_INFO_ATTR = "_mcore_matrix_optimizer_info"
MATRIX_SHARD_SPEC_ATTR = "_mcore_matrix_shard_spec"
MATRIX_OPTIMIZER_STATE_SPEC_ATTR = "_mcore_matrix_optimizer_state_spec"

MATRIX_OPTIMIZER_OWNER_NONE = "none"
MATRIX_OPTIMIZER_OWNER_MUON = "muon"
MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION = "matrix_function"
MATRIX_OPTIMIZER_OWNER_FALLBACK = "fallback"

_MATRIX_OPTIMIZER_OWNERS = {
    MATRIX_OPTIMIZER_OWNER_NONE,
    MATRIX_OPTIMIZER_OWNER_MUON,
    MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
    MATRIX_OPTIMIZER_OWNER_FALLBACK,
}
_MATRIX_UPDATE_FAMILY_BY_OPTIMIZER_NAME = {
    "none": "none",
    "sgd": "sgd",
    "muon": "muon",
}


def matrix_update_family_from_optimizer_name(
    optimizer_name: Optional[str],
) -> Literal["none", "sgd", "muon"]:
    """Map public optimizer names to stable matrix-update contract families.

    This deliberately rejects optimizer variants such as ``adaptive_muon`` or
    legacy/product names such as ``newton_muon``. Those names may select a
    concrete implementation elsewhere, but they should not leak into metadata
    consumed by distributed layout and checkpoint code.
    """

    if optimizer_name is None:
        optimizer_name = "none"
    if optimizer_name not in _MATRIX_UPDATE_FAMILY_BY_OPTIMIZER_NAME:
        raise ValueError(
            f"{optimizer_name!r} is not a stable matrix update_family. Expected one of: "
            f"{', '.join(_MATRIX_UPDATE_FAMILY_BY_OPTIMIZER_NAME)}."
        )
    return _MATRIX_UPDATE_FAMILY_BY_OPTIMIZER_NAME[optimizer_name]


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


@dataclass(frozen=True)
class MatrixOptimizerInfo:
    """Internal optimizer ownership metadata for matrix-shaped parameters.

    This deliberately does not replace ``parameterization_role``. The
    parameterization role describes model/scaling semantics, while this record
    describes optimizer ownership and buffer-layout routing.
    """

    owner: Literal["none", "muon", "matrix_function", "fallback"]
    update_family: Literal["none", "sgd", "muon"]
    requires_layerwise_layout: bool = False


@dataclass(frozen=True)
class MatrixOptimizerStateSpec:
    """Internal optimizer-state sharding contract for matrix-owned params.

    ``same_shard_state_names`` names optimizer state tensors that, when present,
    must have the same matrix shape and shard layout as the parameter. The names
    are optional because optimizer state is usually materialized by the first
    real ``optimizer.step()`` or by checkpoint load.
    """

    same_shard_state_names: tuple[str, ...] = ()
    allow_discovered_same_shard_state: bool = True


@dataclass(frozen=True)
class MatrixShardSpec:
    """Internal logical matrix layout contract for distributed matrix optimizers.

    ``tp_shard_axis`` describes the current tensor-parallel local matrix shard,
    when the local shape proves that an axis is actually sharded. The DP/FSDP
    fields are reserved for matrix-axis-aware FSDP planner output; their
    presence is not a claim that generic Megatron-FSDP matrix optimizer support
    is wired.
    """

    logical_shape: tuple[int, int]
    local_shape: tuple[int, int]
    tp_layout: Literal["none", "duplicated", "column_parallel", "row_parallel", "grouped_expert"]
    tp_shard_axis: Optional[int] = None
    dp_shard_axis: Optional[int] = None
    dp_local_start: Optional[int] = None
    dp_local_end: Optional[int] = None
    pre_dp_local_shape: Optional[tuple[int, int]] = None

    def __post_init__(self) -> None:
        if len(self.logical_shape) != 2 or len(self.local_shape) != 2:
            raise ValueError(
                "MatrixShardSpec requires 2D logical_shape and local_shape, got "
                f"logical_shape={self.logical_shape}, local_shape={self.local_shape}."
            )
        if self.pre_dp_local_shape is not None:
            if len(self.pre_dp_local_shape) != 2:
                raise ValueError(
                    "MatrixShardSpec pre_dp_local_shape must be 2D, got "
                    f"{self.pre_dp_local_shape}."
                )
            if self.dp_shard_axis not in (0, 1):
                raise ValueError(
                    "MatrixShardSpec pre_dp_local_shape requires dp_shard_axis 0 or 1."
                )
            dp_axis = self.dp_shard_axis
            other_axis = 1 - dp_axis
            if self.local_shape[dp_axis] > self.pre_dp_local_shape[dp_axis]:
                raise ValueError(
                    "MatrixShardSpec local_shape cannot be larger than "
                    f"pre_dp_local_shape: local_shape={self.local_shape}, "
                    f"pre_dp_local_shape={self.pre_dp_local_shape}, dp_shard_axis={dp_axis}."
                )
            if self.local_shape[other_axis] != self.pre_dp_local_shape[other_axis]:
                raise ValueError(
                    "MatrixShardSpec local_shape non-DP dimension must match "
                    f"pre_dp_local_shape: local_shape={self.local_shape}, "
                    f"pre_dp_local_shape={self.pre_dp_local_shape}, dp_shard_axis={dp_axis}."
                )
        if (self.dp_local_start is None) != (self.dp_local_end is None):
            raise ValueError("MatrixShardSpec dp_local_start and dp_local_end must be set together.")
        if self.dp_local_start is not None:
            if self.dp_shard_axis not in (0, 1):
                raise ValueError(
                    "MatrixShardSpec DP local ranges require dp_shard_axis 0 or 1."
                )
            dp_axis = self.dp_shard_axis
            if self.dp_local_start < 0 or self.dp_local_end < self.dp_local_start:
                raise ValueError(
                    "MatrixShardSpec has an invalid DP local range: "
                    f"start={self.dp_local_start}, end={self.dp_local_end}."
                )
            if (
                self.pre_dp_local_shape is not None
                and self.dp_local_end > self.pre_dp_local_shape[dp_axis]
            ):
                raise ValueError(
                    "MatrixShardSpec DP local range exceeds the pre-DP local shape: "
                    f"end={self.dp_local_end}, pre_dp_local_shape={self.pre_dp_local_shape}, "
                    f"dp_shard_axis={dp_axis}."
                )
            expected_axis_size = self.dp_local_end - self.dp_local_start
            if self.local_shape[dp_axis] != expected_axis_size:
                raise ValueError(
                    "MatrixShardSpec local_shape DP-axis size must match the DP local range: "
                    f"local_shape={self.local_shape}, start={self.dp_local_start}, "
                    f"end={self.dp_local_end}, dp_shard_axis={dp_axis}."
                )

    @property
    def small_gram_side(self) -> Literal["right", "left"]:
        """Derived small-Gram side for this matrix shard contract."""

        return matrix_small_gram_side_for_spec(self)


MatrixSmallGramSide = Literal["right", "left"]
# Backwards-compatible name for older local tests/helpers. "Side" is the stable
# contract term: it records whether the small Gram is applied on the right
# (M.T @ M) or left (M @ M.T).
MatrixSmallGramOrientation = MatrixSmallGramSide


@dataclass
class MatrixInputPreconditionerRecipe:
    """Collection and consumption policy for input-side ``FEATURE_GRAM = X.T @ X``."""

    kind: MatrixPreconditionerKind
    approximation: MatrixPreconditionerApproximation
    scope: MatrixPreconditionerScope
    normalization: MatrixPreconditionerNormalization
    activation_dtype: Literal["bf16_saved", "fp32_cast", "fp8_dequant"]
    accumulation_dtype: torch.dtype
    refresh_interval: int
    token_sample_size: Optional[int]
    ridge: float
    ema_beta: Optional[float]
    min_samples_per_feature: Optional[float]
    block_size: int = 128


@dataclass
class MatrixOutputPreconditionerRecipe:
    """Collection and consumption policy for output-side ``GRAD_GRAM = dY.T @ dY``."""

    kind: MatrixPreconditionerKind
    approximation: MatrixPreconditionerApproximation
    scope: MatrixPreconditionerScope
    normalization: MatrixPreconditionerNormalization
    gradient_dtype: Literal["bf16_saved", "fp32_cast"]
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
    feature_gram_scope: Optional[MatrixPreconditionerScope]
    grad_gram_scope: Optional[MatrixPreconditionerScope]
    requires_full_logical_gradient: bool
    requires_full_logical_weight_for_direction: bool = False
    approximation_label: Optional[str] = None


def _enum_from_string(enum_type, value):
    if isinstance(value, enum_type):
        return value
    return enum_type(value)


def _tp_shard_axis_from_linear_weight_info(info: LinearWeightInfo) -> Optional[int]:
    if (
        info.tp_layout == "column_parallel"
        and len(info.local_shape) == 2
        and len(info.logical_shape) == 2
        and info.local_shape[0] != info.logical_shape[0]
    ):
        return 0
    if (
        info.tp_layout == "row_parallel"
        and len(info.local_shape) == 2
        and len(info.logical_shape) == 2
        and info.local_shape[1] != info.logical_shape[1]
    ):
        return 1
    return None


def matrix_shard_spec_from_linear_weight_info(info: LinearWeightInfo) -> MatrixShardSpec:
    return MatrixShardSpec(
        logical_shape=info.logical_shape,
        local_shape=info.local_shape,
        tp_layout=info.tp_layout,
        tp_shard_axis=_tp_shard_axis_from_linear_weight_info(info),
    )


def set_matrix_shard_spec(param: torch.nn.Parameter, spec: MatrixShardSpec) -> None:
    setattr(param, MATRIX_SHARD_SPEC_ATTR, spec)


def get_matrix_shard_spec(param: torch.nn.Parameter) -> Optional[MatrixShardSpec]:
    return getattr(param, MATRIX_SHARD_SPEC_ATTR, None)


def _ensure_matrix_shard_spec(param: torch.nn.Parameter) -> MatrixShardSpec:
    """Attach and return a MatrixShardSpec for a matrix-owned parameter.

    Affine weights should normally already carry LinearWeightInfo, which records
    TP layout and logical shape. Plain 2D Muon parameters do not necessarily have
    that metadata, so they get a conservative unsharded local matrix contract.
    """

    spec = get_matrix_shard_spec(param)
    if spec is not None:
        return spec
    info = getattr(param, "_mcore_linear_weight_info", None)
    if info is not None:
        spec = matrix_shard_spec_from_linear_weight_info(info)
    else:
        if param.dim() != 2:
            raise ValueError(
                "MatrixShardSpec can only be synthesized for 2D parameters, got "
                f"shape={tuple(param.shape)}."
            )
        spec = MatrixShardSpec(
            logical_shape=tuple(param.shape),
            local_shape=tuple(param.shape),
            tp_layout="none",
        )
    set_matrix_shard_spec(param, spec)
    return spec


def matrix_fsdp_shard_axis_for_spec(spec: MatrixShardSpec) -> int:
    """Return the future DP/FSDP matrix axis required by small-Gram Muon.

    This helper is a layout-planning rule, not active generic FSDP support. A
    future matrix-aware DP/FSDP shard should align with the existing matrix
    shard axis when TP already shards the matrix. Sharding the opposite axis
    would create 2D patches, which are not a valid input to the simple
    row/column small-Gram distributed Muon rule. If TP does not shard the
    matrix, choose the axis that keeps the small Gram on the smaller matrix
    dimension: row-axis shards for tall matrices and column-axis shards for wide
    matrices.
    """

    if spec.tp_shard_axis is None:
        rows, cols = spec.logical_shape
        return 0 if rows >= cols else 1
    return spec.tp_shard_axis


def matrix_small_gram_side_for_spec(
    spec: MatrixShardSpec,
) -> MatrixSmallGramSide:
    """Return the exact small-Gram side implied by a matrix shard spec.

    Row-axis shards use the small right Gram ``M.T @ M`` and locally apply
    ``M @ G^{-1/2}``; column-axis shards use the small left Gram ``M @ M.T``
    and locally apply ``G^{-1/2} @ M``. If TP and DP both shard the matrix,
    they must agree on the same axis, otherwise the local tensor is a 2D patch
    rather than a valid small-Gram Muon shard.
    """

    shard_axes = {axis for axis in (spec.tp_shard_axis, spec.dp_shard_axis) if axis is not None}
    if len(shard_axes) > 1:
        raise ValueError(
            "MatrixShardSpec has conflicting TP/DP shard axes; small-Gram Muon "
            f"requires a single matrix shard axis, got {sorted(shard_axes)}."
        )
    if shard_axes:
        axis = next(iter(shard_axes))
        if axis == 0:
            return "right"
        if axis == 1:
            return "left"
        raise ValueError(f"MatrixShardSpec shard axis must be 0 or 1, got {axis}.")

    rows, cols = spec.logical_shape
    return "right" if rows >= cols else "left"


def matrix_shard_spec_with_dp_axis(
    spec: MatrixShardSpec,
    *,
    dp_shard_axis: int,
    dp_local_start: Optional[int] = None,
    dp_local_end: Optional[int] = None,
) -> MatrixShardSpec:
    if dp_shard_axis not in (0, 1):
        raise ValueError("dp_shard_axis must be 0 or 1 for matrix parameters")
    if spec.tp_shard_axis is not None and dp_shard_axis != spec.tp_shard_axis:
        raise ValueError(
            "MatrixShardSpec cannot add a DP shard axis that differs from the TP "
            f"shard axis: tp_shard_axis={spec.tp_shard_axis}, dp_shard_axis={dp_shard_axis}."
        )
    local_shape = spec.local_shape
    if dp_local_start is not None or dp_local_end is not None:
        if dp_local_start is None or dp_local_end is None:
            raise ValueError("dp_local_start and dp_local_end must be provided together.")
        if dp_local_start < 0 or dp_local_end < dp_local_start:
            raise ValueError(
                f"Invalid DP local range: start={dp_local_start}, end={dp_local_end}."
            )
        pre_dp_local_shape = spec.pre_dp_local_shape or spec.local_shape
        if dp_local_end > pre_dp_local_shape[dp_shard_axis]:
            raise ValueError(
                "DP local range exceeds the pre-DP local matrix shape: "
                f"end={dp_local_end}, pre_dp_local_shape={pre_dp_local_shape}."
            )
        local_shape_list = list(pre_dp_local_shape)
        local_shape_list[dp_shard_axis] = dp_local_end - dp_local_start
        local_shape = tuple(local_shape_list)
    else:
        pre_dp_local_shape = spec.pre_dp_local_shape
    return MatrixShardSpec(
        logical_shape=spec.logical_shape,
        local_shape=local_shape,
        tp_layout=spec.tp_layout,
        tp_shard_axis=spec.tp_shard_axis,
        dp_shard_axis=dp_shard_axis,
        dp_local_start=dp_local_start,
        dp_local_end=dp_local_end,
        pre_dp_local_shape=pre_dp_local_shape,
    )


def set_matrix_optimizer_info(
    param: torch.nn.Parameter,
    *,
    owner: Literal["none", "muon", "matrix_function", "fallback"],
    update_family: Literal["none", "sgd", "muon"],
    requires_layerwise_layout: bool = False,
) -> MatrixOptimizerInfo:
    if owner not in _MATRIX_OPTIMIZER_OWNERS:
        raise ValueError(
            f"{owner!r} is not a stable matrix optimizer owner. Expected one of: "
            f"{', '.join(sorted(_MATRIX_OPTIMIZER_OWNERS))}."
        )
    if update_family not in _MATRIX_UPDATE_FAMILY_BY_OPTIMIZER_NAME.values():
        raise ValueError(
            f"{update_family!r} is not a stable matrix update_family. Expected one of: "
            f"{', '.join(_MATRIX_UPDATE_FAMILY_BY_OPTIMIZER_NAME)}."
        )
    if owner in (MATRIX_OPTIMIZER_OWNER_NONE, MATRIX_OPTIMIZER_OWNER_FALLBACK):
        if update_family != "none":
            raise ValueError(
                f"matrix optimizer owner {owner!r} requires update_family='none', "
                f"got {update_family!r}."
            )
    elif owner == MATRIX_OPTIMIZER_OWNER_MUON:
        if update_family != "muon":
            raise ValueError(
                "matrix optimizer owner 'muon' requires update_family='muon', "
                f"got {update_family!r}."
            )
    elif owner == MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION:
        if update_family == "none":
            raise ValueError(
                "matrix optimizer owner 'matrix_function' requires an active "
                "update_family, got 'none'."
            )
    info = MatrixOptimizerInfo(
        owner=owner,
        update_family=update_family,
        requires_layerwise_layout=requires_layerwise_layout,
    )
    setattr(param, MATRIX_OPTIMIZER_INFO_ATTR, info)
    set_matrix_optimizer_state_spec(
        param, matrix_optimizer_state_spec_from_info(info)
    )
    return info


def get_matrix_optimizer_info(param: torch.nn.Parameter) -> Optional[MatrixOptimizerInfo]:
    return getattr(param, MATRIX_OPTIMIZER_INFO_ATTR, None)


def matrix_optimizer_state_spec_from_info(
    info: MatrixOptimizerInfo,
) -> MatrixOptimizerStateSpec:
    if info.owner == MATRIX_OPTIMIZER_OWNER_MUON or info.update_family == "muon":
        return MatrixOptimizerStateSpec(
            same_shard_state_names=("master_param", "momentum_buffer"),
            allow_discovered_same_shard_state=True,
        )
    return MatrixOptimizerStateSpec()


def set_matrix_optimizer_state_spec(
    param: torch.nn.Parameter, spec: MatrixOptimizerStateSpec
) -> None:
    setattr(param, MATRIX_OPTIMIZER_STATE_SPEC_ATTR, spec)


def get_matrix_optimizer_state_spec(
    param: torch.nn.Parameter,
) -> Optional[MatrixOptimizerStateSpec]:
    spec = getattr(param, MATRIX_OPTIMIZER_STATE_SPEC_ATTR, None)
    if spec is not None:
        return spec
    info = get_matrix_optimizer_info(param)
    if info is None:
        return None
    spec = matrix_optimizer_state_spec_from_info(info)
    set_matrix_optimizer_state_spec(param, spec)
    return spec


def get_matrix_optimizer_owner(param: torch.nn.Parameter) -> str:
    info = get_matrix_optimizer_info(param)
    if info is None:
        return MATRIX_OPTIMIZER_OWNER_NONE
    return info.owner


def is_matrix_optimizer_owned_parameter(param: torch.nn.Parameter) -> bool:
    return get_matrix_optimizer_owner(param) in (
        MATRIX_OPTIMIZER_OWNER_MUON,
        MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
    )


def requires_matrix_layerwise_layout(param: torch.nn.Parameter) -> bool:
    """Return whether a parameter must be routed through LayerWise buffers."""

    info = get_matrix_optimizer_info(param)
    if info is None:
        return False
    return is_matrix_optimizer_owned_parameter(param) and info.requires_layerwise_layout


def is_matrix_optimizer_fallback_parameter(param: torch.nn.Parameter) -> bool:
    """Return whether a parameter is explicitly owned by the fallback optimizer."""

    return get_matrix_optimizer_owner(param) == MATRIX_OPTIMIZER_OWNER_FALLBACK


def register_matrix_optimizer_param(
    param: torch.nn.Parameter,
    *,
    owner: Literal["none", "muon", "matrix_function", "fallback"],
    update_family: Literal["none", "sgd", "muon"],
    requires_layerwise_layout: bool = False,
    ensure_shard_spec: bool = True,
) -> MatrixOptimizerInfo:
    """Register matrix-optimizer ownership and derived routing metadata.

    This is the single boundary for optimizer ownership metadata. Callers should
    not separately set ``MatrixOptimizerInfo``, synthesize ``MatrixShardSpec``,
    and tag LayerWise routing; those invariants must move together.
    """

    info = set_matrix_optimizer_info(
        param,
        owner=owner,
        update_family=update_family,
        requires_layerwise_layout=requires_layerwise_layout,
    )
    if owner in (MATRIX_OPTIMIZER_OWNER_MUON, MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION):
        if ensure_shard_spec:
            _ensure_matrix_shard_spec(param)
        param.is_managed_by_layer_wise_optimizer = info.requires_layerwise_layout
    else:
        param.is_managed_by_layer_wise_optimizer = False
    return info


def copy_matrix_optimizer_registration(
    src_param: torch.nn.Parameter,
    dst_param: torch.Tensor,
    *,
    shard_spec: Optional[MatrixShardSpec] = None,
) -> Optional[MatrixOptimizerInfo]:
    """Copy validated matrix optimizer metadata from one param object to another."""

    info = get_matrix_optimizer_info(src_param)
    copied_info = None
    if info is not None:
        copied_info = set_matrix_optimizer_info(
            dst_param,
            owner=info.owner,
            update_family=info.update_family,
            requires_layerwise_layout=info.requires_layerwise_layout,
        )
        dst_param.is_managed_by_layer_wise_optimizer = (
            info.owner in (MATRIX_OPTIMIZER_OWNER_MUON, MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION)
            and info.requires_layerwise_layout
        )
        state_spec = get_matrix_optimizer_state_spec(src_param)
        if state_spec is not None:
            set_matrix_optimizer_state_spec(dst_param, state_spec)
    spec = shard_spec if shard_spec is not None else get_matrix_shard_spec(src_param)
    if spec is not None:
        set_matrix_shard_spec(dst_param, spec)
    return copied_info


def input_preconditioner_scope_for(
    approximation: MatrixPreconditionerApproximation,
    tp_layout: str,
    *,
    is_feature_axis_sharded: bool = False,
) -> MatrixPreconditionerScope:
    """Infer the factor scope implied by approximation and TP layout.

    Row-parallel linears own only a feature-axis shard. Until cross-TP Gram
    support exists, any local row-parallel factor is TP-local metadata rather
    than a global exact or generic diagonal approximation.
    """

    if tp_layout == "row_parallel" and is_feature_axis_sharded:
        return MatrixPreconditionerScope.TP_LOCAL_BLOCK_DIAG
    if approximation == MatrixPreconditionerApproximation.BLOCK_DIAG:
        return MatrixPreconditionerScope.BLOCK_DIAG_APPROX
    if approximation == MatrixPreconditionerApproximation.DIAG:
        return MatrixPreconditionerScope.DIAG_APPROX
    return MatrixPreconditionerScope.GLOBAL_EXACT


def output_preconditioner_scope_for(
    approximation: MatrixPreconditionerApproximation,
    tp_layout: str,
    *,
    is_output_axis_sharded: bool = False,
) -> MatrixPreconditionerScope:
    """Infer output-side factor scope implied by approximation and TP layout.

    Column-parallel linears own only an output-axis shard, so their local
    ``dY.T @ dY`` is a TP-local block of the logical output Gram. Row-parallel
    linears own the full output axis on every rank and can consume the same
    output-side factor for each input shard.
    """

    if tp_layout == "column_parallel" and is_output_axis_sharded:
        return MatrixPreconditionerScope.TP_LOCAL_BLOCK_DIAG
    if approximation == MatrixPreconditionerApproximation.BLOCK_DIAG:
        return MatrixPreconditionerScope.BLOCK_DIAG_APPROX
    if approximation == MatrixPreconditionerApproximation.DIAG:
        return MatrixPreconditionerScope.DIAG_APPROX
    return MatrixPreconditionerScope.GLOBAL_EXACT


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
    info = LinearWeightInfo(
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
    )
    setattr(
        param,
        "_mcore_linear_weight_info",
        info,
    )
    set_matrix_shard_spec(param, matrix_shard_spec_from_linear_weight_info(info))
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


def _output_dim(param: torch.nn.Parameter) -> int:
    info = getattr(param, "_mcore_linear_weight_info", None)
    if info is None:
        if param.ndim != 2:
            raise ValueError("GRAD_GRAM requires a 2D affine weight or LinearWeightInfo.")
        return param.shape[0]
    return info.local_shape[0]


def _is_row_parallel_feature_axis_sharded(info: LinearWeightInfo) -> bool:
    return (
        info.tp_layout == "row_parallel"
        and len(info.local_shape) == 2
        and len(info.logical_shape) == 2
        and info.local_shape[1] != info.logical_shape[1]
    )


def _is_column_parallel_output_axis_sharded(info: LinearWeightInfo) -> bool:
    return (
        info.tp_layout == "column_parallel"
        and len(info.local_shape) == 2
        and len(info.logical_shape) == 2
        and info.local_shape[0] != info.logical_shape[0]
    )


def _feature_gram_shape(param: torch.nn.Parameter, recipe: MatrixInputPreconditionerRecipe) -> tuple[int, ...]:
    dim = _feature_dim(param)
    if recipe.approximation == MatrixPreconditionerApproximation.FULL:
        return (dim, dim)
    if recipe.approximation == MatrixPreconditionerApproximation.DIAG:
        return (dim,)
    if recipe.approximation == MatrixPreconditionerApproximation.BLOCK_DIAG:
        num_blocks = (dim + recipe.block_size - 1) // recipe.block_size
        return (num_blocks, recipe.block_size, recipe.block_size)
    raise NotImplementedError(
        f"FEATURE_GRAM approximation {recipe.approximation.value!r} is not implemented yet. "
        "Use 'full', 'diag', or 'block_diag', or add an explicit storage format for this approximation."
    )


def _grad_gram_shape(param: torch.nn.Parameter, recipe: MatrixOutputPreconditionerRecipe) -> tuple[int, ...]:
    dim = _output_dim(param)
    if recipe.approximation == MatrixPreconditionerApproximation.FULL:
        return (dim, dim)
    if recipe.approximation == MatrixPreconditionerApproximation.DIAG:
        return (dim,)
    if recipe.approximation == MatrixPreconditionerApproximation.BLOCK_DIAG:
        num_blocks = (dim + recipe.block_size - 1) // recipe.block_size
        return (num_blocks, recipe.block_size, recipe.block_size)
    raise NotImplementedError(
        f"GRAD_GRAM approximation {recipe.approximation.value!r} is not implemented yet. "
        "Use 'full', 'diag', or 'block_diag', or add an explicit storage format for this approximation."
    )


def _validate_feature_gram_recipe(param: torch.nn.Parameter, recipe: MatrixInputPreconditionerRecipe) -> None:
    if recipe.refresh_interval < 1:
        raise ValueError("matrix-input-preconditioner-refresh-interval must be >= 1")
    if recipe.block_size < 1:
        raise ValueError("matrix-input-preconditioner-block-size must be >= 1")
    if recipe.ridge < 0.0:
        raise ValueError("matrix-input-preconditioner-ridge must be >= 0")
    if recipe.token_sample_size is not None and recipe.token_sample_size < 1:
        raise ValueError("matrix-input-preconditioner-token-sample-size must be >= 1 when set")
    feature_dim = _feature_dim(param)
    if (
        recipe.approximation == MatrixPreconditionerApproximation.BLOCK_DIAG
        and feature_dim % recipe.block_size != 0
        and recipe.ridge == 0.0
    ):
        raise ValueError(
            "block_diag FEATURE_GRAM with a non-divisible feature dimension requires "
            "matrix-input-preconditioner-ridge > 0; otherwise padded coordinates make the "
            "last block singular."
        )
    if recipe.min_samples_per_feature is not None and recipe.token_sample_size is not None:
        min_samples = recipe.min_samples_per_feature * feature_dim
        if (
            recipe.approximation == MatrixPreconditionerApproximation.FULL
            and recipe.token_sample_size < min_samples
        ):
            raise ValueError(
                "Full FEATURE_GRAM is rank-deficient under this token sample size; "
                "use diag/block_diag or increase matrix-input-preconditioner-token-sample-size."
            )


def _validate_grad_gram_recipe(param: torch.nn.Parameter, recipe: MatrixOutputPreconditionerRecipe) -> None:
    if recipe.refresh_interval < 1:
        raise ValueError("matrix-output-preconditioner-refresh-interval must be >= 1")
    if recipe.block_size < 1:
        raise ValueError("matrix-output-preconditioner-block-size must be >= 1")
    if recipe.ridge < 0.0:
        raise ValueError("matrix-output-preconditioner-ridge must be >= 0")
    if recipe.token_sample_size is not None and recipe.token_sample_size < 1:
        raise ValueError("matrix-output-preconditioner-token-sample-size must be >= 1 when set")
    output_dim = _output_dim(param)
    if (
        recipe.approximation == MatrixPreconditionerApproximation.BLOCK_DIAG
        and output_dim % recipe.block_size != 0
        and recipe.ridge == 0.0
    ):
        raise ValueError(
            "block_diag GRAD_GRAM with a non-divisible output dimension requires "
            "matrix-output-preconditioner-ridge > 0; otherwise padded coordinates make the "
            "last block singular."
        )
    if recipe.min_samples_per_feature is not None and recipe.token_sample_size is not None:
        min_samples = recipe.min_samples_per_feature * output_dim
        if (
            recipe.approximation == MatrixPreconditionerApproximation.FULL
            and recipe.token_sample_size < min_samples
        ):
            raise ValueError(
                "Full GRAD_GRAM is rank-deficient under this token sample size; "
                "use diag/block_diag or increase matrix-output-preconditioner-token-sample-size."
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


def _mark_grad_gram_unfinalized(param: torch.nn.Parameter) -> None:
    param._grad_gram_finalized = not getattr(param, "_grad_gram_finalization_required", False)


def set_feature_gram_finalization_required(
    params: Iterable[torch.nn.Parameter], *, required: bool
) -> None:
    for param in params:
        param._feature_gram_finalization_required = required
        _mark_feature_gram_unfinalized(param)


def set_grad_gram_finalization_required(
    params: Iterable[torch.nn.Parameter], *, required: bool
) -> None:
    for param in params:
        param._grad_gram_finalization_required = required
        _mark_grad_gram_unfinalized(param)


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
    params = [
        param
        for param in params
        if hasattr(param, "main_grad_feature_gram")
        and not getattr(param, "_feature_gram_finalized", False)
    ]
    if any(getattr(param, "_feature_gram_finalization_required", False) for param in params) and not groups:
        raise RuntimeError(
            "FEATURE_GRAM finalization requires explicit process_groups for these parameters."
        )

    for param in params:
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


def finalize_grad_gram_buffers(
    params: Iterable[torch.nn.Parameter],
    *,
    process_groups: Iterable[torch.distributed.ProcessGroup] = (),
) -> None:
    """Reduce output-side GRAD_GRAM buffers across caller-specified groups."""

    groups = tuple(process_groups)
    params = [
        param
        for param in params
        if hasattr(param, "main_grad_grad_gram")
        and not getattr(param, "_grad_gram_finalized", False)
    ]
    if any(getattr(param, "_grad_gram_finalization_required", False) for param in params) and not groups:
        raise RuntimeError(
            "GRAD_GRAM finalization requires explicit process_groups for these parameters."
        )

    for param in params:
        for group in groups:
            torch.distributed.all_reduce(
                param.main_grad_grad_gram, op=torch.distributed.ReduceOp.SUM, group=group
            )
            torch.distributed.all_reduce(
                param.main_grad_grad_count, op=torch.distributed.ReduceOp.SUM, group=group
            )
        param._grad_gram_finalized = True


def allocate_feature_gram_buffers(param: torch.nn.Parameter, recipe: MatrixInputPreconditionerRecipe) -> None:
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


def allocate_grad_gram_buffers(param: torch.nn.Parameter, recipe: MatrixOutputPreconditionerRecipe) -> None:
    """Allocate per-parameter output-side GRAD_GRAM buffers for a requested recipe."""

    _validate_grad_gram_recipe(param, recipe)
    gram_shape = _grad_gram_shape(param, recipe)
    gram = getattr(param, "main_grad_grad_gram", None)
    if (
        gram is None
        or tuple(gram.shape) != gram_shape
        or gram.dtype != recipe.accumulation_dtype
        or gram.device != param.device
    ):
        param.main_grad_grad_gram = torch.zeros(
            gram_shape,
            device=param.device,
            dtype=recipe.accumulation_dtype,
        )
    if not hasattr(param, "main_grad_grad_count"):
        param.main_grad_grad_count = torch.zeros((), device=param.device, dtype=torch.float64)
    param._grad_gram_recipe = recipe
    param._grad_gram_scope = recipe.scope
    if not hasattr(param, "_grad_gram_generation"):
        param._grad_gram_generation = 0
    _mark_grad_gram_unfinalized(param)


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


def reset_grad_gram_buffers(
    param: torch.nn.Parameter, *, active: bool = True, zero: bool = True
) -> None:
    """Prepare transient output-side factor buffers for a collection window."""

    if zero and hasattr(param, "main_grad_grad_gram"):
        param.main_grad_grad_gram.zero_()
    if zero and hasattr(param, "main_grad_grad_count"):
        param.main_grad_grad_count.zero_()
    if zero:
        param._grad_gram_generation = getattr(param, "_grad_gram_generation", 0) + 1
        _mark_grad_gram_unfinalized(param)
    param._grad_gram_active = active


def iter_matrix_update_params(modules: Iterable[torch.nn.Module]):
    """Yield parameters that carry matrix-update metadata."""

    for module in modules:
        for param in module.parameters():
            if hasattr(param, "_mcore_linear_weight_info"):
                yield param


def configure_matrix_update_param(
    param: torch.nn.Parameter,
    *,
    recipe: Optional[MatrixInputPreconditionerRecipe] = None,
    output_recipe: Optional[MatrixOutputPreconditionerRecipe] = None,
    factors: ExtraWgradFactor = ExtraWgradFactor.FEATURE_GRAM,
) -> None:
    """Enable requested extra wgrad factors on an eligible parameter."""

    enabled_factors = factors
    if recipe is None:
        enabled_factors &= ~ExtraWgradFactor.FEATURE_GRAM
    if output_recipe is not None:
        enabled_factors |= ExtraWgradFactor.GRAD_GRAM
    param._extra_wgrad_factors = enabled_factors
    if recipe is not None:
        allocate_feature_gram_buffers(param, recipe)
        reset_feature_gram_buffers(param, active=True)
    if output_recipe is not None:
        allocate_grad_gram_buffers(param, output_recipe)
        reset_grad_gram_buffers(param, active=True)


def recipe_from_optimizer_config(config, info: LinearWeightInfo) -> MatrixInputPreconditionerRecipe:
    """Build a feature Gram recipe from an ``OptimizerConfig``-like object."""

    approximation = _enum_from_string(MatrixPreconditionerApproximation, config.matrix_input_preconditioner_approximation)
    scope = input_preconditioner_scope_for(
        approximation,
        info.tp_layout,
        is_feature_axis_sharded=_is_row_parallel_feature_axis_sharded(info),
    )
    normalization = _enum_from_string(
        MatrixPreconditionerNormalization, config.matrix_input_preconditioner_normalization
    )
    return MatrixInputPreconditionerRecipe(
        kind=MatrixPreconditionerKind.FEATURE_GRAM,
        approximation=approximation,
        scope=scope,
        normalization=normalization,
        activation_dtype=config.matrix_input_preconditioner_activation_dtype,
        accumulation_dtype=getattr(config, "matrix_input_preconditioner_accumulation_dtype", torch.float32),
        refresh_interval=config.matrix_input_preconditioner_refresh_interval,
        token_sample_size=config.matrix_input_preconditioner_token_sample_size,
        ridge=config.matrix_input_preconditioner_ridge,
        ema_beta=config.matrix_input_preconditioner_ema_beta,
        min_samples_per_feature=config.matrix_input_preconditioner_min_samples_per_feature,
        block_size=getattr(config, "matrix_input_preconditioner_block_size", 128),
    )


def output_recipe_from_optimizer_config(config, info: LinearWeightInfo) -> MatrixOutputPreconditionerRecipe:
    """Build a grad Gram recipe from an ``OptimizerConfig``-like object."""

    approximation = _enum_from_string(MatrixPreconditionerApproximation, config.matrix_output_preconditioner_approximation)
    scope = output_preconditioner_scope_for(
        approximation,
        info.tp_layout,
        is_output_axis_sharded=_is_column_parallel_output_axis_sharded(info),
    )
    normalization = _enum_from_string(
        MatrixPreconditionerNormalization, config.matrix_output_preconditioner_normalization
    )
    return MatrixOutputPreconditionerRecipe(
        kind=MatrixPreconditionerKind.GRAD_GRAM,
        approximation=approximation,
        scope=scope,
        normalization=normalization,
        gradient_dtype=config.matrix_output_preconditioner_gradient_dtype,
        accumulation_dtype=getattr(config, "matrix_output_preconditioner_accumulation_dtype", torch.float32),
        refresh_interval=config.matrix_output_preconditioner_refresh_interval,
        token_sample_size=config.matrix_output_preconditioner_token_sample_size,
        ridge=config.matrix_output_preconditioner_ridge,
        ema_beta=config.matrix_output_preconditioner_ema_beta,
        min_samples_per_feature=config.matrix_output_preconditioner_min_samples_per_feature,
        block_size=getattr(config, "matrix_output_preconditioner_block_size", 128),
    )


def configure_model_matrix_updates(modules: Iterable[torch.nn.Module], config) -> list[torch.nn.Parameter]:
    """Enable requested extra wgrad factor collection for eligible parameters.

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
        input_recipe = None
        output_recipe = None
        factors = ExtraWgradFactor.NONE
        register_matrix_optimizer_param(
            param,
            owner=MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
            update_family=matrix_update_family_from_optimizer_name(config.matrix_optimizer),
            requires_layerwise_layout=True,
        )
        if config.matrix_input_preconditioner == "feature_gram":
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
                _is_row_parallel_feature_axis_sharded(info)
                and recipe.approximation == MatrixPreconditionerApproximation.FULL
            ):
                raise RuntimeError(
                    "matrix-input-preconditioner-approximation=full is not supported for "
                    "row-parallel weights without cross-TP FEATURE_GRAM collection; use "
                    "diag/block_diag or disable the matrix optimizer for this parameter "
                    f"(role={info.role!r})."
                )
            if collector == "native" and recipe.activation_dtype == "fp8_dequant":
                raise RuntimeError(
                    "FEATURE_GRAM activation_dtype=fp8_dequant requires Transformer Engine "
                    "collection at the wgrad site; native linears cannot dequantize FP8 sources."
                )
            input_recipe = recipe
            factors |= ExtraWgradFactor.FEATURE_GRAM
        if config.matrix_output_preconditioner == "grad_gram":
            collector = getattr(param, "_feature_gram_collector", "unknown")
            if collector != "native":
                raise RuntimeError(
                    f"GRAD_GRAM collection for {collector!r} linears is not available in this "
                    f"checkout (role={info.role!r}, tp_layout={info.tp_layout!r})."
                )
            recipe = output_recipe_from_optimizer_config(config, info)
            if (
                _is_column_parallel_output_axis_sharded(info)
                and recipe.approximation == MatrixPreconditionerApproximation.FULL
            ):
                raise RuntimeError(
                    "matrix-output-preconditioner-approximation=full is not supported for "
                    "column-parallel weights without cross-TP GRAD_GRAM collection; use "
                    "diag/block_diag or disable the matrix optimizer for this parameter "
                    f"(role={info.role!r})."
                )
            output_recipe = recipe
            factors |= ExtraWgradFactor.GRAD_GRAM
        if factors != ExtraWgradFactor.NONE:
            configure_matrix_update_param(
                param, recipe=input_recipe, output_recipe=output_recipe, factors=factors
            )
        configured.append(param)
    return configured


def _cast_feature_input(inputmat: torch.Tensor, recipe: MatrixInputPreconditionerRecipe) -> torch.Tensor:
    if recipe.activation_dtype not in ("bf16_saved", "fp32_cast", "fp8_dequant"):
        raise ValueError(f"Unsupported feature Gram activation dtype: {recipe.activation_dtype}")
    if recipe.activation_dtype == "fp8_dequant":
        raise NotImplementedError(
            "FEATURE_GRAM activation_dtype=fp8_dequant requires Transformer Engine dequantization "
            "support at the wgrad site; this native collector fails closed."
        )
    if recipe.activation_dtype == "fp32_cast":
        return inputmat.to(torch.float32)
    return inputmat.to(recipe.accumulation_dtype)


def _cast_grad_output(grad_output: torch.Tensor, recipe: MatrixOutputPreconditionerRecipe) -> torch.Tensor:
    if recipe.gradient_dtype not in ("bf16_saved", "fp32_cast"):
        raise ValueError(f"Unsupported grad Gram gradient dtype: {recipe.gradient_dtype}")
    if recipe.gradient_dtype == "fp32_cast":
        return grad_output.to(torch.float32)
    return grad_output.to(recipe.accumulation_dtype)


def _try_accumulate_cuda_diag_gram(gram: torch.Tensor, value: torch.Tensor) -> bool:
    if not (gram.is_cuda and value.is_cuda):
        return False
    try:
        from emerging_optimizers.triton_kernels.diag_gram import diag_gram_reduce
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "")
        if missing_name and not missing_name.startswith("emerging_optimizers"):
            raise
        return False
    except ImportError:
        return False

    diag_gram_reduce(value, out=gram, accumulate=True)
    return True


def _accumulate_diag_feature_gram(gram: torch.Tensor, x: torch.Tensor) -> None:
    if _try_accumulate_cuda_diag_gram(gram, x):
        return
    gram.add_(torch.sum(x * x, dim=0))


def _accumulate_diag_grad_gram(gram: torch.Tensor, dy: torch.Tensor) -> None:
    if _try_accumulate_cuda_diag_gram(gram, dy):
        return
    gram.add_(torch.sum(dy * dy, dim=0))


def _accumulate_block_diag_feature_gram(
    gram: torch.Tensor,
    x: torch.Tensor,
    *,
    feature_dim: int,
    block_size: int,
) -> None:
    """Accumulate padded block-diagonal Gram storage via batched GEMM."""

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
        raise RuntimeError("FEATURE_GRAM requested without a MatrixInputPreconditionerRecipe.")
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
    if recipe.approximation == MatrixPreconditionerApproximation.FULL:
        gram.add_(x.t().matmul(x))
    elif recipe.approximation == MatrixPreconditionerApproximation.DIAG:
        _accumulate_diag_feature_gram(gram, x)
    elif recipe.approximation == MatrixPreconditionerApproximation.BLOCK_DIAG:
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


@torch.no_grad()
def maybe_accumulate_grad_gram(weight: torch.Tensor, grad_output: torch.Tensor) -> None:
    """Accumulate ``dY.T @ dY`` for a linear weight when requested."""

    flags = getattr(weight, "_extra_wgrad_factors", ExtraWgradFactor.NONE)
    if not (flags & ExtraWgradFactor.GRAD_GRAM):
        return
    if not getattr(weight, "_grad_gram_active", True):
        return
    recipe = getattr(weight, "_grad_gram_recipe", None)
    if recipe is None:
        raise RuntimeError("GRAD_GRAM requested without a MatrixOutputPreconditionerRecipe.")
    if grad_output.dim() != 2:
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    dy = _cast_grad_output(grad_output, recipe)
    if recipe.token_sample_size is not None:
        remaining = recipe.token_sample_size - int(weight.main_grad_grad_count.item())
        if remaining <= 0:
            return
        if dy.shape[0] > remaining:
            dy = dy[:remaining]

    gram = weight.main_grad_grad_gram
    if recipe.approximation == MatrixPreconditionerApproximation.FULL:
        gram.add_(dy.t().matmul(dy))
    elif recipe.approximation == MatrixPreconditionerApproximation.DIAG:
        _accumulate_diag_grad_gram(gram, dy)
    elif recipe.approximation == MatrixPreconditionerApproximation.BLOCK_DIAG:
        _accumulate_block_diag_feature_gram(
            gram,
            dy,
            feature_dim=_output_dim(weight),
            block_size=recipe.block_size,
        )
    else:
        raise NotImplementedError(
            f"GRAD_GRAM approximation {recipe.approximation.value!r} is not implemented yet."
        )
    weight.main_grad_grad_count.add_(float(dy.shape[0]))
    _mark_grad_gram_unfinalized(weight)


def get_feature_gram_for_optimizer(param: torch.nn.Parameter) -> torch.Tensor:
    """Return the feature Gram using the recipe's consumption normalization."""

    recipe = getattr(param, "_feature_gram_recipe", None)
    if recipe is None:
        raise RuntimeError("Parameter has no MatrixInputPreconditionerRecipe.")
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
    if recipe.normalization == MatrixPreconditionerNormalization.MEAN:
        count = param.main_grad_feature_count.clamp_min(1.0).to(gram.dtype)
        return gram / count
    return gram


def get_grad_gram_for_optimizer(param: torch.nn.Parameter) -> torch.Tensor:
    """Return the grad Gram using the recipe's consumption normalization."""

    recipe = getattr(param, "_grad_gram_recipe", None)
    if recipe is None:
        raise RuntimeError("Parameter has no MatrixOutputPreconditionerRecipe.")
    if _needs_distributed_finalization() and not getattr(param, "_grad_gram_finalized", False):
        raise RuntimeError(
            "GRAD_GRAM has not been finalized across distributed groups; call "
            "finalize_grad_gram_buffers before optimizer consumption."
        )
    gram = param.main_grad_grad_gram
    if param.main_grad_grad_count.item() <= 0.0:
        raise RuntimeError(
            "GRAD_GRAM has zero collected grad-output rows; ensure the wgrad path collected "
            "main_grad_grad_gram before optimizer consumption."
        )
    if recipe.normalization == MatrixPreconditionerNormalization.MEAN:
        count = param.main_grad_grad_count.clamp_min(1.0).to(gram.dtype)
        return gram / count
    return gram
