# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import types
from enum import IntEnum
from typing import Callable, Optional, Sequence, Type

import torch
from torch.distributed import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from .megatron_fsdp import MegatronFSDP
from .mixed_precision import MixedPrecisionPolicy
from .uneven_dtensor import preprocess_state_dict_for_uneven_dtensor
from .utils import FSDPDistributedIndex, create_updated_function_signature

try:
    # Default to Megatron-LM FW.
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
except ImportError:
    # Megatron-LM is not installed, use Megatron-FSDP as a standalone module.
    from .distributed_data_parallel_config import DistributedDataParallelConfig

try:
    from megatron.core.matrix_update import (
        MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
        MATRIX_OPTIMIZER_OWNER_MUON,
        get_matrix_optimizer_info,
        get_matrix_shard_spec,
        matrix_small_gram_side_for_spec,
        set_matrix_shard_spec,
    )

    HAVE_MCORE_MATRIX_UPDATE = True
except ImportError:
    HAVE_MCORE_MATRIX_UPDATE = False


logger = logging.getLogger(__name__)

MATRIX_OPTIMIZER_STATE_METADATA_KEY = "_mcore_matrix_optimizer_state"
MATRIX_OPTIMIZER_STATE_METADATA_VERSION = 6
MATRIX_OPTIMIZER_SAME_SHARD_STATE_LAYOUT = "same_as_param"
MATRIX_OPTIMIZER_DP_SHARD_LAYOUT_ROW_CONTIGUOUS = "row_contiguous_flat_buffer"
MATRIX_OPTIMIZER_DP_SHARD_LAYOUT_COLUMN_CONTIGUOUS = "column_contiguous_flat_buffer"


def _is_matrix_optimizer_owned_param(param: torch.Tensor) -> bool:
    if not HAVE_MCORE_MATRIX_UPDATE:
        return False
    matrix_optimizer_info = get_matrix_optimizer_info(param)
    return matrix_optimizer_info is not None and getattr(matrix_optimizer_info, "owner", None) in (
        MATRIX_OPTIMIZER_OWNER_MUON,
        MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
    )


def _matrix_shard_spec_to_checkpoint_dict(spec) -> dict:
    return {
        "logical_shape": list(spec.logical_shape),
        "local_shape": list(spec.local_shape),
        "pre_dp_local_shape": (
            list(spec.pre_dp_local_shape) if spec.pre_dp_local_shape is not None else None
        ),
        "tp_layout": spec.tp_layout,
        "tp_shard_axis": spec.tp_shard_axis,
        "dp_shard_axis": spec.dp_shard_axis,
        "dp_shard_layout": _matrix_dp_shard_layout_to_checkpoint_value(spec),
        "dp_local_start": spec.dp_local_start,
        "dp_local_end": spec.dp_local_end,
        "small_gram_side": matrix_small_gram_side_for_spec(spec),
    }


def _matrix_shard_global_contract_to_checkpoint_dict(spec) -> dict:
    """Return MatrixShardSpec fields that must match across checkpoint resume.

    ``dp_local_start`` and ``dp_local_end`` are rank-local residency metadata.
    They are useful to save for debugging, but they are not stable under DP
    resharding and should not be used as the global checkpoint compatibility
    contract. ``local_shape`` is also saved in the full spec, but after FSDP
    row slicing it describes this rank's current matrix rows and is therefore
    not a global resume key.
    """

    return {
        "logical_shape": list(spec.logical_shape),
        "tp_layout": spec.tp_layout,
        "tp_shard_axis": spec.tp_shard_axis,
        "dp_shard_axis": spec.dp_shard_axis,
        "dp_shard_layout": _matrix_dp_shard_layout_to_checkpoint_value(spec),
        "small_gram_side": matrix_small_gram_side_for_spec(spec),
    }


def _matrix_shard_global_contract_from_checkpoint_dict(metadata: dict) -> dict:
    return {
        "logical_shape": metadata.get("logical_shape"),
        "tp_layout": metadata.get("tp_layout"),
        "tp_shard_axis": metadata.get("tp_shard_axis"),
        "dp_shard_axis": metadata.get("dp_shard_axis"),
        "dp_shard_layout": metadata.get("dp_shard_layout"),
        "small_gram_side": metadata.get("small_gram_side"),
    }


def _matrix_shard_spec_matches_checkpoint_dict(spec, metadata: dict) -> bool:
    return _matrix_shard_global_contract_to_checkpoint_dict(
        spec
    ) == _matrix_shard_global_contract_from_checkpoint_dict(metadata)


def _validate_matrix_shape_checkpoint_field(
    metadata: dict, field_name: str, param_idx: str
) -> None:
    shape = metadata.get(field_name)
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata field "
            f"{field_name!r} for optimizer state index {param_idx} must be a list of integers."
        )


def _validate_optional_matrix_shape_checkpoint_field(
    metadata: dict, field_name: str, param_idx: str
) -> None:
    shape = metadata.get(field_name)
    if shape is None:
        return
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata field "
            f"{field_name!r} for optimizer state index {param_idx} must be None or "
            "a list of integers."
        )


def _validate_matrix_shard_spec_checkpoint_metadata(
    metadata: object, param_idx: str, field_name: str
) -> None:
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata field "
            f"{field_name!r} for optimizer state index {param_idx} must be a dict."
        )
    _validate_matrix_shape_checkpoint_field(metadata, "logical_shape", param_idx)
    _validate_matrix_shape_checkpoint_field(metadata, "local_shape", param_idx)
    _validate_optional_matrix_shape_checkpoint_field(
        metadata, "pre_dp_local_shape", param_idx
    )
    logical_shape = metadata.get("logical_shape")
    local_shape = metadata.get("local_shape")
    pre_dp_local_shape = metadata.get("pre_dp_local_shape")
    if len(logical_shape) != 2 or len(local_shape) != 2:
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata shapes must be 2D "
            f"for optimizer state index {param_idx}: logical_shape={logical_shape}, "
            f"local_shape={local_shape}."
        )
    if pre_dp_local_shape is not None:
        if len(pre_dp_local_shape) != 2:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata pre_dp_local_shape "
                f"must be 2D for optimizer state index {param_idx}: "
                f"pre_dp_local_shape={pre_dp_local_shape}."
            )
        if local_shape[0] > pre_dp_local_shape[0] or local_shape[1:] != pre_dp_local_shape[1:]:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata local_shape is "
                f"inconsistent with pre_dp_local_shape for optimizer state index {param_idx}: "
                f"local_shape={local_shape}, pre_dp_local_shape={pre_dp_local_shape}."
            )
    dp_shard_axis = metadata.get("dp_shard_axis")
    dp_shard_layout = metadata.get("dp_shard_layout")
    if dp_shard_axis is None:
        if dp_shard_layout is not None:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata has a DP shard "
                f"layout without a DP shard axis for optimizer state index {param_idx}."
            )
    elif dp_shard_axis == 0:
        if dp_shard_layout != MATRIX_OPTIMIZER_DP_SHARD_LAYOUT_ROW_CONTIGUOUS:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata has unsupported "
                f"DP shard layout {dp_shard_layout!r} for optimizer state index {param_idx}."
            )
    elif dp_shard_axis == 1:
        if dp_shard_layout != MATRIX_OPTIMIZER_DP_SHARD_LAYOUT_COLUMN_CONTIGUOUS:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata has unsupported "
                f"DP shard layout {dp_shard_layout!r} for optimizer state index {param_idx}."
            )
    else:
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata has unsupported DP "
            f"shard axis {dp_shard_axis!r} for optimizer state index {param_idx}."
        )
    dp_local_start = metadata.get("dp_local_start")
    dp_local_end = metadata.get("dp_local_end")
    if (dp_local_start is None) != (dp_local_end is None):
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata dp_local_start and "
            f"dp_local_end must be set together for optimizer state index {param_idx}."
        )
    if dp_local_start is not None:
        if not isinstance(dp_local_start, int) or not isinstance(dp_local_end, int):
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata DP local row "
                f"range must use integer offsets for optimizer state index {param_idx}."
            )
        if dp_shard_axis != 0 or dp_local_start < 0 or dp_local_end < dp_local_start:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata has an invalid "
                f"DP local row range for optimizer state index {param_idx}: "
                f"axis={dp_shard_axis}, start={dp_local_start}, end={dp_local_end}."
            )
        if local_shape[0] != dp_local_end - dp_local_start:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata local_shape row "
                f"count does not match DP local row range for optimizer state index "
                f"{param_idx}: local_shape={local_shape}, start={dp_local_start}, "
                f"end={dp_local_end}."
            )
        if pre_dp_local_shape is not None and dp_local_end > pre_dp_local_shape[0]:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata DP local row "
                f"range exceeds pre_dp_local_shape for optimizer state index {param_idx}: "
                f"end={dp_local_end}, pre_dp_local_shape={pre_dp_local_shape}."
            )


def _validate_matrix_shard_contract_checkpoint_metadata(
    metadata: object, param_idx: str
) -> None:
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata field "
            f"'matrix_shard_contract' for optimizer state index {param_idx} must be a dict."
        )
    _validate_matrix_shape_checkpoint_field(metadata, "logical_shape", param_idx)
    logical_shape = metadata.get("logical_shape")
    if len(logical_shape) != 2:
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata logical_shape must "
            f"be 2D for optimizer state index {param_idx}: logical_shape={logical_shape}."
        )
    for axis_name in ("tp_shard_axis", "dp_shard_axis"):
        axis = metadata.get(axis_name)
        if axis is not None and axis not in (0, 1):
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata field "
                f"{axis_name!r} for optimizer state index {param_idx} must be None, 0, or 1."
            )
    tp_shard_axis = metadata.get("tp_shard_axis")
    dp_shard_axis = metadata.get("dp_shard_axis")
    if tp_shard_axis is not None and dp_shard_axis is not None and tp_shard_axis != dp_shard_axis:
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata has conflicting TP/DP "
            f"matrix shard axes for optimizer state index {param_idx}: "
            f"tp_shard_axis={tp_shard_axis}, dp_shard_axis={dp_shard_axis}."
        )
    dp_shard_layout = metadata.get("dp_shard_layout")
    if dp_shard_axis is None:
        if dp_shard_layout is not None:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata has a DP shard "
                f"layout without a DP shard axis for optimizer state index {param_idx}."
            )
    elif dp_shard_axis == 0:
        if dp_shard_layout != MATRIX_OPTIMIZER_DP_SHARD_LAYOUT_ROW_CONTIGUOUS:
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata has unsupported "
                f"DP shard layout {dp_shard_layout!r} for optimizer state index {param_idx}."
            )
    else:
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata has unsupported DP "
            f"shard axis {dp_shard_axis!r} for optimizer state index {param_idx}."
        )
    if metadata.get("small_gram_side") not in ("right", "left"):
        raise RuntimeError(
            "[MegatronFSDP] Matrix optimizer checkpoint metadata field "
            f"'small_gram_side' for optimizer state index {param_idx} must be 'right' or 'left'."
        )


def _matrix_dp_shard_layout_to_checkpoint_value(spec) -> Optional[str]:
    if spec.dp_shard_axis is None:
        return None
    if spec.dp_shard_axis == 0:
        return MATRIX_OPTIMIZER_DP_SHARD_LAYOUT_ROW_CONTIGUOUS
    if spec.dp_shard_axis == 1:
        return MATRIX_OPTIMIZER_DP_SHARD_LAYOUT_COLUMN_CONTIGUOUS
    raise RuntimeError(
        "[MegatronFSDP] Matrix optimizer checkpoint metadata cannot represent "
        f"unsupported DP matrix shard axis {spec.dp_shard_axis}."
    )


def _optimizer_param_state_indices(optimizer: torch.optim.Optimizer) -> dict:
    """Return the param-id to state-index map used by Optimizer.state_dict()."""

    param_state_idx = {}
    idx = 0
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if id(param) not in param_state_idx:
                param_state_idx[id(param)] = idx
                idx += 1
    return param_state_idx


def _matrix_state_tensor_matches_param_shape(state_value: torch.Tensor, param: DTensor) -> bool:
    state_shape = tuple(state_value.shape)
    param_global_shape = tuple(param.shape)
    param_local_shape = (
        tuple(param._local_tensor.shape) if hasattr(param, "_local_tensor") else None
    )
    return state_shape == param_global_shape or state_shape == param_local_shape


def _is_same_shard_matrix_state_tensor(state_value: object, param: DTensor) -> bool:
    return (
        isinstance(state_value, torch.Tensor)
        and state_value.numel() > 1
        and isinstance(param, DTensor)
        and _matrix_state_tensor_matches_param_shape(state_value, param)
    )


def _raise_if_uncontracted_matrix_state_sidecar(
    state_name: str, state_value: object, param: DTensor
) -> None:
    if (
        isinstance(state_value, torch.Tensor)
        and state_value.numel() > 1
        and isinstance(param, DTensor)
        and not _matrix_state_tensor_matches_param_shape(state_value, param)
    ):
        _raise_unsupported_matrix_sidecar_state(state_name, state_value)


def _matrix_state_names_sharded_like_param(state: dict, param: DTensor) -> list[str]:
    return sorted(
        state_name
        for state_name, state_value in state.items()
        if _is_same_shard_matrix_state_tensor(state_value, param)
    )


def _matrix_same_shard_state_metadata(state: dict, param: DTensor) -> tuple[list[str], dict]:
    state_names = []
    for state_name, state_value in state.items():
        _raise_if_uncontracted_matrix_state_sidecar(state_name, state_value, param)
        if _is_same_shard_matrix_state_tensor(state_value, param):
            state_names.append(state_name)
    state_names = sorted(state_names)
    state_shapes = {state_name: list(state[state_name].shape) for state_name in state_names}
    return state_names, state_shapes


def _raise_unsupported_matrix_sidecar_state(state_name: str, state_value: torch.Tensor) -> None:
    raise RuntimeError(
        "[MegatronFSDP] Matrix optimizer tensor state that is not shaped like its "
        f"parameter is not supported yet. State {state_name!r} has shape "
        f"{tuple(state_value.shape)}. Add an explicit sidecar sharding/checkpoint "
        "contract before using this state with matrix-sharded FSDP."
    )


def _dtensor_layout_matches_param(state_value: DTensor, param: DTensor) -> bool:
    if state_value.placements != param.placements:
        return False
    state_mesh = state_value.device_mesh
    param_mesh = param.device_mesh
    state_dim_names = tuple(getattr(state_mesh, "mesh_dim_names", ()) or ())
    param_dim_names = tuple(getattr(param_mesh, "mesh_dim_names", ()) or ())
    if state_dim_names != param_dim_names:
        return False
    state_mesh_tensor = getattr(state_mesh, "mesh", None)
    param_mesh_tensor = getattr(param_mesh, "mesh", None)
    if state_mesh_tensor is None or param_mesh_tensor is None:
        return state_mesh == param_mesh
    return torch.equal(state_mesh_tensor, param_mesh_tensor)


def _validate_matrix_optimizer_state_sharding(
    optimizer: torch.optim.Optimizer, mfsdp_model: MegatronFSDP
) -> None:
    """Validate same-shaped matrix optimizer state follows parameter sharding.

    Muon momentum and mixed-precision master parameters are matrix-shaped state
    tensors and must be sharded exactly like the matrix parameter they belong
    to. Megatron-FSDP initializes optimizer state via a dummy optimizer step;
    checking immediately after that step prevents plain local tensors from
    being checkpointed as if they were valid matrix-axis shards.
    """

    if not HAVE_MCORE_MATRIX_UPDATE:
        return
    if (
        mfsdp_model.param_and_grad_buffer.bucketing_policy.data_parallel_sharding_strategy
        == "no_shard"
    ):
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if not _is_matrix_optimizer_owned_param(param):
                continue
            matrix_shard_spec = get_matrix_shard_spec(param)
            if matrix_shard_spec is None:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer-owned optimizer parameter is missing "
                    "MatrixShardSpec metadata after FSDP parameter replacement."
                )
            if not isinstance(param, DTensor):
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer-owned parameters require DTensor optimizer "
                    "parameters so momentum/master state is sharded by matrix axis."
                )
            for state_name, state_value in optimizer.state.get(param, {}).items():
                if not isinstance(state_value, torch.Tensor) or state_value.numel() <= 1:
                    continue
                _raise_if_uncontracted_matrix_state_sidecar(state_name, state_value, param)
                if not isinstance(state_value, DTensor):
                    raise RuntimeError(
                        "[MegatronFSDP] Matrix optimizer state must be DTensor-sharded with "
                        f"the same matrix spec as its parameter. State {state_name!r} for "
                        f"shape {tuple(state_value.shape)} is a local tensor."
                    )
                if not _dtensor_layout_matches_param(state_value, param):
                    raise RuntimeError(
                        "[MegatronFSDP] Matrix optimizer state DTensor placement does not "
                        f"match its parameter for state {state_name!r}: "
                        f"state placements={state_value.placements}, "
                        f"param placements={param.placements}."
                    )
                set_matrix_shard_spec(state_value, matrix_shard_spec)


def _matrix_optimizer_checkpoint_metadata(
    optimizer: torch.optim.Optimizer, mfsdp_model: MegatronFSDP
) -> dict:
    if not HAVE_MCORE_MATRIX_UPDATE:
        return {}
    if (
        mfsdp_model.param_and_grad_buffer.bucketing_policy.data_parallel_sharding_strategy
        == "no_shard"
    ):
        return {}

    param_state_indices = _optimizer_param_state_indices(optimizer)
    metadata = {}
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if not _is_matrix_optimizer_owned_param(param):
                continue
            matrix_shard_spec = get_matrix_shard_spec(param)
            matrix_optimizer_info = get_matrix_optimizer_info(param)
            if matrix_shard_spec is None or matrix_optimizer_info is None:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer-owned parameter is missing matrix "
                    "metadata while writing optimizer checkpoint metadata."
                )
            state_names, state_shapes = _matrix_same_shard_state_metadata(
                optimizer.state.get(param, {}), param
            )
            if not state_names:
                continue
            metadata[str(param_state_indices[id(param)])] = {
                "owner": matrix_optimizer_info.owner,
                "update_family": matrix_optimizer_info.update_family,
                "matrix_shard_contract": _matrix_shard_global_contract_to_checkpoint_dict(
                    matrix_shard_spec
                ),
                "matrix_shard_spec": _matrix_shard_spec_to_checkpoint_dict(matrix_shard_spec),
                "same_shard_state_layout": MATRIX_OPTIMIZER_SAME_SHARD_STATE_LAYOUT,
                "same_shard_state_names": sorted(state_names),
                "same_shard_state_shapes": state_shapes,
            }
    return metadata


def _add_matrix_optimizer_checkpoint_metadata(
    optimizer: torch.optim.Optimizer, mfsdp_model: MegatronFSDP, state_dict: dict
) -> None:
    _validate_matrix_optimizer_state_sharding(optimizer, mfsdp_model)
    metadata = _matrix_optimizer_checkpoint_metadata(optimizer, mfsdp_model)
    if metadata:
        state_dict[MATRIX_OPTIMIZER_STATE_METADATA_KEY] = {
            "version": MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
            "params": metadata,
        }
    else:
        state_dict.pop(MATRIX_OPTIMIZER_STATE_METADATA_KEY, None)


def _validate_matrix_optimizer_checkpoint_metadata(
    optimizer: torch.optim.Optimizer, mfsdp_model: MegatronFSDP, state_dict: dict
) -> None:
    if not HAVE_MCORE_MATRIX_UPDATE:
        return
    if (
        mfsdp_model.param_and_grad_buffer.bucketing_policy.data_parallel_sharding_strategy
        == "no_shard"
    ):
        return

    metadata_block = state_dict.get(MATRIX_OPTIMIZER_STATE_METADATA_KEY)
    metadata = None
    if metadata_block is not None:
        if not isinstance(metadata_block, dict):
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata must be a dict."
            )
        if metadata_block.get("version") != MATRIX_OPTIMIZER_STATE_METADATA_VERSION:
            raise RuntimeError(
                "[MegatronFSDP] Unsupported matrix optimizer checkpoint metadata version: "
                f"{metadata_block.get('version')!r}."
            )
        metadata = metadata_block.get("params")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "[MegatronFSDP] Matrix optimizer checkpoint metadata is missing params."
            )
    param_state_indices = _optimizer_param_state_indices(optimizer)
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if not _is_matrix_optimizer_owned_param(param):
                continue
            matrix_shard_spec = get_matrix_shard_spec(param)
            matrix_optimizer_info = get_matrix_optimizer_info(param)
            if matrix_shard_spec is None:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer-owned parameter is missing "
                    "MatrixShardSpec metadata while loading optimizer state."
                )
            if matrix_optimizer_info is None:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer-owned parameter is missing "
                    "MatrixOptimizerInfo metadata while loading optimizer state."
                )
            param_idx = str(param_state_indices[id(param)])
            loaded_state_by_idx = state_dict.get("state", {})
            loaded_state = loaded_state_by_idx.get(
                int(param_idx), loaded_state_by_idx.get(param_idx, {})
            )
            if not isinstance(loaded_state, dict):
                continue
            for state_name, state_value in loaded_state.items():
                _raise_if_uncontracted_matrix_state_sidecar(state_name, state_value, param)
                if (
                    _is_same_shard_matrix_state_tensor(state_value, param)
                    and not isinstance(state_value, DTensor)
                ):
                    raise RuntimeError(
                        "[MegatronFSDP] Matrix optimizer checkpoint contains local tensor "
                        f"state {state_name!r} for optimizer state index {param_idx}; "
                        "matrix-shaped Muon state must be checkpointed as a DTensor."
                    )
            same_shape_state_names = _matrix_state_names_sharded_like_param(loaded_state, param)
            if not same_shape_state_names:
                continue
            if metadata is None or param_idx not in metadata:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint is missing matrix-shard "
                    f"metadata for optimizer state index {param_idx}."
                )
            param_metadata = metadata[param_idx]
            if not isinstance(param_metadata, dict):
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint metadata for optimizer "
                    f"state index {param_idx} must be a dict."
                )
            matrix_shard_contract = param_metadata.get("matrix_shard_contract")
            if not isinstance(matrix_shard_contract, dict):
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint metadata is missing "
                    f"matrix_shard_contract for optimizer state index {param_idx}."
                )
            _validate_matrix_shard_contract_checkpoint_metadata(
                matrix_shard_contract, param_idx
            )
            if not _matrix_shard_spec_matches_checkpoint_dict(
                matrix_shard_spec, matrix_shard_contract
            ):
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint shard metadata does not "
                    f"match the current MatrixShardSpec for optimizer state index {param_idx}."
                )
            matrix_shard_spec_metadata = param_metadata.get("matrix_shard_spec")
            if matrix_shard_spec_metadata is not None:
                _validate_matrix_shard_spec_checkpoint_metadata(
                    matrix_shard_spec_metadata, param_idx, "matrix_shard_spec"
                )
            if param_metadata.get("owner") != matrix_optimizer_info.owner:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint owner does not match "
                    f"the current MatrixOptimizerInfo for optimizer state index {param_idx}: "
                    f"checkpoint={param_metadata.get('owner')!r}, "
                    f"current={matrix_optimizer_info.owner!r}."
                )
            if param_metadata.get("update_family") != matrix_optimizer_info.update_family:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint update family does not match "
                    f"the current MatrixOptimizerInfo for optimizer state index {param_idx}: "
                    f"checkpoint={param_metadata.get('update_family')!r}, "
                    f"current={matrix_optimizer_info.update_family!r}."
                )
            if (
                param_metadata.get("same_shard_state_layout")
                != MATRIX_OPTIMIZER_SAME_SHARD_STATE_LAYOUT
            ):
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint state layout contract does "
                    f"not match for optimizer state index {param_idx}: "
                    f"checkpoint={param_metadata.get('same_shard_state_layout')!r}, "
                    f"current={MATRIX_OPTIMIZER_SAME_SHARD_STATE_LAYOUT!r}."
                )
            metadata_state_names = param_metadata.get("same_shard_state_names")
            if not isinstance(metadata_state_names, list):
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint metadata is missing "
                    f"same_shard_state_names for optimizer state index {param_idx}."
                )
            metadata_state_names = sorted(metadata_state_names)
            if same_shape_state_names != metadata_state_names:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint state names do not match "
                    f"the matrix-sharded metadata for optimizer state index {param_idx}: "
                    f"checkpoint={metadata_state_names}, loaded={same_shape_state_names}."
                )
            metadata_state_shapes = param_metadata.get("same_shard_state_shapes")
            if not isinstance(metadata_state_shapes, dict):
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint metadata is missing "
                    f"same_shard_state_shapes for optimizer state index {param_idx}."
                )
            if sorted(metadata_state_shapes) != metadata_state_names:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint state shape keys do not "
                    f"match same_shard_state_names for optimizer state index {param_idx}: "
                    f"names={metadata_state_names}, shape_keys={sorted(metadata_state_shapes)}."
                )
            for state_name, state_shape in metadata_state_shapes.items():
                if not isinstance(state_shape, list) or not all(
                    isinstance(dim, int) for dim in state_shape
                ):
                    raise RuntimeError(
                        "[MegatronFSDP] Matrix optimizer checkpoint state shape metadata "
                        f"for state {state_name!r} at optimizer state index {param_idx} "
                        "must be a list of integers."
                    )
            loaded_state_shapes = {
                state_name: list(loaded_state[state_name].shape)
                for state_name in same_shape_state_names
            }
            if loaded_state_shapes != metadata_state_shapes:
                raise RuntimeError(
                    "[MegatronFSDP] Matrix optimizer checkpoint state shapes do not match "
                    f"the matrix-sharded metadata for optimizer state index {param_idx}: "
                    f"checkpoint={metadata_state_shapes}, loaded={loaded_state_shapes}."
                )


class ShardingStrategy(IntEnum):
    """
    IntEnum to track the abbreviated sharding strategy for Megatron-FSDP.

    - `0` or `no_shard` implies that your model is not sharded. Similar memory usage to `DDP`.
    - `1` or `optim` implies that your optimizer state is sharded. Similar to optimizer
        state sharding in `ZeRO-DP`.
    - `2` or `optim_grads` implies that your optimizer state and gradients are sharded.
        Similar to optimizer state and gradient sharding in `ZeRO-2`.
    - `3` or `optim_grads_params` implies that your optimizer state, gradients, and
        training parameters are sharded. Similar to optimizer state, gradient, and
        training parameter sharding in `ZeRO-3`.
    """

    NO_SHARD = 0
    OPTIM = 1
    OPTIM_GRADS = 2
    OPTIM_GRADS_PARAMS = 3


def experimental_api(func: Callable) -> Callable:
    """
    Mark a function or class as experimental API in Megatron CI/CD.

    TODO(@cspades): Copied from megatron.core.utils to avoid depending on MCore
    for Megatron-FSDP. Should remove when the API is no longer experimental.
    """
    func._experimental_api = True
    return func


@experimental_api
def fully_shard_model(
    module: torch.nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    dp_shard_dim: Optional[str] = None,
    dp_outer_dim: Optional[str] = None,
    tp_dim: Optional[str] = None,
    hybrid_fsdp_group: Optional[torch.distributed.ProcessGroup] = None,
    hybrid_fsdp_expt_group: Optional[torch.distributed.ProcessGroup] = None,
    expt_device_mesh: Optional[DeviceMesh] = None,
    fsdp_group_ag: Optional[torch.distributed.ProcessGroup] = None,
    expt_fsdp_group_ag: Optional[torch.distributed.ProcessGroup] = None,
    fsdp_unit_modules: Optional[Sequence[Type[torch.nn.Module]] | Sequence[str]] = None,
    zero_dp_strategy: str | int = 3,
    outer_dp_sharding_strategy: str | int = 0,
    device: Optional[torch.device] = None,
    init_model_with_meta_device: bool = False,
    mixed_precision_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    sync_model_each_microbatch: bool = True,
    preproc_state_dict_for_dcp_ckpt: bool = True,
    report_nan_in_param_grad: bool = False,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    fsdp_db_use_persist_buf_on_alloc_fail: bool = False,
    disable_symmetric_registration: bool = False,
    enable_fine_grained_param_gather: bool = False,
    use_decoupled_grad: bool = False,
) -> torch.nn.Module:
    """
    Fully-shard the model for Megatron-FSDP. This wraps the model in a MegatronFSDP
    class that schedules the sharding lifecycle of the model parameters and gradients
    during training and inference.

    The original `torch.nn.Module` can be accessed at `MegatronFSDP.module`.

    Args:
        module (torch.nn.Module):
            The PyTorch module fully-sharded and managed by Megatron-FSDP.

        device_mesh (Optional[DeviceMesh]):
            Device mesh object defining the topology for distributed training. If not provided,
            Megatron-FSDP will build a default FSDP DeviceMesh.

        dp_shard_dim (Optional[str]):
            Name of the data parallel sharding sub-mesh in the device_mesh. Supports
            a flattened DP-CP sub-mesh, in which case parameters, gradients, and
            optimizer state will be sharded across both DP and CP ranks.

        dp_outer_dim (Optional[str]):
            Name of the "outer" DP sub-mesh in the device_mesh for hybrid-sharding (HSDP),
            which supports "DP-Replicate" as well as optimizer state sharding (HFSDP).
            Defaults to None. Required for HSDP, which is enabled by this argument.

        tp_dim (Optional[str]):
            Name of the tensor parallel sub-mesh in the device_mesh, which is necessary
            for strided sharding between TP and FSDP (and fully-sharded HSDP) dimensions.
            Defaults to None. Required if TP is used in the model, or if TransformerEngine
            layers are utilized, as TE defaults to "TP=1".

        hybrid_fsdp_group (Optional[torch.distributed.ProcessGroup]):
            Cumulative data parallel process group for hybrid FSDP that can be manufactured
            by flattening the outer-FSDP (dp_outer_dim) and FSDP (dp_shard_dim) process groups
            or sub-meshes. Defaults to None. Required for HSDP, i.e. if dp_outer_dim is not None.

        expt_device_mesh (Optional[DeviceMesh]):
            Expert parallel device mesh object defining the topology for MoE distributed training.
            Utilizes the mesh dimension names specified by the *_dim arguments.

        fsdp_group_ag (Optional[torch.distributed.ProcessGroup]):
            Independent all-gather process group for overlapping all-gather and reduce-scatter
            operations. When provided, enables AG/RS overlap optimization for regular (non-expert)
            parameters. Users should create this group with the same ranks as the dp-cp group.
            Defaults to None.

        expt_fsdp_group_ag (Optional[torch.distributed.ProcessGroup]):
            Independent all-gather process group for expert parameters in MoE models. When provided,
            enables AG/RS overlap optimization for expert parameters. Users should create this group
            with the same ranks as the expert data parallel group. Defaults to None.

        fsdp_unit_modules (Optional[Sequence[Type[torch.nn.Module]] | Sequence[str]]):
            List of (sub-)module classes or (sub-)module class import paths that are "units",
            which are torch.nn.Module(s) that are sharded and scheduled by Megatron-FSDP.
            In particular, FSDP unit module parameters can be "safely" deallocated after
            the forward() or backward() pass without interfering with other computational
            operations that rely on those parameters in the complete PyTorch model.
            This information is utilized by Megatron-FSDP to optimally shard, gather, and
            overlap communications during the forward and backward pass of the module.
            Defaults to None, which is peak-memory-equivalent to DDP / "no_shard".

        zero_dp_strategy (str | int):
            Zero-redundancy sharding strategy for sharding data parallel parameters and gradients.
            - "no_shard" / 0: No optimizer, gradient, or parameter sharding. Similar
                memory usage to DDP.
            - "optim" / 1: Shards optimizer states (and main weights for mixed precision training),
                which is conceptually similar to optimizer state sharding in `ZeRO-DP`.
            - "optim_grads" / 2: Shards gradients and optimizer states, which is conceptually
                similar to "ZeRO-2".
            - "optim_grads_params" / 3: Shards parameters, gradients and optimizer states, which
                is conceptually similar to "ZeRO-3".
            Defaults to "optim_grads_params" / 3.

        outer_dp_sharding_strategy (str | int):
            Sharding strategy for outer data parallel group in Hybrid Sharded Data Parallel (HSDP).
            Shares the same semantics as zero_dp_strategy, but only 'no_shard' / 0 (DP Replication)
            and 'optim' / 1 (Optimizer State Hybrid Sharding) are supported, and 'optim' / 1 is only
            supported when zero_dp_strategy='optim_grads_params'.
            This option is only effective when HSDP is enabled, i.e. when dp_outer_dim is not None.
            Defaults to "no_shard" / 0, which replicates model parameters across the dp_outer group.

        device (Optional[torch.device]):
            Target device for the sharded model. Used to migrate all parameters in the model
            to an expected device. If init_model_with_meta_device=True, this argument is ignored.
            Defaults to None.

        init_model_with_meta_device (bool):
            Utilized to initialize large models that do not fit on a single device, and requires
            implementing a custom Module.reset_parameters() or Module._reset_parameters() method.
            Defaults to False.

        mixed_precision_policy (megatron_fsdp.MixedPrecisionPolicy):
            Megatron-FSDP mixed-precision config that controls compute and communication precision.
            Default values are defined in `megatron_fsdp.MixedPrecisionPolicy`.

        overlap_grad_reduce (bool):
            Whether to overlap gradient reduce-scatter (or all-reduce) with backward compute.
            Defaults to True.

        overlap_param_gather (bool):
            Whether to overlap parameter all-gather with forward and backward compute.
            Defaults to True.

        sync_model_each_microbatch (bool): Whether to sync parameters and install gradients on
            each training step. When disabled, Megatron-FSDP will overlap reduce-scatter with
            subsequent compute and delay HSDP gather and reduce operations per optimization cycle,
            which improves performance and throughput when using delayed optimization strategies
            such as gradient accumulation. Defaults to True, can be modified before the model
            forward / backward pass via MegatronFSDP.set_model_auto_sync(bool) or controlled
            with the (no_)sync context managers or microbatch_count and is_last_microbatch.

        preproc_state_dict_for_dcp_ckpt (bool):
            Whether to preprocess the unevenly-sharded state dict for DCP checkpointing,
            for both the model and the optimizer.
            Defaults to True.

        report_nan_in_param_grad (bool):
            Whether to precisely check for NaN values in gradients for every weight. Can
            significantly degrade performance. Defaults to False.

        average_in_collective (bool):
            Whether to average gradients in collective communication. Defaults to False.

        disable_bucketing (bool):
            Whether to disable gradient bucketing optimization, which permits more granular
            and precise communication of parameters and gradients. Defaults to False.

        calculate_per_token_loss (bool):
            Whether to calculate loss per token, which deactivates gradient scaling.
            Defaults to False.

        keep_fp8_transpose_cache (bool):
            Whether to keep the FP8 transpose cache when using a Megatron FSDP.
            Defaults to False.

        nccl_ub (bool):
            Whether to use NCCL UCC for communication. Defaults to False.

        fsdp_double_buffer (bool):
            Whether to use double buffer for FSDP. Defaults to False.

        fsdp_db_use_persist_buf_on_alloc_fail (bool):
            Whether to fall back to persistent buffer allocator when a bucket does not
            fit FSDP double buffer size.

        disable_symmetric_registration (bool):
            Whether to disable symmetric (window) registration for NCCL UB registration.
            This option forces conventional (local) UB registration when nccl_ub is set.
            Defaults to False.

        enable_fine_grained_param_gather (bool):
            Whether to enable "fine-grained" param all-gather, which can improve performance
            when using MXFP8 parameters with activation recomputation. Specifically, it
            unshards parameters per-Module instead of unsharding all sub-modules of an FSDP
            unit module simultaneously. Defaults to False.

        use_decoupled_grad (bool):
            If true, reduced gradients are installed into `Parameter.decoupled_grad` instead
            of `Parameter.grad`. Defaults to False.

    Returns:
        model (MegatronFSDP): The wrapped Megatron-FSDP model configured for FSDP.
    """
    # If no DeviceMesh or FSDP dimension is provided, then build an FSDP DeviceMesh.
    # Modify arguments into arguments necessary for vanilla FSDP.
    if device_mesh is None:
        if dp_shard_dim is None:
            dp_shard_dim = "fsdp"
        if tp_dim is None:
            # Trivial TP dimension to seamlessly support TransformerEngine.
            tp_dim = "tp"
        # Deactivate DP-Outer, which needs to be consistent with Expert DeviceMesh.
        dp_outer_dim = None
        hybrid_fsdp_group = None
        outer_dp_sharding_strategy = ShardingStrategy.NO_SHARD
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(torch.distributed.get_world_size(), 1),
            mesh_dim_names=(dp_shard_dim, tp_dim),
        )

    # Parse zero_dp_strategy and outer_dp_sharding_strategy.
    # TODO(@cspades): Integrate this Enum into MegatronFSDP.
    if zero_dp_strategy == ShardingStrategy.NO_SHARD:
        zero_dp_strategy = "no_shard"
    elif zero_dp_strategy == ShardingStrategy.OPTIM:
        zero_dp_strategy = "optim"
    elif zero_dp_strategy == ShardingStrategy.OPTIM_GRADS:
        zero_dp_strategy = "optim_grads"
    elif zero_dp_strategy == ShardingStrategy.OPTIM_GRADS_PARAMS:
        zero_dp_strategy = "optim_grads_params"
    elif zero_dp_strategy in ["no_shard", "optim", "optim_grads", "optim_grads_params"]:
        # Valid string sharding strategy.
        pass
    else:
        # Invalid sharding strategy.
        raise ValueError(
            f"Invalid FSDP / Inner DP Sharding Strategy: {zero_dp_strategy}\n"
            f"Valid Sharding Strategies: {ShardingStrategy.NO_SHARD}, "
            f"{ShardingStrategy.OPTIM}, {ShardingStrategy.OPTIM_GRADS}, "
            f"{ShardingStrategy.OPTIM_GRADS_PARAMS}, "
            "no_shard, optim, optim_grads, optim_grads_params"
        )
    if outer_dp_sharding_strategy == ShardingStrategy.NO_SHARD:
        outer_dp_sharding_strategy = "no_shard"
    elif outer_dp_sharding_strategy == ShardingStrategy.OPTIM:
        outer_dp_sharding_strategy = "optim"
    elif outer_dp_sharding_strategy in ["no_shard", "optim"]:
        # Valid string sharding strategy.
        pass
    else:
        # Invalid sharding strategy.
        raise ValueError(
            f"Invalid Hybrid DP-Outer Sharding Strategy: {outer_dp_sharding_strategy}\n"
            f"Valid Sharding Strategies: {ShardingStrategy.NO_SHARD}, "
            f"{ShardingStrategy.OPTIM}, no_shard, optim"
        )

    # Validate more arguments.
    _outer_fsdp_sharding = outer_dp_sharding_strategy == "optim"
    if _outer_fsdp_sharding and zero_dp_strategy != "optim_grads_params":
        # If sharding on outer DP using HSDP, then we must use HSDP buffers and
        # we must be fully-sharding on inner DP. HSDP is an extension of FSDP.
        # TODO(@shjwudp, @cspades): Requires various modifications to support.
        raise ValueError(
            f"Sharding with Hybrid (Fully) Sharded Data Parallel (HSDP) requires "
            "zero_dp_strategy to use FSDP ('optim_grads_params', 3), because "
            "outer sharding is dependent on inner sharding."
        )
    if (dp_outer_dim is None) ^ (hybrid_fsdp_group is None):
        # XOR - HSDP requires both or neither of dp_outer_dim and hybrid_fsdp_group
        # to be specified, so if XOR then raise an error.
        raise ValueError(
            f"dp_outer_dim={dp_outer_dim} and hybrid_fsdp_group={hybrid_fsdp_group} must be "
            "specified together for Hybrid FSDP (HSDP), or both set to None (for FSDP)."
        )
    if init_model_with_meta_device and zero_dp_strategy == "no_shard":
        raise ValueError(
            "Meta device initialization (init_model_with_meta_device=True) is not "
            "supported or necessary for the 'no_shard' / 0 sharding strategy."
        )

    # DDP Config for Megatron FSDP.
    ddp_config = DistributedDataParallelConfig(
        data_parallel_sharding_strategy=zero_dp_strategy,
        outer_dp_sharding_strategy=outer_dp_sharding_strategy,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        average_in_collective=average_in_collective,
        keep_fp8_transpose_cache=keep_fp8_transpose_cache,  # pylint: disable=C0301
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer or nccl_ub,
        fsdp_db_use_persist_buf_on_alloc_fail=fsdp_db_use_persist_buf_on_alloc_fail,
        disable_symmetric_registration=disable_symmetric_registration,
        megatron_fsdp_use_decoupled_grad=use_decoupled_grad,
    )

    # Create FSDPDistributedIndex.
    dist_index = FSDPDistributedIndex(
        device_mesh=device_mesh,
        # Always required for Megatron-FSDP.
        dp_shard_dim=dp_shard_dim,
        # Only required for HSDP.
        dp_outer_dim=dp_outer_dim,
        # TODO(@cspades): TP sub-mesh should be optional if not using TP, but is
        # required for Megatron, TransformerEngine (default TP=1), and strided
        # sharding when using DTensor-based TP.
        tp_dim=tp_dim,
        # Only required for HSDP.
        hybrid_fsdp_group=hybrid_fsdp_group,
        # Only required for HSDP + EP.
        hybrid_fsdp_expt_group=hybrid_fsdp_expt_group,
        # Access to flattened DP rank assignments for HSDP.
        hsdp_outer_dp_shard=_outer_fsdp_sharding,
        # Only required for Megatron-FSDP + EP.
        expt_device_mesh=expt_device_mesh,
        # AG groups for AG/RS overlap optimization.
        fsdp_group_ag=fsdp_group_ag,
        expt_fsdp_group_ag=expt_fsdp_group_ag,
    )

    # Wrap model in Megatron FSDP.
    model = MegatronFSDP(
        module=module,
        dist_index=dist_index,
        ddp_config=ddp_config,
        mixed_precision_policy=mixed_precision_policy,
        fsdp_unit_modules=fsdp_unit_modules,
        disable_bucketing=disable_bucketing,
        device=device,
        calculate_per_token_loss=calculate_per_token_loss,
        init_model_with_meta_device=init_model_with_meta_device,
        sync_model_each_microbatch=sync_model_each_microbatch,
        enable_fine_grained_param_gather_hook=enable_fine_grained_param_gather,
        report_nan_in_param_grad=report_nan_in_param_grad,
    )

    # Register a state dict post-hook to add Torch DCP metadata for writing checkpoints.
    if preproc_state_dict_for_dcp_ckpt and zero_dp_strategy != "no_shard":

        def remove_te_extra_state(state_dict):
            # Megatron-FSDP does not support FP8 extra state checkpointing in TE.
            extra_state_keys = [k for k in state_dict.keys() if k.endswith("_extra_state")]
            for key in extra_state_keys:
                state_dict.pop(key)

        def preprocess_dcp_and_te_extra_state(state_dict):
            # Preprocess the state dict for uneven DTensor checkpointing.
            remove_te_extra_state(state_dict)
            return preprocess_state_dict_for_uneven_dtensor(state_dict)

        model._register_state_dict_hook(
            lambda module, state_dict, prefix, local_metadata: preprocess_dcp_and_te_extra_state(
                state_dict
            )
        )

    # Return the wrapped Megatron-FSDP model.
    return model


@experimental_api
def fully_shard_optimizer(
    optimizer: torch.optim.Optimizer, preproc_state_dict_for_dcp_ckpt: bool = True
) -> torch.optim.Optimizer:
    """
    Fully shard the optimizer for Megatron-FSDP. This is an in-place operation on the optimizer
    instance, which modifies the optimizer to call methods exposed by the MegatronFSDP model API.

    The optimizer should be registered on the MegatronFSDP distributed model parameters:
    ```
        # Fully-shard the model.
        mfsdp_model = fully_shard_model(model, ...)

        # Register the fully-sharded parameters with the optimizer.
        # Use MegatronFSDP._replace_param_with_distributed_if_needed()
        # to swap to the distributed optimizer state parameters.
        optimizer = fully_shard_optimizer(Adam(params=mfsdp_model.parameters()))
    ```

    Args:
        optimizer (torch.optim.Optimizer):
            (Distributed) optimizer for training the model, which is extended to automatically
            execute necessary Megatron-FSDP operations during the training loop.

        preproc_state_dict_for_dcp_ckpt (bool):
            Whether to preprocess the state dict for DCP checkpointing. Defaults to True.

    Returns:
        optimizer (torch.optim.Optimizer): The in-place modified optimizer for Megatron-FSDP.
    """
    # Extract a reference to MegatronFSDP from the first registered Parameter.
    if not optimizer.param_groups:
        raise ValueError(
            f"[MegatronFSDP fully_shard_optimizer()] Provided optimizer doesn't "
            f"have any registered parameters: {optimizer}"
        )
    first_mfsdp_param = optimizer.param_groups[0][next(iter(optimizer.param_groups[0]))][0]
    if not getattr(first_mfsdp_param, "_megatron_fsdp_model", None):
        raise ValueError(
            f"[MegatronFSDP fully_shard_optimizer()] Could not retrieve a reference to "
            f"MegatronFSDP from the first registered Parameter: {first_mfsdp_param} \n"
            "Make sure the optimizer is registered to the MegatronFSDP distributed "
            "parameters via MegatronFSDP._replace_param_with_distributed_if_needed() "
            "before initializing the optimizer on the MegatronFSDP model. "
        )
    mfsdp_model = first_mfsdp_param._megatron_fsdp_model

    # Save a reference to the optimizer.step() and optimizer.zero_grad() methods.
    optimizer_step_base_func = type(optimizer).step
    optimizer_zero_grad_base_func = type(optimizer).zero_grad

    # Pre-initialize the optimizer state for checkpoint loading via DCP.
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.numel() == 0 or (
                hasattr(param, "_local_tensor") and param._local_tensor.numel() == 0
            ):
                # Avoid FusedAdam errors on empty tensor input.
                continue
            # Optimizer state is built from wgrad.
            param.grad = torch.zeros_like(param)
    # Non-lazy optimizer state initialization.
    optimizer.step()
    _validate_matrix_optimizer_state_sharding(optimizer, mfsdp_model)
    optimizer.zero_grad()

    def add_matrix_optimizer_checkpoint_metadata(optimizer, state_dict):
        _add_matrix_optimizer_checkpoint_metadata(optimizer, mfsdp_model, state_dict)

    def validate_matrix_optimizer_load_state_dict(optimizer, state_dict):
        _validate_matrix_optimizer_checkpoint_metadata(optimizer, mfsdp_model, state_dict)
        return state_dict

    def restore_matrix_optimizer_loaded_state_metadata(optimizer):
        _validate_matrix_optimizer_state_sharding(optimizer, mfsdp_model)

    optimizer.register_state_dict_post_hook(add_matrix_optimizer_checkpoint_metadata)
    optimizer.register_load_state_dict_pre_hook(validate_matrix_optimizer_load_state_dict)
    optimizer.register_load_state_dict_post_hook(restore_matrix_optimizer_loaded_state_metadata)

    # Define a new optimizer.step() method that distributes optimizer state and gradients,
    # waits for asynchronous gradient reduce-scatter work to be completed, and updates
    # model weights. These options can be turned off via arguments in optimizer.step().
    def megatron_fsdp_optimizer_step(optimizer, *args, **kwargs):
        # Extract extended kwargs.
        sync_grad_before_optimizer_step = kwargs.pop("sync_grad_before_optimizer_step", True)
        install_optimized_model_weights = kwargs.pop("install_optimized_model_weights", True)

        # Synchronize reduce-scatter and all-gather operations for all model gradients
        # and parameters, attach gradients to the optimizer state, and replace the raw
        # module parameters with Megatron-FSDP-managed optimizer parameters & states in
        # preparation for (distributed) optimization.
        # NOTE: Only necessary if MegatronFSDP.model_auto_sync = False, in which case
        # gradient synchronization is not automatically handled by MegatronFSDP during
        # the post-backward hook and we need to synchronize manually.
        if sync_grad_before_optimizer_step and not mfsdp_model.model_auto_sync:
            mfsdp_model.finish_grad_sync()

        # Execute the base optimizer.step() on the model optimizer named parameters.
        optimizer_step_base_func(optimizer, *args, **kwargs)

        # Update the raw module training parameters with optimized values.
        if install_optimized_model_weights:
            mfsdp_model.install_optimized_model_weights()

    # Define a new optimizer.zero_grad() method that zeros the gradient in both
    # the optimizer as well as the Megatron-FSDP gradient buffer. These options
    # can be turned off via arguments in optimizer.zero_grad().
    def megatron_fsdp_optimizer_zero_grad(optimizer, *args, **kwargs):
        # Extract extended kwargs.
        zero_grad_buffer = kwargs.pop("zero_grad_buffer", True)

        # Execute the base optimizer.zero_grad() on the model optimizer named parameters.
        optimizer_zero_grad_base_func(optimizer, *args, **kwargs)

        # Zero out the gradient in the Megatron-FSDP gradient buffer.
        if zero_grad_buffer:
            mfsdp_model.zero_grad_buffer()

    # Override the optimizer.step() and optimizer.zero_grad() methods to support
    # Megatron-FSDP operations.
    megatron_fsdp_optimizer_step.__signature__ = create_updated_function_signature(
        optimizer_step_base_func,
        sync_grad_before_optimizer_step=True,
        install_optimized_model_weights=True,
    )
    optimizer.step = types.MethodType(megatron_fsdp_optimizer_step, optimizer)
    megatron_fsdp_optimizer_zero_grad.__signature__ = create_updated_function_signature(
        optimizer_zero_grad_base_func, zero_grad_buffer=True
    )
    optimizer.zero_grad = types.MethodType(megatron_fsdp_optimizer_zero_grad, optimizer)

    if preproc_state_dict_for_dcp_ckpt:

        def dict_nested_shallow_copy(d: dict):
            """Create a nested shallow copy of a dict. Same values, different pointers."""
            if not isinstance(d, dict):
                return d
            return {
                k: dict_nested_shallow_copy(v) if isinstance(v, dict) else v for k, v in d.items()
            }

        def preprocess_optimizer_state_dict_for_uneven_dtensor(optimizer, state_dict):
            """
            Hook that mocks the global optimizer state for unevenly-distributed
            DTensors, as the optimizer state is only initialized for non-empty
            parameters, and preprocesses the optimizer `state_dict` DTensors
            in-place for Torch DCP.
            """
            # Retrieve a template optimizer state.
            optim_state_template = next(iter(optimizer.state.values())) if optimizer.state else {}
            # All-gather the optimizer state keys as this rank could have empty state.
            optim_state_dtensor_keys = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(
                optim_state_dtensor_keys,
                [
                    # Only track keys associated with DTensors for DCP.
                    key
                    for key, val in optim_state_template.items()
                    if isinstance(val, DTensor)
                ],
            )
            optim_state_dtensor_keys = list(
                set([key for state in optim_state_dtensor_keys for key in state])
            )

            # NOTE(@cspades): Re-construct the Megatron-FSDP distributed parameter
            # to index mapping as implemented in torch.optim.Optimizer.state_dict():
            # https://github.com/pytorch/pytorch/blob/main/torch/optim/optimizer.py
            # Simply put, the index maps to the very first appearance of id(param)
            # looping through all parameters in all groups with memory address
            # equivalent to the distributed parameter managed by Megatron-FSDP.
            param_state_idx = {}
            idx = 0
            # For all empty parameters, mock empty DTensors for all empty parameters
            # of Megatron-FSDP's unevenly-distributed optimizer state into a shallow
            # copy of the state dictionary to synchronize and pre-process a global
            # variant of the optimizer state in preparation for Torch DCP. This allows
            # us to sync the non-empty DTensor shard metadata across sharding groups
            # while excluding empty DTensor shards from the optimizer checkpoint.
            optim_state_extended = dict_nested_shallow_copy(state_dict)
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    # Update the parameter state index.
                    # For shared params, use same index.
                    if id(param) not in param_state_idx:
                        # New parameter, assign an index.
                        param_state_idx[id(param)] = idx
                        idx += 1
                    if param in optimizer.state or not isinstance(param, DTensor):
                        # Only mock optimizer state for parameters that are missing state.
                        # No need to mock for non-DTensor params. Not relevant to DCP.
                        continue
                    for key in optim_state_dtensor_keys:
                        # Construct a mock DTensor state for the empty DTensor parameter.
                        param_idx = param_state_idx[id(param)]
                        optim_state_extended["state"].setdefault(param_idx, {})[key] = (
                            DTensor.from_local(
                                local_tensor=torch.empty(0, dtype=param.dtype, device=param.device),
                                device_mesh=param.device_mesh,
                                placements=param.placements,
                                shape=param.shape,
                                stride=param.stride(),
                            )
                        )

            # Synchronize and preprocess DTensor metadata for Torch DCP.
            preprocess_state_dict_for_uneven_dtensor(optim_state_extended)

        # Attach the optimizer state_dict() post-hook to prepare DTensors for Torch DCP.
        # args = (optimizer, state_dict)
        optimizer.register_state_dict_post_hook(
            lambda *args, **kwargs: preprocess_optimizer_state_dict_for_uneven_dtensor(
                args[0], args[1]
            )
        )

    # Return the in-place modified optimizer.
    return optimizer


@experimental_api
def fully_shard(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device_mesh: Optional[DeviceMesh] = None,
    dp_shard_dim: Optional[str] = None,
    dp_outer_dim: Optional[str] = None,
    tp_dim: Optional[str] = None,
    hybrid_fsdp_group: Optional[torch.distributed.ProcessGroup] = None,
    hybrid_fsdp_expt_group: Optional[torch.distributed.ProcessGroup] = None,
    expt_device_mesh: Optional[DeviceMesh] = None,
    fsdp_group_ag: Optional[torch.distributed.ProcessGroup] = None,
    expt_fsdp_group_ag: Optional[torch.distributed.ProcessGroup] = None,
    fsdp_unit_modules: Optional[Sequence[Type[torch.nn.Module]] | Sequence[str]] = None,
    zero_dp_strategy: str | int = 3,
    outer_dp_sharding_strategy: str | int = 0,
    device: Optional[torch.device] = None,
    init_model_with_meta_device: bool = False,
    mixed_precision_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    sync_model_each_microbatch: bool = True,
    preproc_state_dict_for_dcp_ckpt: bool = True,
    report_nan_in_param_grad: bool = False,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    fsdp_db_use_persist_buf_on_alloc_fail: bool = False,
    disable_symmetric_registration: bool = False,
    enable_fine_grained_param_gather: bool = False,
    use_decoupled_grad: bool = False,
) -> tuple[MegatronFSDP, torch.optim.Optimizer]:
    """
    Fully shard the model and the optimizer for Megatron-FSDP.

    Wraps the model as an Megatron-FSDP module, and modifies the optimizer to
    be compatible with the Megatron-FSDP training strategy.

    Args:
        Union of arguments from fully_shard_model and fully_shard_optimizer.

    Returns:
        torch.nn.Module: The wrapped Megatron-FSDP model configured for distributed training.
        torch.optim.Optimizer: The Megatron-FSDP-compliant optimizer for training the model.

    Note:
        This implementation uses NVIDIA's FSDP which includes optimizations specific
        to NVIDIA hardware and software stack.
    """

    model = fully_shard_model(
        module=module,
        device_mesh=device_mesh,
        dp_shard_dim=dp_shard_dim,
        dp_outer_dim=dp_outer_dim,
        tp_dim=tp_dim,
        hybrid_fsdp_group=hybrid_fsdp_group,
        hybrid_fsdp_expt_group=hybrid_fsdp_expt_group,
        expt_device_mesh=expt_device_mesh,
        fsdp_group_ag=fsdp_group_ag,
        expt_fsdp_group_ag=expt_fsdp_group_ag,
        fsdp_unit_modules=fsdp_unit_modules,
        zero_dp_strategy=zero_dp_strategy,
        outer_dp_sharding_strategy=outer_dp_sharding_strategy,
        device=device,
        init_model_with_meta_device=init_model_with_meta_device,
        mixed_precision_policy=mixed_precision_policy,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        sync_model_each_microbatch=sync_model_each_microbatch,
        preproc_state_dict_for_dcp_ckpt=preproc_state_dict_for_dcp_ckpt,
        report_nan_in_param_grad=report_nan_in_param_grad,
        average_in_collective=average_in_collective,
        disable_bucketing=disable_bucketing,
        calculate_per_token_loss=calculate_per_token_loss,
        keep_fp8_transpose_cache=keep_fp8_transpose_cache,
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer,
        fsdp_db_use_persist_buf_on_alloc_fail=fsdp_db_use_persist_buf_on_alloc_fail,
        disable_symmetric_registration=disable_symmetric_registration,
        use_decoupled_grad=use_decoupled_grad,
    )

    # Extend optimizer methods to support Megatron-FSDP operations.
    # Replace the optimizer module parameter references with
    # Megatron-FSDP-managed distributed parameters.
    model._replace_param_with_distributed_if_needed()
    optimizer.param_groups.clear()
    optimizer.state.clear()
    optimizer.add_param_group({"params": model.parameters()})
    fully_shard_optimizer(
        optimizer, preproc_state_dict_for_dcp_ckpt=preproc_state_dict_for_dcp_ckpt
    )

    # Return model and optimizer.
    return model, optimizer
