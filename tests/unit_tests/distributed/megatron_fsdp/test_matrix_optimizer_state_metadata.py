# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
import importlib

import pytest
import torch

fully_shard_module = importlib.import_module(
    "megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard"
)
import megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer as param_buffer_module
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    BucketingPolicy,
    DataParallelBuffer,
    _get_parameter_groups,
    build_data_parallel_buffer_index,
)
from megatron.core.matrix_update import (
    MATRIX_OPTIMIZER_OWNER_MUON,
    MatrixShardSpec,
    get_matrix_optimizer_info,
    get_matrix_shard_spec,
    set_matrix_optimizer_info,
    set_matrix_shard_spec,
)


class _FakeDTensor(torch.nn.Parameter):
    pass


class _FakeFSDPDTensor:
    @staticmethod
    def from_local(
        local_tensor,
        device_mesh,
        placements,
        run_check=False,
        shape=None,
        stride=None,
    ):
        del device_mesh, placements, run_check, stride
        result = torch.nn.Parameter(local_tensor.new_empty(tuple(shape)), requires_grad=False)
        result._local_tensor = local_tensor
        return result


class _FakeDistIndex:
    use_hybrid_fsdp = False
    dp_shard_dim = "dp"

    def get_submesh(self, mesh_dim_names, is_expert_parallel=False):
        del mesh_dim_names, is_expert_parallel
        return _fake_mesh()


def _fake_mesh():
    return SimpleNamespace(mesh_dim_names=("dp",), mesh=torch.tensor([0]))


def _fake_mfsdp_model():
    bucketing_policy = SimpleNamespace(data_parallel_sharding_strategy="optim")
    param_and_grad_buffer = SimpleNamespace(bucketing_policy=bucketing_policy)
    return SimpleNamespace(param_and_grad_buffer=param_and_grad_buffer)


def _matrix_param(shape=(2, 3)):
    param = _FakeDTensor(torch.zeros(*shape))
    param._local_tensor = torch.zeros(1, shape[1])
    param.device_mesh = _fake_mesh()
    param.placements = ("shard0",)
    set_matrix_optimizer_info(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    set_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=shape,
            local_shape=(1, shape[1]),
            tp_layout="none",
            tp_shard_axis=None,
            dp_shard_axis=0,
            dp_local_start=0,
            dp_local_end=1,
        ),
    )
    return param


def _tag_muon_matrix_param(param):
    set_matrix_optimizer_info(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    set_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=tuple(param.shape),
            local_shape=tuple(param.shape),
            tp_layout="none",
        ),
    )
    return param


def _tag_muon_matrix_param_with_axis1_fsdp(param):
    set_matrix_optimizer_info(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    set_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=(param.shape[0], param.shape[1] * 2),
            local_shape=tuple(param.shape),
            tp_layout="row_parallel",
            tp_shard_axis=1,
        ),
    )
    return param


def _fake_dtensor_state(shape=(2, 3)):
    state = _FakeDTensor(torch.ones(*shape), requires_grad=False)
    state._local_tensor = torch.ones(1, shape[1])
    state.device_mesh = _fake_mesh()
    state.placements = ("shard0",)
    return state


def _metadata_for_param(param, **overrides):
    matrix_shard_spec = fully_shard_module.get_matrix_shard_spec(param)
    metadata = {
        "owner": "muon",
        "update_family": "muon",
        "matrix_shard_contract": (
            fully_shard_module._matrix_shard_global_contract_to_checkpoint_dict(
                matrix_shard_spec
            )
        ),
        "matrix_shard_spec": fully_shard_module._matrix_shard_spec_to_checkpoint_dict(
            matrix_shard_spec
        ),
        "same_shard_state_layout": fully_shard_module.MATRIX_OPTIMIZER_SAME_SHARD_STATE_LAYOUT,
        "same_shard_state_names": ["momentum_buffer"],
        "same_shard_state_shapes": {"momentum_buffer": list(param.shape)},
    }
    metadata.update(overrides)
    return metadata


def test_matrix_optimizer_checkpoint_metadata_round_trip_contract(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.state[param]["momentum_buffer"] = _fake_dtensor_state()
    optimizer.state[param]["step"] = torch.tensor(1.0)
    state_dict = {"state": {}, "param_groups": []}

    fully_shard_module._add_matrix_optimizer_checkpoint_metadata(
        optimizer, _fake_mfsdp_model(), state_dict
    )

    metadata_block = state_dict[fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY]
    assert metadata_block["version"] == fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION
    metadata = metadata_block["params"]
    assert metadata["0"]["owner"] == "muon"
    assert metadata["0"]["update_family"] == "muon"
    assert metadata["0"]["same_shard_state_layout"] == "same_as_param"
    assert metadata["0"]["same_shard_state_names"] == ["momentum_buffer"]
    assert metadata["0"]["same_shard_state_shapes"] == {"momentum_buffer": [2, 3]}
    assert metadata["0"]["matrix_shard_contract"]["dp_shard_axis"] == 0
    assert metadata["0"]["matrix_shard_contract"]["dp_shard_layout"] == "row_contiguous_flat_buffer"
    assert metadata["0"]["matrix_shard_contract"]["small_gram_side"] == "right"
    assert metadata["0"]["matrix_shard_spec"]["dp_shard_axis"] == 0
    assert metadata["0"]["matrix_shard_spec"]["pre_dp_local_shape"] is None
    assert metadata["0"]["matrix_shard_spec"]["dp_shard_layout"] == "row_contiguous_flat_buffer"
    assert metadata["0"]["matrix_shard_spec"]["small_gram_side"] == "right"

    load_state_dict = {
        "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
        "param_groups": [],
        fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: metadata_block,
    }
    fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
        optimizer, _fake_mfsdp_model(), load_state_dict
    )


def test_matrix_shard_checkpoint_spec_records_pre_dp_local_shape():
    spec = MatrixShardSpec(
        logical_shape=(8, 3),
        local_shape=(2, 3),
        pre_dp_local_shape=(8, 3),
        tp_layout="none",
        dp_shard_axis=0,
        dp_local_start=2,
        dp_local_end=4,
    )

    metadata = fully_shard_module._matrix_shard_spec_to_checkpoint_dict(spec)

    assert metadata["local_shape"] == [2, 3]
    assert metadata["pre_dp_local_shape"] == [8, 3]
    assert metadata["dp_local_start"] == 2
    assert metadata["dp_local_end"] == 4


def test_matrix_optimizer_checkpoint_rejects_local_same_shaped_state(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    local_state = torch.ones(tuple(param.shape))
    metadata = {"0": _metadata_for_param(param)}
    load_state_dict = {
        "state": {0: {"momentum_buffer": local_state}},
        "param_groups": [],
        fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
            "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
            "params": metadata,
        },
    }

    with pytest.raises(RuntimeError, match="local tensor state"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer, _fake_mfsdp_model(), load_state_dict
        )


def test_matrix_optimizer_runtime_state_rejects_local_same_shaped_state(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.state[param]["momentum_buffer"] = torch.ones(tuple(param.shape))

    with pytest.raises(RuntimeError, match="local tensor"):
        fully_shard_module._validate_matrix_optimizer_state_sharding(
            optimizer, _fake_mfsdp_model()
        )


def test_matrix_optimizer_runtime_state_gets_matrix_shard_spec(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    state = _fake_dtensor_state()
    optimizer.state[param]["momentum_buffer"] = state

    fully_shard_module._validate_matrix_optimizer_state_sharding(
        optimizer, _fake_mfsdp_model()
    )

    assert get_matrix_shard_spec(state) == get_matrix_shard_spec(param)


def test_matrix_optimizer_runtime_master_param_state_gets_matrix_shard_spec(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    momentum = _fake_dtensor_state()
    master_param = _fake_dtensor_state()
    optimizer.state[param]["momentum_buffer"] = momentum
    optimizer.state[param]["master_param"] = master_param

    fully_shard_module._validate_matrix_optimizer_state_sharding(
        optimizer, _fake_mfsdp_model()
    )

    assert get_matrix_shard_spec(momentum) == get_matrix_shard_spec(param)
    assert get_matrix_shard_spec(master_param) == get_matrix_shard_spec(param)


def test_matrix_optimizer_checkpoint_metadata_records_master_param_state(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.state[param]["momentum_buffer"] = _fake_dtensor_state()
    optimizer.state[param]["master_param"] = _fake_dtensor_state()
    state_dict = {"state": {}, "param_groups": []}

    fully_shard_module._add_matrix_optimizer_checkpoint_metadata(
        optimizer, _fake_mfsdp_model(), state_dict
    )

    metadata_block = state_dict[fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY]
    metadata = metadata_block["params"]["0"]
    assert metadata["same_shard_state_names"] == ["master_param", "momentum_buffer"]
    assert metadata["same_shard_state_shapes"] == {
        "master_param": [2, 3],
        "momentum_buffer": [2, 3],
    }

    load_state_dict = {
        "state": {
            0: {
                "momentum_buffer": _fake_dtensor_state(),
                "master_param": _fake_dtensor_state(),
            }
        },
        "param_groups": [],
        fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: metadata_block,
    }
    fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
        optimizer, _fake_mfsdp_model(), load_state_dict
    )


def test_matrix_optimizer_checkpoint_rejects_shard_spec_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = {
        "0": _metadata_for_param(
            param,
            matrix_shard_contract={
                "logical_shape": [2, 3],
                "local_shape": [2, 3],
                "tp_layout": "none",
                "tp_shard_axis": None,
                "dp_shard_axis": None,
                "dp_shard_layout": None,
                "small_gram_side": "right",
            },
        )
    }
    load_state_dict = {
        "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
        "param_groups": [],
        fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
            "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
            "params": metadata,
        },
    }

    with pytest.raises(RuntimeError, match="shard metadata does not match"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer, _fake_mfsdp_model(), load_state_dict
        )


def test_matrix_optimizer_checkpoint_rejects_malformed_shard_contract_shape(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param)
    metadata["matrix_shard_contract"]["logical_shape"] = "2x3"

    with pytest.raises(RuntimeError, match="logical_shape.*list of integers"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_conflicting_contract_axes(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param)
    metadata["matrix_shard_contract"]["tp_shard_axis"] = 1
    metadata["matrix_shard_contract"]["dp_shard_axis"] = 0

    with pytest.raises(RuntimeError, match="conflicting TP/DP"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_bad_contract_small_gram_side(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param)
    metadata["matrix_shard_contract"]["small_gram_side"] = "diagonal"

    with pytest.raises(RuntimeError, match="small_gram_side"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_shard_layout_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param)
    metadata["matrix_shard_contract"]["dp_shard_layout"] = "column_axis_packed"

    with pytest.raises(RuntimeError, match="unsupported DP shard layout"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_malformed_full_shard_spec(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param, matrix_shard_spec="not-a-dict")

    with pytest.raises(RuntimeError, match="matrix_shard_spec.*must be a dict"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_malformed_full_shard_spec_shape(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    full_spec = fully_shard_module._matrix_shard_spec_to_checkpoint_dict(
        fully_shard_module.get_matrix_shard_spec(param)
    )
    full_spec["local_shape"] = "2x3"
    metadata = _metadata_for_param(param, matrix_shard_spec=full_spec)

    with pytest.raises(RuntimeError, match="local_shape.*list of integers"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_full_shard_spec_range_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    full_spec = fully_shard_module._matrix_shard_spec_to_checkpoint_dict(
        fully_shard_module.get_matrix_shard_spec(param)
    )
    full_spec["local_shape"] = [2, 3]
    full_spec["dp_local_start"] = 0
    full_spec["dp_local_end"] = 1
    metadata = _metadata_for_param(param, matrix_shard_spec=full_spec)

    with pytest.raises(RuntimeError, match="row count does not match"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_full_shard_spec_pre_dp_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    full_spec = fully_shard_module._matrix_shard_spec_to_checkpoint_dict(
        fully_shard_module.get_matrix_shard_spec(param)
    )
    full_spec["pre_dp_local_shape"] = [1, 4]
    metadata = _metadata_for_param(param, matrix_shard_spec=full_spec)

    with pytest.raises(RuntimeError, match="inconsistent with pre_dp_local_shape"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_missing_shard_contract(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param)
    metadata.pop("matrix_shard_contract")

    with pytest.raises(RuntimeError, match="missing matrix_shard_contract"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_malformed_param_metadata(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)

    with pytest.raises(RuntimeError, match="must be a dict"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": "not-a-dict"},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_missing_state_names(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param)
    metadata.pop("same_shard_state_names")

    with pytest.raises(RuntimeError, match="missing same_shard_state_names"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_state_shape_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(
        param,
        same_shard_state_shapes={"momentum_buffer": [3, 2]},
    )

    with pytest.raises(RuntimeError, match="state shapes do not match"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_state_shape_key_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(
        param,
        same_shard_state_shapes={"other_state": [2, 3]},
    )

    with pytest.raises(RuntimeError, match="state shape keys do not match"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_malformed_state_shape(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(
        param,
        same_shard_state_shapes={"momentum_buffer": "2x3"},
    )

    with pytest.raises(RuntimeError, match="list of integers"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_allows_different_rank_local_offsets(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = {
        "0": _metadata_for_param(
            param,
                matrix_shard_spec={
                    "logical_shape": [2, 3],
                    "local_shape": [1, 3],
                    "tp_layout": "none",
                    "tp_shard_axis": None,
                    "dp_shard_axis": 0,
                "dp_shard_layout": "row_contiguous_flat_buffer",
                "dp_local_start": 1,
                "dp_local_end": 2,
                "small_gram_side": "right",
            },
            matrix_shard_contract={
                "logical_shape": [2, 3],
                "local_shape": [99, 3],
                "tp_layout": "none",
                "tp_shard_axis": None,
                "dp_shard_axis": 0,
                "dp_shard_layout": "row_contiguous_flat_buffer",
                "small_gram_side": "right",
            },
        )
    }
    load_state_dict = {
        "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
        "param_groups": [],
        fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
            "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
            "params": metadata,
        },
    }

    fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
        optimizer, _fake_mfsdp_model(), load_state_dict
    )


def test_matrix_optimizer_checkpoint_rejects_owner_or_family_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    base_metadata = _metadata_for_param(param)

    owner_mismatch = dict(base_metadata, owner="matrix_function")
    with pytest.raises(RuntimeError, match="checkpoint owner does not match"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": owner_mismatch},
                },
            },
        )

    family_mismatch = dict(base_metadata, update_family="sgd")
    with pytest.raises(RuntimeError, match="checkpoint update family does not match"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": family_mismatch},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_state_layout_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param, same_shard_state_layout="flat_local")

    with pytest.raises(RuntimeError, match="state layout contract does not match"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
                    "params": {"0": metadata},
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_metadata_version_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = {"0": _metadata_for_param(param)}

    with pytest.raises(RuntimeError, match="Unsupported matrix optimizer checkpoint metadata version"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer,
            _fake_mfsdp_model(),
            {
                "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
                "param_groups": [],
                fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
                    "version": 0,
                    "params": metadata,
                },
            },
        )


def test_matrix_optimizer_checkpoint_rejects_uncontracted_sidecar_state(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.state[param]["momentum_buffer"] = _fake_dtensor_state()
    optimizer.state[param]["small_gram_factor"] = torch.ones(3, 3)

    with pytest.raises(RuntimeError, match="sidecar sharding/checkpoint contract"):
        fully_shard_module._add_matrix_optimizer_checkpoint_metadata(
            optimizer, _fake_mfsdp_model(), {"state": {}, "param_groups": []}
        )


def test_matrix_optimizer_owned_params_are_singleton_fsdp_buckets():
    module = torch.nn.Module()
    module.fallback = torch.nn.Parameter(torch.zeros(2))
    module.matrix_a = _tag_muon_matrix_param(torch.nn.Parameter(torch.zeros(4, 3)))
    module.matrix_b = _tag_muon_matrix_param(torch.nn.Parameter(torch.zeros(4, 3)))

    groups, _, _ = _get_parameter_groups(
        module,
        BucketingPolicy(
            suggested_bucket_size=None,
            data_parallel_sharding_strategy="optim",
        ),
        meta_device_init_fp8_params={},
    )

    matrix_groups = [
        group
        for group in groups
        if group.matrix_optimizer_owned
    ]
    assert len(matrix_groups) == 2
    assert all(len(group.params) == 1 for group in matrix_groups)
    assert all(group.matrix_dp_shard_axis == 0 for group in matrix_groups)
    assert all(group.chunk_size_factor == 3 for group in matrix_groups)


def test_matrix_optimizer_axis1_fsdp_rejected_at_grouping():
    module = torch.nn.Module()
    module.matrix = _tag_muon_matrix_param_with_axis1_fsdp(
        torch.nn.Parameter(torch.zeros(4, 3))
    )

    with pytest.raises(NotImplementedError, match="matrix axis 1"):
        _get_parameter_groups(
            module,
            BucketingPolicy(
                suggested_bucket_size=None,
                data_parallel_sharding_strategy="optim",
            ),
            meta_device_init_fp8_params={},
        )


def test_matrix_optimizer_axis1_fsdp_rejected_by_buffer_planner(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = _tag_muon_matrix_param_with_axis1_fsdp(
        torch.nn.Parameter(torch.zeros(4, 3))
    )

    with pytest.raises(NotImplementedError, match="matrix axis 1"):
        DataParallelBuffer(
            ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
            params=[param],
            is_data_distributed=True,
            bucket_id=0,
            data_parallel_group=None,
        )


def test_make_fsdp_dtensor_preserves_matrix_metadata(monkeypatch):
    monkeypatch.setattr(param_buffer_module, "DTensor", _FakeFSDPDTensor)
    monkeypatch.setattr(param_buffer_module, "using_tensor_parallel", lambda *args, **kwargs: False)
    param = _tag_muon_matrix_param(torch.nn.Parameter(torch.zeros(4, 3)))

    fsdp_param = param_buffer_module.make_fsdp_dtensor(
        local_tensor=torch.zeros(6),
        param=param,
        dist_index=_FakeDistIndex(),
        is_sharded_param=True,
    )

    assert get_matrix_optimizer_info(fsdp_param) == get_matrix_optimizer_info(param)
    assert get_matrix_shard_spec(fsdp_param) == get_matrix_shard_spec(param)
    assert get_matrix_shard_spec(fsdp_param).dp_shard_axis == 0
    assert get_matrix_shard_spec(fsdp_param).small_gram_side == "right"


def test_matrix_optimizer_wide_unsharded_matrix_rejected_at_grouping():
    module = torch.nn.Module()
    module.matrix = _tag_muon_matrix_param(torch.nn.Parameter(torch.zeros(3, 8)))

    with pytest.raises(NotImplementedError, match="matrix axis 1"):
        _get_parameter_groups(
            module,
            BucketingPolicy(
                suggested_bucket_size=None,
                data_parallel_sharding_strategy="optim",
            ),
            meta_device_init_fp8_params={},
        )


def test_matrix_singleton_bucket_uneven_rows_remain_row_aligned():
    row_stride = 3
    ddp_config = SimpleNamespace(data_parallel_sharding_strategy="optim")

    for dp_rank in (0, 1):
        item_index_map, _, shard_bucket_index = build_data_parallel_buffer_index(
            [torch.Size([5, row_stride])],
            data_parallel_rank=dp_rank,
            data_parallel_world_size=2,
            is_data_distributed=True,
            ddp_config=ddp_config,
            chunk_size_factor=row_stride,
        )
        item_index = item_index_map[0]
        item_global_start = item_index.global_data_index
        item_global_end = item_index.global_data_index + item_index.size
        shard_global_start = shard_bucket_index.global_data_index
        shard_global_end = shard_bucket_index.global_data_index + shard_bucket_index.size

        slice_start = max(item_global_start, shard_global_start) - item_global_start
        slice_end = min(item_global_end, shard_global_end) - item_global_start

        assert shard_bucket_index.size % row_stride == 0
        assert slice_start % row_stride == 0
        assert slice_end % row_stride == 0
