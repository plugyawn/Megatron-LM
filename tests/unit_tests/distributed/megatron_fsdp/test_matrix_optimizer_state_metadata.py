# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

fully_shard_module = importlib.import_module(
    "megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard"
)
import megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer as param_buffer_module
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.matrix_optimizer import MegatronFSDPOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    Bucket,
    BucketingPolicy,
    DataParallelBuffer,
    _build_matrix_fsdp_shard_plan,
    _get_parameter_groups,
    _pack_matrix_fsdp_local_shard,
    _pack_matrix_fsdp_global_bucket,
    _unpack_matrix_fsdp_global_bucket,
    _unpack_matrix_fsdp_local_shard,
    build_data_parallel_buffer_index,
)
from megatron.core.matrix_update import (
    MATRIX_OPTIMIZER_OWNER_MUON,
    MatrixOptimizerStateSpec,
    MatrixShardSpec,
    get_matrix_optimizer_info,
    get_matrix_shard_spec,
    register_matrix_optimizer_param,
    update_matrix_optimizer_state_spec,
    update_matrix_shard_spec,
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
        del run_check, stride
        result = torch.nn.Parameter(local_tensor.new_empty(tuple(shape)), requires_grad=False)
        result.device_mesh = device_mesh
        result.placements = placements
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
    param_and_grad_buffer = SimpleNamespace(bucketing_policy=bucketing_policy, param_to_name={})
    return SimpleNamespace(param_and_grad_buffer=param_and_grad_buffer)


def _fake_distributed_optimizer(param, optimizer):
    distopt = object.__new__(DistributedOptimizer)
    distopt.ddp_config = SimpleNamespace(use_megatron_fsdp=True)
    distopt.optimizer = optimizer
    distopt.model_chunks = []
    distopt.param_to_name = {param: param.megatron_fsdp_param_name}
    return distopt


def _matrix_param(shape=(2, 3), name="layers.0.weight"):
    param = _FakeDTensor(torch.zeros(*shape))
    param._local_tensor = torch.zeros(1, shape[1])
    param.device_mesh = _fake_mesh()
    param.placements = ("shard0",)
    param.megatron_fsdp_param_name = name
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
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
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=tuple(param.shape),
            local_shape=tuple(param.shape),
            tp_layout="none",
        ),
    )
    return param


class _TinyMatrixFsdpModule(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5, bias=True, device=device)
        _tag_muon_matrix_param(self.linear.weight)
        update_matrix_optimizer_state_spec(
            self.linear.weight,
            MatrixOptimizerStateSpec(("momentum_buffer", "master_param")),
        )

    def forward(self, x):
        return self.linear(x)


def _fake_distributed_optimizer_for_params(params, optimizer):
    distopt = object.__new__(DistributedOptimizer)
    distopt.ddp_config = SimpleNamespace(use_megatron_fsdp=True)
    distopt.optimizer = optimizer
    distopt.model_chunks = []
    distopt.param_to_name = {
        param: param.megatron_fsdp_param_name
        for param in params
        if hasattr(param, "megatron_fsdp_param_name")
    }
    return distopt


def _real_fsdp_matrix_distopt_state(seed):
    torch.manual_seed(seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    module = _TinyMatrixFsdpModule(device=torch.device("cuda", local_rank))
    mfsdp_model = fully_shard_module.fully_shard_model(
        module,
        fsdp_unit_modules=[torch.nn.Linear],
        zero_dp_strategy="optim_grads_params",
        device=torch.device("cuda", local_rank),
        sync_model_each_microbatch=True,
    )
    params = list(mfsdp_model.parameters())
    optimizer = torch.optim.SGD(params, lr=0.1)
    matrix_param = next(
        param
        for param in params
        if get_matrix_optimizer_info(param) is not None
        and get_matrix_optimizer_info(param).owner == MATRIX_OPTIMIZER_OWNER_MUON
    )
    momentum = matrix_param.detach().clone()
    optimizer.state[matrix_param]["momentum_buffer"] = momentum
    return (
        _fake_distributed_optimizer_for_params(params, optimizer),
        optimizer,
        matrix_param,
        momentum,
    )


def _tag_muon_matrix_param_with_axis1_fsdp(param):
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
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
        "param_identity": fully_shard_module._matrix_optimizer_param_checkpoint_identity(
            param, _fake_mfsdp_model()
        ),
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
        "declared_same_shard_state_names": ["master_param", "momentum_buffer"],
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
    assert metadata["0"]["param_identity"] == "layers.0.weight"
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


def test_matrix_shard_checkpoint_spec_records_column_axis_range():
    spec = MatrixShardSpec(
        logical_shape=(3, 8),
        local_shape=(3, 2),
        pre_dp_local_shape=(3, 8),
        tp_layout="none",
        dp_shard_axis=1,
        dp_local_start=4,
        dp_local_end=6,
    )

    metadata = fully_shard_module._matrix_shard_spec_to_checkpoint_dict(spec)

    assert metadata["local_shape"] == [3, 2]
    assert metadata["pre_dp_local_shape"] == [3, 8]
    assert metadata["dp_shard_axis"] == 1
    assert metadata["dp_local_start"] == 4
    assert metadata["dp_local_end"] == 6


def test_matrix_shard_checkpoint_validates_column_axis_contract_and_spec():
    spec = MatrixShardSpec(
        logical_shape=(3, 8),
        local_shape=(3, 2),
        pre_dp_local_shape=(3, 8),
        tp_layout="none",
        dp_shard_axis=1,
        dp_local_start=4,
        dp_local_end=6,
    )

    spec_metadata = fully_shard_module._matrix_shard_spec_to_checkpoint_dict(spec)
    contract_metadata = fully_shard_module._matrix_shard_global_contract_to_checkpoint_dict(
        spec
    )

    assert spec_metadata["dp_shard_layout"] == "column_contiguous_flat_buffer"
    assert contract_metadata["dp_shard_layout"] == "column_contiguous_flat_buffer"
    assert spec_metadata["small_gram_side"] == "left"
    assert contract_metadata["small_gram_side"] == "left"
    fully_shard_module._validate_matrix_shard_spec_checkpoint_metadata(
        spec_metadata, "0", "matrix_shard_spec"
    )
    fully_shard_module._validate_matrix_shard_contract_checkpoint_metadata(
        contract_metadata, "0"
    )


def test_matrix_shard_checkpoint_rejects_column_axis_range_mismatch():
    spec = MatrixShardSpec(
        logical_shape=(3, 8),
        local_shape=(3, 2),
        pre_dp_local_shape=(3, 8),
        tp_layout="none",
        dp_shard_axis=1,
        dp_local_start=4,
        dp_local_end=6,
    )
    metadata = fully_shard_module._matrix_shard_spec_to_checkpoint_dict(spec)
    metadata["local_shape"] = [3, 3]

    with pytest.raises(RuntimeError, match="DP-axis size"):
        fully_shard_module._validate_matrix_shard_spec_checkpoint_metadata(
            metadata, "0", "matrix_shard_spec"
        )


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


def test_matrix_optimizer_checkpoint_rejects_missing_param_identity(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    delattr(param, "megatron_fsdp_param_name")
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.state[param]["momentum_buffer"] = _fake_dtensor_state()

    with pytest.raises(RuntimeError, match="stable parameter identity"):
        fully_shard_module._add_matrix_optimizer_checkpoint_metadata(
            optimizer, _fake_mfsdp_model(), {"state": {}, "param_groups": []}
        )


def test_matrix_optimizer_checkpoint_rejects_param_identity_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param(name="layers.0.weight")
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = {
        "0": _metadata_for_param(param, param_identity="layers.1.weight")
    }
    load_state_dict = {
        "state": {0: {"momentum_buffer": _fake_dtensor_state()}},
        "param_groups": [],
        fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
            "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
            "params": metadata,
        },
    }

    with pytest.raises(RuntimeError, match="parameter identity"):
        fully_shard_module._validate_matrix_optimizer_checkpoint_metadata(
            optimizer, _fake_mfsdp_model(), load_state_dict
        )


def test_matrix_optimizer_checkpoint_rejects_duplicate_saved_param_identity(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param_a = _matrix_param(name="layers.duplicate.weight")
    param_b = _matrix_param(name="layers.duplicate.weight")
    optimizer = torch.optim.SGD([param_a, param_b], lr=0.1)
    optimizer.state[param_a]["momentum_buffer"] = _fake_dtensor_state()
    optimizer.state[param_b]["momentum_buffer"] = _fake_dtensor_state()

    with pytest.raises(RuntimeError, match="parameter identity must be unique"):
        fully_shard_module._add_matrix_optimizer_checkpoint_metadata(
            optimizer, _fake_mfsdp_model(), {"state": {}, "param_groups": []}
        )


def test_matrix_optimizer_checkpoint_rejects_duplicate_loaded_param_identity(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param_a = _matrix_param(name="layers.0.weight")
    param_b = _matrix_param(name="layers.1.weight")
    optimizer = torch.optim.SGD([param_a, param_b], lr=0.1)
    duplicate_metadata = _metadata_for_param(param_a, param_identity="layers.0.weight")
    load_state_dict = {
        "state": {
            0: {"momentum_buffer": _fake_dtensor_state()},
            1: {"momentum_buffer": _fake_dtensor_state()},
        },
        "param_groups": [],
        fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY: {
            "version": fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_VERSION,
            "params": {
                "0": _metadata_for_param(param_a, param_identity="layers.0.weight"),
                "1": duplicate_metadata,
            },
        },
    }

    with pytest.raises(RuntimeError, match="parameter identity must be unique"):
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


def test_fully_shard_optimizer_registration_does_not_step():
    class _StepCountingSGD(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_calls = 0

        def step(self, *args, **kwargs):
            self.step_calls += 1
            return super().step(*args, **kwargs)

    param = torch.nn.Parameter(torch.ones(2, 3))
    mfsdp_model = SimpleNamespace(
        model_auto_sync=True,
        param_and_grad_buffer=SimpleNamespace(
            bucketing_policy=SimpleNamespace(data_parallel_sharding_strategy="no_shard")
        ),
        finish_grad_sync=lambda: None,
        install_optimized_model_weights=lambda: None,
        zero_grad_buffer=lambda: None,
    )
    param._megatron_fsdp_model = mfsdp_model
    optimizer = _StepCountingSGD([param], lr=0.1, momentum=0.9, weight_decay=0.1)
    param_before = param.detach().clone()

    fully_shard_module.fully_shard_optimizer(
        optimizer, preproc_state_dict_for_dcp_ckpt=False
    )

    assert optimizer.step_calls == 0
    assert optimizer.state == {}
    torch.testing.assert_close(param, param_before)


def test_megatron_fsdp_optimizer_loading_schema_does_not_step(monkeypatch):
    class _StepCountingSGD(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_calls = 0

        def step(self, *args, **kwargs):
            self.step_calls += 1
            return super().step(*args, **kwargs)

    param = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = _StepCountingSGD([param], lr=0.1, momentum=0.9, weight_decay=0.1)
    param_before = param.detach().clone()

    monkeypatch.setattr(
        fully_shard_module,
        "_megatron_fsdp_model_from_optimizer_params",
        lambda optimizer: SimpleNamespace(
            param_and_grad_buffer=SimpleNamespace(
                bucketing_policy=SimpleNamespace(data_parallel_sharding_strategy="no_shard")
            )
        ),
    )
    monkeypatch.setattr(
        fully_shard_module,
        "_optimizer_param_state_indices",
        lambda optimizer: {id(param): 0},
    )
    monkeypatch.setattr(
        fully_shard_module,
        "_matrix_optimizer_param_checkpoint_identity",
        lambda param, mfsdp_model: "param0",
    )

    wrapped = MegatronFSDPOptimizer(
        optimizer,
        OptimizerConfig(),
        SimpleNamespace(tp=None, dp=None, dp_cp=None, tp_dp_cp=None, mp=None),
    )
    state_dict = wrapped.sharded_state_dict({}, is_loading=True)

    assert optimizer.step_calls == 0
    assert optimizer.state == {}
    torch.testing.assert_close(param, param_before)
    assert "momentum_buffer" in state_dict["state"]["param0"]


def test_distributed_optimizer_fsdp_dtensor_loading_schema_does_not_step():
    class _StepCountingSGD(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_calls = 0

        def step(self, *args, **kwargs):
            self.step_calls += 1
            return super().step(*args, **kwargs)

    param = torch.nn.Parameter(torch.ones(2, 3))
    param.megatron_fsdp_param_name = "param0"
    optimizer = _StepCountingSGD([param], lr=0.1, momentum=0.9, weight_decay=0.1)
    distopt = _fake_distributed_optimizer(param, optimizer)
    param_before = param.detach().clone()

    state_dict = distopt.sharded_param_state_fsdp_dtensor(is_loading=True)

    assert optimizer.step_calls == 0
    assert optimizer.state == {}
    torch.testing.assert_close(param, param_before)
    assert "momentum_buffer" in state_dict["state"]["param0"]


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
    assert metadata["declared_same_shard_state_names"] == ["master_param", "momentum_buffer"]
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


def test_distributed_optimizer_fsdp_dtensor_state_records_matrix_metadata(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    param._megatron_fsdp_model = _fake_mfsdp_model()
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.state[param]["momentum_buffer"] = _fake_dtensor_state()
    distopt = _fake_distributed_optimizer(param, optimizer)

    state_dict = distopt.sharded_param_state_fsdp_dtensor()

    metadata_key = fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY
    assert "layers.0.weight" in state_dict["state"]
    assert metadata_key in state_dict
    metadata = state_dict[metadata_key]["params"]["0"]
    assert metadata["param_identity"] == "layers.0.weight"
    assert metadata["same_shard_state_names"] == ["momentum_buffer"]
    assert metadata["declared_same_shard_state_names"] == [
        "master_param",
        "momentum_buffer",
    ]


def test_distributed_optimizer_fsdp_dtensor_load_preserves_matrix_state(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    save_param = _matrix_param()
    save_param._megatron_fsdp_model = _fake_mfsdp_model()
    save_optimizer = torch.optim.SGD([save_param], lr=0.1)
    saved_momentum = _fake_dtensor_state()
    save_optimizer.state[save_param]["momentum_buffer"] = saved_momentum
    save_distopt = _fake_distributed_optimizer(save_param, save_optimizer)
    state_dict = save_distopt.sharded_param_state_fsdp_dtensor()

    load_param = _matrix_param()
    load_param._megatron_fsdp_model = _fake_mfsdp_model()
    load_optimizer = torch.optim.SGD([load_param], lr=0.1)
    load_distopt = _fake_distributed_optimizer(load_param, load_optimizer)

    load_distopt.load_state_dict(state_dict)

    assert "momentum_buffer" in load_optimizer.state[load_param]
    loaded_momentum = load_optimizer.state[load_param]["momentum_buffer"]
    assert tuple(loaded_momentum.shape) == tuple(saved_momentum.shape)
    torch.testing.assert_close(loaded_momentum, saved_momentum)


def test_real_fsdp_dtensor_distributed_optimizer_load_preserves_matrix_state():
    if not torch.cuda.is_available():
        pytest.skip("real Megatron-FSDP DTensor smoke requires CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("run with torchrun --nproc-per-node=2 to exercise real FSDP sharding")

    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        initialized_here = True
    try:
        save_distopt, _, _, saved_momentum = _real_fsdp_matrix_distopt_state(seed=1234)
        state_dict = save_distopt.sharded_param_state_fsdp_dtensor()
        metadata_key = fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY
        assert metadata_key in state_dict
        assert state_dict[metadata_key]["params"]["0"]["same_shard_state_names"] == [
            "momentum_buffer"
        ]

        load_distopt, load_optimizer, load_matrix_param, _ = _real_fsdp_matrix_distopt_state(
            seed=5678
        )
        load_distopt.load_state_dict(state_dict)

        loaded_momentum = load_optimizer.state[load_matrix_param]["momentum_buffer"]
        assert type(loaded_momentum) is type(saved_momentum)
        assert loaded_momentum.placements == saved_momentum.placements
        torch.testing.assert_close(loaded_momentum.to_local(), saved_momentum.to_local())
        dist.barrier()
    finally:
        if initialized_here:
            dist.destroy_process_group()


def test_distributed_optimizer_fsdp_dtensor_load_validates_matrix_metadata(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    param._megatron_fsdp_model = _fake_mfsdp_model()
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.state[param]["momentum_buffer"] = _fake_dtensor_state()
    distopt = _fake_distributed_optimizer(param, optimizer)
    state_dict = distopt.sharded_param_state_fsdp_dtensor()
    metadata_key = fully_shard_module.MATRIX_OPTIMIZER_STATE_METADATA_KEY
    state_dict[metadata_key]["params"]["0"]["owner"] = "fallback"

    with pytest.raises(RuntimeError, match="checkpoint owner does not match"):
        distopt.load_state_dict(state_dict)


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

    with pytest.raises(RuntimeError, match="DP-axis size does not match"):
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


def test_matrix_optimizer_checkpoint_rejects_malformed_declared_state_names(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param, declared_same_shard_state_names="momentum_buffer")

    with pytest.raises(RuntimeError, match="declared_same_shard_state_names"):
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


def test_matrix_optimizer_checkpoint_rejects_declared_state_contract_mismatch(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(
        param, declared_same_shard_state_names=["momentum_buffer"]
    )

    with pytest.raises(RuntimeError, match="declared state contract"):
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


def test_matrix_optimizer_checkpoint_allows_legacy_missing_declared_state_names(monkeypatch):
    monkeypatch.setattr(fully_shard_module, "DTensor", _FakeDTensor)
    param = _matrix_param()
    optimizer = torch.optim.SGD([param], lr=0.1)
    metadata = _metadata_for_param(param)
    metadata.pop("declared_same_shard_state_names")

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


def test_matrix_fsdp_shard_plan_row_axis_matches_flat_row_layout():
    rank0_plan = _build_matrix_fsdp_shard_plan(
        torch.Size([5, 3]), dp_shard_axis=0, dp_rank=0, dp_world_size=2
    )
    rank1_plan = _build_matrix_fsdp_shard_plan(
        torch.Size([5, 3]), dp_shard_axis=0, dp_rank=1, dp_world_size=2
    )

    assert rank0_plan.dp_shard_layout == "row_contiguous_flat_buffer"
    assert rank0_plan.local_axis_start == 0
    assert rank0_plan.local_axis_end == 3
    assert rank0_plan.local_shape == (3, 3)
    assert rank0_plan.padded_local_shape == (3, 3)
    assert rank0_plan.chunk_size_factor == 3
    assert not rank0_plan.requires_matrix_axis_packing
    assert rank1_plan.local_axis_start == 3
    assert rank1_plan.local_axis_end == 5
    assert rank1_plan.local_shape == (2, 3)
    assert rank1_plan.padded_local_shape == (3, 3)


def test_matrix_fsdp_shard_plan_column_axis_requires_column_packing():
    rank0_plan = _build_matrix_fsdp_shard_plan(
        torch.Size([3, 5]), dp_shard_axis=1, dp_rank=0, dp_world_size=2
    )
    rank1_plan = _build_matrix_fsdp_shard_plan(
        torch.Size([3, 5]), dp_shard_axis=1, dp_rank=1, dp_world_size=2
    )

    assert rank0_plan.dp_shard_layout == "column_contiguous_flat_buffer"
    assert rank0_plan.local_axis_start == 0
    assert rank0_plan.local_axis_end == 3
    assert rank0_plan.local_shape == (3, 3)
    assert rank0_plan.padded_local_shape == (3, 3)
    assert rank0_plan.chunk_size_factor == 3
    assert rank0_plan.requires_matrix_axis_packing
    assert rank1_plan.local_axis_start == 3
    assert rank1_plan.local_axis_end == 5
    assert rank1_plan.local_shape == (3, 2)
    assert rank1_plan.padded_local_shape == (3, 3)
    assert rank1_plan.requires_matrix_axis_packing


def test_matrix_fsdp_pack_unpack_row_axis_local_shard():
    matrix = torch.arange(15, dtype=torch.float32).view(5, 3)
    plan = _build_matrix_fsdp_shard_plan(
        torch.Size([5, 3]), dp_shard_axis=0, dp_rank=1, dp_world_size=2
    )

    packed = _pack_matrix_fsdp_local_shard(matrix, plan)

    expected_padded = torch.tensor(
        [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [0.0, 0.0, 0.0]]
    )
    assert torch.equal(packed, expected_padded.reshape(-1))
    assert torch.equal(_unpack_matrix_fsdp_local_shard(packed, plan), matrix[3:5, :])


def test_matrix_fsdp_pack_unpack_column_axis_local_shard():
    matrix = torch.arange(15, dtype=torch.float32).view(3, 5)
    plan = _build_matrix_fsdp_shard_plan(
        torch.Size([3, 5]), dp_shard_axis=1, dp_rank=1, dp_world_size=2
    )

    packed = _pack_matrix_fsdp_local_shard(matrix, plan)

    expected_padded = torch.tensor(
        [[3.0, 4.0, 0.0], [8.0, 9.0, 0.0], [13.0, 14.0, 0.0]]
    )
    assert torch.equal(packed, expected_padded.reshape(-1))
    assert torch.equal(_unpack_matrix_fsdp_local_shard(packed, plan), matrix[:, 3:5])


def test_matrix_fsdp_global_bucket_round_trips_row_axis():
    matrix = torch.arange(15, dtype=torch.float32).view(5, 3)

    packed = _pack_matrix_fsdp_global_bucket(
        matrix, dp_shard_axis=0, dp_world_size=2
    )

    expected = torch.tensor(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert torch.equal(packed, expected)
    assert torch.equal(
        _unpack_matrix_fsdp_global_bucket(
            packed, torch.Size([5, 3]), dp_shard_axis=0, dp_world_size=2
        ),
        matrix,
    )


def test_matrix_fsdp_global_bucket_round_trips_column_axis():
    matrix = torch.arange(15, dtype=torch.float32).view(3, 5)

    packed = _pack_matrix_fsdp_global_bucket(
        matrix, dp_shard_axis=1, dp_world_size=2
    )

    expected = torch.tensor(
        [
            0.0,
            1.0,
            2.0,
            5.0,
            6.0,
            7.0,
            10.0,
            11.0,
            12.0,
            3.0,
            4.0,
            0.0,
            8.0,
            9.0,
            0.0,
            13.0,
            14.0,
            0.0,
        ]
    )
    assert torch.equal(packed, expected)
    assert torch.equal(
        _unpack_matrix_fsdp_global_bucket(
            packed, torch.Size([3, 5]), dp_shard_axis=1, dp_world_size=2
        ),
        matrix,
    )


def test_matrix_fsdp_buffer_set_get_row_axis_packed_item(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = torch.nn.Parameter(torch.zeros(5, 3))
    buffer = DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=3,
    )
    buffer.init_data(torch.full((buffer.data_size,), -1.0))
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=(5, 3),
            local_shape=(2, 3),
            pre_dp_local_shape=(5, 3),
            tp_layout="none",
            dp_shard_axis=0,
            dp_local_start=3,
            dp_local_end=5,
        ),
    )
    matrix = torch.arange(15, dtype=torch.float32).view(5, 3)

    buffer.set_item(0, matrix)

    expected = torch.tensor([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 0.0, 0.0, 0.0])
    assert torch.equal(buffer.get_item(0), expected)


def test_matrix_fsdp_buffer_set_get_column_axis_packed_item(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = torch.nn.Parameter(torch.zeros(3, 5))
    buffer = DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=3,
    )
    buffer.init_data(torch.full((buffer.data_size,), -1.0))
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=(3, 5),
            local_shape=(3, 2),
            pre_dp_local_shape=(3, 5),
            tp_layout="none",
            dp_shard_axis=1,
            dp_local_start=3,
            dp_local_end=5,
        ),
    )
    matrix = torch.arange(15, dtype=torch.float32).view(3, 5)

    buffer.set_item(0, matrix)

    expected = torch.tensor([3.0, 4.0, 0.0, 8.0, 9.0, 0.0, 13.0, 14.0, 0.0])
    assert torch.equal(buffer.get_item(0), expected)


def test_matrix_fsdp_buffer_unpacks_global_bucket_to_row_major_item(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = torch.nn.Parameter(torch.zeros(3, 5))
    buffer = DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=3,
    )
    buffer.init_data(torch.empty(buffer.data_size))
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=(3, 5),
            local_shape=(3, 2),
            pre_dp_local_shape=(3, 5),
            tp_layout="none",
            dp_shard_axis=1,
            dp_local_start=3,
            dp_local_end=5,
        ),
    )
    matrix = torch.arange(15, dtype=torch.float32).view(3, 5)
    bucket = Bucket(
        data=_pack_matrix_fsdp_global_bucket(
            matrix, dp_shard_axis=1, dp_world_size=2
        )
    )

    item = buffer.get_item_from_bucket(bucket, 0)

    assert torch.equal(item, matrix.reshape(-1))


def test_matrix_fsdp_buffer_row_axis_global_bucket_item_stays_bucket_view(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = torch.nn.Parameter(torch.zeros(5, 3))
    buffer = DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=3,
    )
    buffer.init_data(torch.empty(buffer.data_size))
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=(5, 3),
            local_shape=(2, 3),
            pre_dp_local_shape=(5, 3),
            tp_layout="none",
            dp_shard_axis=0,
            dp_local_start=3,
            dp_local_end=5,
        ),
    )
    matrix = torch.arange(15, dtype=torch.float32).view(5, 3)
    bucket = Bucket(
        data=_pack_matrix_fsdp_global_bucket(
            matrix, dp_shard_axis=0, dp_world_size=2
        )
    )

    item = buffer.get_item_from_bucket(bucket, 0)
    bucket.data[0] = 99.0

    assert item[0] == 99.0
    assert torch.equal(item[1:], matrix.reshape(-1)[1:])


def test_matrix_fsdp_buffer_refreshes_column_axis_param_workspace(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = torch.nn.Parameter(torch.full((3, 5), -1.0))
    buffer = DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=3,
    )
    buffer.init_data(torch.empty(buffer.data_size))
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=(3, 5),
            local_shape=(3, 2),
            pre_dp_local_shape=(3, 5),
            tp_layout="none",
            dp_shard_axis=1,
            dp_local_start=3,
            dp_local_end=5,
        ),
    )
    matrix = torch.arange(15, dtype=torch.float32).view(3, 5)
    bucket = Bucket(
        data=_pack_matrix_fsdp_global_bucket(
            matrix, dp_shard_axis=1, dp_world_size=2
        )
    )

    buffer.refresh_matrix_axis_params_from_bucket(bucket)

    assert torch.equal(param.data, matrix)


def test_matrix_fsdp_buffer_packs_column_axis_grad_workspace(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = torch.nn.Parameter(torch.zeros(3, 5))
    buffer = DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=3,
    )
    buffer.init_data(torch.empty(buffer.data_size))
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=(3, 5),
            local_shape=(3, 2),
            pre_dp_local_shape=(3, 5),
            tp_layout="none",
            dp_shard_axis=1,
            dp_local_start=3,
            dp_local_end=5,
        ),
    )
    bucket = Bucket(data=torch.zeros(18))
    matrix = torch.arange(15, dtype=torch.float32).view(3, 5)

    grad_workspace = buffer.get_item_from_bucket(bucket, 0).view(3, 5)
    grad_workspace.copy_(matrix)
    buffer.pack_matrix_axis_bucket_items(bucket)

    assert torch.equal(
        bucket.data,
        _pack_matrix_fsdp_global_bucket(
            matrix, dp_shard_axis=1, dp_world_size=2
        ),
    )


def test_matrix_fsdp_column_axis_grad_workspace_survives_fresh_bucket_wrapper(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = torch.nn.Parameter(torch.zeros(3, 5))
    buffer = DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=3,
    )
    buffer.init_data(torch.empty(buffer.data_size))
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=(3, 5),
            local_shape=(3, 2),
            pre_dp_local_shape=(3, 5),
            tp_layout="none",
            dp_shard_axis=1,
            dp_local_start=3,
            dp_local_end=5,
        ),
    )
    first_bucket = Bucket(data=torch.zeros(18))
    matrix = torch.arange(15, dtype=torch.float32).view(3, 5)

    grad_workspace = buffer.get_item_from_bucket(first_bucket, 0).view(3, 5)
    grad_workspace.copy_(matrix)
    second_bucket = Bucket(data=first_bucket.data)
    buffer.pack_matrix_axis_bucket_items(second_bucket)

    assert torch.equal(
        second_bucket.data,
        _pack_matrix_fsdp_global_bucket(
            matrix, dp_shard_axis=1, dp_world_size=2
        ),
    )
    assert buffer._matrix_axis_unpacked_items == {}


def test_matrix_optimizer_axis1_fsdp_grouping_uses_column_packing():
    module = torch.nn.Module()
    module.matrix = _tag_muon_matrix_param_with_axis1_fsdp(
        torch.nn.Parameter(torch.zeros(4, 3))
    )

    groups, _, _ = _get_parameter_groups(
        module,
        BucketingPolicy(
            suggested_bucket_size=None,
            data_parallel_sharding_strategy="optim",
        ),
        meta_device_init_fp8_params={},
    )

    matrix_groups = [group for group in groups if group.matrix_optimizer_owned]
    assert len(matrix_groups) == 1
    assert matrix_groups[0].matrix_dp_shard_axis == 1
    assert matrix_groups[0].chunk_size_factor == 4


def test_matrix_optimizer_axis1_fsdp_buffer_planner_sets_column_spec(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = _tag_muon_matrix_param_with_axis1_fsdp(
        torch.nn.Parameter(torch.zeros(4, 3))
    )

    DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=4,
    )

    matrix_shard_spec = get_matrix_shard_spec(param)
    assert matrix_shard_spec.dp_shard_axis == 1
    assert matrix_shard_spec.dp_local_start == 0
    assert matrix_shard_spec.dp_local_end == 2
    assert matrix_shard_spec.local_shape == (4, 2)
    assert matrix_shard_spec.pre_dp_local_shape == (4, 3)
    assert matrix_shard_spec.small_gram_side == "left"


def test_matrix_optimizer_unsharded_buffer_virtual_axis1_shard_uses_column_plan(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = _tag_muon_matrix_param_with_axis1_fsdp(
        torch.nn.Parameter(torch.zeros(4, 3))
    )
    buffer = DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=False,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=4,
    )
    matrix = torch.arange(12, dtype=torch.float32).view(4, 3)
    data = torch.empty(buffer.data_size, dtype=torch.float32)
    data[: matrix.numel()] = matrix.flatten()
    buffer.init_data(data)

    shard = buffer.get_item(0, only_shard=True)

    assert torch.equal(shard, matrix[:, :2].reshape(-1))


def test_matrix_optimizer_helper_buffer_does_not_overwrite_canonical_column_spec(monkeypatch):
    rank = 0
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    param = _tag_muon_matrix_param_with_axis1_fsdp(
        torch.nn.Parameter(torch.zeros(4, 3))
    )

    DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=4,
    )
    canonical_spec = get_matrix_shard_spec(param)
    assert canonical_spec.dp_local_start == 0
    assert canonical_spec.dp_local_end == 2

    rank = 1
    DataParallelBuffer(
        ddp_config=SimpleNamespace(data_parallel_sharding_strategy="optim"),
        params=[param],
        is_data_distributed=True,
        bucket_id=0,
        data_parallel_group=None,
        chunk_size_factor=4,
    )

    assert get_matrix_shard_spec(param) == canonical_spec


def test_make_fsdp_dtensor_preserves_matrix_metadata(monkeypatch):
    monkeypatch.setattr(param_buffer_module, "DTensor", _FakeFSDPDTensor)
    monkeypatch.setattr(param_buffer_module, "using_tensor_parallel", lambda *args, **kwargs: False)
    param = _tag_muon_matrix_param(torch.nn.Parameter(torch.zeros(4, 3)))
    update_matrix_shard_spec(
        param,
        param_buffer_module.matrix_shard_spec_with_dp_axis(
            get_matrix_shard_spec(param),
            dp_shard_axis=0,
            dp_local_start=0,
            dp_local_end=2,
        ),
    )

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


def test_make_fsdp_dtensor_rejects_unplanned_matrix_metadata(monkeypatch):
    monkeypatch.setattr(param_buffer_module, "DTensor", _FakeFSDPDTensor)
    monkeypatch.setattr(param_buffer_module, "using_tensor_parallel", lambda *args, **kwargs: False)
    param = _tag_muon_matrix_param(torch.nn.Parameter(torch.zeros(4, 3)))

    with pytest.raises(RuntimeError, match="DP local range metadata"):
        param_buffer_module.make_fsdp_dtensor(
            local_tensor=torch.zeros(6),
            param=param,
            dist_index=_FakeDistIndex(),
            is_sharded_param=True,
        )


def test_make_fsdp_dtensor_accepts_packed_matrix_axis1_local_tensor(monkeypatch):
    monkeypatch.setattr(param_buffer_module, "DTensor", _FakeFSDPDTensor)
    monkeypatch.setattr(param_buffer_module, "using_tensor_parallel", lambda *args, **kwargs: False)
    param = torch.nn.Parameter(torch.zeros(3, 5))
    register_matrix_optimizer_param(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        param,
        MatrixShardSpec(
            logical_shape=(3, 5),
            local_shape=(3, 2),
            pre_dp_local_shape=(3, 5),
            tp_layout="none",
            dp_shard_axis=1,
            dp_local_start=3,
            dp_local_end=5,
        ),
    )
    matrix = torch.arange(15, dtype=torch.float32).view(3, 5)
    shard_plan = _build_matrix_fsdp_shard_plan(
        torch.Size([3, 5]), dp_shard_axis=1, dp_rank=1, dp_world_size=2
    )
    packed = _pack_matrix_fsdp_local_shard(matrix, shard_plan)

    fsdp_param = param_buffer_module.make_fsdp_dtensor(
        local_tensor=packed,
        param=param,
        dist_index=_FakeDistIndex(),
        is_sharded_param=True,
    )

    assert fsdp_param._local_tensor.shape == torch.Size([3, 3])
    assert torch.equal(fsdp_param._local_tensor, packed.view(3, 3))
    assert fsdp_param.placements[0].dim == 1
    assert get_matrix_optimizer_info(fsdp_param) == get_matrix_optimizer_info(param)
    assert get_matrix_shard_spec(fsdp_param) == get_matrix_shard_spec(param)
    assert get_matrix_shard_spec(fsdp_param).small_gram_side == "left"


def test_matrix_optimizer_wide_unsharded_matrix_groups_for_column_packing():
    module = torch.nn.Module()
    module.matrix = _tag_muon_matrix_param(torch.nn.Parameter(torch.zeros(3, 8)))

    groups, _, _ = _get_parameter_groups(
        module,
        BucketingPolicy(
            suggested_bucket_size=None,
            data_parallel_sharding_strategy="optim",
        ),
        meta_device_init_fp8_params={},
    )

    matrix_groups = [group for group in groups if group.matrix_optimizer_owned]
    assert len(matrix_groups) == 1
    assert matrix_groups[0].matrix_dp_shard_axis == 1
    assert matrix_groups[0].chunk_size_factor == 3


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
