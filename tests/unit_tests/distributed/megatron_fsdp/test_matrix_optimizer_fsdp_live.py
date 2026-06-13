# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Live Megatron-FSDP smoke for matrix optimizer-owned parameters.

Run with:
    torchrun --nproc_per_node=2 pytest \
      tests/unit_tests/distributed/megatron_fsdp/test_matrix_optimizer_fsdp_live.py -q
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from packaging import version
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.functional import mse_loss

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    MATRIX_OPTIMIZER_STATE_METADATA_KEY,
    fully_shard_model,
    fully_shard_optimizer,
)
from megatron.core.matrix_update import (
    MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
    MATRIX_OPTIMIZER_OWNER_MUON,
    MatrixShardSpec,
    get_matrix_shard_spec,
    register_matrix_optimizer_param,
    update_matrix_shard_spec,
)
from megatron.core.optimizer.matrix_function_optimizer import MatrixFunctionOptimizer
from megatron.core.optimizer.matrix_optimizer import (
    _make_matrix_update_rule,
    get_megatron_matrix_optimizer,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection


class _ToyMatrixModel(torch.nn.Module):
    def __init__(self, device, in_features=3, out_features=4, bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias, device=device)

    def forward(self, x):
        return self.linear(x)


class _SGDWithMasterParamState(torch.optim.SGD):
    """SGD test optimizer that materializes a same-shard master-param state."""

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state.get(param)
                if state is not None and "momentum_buffer" in state:
                    state["master_param"] = param.detach().clone()
        return result


def _world_process_groups():
    world = dist.group.WORLD
    return ProcessGroupCollection(
        tp=world,
        pp=world,
        mp=world,
        embd=world,
        pos_embd=world,
        cp=world,
        tp_cp=world,
        hcp=[],
        ep=world,
        expt_tp=world,
        tp_ep=world,
        tp_ep_pp=world,
        tp_dp_cp=world,
        dp=world,
        dp_cp=world,
        dp_cp_ag=None,
        expt_dp=world,
        expt_dp_ag=None,
        intra_dp_cp=world,
        intra_expt_dp=world,
        inter_dist_opt=world,
        intra_dist_opt=world,
    )


def _local_param_tensor(param):
    return param.to_local() if isinstance(param, DTensor) else param


@pytest.fixture(scope="module")
def distributed_cuda_setup():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")
    if version.parse(torch.__version__) < version.parse("2.4.0"):
        pytest.skip("Requires DTensor and DeviceMesh support in PyTorch 2.4.0 or later.")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    yield {
        "rank": rank,
        "world_size": world_size,
        "device": torch.device(f"cuda:{local_rank}"),
    }

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.distributed
def test_matrix_optimizer_owned_param_state_checkpoint_contract(distributed_cuda_setup):
    setup = distributed_cuda_setup
    if setup["world_size"] < 2:
        pytest.skip("Matrix-sharded Megatron-FSDP smoke requires at least 2 ranks.")

    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(setup["world_size"], 1), mesh_dim_names=("dp_shard", "tp")
    )
    toy_model = _ToyMatrixModel(setup["device"])
    register_matrix_optimizer_param(
        toy_model.linear.weight,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        toy_model.linear.weight,
        MatrixShardSpec(
            logical_shape=tuple(toy_model.linear.weight.shape),
            local_shape=tuple(toy_model.linear.weight.shape),
            tp_layout="none",
        ),
    )

    mfsdp_model = fully_shard_model(
        module=toy_model,
        device_mesh=device_mesh,
        dp_shard_dim="dp_shard",
        tp_dim="tp",
        zero_dp_strategy="optim_grads_params",
        fsdp_unit_modules=[torch.nn.Linear],
        disable_bucketing=True,
    )
    optimizer = fully_shard_optimizer(
        _SGDWithMasterParamState(mfsdp_model.parameters(), lr=0.01, momentum=0.9)
    )

    matrix_params = [
        param
        for group in optimizer.param_groups
        for param in group["params"]
        if get_matrix_shard_spec(param) is not None
    ]
    assert len(matrix_params) == 1
    matrix_param = matrix_params[0]
    matrix_spec = get_matrix_shard_spec(matrix_param)
    assert matrix_spec.dp_shard_axis == 0
    assert matrix_spec.small_gram_side == "right"

    x = torch.randn(2, 3, device=setup["device"])
    y = torch.randn(2, 4, device=setup["device"])
    loss = mse_loss(mfsdp_model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    state = optimizer.state[matrix_param]
    assert isinstance(state["momentum_buffer"], DTensor)
    assert isinstance(state["master_param"], DTensor)
    assert get_matrix_shard_spec(state["momentum_buffer"]) == matrix_spec
    assert get_matrix_shard_spec(state["master_param"]) == matrix_spec
    saved_momentum_local = state["momentum_buffer"].to_local().detach().clone()
    saved_master_param_local = state["master_param"].to_local().detach().clone()

    state_dict = optimizer.state_dict()
    metadata_block = state_dict[MATRIX_OPTIMIZER_STATE_METADATA_KEY]
    metadata = metadata_block["params"]["0"]
    assert metadata["owner"] == "muon"
    assert metadata["update_family"] == "muon"
    assert metadata["matrix_shard_contract"]["dp_shard_axis"] == 0
    assert metadata["same_shard_state_names"] == ["master_param", "momentum_buffer"]

    reloaded_optimizer = fully_shard_optimizer(
        _SGDWithMasterParamState(mfsdp_model.parameters(), lr=0.01, momentum=0.9)
    )
    reloaded_optimizer.load_state_dict(state_dict)
    reloaded_state = reloaded_optimizer.state[matrix_param]
    assert isinstance(reloaded_state["momentum_buffer"], DTensor)
    assert isinstance(reloaded_state["master_param"], DTensor)
    assert get_matrix_shard_spec(reloaded_state["momentum_buffer"]) == matrix_spec
    assert get_matrix_shard_spec(reloaded_state["master_param"]) == matrix_spec
    torch.testing.assert_close(
        reloaded_state["momentum_buffer"].to_local(), saved_momentum_local
    )
    torch.testing.assert_close(
        reloaded_state["master_param"].to_local(), saved_master_param_local
    )
    dist.barrier()


@pytest.mark.distributed
def test_matrix_function_muon_step_uses_fsdp_dtensor_local_shard(distributed_cuda_setup):
    setup = distributed_cuda_setup
    if setup["world_size"] < 2:
        pytest.skip("Matrix-sharded Megatron-FSDP smoke requires at least 2 ranks.")

    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(setup["world_size"], 1), mesh_dim_names=("dp_shard", "tp")
    )
    toy_model = _ToyMatrixModel(setup["device"])
    register_matrix_optimizer_param(
        toy_model.linear.weight,
        owner=MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        toy_model.linear.weight,
        MatrixShardSpec(
            logical_shape=tuple(toy_model.linear.weight.shape),
            local_shape=tuple(toy_model.linear.weight.shape),
            tp_layout="none",
        ),
    )

    mfsdp_model = fully_shard_model(
        module=toy_model,
        device_mesh=device_mesh,
        dp_shard_dim="dp_shard",
        tp_dim="tp",
        zero_dp_strategy="optim_grads_params",
        fsdp_unit_modules=[torch.nn.Linear],
        disable_bucketing=True,
    )
    matrix_param = next(mfsdp_model.parameters())
    matrix_spec = get_matrix_shard_spec(matrix_param)
    assert matrix_spec is not None
    assert tuple(matrix_param.to_local().shape) == matrix_spec.local_shape
    before_local = matrix_param.to_local().detach().clone()

    config = OptimizerConfig(
        matrix_optimizer="muon",
        muon_num_ns_steps=2,
        muon_coefficient_type="simple",
        muon_scale_mode="unit_rms_norm",
    )
    process_groups = SimpleNamespace(
        tp=dist.group.WORLD,
        dp=dist.group.WORLD,
        dp_cp=dist.group.WORLD,
        tp_dp_cp=dist.group.WORLD,
    )
    optimizer = fully_shard_optimizer(
        MatrixFunctionOptimizer(
            [matrix_param],
            lr=0.01,
            update_rule=_make_matrix_update_rule(config, process_groups),
            weight_decay=0.0,
        )
    )

    x = torch.randn(2, 3, device=setup["device"])
    y = torch.randn(2, 4, device=setup["device"])
    loss = mse_loss(mfsdp_model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    after_local = matrix_param.to_local().detach()
    assert torch.linalg.vector_norm(after_local - before_local) > 0
    dist.barrier()


@pytest.mark.distributed
def test_public_matrix_optimizer_builder_steps_megatron_fsdp_dtensor_shard(
    distributed_cuda_setup,
):
    setup = distributed_cuda_setup
    if setup["world_size"] < 2:
        pytest.skip("Matrix-sharded Megatron-FSDP smoke requires at least 2 ranks.")

    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(setup["world_size"], 1), mesh_dim_names=("dp_shard", "tp")
    )
    toy_model = _ToyMatrixModel(setup["device"])
    register_matrix_optimizer_param(
        toy_model.linear.weight,
        owner=MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        toy_model.linear.weight,
        MatrixShardSpec(
            logical_shape=tuple(toy_model.linear.weight.shape),
            local_shape=tuple(toy_model.linear.weight.shape),
            tp_layout="none",
        ),
    )

    mfsdp_model = fully_shard_model(
        module=toy_model,
        device_mesh=device_mesh,
        dp_shard_dim="dp_shard",
        tp_dim="tp",
        zero_dp_strategy="optim_grads_params",
        fsdp_unit_modules=[torch.nn.Linear],
        disable_bucketing=True,
    )
    matrix_param = next(mfsdp_model.parameters())
    before_local = matrix_param.to_local().detach().clone()

    config = OptimizerConfig(
        lr=0.01,
        weight_decay=0.0,
        clip_grad=0.0,
        matrix_optimizer="muon",
        use_distributed_optimizer=True,
        use_layer_wise_distributed_optimizer=True,
        use_megatron_fsdp=True,
        muon_num_ns_steps=2,
        muon_coefficient_type="simple",
        muon_scale_mode="unit_rms_norm",
    )
    optimizer = get_megatron_matrix_optimizer(
        config,
        [mfsdp_model],
        pg_collection=_world_process_groups(),
        use_gloo_process_groups=False,
    )

    x = torch.randn(2, 3, device=setup["device"])
    y = torch.randn(2, 4, device=setup["device"])
    loss = mse_loss(mfsdp_model(x), y)
    loss.backward()
    update_successful, _, _ = optimizer.step()
    optimizer.zero_grad()

    after_local = matrix_param.to_local().detach()
    assert update_successful
    assert torch.linalg.vector_norm(after_local - before_local) > 0
    dist.barrier()


@pytest.mark.distributed
def test_public_matrix_optimizer_builder_routes_fsdp_fallback_param(
    distributed_cuda_setup,
):
    setup = distributed_cuda_setup
    if setup["world_size"] < 2:
        pytest.skip("Matrix-sharded Megatron-FSDP smoke requires at least 2 ranks.")

    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(setup["world_size"], 1), mesh_dim_names=("dp_shard", "tp")
    )
    toy_model = _ToyMatrixModel(setup["device"], bias=True)
    register_matrix_optimizer_param(
        toy_model.linear.weight,
        owner=MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        toy_model.linear.weight,
        MatrixShardSpec(
            logical_shape=tuple(toy_model.linear.weight.shape),
            local_shape=tuple(toy_model.linear.weight.shape),
            tp_layout="none",
        ),
    )

    mfsdp_model = fully_shard_model(
        module=toy_model,
        device_mesh=device_mesh,
        dp_shard_dim="dp_shard",
        tp_dim="tp",
        zero_dp_strategy="optim_grads_params",
        fsdp_unit_modules=[torch.nn.Linear],
        disable_bucketing=True,
    )
    matrix_param = next(
        param for param in mfsdp_model.parameters() if get_matrix_shard_spec(param) is not None
    )
    fallback_param = next(
        param for param in mfsdp_model.parameters() if get_matrix_shard_spec(param) is None
    )
    matrix_before = _local_param_tensor(matrix_param).detach().clone()
    fallback_before = _local_param_tensor(fallback_param).detach().clone()

    config = OptimizerConfig(
        optimizer="sgd",
        lr=0.01,
        weight_decay=0.0,
        sgd_momentum=0.0,
        clip_grad=0.0,
        matrix_optimizer="muon",
        use_distributed_optimizer=True,
        use_layer_wise_distributed_optimizer=True,
        use_megatron_fsdp=True,
        muon_num_ns_steps=2,
        muon_coefficient_type="simple",
        muon_scale_mode="unit_rms_norm",
    )
    optimizer = get_megatron_matrix_optimizer(
        config,
        [mfsdp_model],
        pg_collection=_world_process_groups(),
        use_gloo_process_groups=False,
    )

    x = torch.randn(2, 3, device=setup["device"])
    y = torch.randn(2, 4, device=setup["device"])
    loss = mse_loss(mfsdp_model(x), y)
    loss.backward()
    update_successful, _, _ = optimizer.step()
    optimizer.zero_grad()

    matrix_after = _local_param_tensor(matrix_param).detach()
    fallback_after = _local_param_tensor(fallback_param).detach()
    assert update_successful
    assert torch.linalg.vector_norm(matrix_after - matrix_before) > 0
    assert torch.linalg.vector_norm(fallback_after - fallback_before) > 0
    dist.barrier()


@pytest.mark.distributed
def test_matrix_optimizer_owned_column_axis_param_state_checkpoint_contract(
    distributed_cuda_setup,
):
    setup = distributed_cuda_setup
    if setup["world_size"] < 2:
        pytest.skip("Matrix-sharded Megatron-FSDP smoke requires at least 2 ranks.")

    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(setup["world_size"], 1), mesh_dim_names=("dp_shard", "tp")
    )
    toy_model = _ToyMatrixModel(setup["device"], in_features=5, out_features=3)
    register_matrix_optimizer_param(
        toy_model.linear.weight,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    update_matrix_shard_spec(
        toy_model.linear.weight,
        MatrixShardSpec(
            logical_shape=tuple(toy_model.linear.weight.shape),
            local_shape=tuple(toy_model.linear.weight.shape),
            tp_layout="none",
        ),
    )

    mfsdp_model = fully_shard_model(
        module=toy_model,
        device_mesh=device_mesh,
        dp_shard_dim="dp_shard",
        tp_dim="tp",
        zero_dp_strategy="optim_grads_params",
        fsdp_unit_modules=[torch.nn.Linear],
        disable_bucketing=True,
    )
    optimizer = fully_shard_optimizer(
        _SGDWithMasterParamState(mfsdp_model.parameters(), lr=0.01, momentum=0.9)
    )

    matrix_params = [
        param
        for group in optimizer.param_groups
        for param in group["params"]
        if get_matrix_shard_spec(param) is not None
    ]
    assert len(matrix_params) == 1
    matrix_param = matrix_params[0]
    matrix_spec = get_matrix_shard_spec(matrix_param)
    assert matrix_spec.dp_shard_axis == 1
    assert matrix_spec.small_gram_side == "left"

    x = torch.randn(2, 5, device=setup["device"])
    y = torch.randn(2, 3, device=setup["device"])
    loss = mse_loss(mfsdp_model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    state = optimizer.state[matrix_param]
    assert isinstance(state["momentum_buffer"], DTensor)
    assert isinstance(state["master_param"], DTensor)
    assert get_matrix_shard_spec(state["momentum_buffer"]) == matrix_spec
    assert get_matrix_shard_spec(state["master_param"]) == matrix_spec
    saved_momentum_local = state["momentum_buffer"].to_local().detach().clone()
    saved_master_param_local = state["master_param"].to_local().detach().clone()

    state_dict = optimizer.state_dict()
    metadata_block = state_dict[MATRIX_OPTIMIZER_STATE_METADATA_KEY]
    metadata = metadata_block["params"]["0"]
    assert metadata["owner"] == "muon"
    assert metadata["update_family"] == "muon"
    assert metadata["matrix_shard_contract"]["dp_shard_axis"] == 1
    assert metadata["matrix_shard_contract"]["dp_shard_layout"] == "column_contiguous_flat_buffer"
    assert metadata["same_shard_state_names"] == ["master_param", "momentum_buffer"]

    reloaded_optimizer = fully_shard_optimizer(
        _SGDWithMasterParamState(mfsdp_model.parameters(), lr=0.01, momentum=0.9)
    )
    reloaded_optimizer.load_state_dict(state_dict)
    reloaded_state = reloaded_optimizer.state[matrix_param]
    assert isinstance(reloaded_state["momentum_buffer"], DTensor)
    assert isinstance(reloaded_state["master_param"], DTensor)
    assert get_matrix_shard_spec(reloaded_state["momentum_buffer"]) == matrix_spec
    assert get_matrix_shard_spec(reloaded_state["master_param"]) == matrix_spec
    torch.testing.assert_close(
        reloaded_state["momentum_buffer"].to_local(), saved_momentum_local
    )
    torch.testing.assert_close(
        reloaded_state["master_param"].to_local(), saved_master_param_local
    )
    dist.barrier()
