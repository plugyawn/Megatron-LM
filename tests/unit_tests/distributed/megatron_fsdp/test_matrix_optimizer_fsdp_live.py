# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Live Megatron-FSDP smoke for matrix optimizer-owned parameters.

Run with:
    torchrun --nproc_per_node=2 pytest \
      tests/unit_tests/distributed/megatron_fsdp/test_matrix_optimizer_fsdp_live.py -q
"""

import os

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
    MATRIX_OPTIMIZER_OWNER_MUON,
    MatrixShardSpec,
    get_matrix_shard_spec,
    set_matrix_optimizer_info,
    set_matrix_shard_spec,
)


class _ToyMatrixModel(torch.nn.Module):
    def __init__(self, device, in_features=3, out_features=4):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False, device=device)

    def forward(self, x):
        return self.linear(x)


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
    set_matrix_optimizer_info(
        toy_model.linear.weight,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    set_matrix_shard_spec(
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
        torch.optim.SGD(mfsdp_model.parameters(), lr=0.01, momentum=0.9)
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

    state = optimizer.state[matrix_param]
    assert isinstance(state["momentum_buffer"], DTensor)
    assert get_matrix_shard_spec(state["momentum_buffer"]) == matrix_spec

    x = torch.randn(2, 3, device=setup["device"])
    y = torch.randn(2, 4, device=setup["device"])
    loss = mse_loss(mfsdp_model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    state_dict = optimizer.state_dict()
    metadata_block = state_dict[MATRIX_OPTIMIZER_STATE_METADATA_KEY]
    metadata = metadata_block["params"]["0"]
    assert metadata["owner"] == "muon"
    assert metadata["update_family"] == "muon"
    assert metadata["matrix_shard_contract"]["dp_shard_axis"] == 0
    assert metadata["same_shard_state_names"] == ["momentum_buffer"]

    reloaded_optimizer = fully_shard_optimizer(
        torch.optim.SGD(mfsdp_model.parameters(), lr=0.01, momentum=0.9)
    )
    reloaded_optimizer.load_state_dict(state_dict)
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
    set_matrix_optimizer_info(
        toy_model.linear.weight,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
        requires_layerwise_layout=True,
    )
    set_matrix_shard_spec(
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
        torch.optim.SGD(mfsdp_model.parameters(), lr=0.01, momentum=0.9)
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

    state = optimizer.state[matrix_param]
    assert isinstance(state["momentum_buffer"], DTensor)
    assert get_matrix_shard_spec(state["momentum_buffer"]) == matrix_spec

    x = torch.randn(2, 5, device=setup["device"])
    y = torch.randn(2, 3, device=setup["device"])
    loss = mse_loss(mfsdp_model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    state_dict = optimizer.state_dict()
    metadata_block = state_dict[MATRIX_OPTIMIZER_STATE_METADATA_KEY]
    metadata = metadata_block["params"]["0"]
    assert metadata["owner"] == "muon"
    assert metadata["update_family"] == "muon"
    assert metadata["matrix_shard_contract"]["dp_shard_axis"] == 1
    assert metadata["matrix_shard_contract"]["dp_shard_layout"] == "column_contiguous_flat_buffer"
    assert metadata["same_shard_state_names"] == ["momentum_buffer"]

    reloaded_optimizer = fully_shard_optimizer(
        torch.optim.SGD(mfsdp_model.parameters(), lr=0.01, momentum=0.9)
    )
    reloaded_optimizer.load_state_dict(state_dict)
    dist.barrier()
