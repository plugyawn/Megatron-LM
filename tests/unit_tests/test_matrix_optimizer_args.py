# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
from types import SimpleNamespace

from megatron.training.arguments import (
    normalize_matrix_and_emerging_optimizer_args,
    validate_matrix_optimizer_fsdp_support,
)


def test_matrix_optimizer_torch_fsdp2_rejected_until_matrix_axis_optimizer_routing():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        use_megatron_fsdp=False,
        use_torch_fsdp2=True,
    )

    with pytest.raises(ValueError, match="matrix-axis-aware FSDP sharding"):
        validate_matrix_optimizer_fsdp_support(args)


def test_matrix_optimizer_megatron_fsdp_no_sidecar_allowed():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        matrix_input_preconditioner="none",
        matrix_output_preconditioner="none",
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
    )

    validate_matrix_optimizer_fsdp_support(args)


def test_matrix_optimizer_megatron_fsdp_sidecar_rejected_until_buffer_routing():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        matrix_input_preconditioner="feature_gram",
        matrix_output_preconditioner="none",
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
    )

    with pytest.raises(ValueError, match="no-sidecar matrix updates only"):
        validate_matrix_optimizer_fsdp_support(args)


def test_matrix_optimizer_megatron_fsdp_preserves_distributed_optimizer_flag():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        optimizer="adam",
        use_distributed_optimizer=True,
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        use_layer_wise_param_layout=True,
    )

    normalize_matrix_and_emerging_optimizer_args(args)

    assert args.use_layer_wise_distributed_optimizer is True
    assert args.use_distributed_optimizer is True


def test_matrix_optimizer_standard_distopt_rewritten_to_layerwise_split():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        optimizer="adam",
        use_distributed_optimizer=True,
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        use_layer_wise_param_layout=True,
    )

    normalize_matrix_and_emerging_optimizer_args(args)

    assert args.use_layer_wise_distributed_optimizer is True
    assert args.use_distributed_optimizer is False
