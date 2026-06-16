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


def test_matrix_optimizer_megatron_fsdp_diag_sidecar_allowed():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="diag",
        matrix_input_preconditioner_refresh_interval=1,
        matrix_input_preconditioner_ema_beta=None,
        matrix_input_preconditioner_activation_dtype="bf16_saved",
        matrix_output_preconditioner="none",
        matrix_tp_update_mode="small_gram_ns",
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
    )

    validate_matrix_optimizer_fsdp_support(args)


def test_matrix_optimizer_megatron_fsdp_full_sidecar_rejected():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="full",
        matrix_input_preconditioner_refresh_interval=1,
        matrix_input_preconditioner_ema_beta=None,
        matrix_input_preconditioner_activation_dtype="bf16_saved",
        matrix_output_preconditioner="none",
        matrix_tp_update_mode="small_gram_ns",
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
    )

    with pytest.raises(ValueError, match="FEATURE_GRAM diag sidecars only"):
        validate_matrix_optimizer_fsdp_support(args)


def test_matrix_optimizer_megatron_fsdp_fp8_dequant_sidecar_rejected():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="diag",
        matrix_input_preconditioner_refresh_interval=1,
        matrix_input_preconditioner_ema_beta=None,
        matrix_input_preconditioner_activation_dtype="fp8_dequant",
        matrix_output_preconditioner="none",
        matrix_tp_update_mode="small_gram_ns",
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
    )

    with pytest.raises(ValueError, match="activation_dtype=fp8_dequant"):
        validate_matrix_optimizer_fsdp_support(args)


def test_matrix_optimizer_megatron_fsdp_block_local_rejected():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        matrix_input_preconditioner="none",
        matrix_output_preconditioner="none",
        matrix_tp_update_mode="block_local",
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
    )

    with pytest.raises(ValueError, match="matrix-tp-update-mode=block_local"):
        validate_matrix_optimizer_fsdp_support(args)


@pytest.mark.parametrize(
    "hsdp_kwargs",
    [
        {"num_distributed_optimizer_instances": 2},
        {"enable_full_sharding_in_hsdp": True},
    ],
)
def test_matrix_optimizer_megatron_fsdp_hsdp_rejected_until_matrix_contract(hsdp_kwargs):
    args_kwargs = dict(
        matrix_optimizer="muon",
        matrix_input_preconditioner="none",
        matrix_output_preconditioner="none",
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        num_distributed_optimizer_instances=1,
        enable_full_sharding_in_hsdp=False,
    )
    args_kwargs.update(hsdp_kwargs)
    args = SimpleNamespace(**args_kwargs)

    with pytest.raises(ValueError, match="hybrid FSDP/HSDP"):
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
