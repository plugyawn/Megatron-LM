# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
from types import SimpleNamespace

from megatron.training.arguments import validate_matrix_optimizer_fsdp_support


def test_matrix_optimizer_megatron_fsdp_rejected_until_matrix_axis_optimizer_routing(
):
    args = SimpleNamespace(
        matrix_optimizer="muon",
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
    )

    with pytest.raises(ValueError, match="matrix-axis-aware FSDP sharding"):
        validate_matrix_optimizer_fsdp_support(args)
