# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.matrix_function_optimizer import (
    MatrixFunctionOptimizer,
    default_matrix_apply_plan,
)
from megatron.core.optimizer.layer_wise_optimizer import tag_params_for_buffer_routing
from megatron.core.optimizer.matrix_update import (
    MATRIX_OPTIMIZER_OWNER_FALLBACK,
    MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
    MATRIX_OPTIMIZER_OWNER_MUON,
    MatrixShardSpec,
    MatrixPreconditionerApproximation,
    MatrixPreconditionerKind,
    MatrixPreconditionerNormalization,
    MatrixPreconditionerScope,
    MatrixInputPreconditionerRecipe,
    MatrixOutputPreconditionerRecipe,
    TPUpdateMode,
    configure_model_matrix_updates,
    configure_matrix_update_param,
    finalize_feature_gram_buffers,
    finalize_grad_gram_buffers,
    get_feature_gram_for_optimizer,
    get_grad_gram_for_optimizer,
    get_matrix_optimizer_info,
    get_matrix_optimizer_state_spec,
    get_matrix_shard_spec,
    input_preconditioner_scope_for,
    is_matrix_update_eligible,
    matrix_fsdp_shard_axis_for_spec,
    matrix_shard_spec_with_dp_axis,
    matrix_small_gram_side_for_spec,
    matrix_update_family_from_optimizer_name,
    maybe_accumulate_feature_gram,
    maybe_accumulate_grad_gram,
    output_preconditioner_scope_for,
    set_feature_gram_finalization_required,
    set_grad_gram_finalization_required,
    set_linear_weight_info,
    set_matrix_optimizer_info,
    set_matrix_shard_spec,
)
from megatron.training.arguments import normalize_matrix_and_emerging_optimizer_args


def _param_with_info(tp_layout="column_parallel", shape=(4, 3), logical_shape=None):
    if logical_shape is None:
        logical_shape = shape
    param = torch.nn.Parameter(torch.empty(*shape))
    set_linear_weight_info(
        param,
        logical_shape=logical_shape,
        tp_layout=tp_layout,
        sequence_parallel=False,
        expert_parallel=False,
        has_bias=True,
    )
    return param


def _recipe(
    approximation=MatrixPreconditionerApproximation.FULL,
    normalization=MatrixPreconditionerNormalization.SUM,
    refresh_interval=1,
):
    return MatrixInputPreconditionerRecipe(
        kind=MatrixPreconditionerKind.FEATURE_GRAM,
        approximation=approximation,
        scope=MatrixPreconditionerScope.GLOBAL_EXACT,
        normalization=normalization,
        activation_dtype="fp32_cast",
        accumulation_dtype=torch.float32,
        refresh_interval=refresh_interval,
        token_sample_size=None,
        ridge=0.0,
        ema_beta=None,
        min_samples_per_feature=None,
    )


def _output_recipe(
    approximation=MatrixPreconditionerApproximation.FULL,
    normalization=MatrixPreconditionerNormalization.SUM,
    refresh_interval=1,
):
    return MatrixOutputPreconditionerRecipe(
        kind=MatrixPreconditionerKind.GRAD_GRAM,
        approximation=approximation,
        scope=MatrixPreconditionerScope.GLOBAL_EXACT,
        normalization=normalization,
        gradient_dtype="fp32_cast",
        accumulation_dtype=torch.float32,
        refresh_interval=refresh_interval,
        token_sample_size=None,
        ridge=0.0,
        ema_beta=None,
        min_samples_per_feature=None,
    )


def test_full_feature_gram_accumulates_raw_sum():
    param = _param_with_info()
    recipe = _recipe()
    configure_matrix_update_param(param, recipe=recipe)
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])

    maybe_accumulate_feature_gram(param, x)

    torch.testing.assert_close(param.main_grad_feature_gram, x.t().matmul(x))
    torch.testing.assert_close(param.main_grad_feature_count, torch.tensor(2.0, dtype=torch.float64))


def test_diag_feature_gram_accumulates_diagonal_only():
    param = _param_with_info()
    recipe = _recipe(approximation=MatrixPreconditionerApproximation.DIAG)
    configure_matrix_update_param(param, recipe=recipe)
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])

    maybe_accumulate_feature_gram(param, x)

    torch.testing.assert_close(param.main_grad_feature_gram, (x * x).sum(dim=0))


def test_block_diag_feature_gram_accumulates_padded_blocks():
    param = _param_with_info()
    recipe = _recipe(approximation=MatrixPreconditionerApproximation.BLOCK_DIAG)
    recipe.block_size = 2
    recipe.ridge = 1e-6
    configure_matrix_update_param(param, recipe=recipe)
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])

    maybe_accumulate_feature_gram(param, x)

    expected = torch.zeros(2, 2, 2)
    expected[0] = x[:, :2].t().matmul(x[:, :2])
    padded_last = torch.nn.functional.pad(x[:, 2:], (0, 1))
    expected[1] = padded_last.t().matmul(padded_last)
    torch.testing.assert_close(param.main_grad_feature_gram, expected)


def test_full_grad_gram_accumulates_raw_sum():
    param = _param_with_info()
    recipe = _output_recipe()
    configure_matrix_update_param(param, output_recipe=recipe)
    dy = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 11.0]])

    maybe_accumulate_grad_gram(param, dy)

    torch.testing.assert_close(param.main_grad_grad_gram, dy.t().matmul(dy))
    torch.testing.assert_close(param.main_grad_grad_count, torch.tensor(2.0, dtype=torch.float64))


def test_diag_grad_gram_accumulates_diagonal_only():
    param = _param_with_info()
    recipe = _output_recipe(approximation=MatrixPreconditionerApproximation.DIAG)
    configure_matrix_update_param(param, output_recipe=recipe)
    dy = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 11.0]])

    maybe_accumulate_grad_gram(param, dy)

    torch.testing.assert_close(param.main_grad_grad_gram, (dy * dy).sum(dim=0))


def test_diag_gram_cuda_routes_use_generic_reducer(monkeypatch):
    import sys
    import types

    import megatron.core.matrix_update as matrix_update_module

    calls = []

    def fake_diag_gram_reduce(x, *, out, accumulate):
        calls.append((x, out, accumulate))
        return out

    emerging_pkg = types.ModuleType("emerging_optimizers")
    emerging_pkg.__path__ = []
    kernels_pkg = types.ModuleType("emerging_optimizers.triton_kernels")
    kernels_pkg.__path__ = []
    diag_gram_module = types.ModuleType("emerging_optimizers.triton_kernels.diag_gram")
    diag_gram_module.diag_gram_reduce = fake_diag_gram_reduce
    monkeypatch.setitem(sys.modules, "emerging_optimizers", emerging_pkg)
    monkeypatch.setitem(sys.modules, "emerging_optimizers.triton_kernels", kernels_pkg)
    monkeypatch.setitem(
        sys.modules, "emerging_optimizers.triton_kernels.diag_gram", diag_gram_module
    )

    class FakeCudaTensor:
        is_cuda = True

        def __mul__(self, other):
            raise AssertionError("fallback path should not run")

    gram = FakeCudaTensor()
    x = FakeCudaTensor()
    dy = FakeCudaTensor()

    matrix_update_module._accumulate_diag_feature_gram(gram, x)
    matrix_update_module._accumulate_diag_grad_gram(gram, dy)

    assert calls == [(x, gram, True), (dy, gram, True)]


def test_diag_gram_cuda_kernel_failure_fails_closed(monkeypatch):
    import sys
    import types

    import megatron.core.matrix_update as matrix_update_module

    def fake_diag_gram_reduce(x, *, out, accumulate):
        raise RuntimeError("diag kernel failed")

    emerging_pkg = types.ModuleType("emerging_optimizers")
    emerging_pkg.__path__ = []
    kernels_pkg = types.ModuleType("emerging_optimizers.triton_kernels")
    kernels_pkg.__path__ = []
    diag_gram_module = types.ModuleType("emerging_optimizers.triton_kernels.diag_gram")
    diag_gram_module.diag_gram_reduce = fake_diag_gram_reduce
    monkeypatch.setitem(sys.modules, "emerging_optimizers", emerging_pkg)
    monkeypatch.setitem(sys.modules, "emerging_optimizers.triton_kernels", kernels_pkg)
    monkeypatch.setitem(
        sys.modules, "emerging_optimizers.triton_kernels.diag_gram", diag_gram_module
    )

    class FakeCudaTensor:
        is_cuda = True

        def __mul__(self, other):
            raise AssertionError("fallback path should not run after kernel failure")

    with pytest.raises(RuntimeError, match="diag kernel failed"):
        matrix_update_module._accumulate_diag_feature_gram(FakeCudaTensor(), FakeCudaTensor())

    with pytest.raises(RuntimeError, match="diag kernel failed"):
        matrix_update_module._accumulate_diag_grad_gram(FakeCudaTensor(), FakeCudaTensor())


def test_block_diag_grad_gram_accumulates_padded_blocks():
    param = _param_with_info()
    recipe = _output_recipe(approximation=MatrixPreconditionerApproximation.BLOCK_DIAG)
    recipe.block_size = 3
    recipe.ridge = 1e-6
    configure_matrix_update_param(param, output_recipe=recipe)
    dy = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 11.0]])

    maybe_accumulate_grad_gram(param, dy)

    expected = torch.zeros(2, 3, 3)
    expected[0] = dy[:, :3].t().matmul(dy[:, :3])
    padded_last = torch.nn.functional.pad(dy[:, 3:], (0, 2))
    expected[1] = padded_last.t().matmul(padded_last)
    torch.testing.assert_close(param.main_grad_grad_gram, expected)


def test_feature_gram_mean_normalization_is_applied_on_consumption():
    param = _param_with_info()
    recipe = _recipe(normalization=MatrixPreconditionerNormalization.MEAN)
    configure_matrix_update_param(param, recipe=recipe)
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])

    maybe_accumulate_feature_gram(param, x)

    torch.testing.assert_close(get_feature_gram_for_optimizer(param), x.t().matmul(x) / 2.0)


def test_grad_gram_mean_normalization_is_applied_on_consumption():
    param = _param_with_info()
    recipe = _output_recipe(normalization=MatrixPreconditionerNormalization.MEAN)
    configure_matrix_update_param(param, output_recipe=recipe)
    dy = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 11.0]])

    maybe_accumulate_grad_gram(param, dy)

    torch.testing.assert_close(get_grad_gram_for_optimizer(param), dy.t().matmul(dy) / 2.0)


def test_feature_gram_consumption_requires_collected_rows():
    param = _param_with_info()
    configure_matrix_update_param(param, recipe=_recipe())

    with pytest.raises(RuntimeError, match="zero collected feature rows"):
        get_feature_gram_for_optimizer(param)


def test_grad_gram_consumption_requires_collected_rows():
    param = _param_with_info()
    configure_matrix_update_param(param, output_recipe=_output_recipe())

    with pytest.raises(RuntimeError, match="zero collected grad-output rows"):
        get_grad_gram_for_optimizer(param)


def test_token_sample_size_caps_total_collection_window():
    param = _param_with_info()
    recipe = _recipe()
    recipe.token_sample_size = 3
    configure_matrix_update_param(param, recipe=recipe)
    x0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    x1 = torch.tensor([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]])

    maybe_accumulate_feature_gram(param, x0)
    maybe_accumulate_feature_gram(param, x1)

    sampled = torch.cat([x0, x1[:1]], dim=0)
    torch.testing.assert_close(param.main_grad_feature_gram, sampled.t().matmul(sampled))
    torch.testing.assert_close(param.main_grad_feature_count, torch.tensor(3.0, dtype=torch.float64))


def test_row_parallel_feature_gram_scope_is_diag_approx_for_diag():
    assert (
        input_preconditioner_scope_for(MatrixPreconditionerApproximation.DIAG, "row_parallel")
        == MatrixPreconditionerScope.DIAG_APPROX
    )


def test_column_parallel_grad_gram_scope_is_diag_approx_for_diag():
    assert (
        output_preconditioner_scope_for(MatrixPreconditionerApproximation.DIAG, "column_parallel")
        == MatrixPreconditionerScope.DIAG_APPROX
    )


def test_full_feature_gram_rejects_undersampled_rank_deficient_recipe():
    param = _param_with_info()
    recipe = _recipe()
    recipe.token_sample_size = 2
    recipe.min_samples_per_feature = 1.0

    with pytest.raises(ValueError, match="rank-deficient"):
        configure_matrix_update_param(param, recipe=recipe)


def test_sketch_approximation_is_not_public_api():
    with pytest.raises(ValueError, match="matrix_input_preconditioner_approximation"):
        OptimizerConfig(
            matrix_optimizer="sgd",
            matrix_input_preconditioner="feature_gram",
            matrix_input_preconditioner_approximation="sketch",
        )


def test_lm_head_marker_excludes_matrix_update_eligibility():
    param = _param_with_info()
    param.is_embedding_or_output_parameter = True

    assert not is_matrix_update_eligible(param, min_matrix_dim=2)


def test_matrix_shard_spec_records_actual_tp_shard_axis():
    param = _param_with_info(tp_layout="column_parallel", shape=(2, 3), logical_shape=(4, 3))

    spec = get_matrix_shard_spec(param)

    assert spec.logical_shape == (4, 3)
    assert spec.local_shape == (2, 3)
    assert spec.tp_layout == "column_parallel"
    assert spec.tp_shard_axis == 0


def test_matrix_shard_spec_does_not_infer_shard_axis_from_tp1_layout_label():
    param = _param_with_info(tp_layout="column_parallel", shape=(4, 3), logical_shape=(4, 3))

    spec = get_matrix_shard_spec(param)

    assert spec.tp_shard_axis is None


def test_matrix_fsdp_shard_axis_aligns_with_tp_axis():
    row_sharded = _param_with_info(
        tp_layout="column_parallel", shape=(2, 3), logical_shape=(4, 3)
    )
    col_sharded = _param_with_info(
        tp_layout="row_parallel", shape=(4, 2), logical_shape=(4, 4)
    )
    unsharded = _param_with_info(tp_layout="column_parallel", shape=(4, 3), logical_shape=(4, 3))
    wide_unsharded = _param_with_info(tp_layout="none", shape=(3, 8), logical_shape=(3, 8))

    assert matrix_fsdp_shard_axis_for_spec(get_matrix_shard_spec(row_sharded)) == 0
    assert matrix_fsdp_shard_axis_for_spec(get_matrix_shard_spec(col_sharded)) == 1
    assert matrix_fsdp_shard_axis_for_spec(get_matrix_shard_spec(unsharded)) == 0
    assert matrix_fsdp_shard_axis_for_spec(get_matrix_shard_spec(wide_unsharded)) == 1


def test_matrix_shard_spec_small_gram_side():
    row_sharded = _param_with_info(
        tp_layout="column_parallel", shape=(2, 3), logical_shape=(4, 3)
    )
    col_sharded = _param_with_info(
        tp_layout="row_parallel", shape=(4, 2), logical_shape=(4, 4)
    )
    tall_unsharded = _param_with_info(tp_layout="none", shape=(8, 3), logical_shape=(8, 3))
    wide_unsharded = _param_with_info(tp_layout="none", shape=(3, 8), logical_shape=(3, 8))

    assert matrix_small_gram_side_for_spec(get_matrix_shard_spec(row_sharded)) == "right"
    assert matrix_small_gram_side_for_spec(get_matrix_shard_spec(col_sharded)) == "left"
    assert matrix_small_gram_side_for_spec(get_matrix_shard_spec(tall_unsharded)) == "right"
    assert matrix_small_gram_side_for_spec(get_matrix_shard_spec(wide_unsharded)) == "left"
    assert get_matrix_shard_spec(row_sharded).small_gram_side == "right"


def test_matrix_shard_spec_rejects_conflicting_tp_dp_axes():
    col_sharded = _param_with_info(
        tp_layout="row_parallel", shape=(4, 2), logical_shape=(4, 4)
    )
    spec = get_matrix_shard_spec(col_sharded)

    with pytest.raises(ValueError, match="differs from the TP shard axis"):
        matrix_shard_spec_with_dp_axis(spec, dp_shard_axis=0)


def test_matrix_shard_spec_dp_row_range_updates_local_shape():
    param = _param_with_info(tp_layout="none", shape=(8, 3), logical_shape=(8, 3))
    spec = matrix_shard_spec_with_dp_axis(
        get_matrix_shard_spec(param),
        dp_shard_axis=0,
        dp_local_start=2,
        dp_local_end=5,
    )

    assert spec.local_shape == (3, 3)
    assert spec.pre_dp_local_shape == (8, 3)
    assert spec.dp_local_start == 2
    assert spec.dp_local_end == 5
    assert spec.small_gram_side == "right"


def test_matrix_shard_spec_dp_row_range_uses_pre_dp_shape_when_present():
    param = _param_with_info(tp_layout="none", shape=(8, 3), logical_shape=(8, 3))
    spec = matrix_shard_spec_with_dp_axis(
        get_matrix_shard_spec(param),
        dp_shard_axis=0,
        dp_local_start=2,
        dp_local_end=5,
    )
    resliced = matrix_shard_spec_with_dp_axis(
        spec,
        dp_shard_axis=0,
        dp_local_start=6,
        dp_local_end=8,
    )

    assert resliced.local_shape == (2, 3)
    assert resliced.pre_dp_local_shape == (8, 3)
    assert resliced.dp_local_start == 6
    assert resliced.dp_local_end == 8


def test_matrix_shard_spec_rejects_inconsistent_pre_dp_shape():
    with pytest.raises(ValueError, match="cannot be larger than"):
        MatrixShardSpec(
            logical_shape=(8, 3),
            local_shape=(9, 3),
            tp_layout="none",
            dp_shard_axis=0,
            pre_dp_local_shape=(8, 3),
        )


def test_matrix_shard_spec_direct_constructor_rejects_partial_dp_range():
    with pytest.raises(ValueError, match="must be set together"):
        MatrixShardSpec(
            logical_shape=(8, 3),
            local_shape=(2, 3),
            tp_layout="none",
            dp_shard_axis=0,
            dp_local_start=2,
        )


def test_matrix_shard_spec_direct_constructor_rejects_row_range_shape_mismatch():
    with pytest.raises(ValueError, match="DP-axis size must match"):
        MatrixShardSpec(
            logical_shape=(8, 3),
            local_shape=(3, 3),
            tp_layout="none",
            dp_shard_axis=0,
            dp_local_start=2,
            dp_local_end=4,
            pre_dp_local_shape=(8, 3),
        )


def test_matrix_shard_spec_direct_constructor_allows_dp_local_shape_without_pre_dp_shape():
    spec = MatrixShardSpec(
        logical_shape=(8, 3),
        local_shape=(2, 3),
        tp_layout="none",
        dp_shard_axis=0,
        dp_local_start=6,
        dp_local_end=8,
    )

    assert spec.local_shape == (2, 3)
    assert spec.pre_dp_local_shape is None
    assert spec.dp_local_start == 6
    assert spec.dp_local_end == 8


def test_matrix_shard_spec_dp_empty_row_range_is_valid():
    param = _param_with_info(tp_layout="none", shape=(8, 3), logical_shape=(8, 3))
    spec = matrix_shard_spec_with_dp_axis(
        get_matrix_shard_spec(param),
        dp_shard_axis=0,
        dp_local_start=4,
        dp_local_end=4,
    )

    assert spec.local_shape == (0, 3)
    assert spec.pre_dp_local_shape == (8, 3)
    assert spec.dp_local_start == 4
    assert spec.dp_local_end == 4
    assert spec.small_gram_side == "right"


def test_matrix_shard_spec_dp_row_range_requires_complete_range():
    param = _param_with_info(tp_layout="none", shape=(8, 3), logical_shape=(8, 3))

    with pytest.raises(ValueError, match="provided together"):
        matrix_shard_spec_with_dp_axis(
            get_matrix_shard_spec(param),
            dp_shard_axis=0,
            dp_local_start=2,
        )


def test_matrix_shard_spec_dp_row_range_rejects_out_of_bounds_range():
    param = _param_with_info(tp_layout="none", shape=(8, 3), logical_shape=(8, 3))

    with pytest.raises(ValueError, match="exceeds the pre-DP local matrix shape"):
        matrix_shard_spec_with_dp_axis(
            get_matrix_shard_spec(param),
            dp_shard_axis=0,
            dp_local_start=7,
            dp_local_end=9,
        )


def test_matrix_shard_spec_dp_column_range_updates_local_shape():
    param = _param_with_info(tp_layout="none", shape=(3, 8), logical_shape=(3, 8))
    spec = matrix_shard_spec_with_dp_axis(
        get_matrix_shard_spec(param),
        dp_shard_axis=1,
        dp_local_start=2,
        dp_local_end=5,
    )

    assert spec.local_shape == (3, 3)
    assert spec.pre_dp_local_shape == (3, 8)
    assert spec.dp_local_start == 2
    assert spec.dp_local_end == 5
    assert spec.small_gram_side == "left"


def test_matrix_shard_spec_direct_constructor_allows_column_range():
    spec = MatrixShardSpec(
        logical_shape=(3, 8),
        local_shape=(3, 2),
        tp_layout="none",
        dp_shard_axis=1,
        dp_local_start=6,
        dp_local_end=8,
        pre_dp_local_shape=(3, 8),
    )

    assert spec.local_shape == (3, 2)
    assert spec.pre_dp_local_shape == (3, 8)
    assert spec.dp_local_start == 6
    assert spec.dp_local_end == 8
    assert spec.small_gram_side == "left"


def test_matrix_shard_spec_direct_constructor_rejects_column_range_shape_mismatch():
    with pytest.raises(ValueError, match="DP-axis size must match"):
        MatrixShardSpec(
            logical_shape=(3, 8),
            local_shape=(3, 3),
            tp_layout="none",
            dp_shard_axis=1,
            dp_local_start=6,
            dp_local_end=8,
            pre_dp_local_shape=(3, 8),
        )


def test_layerwise_buffer_routing_marks_matrix_optimizer_owned_params():
    module = torch.nn.Module()
    module.weight = _param_with_info()
    module.bias = torch.nn.Parameter(torch.empty(4))

    tag_params_for_buffer_routing(
        [module],
        optimizer_type="adam",
        matrix_optimizer_type="muon",
        matrix_min_dim=2,
        requires_layerwise_layout=True,
    )

    weight_info = get_matrix_optimizer_info(module.weight)
    bias_info = get_matrix_optimizer_info(module.bias)
    assert weight_info.owner == MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION
    assert weight_info.update_family == "muon"
    assert weight_info.requires_layerwise_layout
    assert module.weight.is_managed_by_layer_wise_optimizer
    assert bias_info.owner == MATRIX_OPTIMIZER_OWNER_FALLBACK
    assert not module.bias.is_managed_by_layer_wise_optimizer


def test_layerwise_buffer_routing_preserves_parameterization_role():
    module = torch.nn.Module()
    module.hidden_weight = _param_with_info()
    module.hidden_weight.parameterization_role = "hidden_matrix"
    module.output_weight = _param_with_info()
    module.output_weight.parameterization_role = "output"
    module.output_weight.is_embedding_or_output_parameter = True

    tag_params_for_buffer_routing(
        [module],
        optimizer_type="muon",
        matrix_optimizer_type=None,
        matrix_min_dim=2,
        requires_layerwise_layout=True,
    )

    hidden_info = get_matrix_optimizer_info(module.hidden_weight)
    output_info = get_matrix_optimizer_info(module.output_weight)
    assert module.hidden_weight.parameterization_role == "hidden_matrix"
    assert module.output_weight.parameterization_role == "output"
    assert hidden_info.owner == MATRIX_OPTIMIZER_OWNER_MUON
    assert hidden_info.update_family == "muon"
    assert hidden_info.requires_layerwise_layout
    assert module.hidden_weight.is_managed_by_layer_wise_optimizer
    assert output_info.owner == MATRIX_OPTIMIZER_OWNER_FALLBACK
    assert output_info.update_family == "none"
    assert not module.output_weight.is_managed_by_layer_wise_optimizer


def test_layerwise_muon_routing_synthesizes_matrix_shard_spec_for_plain_2d_param():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.empty(6, 4))

    tag_params_for_buffer_routing(
        [module],
        optimizer_type="muon",
        matrix_optimizer_type=None,
        matrix_min_dim=2,
        requires_layerwise_layout=True,
    )

    info = get_matrix_optimizer_info(module.weight)
    spec = get_matrix_shard_spec(module.weight)
    assert info.owner == MATRIX_OPTIMIZER_OWNER_MUON
    assert info.update_family == "muon"
    assert spec.logical_shape == (6, 4)
    assert spec.local_shape == (6, 4)
    assert spec.tp_layout == "none"
    assert spec.tp_shard_axis is None


def test_matrix_optimizer_split_routes_fallback_to_standard_distopt(monkeypatch):
    import copy
    import types

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    matrix_param = _param_with_info(shape=(4, 3))
    fallback_param = torch.nn.Parameter(torch.ones(3))

    class DummyModel:
        ddp_config = types.SimpleNamespace(use_distributed_optimizer=True)
        buffers = []

        def named_parameters(self):
            return [("matrix", matrix_param), ("fallback", fallback_param)]

    class FakeMatrixFunctionOptimizer:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = param_groups

    class FakeLayerWiseDistributedOptimizer:
        def __init__(
            self,
            chained_optimizers,
            config,
            pg_collection,
            init_state_fn_list=None,
            model_chunks=None,
        ):
            self.chained_optimizers = chained_optimizers
            self.config = config

    class FakeFallbackOptimizer:
        def __init__(self, config):
            self.config = config

    class FakeChainedOptimizer:
        def __init__(self, chained_optimizers):
            self.chained_optimizers = chained_optimizers

    param_group_calls = []
    fallback_calls = []

    def fake_get_param_groups(model_chunks, config, config_overrides=None, param_filter=None):
        trainable = [
            param
            for model_chunk in model_chunks
            for name, param in model_chunk.named_parameters()
            if param.requires_grad
            and (param_filter is None or param_filter(name, param))
        ]
        param_group_calls.append(
            {
                "params": trainable,
                "use_distributed_optimizer": config.use_distributed_optimizer,
                "use_layer_wise_distributed_optimizer": config.use_layer_wise_distributed_optimizer,
                "matrix_optimizer": config.matrix_optimizer,
            }
        )
        return [{"params": trainable}]

    def fake_fallback_optimizer(**kwargs):
        fallback_calls.append(copy.copy(kwargs["config"]))
        return FakeFallbackOptimizer(kwargs["config"])

    monkeypatch.setattr(matrix_optimizer_module, "HAVE_EMERGING_MATRIX_OPTIMIZERS", True)
    monkeypatch.setattr(
        matrix_optimizer_module,
        "configure_model_matrix_updates",
        lambda model_chunks, config: [matrix_param],
    )
    monkeypatch.setattr(matrix_optimizer_module, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(matrix_optimizer_module, "_get_param_groups", fake_get_param_groups)
    monkeypatch.setattr(
        matrix_optimizer_module,
        "_make_matrix_update_rule",
        lambda config, pg_collection: object(),
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "_make_matrix_inplace_update_rule",
        lambda config, pg_collection: object(),
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "MatrixFunctionOptimizer",
        FakeMatrixFunctionOptimizer,
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "LayerWiseDistributedOptimizer",
        FakeLayerWiseDistributedOptimizer,
    )
    monkeypatch.setattr(matrix_optimizer_module, "ChainedOptimizer", FakeChainedOptimizer)
    monkeypatch.setattr(
        matrix_optimizer_module,
        "_copy_matrix_model_refs_to_main_params",
        lambda optimizer: None,
    )
    monkeypatch.setattr(
        matrix_optimizer_module.ProcessGroupCollection,
        "setup_process_groups_for_optimizer",
        staticmethod(
            lambda pg_collection, model_chunks, use_gloo_process_groups=False: {
                "mp_group": object(),
                "intra_dp_cp_group": object(),
                "intra_dp_cp_group_gloo": object(),
                "intra_dist_opt_group": object(),
            }
        ),
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "_distributed_optimizer_instance_id_from_process_groups",
        lambda process_groups: 0,
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "_get_megatron_optimizer_based_on_param_groups",
        fake_fallback_optimizer,
    )

    config = OptimizerConfig(
        optimizer="adam",
        lr=1e-3,
        matrix_optimizer="muon",
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=True,
    )
    pg_collection = types.SimpleNamespace(dp_cp=object())

    optimizer = matrix_optimizer_module.get_megatron_matrix_optimizer(
        config,
        [DummyModel()],
        pg_collection=pg_collection,
    )

    assert isinstance(optimizer, FakeChainedOptimizer)
    assert param_group_calls[0]["params"] == [matrix_param]
    assert param_group_calls[0]["matrix_optimizer"] == "muon"
    assert param_group_calls[1]["params"] == [fallback_param]
    assert param_group_calls[1]["matrix_optimizer"] == "none"
    assert fallback_calls[0].use_distributed_optimizer
    assert not fallback_calls[0].use_layer_wise_distributed_optimizer
    assert config.matrix_optimizer == "muon"
    assert matrix_param.requires_grad
    assert fallback_param.requires_grad


def test_matrix_optimizer_split_rejects_expert_parallel_fallback(monkeypatch):
    import types

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    matrix_param = _param_with_info(shape=(4, 3))
    fallback_param = torch.nn.Parameter(torch.ones(3))

    class DummyModel:
        ddp_config = types.SimpleNamespace(use_distributed_optimizer=True)
        buffers = []

        def named_parameters(self):
            return [("matrix", matrix_param), ("fallback", fallback_param)]

    class FakeMatrixFunctionOptimizer:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = param_groups

    class FakeLayerWiseDistributedOptimizer:
        def __init__(
            self,
            chained_optimizers,
            config,
            pg_collection,
            init_state_fn_list=None,
            model_chunks=None,
        ):
            self.chained_optimizers = chained_optimizers
            self.config = config

    fallback_optimizer_called = False

    def fake_get_param_groups(model_chunks, config, config_overrides=None, param_filter=None):
        trainable = [
            param
            for model_chunk in model_chunks
            for name, param in model_chunk.named_parameters()
            if param.requires_grad
            and (param_filter is None or param_filter(name, param))
        ]
        group = {"params": trainable}
        if len(trainable) == 1 and trainable[0] is fallback_param:
            group["is_expert_parallel"] = True
        return [group]

    def fake_fallback_optimizer(**kwargs):
        nonlocal fallback_optimizer_called
        fallback_optimizer_called = True
        raise AssertionError("expert fallback should be rejected before optimizer construction")

    monkeypatch.setattr(matrix_optimizer_module, "HAVE_EMERGING_MATRIX_OPTIMIZERS", True)
    monkeypatch.setattr(
        matrix_optimizer_module,
        "configure_model_matrix_updates",
        lambda model_chunks, config: [matrix_param],
    )
    monkeypatch.setattr(matrix_optimizer_module, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(matrix_optimizer_module, "_get_param_groups", fake_get_param_groups)
    monkeypatch.setattr(
        matrix_optimizer_module,
        "_make_matrix_update_rule",
        lambda config, pg_collection: object(),
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "_make_matrix_inplace_update_rule",
        lambda config, pg_collection: object(),
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "MatrixFunctionOptimizer",
        FakeMatrixFunctionOptimizer,
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "LayerWiseDistributedOptimizer",
        FakeLayerWiseDistributedOptimizer,
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "_copy_matrix_model_refs_to_main_params",
        lambda optimizer: None,
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "_get_megatron_optimizer_based_on_param_groups",
        fake_fallback_optimizer,
    )

    config = OptimizerConfig(
        optimizer="adam",
        lr=1e-3,
        matrix_optimizer="muon",
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=True,
    )
    pg_collection = types.SimpleNamespace(dp_cp=object())

    with pytest.raises(RuntimeError, match="expert-parallel fallback"):
        matrix_optimizer_module.get_megatron_matrix_optimizer(
            config,
            [DummyModel()],
            pg_collection=pg_collection,
        )
    assert not fallback_optimizer_called


def test_matrix_function_optimizer_state_dict_preserves_refresh_cadence():
    param = _param_with_info()

    def update_rule(grad, feature_gram, grad_gram, model_param):
        return -grad

    opt = MatrixFunctionOptimizer([param], lr=0.1, update_rule=update_rule)
    opt._matrix_step = 7

    reloaded = MatrixFunctionOptimizer([param], lr=0.1, update_rule=update_rule)
    reloaded.load_state_dict(opt.state_dict())

    assert reloaded._matrix_step == 7


def test_matrix_optimizer_normalization_matches_yaml_and_cli_path():
    args = SimpleNamespace(
        matrix_optimizer="muon",
        optimizer="adam",
        use_distributed_optimizer=True,
        use_layer_wise_param_layout=True,
        use_torch_fsdp2=False,
        use_megatron_fsdp=False,
        ckpt_format="torch_dist",
    )

    normalize_matrix_and_emerging_optimizer_args(args)

    assert args.use_layer_wise_distributed_optimizer
    assert not args.use_distributed_optimizer


def test_fp8_dequant_source_fails_closed_in_native_collector():
    param = _param_with_info()
    recipe = _recipe()
    recipe.activation_dtype = "fp8_dequant"
    configure_matrix_update_param(param, recipe=recipe)

    with pytest.raises(NotImplementedError, match="fp8_dequant"):
        maybe_accumulate_feature_gram(param, torch.ones(2, 3))


def test_matrix_function_optimizer_consumes_feature_gram_and_preserves_buffer_for_cache_reuse():
    param = _param_with_info()
    param.data = torch.zeros_like(param.data)
    param.main_grad = torch.ones_like(param.data)
    recipe = _recipe(normalization=MatrixPreconditionerNormalization.MEAN)
    configure_matrix_update_param(param, recipe=recipe)
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    maybe_accumulate_feature_gram(param, x)

    seen = {}

    def update_rule(grad, feature_gram, grad_gram, model_param):
        seen["feature_gram"] = feature_gram.clone()
        seen["model_param"] = model_param
        return -grad

    opt = MatrixFunctionOptimizer([param], lr=0.1, update_rule=update_rule)
    opt.step()

    torch.testing.assert_close(param, torch.full_like(param, -0.1))
    torch.testing.assert_close(seen["feature_gram"], x.t().matmul(x) / x.shape[0])
    assert seen["model_param"] is param
    torch.testing.assert_close(param.main_grad_feature_gram, x.t().matmul(x))
    torch.testing.assert_close(param.main_grad_feature_count, torch.tensor(2.0, dtype=torch.float64))


def test_matrix_function_optimizer_consumes_feature_and_grad_grams():
    param = _param_with_info()
    param.data = torch.zeros_like(param.data)
    param.main_grad = torch.ones_like(param.data)
    input_recipe = _recipe(normalization=MatrixPreconditionerNormalization.MEAN)
    output_recipe = _output_recipe(normalization=MatrixPreconditionerNormalization.MEAN)
    configure_matrix_update_param(param, recipe=input_recipe, output_recipe=output_recipe)
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    dy = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    maybe_accumulate_feature_gram(param, x)
    maybe_accumulate_grad_gram(param, dy)

    seen = {}

    def update_rule(grad, feature_gram, grad_gram, model_param):
        seen["feature_gram"] = feature_gram.clone()
        seen["grad_gram"] = grad_gram.clone()
        seen["model_param"] = model_param
        return -grad

    opt = MatrixFunctionOptimizer([param], lr=0.1, update_rule=update_rule)
    opt.step()

    torch.testing.assert_close(seen["feature_gram"], x.t().matmul(x) / x.shape[0])
    torch.testing.assert_close(seen["grad_gram"], dy.t().matmul(dy) / dy.shape[0])
    assert seen["model_param"] is param


def test_matrix_function_optimizer_respects_feature_gram_refresh_interval():
    param = _param_with_info()
    param.data = torch.zeros_like(param.data)
    param.main_grad = torch.ones_like(param.data)
    recipe = _recipe(normalization=MatrixPreconditionerNormalization.MEAN, refresh_interval=2)
    configure_matrix_update_param(param, recipe=recipe)
    seen = []

    def update_rule(grad, feature_gram, grad_gram, model_param):
        seen.append(feature_gram.clone())
        return -grad

    opt = MatrixFunctionOptimizer([param], lr=0.1, update_rule=update_rule)
    x0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    x1 = torch.tensor([[3.0, 0.0, 0.0]])
    x2 = torch.tensor([[0.0, 0.0, 4.0]])

    opt.zero_grad()
    maybe_accumulate_feature_gram(param, x0)
    opt.step()
    first_generation = param._feature_gram_generation

    opt.zero_grad()
    assert not param._feature_gram_active
    maybe_accumulate_feature_gram(param, x1)
    opt.step()

    opt.zero_grad()
    assert param._feature_gram_active
    maybe_accumulate_feature_gram(param, x2)
    opt.step()

    torch.testing.assert_close(seen[0], x0.t().matmul(x0) / x0.shape[0])
    torch.testing.assert_close(seen[1], seen[0])
    torch.testing.assert_close(seen[2], x2.t().matmul(x2) / x2.shape[0])
    assert param._feature_gram_generation == first_generation + 1


def test_matrix_function_optimizer_respects_grad_gram_refresh_interval():
    param = _param_with_info()
    param.data = torch.zeros_like(param.data)
    param.main_grad = torch.ones_like(param.data)
    recipe = _output_recipe(
        normalization=MatrixPreconditionerNormalization.MEAN, refresh_interval=2
    )
    configure_matrix_update_param(param, output_recipe=recipe)
    seen = []

    def update_rule(grad, feature_gram, grad_gram, model_param):
        seen.append(grad_gram.clone())
        return -grad

    opt = MatrixFunctionOptimizer([param], lr=0.1, update_rule=update_rule)
    dy0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    dy1 = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    dy2 = torch.tensor([[0.0, 0.0, 4.0, 0.0]])

    opt.zero_grad()
    maybe_accumulate_grad_gram(param, dy0)
    opt.step()
    first_generation = param._grad_gram_generation

    opt.zero_grad()
    assert not param._grad_gram_active
    maybe_accumulate_grad_gram(param, dy1)
    opt.step()

    opt.zero_grad()
    assert param._grad_gram_active
    maybe_accumulate_grad_gram(param, dy2)
    opt.step()

    torch.testing.assert_close(seen[0], dy0.t().matmul(dy0) / dy0.shape[0])
    torch.testing.assert_close(seen[1], seen[0])
    torch.testing.assert_close(seen[2], dy2.t().matmul(dy2) / dy2.shape[0])
    assert param._grad_gram_generation == first_generation + 1


def test_matrix_function_optimizer_does_not_refinalize_cached_grams(monkeypatch):
    param = _param_with_info()
    param.data = torch.zeros_like(param.data)
    param.main_grad = torch.ones_like(param.data)
    input_recipe = _recipe(
        normalization=MatrixPreconditionerNormalization.MEAN, refresh_interval=2
    )
    output_recipe = _output_recipe(
        normalization=MatrixPreconditionerNormalization.MEAN, refresh_interval=2
    )
    configure_matrix_update_param(param, recipe=input_recipe, output_recipe=output_recipe)
    set_feature_gram_finalization_required([param], required=True)
    set_grad_gram_finalization_required([param], required=True)
    reduce_calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        reduce_calls.append((tensor, op, group))
        return tensor

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    def update_rule(grad, feature_gram, grad_gram, model_param):
        return -grad

    process_group = object()
    opt = MatrixFunctionOptimizer(
        [param],
        lr=0.1,
        update_rule=update_rule,
        feature_gram_process_groups=(process_group,),
    )
    x0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    dy0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    x1 = torch.tensor([[3.0, 0.0, 0.0]])
    dy1 = torch.tensor([[3.0, 0.0, 0.0, 0.0]])

    opt.zero_grad()
    maybe_accumulate_feature_gram(param, x0)
    maybe_accumulate_grad_gram(param, dy0)
    opt.step()
    finalized_call_count = len(reduce_calls)
    assert finalized_call_count == 4

    opt.zero_grad()
    assert not param._feature_gram_active
    assert not param._grad_gram_active
    maybe_accumulate_feature_gram(param, x1)
    maybe_accumulate_grad_gram(param, dy1)
    opt.step()

    assert len(reduce_calls) == finalized_call_count


def test_matrix_update_rule_caches_factorization_by_feature_gram_generation(monkeypatch):
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info()
    param._feature_gram_generation = 7
    grad = torch.ones_like(param.data)
    feature_gram = torch.tensor([2.0, 3.0, 4.0])
    config = OptimizerConfig(
        matrix_optimizer="sgd", matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="diag",
        matrix_input_preconditioner_refresh_interval=2,
        matrix_input_preconditioner_ridge=1.0,
    )
    calls = []
    original_factorize = matrix_optimizer_module.factorize_feature_gram

    def counted_factorize(*args, **kwargs):
        calls.append(1)
        return original_factorize(*args, **kwargs)

    monkeypatch.setattr(matrix_optimizer_module, "factorize_feature_gram", counted_factorize)
    rule = matrix_optimizer_module._make_matrix_update_rule(config, SimpleNamespace(tp=None))

    first = rule(grad, feature_gram, None, param)
    second = rule(grad, feature_gram, None, param)
    param._feature_gram_generation += 1
    third = rule(grad, feature_gram, None, param)

    assert len(calls) == 2
    torch.testing.assert_close(first, -grad / (feature_gram + 1.0))
    torch.testing.assert_close(second, first)
    torch.testing.assert_close(third, first)


def test_matrix_update_rule_supports_plain_sgd_without_input_preconditioner():
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info(tp_layout="none")
    grad = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    config = OptimizerConfig(matrix_optimizer="sgd")

    rule = matrix_optimizer_module._make_matrix_update_rule(config, SimpleNamespace(tp=None))

    torch.testing.assert_close(rule(grad, None, None, param), -grad)


def test_matrix_update_rule_supports_output_grad_gram_preconditioner():
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info(tp_layout="none")
    grad = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    grad_gram = torch.tensor([1.0, 3.0, 7.0, 15.0])
    config = OptimizerConfig(
        matrix_optimizer="sgd",
        matrix_output_preconditioner="grad_gram",
        matrix_output_preconditioner_approximation="diag",
        matrix_output_preconditioner_ridge=1.0,
    )

    rule = matrix_optimizer_module._make_matrix_update_rule(config, SimpleNamespace(tp=None))

    expected = -grad / (grad_gram + 1.0).unsqueeze(1)
    torch.testing.assert_close(rule(grad, None, grad_gram, param), expected)


def test_matrix_update_rule_full_output_grad_gram_tp1_matches_unsharded():
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    column_parallel_param = _param_with_info(
        tp_layout="column_parallel", shape=(4, 3), logical_shape=(4, 3)
    )
    unsharded_param = _param_with_info(tp_layout="none", shape=(4, 3), logical_shape=(4, 3))
    grad = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    grad_gram = torch.tensor(
        [
            [5.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 6.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ]
    )
    config = OptimizerConfig(
        matrix_optimizer="sgd",
        matrix_output_preconditioner="grad_gram",
        matrix_output_preconditioner_approximation="full",
        matrix_output_preconditioner_ridge=0.5,
    )

    rule = matrix_optimizer_module._make_matrix_update_rule(config, SimpleNamespace(tp=None))

    expected = -torch.linalg.solve(grad_gram + 0.5 * torch.eye(4), grad)
    torch.testing.assert_close(rule(grad, None, grad_gram, column_parallel_param), expected)
    torch.testing.assert_close(
        rule(grad, None, grad_gram, column_parallel_param),
        rule(grad, None, grad_gram, unsharded_param),
    )


def test_matrix_update_rule_block_diag_output_grad_gram_solves_real_coordinates_only():
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info(tp_layout="none", shape=(3, 2), logical_shape=(3, 2))
    grad = torch.arange(1, 7, dtype=torch.float32).reshape(3, 2)
    grad_gram = torch.tensor(
        [
            [[4.0, 1.0], [1.0, 3.0]],
            [[9.0, 0.0], [0.0, 0.0]],
        ]
    )
    config = OptimizerConfig(
        matrix_optimizer="sgd",
        matrix_output_preconditioner="grad_gram",
        matrix_output_preconditioner_approximation="block_diag",
        matrix_output_preconditioner_block_size=2,
        matrix_output_preconditioner_ridge=0.25,
    )

    rule = matrix_optimizer_module._make_matrix_update_rule(config, SimpleNamespace(tp=None))

    ridge_eye = 0.25 * torch.eye(2)
    first_block = -torch.linalg.solve(grad_gram[0] + ridge_eye, grad[:2])
    last_rhs = torch.stack([grad[2], torch.zeros_like(grad[2])])
    last_block = -torch.linalg.solve(grad_gram[1] + ridge_eye, last_rhs)[:1]
    expected = torch.cat([first_block, last_block], dim=0)
    torch.testing.assert_close(rule(grad, None, grad_gram, param), expected)


def test_matrix_update_rule_supports_two_sided_diag_preconditioning():
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info(tp_layout="none")
    grad = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    feature_gram = torch.tensor([1.0, 4.0, 9.0])
    grad_gram = torch.tensor([2.0, 3.0, 4.0, 5.0])
    config = OptimizerConfig(
        matrix_optimizer="sgd",
        matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="diag",
        matrix_input_preconditioner_ridge=0.5,
        matrix_output_preconditioner="grad_gram",
        matrix_output_preconditioner_approximation="diag",
        matrix_output_preconditioner_ridge=1.0,
    )

    rule = matrix_optimizer_module._make_matrix_update_rule(config, SimpleNamespace(tp=None))

    expected = -grad / (grad_gram + 1.0).unsqueeze(1) / (feature_gram + 0.5).unsqueeze(0)
    torch.testing.assert_close(rule(grad, feature_gram, grad_gram, param), expected)


def test_matrix_update_rule_supports_plain_muon_without_input_preconditioner(monkeypatch):
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info(tp_layout="none")
    grad = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    config = OptimizerConfig(
        matrix_optimizer="muon",
        muon_scale_mode="unit_rms_norm",
        muon_extra_scale_factor=1.0,
    )

    def fake_tp_allgather(matrix, orthogonalize, tp_layout, group):
        return matrix

    monkeypatch.setattr(
        matrix_optimizer_module, "tp_allgather_logical_matrix_update", fake_tp_allgather
    )
    rule = matrix_optimizer_module._make_matrix_update_rule(config, SimpleNamespace(tp=None))

    expected = -grad * (grad.size(-2) / grad.size(-1)) ** 0.5
    torch.testing.assert_close(rule(grad, None, None, param), expected)


def test_matrix_update_rule_small_gram_muon_honors_ns_config(monkeypatch):
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info(tp_layout="column_parallel", shape=(2, 2))
    grad = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    config = OptimizerConfig(
        matrix_optimizer="muon",
        matrix_tp_update_mode="small_gram_ns",
        muon_num_ns_steps=7,
        muon_coefficient_type="polar_express",
        muon_scale_mode="unit_rms_norm",
    )
    calls = []

    def fake_small_gram(matrix, **kwargs):
        calls.append((matrix, kwargs))
        return matrix + 1.0

    monkeypatch.setattr(
        matrix_optimizer_module,
        "tp_small_gram_newton_schulz_allreduce",
        fake_small_gram,
    )
    rule = matrix_optimizer_module._make_matrix_update_rule(config, SimpleNamespace(tp=None))

    update = rule(grad, None, None, param)

    torch.testing.assert_close(update, -(grad + 1.0))
    assert len(calls) == 1
    seen_matrix, kwargs = calls[0]
    torch.testing.assert_close(seen_matrix, grad)
    assert kwargs["tp_layout"] == "column_parallel"
    assert kwargs["group"] is None
    assert kwargs["steps"] == 7
    assert kwargs["coefficient_type"] == "polar_express"
    assert not kwargs["use_syrk"]


def _install_fake_initialized_all_reduce(monkeypatch, *, world_size):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: tensor.mul_(world_size),
    )


def test_matrix_fsdp_small_gram_muon_row_shard_matches_full_reference(monkeypatch):
    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    _install_fake_initialized_all_reduce(monkeypatch, world_size=2)
    local = torch.tensor([[1.0, 0.25], [0.5, 2.0]])
    full = torch.cat([local, local], dim=0)
    spec = matrix_shard_spec_with_dp_axis(
        MatrixShardSpec(logical_shape=(4, 2), local_shape=(4, 2), tp_layout="none"),
        dp_shard_axis=0,
        dp_local_start=0,
        dp_local_end=2,
    )
    config = OptimizerConfig(
        matrix_optimizer="muon",
        muon_num_ns_steps=2,
        muon_coefficient_type="simple",
    )

    update = matrix_optimizer_module._fsdp_small_gram_newton_schulz_allreduce(
        local,
        spec,
        group=object(),
        config=config,
    )
    ref = matrix_optimizer_module.newton_schulz_orthogonalize(
        full,
        steps=2,
        coefficient_type="simple",
    ).chunk(2, dim=0)[0]

    torch.testing.assert_close(update, ref)


def test_matrix_fsdp_small_gram_muon_column_shard_matches_full_reference(monkeypatch):
    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    _install_fake_initialized_all_reduce(monkeypatch, world_size=2)
    local = torch.tensor([[1.0, 0.25], [0.5, 2.0]])
    full = torch.cat([local, local], dim=1)
    spec = matrix_shard_spec_with_dp_axis(
        MatrixShardSpec(logical_shape=(2, 4), local_shape=(2, 4), tp_layout="none"),
        dp_shard_axis=1,
        dp_local_start=0,
        dp_local_end=2,
    )
    config = OptimizerConfig(
        matrix_optimizer="muon",
        muon_num_ns_steps=2,
        muon_coefficient_type="simple",
    )

    update = matrix_optimizer_module._fsdp_small_gram_newton_schulz_allreduce(
        local,
        spec,
        group=object(),
        config=config,
    )
    ref = matrix_optimizer_module.newton_schulz_orthogonalize(
        full,
        steps=2,
        coefficient_type="simple",
    ).chunk(2, dim=1)[0]

    torch.testing.assert_close(update, ref)


def test_matrix_update_rule_uses_fsdp_small_gram_and_logical_scale(monkeypatch):
    import contextlib
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    param = _param_with_info(tp_layout="none", shape=(2, 2), logical_shape=(4, 2))
    set_matrix_shard_spec(
        param,
        matrix_shard_spec_with_dp_axis(
            MatrixShardSpec(logical_shape=(4, 2), local_shape=(4, 2), tp_layout="none"),
            dp_shard_axis=0,
            dp_local_start=0,
            dp_local_end=2,
        ),
    )
    grad = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    config = OptimizerConfig(
        matrix_optimizer="muon",
        muon_scale_mode="unit_rms_norm",
        muon_extra_scale_factor=1.0,
    )
    dp_group = object()
    calls = []

    def fake_fsdp_small_gram(matrix, spec, *, group, config):
        calls.append((matrix, spec, group, config))
        return matrix + 1.0

    monkeypatch.setattr(
        matrix_optimizer_module,
        "_fsdp_small_gram_newton_schulz_allreduce",
        fake_fsdp_small_gram,
    )
    monkeypatch.setattr(
        matrix_optimizer_module,
        "fp32_matmul_precision",
        lambda precision: contextlib.nullcontext(),
        raising=False,
    )
    rule = matrix_optimizer_module._make_matrix_update_rule(
        config,
        SimpleNamespace(tp=None, dp_cp=dp_group),
    )

    update = rule(grad, None, None, param)

    torch.testing.assert_close(update, -(grad + 1.0) * (4 / 2) ** 0.5)
    assert len(calls) == 1
    seen_matrix, seen_spec, seen_group, seen_config = calls[0]
    torch.testing.assert_close(seen_matrix, grad)
    assert seen_spec == get_matrix_shard_spec(param)
    assert seen_group is dp_group
    assert seen_config is config


def test_matrix_function_optimizer_state_dict_does_not_serialize_update_rule():
    param = _param_with_info()

    def update_rule(grad, feature_gram, grad_gram, model_param):
        return -grad

    opt = MatrixFunctionOptimizer([param], lr=0.1, update_rule=update_rule)

    state_dict = opt.state_dict()

    assert "update_rule" not in state_dict["param_groups"][0]


def test_matrix_function_optimizer_applies_decoupled_weight_decay():
    param = _param_with_info()
    param.data.fill_(1.0)
    param.main_grad = torch.zeros_like(param.data)
    configure_matrix_update_param(param, recipe=_recipe())
    maybe_accumulate_feature_gram(param, torch.ones(2, 3))

    def update_rule(grad, feature_gram, grad_gram, model_param):
        return torch.zeros_like(grad)

    opt = MatrixFunctionOptimizer(
        [param],
        lr=0.1,
        update_rule=update_rule,
        weight_decay=0.2,
        decoupled_weight_decay=True,
    )
    opt.step()

    torch.testing.assert_close(param, torch.full_like(param, 0.98))


def test_matrix_function_optimizer_uses_inplace_update_rule_when_available():
    param = _param_with_info()
    param.data.fill_(1.0)
    param.main_grad = torch.tensor(
        [[2.0, 4.0, 6.0], [1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [3.0, 6.0, 9.0]]
    )
    recipe = _recipe(approximation=MatrixPreconditionerApproximation.DIAG)
    configure_matrix_update_param(param, recipe=recipe)
    param.main_grad_feature_gram.copy_(torch.tensor([1.0, 3.0, 5.0]))
    param.main_grad_feature_count.fill_(1.0)
    calls = []

    def update_rule(grad, feature_gram, grad_gram, model_param):
        raise AssertionError("generic update_rule should not run")

    def inplace_update_rule(
        update_param,
        grad,
        feature_gram,
        grad_gram,
        model_param,
        lr,
        weight_decay,
        decoupled_weight_decay,
    ):
        calls.append(1)
        assert grad_gram is None
        update_param.mul_(1.0 - lr * weight_decay)
        update_param.add_(grad / (feature_gram + 1.0), alpha=-lr)
        return True

    opt = MatrixFunctionOptimizer(
        [param],
        lr=0.1,
        update_rule=update_rule,
        inplace_update_rule=inplace_update_rule,
        weight_decay=0.2,
        decoupled_weight_decay=True,
    )
    opt.step()

    expected = torch.ones_like(param) * 0.98 - 0.1 * param.main_grad / torch.tensor([2.0, 4.0, 6.0])
    torch.testing.assert_close(param, expected)
    assert calls == [1]


def test_matrix_inplace_update_rule_uses_two_sided_diag_sgd():
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info()
    param.data.fill_(1.0)
    grad = torch.tensor(
        [[2.0, 4.0, 6.0], [1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [3.0, 6.0, 9.0]]
    )
    feature_gram = torch.tensor([1.0, 3.0, 5.0])
    grad_gram = torch.tensor([1.0, 4.0, 9.0, 16.0])
    config = OptimizerConfig(
        matrix_optimizer="sgd",
        matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="diag",
        matrix_input_preconditioner_ridge=0.5,
        matrix_output_preconditioner="grad_gram",
        matrix_output_preconditioner_approximation="diag",
        matrix_output_preconditioner_ridge=1.0,
    )
    rule = matrix_optimizer_module._make_matrix_inplace_update_rule(
        config, SimpleNamespace(tp=None)
    )

    with torch.no_grad():
        assert rule(param, grad, feature_gram, grad_gram, param, 0.1, 0.2, True)

    expected_direction = grad / (grad_gram + 1.0)[:, None] / (feature_gram + 0.5)
    expected = torch.ones_like(param) * 0.98 - 0.1 * expected_direction
    torch.testing.assert_close(param, expected)


def test_matrix_inplace_update_rule_routes_input_only_diag_sgd_to_right_kernel(monkeypatch):
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info()
    grad = torch.ones_like(param.data)
    feature_gram = torch.tensor([2.0, 3.0, 4.0])
    config = OptimizerConfig(
        matrix_optimizer="sgd",
        matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="diag",
        matrix_input_preconditioner_ridge=1.0,
    )
    calls = []

    def fake_apply(param_arg, grad_arg, diag_feature_gram, **kwargs):
        calls.append((param_arg, grad_arg, diag_feature_gram, kwargs))
        return True

    monkeypatch.setattr(
        matrix_optimizer_module, "apply_diag_right_preconditioned_update_", fake_apply
    )
    rule = matrix_optimizer_module._make_matrix_inplace_update_rule(
        config, SimpleNamespace(tp=None)
    )

    assert rule(param, grad, feature_gram, None, param, 0.1, 0.2, True)
    assert len(calls) == 1
    seen_param, seen_grad, seen_feature_gram, kwargs = calls[0]
    assert seen_param is param
    assert seen_grad is grad
    assert seen_feature_gram is feature_gram
    assert kwargs["lr"] == 0.1
    assert kwargs["weight_decay"] == 0.2
    assert kwargs["decoupled_weight_decay"] is True
    assert kwargs["ridge"] == 1.0


def test_matrix_inplace_update_rule_uses_diag_muon_for_local_matrix(monkeypatch):
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info()
    grad = torch.ones_like(param.data)
    feature_gram = torch.tensor([2.0, 3.0, 4.0])
    config = OptimizerConfig(
        matrix_optimizer="muon", matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="diag",
        matrix_input_preconditioner_ridge=1.0,
        muon_num_ns_steps=2,
        muon_coefficient_type="simple",
        muon_scale_mode="unit_rms_norm",
        muon_extra_scale_factor=0.5,
        muon_fp32_matmul_prec="highest",
    )
    calls = []

    def fake_apply(param, grad, diag_feature_gram, **kwargs):
        calls.append((param, grad, diag_feature_gram, kwargs))
        return param

    monkeypatch.setattr(
        matrix_optimizer_module, "apply_diag_newton_muon_update_", fake_apply
    )
    rule = matrix_optimizer_module._make_matrix_inplace_update_rule(
        config, SimpleNamespace(tp=None)
    )

    assert rule(param, grad, feature_gram, None, param, 0.1, 0.2, True)
    assert len(calls) == 1
    _, seen_grad, seen_feature_gram, kwargs = calls[0]
    assert seen_grad is grad
    assert seen_feature_gram is feature_gram
    assert kwargs["lr"] == 0.1
    assert kwargs["ridge"] == 1.0
    assert kwargs["num_ns_steps"] == 2
    assert kwargs["coefficient_type"] == "simple"
    assert kwargs["scale_mode"] == "unit_rms_norm"
    assert kwargs["extra_scale_factor"] == 0.5
    assert kwargs["weight_decay"] == 0.2
    assert kwargs["decoupled_weight_decay"]
    assert kwargs["fp32_matmul_prec"] == "highest"
    assert not kwargs["use_syrk"]


def test_row_parallel_nonexact_feature_scope_gets_approximation_label():
    param = _param_with_info(tp_layout="row_parallel")
    recipe = _recipe(approximation=MatrixPreconditionerApproximation.DIAG)
    recipe.scope = input_preconditioner_scope_for(
        recipe.approximation, "row_parallel", is_feature_axis_sharded=True
    )
    configure_matrix_update_param(param, recipe=recipe)

    plan = default_matrix_apply_plan(
        param, tp_update_mode=TPUpdateMode.TP_ALLGATHER_LOGICAL_MATRIX
    )

    assert plan.requires_full_logical_gradient
    assert plan.approximation_label == "tp_local_block_diag_feature_gram"


def test_column_parallel_nonexact_grad_scope_gets_approximation_label():
    param = _param_with_info(tp_layout="column_parallel")
    recipe = _output_recipe(approximation=MatrixPreconditionerApproximation.DIAG)
    recipe.scope = output_preconditioner_scope_for(
        recipe.approximation, "column_parallel", is_output_axis_sharded=True
    )
    configure_matrix_update_param(param, output_recipe=recipe)

    plan = default_matrix_apply_plan(
        param, tp_update_mode=TPUpdateMode.TP_ALLGATHER_LOGICAL_MATRIX
    )

    assert plan.requires_full_logical_gradient
    assert plan.approximation_label == "tp_local_block_diag_grad_gram"


def test_matrix_function_optimizer_fails_when_grad_has_no_feature_rows():
    param = _param_with_info()
    param.main_grad = torch.ones_like(param.data)
    configure_matrix_update_param(param, recipe=_recipe())

    def update_rule(grad, feature_gram, grad_gram, model_param):
        return -grad

    opt = MatrixFunctionOptimizer([param], lr=0.1, update_rule=update_rule)

    with pytest.raises(RuntimeError, match="zero collected feature rows"):
        opt.step()


def test_feature_gram_finalization_requires_explicit_process_groups():
    param = _param_with_info()
    configure_matrix_update_param(param, recipe=_recipe())
    maybe_accumulate_feature_gram(param, torch.ones(2, 3))
    set_feature_gram_finalization_required([param], required=True)

    with pytest.raises(RuntimeError, match="explicit process_groups"):
        finalize_feature_gram_buffers([param])


def test_grad_gram_finalization_requires_explicit_process_groups():
    param = _param_with_info()
    configure_matrix_update_param(param, output_recipe=_output_recipe())
    maybe_accumulate_grad_gram(param, torch.ones(2, 4))
    set_grad_gram_finalization_required([param], required=True)

    with pytest.raises(RuntimeError, match="explicit process_groups"):
        finalize_grad_gram_buffers([param])


def test_feature_gram_finalization_skips_cached_finalized_buffer_without_groups():
    param = _param_with_info()
    configure_matrix_update_param(param, recipe=_recipe())
    maybe_accumulate_feature_gram(param, torch.ones(2, 3))
    set_feature_gram_finalization_required([param], required=True)
    param._feature_gram_finalized = True

    finalize_feature_gram_buffers([param])


def test_grad_gram_finalization_skips_cached_finalized_buffer_without_groups():
    param = _param_with_info()
    configure_matrix_update_param(param, output_recipe=_output_recipe())
    maybe_accumulate_grad_gram(param, torch.ones(2, 4))
    set_grad_gram_finalization_required([param], required=True)
    param._grad_gram_finalized = True

    finalize_grad_gram_buffers([param])


def test_optimizer_config_rejects_standard_distopt_with_matrix_optimizer():
    with pytest.raises(ValueError, match="standard DistributedOptimizer"):
        OptimizerConfig(matrix_optimizer="muon", use_distributed_optimizer=True)


def test_optimizer_config_accepts_block_diag_feature_gram():
    config = OptimizerConfig(
        matrix_optimizer="sgd", matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="block_diag",
        matrix_input_preconditioner_block_size=2,
    )

    assert config.matrix_input_preconditioner_approximation == "block_diag"


def test_optimizer_config_accepts_block_diag_grad_gram():
    config = OptimizerConfig(
        matrix_optimizer="sgd",
        matrix_output_preconditioner="grad_gram",
        matrix_output_preconditioner_approximation="block_diag",
        matrix_output_preconditioner_block_size=2,
    )

    assert config.matrix_output_preconditioner_approximation == "block_diag"


def test_optimizer_config_rejects_output_preconditioner_options_when_disabled():
    with pytest.raises(ValueError, match="matrix_output_preconditioner-specific options"):
        OptimizerConfig(
            matrix_optimizer="sgd",
            matrix_output_preconditioner_approximation="block_diag",
        )


def test_optimizer_config_normalizes_input_preconditioner_accumulation_dtype_strings():
    config = OptimizerConfig(
        matrix_optimizer="sgd", matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_accumulation_dtype="fp32",
    )
    assert config.matrix_input_preconditioner_accumulation_dtype is torch.float32

    config = OptimizerConfig(
        matrix_optimizer="sgd", matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_accumulation_dtype="bf16",
    )
    assert config.matrix_input_preconditioner_accumulation_dtype is torch.bfloat16

    with pytest.raises(ValueError, match="matrix_input_preconditioner_accumulation_dtype"):
        OptimizerConfig(
            matrix_optimizer="sgd", matrix_input_preconditioner="feature_gram",
            matrix_input_preconditioner_accumulation_dtype="int32",
        )


def test_optimizer_config_rejects_unimplemented_active_matrix_options():
    OptimizerConfig(
        matrix_optimizer="muon",
        matrix_input_preconditioner="feature_gram",
        matrix_input_preconditioner_approximation="block_diag",
    )
    OptimizerConfig(matrix_optimizer="muon", matrix_input_preconditioner="feature_gram", matrix_input_preconditioner_activation_dtype="fp8_dequant")
    with pytest.raises(ValueError, match="augmented_feature_sum"):
        OptimizerConfig(matrix_optimizer="muon", matrix_input_preconditioner="feature_gram", matrix_bias_mode="augmented_feature_sum")
    OptimizerConfig(matrix_optimizer="muon", matrix_input_preconditioner="feature_gram", matrix_input_preconditioner_refresh_interval=2)
    with pytest.raises(ValueError, match="ema_beta"):
        OptimizerConfig(matrix_optimizer="muon", matrix_input_preconditioner="feature_gram", matrix_input_preconditioner_ema_beta=0.9)
    with pytest.raises(ValueError, match="matrix_output_preconditioner_ema_beta"):
        OptimizerConfig(matrix_optimizer="muon", matrix_output_preconditioner="grad_gram", matrix_output_preconditioner_ema_beta=0.9)


def test_optimizer_config_maps_deprecated_muon_ns_coefficients_alias():
    config = OptimizerConfig(matrix_optimizer="muon", muon_ns_coefficients="simple")

    assert config.muon_ns_coefficients == "simple"
    assert config.muon_coefficient_type == "simple"

    with pytest.raises(ValueError, match="deprecated alias"):
        OptimizerConfig(
            matrix_optimizer="muon",
            muon_coefficient_type="simple",
            muon_ns_coefficients="polar_express",
        )


def test_matrix_update_family_rejects_optimizer_variants():
    assert matrix_update_family_from_optimizer_name(None) == "none"
    assert matrix_update_family_from_optimizer_name("sgd") == "sgd"
    assert matrix_update_family_from_optimizer_name("muon") == "muon"

    with pytest.raises(ValueError, match="stable matrix update_family"):
        matrix_update_family_from_optimizer_name("adaptive_muon")

    with pytest.raises(ValueError, match="stable matrix update_family"):
        matrix_update_family_from_optimizer_name("newton_muon")


def test_matrix_optimizer_info_owner_family_invariants():
    param = torch.nn.Parameter(torch.empty(2, 2))

    set_matrix_optimizer_info(
        param,
        owner=MATRIX_OPTIMIZER_OWNER_MUON,
        update_family="muon",
    )
    state_spec = get_matrix_optimizer_state_spec(param)
    assert state_spec.same_shard_state_names == ("master_param", "momentum_buffer")

    with pytest.raises(ValueError, match="owner 'muon' requires update_family='muon'"):
        set_matrix_optimizer_info(
            param,
            owner=MATRIX_OPTIMIZER_OWNER_MUON,
            update_family="sgd",
        )

    with pytest.raises(ValueError, match="owner 'fallback' requires update_family='none'"):
        set_matrix_optimizer_info(
            param,
            owner=MATRIX_OPTIMIZER_OWNER_FALLBACK,
            update_family="muon",
        )

    with pytest.raises(ValueError, match="owner 'matrix_function' requires an active"):
        set_matrix_optimizer_info(
            param,
            owner=MATRIX_OPTIMIZER_OWNER_MATRIX_FUNCTION,
            update_family="none",
        )


def test_configure_model_matrix_updates_rejects_native_fp8_dequant_source():
    module = torch.nn.Module()
    module.weight = _param_with_info()
    config = OptimizerConfig(
        matrix_optimizer="muon", matrix_input_preconditioner="feature_gram", matrix_input_preconditioner_activation_dtype="fp8_dequant"
    )

    with pytest.raises(RuntimeError, match="native linears cannot dequantize FP8"):
        configure_model_matrix_updates([module], config)


def test_configure_model_matrix_updates_rejects_row_parallel_full_gram():
    module = torch.nn.Module()
    module.weight = _param_with_info(
        tp_layout="row_parallel", shape=(4, 2), logical_shape=(4, 4)
    )
    config = OptimizerConfig(matrix_optimizer="muon", matrix_input_preconditioner="feature_gram", matrix_input_preconditioner_approximation="full")

    with pytest.raises(RuntimeError, match="row-parallel"):
        configure_model_matrix_updates([module], config)


def test_configure_model_matrix_updates_allows_unsharded_column_parallel_full_grad_gram():
    module = torch.nn.Module()
    module.weight = _param_with_info(tp_layout="column_parallel")
    config = OptimizerConfig(
        matrix_optimizer="muon",
        matrix_output_preconditioner="grad_gram",
        matrix_output_preconditioner_approximation="full",
    )

    configure_model_matrix_updates([module], config)


def test_configure_model_matrix_updates_rejects_sharded_column_parallel_full_grad_gram():
    module = torch.nn.Module()
    module.weight = _param_with_info(
        tp_layout="column_parallel", shape=(2, 3), logical_shape=(4, 3)
    )
    config = OptimizerConfig(
        matrix_optimizer="muon",
        matrix_output_preconditioner="grad_gram",
        matrix_output_preconditioner_approximation="full",
    )

    with pytest.raises(RuntimeError, match="column-parallel"):
        configure_model_matrix_updates([module], config)


def test_configure_matrix_update_rejects_padded_block_diag_grad_gram_without_ridge():
    param = _param_with_info(shape=(3, 2), logical_shape=(3, 2))
    recipe = _output_recipe(approximation=MatrixPreconditionerApproximation.BLOCK_DIAG)
    recipe.block_size = 2

    with pytest.raises(ValueError, match="non-divisible output dimension"):
        configure_matrix_update_param(param, output_recipe=recipe)


def test_configure_matrix_update_accepts_padded_block_diag_grad_gram_with_ridge():
    param = _param_with_info(shape=(3, 2), logical_shape=(3, 2))
    recipe = _output_recipe(approximation=MatrixPreconditionerApproximation.BLOCK_DIAG)
    recipe.block_size = 2
    recipe.ridge = 1e-6

    configure_matrix_update_param(param, output_recipe=recipe)


def test_configure_model_matrix_updates_rejects_grouped_expert_weights():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.empty(4, 3))
    set_linear_weight_info(
        module.weight,
        logical_shape=(4, 3),
        tp_layout="grouped_expert",
        sequence_parallel=False,
        expert_parallel=True,
        has_bias=False,
    )
    config = OptimizerConfig(matrix_optimizer="muon", matrix_input_preconditioner="feature_gram")

    with pytest.raises(RuntimeError, match="grouped/expert"):
        configure_model_matrix_updates([module], config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_native_column_parallel_linear_backward_collects_feature_gram():
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=4,
            num_attention_heads=1,
            use_cpu_initialization=True,
            perform_initialization=True,
            gradient_accumulation_fusion=False,
            params_dtype=torch.float32,
        )
        layer = ColumnParallelLinear(
            input_size=3,
            output_size=4,
            config=config,
            init_method=init_method_normal(config.init_method_std),
            bias=True,
            gather_output=False,
            skip_bias_add=False,
        ).cuda()
        recipe = _recipe(normalization=MatrixPreconditionerNormalization.SUM)
        output_recipe = _output_recipe(normalization=MatrixPreconditionerNormalization.SUM)
        configure_matrix_update_param(layer.weight, recipe=recipe, output_recipe=output_recipe)
        x = torch.randn(2, 5, 3, device="cuda")

        out, _ = layer(x)
        out.square().sum().backward()

        x_2d = x.reshape(-1, x.shape[-1]).float()
        dy_2d = (2 * out.detach()).reshape(-1, out.shape[-1]).float()
        torch.testing.assert_close(
            layer.weight.main_grad_feature_gram,
            x_2d.t().matmul(x_2d),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            layer.weight.main_grad_feature_count,
            torch.tensor(float(x_2d.shape[0]), device="cuda", dtype=torch.float64),
        )
        torch.testing.assert_close(
            layer.weight.main_grad_grad_gram,
            dy_2d.t().matmul(dy_2d),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            layer.weight.main_grad_grad_count,
            torch.tensor(float(dy_2d.shape[0]), device="cuda", dtype=torch.float64),
        )
    finally:
        Utils.destroy_model_parallel()
