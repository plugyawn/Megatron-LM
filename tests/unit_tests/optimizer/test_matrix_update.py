# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.matrix_function_optimizer import (
    MatrixFunctionOptimizer,
    default_matrix_apply_plan,
)
from megatron.core.optimizer.matrix_update import (
    FeatureGramApproximation,
    FeatureGramNormalization,
    FeatureGramScope,
    FeatureGramRecipe,
    TPUpdateMode,
    configure_model_matrix_updates,
    configure_matrix_update_param,
    feature_gram_scope_for,
    finalize_feature_gram_buffers,
    get_feature_gram_for_optimizer,
    is_matrix_update_eligible,
    maybe_accumulate_feature_gram,
    set_feature_gram_finalization_required,
    set_linear_weight_info,
)


def _param_with_info(tp_layout="column_parallel"):
    param = torch.nn.Parameter(torch.empty(4, 3))
    set_linear_weight_info(
        param,
        logical_shape=(4, 3),
        tp_layout=tp_layout,
        sequence_parallel=False,
        expert_parallel=False,
        has_bias=True,
    )
    return param


def _recipe(
    approximation=FeatureGramApproximation.FULL,
    normalization=FeatureGramNormalization.SUM,
    refresh_interval=1,
):
    return FeatureGramRecipe(
        approximation=approximation,
        scope=FeatureGramScope.GLOBAL_EXACT,
        normalization=normalization,
        source_dtype="fp32_cast",
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
    recipe = _recipe(approximation=FeatureGramApproximation.DIAG)
    configure_matrix_update_param(param, recipe=recipe)
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])

    maybe_accumulate_feature_gram(param, x)

    torch.testing.assert_close(param.main_grad_feature_gram, (x * x).sum(dim=0))


def test_block_diag_feature_gram_accumulates_padded_blocks():
    param = _param_with_info()
    recipe = _recipe(approximation=FeatureGramApproximation.BLOCK_DIAG)
    recipe.block_size = 2
    configure_matrix_update_param(param, recipe=recipe)
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])

    maybe_accumulate_feature_gram(param, x)

    expected = torch.zeros(2, 2, 2)
    expected[0] = x[:, :2].t().matmul(x[:, :2])
    padded_last = torch.nn.functional.pad(x[:, 2:], (0, 1))
    expected[1] = padded_last.t().matmul(padded_last)
    torch.testing.assert_close(param.main_grad_feature_gram, expected)


def test_feature_gram_mean_normalization_is_applied_on_consumption():
    param = _param_with_info()
    recipe = _recipe(normalization=FeatureGramNormalization.MEAN)
    configure_matrix_update_param(param, recipe=recipe)
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])

    maybe_accumulate_feature_gram(param, x)

    torch.testing.assert_close(get_feature_gram_for_optimizer(param), x.t().matmul(x) / 2.0)


def test_feature_gram_consumption_requires_collected_rows():
    param = _param_with_info()
    configure_matrix_update_param(param, recipe=_recipe())

    with pytest.raises(RuntimeError, match="zero collected feature rows"):
        get_feature_gram_for_optimizer(param)


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


def test_row_parallel_feature_gram_scope_is_tp_local_block_diag_even_for_diag():
    assert (
        feature_gram_scope_for(FeatureGramApproximation.DIAG, "row_parallel")
        == FeatureGramScope.TP_LOCAL_BLOCK_DIAG
    )


def test_full_feature_gram_rejects_undersampled_rank_deficient_recipe():
    param = _param_with_info()
    recipe = _recipe()
    recipe.token_sample_size = 2
    recipe.min_samples_per_feature = 1.0

    with pytest.raises(ValueError, match="rank-deficient"):
        configure_matrix_update_param(param, recipe=recipe)


def test_sketch_approximation_fails_closed():
    param = _param_with_info()
    recipe = _recipe(approximation=FeatureGramApproximation.SKETCH)

    with pytest.raises(NotImplementedError, match="sketch"):
        configure_matrix_update_param(param, recipe=recipe)


def test_lm_head_marker_excludes_matrix_update_eligibility():
    param = _param_with_info()
    param.is_embedding_or_output_parameter = True

    assert not is_matrix_update_eligible(param, min_matrix_dim=2)


def test_fp8_dequant_source_fails_closed_in_native_collector():
    param = _param_with_info()
    recipe = _recipe()
    recipe.source_dtype = "fp8_dequant"
    configure_matrix_update_param(param, recipe=recipe)

    with pytest.raises(NotImplementedError, match="fp8_dequant"):
        maybe_accumulate_feature_gram(param, torch.ones(2, 3))


def test_matrix_function_optimizer_consumes_feature_gram_and_preserves_buffer_for_cache_reuse():
    param = _param_with_info()
    param.data = torch.zeros_like(param.data)
    param.main_grad = torch.ones_like(param.data)
    recipe = _recipe(normalization=FeatureGramNormalization.MEAN)
    configure_matrix_update_param(param, recipe=recipe)
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    maybe_accumulate_feature_gram(param, x)

    seen = {}

    def update_rule(grad, feature_gram, model_param):
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


def test_matrix_function_optimizer_respects_feature_gram_refresh_interval():
    param = _param_with_info()
    param.data = torch.zeros_like(param.data)
    param.main_grad = torch.ones_like(param.data)
    recipe = _recipe(normalization=FeatureGramNormalization.MEAN, refresh_interval=2)
    configure_matrix_update_param(param, recipe=recipe)
    seen = []

    def update_rule(grad, feature_gram, model_param):
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
        matrix_optimizer="locoprop_s",
        matrix_feature_gram="diag",
        matrix_feature_gram_refresh_interval=2,
        matrix_feature_gram_ridge=1.0,
    )
    calls = []
    original_factorize = matrix_optimizer_module.factorize_feature_gram

    def counted_factorize(*args, **kwargs):
        calls.append(1)
        return original_factorize(*args, **kwargs)

    monkeypatch.setattr(matrix_optimizer_module, "factorize_feature_gram", counted_factorize)
    rule = matrix_optimizer_module._make_matrix_update_rule(config, SimpleNamespace(tp=None))

    first = rule(grad, feature_gram, param)
    second = rule(grad, feature_gram, param)
    param._feature_gram_generation += 1
    third = rule(grad, feature_gram, param)

    assert len(calls) == 2
    torch.testing.assert_close(first, -grad / (feature_gram + 1.0))
    torch.testing.assert_close(second, first)
    torch.testing.assert_close(third, first)


def test_matrix_function_optimizer_state_dict_does_not_serialize_update_rule():
    param = _param_with_info()

    def update_rule(grad, feature_gram, model_param):
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

    def update_rule(grad, feature_gram, model_param):
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
    recipe = _recipe(approximation=FeatureGramApproximation.DIAG)
    configure_matrix_update_param(param, recipe=recipe)
    param.main_grad_feature_gram.copy_(torch.tensor([1.0, 3.0, 5.0]))
    param.main_grad_feature_count.fill_(1.0)
    calls = []

    def update_rule(grad, feature_gram, model_param):
        raise AssertionError("generic update_rule should not run")

    def inplace_update_rule(
        update_param,
        grad,
        feature_gram,
        model_param,
        lr,
        weight_decay,
        decoupled_weight_decay,
    ):
        calls.append(1)
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


def test_matrix_inplace_update_rule_uses_diag_newton_muon_for_local_matrix(monkeypatch):
    from types import SimpleNamespace

    import megatron.core.optimizer.matrix_optimizer as matrix_optimizer_module

    if not matrix_optimizer_module.HAVE_EMERGING_MATRIX_OPTIMIZERS:
        pytest.skip("emerging_optimizers is not installed")

    param = _param_with_info()
    grad = torch.ones_like(param.data)
    feature_gram = torch.tensor([2.0, 3.0, 4.0])
    config = OptimizerConfig(
        matrix_optimizer="newton_muon",
        matrix_feature_gram="diag",
        matrix_feature_gram_ridge=1.0,
        muon_num_ns_steps=2,
        muon_ns_coefficients="simple",
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

    assert rule(param, grad, feature_gram, param, 0.1, 0.2, True)
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
    recipe = _recipe(approximation=FeatureGramApproximation.DIAG)
    recipe.scope = feature_gram_scope_for(recipe.approximation, "row_parallel")
    configure_matrix_update_param(param, recipe=recipe)

    plan = default_matrix_apply_plan(
        param, tp_update_mode=TPUpdateMode.TP_ALLGATHER_LOGICAL_MATRIX
    )

    assert plan.requires_full_logical_gradient
    assert plan.approximation_label == "tp_local_block_diag_feature_gram"


def test_matrix_function_optimizer_fails_when_grad_has_no_feature_rows():
    param = _param_with_info()
    param.main_grad = torch.ones_like(param.data)
    configure_matrix_update_param(param, recipe=_recipe())

    def update_rule(grad, feature_gram, model_param):
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


def test_optimizer_config_rejects_standard_distopt_with_matrix_optimizer():
    with pytest.raises(ValueError, match="standard DistributedOptimizer"):
        OptimizerConfig(matrix_optimizer="newton_muon", use_distributed_optimizer=True)


def test_optimizer_config_accepts_block_diag_feature_gram():
    config = OptimizerConfig(
        matrix_optimizer="locoprop_s",
        matrix_feature_gram="block_diag",
        matrix_feature_gram_block_size=2,
    )

    assert config.matrix_feature_gram == "block_diag"


def test_optimizer_config_rejects_unimplemented_active_matrix_options():
    with pytest.raises(ValueError, match="block_diag/sketch"):
        OptimizerConfig(matrix_optimizer="newton_muon", matrix_feature_gram="block_diag")
    OptimizerConfig(matrix_optimizer="newton_muon", matrix_feature_gram_source_dtype="fp8_dequant")
    with pytest.raises(ValueError, match="augmented_feature_sum"):
        OptimizerConfig(matrix_optimizer="newton_muon", matrix_bias_mode="augmented_feature_sum")
    OptimizerConfig(matrix_optimizer="newton_muon", matrix_feature_gram_refresh_interval=2)
    with pytest.raises(ValueError, match="ema_beta"):
        OptimizerConfig(matrix_optimizer="newton_muon", matrix_feature_gram_ema_beta=0.9)


def test_optimizer_config_exposes_muon_ns_coefficients():
    config = OptimizerConfig(matrix_optimizer="newton_muon", muon_ns_coefficients="simple")

    assert config.muon_ns_coefficients == "simple"


def test_configure_model_matrix_updates_rejects_native_fp8_dequant_source():
    module = torch.nn.Module()
    module.weight = _param_with_info()
    config = OptimizerConfig(
        matrix_optimizer="newton_muon", matrix_feature_gram_source_dtype="fp8_dequant"
    )

    with pytest.raises(RuntimeError, match="native linears cannot dequantize FP8"):
        configure_model_matrix_updates([module], config)


def test_configure_model_matrix_updates_rejects_row_parallel_full_gram():
    module = torch.nn.Module()
    module.weight = _param_with_info(tp_layout="row_parallel")
    config = OptimizerConfig(matrix_optimizer="newton_muon", matrix_feature_gram="full")

    with pytest.raises(RuntimeError, match="row-parallel"):
        configure_model_matrix_updates([module], config)


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
    config = OptimizerConfig(matrix_optimizer="newton_muon")

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
        recipe = _recipe(normalization=FeatureGramNormalization.SUM)
        configure_matrix_update_param(layer.weight, recipe=recipe)
        x = torch.randn(2, 5, 3, device="cuda")

        out, _ = layer(x)
        out.square().sum().backward()

        x_2d = x.reshape(-1, x.shape[-1]).float()
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
    finally:
        Utils.destroy_model_parallel()
