# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Matrix-shaped optimizer shim for matrix update rules.

This module provides the Megatron-side optimizer surface for rules that own
eligible matrix-shaped parameters. Rules may optionally consume an input-side
feature gram and/or output-side grad gram preconditioner, but plain matrix
SGD/Muon paths do not require either sidecar.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import torch

from megatron.core.matrix_update import (
    MatrixApplyPlan,
    MatrixPreconditionerScope,
    TPUpdateMode,
    finalize_feature_gram_buffers,
    finalize_grad_gram_buffers,
    get_feature_gram_for_optimizer,
    get_grad_gram_for_optimizer,
    get_matrix_shard_spec,
    reset_feature_gram_buffers,
    reset_grad_gram_buffers,
    update_matrix_shard_spec,
)

MatrixUpdateRuleFn = Callable[
    [torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.nn.Parameter], torch.Tensor
]
MatrixInplaceUpdateRuleFn = Callable[
    [
        torch.nn.Parameter,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.nn.Parameter,
        float,
        float,
        bool,
    ],
    bool,
]


def default_matrix_apply_plan(
    param: torch.nn.Parameter, *, tp_update_mode: TPUpdateMode
) -> MatrixApplyPlan:
    """Construct a conservative apply plan from per-parameter metadata."""

    feature_gram_scope = getattr(param, "_feature_gram_scope", None)
    if feature_gram_scope is None:
        model_param = getattr(param, "_matrix_update_model_param", None)
        feature_gram_scope = getattr(model_param, "_feature_gram_scope", None)
    grad_gram_scope = getattr(param, "_grad_gram_scope", None)
    if grad_gram_scope is None:
        model_param = getattr(param, "_matrix_update_model_param", None)
        grad_gram_scope = getattr(model_param, "_grad_gram_scope", None)
    if feature_gram_scope is None and grad_gram_scope is None:
        approximation_labels: list[str] = []
        if tp_update_mode == TPUpdateMode.TP_BLOCK_LOCAL_APPROX:
            approximation_labels.append("tp_block_local_matrix_update")
        return MatrixApplyPlan(
            dp_apply_mode="layerwise_whole_param_owner",
            tp_update_mode=tp_update_mode,
            feature_gram_scope=None,
            grad_gram_scope=None,
            requires_full_logical_gradient=tp_update_mode != TPUpdateMode.TP_BLOCK_LOCAL_APPROX,
            requires_full_logical_weight_for_direction=False,
            approximation_label="+".join(approximation_labels) if approximation_labels else None,
        )
    approximation_labels: list[str] = []
    if feature_gram_scope == MatrixPreconditionerScope.TP_LOCAL:
        approximation_labels.append("tp_local_feature_gram")
    elif feature_gram_scope == MatrixPreconditionerScope.BLOCK_DIAG_APPROX:
        approximation_labels.append("block_diag_feature_gram")
    elif feature_gram_scope == MatrixPreconditionerScope.DIAG_APPROX:
        approximation_labels.append("diag_feature_gram")
    if grad_gram_scope == MatrixPreconditionerScope.TP_LOCAL:
        approximation_labels.append("tp_local_grad_gram")
    elif grad_gram_scope == MatrixPreconditionerScope.BLOCK_DIAG_APPROX:
        approximation_labels.append("block_diag_grad_gram")
    elif grad_gram_scope == MatrixPreconditionerScope.DIAG_APPROX:
        approximation_labels.append("diag_grad_gram")
    if tp_update_mode == TPUpdateMode.TP_BLOCK_LOCAL_APPROX:
        approximation_labels.append("tp_block_local_matrix_update")
    return MatrixApplyPlan(
        dp_apply_mode="layerwise_whole_param_owner",
        tp_update_mode=tp_update_mode,
        feature_gram_scope=feature_gram_scope,
        grad_gram_scope=grad_gram_scope,
        requires_full_logical_gradient=tp_update_mode != TPUpdateMode.TP_BLOCK_LOCAL_APPROX,
        requires_full_logical_weight_for_direction=False,
        approximation_label="+".join(approximation_labels) if approximation_labels else None,
    )


def validate_matrix_apply_plan(plan: MatrixApplyPlan) -> None:
    """Fail closed on unsupported or unlabeled matrix apply modes."""

    if plan.tp_update_mode == TPUpdateMode.TP_BLOCK_LOCAL_APPROX and not plan.approximation_label:
        raise RuntimeError("TP_BLOCK_LOCAL_APPROX requires an approximation_label.")
    if plan.dp_apply_mode == "standard_distopt_forbidden":
        raise RuntimeError(
            "Matrix optimizers cannot use standard DistributedOptimizer until logical matrix "
            "gather/apply/scatter views exist."
        )


class MatrixFunctionOptimizer(torch.optim.Optimizer):
    """Torch optimizer that applies a matrix update rule to whole parameters.

    ``update_rule`` must return a delta with the same shape as ``param``. The
    delta is added as ``param += lr * delta``. This sign convention keeps the
    scientific rule explicit: LocoProp-S can return ``-gamma * G C^-1`` while a
    Muon-like rule can return its own signed direction.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        update_rule: MatrixUpdateRuleFn,
        inplace_update_rule: Optional[MatrixInplaceUpdateRuleFn] = None,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
        momentum: float = 0.0,
        nesterov: bool = False,
        tp_update_mode: TPUpdateMode = TPUpdateMode.TP_ALLGATHER_LOGICAL_MATRIX,
        gram_process_groups: Iterable[torch.distributed.ProcessGroup] = (),
    ) -> None:
        if update_rule is None:
            raise ValueError("MatrixFunctionOptimizer requires an explicit update_rule")
        if momentum < 0.0:
            raise ValueError("MatrixFunctionOptimizer momentum must be non-negative")
        if nesterov and momentum == 0.0:
            raise ValueError("MatrixFunctionOptimizer nesterov requires nonzero momentum")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)
        self.update_rule = update_rule
        self.inplace_update_rule = inplace_update_rule
        self.tp_update_mode = tp_update_mode
        self.gram_process_groups = tuple(gram_process_groups)
        self._matrix_step = 0

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["matrix_step"] = self._matrix_step
        return state_dict

    def load_state_dict(self, state_dict):
        matrix_step = state_dict.get("matrix_step", 0)
        super().load_state_dict(state_dict)
        self._matrix_step = int(matrix_step)
        self._refresh_matrix_state_specs()

    def _model_param_for_factor(self, param: torch.nn.Parameter) -> torch.nn.Parameter:
        return getattr(param, "_matrix_update_model_param", param)

    def _factor_params(self) -> list[torch.nn.Parameter]:
        factor_params = []
        seen = set()
        for group in self.param_groups:
            for param in group["params"]:
                factor_param = self._model_param_for_factor(param)
                if id(factor_param) not in seen:
                    factor_params.append(factor_param)
                    seen.add(id(factor_param))
        return factor_params

    def _matrix_shard_spec_for_state(
        self, param: torch.nn.Parameter
    ):
        matrix_shard_spec = get_matrix_shard_spec(param)
        if matrix_shard_spec is not None:
            return matrix_shard_spec
        model_param = self._model_param_for_factor(param)
        return get_matrix_shard_spec(model_param)

    def _refresh_matrix_state_specs(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                matrix_shard_spec = self._matrix_shard_spec_for_state(param)
                if matrix_shard_spec is None:
                    continue
                for state_value in self.state.get(param, {}).values():
                    if isinstance(state_value, torch.Tensor) and state_value.shape == param.shape:
                        update_matrix_shard_spec(state_value, matrix_shard_spec)

    def _apply_momentum(
        self,
        param: torch.nn.Parameter,
        grad: torch.Tensor,
        *,
        momentum: float,
        nesterov: bool,
    ) -> torch.Tensor:
        if momentum == 0.0:
            return grad
        state = self.state[param]
        momentum_buffer = state.get("momentum_buffer")
        if momentum_buffer is None:
            momentum_buffer = torch.clone(grad).detach()
            state["momentum_buffer"] = momentum_buffer
        else:
            momentum_buffer.mul_(momentum).add_(grad)
        matrix_shard_spec = self._matrix_shard_spec_for_state(param)
        if matrix_shard_spec is not None:
            update_matrix_shard_spec(momentum_buffer, matrix_shard_spec)
        if nesterov:
            return grad.add(momentum_buffer, alpha=momentum)
        return momentum_buffer

    def _prepare_grad_for_update(
        self,
        param: torch.nn.Parameter,
        grad: torch.Tensor,
        *,
        lr: float,
        weight_decay: float,
        decoupled_weight_decay: bool,
        momentum: float,
        nesterov: bool,
    ) -> tuple[torch.Tensor, float]:
        if momentum == 0.0:
            return grad, weight_decay
        grad_for_update = grad
        if weight_decay != 0.0:
            if decoupled_weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            else:
                grad_for_update = grad.add(param, alpha=weight_decay)
            weight_decay = 0.0
        return (
            self._apply_momentum(
                param,
                grad_for_update,
                momentum=momentum,
                nesterov=nesterov,
            ),
            weight_decay,
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for param in self._factor_params():
            recipe = getattr(param, "_feature_gram_recipe", None)
            if recipe is not None:
                refresh_interval = getattr(recipe, "refresh_interval", 1)
                refresh_active = self._matrix_step % refresh_interval == 0
                reset_feature_gram_buffers(param, active=refresh_active, zero=refresh_active)
            recipe = getattr(param, "_grad_gram_recipe", None)
            if recipe is not None:
                refresh_interval = getattr(recipe, "refresh_interval", 1)
                refresh_active = self._matrix_step % refresh_interval == 0
                reset_grad_gram_buffers(param, active=refresh_active, zero=refresh_active)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        factor_params = [
            param for param in self._factor_params() if hasattr(param, "_feature_gram_recipe")
        ]
        finalize_feature_gram_buffers(
            factor_params, process_groups=self.gram_process_groups
        )
        factor_params = [
            param for param in self._factor_params() if hasattr(param, "_grad_gram_recipe")
        ]
        finalize_grad_gram_buffers(
            factor_params, process_groups=self.gram_process_groups
        )

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            decoupled_weight_decay = group["decoupled_weight_decay"]
            momentum = group.get("momentum", 0.0)
            nesterov = group.get("nesterov", False)
            for param in group["params"]:
                if param is None:
                    continue
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    grad = param.grad
                if grad is None:
                    continue
                model_param = self._model_param_for_factor(param)
                plan = default_matrix_apply_plan(model_param, tp_update_mode=self.tp_update_mode)
                validate_matrix_apply_plan(plan)
                feature_gram = (
                    get_feature_gram_for_optimizer(model_param)
                    if hasattr(model_param, "_feature_gram_recipe")
                    else None
                )
                grad_gram = (
                    get_grad_gram_for_optimizer(model_param)
                    if hasattr(model_param, "_grad_gram_recipe")
                    else None
                )
                grad_for_update, remaining_weight_decay = self._prepare_grad_for_update(
                    param,
                    grad,
                    lr=lr,
                    weight_decay=weight_decay,
                    decoupled_weight_decay=decoupled_weight_decay,
                    momentum=momentum,
                    nesterov=nesterov,
                )
                if self.inplace_update_rule is not None and self.inplace_update_rule(
                    param,
                    grad_for_update,
                    feature_gram,
                    grad_gram,
                    model_param,
                    lr,
                    remaining_weight_decay,
                    decoupled_weight_decay,
                ):
                    continue
                if remaining_weight_decay != 0.0:
                    if decoupled_weight_decay:
                        param.mul_(1.0 - lr * remaining_weight_decay)
                    else:
                        grad_for_update = grad_for_update.add(
                            param, alpha=remaining_weight_decay
                        )
                delta = self.update_rule(grad_for_update, feature_gram, grad_gram, model_param)
                if delta.shape != param.shape:
                    raise RuntimeError(
                        f"Matrix update rule returned shape {tuple(delta.shape)}, expected "
                        f"{tuple(param.shape)}."
                    )
                param.add_(delta.to(dtype=param.dtype), alpha=lr)

        self._matrix_step += 1
        return loss
