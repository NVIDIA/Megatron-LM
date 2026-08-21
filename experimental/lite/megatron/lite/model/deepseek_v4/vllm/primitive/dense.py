from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F

from megatron.lite.primitive.modules.attention.hca import HyperConnection, split_sinkhorn

def parameter_versions(parameters: Iterable[torch.Tensor]) -> tuple[int, ...]:
    return tuple(parameter._version for parameter in parameters)


def check_parameter_versions(
    parameters: Iterable[torch.Tensor], expected: tuple[int, ...]
) -> None:
    actual = parameter_versions(parameters)
    if actual != expected:
        raise RuntimeError(
            "DS4 vLLM master parameter changed between forward and backward; "
            f"versions={expected}->{actual}"
        )


def fp32_linear_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x2d = value.reshape(-1, value.shape[-1]).float()
    dy2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
    if grad_output.shape[:-1] != value.shape[:-1]:
        raise RuntimeError("linear bridge received incompatible grad_output shape")
    return (
        torch.mm(dy2d, weight.float()).to(value.dtype).reshape(value.shape),
        torch.mm(dy2d.T, x2d).to(weight.dtype),
    )


def _own(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple):
        return tuple(_own(item) for item in value)
    if isinstance(value, list):
        return tuple(_own(item) for item in value)
    raise TypeError("training bridge outputs must be tensors")


class _VisibleFunctionalVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, visible_op, functional_op, version_indices, *inputs):
        output = _own(visible_op(*inputs))
        ctx.functional_op = functional_op
        ctx.inputs = inputs
        ctx.version_indices = tuple(version_indices)
        versioned = tuple(inputs[index] for index in ctx.version_indices)
        ctx.versions = parameter_versions(versioned)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs):
        versioned = tuple(ctx.inputs[index] for index in ctx.version_indices)
        check_parameter_versions(versioned, ctx.versions)
        with torch.enable_grad():
            recompute_inputs = tuple(
                value.detach().requires_grad_(value.is_floating_point())
                for value in ctx.inputs
            )
            outputs = ctx.functional_op(*recompute_inputs)
            outputs = outputs if isinstance(outputs, tuple) else (outputs,)
            differentiable_inputs = tuple(
                value for value in recompute_inputs if value.requires_grad
            )
            grads = torch.autograd.grad(
                outputs,
                differentiable_inputs,
                grad_outputs=grad_outputs,
                allow_unused=True,
            )
        grad_iter = iter(grads)
        input_grads = tuple(
            next(grad_iter) if value.requires_grad else None
            for value in recompute_inputs
        )
        return None, None, None, *input_grads


def visible_functional_vjp(
    visible_op: Callable,
    functional_op: Callable,
    inputs: tuple[torch.Tensor, ...],
    *,
    version_indices: tuple[int, ...] = (),
):
    # Forward-only callers use the visible implementation without an autograd owner.
    if not torch.is_grad_enabled():
        return visible_op(*inputs)
    return _VisibleFunctionalVJP.apply(
        visible_op, functional_op, version_indices, *inputs
    )


class _VisibleLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible_op: Callable, value: torch.Tensor, *weights):
        output = visible_op(value, *weights)
        if isinstance(output, (tuple, list)):
            output = output[0]
        ctx.save_for_backward(value)
        ctx.weights = weights
        ctx.versions = parameter_versions(weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        check_parameter_versions(ctx.weights, ctx.versions)
        (value,) = ctx.saved_tensors
        fused = torch.cat(ctx.weights, dim=0)
        grad_value, grad_weight = fp32_linear_vjp(grad_output, value, fused)
        return (
            None,
            grad_value,
            *grad_weight.split([weight.shape[0] for weight in ctx.weights], dim=0),
        )


def _empty_output(value: torch.Tensor, weights: tuple[torch.Tensor, ...]):
    return value.new_empty(
        (*value.shape[:-1], sum(weight.shape[0] for weight in weights))
    )


def visible_linear(visible_op, value, master_weight):
    if value.numel() == 0:
        def visible_op(_value, _weight):
            return _empty_output(_value, (_weight,))
    if not torch.is_grad_enabled():
        output = visible_op(value, master_weight)
        return output[0] if isinstance(output, (tuple, list)) else output
    return _VisibleLinear.apply(visible_op, value, master_weight)


block_fp8_linear = visible_linear


def fused_block_fp8_linear(visible_op, value, *master_weights):
    if value.numel() == 0:
        def visible_op(_value, *_weights):
            return _empty_output(_value, _weights)
    if not torch.is_grad_enabled():
        output = visible_op(value, *master_weights)
        return output[0] if isinstance(output, (tuple, list)) else output
    return _VisibleLinear.apply(visible_op, value, *master_weights)


def gate_linear(visible_op, value, master_weight):
    if not torch.is_grad_enabled():
        output = visible_op(value)
        return output[0] if isinstance(output, (tuple, list)) else output
    return _VisibleLinear.apply(
        lambda input_value, _weight: visible_op(input_value),
        value,
        master_weight,
    )


def _norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(value.float(), (value.shape[-1],), weight.float(), eps).to(
        value.dtype
    )


def rms_norm(visible_op: Callable, value, weight, eps):
    return visible_functional_vjp(
        lambda value_, weight_: visible_op(value_, weight_, eps),
        lambda value_, weight_: _norm(value_, weight_, eps),
        (value, weight),
        version_indices=(1,),
    )


def fused_qkv_rms_norm(visible_op: Callable, q, kv, q_weight, kv_weight, eps):
    return visible_functional_vjp(
        lambda q_, kv_, qw_, kvw_: visible_op(q_, kv_, qw_, kvw_, eps),
        lambda q_, kv_, qw_, kvw_: (_norm(q_, qw_, eps), _norm(kv_, kvw_, eps)),
        (q, kv, q_weight, kv_weight),
        version_indices=(2, 3),
    )


def _pre_graph(x, fn, scale, base, *, mult, iters, eps):
    residual = x
    if residual.ndim == 2:
        residual = residual.unsqueeze(-2).expand(
            *residual.shape[:-1], mult, residual.shape[-1]
        )
    flat = residual.flatten(-2)
    rms_inv = 1.0 / (
        flat.norm(dim=-1, keepdim=True) / math.sqrt(flat.shape[-1]) + eps
    )
    mixes = F.linear(flat, fn.to(flat.dtype)) * rms_inv
    pre, post, comb = split_sinkhorn(mixes, scale, base, mult, iters, eps)
    hidden = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)
    return residual, post.unsqueeze(-1), comb, hidden


def _post_graph(x, residual, post, comb):
    return HyperConnection.post(x, residual, post.squeeze(-1), comb)


def mhc_pre_broadcast(
    visible_op: Callable,
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    mult: int,
    iters: int,
    eps: float,
    norm_eps: float,
):
    def functional(x_, fn_, scale_, base_, norm_weight_):
        residual, post, comb, hidden = _pre_graph(
            x_, fn_, scale_, base_, mult=mult, iters=iters, eps=eps
        )
        hidden = F.rms_norm(hidden, (hidden.shape[-1],), norm_weight_, norm_eps)
        return residual, post, comb, hidden

    return visible_functional_vjp(
        visible_op,
        functional,
        (x, fn, scale, base, norm_weight),
        version_indices=(1, 2, 3, 4),
    )


def mhc_post(visible_op: Callable, x, residual, post, comb):
    return visible_functional_vjp(
        visible_op, _post_graph, (x, residual, post, comb)
    )


def mhc_head(visible_op: Callable, x, fn, scale, base, *, eps: float):
    def functional(x_, fn_, scale_, base_):
        flat = x_.flatten(-2).float()
        rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        mixes = F.linear(flat, fn_.float()) * rstd
        pre = torch.sigmoid(mixes * scale_.float() + base_.float()) + eps
        return torch.sum(pre.unsqueeze(-1) * x_.float(), dim=-2).to(x_.dtype)

    return visible_functional_vjp(
        visible_op, functional, (x, fn, scale, base), version_indices=(1, 2, 3)
    )


def _inverse_rope(o, positions, cache, nope_dim, rope_dim):
    prefix, rope = o[..., :nope_dim], o[..., nope_dim : nope_dim + rope_dim]
    selected = cache.index_select(0, positions.long()).float()
    cos = selected[..., : rope_dim // 2].unsqueeze(-2)
    sin = selected[..., rope_dim // 2 : rope_dim].unsqueeze(-2)
    even, odd = rope[..., 0::2].float(), rope[..., 1::2].float()
    rotated = torch.stack(
        (even * cos + odd * sin, odd * cos - even * sin), dim=-1
    )
    return torch.cat((prefix.float(), rotated.flatten(-2)), dim=-1)


def o_projection(
    visible_op: Callable,
    o: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    *,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
):
    def functional(o_, wa_, wb_):
        inverse = _inverse_rope(o_, positions, cos_sin_cache, nope_dim, rope_dim)
        grouped = inverse.reshape(inverse.shape[0], n_groups, -1)
        wa = wa_.float().reshape(n_groups, o_lora_rank, -1)
        z = torch.einsum("tgd,grd->tgr", grouped, wa)
        return F.linear(z.flatten(1), wb_.float()).to(o_.dtype)

    return visible_functional_vjp(
        visible_op, functional, (o, wo_a, wo_b), version_indices=(1, 2)
    )


class _VLLMFixedRouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits, visible_op, renormalize, route_scale):
        weights, ids = visible_op(logits)
        ctx.save_for_backward(logits, ids.clone())
        ctx.renormalize = renormalize
        ctx.route_scale = route_scale
        ctx.mark_non_differentiable(ids)
        return weights, ids

    @staticmethod
    def backward(ctx: Any, grad_weights, _grad_ids):
        logits, ids = ctx.saved_tensors
        with torch.enable_grad():
            replay = logits.detach().float().requires_grad_(True)
            scores = torch.sqrt(F.softplus(replay))
            selected = scores.gather(-1, ids.long())
            if ctx.renormalize:
                selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            selected = selected * ctx.route_scale
            (dlogits,) = torch.autograd.grad(selected, replay, grad_weights.float())
        return dlogits.to(logits.dtype), None, None, None


def fixed_route_vjp(visible_op, logits, *, renormalize: bool, route_scale: float):
    if not torch.is_grad_enabled():
        return visible_op(logits)
    return _VLLMFixedRouteFunction.apply(logits, visible_op, renormalize, route_scale)
