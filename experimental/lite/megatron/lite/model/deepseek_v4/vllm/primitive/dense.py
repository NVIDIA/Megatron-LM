"""Autograd bridges from vLLM-visible dense kernels to BF16-master VJPs."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_broadcast_tilelang,
    mhc_pre_tilelang,
)

from megatron.lite.primitive.modules.attention.hca import (
    HyperConnection,
    split_sinkhorn,
)


_MHC_ENTRIES = {
    "pre": mhc_pre_tilelang,
    "pre_broadcast": mhc_pre_broadcast_tilelang,
    "post": mhc_post_tilelang,
    "post_pre": mhc_fused_post_pre_tilelang,
    "head": hc_head_fused_kernel_tilelang,
}


def _compiled_vjp_or_eager(compiled, eager, *args):
    if not any(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args):
        return eager(*args)
    try:
        return compiled(*args)
    except torch._dynamo.exc.FailOnRecompileLimitHit:
        return eager(*args)


def mhc_kernel(name: str, *args, **kwargs):
    return _MHC_ENTRIES[name](*args, **kwargs)


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


def native_linear_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x2d = value.reshape(-1, value.shape[-1]).contiguous()
    dy2d = (
        grad_output.reshape(-1, grad_output.shape[-1])
        .to(value.dtype)
        .contiguous()
    )
    if grad_output.shape[:-1] != value.shape[:-1]:
        raise RuntimeError("linear bridge received incompatible grad_output shape")
    if value.is_cuda:
        from megatron.lite.primitive.utils.moe import _te_general_gemm

        grad_value = _te_general_gemm(
            weight.to(dy2d.dtype),
            dy2d,
            value.dtype,
            layout="NN",
            grad=True,
        )
        grad_weight = _te_general_gemm(
            x2d,
            dy2d,
            weight.dtype,
            layout="NT",
            grad=True,
        )
        if grad_value is not None and grad_weight is not None:
            return (
                grad_value[0].reshape(value.shape),
                grad_weight[0],
            )
    return (
        torch.mm(dy2d, weight.to(dy2d.dtype)).to(value.dtype).reshape(value.shape),
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


def visible_clamped_swiglu(value: torch.Tensor, limit: float) -> torch.Tensor:
    """Use vLLM's activation bytes with the Lite functional graph as its VJP."""
    from vllm.model_executor.layers.fused_moe.activation import (
        silu_and_mul_with_clamp,
    )

    from megatron.lite.primitive.modules.experts import swiglu_with_probs

    def visible_op(value_: torch.Tensor) -> torch.Tensor:
        if not value_.is_cuda:
            return swiglu_with_probs(value_, None, float(limit))
        # The training process does not construct a vLLM worker, so register
        # the stable-libtorch operators explicitly before calling the official
        # activation entry.
        import vllm._C_stable_libtorch  # noqa: F401

        output = value_.new_empty((*value_.shape[:-1], value_.shape[-1] // 2))
        silu_and_mul_with_clamp(output, value_, float(limit))
        return output

    if not torch.is_grad_enabled():
        return visible_op(value)
    return _ClampedSwiGLUVJP.apply(visible_op, value, float(limit))


def _clamped_swiglu_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    gate_source, up_source = value.chunk(2, dim=-1)
    gate = gate_source.float()
    up = up_source.float()
    if limit > 0:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
        gate_mask = gate_source.float() <= limit
        up_mask = (up_source.float() >= -limit) & (
            up_source.float() <= limit
        )
    else:
        gate_mask = up_mask = 1.0
    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    grad = grad_output.float()
    grad_gate = (
        grad
        * up
        * sigmoid
        * (1.0 + gate * (1.0 - sigmoid))
        * gate_mask
    )
    grad_up = grad * silu * up_mask
    return torch.cat((grad_gate, grad_up), dim=-1).to(value.dtype)


_compiled_clamped_swiglu_vjp = torch.compile(
    _clamped_swiglu_vjp,
    fullgraph=True,
    dynamic=False,
)


class _ClampedSwiGLUVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible_op: Callable, value: torch.Tensor, limit: float):
        output = visible_op(value)
        ctx.save_for_backward(value)
        ctx.limit = float(limit)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (value,) = ctx.saved_tensors
        grad_value = _compiled_vjp_or_eager(
            _compiled_clamped_swiglu_vjp,
            _clamped_swiglu_vjp,
            grad_output,
            value,
            ctx.limit,
        )
        return None, grad_value, None


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
        grad_value, grad_weight = native_linear_vjp(grad_output, value, fused)
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


def _rms_norm_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = value.float()
    w = weight.float()
    grad = grad_output.float()
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    scaled_grad = grad * w
    correction = (scaled_grad * x).mean(dim=-1, keepdim=True)
    grad_value = (scaled_grad * rstd - x * rstd.pow(3) * correction).to(
        value.dtype
    )
    reduce_dims = tuple(range(grad.ndim - 1))
    grad_weight = (grad * x * rstd).sum(dim=reduce_dims).to(weight.dtype)
    return grad_value, grad_weight


_compiled_rms_norm_vjp = torch.compile(
    _rms_norm_vjp,
    fullgraph=True,
    dynamic=False,
)


def _dispatch_rms_norm_vjp(grad_output, value, weight, eps):
    return _compiled_vjp_or_eager(
        _compiled_rms_norm_vjp,
        _rms_norm_vjp,
        grad_output,
        value,
        weight,
        eps,
    )


class _RMSNormVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible_op: Callable, value, weight, eps):
        output = visible_op(value, weight, eps)
        ctx.save_for_backward(value, weight)
        ctx.eps = float(eps)
        ctx.versions = parameter_versions((weight,))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, weight = ctx.saved_tensors
        check_parameter_versions((weight,), ctx.versions)
        grad_value, grad_weight = _dispatch_rms_norm_vjp(
            grad_output, value, weight, ctx.eps
        )
        return None, grad_value, grad_weight, None


def rms_norm(visible_op: Callable, value, weight, eps):
    if not torch.is_grad_enabled():
        return visible_op(value, weight, eps)
    return _RMSNormVJP.apply(visible_op, value, weight, eps)


class _FusedQKVRMSNormVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible_op: Callable, q, kv, q_weight, kv_weight, eps):
        q_out, kv_out = visible_op(q, kv, q_weight, kv_weight, eps)
        ctx.save_for_backward(q, kv, q_weight, kv_weight)
        ctx.eps = float(eps)
        ctx.versions = parameter_versions((q_weight, kv_weight))
        return q_out, kv_out

    @staticmethod
    def backward(ctx, grad_q, grad_kv):
        q, kv, q_weight, kv_weight = ctx.saved_tensors
        check_parameter_versions((q_weight, kv_weight), ctx.versions)
        grad_q_value, grad_q_weight = _dispatch_rms_norm_vjp(
            grad_q, q, q_weight, ctx.eps
        )
        grad_kv_value, grad_kv_weight = _dispatch_rms_norm_vjp(
            grad_kv, kv, kv_weight, ctx.eps
        )
        return (
            None,
            grad_q_value,
            grad_kv_value,
            grad_q_weight,
            grad_kv_weight,
            None,
        )


def fused_qkv_rms_norm(visible_op: Callable, q, kv, q_weight, kv_weight, eps):
    if not torch.is_grad_enabled():
        return visible_op(q, kv, q_weight, kv_weight, eps)
    return _FusedQKVRMSNormVJP.apply(
        visible_op, q, kv, q_weight, kv_weight, eps
    )


def _pre_graph(x, fn, scale, base, *, mult, iters, eps):
    residual = x
    if residual.ndim == 2:
        residual = residual.unsqueeze(-2).expand(
            *residual.shape[:-1], mult, residual.shape[-1]
        )
    flat = residual.flatten(-2)
    rms_inv = 1.0 / (flat.norm(dim=-1, keepdim=True) / math.sqrt(flat.shape[-1]) + eps)
    mixes = F.linear(flat, fn.to(flat.dtype)) * rms_inv
    pre, post, comb = split_sinkhorn(mixes, scale, base, mult, iters, eps)
    hidden = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)
    return residual, post.unsqueeze(-1), comb, hidden


def _post_graph(x, residual, post, comb):
    return HyperConnection.post(x, residual, post.squeeze(-1), comb)


def _normalization_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    dim: int,
    eps: float,
) -> torch.Tensor:
    total = value.sum(dim=dim, keepdim=True)
    denominator = total.clamp(min=eps)
    grad_value = grad_output / denominator
    active = total >= eps
    correction = (grad_output * value).sum(dim=dim, keepdim=True)
    return grad_value - torch.where(
        active,
        correction / denominator.square(),
        torch.zeros_like(correction),
    )


def _sinkhorn_vjp(
    grad_output: torch.Tensor,
    logits: torch.Tensor,
    iters: int,
    eps: float,
) -> torch.Tensor:
    maximum, maximum_indices = logits.max(dim=-1, keepdim=True)
    values = [torch.exp(logits - maximum)]
    comb = values[0]
    for _ in range(iters):
        comb = comb / comb.sum(dim=-1, keepdim=True).clamp(min=eps)
        values.append(comb)
        comb = comb / comb.sum(dim=-2, keepdim=True).clamp(min=eps)
        values.append(comb)

    grad = grad_output
    for iteration in reversed(range(iters)):
        after_row = values[2 * iteration + 1]
        before_row = values[2 * iteration]
        grad = _normalization_vjp(grad, after_row, -2, eps)
        grad = _normalization_vjp(grad, before_row, -1, eps)

    grad_logits = grad * values[0]
    grad_maximum = -grad_logits.sum(dim=-1, keepdim=True)
    return grad_logits.scatter_add(-1, maximum_indices, grad_maximum)


_compiled_sinkhorn_vjp = torch.compile(
    _sinkhorn_vjp,
    fullgraph=True,
    dynamic=False,
)


def _mhc_pre_vjp(
    grad_residual,
    grad_post,
    grad_comb,
    grad_hidden,
    x,
    fn,
    scale,
    base,
    norm_weight,
    mult,
    iters,
    eps,
    norm_eps,
):
    residual = x
    broadcast = residual.ndim == 2
    if broadcast:
        residual = residual.unsqueeze(-2).expand(
            *residual.shape[:-1], mult, residual.shape[-1]
        )
    flat = residual.flatten(-2)
    flat_norm = flat.norm(dim=-1, keepdim=True)
    rms_inv = 1.0 / (flat_norm / math.sqrt(flat.shape[-1]) + eps)
    fn_value = fn.to(flat.dtype)
    linear = F.linear(flat, fn_value)
    mixes = linear * rms_inv
    split_sizes = [mult, mult, mult * mult]
    pre_mix, post_mix, comb_mix = mixes.split(split_sizes, dim=-1)
    base_pre, base_post, base_comb = base.to(mixes.dtype).split(
        split_sizes, dim=-1
    )
    scale_value = scale.to(mixes.dtype)
    pre = torch.sigmoid(pre_mix * scale_value[0] + base_pre)
    post = 2 * torch.sigmoid(post_mix * scale_value[1] + base_post)
    comb_logits = (
        comb_mix * scale_value[2] + base_comb
    ).view(*comb_mix.shape[:-1], mult, mult)

    hidden = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)
    grad_hidden_input, grad_norm_weight = _rms_norm_vjp(
        grad_hidden, hidden, norm_weight, norm_eps
    )
    grad_pre = (grad_hidden_input.unsqueeze(-2) * residual).sum(dim=-1)
    grad_residual_value = grad_hidden_input.unsqueeze(-2) * pre.unsqueeze(-1)
    if grad_residual is not None:
        grad_residual_value = grad_residual_value + grad_residual.to(
            grad_residual_value.dtype
        )

    grad_pre_logits = grad_pre * pre * (1.0 - pre)
    grad_post_value = grad_post.squeeze(-1).to(post.dtype)
    grad_post_logits = grad_post_value * post * (1.0 - post * 0.5)
    grad_comb_logits = _sinkhorn_vjp(
        grad_comb.to(comb_logits.dtype), comb_logits, iters, eps
    )
    grad_comb_flat = grad_comb_logits.flatten(-2)
    grad_mixes = torch.cat(
        (
            grad_pre_logits * scale_value[0],
            grad_post_logits * scale_value[1],
            grad_comb_flat * scale_value[2],
        ),
        dim=-1,
    )
    grad_base = torch.cat(
        (grad_pre_logits, grad_post_logits, grad_comb_flat), dim=-1
    ).flatten(0, -2).sum(dim=0).to(base.dtype)
    grad_scale = torch.stack(
        (
            (grad_pre_logits * pre_mix).sum(),
            (grad_post_logits * post_mix).sum(),
            (grad_comb_flat * comb_mix).sum(),
        )
    ).to(scale.dtype)

    grad_linear = grad_mixes * rms_inv
    flat_2d = flat.flatten(0, -2)
    grad_linear_2d = grad_linear.flatten(0, -2)
    grad_fn = torch.matmul(
        grad_linear_2d.transpose(0, 1), flat_2d
    ).to(fn.dtype)
    grad_flat = torch.matmul(grad_linear, fn_value)
    grad_rms_inv = (grad_mixes * linear).sum(dim=-1, keepdim=True)
    norm_scale = math.sqrt(flat.shape[-1])
    grad_norm = -grad_rms_inv * rms_inv.square() / norm_scale
    grad_flat = grad_flat + torch.where(
        flat_norm > 0,
        grad_norm * flat / flat_norm.clamp_min(torch.finfo(flat.dtype).tiny),
        torch.zeros_like(flat),
    )
    grad_residual_value = grad_residual_value + grad_flat.view_as(residual)
    grad_x = (
        grad_residual_value.sum(dim=-2)
        if broadcast
        else grad_residual_value
    ).to(x.dtype)
    return grad_x, grad_fn, grad_scale, grad_base, grad_norm_weight


_compiled_mhc_pre_vjp = torch.compile(
    _mhc_pre_vjp,
    fullgraph=True,
    dynamic=False,
)


class _MHCPreVJP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        visible_op: Callable,
        x,
        fn,
        scale,
        base,
        norm_weight,
        mult,
        iters,
        eps,
        norm_eps,
    ):
        outputs = visible_op(x, fn, scale, base, norm_weight)
        ctx.save_for_backward(x, fn, scale, base, norm_weight)
        ctx.mult = int(mult)
        ctx.iters = int(iters)
        ctx.eps = float(eps)
        ctx.norm_eps = float(norm_eps)
        ctx.versions = parameter_versions((fn, scale, base, norm_weight))
        return outputs

    @staticmethod
    def backward(
        ctx,
        grad_residual,
        grad_post,
        grad_comb,
        grad_hidden,
    ):
        x, fn, scale, base, norm_weight = ctx.saved_tensors
        check_parameter_versions((fn, scale, base, norm_weight), ctx.versions)
        results = _compiled_vjp_or_eager(
            _compiled_mhc_pre_vjp,
            _mhc_pre_vjp,
            grad_residual,
            grad_post,
            grad_comb,
            grad_hidden,
            x,
            fn,
            scale,
            base,
            norm_weight,
            ctx.mult,
            ctx.iters,
            ctx.eps,
            ctx.norm_eps,
        )
        grad_x, grad_fn, grad_scale, grad_base, grad_norm_weight = results
        return (
            None,
            grad_x,
            grad_fn,
            grad_scale,
            grad_base,
            grad_norm_weight,
            None,
            None,
            None,
            None,
        )


class _MHCPostVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible_op: Callable, x, residual, post, comb):
        output = visible_op(x, residual, post, comb)
        ctx.save_for_backward(x, residual, post, comb)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, residual, post, comb = ctx.saved_tensors
        dtype = x.dtype
        grad = grad_output.to(dtype)
        post_value = post.squeeze(-1).to(dtype)
        comb_value = comb.to(dtype)
        residual_value = residual.to(dtype)

        grad_x = (grad * post_value.unsqueeze(-1)).sum(dim=-2).to(x.dtype)
        grad_post = (grad * x.to(dtype).unsqueeze(-2)).sum(dim=-1)
        if post.ndim == grad_post.ndim + 1:
            grad_post = grad_post.unsqueeze(-1)
        if grad.is_cuda:
            grad_residual = mhc_kernel(
                "post",
                torch.zeros_like(x),
                grad,
                torch.zeros_like(post, dtype=torch.float32),
                comb_value.float().contiguous(),
            ).to(residual.dtype)
        else:
            grad_residual = torch.matmul(
                comb_value.transpose(-2, -1), grad
            ).to(residual.dtype)
        grad_comb = torch.matmul(
            grad, residual_value.transpose(-2, -1)
        ).to(comb.dtype)
        return None, grad_x, grad_residual, grad_post.to(post.dtype), grad_comb


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
    if not torch.is_grad_enabled():
        return visible_op(x, fn, scale, base, norm_weight)
    return _MHCPreVJP.apply(
        visible_op,
        x,
        fn,
        scale,
        base,
        norm_weight,
        mult,
        iters,
        eps,
        norm_eps,
    )


def mhc_post(visible_op: Callable, x, residual, post, comb):
    if not torch.is_grad_enabled():
        return visible_op(x, residual, post, comb)
    return _MHCPostVJP.apply(visible_op, x, residual, post, comb)


def mhc_head(visible_op: Callable, x, fn, scale, base, *, eps: float):
    if not torch.is_grad_enabled():
        return visible_op(x, fn, scale, base)
    return _MHCHeadVJP.apply(
        visible_op, x, fn, scale, base, float(eps)
    )


def _mhc_head_vjp(grad_output, x, fn, scale, base, eps):
    flat = x.flatten(-2).float()
    fn_value = fn.float()
    scale_value = scale.float()
    base_value = base.float()
    rstd = torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + eps
    )
    raw_mixes = F.linear(flat, fn_value)
    mixes = raw_mixes * rstd
    sigmoid = torch.sigmoid(mixes * scale_value + base_value)

    grad = grad_output.float()
    grad_pre = (grad.unsqueeze(-2) * x.float()).sum(dim=-1)
    grad_logits = grad_pre * sigmoid * (1.0 - sigmoid)
    reduce_dims = tuple(range(grad_logits.ndim - 1))
    grad_scale = (grad_logits * mixes).sum(dim=reduce_dims)
    grad_base = grad_logits.sum(dim=reduce_dims)

    grad_mixes = grad_logits * scale_value
    grad_raw = grad_mixes * rstd
    grad_rstd = (grad_mixes * raw_mixes).sum(dim=-1, keepdim=True)
    grad_flat = torch.matmul(grad_raw, fn_value)
    grad_flat = grad_flat - (
        grad_rstd * flat * rstd.pow(3) / flat.shape[-1]
    )
    grad_fn = torch.matmul(
        grad_raw.reshape(-1, grad_raw.shape[-1]).transpose(0, 1),
        flat.reshape(-1, flat.shape[-1]),
    )

    direct_grad_x = grad.unsqueeze(-2) * (sigmoid + eps).unsqueeze(-1)
    grad_x = direct_grad_x + grad_flat.reshape_as(x)
    return (
        grad_x.to(x.dtype),
        grad_fn.to(fn.dtype),
        grad_scale.to(scale.dtype),
        grad_base.to(base.dtype),
    )


_compiled_mhc_head_vjp = torch.compile(
    _mhc_head_vjp,
    fullgraph=True,
    dynamic=False,
)


class _MHCHeadVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible_op: Callable, x, fn, scale, base, eps):
        output = visible_op(x, fn, scale, base)
        ctx.save_for_backward(x, fn, scale, base)
        ctx.eps = float(eps)
        ctx.versions = parameter_versions((fn, scale, base))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, fn, scale, base = ctx.saved_tensors
        check_parameter_versions((fn, scale, base), ctx.versions)
        results = _compiled_vjp_or_eager(
            _compiled_mhc_head_vjp,
            _mhc_head_vjp,
            grad_output,
            x,
            fn,
            scale,
            base,
            ctx.eps,
        )
        grad_x, grad_fn, grad_scale, grad_base = results
        return (
            None,
            grad_x,
            grad_fn,
            grad_scale,
            grad_base,
            None,
        )
