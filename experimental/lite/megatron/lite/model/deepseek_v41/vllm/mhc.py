# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Caller-stream overlap kernels with the W9-b5 owned mHC graph VJP."""
import torch

from megatron.core.transformer.hyper_connection import _sinkhorn_iterations


def _reference(values, pre, gamma, fn, scale, base, output, residual, post, comb, options):
    eps, hc_eps, sinkhorn_eps, iterations, broadcast, pending = options
    copies, width = values.shape[-2:]
    stream = values
    if pending:
        mixed = torch.einsum("...ij,...id->...jd", comb.float(), residual.float())
        stream = (mixed + post.float().unsqueeze(-1) * output.float().unsqueeze(-2)).to(
            values.dtype
        )
    if broadcast:
        token = stream[..., 0, :].float()
        weight = fn.reshape(-1, copies, width).sum(1)
    else:
        token, weight = stream.flatten(-2).float(), fn
    projected = torch.nn.functional.linear(token, weight.float()) * torch.rsqrt(
        token.square().mean(-1, keepdim=True) + eps
    )
    a, b, c = projected.split((copies, copies, copies * copies), -1)
    ba, bb, bc = base.float().split((copies, copies, copies * copies))
    next_pre = torch.sigmoid(a * scale[0] + ba) + hc_eps
    next_post = 2 * torch.sigmoid(b * scale[1] + bb)
    next_comb = _sinkhorn_iterations(
        (c * scale[2] + bc).reshape(*token.shape[:-1], copies, copies), iterations, sinkhorn_eps
    )
    collapsed = (
        stream[..., 0, :]
        if broadcast
        else (stream.float() * pre.unsqueeze(-1)).sum(-2).to(stream.dtype)
    )
    rounded_gamma = gamma + (gamma.bfloat16().to(gamma.dtype) - gamma).detach()
    normed = torch.nn.functional.rms_norm(collapsed, (width,), rounded_gamma, eps)
    return stream, next_pre, next_post, next_comb, normed


class _Joint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, pre, gamma, fn, scale, base, output, residual, post, comb, options):
        if not values.is_cuda:
            raise RuntimeError("Deployment mHC overlap requires CUDA")
        from vllm.models.deepseek_v41.nvidia.ops.mhc import (
            mhc_pre_delayed_overlap,
            mhc_shifted_post_pre,
        )

        eps, hc_eps, sinkhorn_eps, iterations, broadcast, pending = options
        copies, width = values.shape[-2:]
        ctx.save_for_backward(values, pre, gamma, fn, scale, base, output, residual, post, comb)
        ctx.options = options
        kwargs = dict(
            norm_weight=gamma.bfloat16(), norm_eps=eps, stream=torch.cuda.current_stream()
        )
        if pending:
            stream, new_post, new_comb, normed, new_pre, _ = mhc_shifted_post_pre(
                output.reshape(-1, width).contiguous(),
                residual.reshape(-1, copies, width).contiguous(),
                post.reshape(-1, copies, 1).contiguous(),
                comb.reshape(-1, copies, copies).contiguous(),
                fn,
                scale,
                base,
                eps,
                hc_eps,
                hc_eps,
                2.0,
                iterations,
                pre_mix=pre.reshape(-1, copies).contiguous(),
                **kwargs,
            )
        else:
            stream = values.reshape(-1, copies, width).contiguous()
            if broadcast:
                fn = fn.reshape(-1, copies, width).sum(1)
                kwargs["x"] = stream[:, 0].contiguous()
            else:
                kwargs["pre_mix"] = pre.reshape(-1, copies).contiguous()
            new_post, new_comb, normed, new_pre = mhc_pre_delayed_overlap(
                stream, fn, scale, base, eps, hc_eps, sinkhorn_eps, 2.0, iterations, **kwargs
            )
        prefix = values.shape[:-2]
        return (
            stream.reshape_as(values),
            new_pre.reshape(*prefix, copies),
            new_post.reshape(*prefix, copies),
            new_comb.reshape(*prefix, copies, copies),
            normed.reshape(*prefix, width),
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        required = ctx.needs_input_grad[:10]
        with (
            torch.enable_grad(),
            torch.autocast(device_type=ctx.saved_tensors[0].device.type, enabled=False),
        ):
            live = [x.detach().requires_grad_(need) for x, need in zip(ctx.saved_tensors, required)]
            result = _reference(*live, ctx.options)
            pairs = [
                (v, g) for v, g in zip(result, grad_outputs) if g is not None and v.requires_grad
            ]
            parameters = [x for x, need in zip(live, required) if need]
            gradients = iter(
                torch.autograd.grad(
                    [v for v, _ in pairs], parameters, [g for _, g in pairs], allow_unused=True
                )
            )
        return (*[next(gradients) if need else None for need in required], None)


def joint(values, previous_pre, gamma, mixes, *, pending=None, broadcast=False):
    empty = values.new_empty(0)
    operands = (empty, empty, empty, empty) if pending is None else pending
    options = (
        mixes.norm_eps,
        mixes.compute_h_eps,
        mixes.sinkhorn_eps,
        mixes.sinkhorn_iterations,
        broadcast,
        pending is not None,
    )
    scale = torch.cat((mixes.alpha_pre, mixes.alpha_post, mixes.alpha_res))
    return _Joint.apply(
        values,
        previous_pre,
        gamma,
        mixes.mapping_proj.weight,
        scale,
        mixes.bias,
        *operands,
        options,
    )


class _Post(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, residual, post, comb):
        if not residual.is_cuda:
            raise RuntimeError("Deployment mHC post requires CUDA")
        from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang

        ctx.save_for_backward(output, residual, post, comb)
        copies, width = residual.shape[-2:]
        return mhc_post_tilelang(
            output.reshape(-1, width).contiguous(),
            residual.reshape(-1, copies, width).contiguous(),
            post.reshape(-1, copies, 1).contiguous(),
            comb.reshape(-1, copies, copies).contiguous(),
        ).reshape_as(residual)

    @staticmethod
    def backward(ctx, grad):
        output, residual, post, comb = ctx.saved_tensors
        with torch.autocast(device_type=grad.device.type, enabled=False):
            g = grad.float()
            dy = (g * post.float().unsqueeze(-1)).sum(-2)
            dh = torch.einsum("...ij,...jd->...id", comb.float(), g)
            dp = (g * output.float().unsqueeze(-2)).sum(-1)
            dc = torch.einsum("...id,...jd->...ij", residual.float(), g)
        return dy.to(output.dtype), dh.to(residual.dtype), dp.to(post.dtype), dc.to(comb.dtype)


def post_mix(output, residual, post, comb):
    return _Post.apply(output, residual, post, comb)


class _Collapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, pre):
        if not values.is_cuda:
            raise RuntimeError("Deployment mHC collapse requires CUDA")
        from vllm.model_executor.kernels.batch_invariant import hc_collapse

        ctx.save_for_backward(values, pre)
        copies, width = values.shape[-2:]
        return hc_collapse(
            values.reshape(-1, copies, width).contiguous(), pre.reshape(-1, copies).contiguous()
        ).reshape(*values.shape[:-2], width)

    @staticmethod
    def backward(ctx, grad):
        values, pre = ctx.saved_tensors
        with torch.autocast(device_type=grad.device.type, enabled=False):
            g = grad.float().unsqueeze(-2)
            dh, dp = g * pre.float().unsqueeze(-1), (g * values.float()).sum(-1)
        return dh.to(values.dtype), dp.to(pre.dtype)


def collapse(values, pre):
    return _Collapse.apply(values, pre)
