# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pinned BI linear forwards with the W9-b5 decoded-operand VJPs."""
from types import SimpleNamespace

import torch

from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8, quantize_block_fp8


def _row_scales(scale, rows, inner):
    tiles, groups = (rows + 127) // 128, inner // 32
    ktiles = (groups + 3) // 4
    return (
        scale.view(tiles, ktiles, 32, 4, 4)
        .transpose(1, 3)
        .contiguous()
        .view(tiles * 128, ktiles * 4)[:rows, :groups]
    )


def tc_mm(a, b, out_dtype=None):
    """BF16 tensor-core operands with optional FP32 output; CPU is a reference."""
    a, b = a.bfloat16(), b.bfloat16()
    if not a.is_cuda:
        out = a.float() @ b.float()
        return out.bfloat16() if out_dtype is None else out.to(out_dtype)
    return a @ b if out_dtype is None else torch.mm(a, b, out_dtype=out_dtype)


def _like(weight):
    """What a placeholder gradient needs, without keeping a stacked weight alive."""
    return SimpleNamespace(shape=weight.shape, dtype=weight.dtype, device=weight.device)


def accumulate_wgrad(owners, dw, like):
    """Weight gradient for the forward input ``like`` whose rows stack ``owners``.

    When every owner has an FP32 ``main_grad`` (DDP), the FP32 ``dw`` is added there
    and DDP's placeholder gradient is returned, as Megatron's gradient-accumulation
    fusion does; otherwise ``dw`` itself in ``like``'s dtype."""
    if not owners or any(
        getattr(o, "main_grad", None) is None or o.main_grad.dtype != torch.float32 for o in owners
    ):
        return dw.to(like.dtype)
    from megatron.lite.primitive.parallel.linear import _dummy_wgrad

    start = 0
    for owner in owners:
        rows = owner.shape[0] if len(owners) > 1 else dw.shape[0]
        owner.main_grad.add_(dw[start : start + rows].reshape(owner.main_grad.shape))
        owner.grad_added_to_main_grad = True
        start += rows
    return _dummy_wgrad(like)


class _MXFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, activation, owners=None, row_perm=None):
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError("Deployment MXFP8 linear requires CUDA")
        ctx.owners, ctx.like, ctx.row_perm = owners, _like(weight), row_perm
        from vllm.model_executor.kernels.linear.mxfp8.batch_invariant import _uses_tensor_core
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
            swizzle_mxfp8_scale,
        )

        flat = x.reshape(-1, x.shape[-1])
        ctx.input_shape, ctx.weight_dtype = x.shape, weight.dtype
        if not flat.shape[0]:
            ctx.save_for_backward(flat, weight.to(x.dtype))
            return x.new_empty((*x.shape[:-1], weight.shape[0]))
        tc = _uses_tensor_core(tuple(weight.shape))
        if activation is None:
            codes, scales = mxfp8_e4m3_quantize(flat, is_sf_swizzled_layout=tc)
        else:
            if not tc:
                raise ValueError("Swizzled prequant requires the tensor-core consumer")
            codes, scales = activation.data.reshape(flat.shape), activation.scale
        raw = _row_scales(scales, *flat.shape) if tc else scales
        packed, weight_scales = quantize_block_fp8(weight, (32, 32), scale_format="e8m0")
        row_scales = weight_scales.view(torch.uint8).repeat_interleave(32, dim=0)
        kernel_codes = packed
        if row_perm is not None:
            # vLLM gathers whole rows of the loaded per-row-scale weight
            # (permute on load): quantize first, then permute codes and scales.
            kernel_codes = packed.view(torch.uint8).index_select(0, row_perm).view(packed.dtype)
            row_scales = row_scales.index_select(0, row_perm)
        if tc:
            row_scales = swizzle_mxfp8_scale(row_scales, *weight.shape)
            output = torch.ops.vllm.bi_pinned_mxfp8_mm(
                codes, kernel_codes.t(), scales, row_scales, x.dtype
            )
        else:
            from vllm.model_executor.kernels.batch_invariant import block32_scaled_mm

            output = block32_scaled_mm(codes, scales, kernel_codes, row_scales, x.dtype)
        ctx.compact, ctx.input_dtype = True, x.dtype
        ctx.save_for_backward(codes, raw, packed, weight_scales)
        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        if getattr(ctx, 'compact', False):
            codes, raw, packed, scales = ctx.saved_tensors
            activation = (
                (
                    codes.float().reshape(codes.shape[0], -1, 32)
                    * torch.exp2(raw.float() - 127).unsqueeze(-1)
                )
                .reshape(codes.shape)
                .bfloat16()
                .to(ctx.input_dtype)
            )
            weight = dequantize_block_fp8(packed, scales, (32, 32)).to(ctx.input_dtype)
        else:
            activation, weight = ctx.saved_tensors
        perm = getattr(ctx, 'row_perm', None)
        if perm is not None:
            # Output column j is natural row perm[j]: un-permute the gradient once.
            flat = grad.reshape(-1, grad.shape[-1])
            grad = torch.empty_like(flat).index_copy_(1, perm, flat)
        dx, dw, _ = linear_vjp(ctx, grad, activation, weight)
        return dx, dw, None, None, None


def linear_vjp(ctx, grad, activation, weight):
    flat_grad = grad.reshape(-1, weight.shape[0])
    with torch.autocast(device_type=grad.device.type, enabled=False):
        dx = tc_mm(flat_grad, weight).reshape(ctx.input_shape).to(activation.dtype)
        dw = tc_mm(flat_grad.T, activation, torch.float32)
    owners = getattr(ctx, 'owners', None)
    if owners is None:
        return dx, dw.to(ctx.weight_dtype), None
    return dx, accumulate_wgrad(owners, dw, ctx.like), None


class _BF16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, owners=None):
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError("Deployment BF16 linear requires CUDA")
        ctx.owners, ctx.like = owners, _like(weight)
        from vllm.model_executor.kernels.linear.cute_dsl.batch_invariant import bf16_fixed_mm

        compute = weight.bfloat16()
        ctx.save_for_backward(x, compute)
        ctx.weight_dtype = weight.dtype
        if not x.reshape(-1, x.shape[-1]).shape[0]:
            return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=torch.float32)
        out = bf16_fixed_mm(x.reshape(-1, x.shape[-1]).bfloat16(), compute)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x, compute = ctx.saved_tensors
        flat_grad = grad.reshape(-1, compute.shape[0])
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = tc_mm(flat_grad, compute).reshape(x.shape).to(x.dtype)
            dw = tc_mm(flat_grad.T, x.reshape(-1, compute.shape[1]), torch.float32)
        owners = getattr(ctx, 'owners', None)
        if owners is None:
            return dx, dw.to(ctx.weight_dtype), None
        return dx, accumulate_wgrad(owners, dw, ctx.like), None


def linear(x, weight, *, fp8, activation=None, row_perm=None):
    """BI GEMM ``x @ weight.T``. ``weight``: a tensor (gradient returned through
    autograd), or a tuple of parameters whose rows are stacked and whose FP32 dW goes
    straight to each owner's ``main_grad``. Pass a tuple only for parameters that have
    no other gradient path: DDP then skips their ``.grad``. ``row_perm`` (MXFP8 only)
    gathers output rows of the quantized weight, as vLLM's permute-on-load does."""
    owners = tuple(weight) if isinstance(weight, (tuple, list)) else None
    if owners is not None:
        weight = owners[0] if len(owners) == 1 else torch.cat(owners)
    if fp8:
        if x.shape[-1] != weight.shape[-1] or any(n % 32 for n in weight.shape):
            raise ValueError("MXFP8 requires matching K and 32-aligned weights")
        return _MXFP8Linear.apply(x, weight, activation, owners, row_perm)
    if row_perm is not None:
        raise ValueError("Output row permutation is only implemented for MXFP8")
    if activation is not None:
        raise ValueError("BF16 linear cannot consume MXFP8 prequant")
    return _BF16Linear.apply(x, weight, owners)
