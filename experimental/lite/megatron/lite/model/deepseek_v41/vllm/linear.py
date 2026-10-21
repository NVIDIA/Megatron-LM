# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pinned BI linear forwards with the W9-b5 decoded-operand VJPs."""
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
    """``a @ b`` with BF16 operands on tensor cores and FP32 accumulation; the result is
    BF16 unless ``out_dtype`` asks for the FP32 accumulator. CPU (derivative references
    only): the same rounded operands in an FP32 GEMM."""
    a, b = a.bfloat16(), b.bfloat16()
    if not a.is_cuda:
        out = a.float() @ b.float()
        return out.bfloat16() if out_dtype is None else out.to(out_dtype)
    return a @ b if out_dtype is None else torch.mm(a, b, out_dtype=out_dtype)


class _MXFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, activation):
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError("Deployment MXFP8 linear requires CUDA")
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
        if tc:
            row_scales = swizzle_mxfp8_scale(row_scales, *weight.shape)
            output = torch.ops.vllm.bi_pinned_mxfp8_mm(
                codes, packed.t(), scales, row_scales, x.dtype
            )
        else:
            from vllm.model_executor.kernels.batch_invariant import block32_scaled_mm

            output = block32_scaled_mm(codes, scales, packed, row_scales, x.dtype)
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
        return linear_vjp(ctx, grad, activation, weight)


def linear_vjp(ctx, grad, activation, weight):
    flat_grad = grad.reshape(-1, weight.shape[0])
    with torch.autocast(device_type=grad.device.type, enabled=False):
        dx = tc_mm(flat_grad, weight).reshape(ctx.input_shape).to(activation.dtype)
        dw = tc_mm(flat_grad.T, activation, torch.float32).to(ctx.weight_dtype)
    return dx, dw, None


class _BF16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError("Deployment BF16 linear requires CUDA")
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
        return dx, dw.to(ctx.weight_dtype)


def linear(x, weight, *, fp8, activation=None):
    if fp8:
        if x.shape[-1] != weight.shape[-1] or any(n % 32 for n in weight.shape):
            raise ValueError("MXFP8 requires matching K and 32-aligned weights")
        return _MXFP8Linear.apply(x, weight, activation)
    if activation is not None:
        raise ValueError("BF16 linear cannot consume MXFP8 prequant")
    return _BF16Linear.apply(x, weight)
