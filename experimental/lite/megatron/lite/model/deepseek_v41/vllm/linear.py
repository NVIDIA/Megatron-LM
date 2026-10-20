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
        decoded_x = (
            (
                codes.float().reshape(flat.shape[0], -1, 32)
                * torch.exp2(raw.float() - 127).unsqueeze(-1)
            )
            .reshape(flat.shape)
            .bfloat16()
            .to(x.dtype)
        )
        packed, weight_scales = quantize_block_fp8(weight, (32, 32), scale_format="e8m0")
        decoded_w = dequantize_block_fp8(packed, weight_scales, (32, 32)).to(x.dtype)
        row_scales = weight_scales.view(torch.uint8).repeat_interleave(32, dim=0)
        if tc:
            row_scales = swizzle_mxfp8_scale(row_scales, *weight.shape)
            output = torch.ops.vllm.bi_pinned_mxfp8_mm(
                codes, packed.t(), scales, row_scales, x.dtype
            )
        else:
            from vllm.model_executor.kernels.batch_invariant import block32_scaled_mm

            output = block32_scaled_mm(codes, scales, packed, row_scales, x.dtype)
        ctx.save_for_backward(decoded_x, decoded_w)
        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        activation, weight = ctx.saved_tensors
        flat_grad = grad.reshape(-1, weight.shape[0]).float()
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = (flat_grad @ weight.float()).reshape(ctx.input_shape).to(activation.dtype)
            dw = (flat_grad.T @ activation.float()).to(ctx.weight_dtype)
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
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = (grad.float() @ compute.float()).to(x.dtype)
            dw = (
                grad.reshape(-1, compute.shape[0]).float().T
                @ x.reshape(-1, compute.shape[1]).float()
            )
        return dx, dw.to(ctx.weight_dtype)


def linear(x, weight, *, fp8, activation=None):
    if fp8:
        if x.shape[-1] != weight.shape[-1] or any(n % 32 for n in weight.shape):
            raise ValueError("MXFP8 requires matching K and 32-aligned weights")
        return _MXFP8Linear.apply(x, weight, activation)
    if activation is not None:
        raise ValueError("BF16 linear cannot consume MXFP8 prequant")
    return _BF16Linear.apply(x, weight)
