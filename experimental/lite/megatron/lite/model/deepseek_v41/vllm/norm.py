# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""BI RMS forward and owned row-wise RMS derivative."""
import torch


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        if not x.is_cuda:
            raise RuntimeError("Deployment RMS norm requires CUDA")
        from vllm.model_executor.kernels.batch_invariant import rms_norm_cuda

        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return rms_norm_cuda(
            x.reshape(-1, x.shape[-1]), weight.bfloat16().to(x.dtype), eps
        ).reshape_as(x)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        with torch.autocast(device_type=x.device.type, enabled=False):
            values = x.float()
            inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + ctx.eps)
            scaled = grad.float() * weight.bfloat16().float()
            dx = inverse * (
                scaled - values * inverse.square() * (scaled * values).mean(-1, keepdim=True)
            )
            dw = (grad.float() * values * inverse).reshape(-1, x.shape[-1]).sum(0)
        return dx.to(x.dtype), dw.to(weight.dtype), None


def rms_norm(x, weight, eps):
    return _RMSNorm.apply(x, weight, eps)
