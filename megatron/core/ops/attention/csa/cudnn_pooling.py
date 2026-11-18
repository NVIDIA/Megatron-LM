# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cuDNN ratio-2 pooling for CSA2's BF16 projections."""

import logging
from functools import partial

import torch

from megatron.core.ops._backends import require

logger = logging.getLogger(__name__)


class _CudnnR2Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, gate, ape, cu, compressed_cu, forward_fn, backward_fn):
        """Run the frontend with a constant zero positional bias."""
        ctx.backward_fn = backward_fn
        ctx.save_for_backward(kv, gate, ape, cu, compressed_cu)
        return forward_fn(
            kv,
            gate,
            ape,
            cu,
            compressed_cu,
            ratio=2,
            head_dim=kv.shape[-1],
            coff=1,
            total_comp=kv.shape[0] // 2,
        )["out"]

    @staticmethod
    def backward(ctx, grad_output):
        """Keep the two projection gradients; the positional bias is constant."""
        kv, gate, ape, cu, compressed_cu = ctx.saved_tensors
        grad_kv, grad_gate, _ = ctx.backward_fn(
            kv,
            gate,
            ape,
            cu,
            compressed_cu,
            grad_output.contiguous(),
            ratio=2,
            head_dim=kv.shape[-1],
            coff=1,
        )
        return grad_kv, grad_gate, None, None, None, None, None


def _pool_r2(kv, gate, dtype, valid_groups=None, *, forward_fn, backward_fn):
    """Return fused BF16 pooling, or None when the native path is required.

    Batch/channel columns pool independently, so flattening them avoids a
    sequence/batch transpose. Softmax, products and accumulation stay FP32.
    The frontend also computes an unused atomic dAPE; strict determinism keeps
    the native path. Warm up each shape before CUDA Graph capture.
    """
    if (
        not kv.is_cuda
        or kv.dtype != torch.bfloat16
        or gate.dtype != torch.bfloat16
        or dtype != torch.bfloat16
        or kv.numel() == 0
        or torch.are_deterministic_algorithms_enabled()
        or torch.compiler.is_compiling()
    ):
        return None
    if torch.cuda.get_device_capability(kv.device)[0] < 10:
        return None
    total, batch, dim = kv.shape
    if total % 2 or kv.numel() >= 2**31:
        return None
    if valid_groups is not None:
        invalid = ~valid_groups[:, None, None, None]
        kv = kv.reshape(-1, 2, batch, dim).masked_fill(invalid, 0)
        gate = gate.reshape(-1, 2, batch, dim).masked_fill(invalid, 0)
    kv = kv.reshape(total, batch * dim).contiguous()
    gate = gate.reshape(total, batch * dim).contiguous()
    cu = torch.arange(2, device=kv.device, dtype=torch.int32) * total
    compressed_cu = cu // 2
    ape = torch.zeros((2, batch * dim), device=kv.device, dtype=torch.float32)
    pooled = _CudnnR2Pool.apply(kv, gate, ape, cu, compressed_cu, forward_fn, backward_fn)
    pooled = pooled.reshape(total // 2, batch, dim)
    if valid_groups is not None:
        pooled = pooled.masked_fill(~valid_groups[:, None, None], 0)
    return pooled


def get_cudnn_r2_pool():
    """Bind the optional frontend once at compressor construction."""
    try:
        frontend = require(
            "cudnn.csa.compressor",
            "csa_compressor_forward_wrapper",
            "csa_compressor_backward_wrapper",
            needed_by="CSA2 fused ratio-2 pooling",
        )
    except ImportError as exc:
        logger.warning("CSA2 cuDNN pooling unavailable; keeping native pooling: %s", exc)
        return None
    return partial(
        _pool_r2,
        forward_fn=frontend.csa_compressor_forward_wrapper,
        backward_fn=frontend.csa_compressor_backward_wrapper,
    )
