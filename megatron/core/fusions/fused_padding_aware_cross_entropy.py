# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused (vocab-parallel) cross entropy over the live rows of a THD buffer.

The LM head of a padding-aware model produces logits for the capacity ``[rows, V]`` with zeros in
the padding tail; the loss over that tail is discarded by the loss mask, but the native fused
cross entropy still spends several fp32 passes over all rows (an fp32 copy of the logits, max,
exp, sum, the gradient). This module computes the loss with one Triton pass per row and skips
rows ``>= m_valid`` (a device int64 read in-kernel, so the kernels are CUDA-graph capturable);
the backward recomputes the softmax from the bf16 logits (no fp32 copy is kept) and zero-fills
the gradient tail. Numerics: fp32 log-sum-exp per row, like the native path.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from megatron.core.fusions.padding_aware_switches import padding_aware_feature_disabled
from megatron.core.tensor_parallel.utils import VocabUtility

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

logger = logging.getLogger(__name__)

_MAX_BLOCK = 16384

if HAVE_TRITON:

    @triton.jit
    def _pa_ce_fwd_kernel(
        X, X_row_stride, Y, STATS, m_valid_ptr, vocab_start, V, tail_s, BLOCK: tl.constexpr
    ):
        """Per row: local max ``m``, ``s = sum exp(x - m)`` and the predicted logit ``p`` (0 if
        the target lies in another vocab partition); rows beyond the live bound get
        (0, tail_s = 1 / world, 0) so that their combined loss is exactly 0."""
        row = tl.program_id(0)
        m_valid = tl.load(m_valid_ptr)
        if row >= m_valid:
            tl.store(STATS + 3 * row, 0.0)
            tl.store(STATS + 3 * row + 1, tail_s)
            tl.store(STATS + 3 * row + 2, 0.0)
            return
        x_ptr = X + row.to(tl.int64) * X_row_stride
        local_y = tl.load(Y + row) - vocab_start
        m = float("-inf")
        s = 0.0
        p = 0.0
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < V
            x = tl.load(x_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
            block_max = tl.max(x, 0)
            new_m = tl.maximum(m, block_max)
            s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), 0)
            m = new_m
            p += tl.sum(tl.where((cols == local_y) & mask, x, 0.0), 0)
        tl.store(STATS + 3 * row, m)
        tl.store(STATS + 3 * row + 1, s)
        tl.store(STATS + 3 * row + 2, p)

    @triton.jit
    def _pa_ce_bwd_kernel(
        X,
        X_row_stride,
        Y,
        GMAX_S,
        G,
        DX,
        DX_row_stride,
        m_valid_ptr,
        vocab_start,
        V,
        BLOCK: tl.constexpr,
    ):
        """dX = (exp(x - gmax) / S - onehot(target)) * dloss for live rows; zeros for the tail."""
        row = tl.program_id(0)
        m_valid = tl.load(m_valid_ptr)
        dx_ptr = DX + row.to(tl.int64) * DX_row_stride
        if row >= m_valid:
            for off in range(0, V, BLOCK):
                cols = off + tl.arange(0, BLOCK)
                tl.store(
                    dx_ptr + cols,
                    tl.zeros([BLOCK], dtype=tl.float32).to(DX.dtype.element_ty),
                    mask=cols < V,
                )
            return
        x_ptr = X + row.to(tl.int64) * X_row_stride
        local_y = tl.load(Y + row) - vocab_start
        gmax = tl.load(GMAX_S + 2 * row)
        inv_s = 1.0 / tl.load(GMAX_S + 2 * row + 1)
        g = tl.load(G + row)
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < V
            x = tl.load(x_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
            prob = tl.exp(x - gmax) * inv_s
            onehot = tl.where(cols == local_y, 1.0, 0.0)
            tl.store(dx_ptr + cols, ((prob - onehot) * g).to(DX.dtype.element_ty), mask=mask)


def _block_and_warps(vocab: int):
    block = min(_MAX_BLOCK, triton.next_power_of_2(vocab))
    return block, max(4, min(32, block // 512))


class PaddingAwareCrossEntropy(torch.autograd.Function):
    """Vocab-parallel cross entropy over the first ``m_valid`` rows of ``[rows, V]`` logits."""

    @staticmethod
    def forward(ctx, logits2d, target1d, m_valid, tp_group):
        """Per-row stats over the live rows, combined across the vocab partitions."""
        rows, vocab = logits2d.shape
        world = 1 if tp_group is None else torch.distributed.get_world_size(tp_group)
        rank = 0 if tp_group is None else torch.distributed.get_rank(tp_group)
        vocab_start, _ = VocabUtility.vocab_range_from_per_partition_vocab_size(vocab, rank, world)
        block, warps = _block_and_warps(vocab)
        stats = torch.empty((rows, 3), dtype=torch.float32, device=logits2d.device)
        _pa_ce_fwd_kernel[(rows,)](
            logits2d,
            logits2d.stride(0),
            target1d,
            stats,
            m_valid,
            vocab_start,
            vocab,
            1.0 / world,
            BLOCK=block,
            num_warps=warps,
        )
        if world > 1:
            gathered = torch.empty((world * rows, 3), dtype=torch.float32, device=stats.device)
            torch.distributed.all_gather_into_tensor(gathered, stats, group=tp_group)
            gathered = gathered.view(world, rows, 3)
            gmax = gathered[:, :, 0].amax(dim=0)
            s = (gathered[:, :, 1] * torch.exp(gathered[:, :, 0] - gmax)).sum(dim=0)
            p = gathered[:, :, 2].sum(dim=0)
        else:
            gmax, s, p = stats[:, 0], stats[:, 1], stats[:, 2]
        loss = torch.log(s) + gmax - p
        gmax_s = torch.stack((gmax, s), dim=1).contiguous()
        ctx.save_for_backward(logits2d, target1d, gmax_s, m_valid)
        ctx.vocab_start = vocab_start
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        """Softmax minus one-hot, recomputed from the saved logits; zero tail."""
        logits2d, target1d, gmax_s, m_valid = ctx.saved_tensors
        rows, vocab = logits2d.shape
        block, warps = _block_and_warps(vocab)
        grad = torch.empty_like(logits2d)
        _pa_ce_bwd_kernel[(rows,)](
            logits2d,
            logits2d.stride(0),
            target1d,
            gmax_s,
            grad_loss.contiguous().to(torch.float32),
            grad,
            grad.stride(0),
            m_valid,
            ctx.vocab_start,
            vocab,
            BLOCK=block,
            num_warps=warps,
        )
        return grad, None, None, None


_logged = [False]
_full_rows_cache: dict = {}


def full_rows_bound(rows: int, device) -> torch.Tensor:
    """A cached device int64 ``(1,)`` holding ``rows``: the bound to use when no padding-aware
    live-row bound is published (every row is live; static address, graph safe)."""
    key = (rows, str(device))
    t = _full_rows_cache.get(key)
    if t is None:
        t = torch.full((1,), rows, dtype=torch.int64, device=device)
        _full_rows_cache[key] = t
    return t


def triton_cross_entropy_available(logits: torch.Tensor, target: torch.Tensor) -> bool:
    """Structural check: Triton, CUDA bf16/fp32 logits ``[s, b, V]`` with a unit last stride."""
    return (
        HAVE_TRITON
        and logits.is_cuda
        and logits.dim() == 3
        and logits.stride(-1) == 1
        and logits.dtype in (torch.bfloat16, torch.float32)
        and target.dtype == torch.int64
    )


def padding_aware_cross_entropy_available(logits: torch.Tensor, target: torch.Tensor) -> bool:
    """``triton_cross_entropy_available`` plus the ``MCORE_PADDING_AWARE_SKIP=ce`` kill-switch
    for the row-bounded use."""
    return triton_cross_entropy_available(logits, target) and not padding_aware_feature_disabled(
        "ce"
    )


def triton_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, tp_group, live_rows: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``cross_entropy_fusion_impl='triton'``: loss ``[s, b]`` (fp32) of ``logits [s, b, V]``
    against ``target [s, b]``; over the live rows when ``live_rows`` (device int64 ``(1,)``) is
    given, over every row otherwise."""
    s, b, _ = logits.shape
    if live_rows is None:
        live_rows = full_rows_bound(s * b, logits.device)
    return padding_aware_cross_entropy(logits, target, live_rows, tp_group)


def padding_aware_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, m_valid: torch.Tensor, tp_group
) -> torch.Tensor:
    """Loss ``[s, b]`` (fp32) of ``logits [s, b, V]`` against ``target [s, b]`` over the live rows.

    ``m_valid`` is the device int64 ``(1,)`` live-row bound of the ``s * b`` flattened rows.
    """
    s, b, vocab = logits.shape
    logits2d = logits.reshape(s * b, vocab)
    if logits2d.stride(1) != 1:
        logits2d = logits2d.contiguous()
    target1d = target.reshape(-1).contiguous()
    if not _logged[0]:
        _logged[0] = True
        logger.info("padding-aware cross entropy engaged for logits %s", tuple(logits.shape))
    loss = PaddingAwareCrossEntropy.apply(logits2d, target1d, m_valid, tp_group)
    return loss.view(s, b)
