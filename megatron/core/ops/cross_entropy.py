# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Ported from https://github.com/NVIDIA-NeMo/nv-mistralai-megatron
# Revision: f1856ec3ef887125dc7ad81f312c9ec837a3bb11
# Private-logits revision: daa90bd2d26fb44f34b769961c1a1f94a53d6c35
# License: BSD-3-Clause (see the source repository LICENSE).

"""Memory-efficient TP1 cross entropy with FP32 normalization and BF16/FP32 logits.

The forward pass retains logits and two FP32 normalizers per token rather
than a vocabulary-sized FP32 softmax. Backward reconstructs probabilities in
FP32 and writes gradients directly in the input dtype. Vocabulary tiles keep
the 262k-vocabulary case within practical register/shared-memory limits.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_normalizers(
    X,
    MAX,
    SUM,
    ROWS: tl.constexpr,
    VOCAB: tl.constexpr,
    CHUNKS: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BV + tl.arange(0, BV)
    values = tl.load(
        X + rows[:, None].to(tl.int64) * VOCAB + cols[None, :],
        (rows[:, None] < ROWS) & (cols[None, :] < VOCAB),
        other=-float("inf"),
    ).to(tl.float32)
    maxima = tl.max(values, axis=1)
    sums = tl.sum(tl.exp(values - maxima[:, None]), axis=1)
    offsets = rows * CHUNKS + tl.program_id(1)
    tl.store(MAX + offsets, maxima, rows < ROWS)
    tl.store(SUM + offsets, sums, rows < ROWS)


@triton.jit
def _finish_loss(
    X,
    TARGET,
    MAX,
    SUM,
    ROW_MAX,
    INV_SUM,
    LOSS,
    ROWS: tl.constexpr,
    VOCAB: tl.constexpr,
    CHUNKS: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    chunks = tl.arange(0, BC)
    mask = (rows[:, None] < ROWS) & (chunks[None, :] < CHUNKS)
    offsets = rows[:, None] * CHUNKS + chunks[None, :]
    maxima = tl.load(MAX + offsets, mask, other=-float("inf"))
    sums = tl.load(SUM + offsets, mask, other=0.0)
    global_max = tl.max(maxima, axis=1)
    normalizer = tl.sum(sums * tl.exp(maxima - global_max[:, None]), axis=1)
    target = tl.load(TARGET + rows, rows < ROWS, other=0)
    valid = (target >= 0) & (target < VOCAB)
    picked = tl.load(X + rows.to(tl.int64) * VOCAB + target, (rows < ROWS) & valid, other=0.0).to(
        tl.float32
    )
    # Match the TP1 reference's out-of-range-label behavior. Training normally
    # uses valid labels and masks padded tokens with the separate loss mask.
    picked = tl.where(valid, picked, global_max)
    tl.store(ROW_MAX + rows, global_max, rows < ROWS)
    tl.store(INV_SUM + rows, 1.0 / normalizer, rows < ROWS)
    loss = tl.where(target == -100, 0.0, tl.log(normalizer) - (picked - global_max))
    tl.store(LOSS + rows, loss, rows < ROWS)


@triton.jit
def _loss_backward(
    X,
    TARGET,
    ROW_MAX,
    INV_SUM,
    DLOSS,
    DX,
    ROWS: tl.constexpr,
    VOCAB: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BV + tl.arange(0, BV)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < VOCAB)
    offsets = rows[:, None].to(tl.int64) * VOCAB + cols[None, :]
    values = tl.load(X + offsets, mask, other=-float("inf")).to(tl.float32)
    maxima = tl.load(ROW_MAX + rows, rows < ROWS, other=0.0)
    inv_sum = tl.load(INV_SUM + rows, rows < ROWS, other=0.0)
    target = tl.load(TARGET + rows, rows < ROWS, other=-1)
    dloss = tl.load(DLOSS + rows, rows < ROWS, other=0.0).to(tl.float32)
    dloss = tl.where(target == -100, 0.0, dloss)
    grad = tl.exp(values - maxima[:, None]) * inv_sum[:, None]
    grad -= (cols[None, :] == target[:, None]).to(tl.float32)
    grad *= dloss[:, None]
    tl.store(DX + offsets, grad, mask)


class _TiledCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, inplace_backward):
        vocab = logits.shape[-1]
        rows = target.numel()
        chunks = triton.cdiv(vocab, 4096)
        flat_logits = logits.reshape(rows, vocab)
        flat_target = target.reshape(rows).contiguous()
        maxima = torch.empty((rows, chunks), device=logits.device, dtype=torch.float32)
        sums = torch.empty_like(maxima)
        row_max = torch.empty(rows, device=logits.device, dtype=torch.float32)
        inv_sum = torch.empty_like(row_max)
        loss = torch.empty_like(row_max)
        _partial_normalizers[(triton.cdiv(rows, 4), chunks)](
            flat_logits, maxima, sums, rows, vocab, chunks, 4, 4096, num_warps=8
        )
        _finish_loss[(triton.cdiv(rows, 32),)](
            flat_logits,
            flat_target,
            maxima,
            sums,
            row_max,
            inv_sum,
            loss,
            rows,
            vocab,
            chunks,
            32,
            triton.next_power_of_2(chunks),
            num_warps=4,
        )
        ctx.save_for_backward(flat_logits, flat_target, row_max, inv_sum)
        ctx.input_shape = logits.shape
        ctx.inplace_backward = inplace_backward
        ctx.consumed = False
        return loss.view_as(target)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.inplace_backward and ctx.consumed:
            raise RuntimeError("In-place cross entropy logits buffer was already consumed")
        logits, target, row_max, inv_sum = ctx.saved_tensors
        rows, vocab = logits.shape
        ctx.consumed = ctx.inplace_backward
        grad = logits if ctx.inplace_backward else torch.empty_like(logits)
        _loss_backward[(triton.cdiv(rows, 4), triton.cdiv(vocab, 4096))](
            logits,
            target,
            row_max,
            inv_sum,
            grad_output.contiguous(),
            grad,
            rows,
            vocab,
            4,
            4096,
            num_warps=8,
        )
        if ctx.inplace_backward:
            torch.autograd.graph.increment_version(logits)
        return grad.view(ctx.input_shape), None, None


def triton_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
    *,
    inplace_backward: bool = False,
) -> torch.Tensor:
    """Compute unreduced TP1 cross entropy while retaining no FP32 softmax.

    Args:
        logits: Contiguous CUDA logits, vocabulary in the final dimension.
        target: Int64 labels with shape equal to ``logits.shape[:-1]``.
            Rows labeled -100 have zero loss/gradient, matching TE's default ignore index.
        tp_group: Explicit tensor-parallel group; must contain one rank.
        inplace_backward: Explicitly transfer privately owned logits to a single
            backward, which overwrites their storage with gradients. No other
            backward consumer or later reader may use them. Forward preserves logits.

    Returns:
        FP32 per-token losses with the same shape as ``target``.
    """
    if tp_group.size() != 1:
        raise ValueError("Tiled Triton cross entropy currently requires TP1")
    if not logits.is_cuda or not logits.is_contiguous():
        raise ValueError("Tiled Triton cross entropy requires contiguous CUDA logits")
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Unsupported logits dtype")
    if target.dtype != torch.int64 or target.device != logits.device:
        raise ValueError("Targets must be int64 on the logits device")
    if logits.shape[:-1] != target.shape or not target.numel() or not logits.shape[-1]:
        raise ValueError("Nonempty logits and target shapes must match")
    return _TiledCrossEntropy.apply(logits, target, inplace_backward)
