# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA TP1 row-reduction kernels for chunked LM-head cross entropy.

The kernels deliberately keep probability math local to a vocab tile.  Their
only FP32 workspace is ``[tokens, ceil(vocab / block)]`` partial reductions,
never a vocab-shaped FP32 tensor.  This is private to the chunked-head
primitive: TP and CPU callers retain the established PyTorch implementation.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the PyTorch fallback.
    triton = None
    tl = None


ROW_BLOCK = 4096


def is_available() -> bool:
    return triton is not None


def workspace_shape(
    rows: int, vocab: int, block: int = ROW_BLOCK
) -> tuple[tuple[int, int], tuple[int]]:
    """Return the bounded FP32 partial and row workspaces for a CE window."""
    if rows < 0 or vocab <= 0 or block <= 0:
        raise ValueError("rows must be non-negative and vocab/block must be positive")
    return (rows, (vocab + block - 1) // block), (rows,)


if triton is not None:

    @triton.jit
    def _partial_max_kernel(
        logits, partial, vocab: tl.constexpr, tiles: tl.constexpr, BLOCK: tl.constexpr
    ):
        program = tl.program_id(0)
        row = (program // tiles).to(tl.int64)
        tile = program % tiles
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(logits + row * vocab + columns, mask=columns < vocab, other=-float("inf"))
        tl.store(partial + row * tiles + tile, tl.max(values.to(tl.float32), axis=0))

    @triton.jit
    def _final_max_kernel(partial, row_max, tiles: tl.constexpr, REDUCE_BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, REDUCE_BLOCK)
        values = tl.load(partial + row * tiles + offsets, mask=offsets < tiles, other=-float("inf"))
        tl.store(row_max + row, tl.max(values, axis=0))

    @triton.jit
    def _partial_sum_kernel(
        logits, row_max, partial, vocab: tl.constexpr, tiles: tl.constexpr, BLOCK: tl.constexpr
    ):
        program = tl.program_id(0)
        row = (program // tiles).to(tl.int64)
        tile = program % tiles
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(
            logits + row * vocab + columns, mask=columns < vocab, other=-float("inf")
        ).to(tl.float32)
        values -= tl.load(row_max + row)
        exp_values = tl.exp(values)
        tl.store(partial + row * tiles + tile, tl.sum(exp_values, axis=0))

    @triton.jit
    def _final_sum_kernel(partial, row_sum, tiles: tl.constexpr, REDUCE_BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, REDUCE_BLOCK)
        values = tl.load(partial + row * tiles + offsets, mask=offsets < tiles, other=0.0)
        tl.store(row_sum + row, tl.sum(values, axis=0))

    @triton.jit
    def _final_loss_kernel(logits, target, row_max, row_sum, loss, vocab: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        target_column = tl.load(target + row)
        target_logit = tl.load(logits + row * vocab + target_column).to(tl.float32)
        tl.store(loss + row, tl.log(tl.load(row_sum + row)) + tl.load(row_max + row) - target_logit)

    @triton.jit
    def _inplace_gradient_kernel(
        logits,
        target,
        grad_output,
        row_max,
        row_sum,
        temperature,
        vocab: tl.constexpr,
        tiles: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        program = tl.program_id(0)
        row = (program // tiles).to(tl.int64)
        tile = program % tiles
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(
            logits + row * vocab + columns, mask=columns < vocab, other=-float("inf")
        ).to(tl.float32)
        probabilities = tl.exp(values - tl.load(row_max + row)) / tl.load(row_sum + row)
        target_column = tl.load(target + row)
        gradient = probabilities - (columns == target_column).to(tl.float32)
        gradient *= tl.load(grad_output + row).to(tl.float32) / temperature
        tl.store(logits + row * vocab + columns, gradient, mask=columns < vocab)


def _workspaces(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    rows, vocab = logits.shape
    partial_shape, row_shape = workspace_shape(rows, vocab)
    partial = torch.empty(partial_shape, dtype=torch.float32, device=logits.device)
    row_values = torch.empty(row_shape, dtype=torch.float32, device=logits.device)
    return partial, row_values, vocab, partial_shape[1]


def _row_statistics(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return bounded workspaces containing row max and exp sums."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    partial, row_max, vocab, tiles = _workspaces(logits)
    rows = logits.shape[0]
    reduce_block = triton.next_power_of_2(tiles)
    _partial_max_kernel[(rows * tiles,)](logits, partial, vocab=vocab, tiles=tiles, BLOCK=ROW_BLOCK)
    _final_max_kernel[(rows,)](partial, row_max, tiles=tiles, REDUCE_BLOCK=reduce_block)
    _partial_sum_kernel[(rows * tiles,)](
        logits, row_max, partial, vocab=vocab, tiles=tiles, BLOCK=ROW_BLOCK
    )
    row_sum = torch.empty_like(row_max)
    _final_sum_kernel[(rows,)](partial, row_sum, tiles=tiles, REDUCE_BLOCK=reduce_block)
    return partial, row_max, row_sum, tiles


def token_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute TP1 token loss without a vocab-shaped FP32 materialization."""
    _partial_sum, row_max, row_sum, _tiles = _row_statistics(logits)
    loss = torch.empty(target.numel(), dtype=torch.float32, device=logits.device)
    _final_loss_kernel[(target.numel(),)](
        logits, target.reshape(-1), row_max, row_sum, loss, vocab=logits.shape[1]
    )
    return loss


def inplace_gradient(
    logits: torch.Tensor, target: torch.Tensor, grad_output: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Overwrite BF16 logits with BF16 dlogits, keeping FP32 math tile-local."""
    _partial_sum, row_max, row_sum, tiles = _row_statistics(logits)
    _inplace_gradient_kernel[(logits.shape[0] * tiles,)](
        logits,
        target.reshape(-1),
        grad_output.reshape(-1),
        row_max,
        row_sum,
        temperature,
        vocab=logits.shape[1],
        tiles=tiles,
        BLOCK=ROW_BLOCK,
    )
    return logits


__all__ = ["is_available", "inplace_gradient", "token_loss", "workspace_shape"]
