# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Embedding weight gradient accumulated straight into ``main_grad``.

PyTorch's embedding backward builds a dense ``[num_embeddings, hidden]`` weight gradient
(zero-fill, then the rows of the tokens that occur), and Megatron's DDP hook then adds the whole
tensor into ``main_grad``. For a 129280 x 7168 vocabulary that is about 1 ms per embedding use,
nearly all of it on rows of tokens that do not occur. ``accumulate_embedding_wgrad`` adds only
the rows of the tokens that occur, with the same arithmetic, so ``main_grad`` gets the same
values:

* PyTorch's embedding backward (``embedding_dense_backward_cuda``, sort-based for more than 3072
  tokens) sorts the token ids together with their positions, sums each token's gradient rows in
  position order into fp32 partial sums of up to 10 rows (``NROWS_PER_THREAD``), adds the
  partial sums in order starting from 0, and rounds once to the gradient dtype.
* ``main_grad.add_(grad)`` adds that in fp32 and rounds to ``main_grad``'s dtype.

Rows of tokens that do not occur are left untouched instead of having 0 added to them.
"""

from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    # tl.sort and the reverse associative scan need Triton 3.
    HAVE_TRITON = version.parse(triton.__version__) >= version.parse("3.0.0")
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

# PyTorch's embedding backward uses another kernel, with another summation order, for up to
# 3072 tokens (embedding_backward_feature_kernel).
MIN_TOKENS = 3073
# The tokens are sorted by a single Triton program.
MAX_TOKENS = 8192
# Rows per fp32 partial sum (NROWS_PER_THREAD of PyTorch's embedding_backward_cuda_kernel).
_NROWS = 10
_BLOCK_H = 1024


@triton.jit
def _max_fn(a, b):
    return tl.maximum(a, b)


@triton.jit
def _min_fn(a, b):
    return tl.minimum(a, b)


@triton.jit
def _sort_tokens_kernel(
    ids_ptr,
    sorted_ids_ptr,
    positions_ptr,
    run_start_ptr,
    run_end_ptr,
    long_runs_ptr,
    n,
    POS_BITS: tl.constexpr,
    NROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sort (token, position) pairs, then give every sorted slot the first and last slot of its
    token's run and list the runs longer than NROWS (long_runs[0] is their count)."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    ids = tl.load(ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    key = tl.sort(tl.where(mask, (ids << POS_BITS) | offs, 0x7FFFFFFF))
    token = key >> POS_BITS
    tl.store(sorted_ids_ptr + offs, token, mask=mask)
    tl.store(positions_ptr + offs, key & ((1 << POS_BITS) - 1), mask=mask)
    tl.debug_barrier()
    prev = tl.load(sorted_ids_ptr + offs - 1, mask=mask & (offs > 0), other=-1)
    nxt = tl.load(sorted_ids_ptr + offs + 1, mask=(offs + 1) < n, other=-1)
    first = mask & ((offs == 0) | (token != prev))
    last = mask & ((offs == n - 1) | (token != nxt))
    start = tl.associative_scan(tl.where(first, offs, 0), 0, _max_fn)
    end = tl.associative_scan(tl.where(last, offs, BLOCK), 0, _min_fn, reverse=True)
    tl.store(run_start_ptr + offs, start, mask=mask)
    tl.store(run_end_ptr + offs, end, mask=mask)
    is_long = first & (end - start >= NROWS)
    tl.store(long_runs_ptr + tl.cumsum(is_long.to(tl.int32), 0), offs, mask=is_long)
    tl.store(long_runs_ptr, tl.sum(is_long.to(tl.int32), 0))


@triton.jit
def _partial_sums_kernel(
    sorted_ids_ptr,
    positions_ptr,
    run_start_ptr,
    run_end_ptr,
    grad_ptr,
    main_grad_ptr,
    partial_ptr,
    hidden,
    grad_stride,
    main_grad_stride,
    NROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Sum NROWS-row chunks of every run in fp32; a run of at most NROWS rows is finished here."""
    i = tl.program_id(0)
    start = tl.load(run_start_ptr + i)
    if (i - start) % NROWS == 0:
        end = tl.load(run_end_ptr + i)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        col_mask = cols < hidden
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for k in range(0, tl.minimum(NROWS, end - i + 1)):
            row = tl.load(positions_ptr + i + k).to(tl.int64)
            acc += tl.load(grad_ptr + row * grad_stride + cols, mask=col_mask, other=0.0).to(
                tl.float32
            )
        if end - start < NROWS:
            # The run's only partial sum: round it as the dense gradient row and add it.
            token = tl.load(sorted_ids_ptr + i).to(tl.int64)
            grad = (0.0 + acc).to(grad_ptr.dtype.element_ty)
            ptr = main_grad_ptr + token * main_grad_stride + cols
            main_grad = tl.load(ptr, mask=col_mask, other=0.0).to(tl.float32)
            tl.store(
                ptr,
                (main_grad + grad.to(tl.float32)).to(main_grad_ptr.dtype.element_ty),
                mask=col_mask,
            )
        else:
            tl.store(partial_ptr + i.to(tl.int64) * hidden + cols, acc, mask=col_mask)


@triton.jit
def _long_run_sums_kernel(
    sorted_ids_ptr,
    run_end_ptr,
    long_runs_ptr,
    grad_ptr,
    main_grad_ptr,
    partial_ptr,
    hidden,
    main_grad_stride,
    NROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Add the partial sums of a run longer than NROWS rows in order, round, add to main_grad."""
    run = tl.program_id(0)
    if run < tl.load(long_runs_ptr):
        i = tl.load(long_runs_ptr + 1 + run)
        end = tl.load(run_end_ptr + i)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        col_mask = cols < hidden
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for p in range(i, end + 1, NROWS):
            acc += tl.load(partial_ptr + p.to(tl.int64) * hidden + cols, mask=col_mask, other=0.0)
        token = tl.load(sorted_ids_ptr + i).to(tl.int64)
        grad = acc.to(grad_ptr.dtype.element_ty)
        ptr = main_grad_ptr + token * main_grad_stride + cols
        main_grad = tl.load(ptr, mask=col_mask, other=0.0).to(tl.float32)
        tl.store(
            ptr, (main_grad + grad.to(tl.float32)).to(main_grad_ptr.dtype.element_ty), mask=col_mask
        )


def _position_bits(num_tokens: int) -> int:
    return max(1, (num_tokens - 1).bit_length())


def can_accumulate_embedding_wgrad(
    ids: torch.Tensor, grad: torch.Tensor, main_grad: torch.Tensor
) -> bool:
    """Whether ``accumulate_embedding_wgrad`` supports these tensors.

    It requires Triton, between ``MIN_TOKENS`` and ``MAX_TOKENS`` tokens, bf16 or fp32 gradient
    and ``main_grad`` on the GPU, and a token id and position that pack into 31 bits.
    """
    num_tokens = ids.numel()
    return (
        HAVE_TRITON
        and MIN_TOKENS <= num_tokens <= MAX_TOKENS
        and ids.dtype in (torch.int32, torch.int64)
        and grad.is_cuda
        and grad.dtype in (torch.bfloat16, torch.float32)
        and main_grad.dtype in (torch.bfloat16, torch.float32)
        and main_grad.dim() == 2
        and main_grad.stride(-1) == 1
        and grad.shape[-1] == main_grad.shape[-1]
        and (main_grad.shape[0] - 1).bit_length() + _position_bits(num_tokens) <= 31
    )


def accumulate_embedding_wgrad(
    ids: torch.Tensor, grad: torch.Tensor, main_grad: torch.Tensor
) -> None:
    """``main_grad += embedding_dense_backward(grad, ids)``, touching only the rows of ``ids``.

    Args:
        ids: token ids of the embedding lookup, ``num_tokens`` elements.
        grad: gradient of the lookup's output, ``[*ids.shape, hidden]``.
        main_grad: ``[num_embeddings, hidden]`` buffer the weight gradient is added to.

    Requires ``can_accumulate_embedding_wgrad(ids, grad, main_grad)``.
    """
    num_tokens = ids.numel()
    hidden = grad.shape[-1]
    grad_rows = grad.reshape(num_tokens, hidden)
    if grad_rows.stride(-1) != 1:
        grad_rows = grad_rows.contiguous()
    device = grad.device
    sorted_ids = torch.empty(num_tokens, dtype=torch.int32, device=device)
    positions = torch.empty(num_tokens, dtype=torch.int32, device=device)
    run_start = torch.empty(num_tokens, dtype=torch.int32, device=device)
    run_end = torch.empty(num_tokens, dtype=torch.int32, device=device)
    max_long_runs = num_tokens // (_NROWS + 1) + 1
    long_runs = torch.empty(max_long_runs + 1, dtype=torch.int32, device=device)
    _sort_tokens_kernel[(1,)](
        ids.reshape(-1).contiguous(),
        sorted_ids,
        positions,
        run_start,
        run_end,
        long_runs,
        num_tokens,
        POS_BITS=_position_bits(num_tokens),
        NROWS=_NROWS,
        BLOCK=triton.next_power_of_2(num_tokens),
        num_warps=16,
    )
    partial = torch.empty(num_tokens, hidden, dtype=torch.float32, device=device)
    col_blocks = triton.cdiv(hidden, _BLOCK_H)
    _partial_sums_kernel[(num_tokens, col_blocks)](
        sorted_ids,
        positions,
        run_start,
        run_end,
        grad_rows,
        main_grad,
        partial,
        hidden,
        grad_rows.stride(0),
        main_grad.stride(0),
        NROWS=_NROWS,
        BLOCK_H=_BLOCK_H,
        num_warps=4,
    )
    _long_run_sums_kernel[(max_long_runs, col_blocks)](
        sorted_ids,
        run_end,
        long_runs,
        grad_rows,
        main_grad,
        partial,
        hidden,
        main_grad.stride(0),
        NROWS=_NROWS,
        BLOCK_H=_BLOCK_H,
        num_warps=4,
    )
