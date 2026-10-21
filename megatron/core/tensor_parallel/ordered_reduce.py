# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch-invariant tensor-parallel reductions with a fixed rank-order sum.

At TP >= 3 the order in which NCCL adds the TP partials of an element depends on where
the element sits in the message (ring rotation, chunking, algorithm choice by message
size). The training forward ([s, b] rows) and the inference engine (flat decode rows)
place the same token at different positions, so the same token gets different bits.
At TP = 2 the sum is a single commutative addition and NCCL is already exact.

Every path in this module computes, for each element,

    out = cast(((p_0 + p_1) + p_2) + ... + p_{TP-1})

with p_r the partial of TP rank r upcast to fp32 (fp64 for fp64 inputs), the adds done
in that accumulator with round-to-nearest, and a single final rounding to the output
dtype. The bits of an element depend only on its TP partials, never on its position,
the message size or the path that computed it, so the size thresholds below cannot
change results.

Paths:
  * all-to-all + fold (any group, NCCL): one ``all_to_all_single`` of the partials
    (the bytes a ring reduce-scatter moves, (TP-1)/TP of the input per rank) and one
    Triton pass that reads the TP partials of the owned rows and writes the output dtype.
  * symmetric-memory pull (inference, NVLink): the partials are already in a symmetric
    buffer; one kernel barriers through the signal pad, reads the owned slice (RS) or
    everything (one-shot AR) from every peer with plain loads (not multimem, whose
    in-switch reduction order is unspecified) and folds in rank order.

The ordered all-reduce of large messages is ordered reduce-scatter + all-gather.
Only forward reductions use these paths; backward keeps the stock collectives.

``batch_invariant_collective="multimem"`` (the EP combine's exact-sum setting) switches every
path to an fp64 accumulator with one rounding to fp32 before the final cast. All paths stay
bitwise identical to each other in either setting.
"""

from typing import Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import torch

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.kernel_config import (
    SYMM_MEM_KERNEL_CONFIG,
)
from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    get_batch_invariant_collective,
    is_batch_invariant_mode_enabled,
)
from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

# Peer pointers are passed to the pull kernel as separate arguments so that Triton
# specializes them on 16-byte alignment and emits vectorized loads.
_MAX_PULL_WORLD_SIZE = 8


def use_ordered_tp_reduce(group, dtype: torch.dtype) -> bool:
    """Whether a forward TP reduction over ``group`` must use the ordered path.

    True under batch-invariant mode for floating-point tensors at TP >= 3. At TP = 2 the
    single addition is commutative, so the stock collective already gives these bits.
    Integer sums are exact in any order.
    """
    return is_batch_invariant_mode_enabled() and group.size() >= 3 and dtype.is_floating_point


def exact_sum_mode() -> bool:
    """Whether the batch-invariant collective is "multimem" (see module docstring)."""
    return get_batch_invariant_collective() == "multimem"


def _acc_flags(dtype: torch.dtype, exact: Optional[bool]) -> Tuple[bool, bool]:
    """(ACC_FP64, ROUND_FP32) kernel flags for an input dtype and summation mode."""
    if exact is None:
        exact = exact_sum_mode()
    if dtype == torch.float64:
        return True, False
    return exact, exact


# ── Kernels ─────────────────────────────────────────────────────────────────


@triton.jit
def _ordered_fold_kernel(
    src_ptr,
    dst_ptr,
    n,
    part_stride,
    TP: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_FP64: tl.constexpr,
    ROUND_FP32: tl.constexpr,
):
    """dst[i] = cast(((src[0, i] + src[1, i]) + ...) + src[TP-1, i]); src[r] at r * part_stride."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc_ty = tl.float64 if ACC_FP64 else tl.float32
    acc = tl.load(src_ptr + offs, mask=mask).to(acc_ty)
    for r in tl.static_range(1, TP):
        acc += tl.load(src_ptr + r * part_stride + offs, mask=mask).to(acc_ty)
    if ROUND_FP32:
        acc = acc.to(tl.float32)
    tl.store(dst_ptr + offs, acc.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ordered_pull_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    out_ptr,
    signal_pad_ptrs,
    n,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_FP64: tl.constexpr,
    ROUND_FP32: tl.constexpr,
):
    """Read n elements from every peer's symmetric buffer and fold them in rank order.

    p_r points at rank r's slice. The leading barrier makes the peers' writes visible;
    the trailing one keeps any rank from overwriting its buffer while a peer still reads it.
    """
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=True,
    )
    sync_threads()

    acc_ty = tl.float64 if ACC_FP64 else tl.float32
    step = tl.num_programs(0).to(tl.int64) * BLOCK
    start = tl.program_id(0).to(tl.int64) * BLOCK
    for block_start in range(start, n, step):
        offs = block_start + tl.arange(0, BLOCK)
        mask = offs < n
        acc = tl.load(p0 + offs, mask=mask).to(acc_ty)
        acc += tl.load(p1 + offs, mask=mask).to(acc_ty)
        if WORLD_SIZE > 2:
            acc += tl.load(p2 + offs, mask=mask).to(acc_ty)
        if WORLD_SIZE > 3:
            acc += tl.load(p3 + offs, mask=mask).to(acc_ty)
        if WORLD_SIZE > 4:
            acc += tl.load(p4 + offs, mask=mask).to(acc_ty)
        if WORLD_SIZE > 5:
            acc += tl.load(p5 + offs, mask=mask).to(acc_ty)
        if WORLD_SIZE > 6:
            acc += tl.load(p6 + offs, mask=mask).to(acc_ty)
        if WORLD_SIZE > 7:
            acc += tl.load(p7 + offs, mask=mask).to(acc_ty)
        if ROUND_FP32:
            acc = acc.to(tl.float32)
        tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=False,
    )


# ── Fold ────────────────────────────────────────────────────────────────────


def _fold_launch_config(n: int, element_size: int) -> Tuple[int, int]:
    """(BLOCK, num_warps): 16-byte vectors, two per thread, and enough CTAs to fill the GPU."""
    vec = 16 // element_size
    if "fold" in _TUNE_OVERRIDE:
        num_warps, per_thread = _TUNE_OVERRIDE["fold"]
    elif n >= (1 << 20):
        num_warps, per_thread = 8, 2
    else:
        num_warps, per_thread = 4, 1
    return num_warps * 32 * vec * per_thread, num_warps


def ordered_fold(
    parts: torch.Tensor, out: Optional[torch.Tensor] = None, exact: Optional[bool] = None
) -> torch.Tensor:
    """Left fold of ``parts`` [TP, ...] over dim 0 in the accumulator dtype, one rounding.

    Writes ``out`` (shape ``parts.shape[1:]``, contiguous; default dtype ``parts.dtype``).
    Each ``parts[r]`` must be contiguous; the TP slices may be strided apart.
    """
    tp = parts.shape[0]
    if out is None:
        out = torch.empty(parts.shape[1:], dtype=parts.dtype, device=parts.device)
    acc_fp64, round_fp32 = _acc_flags(parts.dtype, exact)
    n = out.numel()
    if (
        HAVE_TRITON
        and parts.is_cuda
        and out.is_contiguous()
        and parts[0].is_contiguous()
        and (tp == 1 or parts.stride(0) >= n)
    ):
        if n == 0:
            return out
        block, num_warps = _fold_launch_config(n, parts.element_size())
        _ordered_fold_kernel[(triton.cdiv(n, block),)](
            parts,
            out,
            n,
            parts.stride(0),
            TP=tp,
            BLOCK=block,
            ACC_FP64=acc_fp64,
            ROUND_FP32=round_fp32,
            num_warps=num_warps,
        )
        return out
    acc_dtype = torch.float64 if acc_fp64 else torch.float32
    acc = parts[0].to(acc_dtype)
    for r in range(1, tp):
        acc = acc + parts[r].to(acc_dtype)
    out.copy_(acc.float() if round_fp32 else acc)
    return out


# ── All-to-all paths (any process group) ────────────────────────────────────


def ordered_reduce_scatter(
    input_: torch.Tensor,
    group,
    output: Optional[torch.Tensor] = None,
    input_split_sizes: Optional[Sequence[int]] = None,
    exact: Optional[bool] = None,
) -> torch.Tensor:
    """Reduce-scatter along dim 0 with the fixed rank-order sum.

    Per rank: one all-to-all of the partials ((TP-1)/TP of the input over the wire, the same
    as a ring reduce-scatter), one transient receive buffer of the input's size, and one
    fold pass that reads TP x (output size) and writes the output. CUDA-graph capturable.
    """
    world_size = group.size()
    rank = group.rank()
    input_ = input_.contiguous()
    if input_split_sizes is None:
        assert (
            input_.shape[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"
        rows = input_.shape[0] // world_size
        out_split = None
        in_split = None
    else:
        rows = input_split_sizes[rank]
        out_split = [rows] * world_size
        in_split = list(input_split_sizes)
    out_shape = (rows, *input_.shape[1:])
    recv = torch.empty(
        (world_size * rows, *input_.shape[1:]), dtype=input_.dtype, device=input_.device
    )
    torch.distributed.all_to_all_single(
        recv, input_, output_split_sizes=out_split, input_split_sizes=in_split, group=group
    )
    if output is None:
        output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)
    return ordered_fold(recv.view(world_size, *out_shape), output, exact)


def ordered_all_reduce(input_: torch.Tensor, group, exact: Optional[bool] = None) -> torch.Tensor:
    """All-reduce with the fixed rank-order sum (ordered reduce-scatter + all-gather).

    Works on any shape: the flattened tensor is split into TP equal chunks (zero-padded
    when the size is not a multiple of TP; padding never mixes with real elements).
    Returns a new tensor; ``input_`` is not modified.
    """
    world_size = group.size()
    shape = input_.shape
    flat = input_.contiguous().view(-1)
    n = flat.numel()
    # Keep every chunk 16-byte aligned so both kernels and NCCL use vector accesses.
    align = max(1, 16 // flat.element_size()) * world_size
    padded = -(-n // align) * align
    if padded != n:
        tmp = torch.zeros(padded, dtype=flat.dtype, device=flat.device)
        tmp[:n].copy_(flat)
        flat = tmp
    chunk = ordered_reduce_scatter(flat, group, exact=exact)
    out = torch.empty(padded, dtype=flat.dtype, device=flat.device)
    torch.distributed.all_gather_into_tensor(out, chunk, group=group)
    if padded != n:
        out = out[:n]
    return out.view(shape)


# ── Symmetric-memory pull paths (inference) ─────────────────────────────────

_PEER_VIEWS: Dict[Tuple, List[torch.Tensor]] = {}


def _peer_views(symm_mem_hdl, offset_elems: int, numel: int, dtype: torch.dtype):
    """Views of every rank's symmetric buffer at [offset, offset + numel) (cached)."""
    key = (id(symm_mem_hdl), offset_elems, numel, dtype)
    views = _PEER_VIEWS.get(key)
    if views is None:
        views = [
            symm_mem_hdl.get_buffer(r, (numel,), dtype, offset_elems)
            for r in range(symm_mem_hdl.world_size)
        ]
        _PEER_VIEWS[key] = views
    return views


def can_use_symm_pull(symm_mem_hdl, tensor: torch.Tensor) -> bool:
    """Whether the symmetric-memory pull kernels can run for this handle and tensor."""
    return (
        HAVE_TRITON
        and symm_mem_hdl is not None
        and 2 <= symm_mem_hdl.world_size <= _MAX_PULL_WORLD_SIZE
        and tensor.is_cuda
        and tensor.dtype.is_floating_point
    )


# Benchmark hook: {"pull": (warps, vecs_per_thread), "fold": (warps, vecs_per_thread)}.
_TUNE_OVERRIDE: Dict[str, Tuple[int, int]] = {}


def _pull_launch_config(n: int, element_size: int) -> Tuple[int, int, int]:
    """(BLOCK, num_warps, num_blocks) for the pull kernel; identical on every rank."""
    vec = 16 // element_size
    if "pull" in _TUNE_OVERRIDE:
        num_warps, per_thread = _TUNE_OVERRIDE["pull"]
    elif n <= SYMM_MEM_KERNEL_CONFIG.max_num_blocks * 512 * vec:
        # Up to one 16-byte load per thread per peer across the whole grid.
        num_warps, per_thread = 16, 1
    else:
        # Bandwidth-bound: two 16-byte loads per peer in flight per thread.
        num_warps, per_thread = 16, 2
    block = num_warps * 32 * vec * per_thread
    # The grid must be identical on every rank and fit the signal pad (kernel_config.py).
    num_blocks = max(1, min(triton.cdiv(n, block), SYMM_MEM_KERNEL_CONFIG.max_num_blocks))
    return block, num_warps, num_blocks


def _launch_pull(views, out: torch.Tensor, symm_mem_hdl, exact: Optional[bool]) -> torch.Tensor:
    n = out.numel()
    block, num_warps, num_blocks = _pull_launch_config(n, out.element_size())
    acc_fp64, round_fp32 = _acc_flags(out.dtype, exact)
    SYMM_MEM_KERNEL_CONFIG.check_num_blocks(num_blocks)
    ws = symm_mem_hdl.world_size
    ptrs = list(views) + [views[0]] * (_MAX_PULL_WORLD_SIZE - ws)
    _ordered_pull_kernel[(num_blocks, 1, 1)](
        *ptrs,
        out,
        symm_mem_hdl.signal_pad_ptrs_dev,
        n,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=ws,
        BLOCK=block,
        ACC_FP64=acc_fp64,
        ROUND_FP32=round_fp32,
        num_warps=num_warps,
    )
    return out


def symm_ordered_reduce_scatter(
    output: torch.Tensor,
    input_: torch.Tensor,
    symm_mem_hdl,
    byte_offset: int = 0,
    exact: Optional[bool] = None,
) -> torch.Tensor:
    """Reduce-scatter dim 0 of ``input_`` (a view of the symmetric buffer at ``byte_offset``).

    One kernel: every rank reads its owned rows from all peers over NVLink and folds them
    in rank order into ``output``. Same bits as :func:`ordered_reduce_scatter`.
    """
    ws = symm_mem_hdl.world_size
    n_out = output.numel()
    assert input_.numel() == n_out * ws, "input numel must be world_size * output numel"
    assert input_.dtype == output.dtype and output.is_contiguous()
    elem = input_.element_size()
    assert byte_offset % elem == 0
    views = _peer_views(
        symm_mem_hdl, byte_offset // elem + symm_mem_hdl.rank * n_out, n_out, input_.dtype
    )
    return _launch_pull(views, output, symm_mem_hdl, exact)


def symm_ordered_all_reduce(
    output: torch.Tensor,
    input_: torch.Tensor,
    symm_mem_hdl,
    byte_offset: int = 0,
    exact: Optional[bool] = None,
) -> torch.Tensor:
    """One-shot all-reduce: every rank reads all of every peer's partial and folds them.

    Reads TP x the message per rank, so it is meant for small (latency-bound) messages
    . Same bits as :func:`ordered_all_reduce`.
    """
    assert input_.numel() == output.numel() and input_.dtype == output.dtype
    assert output.is_contiguous()
    elem = input_.element_size()
    assert byte_offset % elem == 0
    views = _peer_views(symm_mem_hdl, byte_offset // elem, input_.numel(), input_.dtype)
    return _launch_pull(views, output, symm_mem_hdl, exact)
