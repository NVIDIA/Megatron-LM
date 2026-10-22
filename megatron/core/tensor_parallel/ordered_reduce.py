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
  * symmetric-memory push (training, NVLink): the partials are in ordinary memory (the TE
    GEMM output). One kernel stores each peer's rows into that peer's fixed-size symmetric
    staging buffer (chunked, double-buffered, one block-level barrier per chunk) and folds
    the staged partials in rank order. No receive buffer of the input's size, no NCCL.
    Falls back to all-to-all + fold for uneven splits or without symmetric memory.
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
    part_stride,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_FP64: tl.constexpr,
    ROUND_FP32: tl.constexpr,
    C: tl.constexpr = 1,
):
    """Read n elements from every peer's symmetric buffer and fold them in rank order.

    p_r points at rank r's slice. With C > 1 each peer holds C partials, partial j at
    p_r + j * part_stride, and the fold runs over (rank, partial) in that order. The leading
    barrier makes the peers' writes visible; the trailing one keeps any rank from
    overwriting its buffer while a peer still reads it.
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
        for j in tl.static_range(1, C):
            acc += tl.load(p0 + j * part_stride + offs, mask=mask).to(acc_ty)
        for r in tl.static_range(1, WORLD_SIZE):
            pr = _peer_ptr(r, p0, p1, p2, p3, p4, p5, p6, p7)
            for j in tl.static_range(C):
                acc += tl.load(pr + j * part_stride + offs, mask=mask).to(acc_ty)
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


@triton.jit
def _peer_ptr(
    d: tl.constexpr, p0, p1, p2, p3, p4, p5, p6, p7
):  # pylint: disable=too-many-return-statements
    """The d-th of the peer pointers (d is a compile-time constant)."""
    if d == 0:
        return p0
    elif d == 1:
        return p1
    elif d == 2:
        return p2
    elif d == 3:
        return p3
    elif d == 4:
        return p4
    elif d == 5:
        return p5
    elif d == 6:
        return p6
    else:
        return p7


@triton.jit
def _push_chunk(
    src_ptr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    n,
    block_start,
    slot,
    pid,
    nb,
    lane,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    C: tl.constexpr = 1,
    src_part_stride=0,
):
    """Store this rank's partial(s) of output chunk [block_start, +BLOCK) into every peer's staging."""
    offs = block_start + lane
    mask = offs < n
    stage_base = ((slot * nb + pid) * WORLD_SIZE * C).to(tl.int64) * BLOCK
    for d in tl.static_range(WORLD_SIZE):
        if d != RANK:
            dst = _peer_ptr(d, s0, s1, s2, s3, s4, s5, s6, s7)
            for j in tl.static_range(C):
                val = tl.load(src_ptr + j * src_part_stride + d * n + offs, mask=mask)
                tl.store(dst + stage_base + (RANK * C + j) * BLOCK + lane, val, mask=mask)


@triton.jit
def _ordered_push_fold_kernel(
    src_ptr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    out_ptr,
    signal_pad_ptrs,
    n,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_FP64: tl.constexpr,
    ROUND_FP32: tl.constexpr,
    C: tl.constexpr = 1,
    src_part_stride=0,
):
    """Reduce-scatter of a local (non-symmetric) tensor through symmetric staging buffers.

    src holds this rank's partial of all WORLD_SIZE output slices (slice d at d * n). s_r is
    rank r's staging buffer: [2 slots][num_blocks][WORLD_SIZE][BLOCK]. Block b stores the
    BLOCK elements of every peer's slice it handles into that peer's staging (NVLink
    stores), barriers with block b of every peer, pushes its next chunk into the other
    slot, then folds the staged partials of the current chunk (its own from src) in rank
    order. Every block of every rank handles the same element ranges, so after the
    block-level barrier its inputs are complete. A peer writes slot s again only after
    passing the barrier that follows this block's fold of slot s. The trailing barrier
    makes the staging safe to reuse.
    """
    pid = tl.program_id(0)
    nb = tl.num_programs(0)
    acc_ty = tl.float64 if ACC_FP64 else tl.float32
    lane = tl.arange(0, BLOCK)
    my_stage = _peer_ptr(RANK, s0, s1, s2, s3, s4, s5, s6, s7)
    step = nb.to(tl.int64) * BLOCK
    first = pid.to(tl.int64) * BLOCK
    # Software pipeline: chunk k+1 is pushed right after barrier k, so its NVLink stores
    # are in flight while chunk k is folded.
    _push_chunk(
        src_ptr,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        n,
        first,
        0,
        pid,
        nb,
        lane,
        RANK,
        WORLD_SIZE,
        BLOCK,
        C,
        src_part_stride,
    )
    slot = 0
    for block_start in range(first, n, step):
        sync_threads()
        symm_mem_sync(
            signal_pad_ptrs,
            None,
            RANK,
            WORLD_SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )
        sync_threads()
        # Uniform across ranks and threads: every rank runs the same chunk schedule.
        _push_chunk(
            src_ptr,
            s0,
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
            n,
            block_start + step,
            1 - slot,
            pid,
            nb,
            lane,
            RANK,
            WORLD_SIZE,
            BLOCK,
            C,
            src_part_stride,
        )
        offs = block_start + lane
        mask = offs < n
        stage_base = ((slot * nb + pid) * WORLD_SIZE * C).to(tl.int64) * BLOCK
        acc = tl.zeros([BLOCK], dtype=acc_ty)
        for r in tl.static_range(WORLD_SIZE):
            for j in tl.static_range(C):
                if r == RANK:
                    v = tl.load(src_ptr + j * src_part_stride + r * n + offs, mask=mask).to(acc_ty)
                else:
                    v = tl.load(my_stage + stage_base + (r * C + j) * BLOCK + lane, mask=mask).to(
                        acc_ty
                    )
                if r == 0 and j == 0:
                    acc = v
                else:
                    acc += v
        if ROUND_FP32:
            acc = acc.to(tl.float32)
        tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)
        slot = 1 - slot

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


# ── Symmetric-memory push path (learner: local tensor, NVLink group) ────────

# Push staging: 16 warps, PUSH_VECS 16-byte vectors per thread and peer, two slots per block.
_PUSH_WARPS = 16
_PUSH_MAX_VECS = 4
# Set to False to force the all-to-all path (same bits; for benchmarks and debugging).
USE_SYMM_PUSH = True
# Above this input size the push (one barrier per staging chunk) is not faster than NCCL
# all-to-all + fold on GB200 TP4; both give the same bits, so the threshold is speed-only.
PUSH_MAX_INPUT_BYTES = 64 << 20


def _push_staging_bytes(world_size: int, num_parts: int = 1) -> int:
    block_bytes = _PUSH_WARPS * 32 * 16 * _PUSH_MAX_VECS
    return 2 * SYMM_MEM_KERNEL_CONFIG.max_num_blocks * world_size * num_parts * block_bytes


def _push_staging(group, tensor: torch.Tensor, num_parts: int = 1):
    """The symmetric staging handle for the push reduce-scatter on ``group``, or None.

    The staging is fixed-size: 2 x max_num_blocks x TP x 32 KiB per rank (32 MiB at TP 4,
    64 MiB at TP 8), independent of the message size. It is allocated (collectively) on
    the first eligible call; outside of a CUDA-graph capture.
    """
    if not (
        USE_SYMM_PUSH
        and HAVE_TRITON
        and tensor.is_cuda
        and 2 <= group.size() <= _MAX_PULL_WORLD_SIZE
        # The cap is speed-only (see above). With several shard partials per rank the push
        # beat the all-to-all at every measured size (GB200, TP2/TP4, up to 16k x 8192), so
        # it applies only to one partial per rank.
        and (num_parts > 1 or tensor.numel() * tensor.element_size() <= PUSH_MAX_INPUT_BYTES)
    ):
        return None
    # Imported here: megatron.core.inference imports tensor_parallel.
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    key = f"tp_ordered_push:{group.group_name}" + (f":x{num_parts}" if num_parts > 1 else "")
    if not SymmetricMemoryManager.is_initialized(key):
        if torch.cuda.is_current_stream_capturing():
            return None
        size_mb = -(-_push_staging_bytes(group.size(), num_parts) // (1 << 20))
        SymmetricMemoryManager.get_buffer(key, process_group=group, size_mb=size_mb)
    hdl = SymmetricMemoryManager.get_buffer(key).symm_mem_hdl
    if hdl is None or hdl.world_size != group.size():
        return None
    return hdl


def _launch_push_fold(
    input_: torch.Tensor, output: torch.Tensor, hdl, exact: Optional[bool], num_parts: int = 1
) -> torch.Tensor:
    """input_ holds num_parts partials of all world_size output slices: [num_parts, ws * n]."""
    ws = hdl.world_size
    n = output.numel()
    assert input_.numel() == n * ws * num_parts and output.is_contiguous()
    assert input_.is_contiguous()
    if n == 0:
        return output
    vec = 16 // input_.element_size()
    if "push" in _TUNE_OVERRIDE:
        per_thread = _TUNE_OVERRIDE["push"][1]
    else:
        # Small messages: more blocks with less work each; larger: fewer barriers per byte.
        per_thread = 2 if n * input_.element_size() <= (2 << 20) else _PUSH_MAX_VECS
    assert per_thread <= _PUSH_MAX_VECS, "the staging is sized for _PUSH_MAX_VECS"
    block = _PUSH_WARPS * 32 * vec * per_thread
    num_blocks = max(1, min(triton.cdiv(n, block), SYMM_MEM_KERNEL_CONFIG.max_num_blocks))
    SYMM_MEM_KERNEL_CONFIG.check_num_blocks(num_blocks)
    acc_fp64, round_fp32 = _acc_flags(output.dtype, exact)
    stage_numel = 2 * num_blocks * ws * num_parts * block
    views = _peer_views(hdl, 0, stage_numel, input_.dtype)
    ptrs = list(views) + [views[0]] * (_MAX_PULL_WORLD_SIZE - ws)
    _ordered_push_fold_kernel[(num_blocks, 1, 1)](
        input_,
        *ptrs,
        output,
        hdl.signal_pad_ptrs_dev,
        n,
        RANK=hdl.rank,
        WORLD_SIZE=ws,
        BLOCK=block,
        ACC_FP64=acc_fp64,
        ROUND_FP32=round_fp32,
        C=num_parts,
        src_part_stride=n * ws,
        num_warps=_PUSH_WARPS,
    )
    return output


# ── All-to-all paths (any process group) ────────────────────────────────────


def ordered_reduce_scatter(
    input_: torch.Tensor,
    group,
    output: Optional[torch.Tensor] = None,
    input_split_sizes: Optional[Sequence[int]] = None,
    exact: Optional[bool] = None,
) -> torch.Tensor:
    """Reduce-scatter along dim 0 with the fixed rank-order sum.

    With symmetric memory and even splits: the push kernel (see module docstring), which
    stores (TP-1)/TP of the input over NVLink and uses the fixed-size staging buffer.
    Otherwise, per rank: one all-to-all of the partials ((TP-1)/TP of the input over the
    wire, the same as a ring reduce-scatter), one transient receive buffer of the input's
    size, and one fold pass that reads TP x (output size) and writes the output. Both are
    CUDA-graph capturable (the staging is allocated on the first, uncaptured call).
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
    if input_split_sizes is None:
        staging = _push_staging(group, input_)
        if staging is not None:
            if output is None:
                output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)
            return _launch_push_fold(input_, output, staging, exact)
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


def _launch_pull(
    views,
    out: torch.Tensor,
    symm_mem_hdl,
    exact: Optional[bool],
    num_parts: int = 1,
    part_stride: int = 0,
) -> torch.Tensor:
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
        part_stride,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=ws,
        BLOCK=block,
        ACC_FP64=acc_fp64,
        ROUND_FP32=round_fp32,
        C=num_parts,
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

    Reads TP x the message per rank, so it is meant for small (latency-bound) messages.
    Same bits as :func:`ordered_all_reduce`.
    """
    assert input_.numel() == output.numel() and input_.dtype == output.dtype
    assert output.is_contiguous()
    elem = input_.element_size()
    assert byte_offset % elem == 0
    views = _peer_views(symm_mem_hdl, byte_offset // elem, input_.numel(), input_.dtype)
    return _launch_pull(views, output, symm_mem_hdl, exact)


# ── Fixed-shard row-parallel reductions (TP-invariant forward) ──────────────
#
# With ``batch_invariant_tp_shards = G`` a row-parallel output is defined independently of the
# TP size t (any t dividing G): split the reduction dim K into G equal shards, take each
# shard's GEMM output as one partial, and fold the G partials in shard order with the same
# accumulator and single rounding as above. TP rank r owns shards r*c .. r*c + c - 1
# (c = G / t), so the TP partials of the ordered reduction are the shard partials in order.
# At c = 1 this is exactly the ordered TP reduction. A shard's GEMM output does not depend
# on how the GEMM is called (TE or torch, contiguous or strided, single or batched), so a
# rank computes its c shards with one strided batched GEMM and no copies.


def row_parallel_shard_partials(
    x: torch.Tensor, weight: torch.Tensor, num_shards: int, out: Optional[torch.Tensor] = None
):
    """Shard partials ``[num_shards, *x.shape[:-1], N]`` of ``x @ weight.T`` (weight ``[N, K]``).

    Shard ``j`` is ``x[..., j*K/c:(j+1)*K/c] @ weight[:, j*K/c:(j+1)*K/c].T``, computed for all
    ``j`` in one batched GEMM on strided views (written into ``out`` when given, e.g. a
    symmetric buffer).
    """
    *lead, k = x.shape
    n = weight.shape[0]
    assert k % num_shards == 0, f"K ({k}) must be divisible by the local shard count {num_shards}"
    kc = k // num_shards
    x2 = x.reshape(-1, k)
    xb = x2.view(x2.shape[0], num_shards, kc).transpose(0, 1)
    wb = weight.view(n, num_shards, kc).permute(1, 2, 0)
    if out is not None:
        torch.bmm(xb, wb, out=out.view(num_shards, x2.shape[0], n))
        return out.view(num_shards, *lead, n)
    return torch.bmm(xb, wb).view(num_shards, *lead, n)


def ordered_reduce_scatter_shards(
    parts: torch.Tensor, group, exact: Optional[bool] = None
) -> torch.Tensor:
    """Reduce-scatter along dim 1 of every rank's shard partials, folded in global shard order.

    ``parts`` is ``[c, M, ...]``: this rank's c shard partials. Returns ``[M / TP, ...]``.
    Per rank: one copy of the partials into destination order, one all-to-all of
    c x (TP-1)/TP of the output size, and one fold over TP x c partials.
    """
    c = parts.shape[0]
    world_size = 1 if group is None else group.size()
    if world_size == 1:
        return ordered_fold(parts.contiguous(), exact=exact)
    if c == 1:
        return ordered_reduce_scatter(parts[0], group, exact=exact)
    m, rest = parts.shape[1], parts.shape[2:]
    assert m % world_size == 0, "First dimension of the tensor should be divisible by TP size"
    rows = m // world_size
    parts = parts.contiguous()
    staging = _push_staging(group, parts, num_parts=c)
    if staging is not None:
        output = torch.empty((rows, *rest), dtype=parts.dtype, device=parts.device)
        return _launch_push_fold(parts, output, staging, exact, num_parts=c)
    # Every (shard, destination) slice of parts is contiguous, so one batch of NCCL sends and
    # receives moves them without a destination-major copy. recv[r, j] is shard r*c + j.
    rank = group.rank()
    recv = torch.empty((world_size, c, rows, *rest), dtype=parts.dtype, device=parts.device)
    ops = []
    for d in range(world_size):
        if d == rank:
            continue
        peer = torch.distributed.get_global_rank(group, d)
        for j in range(c):
            ops.append(
                torch.distributed.P2POp(
                    torch.distributed.isend, parts[j, d * rows : (d + 1) * rows], peer, group
                )
            )
            ops.append(torch.distributed.P2POp(torch.distributed.irecv, recv[d, j], peer, group))
    recv[rank].copy_(parts[:, rank * rows : (rank + 1) * rows])
    for req in torch.distributed.batch_isend_irecv(ops):
        req.wait()
    return ordered_fold(recv.view(world_size * c, rows, *rest), exact=exact)


def ordered_all_reduce_shards(
    parts: torch.Tensor, group, exact: Optional[bool] = None
) -> torch.Tensor:
    """All-reduce of every rank's shard partials ``[c, ...]`` in global shard order."""
    c = parts.shape[0]
    world_size = 1 if group is None else group.size()
    if world_size == 1:
        return ordered_fold(parts.contiguous(), exact=exact)
    if c == 1:
        return ordered_all_reduce(parts[0], group, exact=exact)
    shape = parts.shape[1:]
    flat = parts.reshape(c, -1)
    n = flat.shape[1]
    align = max(1, 16 // flat.element_size()) * world_size
    padded = -(-n // align) * align
    if padded != n:
        flat = torch.nn.functional.pad(flat, (0, padded - n))
    chunk = ordered_reduce_scatter_shards(flat, group, exact=exact)
    out = torch.empty(padded, dtype=flat.dtype, device=flat.device)
    torch.distributed.all_gather_into_tensor(out, chunk.view(-1), group=group)
    return out[:n].view(shape)


def _accumulate_wgrad(weight: torch.Tensor, grad_out2d: torch.Tensor, input2d: torch.Tensor):
    """Weight gradient with Megatron's main_grad accumulation (as in tensor_parallel.layers)."""
    main_grad = getattr(weight, "main_grad", None)
    if main_grad is None:
        return grad_out2d.t().matmul(input2d)
    try:
        import fused_weight_gradient_mlp_cuda

        if main_grad.dtype == torch.float32:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(input2d, grad_out2d, main_grad)
        else:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(input2d, grad_out2d, main_grad)
    except ImportError:
        main_grad.add_(grad_out2d.t().matmul(input2d).to(main_grad.dtype))
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    weight.grad_added_to_main_grad = True
    # A dummy gradient keeps the DDP backward hooks on the main backprop thread.
    try:
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        return get_dummy_wgrad(
            list(main_grad.shape), input2d.dtype, zero=getattr(weight, "zero_out_wgrad", False)
        )
    except ImportError:
        return torch.zeros(main_grad.shape, dtype=input2d.dtype, device=input2d.device)


class _TPShardRowLinear(torch.autograd.Function):
    """Row-parallel linear whose forward folds this rank's shard partials in global order.

    Only the forward is fixed-order. The backward is the stock row-parallel one: all-gather
    the output gradient under sequence parallelism, then the full local-K input and weight
    gradient GEMMs.
    """

    @staticmethod
    def forward(ctx, x, weight, group, num_shards, sequence_parallel, wgrad_into_main_grad):
        parts = row_parallel_shard_partials(x, weight, num_shards)
        if sequence_parallel:
            out = ordered_reduce_scatter_shards(parts, group)
        else:
            out = ordered_all_reduce_shards(parts, group)
        ctx.save_for_backward(x, weight)
        ctx.group = group
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_into_main_grad = wgrad_into_main_grad
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        group = ctx.group
        world_size = 1 if group is None else group.size()
        if ctx.sequence_parallel and world_size > 1:
            grad_output = grad_output.contiguous()
            full = torch.empty(
                (grad_output.shape[0] * world_size, *grad_output.shape[1:]),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            torch.distributed.all_gather_into_tensor(full, grad_output, group=group)
            grad_output = full
        g2 = grad_output.reshape(-1, grad_output.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        grad_input = g2.matmul(weight).view(x.shape) if ctx.needs_input_grad[0] else None
        grad_weight = None
        if ctx.needs_input_grad[1]:
            if ctx.wgrad_into_main_grad:
                grad_weight = _accumulate_wgrad(weight, g2, x2)
            else:
                grad_weight = g2.t().matmul(x2)
        return grad_input, grad_weight, None, None, None, None


def tp_shard_row_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    group,
    num_shards: int,
    sequence_parallel: bool,
    wgrad_into_main_grad: bool = False,
) -> torch.Tensor:
    """Row-parallel ``x @ weight.T`` with the fixed-shard forward reduction (see above)."""
    return _TPShardRowLinear.apply(
        x, weight, group, num_shards, sequence_parallel, wgrad_into_main_grad
    )


def symm_ordered_reduce_scatter_shards(
    output: torch.Tensor, parts: torch.Tensor, symm_mem_hdl, byte_offset: int = 0, exact=None
) -> torch.Tensor:
    """:func:`ordered_reduce_scatter_shards` of ``parts`` [c, M, ...] in the symmetric buffer.

    ``parts`` is a contiguous view of the buffer at ``byte_offset``. One pull kernel reads
    the owned rows of every peer's c partials and folds them in global shard order.
    """
    ws = symm_mem_hdl.world_size
    c = parts.shape[0]
    n_out = output.numel()
    part_numel = parts[0].numel()
    assert part_numel == n_out * ws and parts.is_contiguous() and output.is_contiguous()
    elem = parts.element_size()
    assert byte_offset % elem == 0
    span = (c - 1) * part_numel + n_out
    views = _peer_views(
        symm_mem_hdl, byte_offset // elem + symm_mem_hdl.rank * n_out, span, parts.dtype
    )
    return _launch_pull(views, output, symm_mem_hdl, exact, num_parts=c, part_stride=part_numel)

