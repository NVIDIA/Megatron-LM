# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch-invariant tensor-parallel reductions with a fixed rank-order sum.

At TP >= 3, NCCL's summation order for an element depends on its position in the message,
so training and inference give the same token different bits. Every path here computes

    out = cast(((p_0 + p_1) + p_2) + ... + p_{TP-1})

in an fp32 accumulator (fp64 for fp64 inputs) with one final rounding, so the bits depend
only on the partials. Training and inference run these same kernels, so they agree bitwise.

Paths (all bitwise identical):
  * pull (training and inference): the GEMM writes into a symmetric buffer; one kernel reads the owned
    rows from every peer and folds them in rank order.
  * all-to-all + fold: NCCL fallback for groups without symmetric memory.

The all-reduce is reduce-scatter + all-gather. Only forward reductions use these paths.
With ``batch_invariant_tp_shards`` the partials are those of fixed reduction-dim shards, so
the bits do not depend on the TP size either (see the last sections)."""

from typing import Optional, Tuple
from unittest.mock import MagicMock

import torch

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

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.kernel_config import (
    SYMM_MEM_KERNEL_CONFIG,
)
from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads


def use_ordered_tp_reduce(group, dtype: torch.dtype) -> bool:
    """Whether a forward TP reduction over ``group`` must use the ordered path.

    True under batch-invariant mode for floating-point tensors at TP >= 3. At TP = 2 the
    single addition is commutative, so the stock collective already gives these bits.
    Integer sums are exact in any order.

    Used by the forward reductions in ``tensor_parallel/mappings.py`` (training) and by
    ``InferenceRowParallelLinear`` / the inference embedding reduce-scatter
    (``inference_layers.py``) to decide whether to leave the stock collective.
    """
    # Imported here: megatron.core.transformer imports tensor_parallel (through
    # context_parallel and mappings), so a module-level import would be circular.
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        is_batch_invariant_mode_enabled,
    )

    return is_batch_invariant_mode_enabled() and group.size() >= 3 and dtype.is_floating_point


@triton.jit
def _ordered_fold_kernel(
    src_ptr, dst_ptr, n, part_stride, TP: tl.constexpr, BLOCK: tl.constexpr, ACC_FP64: tl.constexpr
):
    """dst[i] = cast(((src[0, i] + src[1, i]) + ...) + src[TP-1, i]); src[r] at r * part_stride.

    One program per BLOCK of output elements; elements never interact, so the bits do not
    depend on the grid. Launched by :func:`ordered_fold`: the last step of the all-to-all
    fallback and the local (TP 1) shard folds of the linear layers and of the cross-entropy
    vocab sum.
    """
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc_ty = tl.float64 if ACC_FP64 else tl.float32
    acc = tl.load(src_ptr + offs, mask=mask).to(acc_ty)
    # int64: r * part_stride passes 2**31 for large messages.
    part_stride = tl.cast(part_stride, tl.int64)
    for r in tl.static_range(1, TP):
        acc += tl.load(src_ptr + r * part_stride + offs, mask=mask).to(acc_ty)
    tl.store(dst_ptr + offs, acc.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _peer_ptr(buffer_ptrs, rank, byte_offset, DTYPE: tl.constexpr, ALIGNED: tl.constexpr):
    """Pointer to ``byte_offset`` in rank ``rank``'s symmetric buffer, typed ``DTYPE``.

    ``buffer_ptrs`` is the handle's device array of every rank's buffer base address
    (``symm_mem_hdl.buffer_ptrs_dev``), so the kernels take any number of peers. A pointer
    loaded from memory carries no alignment information, so Triton would fall back to narrow
    loads and stores; with ``ALIGNED`` (every base + offset is 16-byte aligned, checked at
    launch) ``tl.multiple_of`` states it, which keeps the 16-byte vector accesses.
    """
    base = tl.load(buffer_ptrs.to(tl.pointer_type(tl.uint64)) + rank)
    ptr = (base.to(tl.pointer_type(tl.uint8)) + byte_offset).to(tl.pointer_type(DTYPE))
    if ALIGNED:
        ptr = tl.multiple_of(ptr, 16)
    return ptr


@triton.jit
def _ordered_pull_kernel(
    buffer_ptrs,
    byte_offset,
    out_ptr,
    signal_pad_ptrs,
    n,
    part_stride,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_FP64: tl.constexpr,
    ALIGNED: tl.constexpr,
    C: tl.constexpr,
    SYNC_AFTER: tl.constexpr,
):
    """Read n elements from every peer's symmetric buffer and fold them in rank order.

    Rank r's slice starts at ``byte_offset`` in its buffer. With C > 1 each peer holds C
    partials, partial j at slice + j * part_stride, and the fold runs over (rank, partial)
    in that order. The barrier waits until every peer has written its partials (a previous
    kernel). As in ``multimem_reduce_scatter`` there is no trailing barrier: rank r reads only
    its own rows of each buffer, which only rank r's following all-gather writes, and the
    next full-buffer write (the next GEMM) comes after that all-gather.

    Used by the inference engine, whose GEMM writes its partial straight into the "tp"
    symmetric buffer (launched through ``_launch_pull``):
      * ``InferenceRowParallelLinear._matmul_ordered_reduce_scatter`` (TP >= 3) and
        ``inference_reduce_scatter_to_sequence_parallel_region``, via
        ``symm_ordered_reduce_scatter`` (C = 1);
      * ``InferenceRowParallelLinear._matmul_shards_reduce`` (``batch_invariant_tp_shards``
        with several shards per rank), via ``symm_ordered_reduce_scatter_shards`` (C > 1).
    Direct-buffer training producers set SYNC_AFTER so every peer has finished reading
    before the next GEMM overwrites the scratch buffer. Other producers use all-to-all.
    """
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=False,
    )
    sync_threads()

    dtype = out_ptr.dtype.element_ty
    acc_ty = tl.float64 if ACC_FP64 else tl.float32
    part_stride = tl.cast(part_stride, tl.int64)  # j * part_stride can pass 2**31
    # Load every peer's address once, outside the data loop.
    peers = ()
    for r in tl.static_range(WORLD_SIZE):
        peers = peers + (_peer_ptr(buffer_ptrs, r, byte_offset, dtype, ALIGNED),)
    step = tl.num_programs(0).to(tl.int64) * BLOCK
    start = tl.program_id(0).to(tl.int64) * BLOCK
    for block_start in range(start, n, step):
        offs = block_start + tl.arange(0, BLOCK)
        mask = offs < n
        for r in tl.static_range(WORLD_SIZE):
            pr = peers[r]
            for j in tl.static_range(C):
                v = tl.load(pr + j * part_stride + offs, mask=mask).to(acc_ty)
                if r == 0 and j == 0:
                    acc = v
                else:
                    acc += v
        tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)
    if SYNC_AFTER:
        # Training can overwrite the whole buffer before another collective runs.
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


def _fold_launch_config(n: int, element_size: int) -> Tuple[int, int]:
    """(BLOCK, num_warps): 16-byte vectors, one or two per thread, enough CTAs to fill the GPU."""
    vec = 16 // element_size
    if n >= (1 << 20):
        num_warps, per_thread = 8, 2
    else:
        num_warps, per_thread = 4, 1
    return num_warps * 32 * vec * per_thread, num_warps


def ordered_fold(parts: torch.Tensor, output: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Left fold of ``parts`` [TP, ...] over dim 0 in the accumulator dtype, one rounding.

    Writes ``output`` (shape ``parts.shape[1:]``, contiguous; default dtype ``parts.dtype``).
    The TP slices may be strided apart; each is made contiguous if it is not.

    The final step of the all-to-all fallback and of every TP 1 shard reduction (linear
    layers and the cross-entropy vocab sum).
    """
    assert HAVE_TRITON and parts.is_cuda, "the ordered TP reductions need Triton on CUDA"
    tp = parts.shape[0]
    if output is None:
        output = torch.empty(parts.shape[1:], dtype=parts.dtype, device=parts.device)
    acc_fp64 = parts.dtype == torch.float64
    n = output.numel()
    assert output.is_contiguous()
    if not parts[0].is_contiguous() or (tp > 1 and parts.stride(0) < n):
        parts = parts.contiguous()
    if n == 0:
        return output
    block, num_warps = _fold_launch_config(n, parts.element_size())
    _ordered_fold_kernel[(triton.cdiv(n, block),)](
        parts,
        output,
        n,
        parts.stride(0),
        TP=tp,
        BLOCK=block,
        ACC_FP64=acc_fp64,
        num_warps=num_warps,
    )
    return output


def ordered_reduce_scatter(
    input_: torch.Tensor, group, output: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Reduce-scatter along dim 0 with the fixed rank-order sum.

    Ordinary tensors use all-to-all plus a fixed-order fold. Training row-linear GEMMs
    write directly into symmetric scratch and use the pull path instead. This function
    remains the fallback for other producers and groups without peer access.
    """
    return ordered_reduce_scatter_shards(input_.unsqueeze(0), group, output)


def ordered_all_reduce(input_: torch.Tensor, group) -> torch.Tensor:
    """All-reduce with the fixed rank-order sum (ordered reduce-scatter + all-gather).

    Works on any shape: the flattened tensor is split into TP equal chunks (zero-padded
    when the size is not a multiple of TP; padding never mixes with real elements).
    Returns a new tensor; ``input_`` is not modified.

    Used by the training forward without sequence parallelism
    (``_ReduceFromModelParallelRegion`` in ``mappings.py``).
    """
    return ordered_all_reduce_shards(input_.unsqueeze(0), group)


def _pull_launch_config(n: int, element_size: int) -> Tuple[int, int, int]:
    """(BLOCK, num_warps, num_blocks) for the pull kernel; identical on every rank."""
    vec = 16 // element_size
    num_warps = 16
    # One 16-byte load per thread and peer while that covers n with the whole grid; above
    # that (bandwidth-bound), two in flight.
    per_thread = 1 if n <= SYMM_MEM_KERNEL_CONFIG.max_num_blocks * 512 * vec else 2
    block = num_warps * 32 * vec * per_thread
    # The grid must be identical on every rank and fit the signal pad (kernel_config.py).
    num_blocks = max(1, min(triton.cdiv(n, block), SYMM_MEM_KERNEL_CONFIG.max_num_blocks))
    return block, num_warps, num_blocks


def _launch_pull(
    input_: torch.Tensor,
    out: torch.Tensor,
    symm_mem_hdl,
    num_parts: int,
    part_stride: int,
    sync_after: bool = False,
) -> torch.Tensor:
    """Launch the pull kernel; called by the two ``symm_ordered_reduce_scatter*`` wrappers.

    ``input_`` is this rank's tensor in the symmetric buffer; every peer's copy sits at the
    same byte offset in its own buffer. Each rank reads the slice it owns, ``rank * n``
    elements in.
    """
    n = out.numel()
    block, num_warps, num_blocks = _pull_launch_config(n, out.element_size())
    acc_fp64 = out.dtype == torch.float64
    SYMM_MEM_KERNEL_CONFIG.check_num_blocks(num_blocks)
    elem = input_.element_size()
    byte_offset = (
        input_.data_ptr() - symm_mem_hdl.buffer_ptrs[symm_mem_hdl.rank]
    ) + symm_mem_hdl.rank * n * elem
    # 16-byte vector accesses need every peer slice and partial to start 16-byte aligned.
    aligned = (
        byte_offset % 16 == 0
        and (part_stride * elem) % 16 == 0
        and all(ptr % 16 == 0 for ptr in symm_mem_hdl.buffer_ptrs)
    )
    _ordered_pull_kernel[(num_blocks, 1, 1)](
        symm_mem_hdl.buffer_ptrs_dev,
        byte_offset,
        out,
        symm_mem_hdl.signal_pad_ptrs_dev,
        n,
        part_stride,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        BLOCK=block,
        ACC_FP64=acc_fp64,
        ALIGNED=aligned,
        C=num_parts,
        SYNC_AFTER=sync_after,
        num_warps=num_warps,
    )
    return out


def symm_ordered_reduce_scatter(
    input_: torch.Tensor,
    symm_mem_hdl,
    output: Optional[torch.Tensor] = None,
    *,
    sync_after: bool = False,
) -> torch.Tensor:
    """Reduce-scatter dim 0 of ``input_``, a contiguous tensor in the symmetric buffer.

    One kernel: every rank reads its owned rows from all peers over NVLink and folds them
    in rank order into ``output`` (allocated when None). Same bits as
    :func:`ordered_reduce_scatter`.

    Used by the inference engine at TP >= 3: ``InferenceRowParallelLinear`` (its GEMM writes
    into the "tp" buffer) and ``inference_reduce_scatter_to_sequence_parallel_region``.
    """
    return symm_ordered_reduce_scatter_shards(
        input_.unsqueeze(0), symm_mem_hdl, output, sync_after=sync_after
    )


def symm_ordered_reduce_scatter_shards(
    parts: torch.Tensor,
    symm_mem_hdl,
    output: Optional[torch.Tensor] = None,
    *,
    sync_after: bool = False,
) -> torch.Tensor:
    """:func:`ordered_reduce_scatter_shards` of ``parts`` [c, M, ...] in the symmetric buffer.

    ``parts`` is a contiguous tensor in the symmetric buffer. One pull kernel reads
    the owned rows of every peer's c partials and folds them in global shard order into
    ``output`` (allocated when None). ``sync_after=True`` adds a trailing barrier, allowing
    training to overwrite the whole scratch buffer on the next call. Without it, reuse
    requires a later synchronizing collective (the inference engine's next all-gather).

    Used by ``InferenceRowParallelLinear._matmul_shards_reduce`` (the engine with several
    ``batch_invariant_tp_shards`` per rank).
    """
    ws = symm_mem_hdl.world_size
    c = parts.shape[0]
    if output is None:
        output = parts.new_empty((parts.shape[1] // ws, *parts.shape[2:]))
    n_out = output.numel()
    part_numel = parts[0].numel()
    assert part_numel == n_out * ws and parts.is_contiguous() and output.is_contiguous()
    assert parts.dtype == output.dtype
    return _launch_pull(
        parts, output, symm_mem_hdl, num_parts=c, part_stride=part_numel, sync_after=sync_after
    )


def ordered_reduce_scatter_shards(
    parts: torch.Tensor, group, output: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Reduce-scatter along dim 1 of every rank's shard partials, folded in global shard order.

    ``parts`` is ``[c, M, ...]``: this rank's c shard partials. Returns ``[M / TP, ...]``.
    A rank sends c x (TP-1)/TP of one partial's size (c times a reduce-scatter): the fold
    order cannot be pre-combined per rank without changing the bits. One all-to-all
    exchanges the partials and one fold pass adds them in fixed order.

    The one implementation of the ordered reduce-scatter: :func:`ordered_reduce_scatter` is
    the c = 1 case. Also used directly with several shards per rank by the training forward
    (``_TPShardRowLinear``) and the inference fallback when the pull is unavailable
    (``InferenceRowParallelLinear._matmul_shards_reduce``).
    """
    c = parts.shape[0]
    world_size = group.size()
    if world_size == 1:
        return ordered_fold(parts, output)
    m, rest = parts.shape[1], parts.shape[2:]
    assert m % world_size == 0, "First dimension of the tensor should be divisible by TP size"
    rows = m // world_size
    parts = parts.contiguous()
    if output is None:
        output = parts.new_empty((rows, *rest))
    # all_to_all_single sends dim-0 slice d to rank d, so lay the partials out
    # destination-major, [TP, c, rows, ...] (a free view when c = 1). recv[r, j] is source
    # rank r's partial j of this rank's rows: global shard r * c + j, the fold's order.
    send = parts.view(c, world_size, rows, *rest).transpose(0, 1).contiguous()
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=group)
    return ordered_fold(recv.view(world_size * c, rows, *rest), output)


def ordered_all_reduce_shards(parts: torch.Tensor, group) -> torch.Tensor:
    """All-reduce of every rank's shard partials ``[c, ...]`` in global shard order.

    Flattens, zero-pads to a multiple of TP x 16 bytes, reduce-scatters in shard order and
    all-gathers. With c = 1 this is :func:`ordered_all_reduce`.

    Used by the training forward without sequence parallelism (``_TPShardRowLinear``) and
    by the cross-entropy vocab sum (:func:`batch_invariant_vocab_sum`).
    """
    c = parts.shape[0]
    world_size = group.size()
    if world_size == 1:
        return ordered_fold(parts)
    shape = parts.shape[1:]
    flat = parts.reshape(c, -1)
    n = flat.shape[1]
    # Keep every chunk 16-byte aligned so both kernels and NCCL use vector accesses. The zero
    # padding is folded like any element (0 + 0 is exact) and sliced off below, so it never
    # touches the real elements' bits.
    align = max(1, 16 // flat.element_size()) * world_size
    padded = -(-n // align) * align
    if padded != n:
        flat = torch.nn.functional.pad(flat, (0, padded - n))
    chunk = ordered_reduce_scatter_shards(flat, group)
    out = torch.empty(padded, dtype=flat.dtype, device=flat.device)
    torch.distributed.all_gather_into_tensor(out, chunk.view(-1), group=group)
    return out[:n].view(shape)
