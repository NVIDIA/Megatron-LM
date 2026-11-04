# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch-invariant tensor-parallel reductions with a fixed rank-order sum.

At TP >= 3, NCCL's summation order for an element depends on its position in the message,
so training and inference give the same token different bits. Every path here computes

    out = cast(((p_0 + p_1) + p_2) + ... + p_{TP-1})

in an fp32 accumulator with one final rounding (fp64 inputs accumulate in fp64;
``batch_invariant_collective="multimem"`` accumulates in fp64 and rounds to fp32 first), so
the bits depend only on the partials.

Paths (all bitwise identical):
  * push (training): one kernel stores each peer's rows into its symmetric staging buffer
    over NVLink and folds them in rank order.
  * pull (inference): the GEMM writes into a symmetric buffer; one kernel reads the owned
    rows from every peer and folds them in rank order.
  * all-to-all + fold: NCCL fallback for groups without symmetric memory.

The all-reduce is reduce-scatter + all-gather. Only forward reductions use these paths.
With ``batch_invariant_tp_shards`` the partials are those of fixed reduction-dim shards, so
the bits do not depend on the TP size either (see the last sections).
"""

from typing import Optional, Tuple
from unittest.mock import MagicMock

import torch

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.kernel_config import (
    SYMM_MEM_KERNEL_CONFIG,
)
from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    get_batch_invariant_collective,
    is_batch_invariant_mode_enabled,
)
from megatron.core.utils import null_decorator

try:
    import fused_weight_gradient_mlp_cuda

    HAVE_GRAD_ACCUM_FUSION = True
except ImportError:
    HAVE_GRAD_ACCUM_FUSION = False

try:
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


def use_ordered_tp_reduce(group, dtype: torch.dtype) -> bool:
    """Whether a forward TP reduction over ``group`` must use the ordered path.

    True under batch-invariant mode for floating-point tensors at TP >= 3. At TP = 2 the
    single addition is commutative, so the stock collective already gives these bits.
    Integer sums are exact in any order.

    Used by the forward reductions in ``tensor_parallel/mappings.py`` (training) and by
    ``InferenceRowParallelLinear`` / the inference embedding reduce-scatter
    (``inference_layers.py``) to decide whether to leave the stock collective.
    """
    return is_batch_invariant_mode_enabled() and group.size() >= 3 and dtype.is_floating_point


def exact_sum_mode() -> bool:
    """Whether the batch-invariant collective is "multimem" (see module docstring)."""
    return get_batch_invariant_collective() == "multimem"


def _acc_flags(dtype: torch.dtype, exact: Optional[bool]) -> Tuple[bool, bool]:
    """(ACC_FP64, ROUND_FP32) kernel flags for an input dtype and summation mode.

    "ordered" (default): fp32 accumulator, one rounding to the output dtype.
    "multimem" (exact): fp64 accumulator, rounded once to fp32, then to the output dtype,
    mirroring NVLS's correctly rounded fp32 sum. fp64 inputs accumulate in fp64 either way.
    """
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
    if ROUND_FP32:
        acc = acc.to(tl.float32)
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
    ROUND_FP32: tl.constexpr,
    ALIGNED: tl.constexpr,
    C: tl.constexpr = 1,
):
    """Read n elements from every peer's symmetric buffer and fold them in rank order.

    Rank r's slice starts at ``byte_offset`` in its buffer. With C > 1 each peer holds C
    partials, partial j at slice + j * part_stride, and the fold runs over (rank, partial)
    in that order. The leading barrier makes the peers' writes visible; the trailing one
    keeps any rank from overwriting its buffer while a peer still reads it.

    Used by the inference engine, whose GEMM writes its partial straight into the "tp"
    symmetric buffer (launched through ``_launch_pull``):
      * ``InferenceRowParallelLinear._matmul_ordered_reduce_scatter`` (TP >= 3) and
        ``inference_reduce_scatter_to_sequence_parallel_region``, via
        ``symm_ordered_reduce_scatter`` (C = 1);
      * ``InferenceRowParallelLinear._matmul_shards_reduce`` (``batch_invariant_tp_shards``
        with several shards per rank), via ``symm_ordered_reduce_scatter_shards`` (C > 1).
    The training side uses the push kernel instead (its input is not in symmetric memory).
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
        acc = tl.zeros([BLOCK], dtype=acc_ty)
        for r in tl.static_range(WORLD_SIZE):
            pr = peers[r]
            for j in tl.static_range(C):
                v = tl.load(pr + j * part_stride + offs, mask=mask).to(acc_ty)
                if r == 0 and j == 0:
                    acc = v
                else:
                    acc += v
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
def _push_chunk(
    src_ptr,
    stages,
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
    """Store this rank's C partials of chunk [block_start, +BLOCK) into every peer's staging.

    Called by the push kernel for its first chunk and, after each barrier, for the next
    chunk, so those NVLink stores overlap the fold of the current chunk.
    """
    offs = block_start + lane
    mask = offs < n
    stage_base = ((slot * nb + pid) * WORLD_SIZE * C).to(tl.int64) * BLOCK
    for d in tl.static_range(WORLD_SIZE):
        if d != RANK:
            dst = stages[d]
            for j in tl.static_range(C):
                val = tl.load(src_ptr + j * src_part_stride + d * n + offs, mask=mask)
                tl.store(dst + stage_base + (RANK * C + j) * BLOCK + lane, val, mask=mask)


@triton.jit
def _ordered_push_fold_kernel(
    src_ptr,
    stage_ptrs,
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

    src holds this rank's C partials of all WORLD_SIZE output slices (partial j of slice d at
    j * src_part_stride + d * n). Rank r's staging buffer (base at ``stage_ptrs[r]``) is
    [2 slots][num_blocks][WORLD_SIZE * C][BLOCK]. Block b stores the
    BLOCK elements of every peer's slice it handles into that peer's staging (NVLink
    stores), barriers with block b of every peer, pushes its next chunk into the other
    slot, then folds the staged partials of the current chunk (its own from src) in rank
    order. Every block of every rank handles the same element ranges, so after the
    block-level barrier its inputs are complete. A peer writes slot s again only after
    passing the barrier that follows this block's fold of slot s. The trailing barrier
    makes the staging safe to reuse.

    Used by the training forward (learner), whose input is an ordinary tensor (the TE GEMM
    output): :func:`ordered_reduce_scatter_shards` (and its one-partial case
    :func:`ordered_reduce_scatter`), through ``_launch_push_fold``.
    """
    pid = tl.program_id(0)
    nb = tl.num_programs(0)
    # int64: the slice and partial offsets (d * n, j * src_part_stride) pass 2**31 for
    # large messages.
    n = tl.cast(n, tl.int64)
    src_part_stride = tl.cast(src_part_stride, tl.int64)
    acc_ty = tl.float64 if ACC_FP64 else tl.float32
    lane = tl.arange(0, BLOCK)
    # Load every rank's staging address once, outside the chunk loop.
    stages = ()
    for r in tl.static_range(WORLD_SIZE):
        stages = stages + (_peer_ptr(stage_ptrs, r, 0, src_ptr.dtype.element_ty, True),)
    my_stage = stages[RANK]
    step = nb.to(tl.int64) * BLOCK
    first = pid.to(tl.int64) * BLOCK
    # Software pipeline: chunk k+1 is pushed right after barrier k, so its NVLink stores
    # are in flight while chunk k is folded.
    _push_chunk(
        src_ptr, stages, n, first, 0, pid, nb, lane, RANK, WORLD_SIZE, BLOCK, C, src_part_stride
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
            stages,
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
    if n >= (1 << 20):
        num_warps, per_thread = 8, 2
    else:
        num_warps, per_thread = 4, 1
    return num_warps * 32 * vec * per_thread, num_warps


def ordered_fold(
    parts: torch.Tensor, out: Optional[torch.Tensor] = None, exact: Optional[bool] = None
) -> torch.Tensor:
    """Left fold of ``parts`` [TP, ...] over dim 0 in the accumulator dtype, one rounding.

    Writes ``out`` (shape ``parts.shape[1:]``, contiguous; default dtype ``parts.dtype``).
    The TP slices may be strided apart; each is made contiguous if it is not.

    The final step of the all-to-all fallback and of every TP 1 shard reduction (linear
    layers and the cross-entropy vocab sum).
    """
    assert HAVE_TRITON and parts.is_cuda, "the ordered TP reductions need Triton on CUDA"
    tp = parts.shape[0]
    if out is None:
        out = torch.empty(parts.shape[1:], dtype=parts.dtype, device=parts.device)
    acc_fp64, round_fp32 = _acc_flags(parts.dtype, exact)
    n = out.numel()
    assert out.is_contiguous()
    if not parts[0].is_contiguous() or (tp > 1 and parts.stride(0) < n):
        parts = parts.contiguous()
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


# ── Symmetric-memory push path (learner: local tensor, NVLink group) ────────

# Push staging: 16 warps, PUSH_VECS 16-byte vectors per thread and peer, two slots per block.
_PUSH_WARPS = 16
_PUSH_MAX_VECS = 4


def _push_staging_bytes(world_size: int, num_parts: int = 1) -> int:
    """Size of the push staging buffer (2 slots x max blocks x peers x partials x block)."""
    block_bytes = _PUSH_WARPS * 32 * 16 * _PUSH_MAX_VECS
    return 2 * SYMM_MEM_KERNEL_CONFIG.max_num_blocks * world_size * num_parts * block_bytes


def _push_staging_key(group, num_parts: int = 1) -> str:
    return f"tp_ordered_push:{group.group_name}" + (f":x{num_parts}" if num_parts > 1 else "")


def prepare_ordered_reduce(group, num_parts: int = 1) -> None:
    """Allocate the push staging for ``group`` and ``num_parts`` partials per rank.

    Collective over ``group``. Called when a batch-invariant row-parallel layer is built
    (``TELinear``), so the allocation never happens inside a forward or a CUDA-graph capture.
    Any other user allocates it on its first eager call.
    """
    if group is not None and group.size() >= 2:
        _push_staging(group, num_parts)


def _push_staging(group, num_parts: int = 1):
    """The symmetric staging handle for the push reduce-scatter on ``group``, or None.

    The staging is fixed-size: 2 x max_num_blocks x TP x num_parts x 32 KiB per rank (32 MiB
    at TP 4 with one partial, 64 MiB at TP 8), independent of the message size. It is
    allocated (collectively) by :func:`prepare_ordered_reduce` or on the first eager call.
    None (no symmetric memory for the group, e.g. TP across NVLink domains) makes
    :func:`ordered_reduce_scatter_shards` take the all-to-all path.
    """
    key = _push_staging_key(group, num_parts)
    if not SymmetricMemoryManager.is_initialized(key):
        assert not torch.cuda.is_current_stream_capturing(), (
            "the ordered-reduce push staging is allocated at layer construction "
            "(prepare_ordered_reduce) or on the first eager call, not during CUDA-graph capture"
        )
        size_mb = -(-_push_staging_bytes(group.size(), num_parts) // (1 << 20))
        SymmetricMemoryManager.get_buffer(key, process_group=group, size_mb=size_mb)
    return SymmetricMemoryManager.get_buffer(key).symm_mem_hdl


def _launch_push_fold(
    input_: torch.Tensor, output: torch.Tensor, hdl, exact: Optional[bool], num_parts: int = 1
) -> torch.Tensor:
    """Launch the push kernel: ``input_`` holds num_parts partials of all world_size output
    slices ([num_parts, ws * n]). Called by :func:`ordered_reduce_scatter_shards`."""
    ws = hdl.world_size
    n = output.numel()
    assert input_.numel() == n * ws * num_parts and output.is_contiguous()
    assert input_.is_contiguous()
    if n == 0:
        return output
    vec = 16 // input_.element_size()
    # Small messages: more blocks with less work each; larger: fewer barriers per byte.
    per_thread = 2 if n * input_.element_size() <= (2 << 20) else _PUSH_MAX_VECS
    block = _PUSH_WARPS * 32 * vec * per_thread
    num_blocks = max(1, min(triton.cdiv(n, block), SYMM_MEM_KERNEL_CONFIG.max_num_blocks))
    SYMM_MEM_KERNEL_CONFIG.check_num_blocks(num_blocks)
    acc_fp64, round_fp32 = _acc_flags(output.dtype, exact)
    # The staging pointers are asserted 16-byte aligned in the kernel (_peer_ptr).
    assert all(ptr % 16 == 0 for ptr in hdl.buffer_ptrs), "staging buffers must be 16B aligned"
    _ordered_push_fold_kernel[(num_blocks, 1, 1)](
        input_,
        hdl.buffer_ptrs_dev,
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


# ── Ordered reduce-scatter / all-reduce (learner) ───────────────────────────


def ordered_reduce_scatter(
    input_: torch.Tensor, group, output: Optional[torch.Tensor] = None, exact: Optional[bool] = None
) -> torch.Tensor:
    """Reduce-scatter along dim 0 with the fixed rank-order sum.

    With symmetric memory: the push kernel (see module docstring), which
    stores (TP-1)/TP of the input over NVLink and uses the fixed-size staging buffer.
    Otherwise, per rank: one all-to-all of the partials ((TP-1)/TP of the input over the
    wire, the same as a ring reduce-scatter), one transient receive buffer of the input's
    size, and one fold pass that reads TP x (output size) and writes the output. Both are
    CUDA-graph capturable (the staging is allocated before capture, see
    :func:`prepare_ordered_reduce`). This is the c = 1 case of
    :func:`ordered_reduce_scatter_shards`.

    Used by the training forward: the sequence-parallel and tensor-parallel reduce-scatters
    in ``mappings.py`` (TERowParallelLinear's batch-invariant reduction goes through them).
    """
    return ordered_reduce_scatter_shards(input_.unsqueeze(0), group, exact, output=output)


def ordered_all_reduce(input_: torch.Tensor, group, exact: Optional[bool] = None) -> torch.Tensor:
    """All-reduce with the fixed rank-order sum (ordered reduce-scatter + all-gather).

    Works on any shape: the flattened tensor is split into TP equal chunks (zero-padded
    when the size is not a multiple of TP; padding never mixes with real elements).
    Returns a new tensor; ``input_`` is not modified.

    Used by the training forward without sequence parallelism
    (``_ReduceFromModelParallelRegion`` in ``mappings.py``).
    """
    return ordered_all_reduce_shards(input_.unsqueeze(0), group, exact=exact)


# ── Symmetric-memory pull paths (inference) ─────────────────────────────────


def can_use_symm_pull(symm_mem_hdl, tensor: torch.Tensor) -> bool:
    """Whether the symmetric-memory pull kernels can run for this handle and tensor.

    Checked by the inference layers (``inference_layers.py``) before taking the pull;
    otherwise they use the all-to-all path, which gives the same bits.
    """
    return (
        symm_mem_hdl is not None and symm_mem_hdl.world_size >= 2 and tensor.dtype.is_floating_point
    )


def _pull_launch_config(n: int, element_size: int) -> Tuple[int, int, int]:
    """(BLOCK, num_warps, num_blocks) for the pull kernel; identical on every rank."""
    vec = 16 // element_size
    if n <= SYMM_MEM_KERNEL_CONFIG.max_num_blocks * 512 * vec:
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
    input_: torch.Tensor,
    out: torch.Tensor,
    symm_mem_hdl,
    exact: Optional[bool],
    num_parts: int = 1,
    part_stride: int = 0,
) -> torch.Tensor:
    """Launch the pull kernel; called by the two ``symm_ordered_reduce_scatter*`` wrappers.

    ``input_`` is this rank's tensor in the symmetric buffer; every peer's copy sits at the
    same byte offset in its own buffer. Each rank reads the slice it owns, ``rank * n``
    elements in.
    """
    n = out.numel()
    block, num_warps, num_blocks = _pull_launch_config(n, out.element_size())
    acc_fp64, round_fp32 = _acc_flags(out.dtype, exact)
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
        ROUND_FP32=round_fp32,
        ALIGNED=aligned,
        C=num_parts,
        num_warps=num_warps,
    )
    return out


def symm_ordered_reduce_scatter(
    input_: torch.Tensor,
    symm_mem_hdl,
    output: Optional[torch.Tensor] = None,
    exact: Optional[bool] = None,
) -> torch.Tensor:
    """Reduce-scatter dim 0 of ``input_``, a contiguous tensor in the symmetric buffer.

    One kernel: every rank reads its owned rows from all peers over NVLink and folds them
    in rank order into ``output`` (allocated when None). Same bits as
    :func:`ordered_reduce_scatter`.

    Used by the inference engine at TP >= 3: ``InferenceRowParallelLinear`` (its GEMM writes
    into the "tp" buffer) and ``inference_reduce_scatter_to_sequence_parallel_region``.
    """
    return symm_ordered_reduce_scatter_shards(input_.unsqueeze(0), symm_mem_hdl, output, exact)


def symm_ordered_reduce_scatter_shards(
    parts: torch.Tensor,
    symm_mem_hdl,
    output: Optional[torch.Tensor] = None,
    exact: Optional[bool] = None,
) -> torch.Tensor:
    """:func:`ordered_reduce_scatter_shards` of ``parts`` [c, M, ...] in the symmetric buffer.

    ``parts`` is a contiguous tensor in the symmetric buffer. One pull kernel reads
    the owned rows of every peer's c partials and folds them in global shard order into
    ``output`` (allocated when None).

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
    return _launch_pull(parts, output, symm_mem_hdl, exact, num_parts=c, part_stride=part_numel)


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

    Used when a rank holds several ``batch_invariant_tp_shards``: by the training forward
    (``tp_shard_row_linear``) and by ``InferenceRowParallelLinear._matmul_shards_reduce``.
    """
    *lead, k = x.shape
    n = weight.shape[0]
    assert k % num_shards == 0, f"K ({k}) must be divisible by the local shard count {num_shards}"
    kc = k // num_shards
    # Flatten the leading dims (e.g. [s, b]) into the GEMM's M; free for the contiguous
    # activations the callers pass (a non-contiguous x would be copied here).
    x2 = x.reshape(-1, k)
    # Strided views, no copies: batch j reads columns j*kc:(j+1)*kc of x and of the weight.
    # Each batch entry is the same [M, kc] x [kc, N] GEMM a rank holding only shard j runs,
    # so its bits match (cuBLASLt, workspace 0 under batch-invariant mode; tested).
    xb = x2.view(x2.shape[0], num_shards, kc).transpose(0, 1)  # [c, M, kc]
    wb = weight.view(n, num_shards, kc).permute(1, 2, 0)  # [c, kc, N]
    if out is not None:
        torch.bmm(xb, wb, out=out.view(num_shards, x2.shape[0], n))
        return out.view(num_shards, *lead, n)
    return torch.bmm(xb, wb).view(num_shards, *lead, n)


def ordered_reduce_scatter_shards(
    parts: torch.Tensor, group, exact: Optional[bool] = None, output: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Reduce-scatter along dim 1 of every rank's shard partials, folded in global shard order.

    ``parts`` is ``[c, M, ...]``: this rank's c shard partials. Returns ``[M / TP, ...]``.
    A rank sends c x (TP-1)/TP of one partial's size (c times a reduce-scatter): the fold
    order cannot be pre-combined per rank without changing the bits. With symmetric memory
    the push kernel folds the TP x c partials; otherwise one all-to-all and a fold pass.

    The one implementation of the ordered reduce-scatter: :func:`ordered_reduce_scatter` is
    the c = 1 case. Also used directly with several shards per rank by the training forward
    (``_TPShardRowLinear``) and the inference fallback when the pull is unavailable
    (``InferenceRowParallelLinear._matmul_shards_reduce``).
    """
    c = parts.shape[0]
    world_size = 1 if group is None else group.size()
    if world_size == 1:
        return ordered_fold(parts.contiguous(), output, exact)
    m, rest = parts.shape[1], parts.shape[2:]
    assert m % world_size == 0, "First dimension of the tensor should be divisible by TP size"
    rows = m // world_size
    parts = parts.contiguous()
    if output is None:
        output = parts.new_empty((rows, *rest))
    staging = _push_staging(group, num_parts=c)
    if staging is not None:
        return _launch_push_fold(parts, output, staging, exact, num_parts=c)
    # all_to_all_single sends dim-0 slice d to rank d, so lay the partials out
    # destination-major, [TP, c, rows, ...] (a free view when c = 1). recv[r, j] is source
    # rank r's partial j of this rank's rows: global shard r * c + j, the fold's order.
    send = parts.view(c, world_size, rows, *rest).transpose(0, 1).contiguous()
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=group)
    return ordered_fold(recv.view(world_size * c, rows, *rest), output, exact)


def ordered_all_reduce_shards(
    parts: torch.Tensor, group, exact: Optional[bool] = None
) -> torch.Tensor:
    """All-reduce of every rank's shard partials ``[c, ...]`` in global shard order.

    Flattens, zero-pads to a multiple of TP x 16 bytes, reduce-scatters in shard order and
    all-gathers. With c = 1 this is :func:`ordered_all_reduce`.

    Used by the training forward without sequence parallelism (``_TPShardRowLinear``) and
    by the cross-entropy vocab sum (:func:`batch_invariant_vocab_sum`).
    """
    c = parts.shape[0]
    world_size = 1 if group is None else group.size()
    if world_size == 1:
        return ordered_fold(parts.contiguous(), exact=exact)
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
    chunk = ordered_reduce_scatter_shards(flat, group, exact=exact)
    out = torch.empty(padded, dtype=flat.dtype, device=flat.device)
    torch.distributed.all_gather_into_tensor(out, chunk.view(-1), group=group)
    return out[:n].view(shape)


def _accumulate_wgrad(weight: torch.Tensor, grad_out2d: torch.Tensor, input2d: torch.Tensor):
    """Weight gradient with Megatron's main_grad accumulation (as in tensor_parallel.layers)."""
    main_grad = getattr(weight, "main_grad", None)
    if main_grad is None:
        return grad_out2d.t().matmul(input2d)
    if HAVE_GRAD_ACCUM_FUSION and main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(input2d, grad_out2d, main_grad)
    elif HAVE_GRAD_ACCUM_FUSION and main_grad.dtype in (torch.float16, torch.bfloat16):
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(input2d, grad_out2d, main_grad)
    else:
        main_grad.add_(grad_out2d.t().matmul(input2d).to(main_grad.dtype))
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    weight.grad_added_to_main_grad = True
    # A dummy gradient keeps the DDP backward hooks on the main backprop thread.
    zero = getattr(weight, "zero_out_wgrad", False)
    if HAVE_TE:
        return get_dummy_wgrad(list(main_grad.shape), input2d.dtype, zero=zero)
    factory = torch.zeros if zero else torch.empty
    return factory(main_grad.shape, dtype=input2d.dtype, device=input2d.device)


class _TPShardRowLinear(torch.autograd.Function):
    """Row-parallel linear whose forward folds this rank's shard partials in global order.

    Only the forward is fixed-order. The backward is the stock row-parallel one: all-gather
    the output gradient under sequence parallelism, then the full local-K input and weight
    gradient GEMMs. Used by ``TERowParallelLinear`` when a rank holds several
    ``batch_invariant_tp_shards`` (through :func:`tp_shard_row_linear`).
    """

    @staticmethod
    def forward(ctx, x, weight, group, num_shards, sequence_parallel, wgrad_into_main_grad):
        """Shard partials of x @ weight.T, reduced (SP: reduce-scattered) in shard order."""
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
        """Stock row-parallel backward (all-gather under SP, full local-K GEMMs)."""
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


# ── Vocab-parallel sums (cross entropy) ─────────────────────────────────────
#
# The cross-entropy denominator sum_v exp(logit_v - max) is a reduction over the vocab, split
# across TP ranks like a row-parallel K. Under batch-invariant mode it uses the same
# definition: the vocab is cut into fixed shards (one per rank, or batch_invariant_tp_shards),
# each shard is summed in a fixed order that does not depend on the number of rows, and the
# shard sums are folded in shard order across TP.

_ROW_SUM_BLOCK = 1024  # fixed: the in-shard summation order must not depend on any size


@triton.jit
def _row_shard_sum_kernel(x_ptr, out_ptr, num_rows, row_stride, shard_len, BLOCK: tl.constexpr):
    """out[s, r] = sum of row r over shard s: BLOCK-strided fp32 lanes, then one tree sum.

    One program per (row, shard), so the order depends only on the shard length and the
    fixed BLOCK, never on the number of rows. Launched by :func:`batch_invariant_vocab_sum`
    (the cross-entropy softmax denominator).
    """
    row = tl.program_id(0).to(tl.int64)
    shard = tl.program_id(1).to(tl.int64)
    base = x_ptr + row * row_stride + shard * shard_len
    lanes = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, shard_len, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        lanes += tl.load(base + offs, mask=offs < shard_len, other=0.0).to(tl.float32)
    tl.store(out_ptr + shard * num_rows + row, tl.sum(lanes, axis=0))


def batch_invariant_vocab_sum(x: torch.Tensor, group, num_shards: int = 1) -> torch.Tensor:
    """fp32 ``x.sum(-1)`` over a vocab split across ``group``, with batch-invariant bits.

    ``x`` [..., V/TP] holds this rank's ``num_shards`` consecutive vocab shards. Each shard is
    summed per row in a fixed order, and the shard sums of all ranks are folded in global shard
    order (:func:`ordered_all_reduce_shards`), so the result depends neither on the number of
    rows nor, for a fixed total shard count, on the TP size.

    Used by the vocab-parallel cross entropy (``cross_entropy.py``, unfused and fused) for the
    softmax denominator under batch-invariant mode.
    """
    *lead, vocab = x.shape
    assert vocab % num_shards == 0, f"local vocab {vocab} must divide into {num_shards} shards"
    x2 = x.reshape(-1, vocab)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    parts = torch.empty((num_shards, rows), dtype=torch.float32, device=x.device)
    if rows:
        _row_shard_sum_kernel[(rows, num_shards)](
            x2, parts, rows, x2.stride(0), vocab // num_shards, BLOCK=_ROW_SUM_BLOCK
        )
    return ordered_all_reduce_shards(parts, group).view(lead)
