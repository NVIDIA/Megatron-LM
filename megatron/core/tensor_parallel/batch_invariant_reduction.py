# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The tensor-parallel reduction of batch-invariant mode.

NCCL's ring reduce-scatter adds the partials of the chunk owned by ring position u as
``((x[u+1] + x[u+2]) + ...) + x[u]``, rounding to the input dtype after every add, and its
all-reduce chunks the tensor the same way. A token's chunk follows its row in the step, which
the other requests set, so with three or more ranks its sum depends on the batch. In
batch-invariant mode every tensor-parallel reduce-scatter and all-reduce goes through this
module instead. Each element is ``((x_0 + x_1) + x_2) + ...`` over the ranks' partials: fp32
adds in ascending rank order (each rounds to fp32), then one rounding to the input dtype. That
is the same on every rank, at every position and at any TP size. Training and inference use the
same reduction, so rollouts reproduce training at any TP size.

* With a symmetric-memory buffer holding every rank's input (inference-optimized layers write
  their GEMM output into the "tp" buffer), one kernel reads every peer's partial of this rank's
  rows and sums them.
* Otherwise an NCCL all-to-all hands each rank every partial of its rows, and the same kernel
  sums them locally. With ``async_op`` the all-to-all runs on NCCL's stream and the returned
  handle's ``wait()`` launches the kernel, so the caller's work in between (TE's and Megatron's
  weight-gradient GEMM) overlaps the communication as it does with NCCL's reduce-scatter.
* With two ranks every element is a single add of two values. bf16 and fp16 partials are exact
  in fp32 and fp32 has at least 2p + 2 bits for their p-bit significands, so the fp32 add
  rounded to the input dtype equals one correctly rounded add; for fp32 it is one add.
  Without a symmetric buffer, NCCL's reduce-scatter and all-reduce therefore stay in use there,
  provided NCCL's add is correctly rounded (test_batch_invariant_reduction.py checks the bits
  for bf16, fp16 and fp32).
"""

from typing import List, Optional, Tuple, Union
from unittest.mock import MagicMock

import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.kernel_config import (
    SYMM_MEM_KERNEL_CONFIG,
)
from megatron.core.inference.communication.torch_symm_triton.utils import add_rn_f32, sync_threads
from megatron.core.parallel_state import get_global_memory_buffer
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

# Elements each program sums per step: 8 per thread with 4 warps, i.e. 128-bit bf16 loads.
_BLOCK_SIZE = 1024
_NUM_WARPS = 4
# Grid cap for the local sum. Each element is reduced on its own, so the grid only sets
# occupancy; the symmetric-memory launch is capped by its signal pad instead.
_MAX_LOCAL_BLOCKS = 1024
# Dtypes summed in fp32. Integer sums are exact in any order and keep NCCL's reduction.
_ORDERED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _load_partial(
    src_ptr,
    buffer_ptrs_dev,
    src_byte_offset,
    numel_per_rank,
    offsets,
    mask,
    src_rank: tl.constexpr,
    FROM_PEERS: tl.constexpr,
    RANK: tl.constexpr,
):
    """Load src_rank's partial of this rank's elements, as fp32."""
    if FROM_PEERS:
        # Every peer's buffer holds its full input; this rank's elements are segment RANK.
        peer_base = tl.load(buffer_ptrs_dev.to(tl.pointer_type(tl.uint64)) + src_rank)
        peer_ptr = (peer_base.to(tl.pointer_type(tl.uint8)) + src_byte_offset).to(src_ptr.dtype)
        values = tl.load(peer_ptr + RANK * numel_per_rank + offsets, mask=mask)
    else:
        # The all-to-all output holds src_rank's partial of this rank's elements in segment
        # src_rank.
        values = tl.load(src_ptr + src_rank * numel_per_rank + offsets, mask=mask)
    return values.to(tl.float32)


@triton.jit(do_not_specialize=["src_byte_offset", "numel_per_rank"])
def _rank_ordered_sum_kernel(
    out_ptr,
    src_ptr,
    buffer_ptrs_dev,
    signal_pad_ptrs,
    src_byte_offset,
    numel_per_rank,
    FROM_PEERS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    PADDED_WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """out[i] = ((x_0[i] + x_1[i]) + x_2[i]) + ... as fp32 adds, then rounded to out's dtype.

    x_r is rank r's partial of this rank's elements. With FROM_PEERS they are read from the
    peers' symmetric buffers after a barrier that makes every rank's earlier writes visible
    (the barrier pattern of ordered_reduce_scatter_v). This rank reads segment RANK of every
    buffer, the bytes multimem_reduce_scatter reads, so the buffer-reuse rules of that kernel
    apply unchanged. The barrier's arange is padded to PADDED_WORLD_SIZE, a power of two, so
    any WORLD_SIZE works. Otherwise they are segments of the local all-to-all output.

    The sum starts from x_0, not from zero, so the signs of zeros match an NCCL add.
    """
    if FROM_PEERS:
        symm_mem_sync(
            signal_pad_ptrs,
            None,
            RANK,
            WORLD_SIZE,
            hasPreviousMemAccess=False,
            hasSubsequentMemAccess=True,
            padded_world_size=PADDED_WORLD_SIZE,
        )
        sync_threads()

    numel_per_rank = numel_per_rank.to(tl.int64)
    block_start = tl.program_id(axis=0).to(tl.int64) * BLOCK_SIZE
    stride = tl.num_programs(axis=0).to(tl.int64) * BLOCK_SIZE
    while block_start < numel_per_rank:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel_per_rank
        acc = _load_partial(
            src_ptr,
            buffer_ptrs_dev,
            src_byte_offset,
            numel_per_rank,
            offsets,
            mask,
            0,
            FROM_PEERS,
            RANK,
        )
        for src_rank in tl.static_range(1, WORLD_SIZE):
            acc = add_rn_f32(
                acc,
                _load_partial(
                    src_ptr,
                    buffer_ptrs_dev,
                    src_byte_offset,
                    numel_per_rank,
                    offsets,
                    mask,
                    src_rank,
                    FROM_PEERS,
                    RANK,
                ),
            )
        # Triton rounds fp32 to bf16 / fp16 to nearest even, as NCCL's adds do.
        tl.store(out_ptr + offsets, acc.to(out_ptr.dtype.element_ty), mask=mask)
        block_start += stride


def _rank_ordered_sum(
    output: torch.Tensor, partials: torch.Tensor, world_size: int, rank: int, symm_mem_hdl=None
) -> torch.Tensor:
    """Sum this rank's ``output.numel()`` elements over ``world_size`` partials, in rank order.

    Args:
        output: This rank's contiguous result.
        partials: Without ``symm_mem_hdl``, the [world_size * output.numel()] all-to-all output,
            rank r's partial in segment r. With it, this rank's full input in the symmetric
            buffer, laid out alike on every rank, whose segment ``rank`` is reduced.
        world_size: Ranks in the group.
        rank: This rank's index in the group.
        symm_mem_hdl: The symmetric-memory handle of the buffer holding ``partials``.
    """
    assert HAVE_TRITON, "Triton is required for the batch-invariant TP reduction."
    assert output.is_contiguous() and partials.is_contiguous()
    assert partials.dtype == output.dtype, (partials.dtype, output.dtype)
    assert output.dtype in _ORDERED_DTYPES, f"no fp32 rank-order sum for {output.dtype}"
    numel_per_rank = output.numel()
    assert partials.numel() == world_size * numel_per_rank, (partials.shape, output.shape)
    if numel_per_rank == 0:
        return output
    num_blocks = triton.cdiv(numel_per_rank, _BLOCK_SIZE)
    if symm_mem_hdl is not None:
        assert symm_mem_hdl.world_size == world_size and symm_mem_hdl.rank == rank
        # Same grid on every rank (it depends only on the shape), within the signal pad.
        num_blocks = min(num_blocks, SYMM_MEM_KERNEL_CONFIG.max_num_blocks)
        SYMM_MEM_KERNEL_CONFIG.check_num_blocks(num_blocks)
        buffer_ptrs_dev = symm_mem_hdl.buffer_ptrs_dev
        signal_pad_ptrs = symm_mem_hdl.signal_pad_ptrs_dev
        src_byte_offset = partials.data_ptr() - symm_mem_hdl.buffer_ptrs[rank]
    else:
        num_blocks = min(num_blocks, _MAX_LOCAL_BLOCKS)
        buffer_ptrs_dev = 0
        signal_pad_ptrs = 0
        src_byte_offset = 0
    _rank_ordered_sum_kernel[(num_blocks, 1, 1)](
        output,
        partials,
        buffer_ptrs_dev,
        signal_pad_ptrs,
        src_byte_offset,
        numel_per_rank,
        FROM_PEERS=symm_mem_hdl is not None,
        RANK=rank,
        WORLD_SIZE=world_size,
        PADDED_WORLD_SIZE=triton.next_power_of_2(world_size),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )
    return output


class _Scratch:
    """The temporaries of one reduction (received partials, padding, the all-reduce's chunk).

    Synchronous reductions reuse one persistent set (Megatron's global memory buffer, the same
    storage on every call and CUDA graph replay): stream order keeps the next reduction's
    all-to-all from writing it before this one's kernel has read it. An asynchronous
    reduction's all-to-all can still be writing when another reduction starts, so it takes a
    second set, or fresh tensors while another asynchronous reduction holds that set (TE and
    Megatron keep one in flight at a time).
    """

    async_set_in_use = False

    def __init__(self, async_op: bool):
        self.suffix = ""
        if async_op:
            self.suffix = None if _Scratch.async_set_in_use else "_async"
            _Scratch.async_set_in_use = True

    def tensor(self, shape, dtype: torch.dtype, name: str) -> torch.Tensor:
        """A persistent (or, see above, fresh) scratch tensor."""
        if self.suffix is None:
            return torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
        return get_global_memory_buffer().get_tensor(
            list(shape), dtype, f"batch_invariant_tp_{name}{self.suffix}"
        )

    def release(self) -> None:
        """Give the asynchronous set back once the reduction has read it."""
        if self.suffix == "_async":
            _Scratch.async_set_in_use = False


class _PendingReduction:
    """Work-like handle of a rank-ordered reduction whose all-to-all is in flight.

    wait() makes the current stream wait for the all-to-all (no host sync) and then launches
    the rest of the reduction, so the result is ready, in stream order, when it returns.
    """

    def __init__(self, work: dist.Work, finish, keep_alive):
        self._work = work
        self._finish = finish
        # Inputs the all-to-all still reads.
        self._keep_alive = keep_alive

    def wait(self) -> bool:
        """Finish the reduction (once)."""
        if self._work is not None:
            self._work.wait()
            finish, self._finish, self._work, self._keep_alive = self._finish, None, None, None
            finish()
        return True


def _launch_all_to_all(input_, group, output, rows, input_split_sizes, scratch, async_op):
    """Send each rank its rows of every partial; returns the all-to-all's work (None unless
    async_op) and a function that sums the received partials into output."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    partials = scratch.tensor((world_size * rows, *input_.shape[1:]), input_.dtype, "partials")
    work = dist.all_to_all_single(
        partials,
        input_,
        output_split_sizes=None if input_split_sizes is None else [rows] * world_size,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=async_op,
    )
    return work, lambda: _rank_ordered_sum(output, partials, world_size, rank)


def _finish_or_defer(work, finish, async_op, keep_alive):
    if not async_op:
        finish()
        return None
    return _PendingReduction(work, finish, keep_alive)


def batch_invariant_reduce_scatter(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    output: Optional[torch.Tensor] = None,
    input_split_sizes: Optional[List[int]] = None,
    symm_mem_hdl=None,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[Union[dist.Work, "_PendingReduction"]]]:
    """Reduce-scatter along the first dimension, each element summed in rank order.

    Args:
        input_: This rank's partial, [T, ...]. With ``symm_mem_hdl``, the tensor of a
            symmetric buffer that holds every rank's partial at the same offset.
        group: The tensor-parallel group.
        output: Optional output, [T / world_size, ...] (or [input_split_sizes[rank], ...]).
        input_split_sizes: Rows of ``input_`` reduced onto each rank, if not equal.
        symm_mem_hdl: Handle of the symmetric buffer holding ``input_``.
        async_op: Return before the reduction completes, with a handle whose ``wait()`` the
            caller must call before reading the output (NCCL's own work where NCCL reduces).
            The symmetric-memory path is one kernel and completes in stream order.

    Returns:
        The output and a work handle or None, like TE's ``reduce_scatter_along_first_dim``.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        # As NCCL's reduce-scatter does, fill a given output.
        return (input_ if output is None else output.copy_(input_)), None
    rank = dist.get_rank(group)
    if input_split_sizes is None:
        assert input_.size(0) % world_size == 0, (
            f"First dimension of the tensor ({input_.size(0)}) should be divisible by tensor "
            f"parallel size ({world_size})"
        )
        rows = input_.size(0) // world_size
    else:
        assert symm_mem_hdl is None, "uneven splits are reduced without symmetric memory"
        rows = input_split_sizes[rank]
    if output is None:
        output = torch.empty((rows, *input_.shape[1:]), dtype=input_.dtype, device=input_.device)
    if symm_mem_hdl is not None:
        return _rank_ordered_sum(output, input_, world_size, rank, symm_mem_hdl), None
    input_ = input_.contiguous()
    if world_size == 2 or not input_.is_floating_point():
        if input_split_sizes is None:
            work = dist.reduce_scatter_tensor(output, input_, group=group, async_op=async_op)
        else:
            work = dist.reduce_scatter(
                output, list(torch.split(input_, input_split_sizes)), group=group, async_op=async_op
            )
        return output, work
    scratch = _Scratch(async_op)
    work, sum_partials = _launch_all_to_all(
        input_, group, output, rows, input_split_sizes, scratch, async_op
    )

    def finish():
        sum_partials()
        scratch.release()

    return output, _finish_or_defer(work, finish, async_op, input_)


def batch_invariant_all_reduce(
    input_: torch.Tensor, group: Optional[dist.ProcessGroup], async_op: bool = False
) -> Tuple[torch.Tensor, Optional[Union[dist.Work, "_PendingReduction"]]]:
    """All-reduce with every element summed in rank order, in place for a contiguous input.

    A rank-ordered reduce-scatter of the flattened tensor (zero-padded to a multiple of the
    group size) followed by an all-gather, so it moves the bytes a ring all-reduce does.

    Returns:
        The reduced tensor and a work handle or None, like TE's ``allreduce``. With
        ``async_op`` the tensor holds the result once the handle's ``wait()`` has run.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_, None
    input_ = input_.contiguous()
    if world_size == 2 or not input_.is_floating_point():
        return input_, dist.all_reduce(input_, group=group, async_op=async_op)
    flat = input_.view(-1)
    numel = flat.numel()
    padded_numel = -(-numel // world_size) * world_size
    scratch = _Scratch(async_op)
    if padded_numel == numel:
        buffer = flat
    else:
        buffer = scratch.tensor((padded_numel,), flat.dtype, "padded")
        buffer[:numel].copy_(flat)
        buffer[numel:].zero_()
    chunk = scratch.tensor((padded_numel // world_size,), flat.dtype, "chunk")
    work, sum_partials = _launch_all_to_all(
        buffer, group, chunk, chunk.numel(), None, scratch, async_op
    )

    def finish():
        sum_partials()
        dist.all_gather_into_tensor(buffer, chunk, group=group)
        if buffer is not flat:
            flat.copy_(buffer[:numel])
        scratch.release()

    return input_, _finish_or_defer(work, finish, async_op, buffer)
