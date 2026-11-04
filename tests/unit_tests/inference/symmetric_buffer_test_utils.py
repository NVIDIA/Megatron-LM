# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Record operations on inference symmetric-memory buffers and check them for races.

Inference-optimized linears all-gather into and reduce-scatter out of shared symmetric buffers
with the multimem kernels of megatron.core.inference.communication.torch_symm_triton. A kernel
may write a buffer only once no peer can still read the bytes it writes (see
multimem_all_gather and multimem_reduce_scatter). record_symmetric_buffer_ops records those
kernels, and the collectives that synchronize a process group, in launch order, with the ranks
of each buffer's and each collective's group; find_symmetric_buffer_races checks the recording
against that rule.
"""

import contextlib
import dataclasses
import inspect
import sys
from typing import Callable, Iterator, Optional
from unittest import mock

import pytest
import torch

from megatron.core.inference.communication.torch_symm_triton import collectives, fused_collectives
from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

Ranks = frozenset[int]

ALL_GATHER = "all_gather"
REDUCE_SCATTER = "reduce_scatter"
FUSED_REDUCE_SCATTER_ALL_GATHER = "fused_reduce_scatter_all_gather"
SYNC = "sync"


@dataclasses.dataclass(frozen=True)
class BufferOp:
    """One recorded operation.

    Attributes:
        kind: ALL_GATHER (multimem_all_gather or multimem_all_gather_fused), REDUCE_SCATTER,
            FUSED_REDUCE_SCATTER_ALL_GATHER, or SYNC (a collective that synchronizes a group).
        buffer: Multicast pointer of the symmetric buffer; None for SYNC.
        barrier_before: Whether an all-gather barriered before writing; None otherwise.
        segment: (byte offset, bytes per rank) of the segments a kernel writes or reduces, or
            None for SYNC and for the fused three-tensor all-gather.
        ranks: Global ranks of the buffer's process group, or of the collective's for SYNC.
            None if unknown, which counts as every rank.
    """

    kind: str
    buffer: Optional[int] = None
    barrier_before: Optional[bool] = None
    segment: Optional[tuple[int, int]] = None
    ranks: Optional[Ranks] = None


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _group_ranks(group: Optional[torch.distributed.ProcessGroup]) -> Ranks:
    group = group if group is not None else torch.distributed.group.WORLD
    return frozenset(torch.distributed.get_process_group_ranks(group))


def _describe_multimem_op(name: str, arguments: dict, buffer_ranks: dict) -> BufferOp:
    buffer = arguments["symm_mem_hdl"].multicast_ptr
    ranks = buffer_ranks.get(buffer)
    if name == "multimem_all_gather":
        segment = (arguments["byte_offset"], _nbytes(arguments["input_tensor"]))
        return BufferOp(ALL_GATHER, buffer, arguments["barrier_before"], segment, ranks)
    if name == "multimem_all_gather_fused":
        return BufferOp(ALL_GATHER, buffer, arguments["barrier_before"], ranks=ranks)
    if name == "multimem_reduce_scatter":
        segment = (0, _nbytes(arguments["output_tensor"]))
        return BufferOp(REDUCE_SCATTER, buffer, segment=segment, ranks=ranks)
    segment = (0, _nbytes(arguments["residual_output_tensor"]))
    return BufferOp(FUSED_REDUCE_SCATTER_ALL_GATHER, buffer, segment=segment, ranks=ranks)


def _describe_collective(arguments: dict) -> Optional[BufferOp]:
    # An asynchronous collective synchronizes only when waited on, so it does not count.
    if arguments.get("async_op", False):
        return None
    if torch.distributed.get_world_size(arguments.get("group")) < 2:
        return None
    return BufferOp(SYNC, ranks=_group_ranks(arguments.get("group")))


_MULTIMEM_OPS = (
    (collectives, "multimem_all_gather"),
    (collectives, "multimem_all_gather_fused"),
    (collectives, "multimem_reduce_scatter"),
    (fused_collectives, "fused_multimem_rs_add_norm_ag"),
)
# Collectives every rank must enter before any completes. Megatron's mappings and Transformer
# Engine run these through NCCL.
_SYNCHRONIZING_COLLECTIVES = (
    "all_gather",
    "all_gather_into_tensor",
    "_all_gather_base",
    "all_reduce",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "_reduce_scatter_base",
)


def _patch_every_reference(stack: contextlib.ExitStack, replacements: dict) -> None:
    """Replace each original function, keyed by id, in every loaded module that refers to it."""
    for module in list(sys.modules.values()):
        try:
            namespace = vars(module)
        except TypeError:
            continue
        for name, value in list(namespace.items()):
            replacement = replacements.get(id(value))
            if replacement is not None:
                try:
                    stack.enter_context(mock.patch.object(module, name, replacement))
                except (AttributeError, TypeError):
                    pass  # A read-only (extension) module; nothing calls through it.


def _recorder(original: Callable, describe: Callable, ops: list) -> Callable:
    signature = inspect.signature(original)

    def record(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        op = describe(dict(arguments.arguments))
        if op is not None:
            ops.append(op)
        return original(*args, **kwargs)

    return record


def _buffer_group_recorder(buffer_ranks: dict) -> Callable:
    """Wrap SymmetricMemoryManager.get_buffer to note the ranks of each buffer's group."""
    original = SymmetricMemoryManager.get_buffer

    def get_buffer(key, process_group=None, size_mb=None):
        buffer = original(key, process_group=process_group, size_mb=size_mb)
        if process_group is not None and buffer.symm_mem_hdl is not None:
            buffer_ranks[buffer.symm_mem_hdl.multicast_ptr] = _group_ranks(process_group)
        return buffer

    return get_buffer


@contextlib.contextmanager
def record_symmetric_buffer_ops() -> Iterator[list[BufferOp]]:
    """Record the multimem kernels and synchronizing collectives launched inside the block.

    Every module's reference to the kernel wrappers and to the torch.distributed collectives is
    patched, so the recording does not depend on how callers import them. The real operations
    still run. A buffer's group is taken from the callers that fetch it from
    SymmetricMemoryManager with their process group, as the inference layers do before each
    operation.

    Yields:
        The list the BufferOps are appended to, in launch order.
    """
    ops: list[BufferOp] = []
    buffer_ranks: dict = {}
    replacements = {}
    for module, name in _MULTIMEM_OPS:
        original = getattr(module, name)
        describe = lambda arguments, name=name: _describe_multimem_op(name, arguments, buffer_ranks)
        replacements[id(original)] = _recorder(original, describe, ops)
    for name in _SYNCHRONIZING_COLLECTIVES:
        original = getattr(torch.distributed, name)
        replacements[id(original)] = _recorder(original, _describe_collective, ops)
    with contextlib.ExitStack() as stack:
        _patch_every_reference(stack, replacements)
        stack.enter_context(
            mock.patch.object(
                SymmetricMemoryManager,
                "get_buffer",
                staticmethod(_buffer_group_recorder(buffer_ranks)),
            )
        )
        yield ops


def _covers(sync: BufferOp, buffer: BufferOp) -> bool:
    """Whether sync's group contains every rank of buffer's group."""
    if sync.ranks is None:
        return True
    if buffer.ranks is None:
        return len(sync.ranks) == torch.distributed.get_world_size()
    return buffer.ranks <= sync.ranks


def find_symmetric_buffer_races(ops: list[BufferOp]) -> list[str]:
    """Describe each recorded write of a symmetric buffer that a peer may still be reading.

    Per buffer, the state after each operation is "synced" (a collective that synchronizes
    every rank of the buffer's group ran since the buffer's last operation), "gathered" (peers
    may read the whole buffer after an all-gather), or the segment of the last reduce-scatter
    (peers may still reduce their own segments of every rank's buffer: it has no exit barrier).
    Before the first operation the state is unknown, which is treated as "gathered". A
    collective over a group that leaves out a rank of the buffer's group does not synchronize
    the buffer. Then:

    - an all-gather without its barrier must find the buffer synced, or follow a reduce-scatter
      with the same segment;
    - a reduce-scatter's input, written into this rank's buffer before it starts, must not
      follow another reduce-scatter unless a synchronizing collective ran in between.
    """
    races = []
    state: dict = {}
    last_index: dict = {}
    for index, op in enumerate(ops):
        if op.kind == SYNC:
            continue
        previous = state.get(op.buffer, "gathered")
        since = last_index.get(op.buffer, -1) + 1
        if any(other.kind == SYNC and _covers(other, op) for other in ops[since:index]):
            previous = "synced"
        last_index[op.buffer] = index
        if op.kind == ALL_GATHER:
            safe = (
                op.barrier_before
                or previous == "synced"
                or (op.segment is not None and previous == (REDUCE_SCATTER, op.segment))
            )
            if not safe:
                races.append(
                    f"op {index}: all-gather {op.segment} without barrier after {previous}"
                )
            state[op.buffer] = "gathered"
        else:
            if isinstance(previous, tuple):
                races.append(f"op {index}: {op.kind} input written after reduce-scatter {previous}")
            if op.kind == REDUCE_SCATTER:
                state[op.buffer] = (REDUCE_SCATTER, op.segment)
            else:
                state[op.buffer] = "gathered"
    return races


def barrier_trace(ops: list[BufferOp]) -> list[tuple[str, Optional[bool]]]:
    """The recorded multimem operations as (kind, barrier_before) pairs, without the syncs."""
    return [(op.kind, op.barrier_before) for op in ops if op.kind != SYNC]


def skip_unless_nvls(process_group: torch.distributed.ProcessGroup) -> None:
    """Create the "tp" buffer on every rank, then skip on all ranks unless NVLS multicast works.

    Collective over process_group: every rank must call it.
    """
    handle = SymmetricMemoryManager.get_buffer("tp", process_group=process_group).symm_mem_hdl
    nvls = is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device()))
    nvls = torch.tensor(
        [float(nvls and handle is not None and bool(handle.multicast_ptr))], device="cuda"
    )
    torch.distributed.all_reduce(nvls, op=torch.distributed.ReduceOp.MIN, group=process_group)
    if nvls.item() != 1.0:
        pytest.skip("NVLS multicast unavailable")
