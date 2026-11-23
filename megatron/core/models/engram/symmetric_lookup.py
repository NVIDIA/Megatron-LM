# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Symmetric (GIN/LSA) lookup for row-sharded Engram tables.

Ported from NVIDIA/Megatron-LM#7224 by Hongxiao Bai (hxbai), commits 610449cd5 and 906ccd59e,
and reduced to a synchronous lookup inside the Engram layer: the deduplicated request routing,
the lookup ticket, its autograd bridge and the symmetric branch of the gradient delivery. The
prefetch budget, deferred gradients, MXFP8 tables and the HybridEP backend are not part of it.

The device transport, ``megatron.core.ops.engram``, imports cutlass and nccl at module import.
This module reaches it only through :func:`require_symmetric_transport`, so importing the
Engram package loads neither. The Triton row accumulation of the backward,
``megatron.core.ops.engram_row_accumulation``, is bound by :func:`select_row_accumulation`
when a lookup is constructed.

One lookup runs on every rank of the row group, on the current stream, in the same order::

    ensure_params_ready(tables)                 # publish pending DDP parameter gathers
    unique, inverse = unique(table * max(table_sizes) + hash_ids)    # host sync 1
    owner, local row = closed form over the module's table_row_base / table_row_remainder
    requests[U, 2] = (table, local row) grouped by owner; send[N, T] = counts per (owner, table)
    row for peer p = send[p] ++ [changed, error] ++ (offset, stride) of each local table
    rows = transport.exchange(one row per peer)                       # every lookup, every rank
    copy send and the received counts and flags to the host once     # host sync 2
    an error flag on any rank raises on every rank; a change flag re-registers on every rank
    metadata[T, N, 2] = received (offset, stride); rebuilt on the device every lookup
    payload = transport.lookup(requests)   # barrier, LSA reads, GIN request/row exchange, barrier

Backward sums the gradients of duplicate positions in FP32, sends (table, row, gradient)
packets to the owners through one exchange and adds them, row-sparse, into ``main_grad``. The
table receives a zero-copy placeholder gradient with ``grad_added_to_main_grad`` set, so the
DDP hook releases it unread. Without ``main_grad``, or under ``create_graph``, the owner builds
a dense table gradient instead and clears the flag, so the hook adds it.

Every rank of a row group must issue the same transport calls in the same order: the barrier
and exchange kernels spin-wait on peers and cannot time out. Lookups therefore never skip a
call on an empty rank, and whether gradients are exchanged depends only on configuration
(``requires_grad`` of the tables), which startup checks is the same on every rank of a row
group.

Nothing bounds that wait when a peer stops issuing calls: a rank that raises between two
transport calls (out of memory in a lookup, say) or dies leaves its peers spinning, and no
PyTorch watchdog sees it, since the transport issues no ``torch.distributed`` work. The failing
rank aborts its transports on the way out (``pretrain``), so it exits promptly; the spinning
peers then end only when the launcher stops them, as torchrun's agent or
``srun --kill-on-bad-exit`` do when one process fails. A silent peer death needs a hang
detector (such as the NVIDIA Resiliency Extension's) or the job's time limit.
"""

from __future__ import annotations

import ctypes
import logging
import os
import socket
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Callable, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor

from megatron.core.ops._backends import require
from megatron.core.utils import (
    ensure_params_ready,
    get_pg_rank,
    get_pg_size,
    nvtx_range_pop,
    nvtx_range_push,
)

from .distributed_embedding import EPShardedMultiTableEmbedding

logger = logging.getLogger(__name__)

_NEEDED_BY = "Engram symmetric lookup"


# --- EngramConfig and table-sharding coupling ------------------------------------------------
# The only places this module reads EngramConfig fields or the table-sharding groups.


def _communication_backend(engram_config) -> str:
    """The resolved ``EngramConfig.communication_backend``: ``"symmetric"`` or ``"nccl"``."""
    backend = engram_config.communication_backend
    if backend not in ("symmetric", "nccl"):
        raise ValueError(
            f"Engram communication_backend must be resolved to 'symmetric' or 'nccl', got "
            f"{backend!r}"
        )
    return backend


def _workspace_bytes(engram_config) -> int:
    """``EngramConfig.symmetric_workspace_mb`` in bytes."""
    return engram_config.symmetric_workspace_mb * 2**20


def _engram_row_group(pg_collection, engram_config):
    """The row-shard group of the Engram tables on this rank."""
    from .table_sharding import resolve_engram_groups

    row_group, _, _ = resolve_engram_groups(pg_collection, engram_config)
    return row_group


def _nvlink_domain() -> tuple:
    """This rank's NVLink domain: its GPU fabric clique, or its host.

    The transport compares it with NCCL's LSA team to warn when that team is smaller than the
    NVLink domain.

    On multi-node NVLink systems (such as GB200 and GB300 NVL72), NVML reports the fabric
    clique of each GPU, and NCCL keeps the traffic between GPUs of one clique on NVLink.
    Elsewhere, or with NCCL_MNNVL_ENABLE=0, NVLink ends at the host.
    """
    if os.environ.get("NCCL_MNNVL_ENABLE", "").strip() != "0":
        clique = _gpu_fabric_clique()
        if clique is not None:
            return ("fabric", *clique)
    return ("host", socket.gethostname())


def _gpu_fabric_clique() -> Optional[Tuple[str, int]]:
    """The (cluster UUID, clique ID) of this rank's GPU in an NVLink fabric, or None."""
    if not torch.cuda.is_available():
        return None
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # pylint: disable=broad-except
        return None
    try:
        uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        if hasattr(pynvml, "nvmlDeviceGetGpuFabricInfoV"):
            info = pynvml.c_nvmlGpuFabricInfoV_t()
            if not info.version:
                info.version = pynvml.nvmlGpuFabricInfo_v2
            pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(info))
        else:
            info = pynvml.c_nvmlGpuFabricInfo_t()
            pynvml.nvmlDeviceGetGpuFabricInfo(handle, ctypes.byref(info))
        if (
            info.state != pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
            or info.status != pynvml.NVML_SUCCESS
        ):
            return None
        # Read the raw bytes: a c_char array field stops at its first zero byte.
        uuid_field = type(info).clusterUuid
        cluster = ctypes.string_at(ctypes.addressof(info) + uuid_field.offset, uuid_field.size)
        return cluster.hex(), int(info.cliqueId)
    except Exception:  # pylint: disable=broad-except
        # The driver or NVML may not support fabric queries; the host is the fallback domain.
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # pylint: disable=broad-except
            pass


# PyTorch's allocator configuration variables in the order it reads them: it parses the first
# one that is set, even when empty, and ignores the others (c10/core/AllocatorConfig.cpp).
_ALLOC_CONF_VARIABLES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def expandable_segments_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the PyTorch caching allocator in ``environ`` (default: this process) uses
    expandable segments.

    The symmetric transport registers the tables' storage as NCCL windows, which needs CUDA VMM
    allocations; with PyTorch's caching allocator, that means expandable segments. This reads
    the variable PyTorch reads, so launch checks and test skips agree with the allocator. Eager
    window registration at start-up stays the authoritative check.
    """
    environ = os.environ if environ is None else environ
    conf = next((environ[name] for name in _ALLOC_CONF_VARIABLES if name in environ), "")
    enabled = False
    for option in conf.split(","):
        key, _, value = option.partition(":")
        if key.strip() == "expandable_segments":
            # A later setting of the key overrides an earlier one, as in PyTorch's parser.
            enabled = value.strip() == "True"
    return enabled


# --- Backend selection -------------------------------------------------------------------------


def require_symmetric_transport() -> ModuleType:
    """Import the device transport module and check the libraries of the symmetric lookup.

    Raises ``ImportError`` naming the Engram symmetric lookup when CuTe DSL, NCCL4Py, the NCCL
    runtime or its device bitcode, or Triton is missing or mismatched. Called before any table
    is allocated and again before any transport is created; nothing falls back to NCCL
    silently.
    """
    module = require(
        "megatron.core.ops.engram", "EngramDeviceTransport", "check_runtime", needed_by=_NEEDED_BY
    )
    module.check_runtime()
    # The Triton row writer of the backward, which every lookup binds at construction.
    require("megatron.core.ops.engram_row_accumulation", "accumulate_rows", needed_by=_NEEDED_BY)
    return module


def select_row_accumulation() -> Optional[Callable[[Tensor, Tensor, Tensor, Tensor], bool]]:
    """Choose the writer of the backward's row sums once, when a lookup is constructed.

    On a host with CUDA this is the Triton kernel of
    ``megatron.core.ops.engram_row_accumulation``, and a missing Triton raises ``ImportError``
    naming the Engram symmetric lookup. :func:`require_symmetric_transport` makes the same
    check before any table is allocated. A host without CUDA has no device for that kernel and
    gets ``None``: :func:`_accumulate_embedding_rows` then sums the rows with PyTorch.
    """
    if not torch.cuda.is_available():
        return None
    return require(
        "megatron.core.ops.engram_row_accumulation", "accumulate_rows", needed_by=_NEEDED_BY
    ).accumulate_rows


def _module_call(embedding: EPShardedMultiTableEmbedding) -> Callable[[Tensor], Tensor]:
    # The bound __call__ rather than forward: calling it is calling the module, hooks included,
    # exactly as the Engram layer did before the lookup became selectable.
    return embedding.__call__


def select_engram_lookup(
    engram_config, row_group
) -> Callable[[EPShardedMultiTableEmbedding], Callable[[Tensor], Tensor]]:
    """Choose the table lookup once, before the Engram tables are allocated.

    Returns a factory that binds the lookup to the built tables: the module's own call (its
    all-to-all ``forward``) for the ``nccl`` backend, or a :class:`SymmetricEngramLookup` for
    ``symmetric``. The symmetric backend checks its dependencies here, so a missing library
    fails before any parameter exists. A single-rank row group owns every row and needs no
    communication, so it keeps the module's local lookup with either backend.
    """
    if _communication_backend(engram_config) == "nccl":
        return _module_call
    require_symmetric_transport()
    if get_pg_size(row_group) == 1:
        return _module_call
    return SymmetricEngramLookup


# --- Count exchange ----------------------------------------------------------------------------


class EngramExchangePlan(NamedTuple):
    """This rank's side of one row exchange.

    ``send_splits`` and ``recv_splits`` are rows per peer on the host; ``splits`` is the same
    information on the device as a contiguous ``[N, 4]`` int64 tensor of (send start, send
    rows, receive start, receive rows) per peer, built without any host-to-device copy.
    """

    send_splits: tuple
    recv_splits: tuple
    splits: Tensor

    @classmethod
    def from_counts(
        cls, send: Tensor, recv: Tensor, send_splits: Sequence[int], recv_splits: Sequence[int]
    ) -> "EngramExchangePlan":
        """Build a plan from device rows-per-peer tensors and their host copies."""
        splits = torch.stack((send.cumsum(0) - send, send, recv.cumsum(0) - recv, recv), dim=1)
        return cls(tuple(send_splits), tuple(recv_splits), splits)

    def reverse(self) -> "EngramExchangePlan":
        """The plan of the opposite direction (returning rows to their requesters)."""
        return EngramExchangePlan(self.recv_splits, self.send_splits, self.splits.roll(2, 1))


@lru_cache(maxsize=None)
def _uniform_plan(size: int, device: torch.device) -> EngramExchangePlan:
    """One row to and from every peer."""
    ones = torch.ones(size, dtype=torch.int64, device=device)
    return EngramExchangePlan.from_counts(ones, ones, (1,) * size, (1,) * size)


@lru_cache(maxsize=256)
def _layout_tensor(layout: tuple, device: torch.device) -> Tensor:
    """The (offset, stride) pairs of a layout on ``device``, copied once per distinct layout.

    A host-to-device copy of pageable memory may synchronize the stream, so it must not run on
    every lookup. Layouts change only when a table storage changes.
    """
    return torch.tensor([value for pair in layout for value in pair], dtype=torch.int64).to(device)


class LookupCounts(NamedTuple):
    """Result of one count exchange."""

    plan: EngramExchangePlan
    # Requests this rank receives for each of its tables, summed over all requesters.
    table_counts: tuple
    # [T, N, 2] int64 (byte offset, byte row stride) of every peer's tables, on the device.
    metadata: Tensor
    # Row ranks that reported a storage change or a failure to describe their tables.
    changed_ranks: tuple
    error_ranks: tuple


def exchange_lookup_counts(
    transport, send_counts: Tensor, *, changed: bool, error: bool, layout: tuple
) -> LookupCounts:
    """Exchange one count row with every peer of the row group and copy it to the host once.

    The row this rank sends to peer ``p`` holds the requests it sends ``p`` per table
    (``send_counts[p]``), its change and error flags, and the (offset, stride) of each of its
    tables; ``3 T + 2`` int64 words. The flags make window registration a collective decision
    without a host collective, and the offsets become the lookup kernel's metadata.
    """
    size, num_tables = send_counts.shape
    row = send_counts.new_empty((size, 3 * num_tables + 2))
    row[:, :num_tables].copy_(send_counts)
    row[:, num_tables].fill_(int(changed))
    row[:, num_tables + 1].fill_(int(error))
    row[:, num_tables + 2 :].copy_(
        _layout_tensor(tuple(layout), send_counts.device).expand(size, 2 * num_tables)
    )
    received = transport.exchange(row, _uniform_plan(size, send_counts.device))
    recv_counts = received[:, :num_tables]
    # The only device-to-host copy of the lookup: split sizes, owner table counts and flags.
    host = torch.cat((send_counts, received[:, : num_tables + 2]), dim=1).cpu()
    host_recv = host[:, num_tables : 2 * num_tables]
    plan = EngramExchangePlan.from_counts(
        send_counts.sum(dim=1),
        recv_counts.sum(dim=1),
        host[:, :num_tables].sum(dim=1).tolist(),
        host_recv.sum(dim=1).tolist(),
    )
    metadata = (
        received[:, num_tables + 2 :].reshape(size, num_tables, 2).transpose(0, 1).contiguous()
    )
    return LookupCounts(
        plan,
        tuple(host_recv.sum(dim=0).tolist()),
        metadata,
        tuple(host[:, 2 * num_tables].nonzero().flatten().tolist()),
        tuple(host[:, 2 * num_tables + 1].nonzero().flatten().tolist()),
    )


def prepare_table_windows(transport, key, tables: Sequence[Tensor], send_counts: Tensor):
    """Run the count exchange of one lookup and keep every rank's table windows current.

    Describes this rank's table storage, exchanges the count rows, raises on every rank when
    any rank could not describe its tables, and re-registers the windows on every rank when any
    rank's storage changed. Returns the :class:`LookupCounts` of the exchange.
    """
    changed, error, layout = transport.describe_tables(key, tables)
    counts = exchange_lookup_counts(
        transport, send_counts, changed=changed, error=error is not None, layout=layout
    )
    if counts.error_ranks:
        if error is not None:
            logger.error(
                "Engram row rank %d cannot use its table storage: %s", transport.rank, error
            )
        reason = f"; on this rank: {error}" if error is not None else ""
        raise RuntimeError(
            f"Cannot prepare Engram table windows on row ranks {list(counts.error_ranks)}"
            f"{reason}. Each failing rank logs its own cause."
        )
    if counts.changed_ranks:
        transport.register_tables(key, counts.metadata)
    return counts


def route_unique_rows(embedding: EPShardedMultiTableEmbedding, rows: Tensor, table_ids: Tensor):
    """Owner-sorted requests and per-(owner, table) counts for deduplicated global rows.

    Uses the balanced contiguous partition of PR 8225's ``EPShardedMultiTableEmbedding``: ranks
    below ``remainder`` own ``base + 1`` rows of a table and the rest own ``base``, so the owner
    of a global row follows in closed form.

    Returns ``(requests, order, send_counts)``: ``[U, 2]`` int64 (table, owner-local row)
    grouped by owner, the owner order of the input rows, and ``[N, T]`` int64 counts.
    """
    base = embedding.table_row_base.index_select(0, table_ids)
    remainder = embedding.table_row_remainder.index_select(0, table_ids)
    large_region_end = (base + 1) * remainder
    owners = torch.where(
        rows < large_region_end,
        rows // (base + 1),
        remainder + (rows - large_region_end) // base.clamp_min(1),
    )
    local_rows = rows - (owners * base + torch.minimum(owners, remainder))
    order = torch.argsort(owners, stable=True)
    sorted_owners = owners.index_select(0, order)
    requests = torch.stack(
        (table_ids.index_select(0, order), local_rows.index_select(0, order)), dim=-1
    )
    num_tables = len(embedding.tables)
    send_counts = torch.bincount(
        sorted_owners * num_tables + requests[:, 0], minlength=embedding.ep_size * num_tables
    ).view(embedding.ep_size, num_tables)
    return requests, order, send_counts


# --- Gradient delivery -------------------------------------------------------------------------


@dataclass
class EngramLookupTicket:
    """The route of one lookup, kept by autograd until its backward."""

    transport: object
    # [U, 2] int64 (table, owner-local row) grouped by owner, as sent to the owners.
    requests: Tensor
    plan: EngramExchangePlan
    # Requests this rank serves per table.
    table_counts: tuple
    # Shape of the hash ids of the lookup.
    shape: tuple
    # The row writer the lookup bound at construction (see select_row_accumulation).
    accumulate_rows: Optional[Callable[[Tensor, Tensor, Tensor, Tensor], bool]] = None


def _accumulate_embedding_rows(target, rows, gradients, accumulate_rows=None):
    """Deterministically sum duplicates in FP32 and add only touched rows to ``target``.

    ``accumulate_rows`` is the writer a lookup bound at construction
    (:func:`select_row_accumulation`). It declines the inputs its kernel does not cover (CPU
    tensors, FP64, a column stride other than one); those, and a host without CUDA, take the
    PyTorch path below.
    """
    if not rows.numel():
        return
    order = torch.argsort(rows, stable=True)
    if accumulate_rows is not None and accumulate_rows(target, rows, order, gradients):
        return
    ids, counts = torch.unique_consecutive(rows[order], return_counts=True)
    values = gradients[order].to(torch.float64 if target.dtype == torch.float64 else torch.float32)
    values = torch.segment_reduce(values, "sum", lengths=counts)
    target.index_copy_(0, ids, (target.index_select(0, ids) + values).to(target.dtype))


def _embedding_row_gradients(row_ids, grad_output, shape, dtype):
    """Accumulate repeated row IDs in FP32, then cast once per touched row."""
    grad_weight = torch.zeros(shape, dtype=dtype, device=grad_output.device)
    if row_ids.numel():
        order = torch.argsort(row_ids, stable=True)
        sorted_rows = row_ids.index_select(0, order)
        accumulation_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        sorted_grads = grad_output.index_select(0, order).to(accumulation_dtype)
        unique_rows, counts = torch.unique_consecutive(sorted_rows, return_counts=True)
        row_grads = torch.segment_reduce(sorted_grads, "sum", lengths=counts)
        grad_weight.index_copy_(0, unique_rows, row_grads.to(dtype))
    return grad_weight


def _accumulates_into_main_grad(weight: Tensor) -> bool:
    """Whether the owner may add the table gradient straight into ``weight.main_grad``.

    That needs Megatron DDP's gradient buffer, whose backward hook skips a gradient already
    added (``grad_added_to_main_grad``), and gradient mode off: under ``create_graph`` autograd
    needs a real gradient.

    The backward cannot tell ``.backward()`` from ``torch.autograd.grad``. Without
    ``create_graph``, ``torch.autograd.grad`` over tables that have ``main_grad`` therefore
    returns the placeholder (the table values) while the real gradient lands in ``main_grad``.
    Differentiate such tables with ``.backward()``, or pass ``create_graph=True`` for a real
    gradient.
    """
    return (
        getattr(weight, "main_grad", None) is not None
        and hasattr(weight, "grad_added_to_main_grad")
        and not torch.is_grad_enabled()
    )


def deliver_row_gradients(
    ticket: EngramLookupTicket,
    unique_grad: Tensor,
    weights: Sequence[Tensor],
    needs_grad: Sequence[bool],
) -> list:
    """Send unique-row gradients to their owners and apply them; one exchange per backward.

    A packet row holds the 16-byte (table, row) request followed by its FP32 gradient. Owners
    sort the received rows by table (stably, so the requester order is kept) and add them into
    ``main_grad`` with the sorted-segment Triton writer, returning a placeholder; otherwise
    they return a dense gradient in the table dtype. Tables that need no gradient get ``None``.
    """
    packet = torch.cat((ticket.requests.view(torch.uint8), unique_grad.view(torch.uint8)), dim=1)
    received = ticket.transport.exchange(packet, ticket.plan)
    requests = received[:, :16].contiguous().view(torch.int64)
    gradients = received[:, 16:].contiguous().view(unique_grad.dtype)
    rows = requests[:, 1]
    if len(weights) > 1:
        table_order = torch.argsort(requests[:, 0], stable=True)
        rows = rows.index_select(0, table_order)
        gradients = gradients.index_select(0, table_order)
    result, start = [], 0
    for weight, count, needed in zip(weights, ticket.table_counts, needs_grad):
        local_rows, local_grads = rows[start : start + count], gradients[start : start + count]
        start += count
        if not needed:
            result.append(None)
        elif _accumulates_into_main_grad(weight):
            if getattr(weight, "zero_out_wgrad", False):
                raise RuntimeError(
                    "Engram tables accumulate into main_grad and cannot set zero_out_wgrad"
                )
            if weight.grad is not None:
                # The placeholder aliases the table: accumulating onto a gradient the DDP hook
                # did not release would write into the table itself.
                raise RuntimeError("Engram table .grad was not released by the DDP hook")
            _accumulate_embedding_rows(
                weight.main_grad, local_rows, local_grads, ticket.accumulate_rows
            )
            weight.grad_added_to_main_grad = True
            # AccumulateGrad steals this layout-matching view without a copy; DDP's hook sees
            # grad_added_to_main_grad and drops it.
            result.append(weight.detach())
        else:
            if hasattr(weight, "grad_added_to_main_grad"):
                # A dense gradient: the DDP hook must add it, even if an earlier microbatch
                # or a captured step left the flag set.
                weight.grad_added_to_main_grad = False
            result.append(
                _embedding_row_gradients(local_rows, local_grads, weight.shape, weight.dtype)
            )
    return result


class _SymmetricEngramLookup(torch.autograd.Function):
    """Bridge rows fetched without autograd to the owning table parameters."""

    @staticmethod
    def forward(ctx, ticket, payload, mapping, *weights):
        """Restore token/head order from the deduplicated, owner-grouped rows."""
        ctx.ticket = ticket
        ctx.weights = weights
        ctx.save_for_backward(mapping)
        return payload.index_select(0, mapping).view(*ticket.shape, payload.shape[1])

    @staticmethod
    def backward(ctx, grad_output):
        """Reduce duplicate positions per unique row, then deliver to the owners."""
        needs_grad = ctx.needs_input_grad[3:]
        if not any(needs_grad):
            return (None, None, None, *([None] * len(needs_grad)))
        message = "engram.lookup.backward"
        nvtx_range_push(message)
        try:
            ticket = ctx.ticket
            (mapping,) = ctx.saved_tensors
            width = ctx.weights[0].shape[1]
            grad = grad_output.reshape(-1, width)
            dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
            unique_grad = torch.zeros(
                (ticket.requests.shape[0], width), device=grad.device, dtype=dtype
            )
            _accumulate_embedding_rows(unique_grad, mapping, grad, ticket.accumulate_rows)
            gradients = deliver_row_gradients(ticket, unique_grad, ctx.weights, needs_grad)
        finally:
            nvtx_range_pop(message)
        return (None, None, None, *gradients)


# --- Lookup ------------------------------------------------------------------------------------

# One transport per row group: its communicator, workspace and windows are shared by every
# Engram layer of the rank.
_TRANSPORTS: dict = {}
_INITIALIZED = False


def _transport_for(group):
    transport = _TRANSPORTS.get(group)
    if transport is None:
        raise RuntimeError(
            "The Engram symmetric lookup has no transport for its row group: call "
            "initialize_engram_transports() after wrapping the model"
        )
    return transport


class SymmetricEngramLookup:
    """Synchronous symmetric lookup bound to one ``EPShardedMultiTableEmbedding``.

    A plain object rather than a submodule, so the tables keep their state-dict keys. The
    module's row group (``ep_group``) selects the transport, which
    :func:`initialize_engram_transports` creates once the model is wrapped.

    Tables wrapped by Megatron DDP (with ``main_grad``) receive their gradients through
    ``.backward()``. ``torch.autograd.grad`` over them needs ``create_graph=True``: without it,
    it returns the table values as their gradient (see :func:`_accumulates_into_main_grad`).
    """

    def __init__(self, embedding: EPShardedMultiTableEmbedding) -> None:
        self.embedding = embedding
        self.accumulate_rows = select_row_accumulation()

    def __call__(self, hash_ids: Tensor) -> Tensor:
        """Return the rows of ``hash_ids [..., T]`` as ``[..., T, embedding_dim]``."""
        embedding = self.embedding
        num_tables = len(embedding.tables)
        if hash_ids.shape[-1] != num_tables:
            raise ValueError(f"Expected {num_tables} Engram hash heads, got {hash_ids.shape[-1]}.")
        transport = _transport_for(embedding.ep_group)
        weights = tuple(table.weight for table in embedding.tables)
        # The tables are read through their windows, bypassing the modules' forward pre-hooks.
        ensure_params_ready(weights)

        message = "engram.lookup.owner-map"
        nvtx_range_push(message)
        try:
            # Encode (table, row) without materializing an [N, num_shards] comparison.
            stride = max(embedding.table_sizes)
            table_ids = (
                torch.arange(num_tables, device=hash_ids.device).expand(hash_ids.shape).reshape(-1)
            )
            unique, inverse = torch.unique(
                hash_ids.reshape(-1).long() + table_ids * stride, sorted=True, return_inverse=True
            )
            requests, order, send_counts = route_unique_rows(
                embedding, unique % stride, unique // stride
            )
        finally:
            nvtx_range_pop(message)

        message = "engram.lookup.count-exchange"
        nvtx_range_push(message)
        try:
            counts = prepare_table_windows(transport, embedding, weights, send_counts)
        finally:
            nvtx_range_pop(message)

        message = "engram.lookup.rows"
        nvtx_range_push(message)
        try:
            payload = transport.lookup(embedding, weights, requests, counts.plan, counts.metadata)
        finally:
            nvtx_range_pop(message)

        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.numel(), device=order.device)
        ticket = EngramLookupTicket(
            transport,
            requests,
            counts.plan,
            counts.table_counts,
            tuple(hash_ids.shape),
            self.accumulate_rows,
        )
        return _SymmetricEngramLookup.apply(
            ticket, payload.view(weights[0].dtype), inverse_order[inverse], *weights
        )


# --- Lifecycle ---------------------------------------------------------------------------------


class _StartReport(NamedTuple):
    """What one rank reports in the first startup gather over the default group."""

    # _table_signature() of each symmetric Engram embedding of the rank, in module order.
    tables: tuple
    # Global ranks of the rank's row group; None without symmetric Engram tables.
    members: Optional[tuple]
    # The NCCL unique id, created on the first rank of each row group.
    unique_id: Optional[bytes]
    # The rank's NVLink domain (_nvlink_domain), or its host when that cannot be read.
    nvlink_domain: tuple
    # Why the rank cannot start, or None.
    problem: Optional[str]


def _table_signature(embedding: EPShardedMultiTableEmbedding) -> tuple:
    """What every rank of a row group must agree on for one embedding.

    The count rows, row packets and gradient packets of its lookups are sized from the table
    count, width and dtype, and its backward exchanges gradients only if a table requires them.
    """
    weights = [table.weight for table in embedding.tables]
    return (
        embedding.table_sizes,
        embedding.embedding_dim,
        tuple(str(weight.dtype) for weight in weights),
        tuple(weight.requires_grad for weight in weights),
    )


def _gather_world(item) -> list:
    """``item`` from every rank of the default group, in rank order."""
    if not torch.distributed.is_initialized():
        return [item]
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, item)
    return gathered


def _problem(failure: Optional[BaseException]) -> Optional[str]:
    return None if failure is None else f"{type(failure).__name__}: {failure}"


def _raise_on_every_rank(problems: list, failure: Optional[BaseException], step: str) -> None:
    """Raise on every rank if any rank of the default group reported a problem."""
    failed = [rank for rank, problem in enumerate(problems) if problem is not None]
    if not failed:
        return
    reason = f"; on this rank: {_problem(failure)}" if failure is not None else ""
    raise RuntimeError(
        f"Engram symmetric transports cannot {step} on ranks {failed}{reason}. Each failing "
        "rank logs its own cause."
    ) from failure


def initialize_engram_transports(model_chunks, pg_collection, engram_config) -> None:
    """Create this rank's symmetric transport after the model is wrapped by DDP.

    Collective over the default (WORLD) group and called on every rank, including pipeline
    stages without Engram layers, which create nothing. Startup has four steps, and each ends
    with an object gather over the default group, so a failure on any rank raises on every
    rank instead of leaving its peers in a collective or a device spin-wait:

    1. local: find the symmetric Engram tables, check that DDP did not put their buffers in
       torch_memory_saver regions, resolve the row group, check the toolchain, create the NCCL
       unique id on each row group's first rank and read the rank's NVLink domain, which sets
       the LSA team the transport expects. The gather also checks that every rank of a row
       group holds the same tables;
    2. local, on ranks with tables: create the row group's transport and allocate its
       workspace, the start-up's last step that can fail on one rank alone;
    3. on ranks with tables, initialize the transport: blocking NCCL calls over the row group
       (communicator, device communicator, workspace window) that start no device spin-wait.
       A rank that fails inside one of them is NCCL's to report, and may leave its peers
       waiting in it;
    4. publish the workspace and register every table window eagerly, so unregisterable table
       storage fails at startup.

    A failed start aborts what it created and leaves no state behind. Returns at once unless
    the symmetric backend is configured.
    """
    global _INITIALIZED
    if engram_config is None or _communication_backend(engram_config) != "symmetric":
        return
    if _INITIALIZED:
        raise RuntimeError(
            "Engram transports are already initialized; call close_engram_transports() first"
        )
    _INITIALIZED = True
    try:
        _start_engram_transports(model_chunks, pg_collection, engram_config)
    except BaseException:
        # Abort, without any device barrier, whatever this call created. The start failure is
        # the error to report, even if an abort fails too.
        try:
            close_engram_transports()
        except Exception:  # pylint: disable=broad-except
            logger.warning("Cannot abort the Engram symmetric transports", exc_info=True)
        raise


def _sharded_embeddings(chunk) -> list:
    return [
        module
        for module in chunk.modules()
        if isinstance(module, EPShardedMultiTableEmbedding) and module.ep_size > 1
    ]


def _check_table_buffers(chunk) -> None:
    """Reject DDP buffers that torch_memory_saver allocates without a CPU backup.

    It remaps their pages at the same virtual addresses, which the registered table windows
    cannot detect. The command line rejects the two flags too; this covers configurations that
    build the DDP config themselves.
    """
    ddp_config = getattr(chunk, "ddp_config", None)
    for flag in ("disable_param_buffers_cpu_backup", "disable_grad_buffers_cpu_backup"):
        if getattr(ddp_config, flag, False):
            raise ValueError(
                f"The Engram symmetric lookup does not support DDP's {flag}: torch_memory_saver "
                "remaps those buffers at the same virtual addresses, which the registered table "
                "windows cannot detect."
            )


def _start_engram_transports(model_chunks, pg_collection, engram_config) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    embeddings, tables, row_group, members = [], (), None, None
    transport_module = unique_id = transport = failure = None
    domain = ("host", socket.gethostname())
    try:
        for chunk in model_chunks:
            chunk_embeddings = _sharded_embeddings(chunk)
            if chunk_embeddings:
                _check_table_buffers(chunk)
            embeddings.extend(chunk_embeddings)
        tables = tuple(_table_signature(embedding) for embedding in embeddings)
        if embeddings:
            row_group = _engram_row_group(pg_collection, engram_config)
            if any(embedding.ep_group is not row_group for embedding in embeddings):
                raise RuntimeError("Engram tables are not sharded over the Engram row group")
            members = tuple(torch.distributed.get_process_group_ranks(row_group))
            transport_module = require_symmetric_transport()
            if get_pg_rank(row_group) == 0:
                unique_id = transport_module.create_unique_id()
            domain = _nvlink_domain()
    except Exception as exc:
        failure = exc
        logger.error("Engram symmetric transport cannot start on rank %d", rank, exc_info=exc)
    reports = _gather_world(_StartReport(tables, members, unique_id, domain, _problem(failure)))
    _raise_on_every_rank([report.problem for report in reports], failure, "start")
    mixed = sorted(
        {
            report.members
            for report in reports
            if report.members is not None
            and len({reports[member].tables for member in report.members}) > 1
        }
    )
    if mixed:
        raise RuntimeError(
            f"Engram row groups {mixed} mix ranks with different symmetric Engram tables; every "
            "rank of a row group must hold the same Engram layers, in the same order, with the "
            "same table sizes, width, dtype and requires_grad"
        )

    if embeddings:
        try:
            # NCCL's LSA team should hold every row-group rank of this rank's NVLink domain (its
            # GPU fabric clique on multi-node NVLink systems); the transport warns when it is
            # smaller, since those peers are then reached over GIN.
            transport = transport_module.EngramDeviceTransport(
                row_group,
                unique_id=reports[torch.distributed.get_global_rank(row_group, 0)].unique_id,
                workspace_bytes=_workspace_bytes(engram_config),
                expected_lsa_ranks=sum(
                    reports[member].nvlink_domain == domain for member in members
                ),
            )
            _TRANSPORTS[row_group] = transport
            # The workspace allocation can fail on one rank alone (out of memory, say): every
            # rank learns of it here, before any rank enters NCCL calls that wait for the
            # whole row group.
            transport.prepare()
        except Exception as exc:
            failure = exc
            logger.error(
                "Engram symmetric transport cannot allocate its workspace on rank %d",
                rank,
                exc_info=exc,
            )
    _raise_on_every_rank(_gather_world(_problem(failure)), failure, "allocate their workspace")

    if transport is not None:
        try:
            transport.initialize()
        except Exception as exc:
            failure = exc
            logger.error(
                "Engram symmetric transport cannot initialize on rank %d", rank, exc_info=exc
            )
    _raise_on_every_rank(_gather_world(_problem(failure)), failure, "initialize")

    try:
        if transport is not None:
            # Publish the zeroed workspace flags before any peer writes into them.
            transport.barrier()
            # Eager registration: the same modules in the same order on every rank of the row
            # group, as the first gather checked.
            for embedding in embeddings:
                weights = tuple(table.weight for table in embedding.tables)
                send_counts = torch.zeros(
                    (embedding.ep_size, len(weights)),
                    dtype=torch.int64,
                    device=embedding.table_row_base.device,
                )
                prepare_table_windows(transport, embedding, weights, send_counts)
    except Exception as exc:
        failure = exc
        logger.error(
            "Engram symmetric transport cannot register table windows on rank %d",
            rank,
            exc_info=exc,
        )
    _raise_on_every_rank(_gather_world(_problem(failure)), failure, "register table windows")


def close_engram_transports(clean: bool = False) -> None:
    """Close every Engram transport; a no-op when none is open.

    ``clean=True`` is collective (device barrier, synchronize, close windows, destroy) and
    belongs where every rank of a row group arrives together, such as the normal end of
    training. The default aborts the communicators without a device barrier, which is safe on
    any teardown path, including after a peer failed.

    Every transport is closed even when closing one raises: the ones after a failed clean
    close are aborted, since their device barrier might never complete then. The first
    failure is raised at the end.
    """
    global _INITIALIZED
    transports = list(_TRANSPORTS.values())
    _TRANSPORTS.clear()
    _INITIALIZED = False
    failure = None
    for transport in transports:
        try:
            transport.close(clean=clean and failure is None)
        except BaseException as exc:  # pylint: disable=broad-except
            # Re-raised below, once every transport is closed.
            failure = failure or exc
    if failure is not None:
        raise failure
