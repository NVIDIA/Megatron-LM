# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CuTe DSL and NCCL4Py device transport for the Engram symmetric (GIN/LSA) lookup.

Ported from NVIDIA/Megatron-LM#7224 by Hongxiao Bai (hxbai), commits 610449cd5 and 906ccd59e.
Every kernel and device helper, from ``_rows_tensor`` through ``_lookup``, is verbatim:

- ``_exchange`` moves opaque rows between the ranks of one row group. A ring of up to four
  channels, each peer transfer striped over up to 32 CTAs, with two receive slots per sender and
  GPU-side credits. Peers inside the NVLink (LSA) team are written directly; the others go
  through GIN puts from a local staging slot.
- ``_lookup`` reads table rows of LSA peers straight from their registered windows, or packs
  rows that remote (GIN) peers requested.
- ``_table_barrier`` is the group-wide device barrier that brackets every lookup.

The host class :class:`EngramDeviceTransport` is adapted for Megatron's synchronous lookup in
``megatron.core.models.engram.symmetric_lookup``:

- the NCCL unique id is passed in (gathered over the default group after the model is wrapped),
  so no torch collective ever runs on the row group;
- the communicator is configured for the device API alone (:func:`_communicator_config`): one
  CTA and no NVLS, as the transport never launches an NCCL kernel;
- start-up has a local step, :meth:`~EngramDeviceTransport.prepare` (the workspace
  allocation), whose outcome the caller agrees on before any rank enters the NCCL calls of
  :meth:`~EngramDeviceTransport.initialize`; that launches nothing that waits on a peer, so the
  caller can agree on its outcome on every rank before :meth:`~EngramDeviceTransport.barrier`
  publishes the workspace;
- table registration is split into :meth:`~EngramDeviceTransport.describe_tables` (local) and
  :meth:`~EngramDeviceTransport.register_tables` (collective, decided by the change and error
  flags that ride on the per-lookup count exchange, and ending with an exchange of each rank's
  registration outcome). Instead of one window per table, there is one window per table
  storage, shared by every lookup whose tables it holds on all ranks: the Engram tables of a
  model chunk sit in one DDP parameter buffer and need one window;
- an exchange plan carries only this rank's row counts per peer, on the host and as a device
  ``[N, 4]`` tensor, so no split list is copied to the device and no rank needs the global count
  matrix; every rank always launches every exchange;
- :meth:`~EngramDeviceTransport.close` aborts the communicator unless a clean, collective
  shutdown is requested, and also when a step of that shutdown fails.

This module imports cutlass and nccl at module import. The Engram package reaches it only
through ``megatron.core.ops._backends.require``. All calls use the caller's current stream and
must be issued in the same order on every rank of the row group. The barrier and exchange
kernels spin-wait on peers and cannot time out, so every rank launches each of them, even with
no rows, and a rank that skips or adds one hangs its peers. Only the lookup kernel never waits;
a rank without rows skips it.
"""

import logging
import math
from types import SimpleNamespace
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import nccl.core as nccl
import nccl.core.device.cute as nc
import torch
from cuda.bindings import driver as cuda
from cuda.core import Buffer
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import dsl_user_op
from nccl.core.device.cute.runtime import make_fake_dev_comm, make_fake_window
from nccl.core.interop import torch as nccl_torch
from packaging.version import Version

from megatron.core.ops._backends import installed_version

logger = logging.getLogger(__name__)

# CuTe DSL 4.8.0 can fail repeated NCCL barrier compiles (NVIDIA/Megatron-LM#7224).
MIN_CUTE_DSL_VERSION = "4.8.1"
MIN_NCCL4PY_VERSION = "0.6.0"


def check_runtime() -> None:
    """Raise ``ImportError`` unless the CuTe DSL and NCCL4Py versions and the NCCL runtime match.

    The NCCL device bitcode (``$NCCL_HOME/lib/libnccl_device.bc``) is linked into every CuTe
    kernel, so the runtime, the NCCL4Py headers and the bitcode must be the same NCCL version.
    This checks the runtime against the headers only; a missing or mismatched bitcode fails
    when :meth:`EngramDeviceTransport.initialize` compiles the kernels, on every rank alike.
    """
    cute_dsl = installed_version("cutlass", dist="nvidia-cutlass-dsl")
    if cute_dsl is None or cute_dsl < Version(MIN_CUTE_DSL_VERSION):
        raise ImportError(
            f"Engram symmetric requires nvidia-cutlass-dsl >= {MIN_CUTE_DSL_VERSION}; "
            f"found {cute_dsl}"
        )
    version = nccl.get_version()
    if version.nccl4py < Version(MIN_NCCL4PY_VERSION) or version.libnccl is None:
        raise ImportError(
            f"Engram symmetric requires nccl4py >= {MIN_NCCL4PY_VERSION} and its NCCL device "
            "bitcode"
        )
    if version.libnccl.version != version.nccl_bindings:
        raise ImportError(
            "Engram symmetric requires matching NCCL runtime, NCCL4Py headers and device "
            f"bitcode; got runtime {version.libnccl.version}, headers {version.nccl_bindings}"
        )


def create_unique_id() -> bytes:
    """Return a fresh NCCL unique id as bytes, ready for an object gather."""
    return bytes(nccl.get_unique_id())


def _communicator_config() -> "nccl.NCCLConfig":
    """The configuration of the row group's NCCL communicator.

    The transport only launches its own kernels through the device API, never an NCCL kernel,
    yet NCCL sets up collective channels and NVLS state for every new communicator: on GB300,
    32 channels of about 6 MiB of device memory each and 24 NVLS channels of 2 MiB each, plus
    multicast bindings for every symmetric window. One CTA keeps a single channel, and the NVLS
    host mode (NCCL 2.32) turns NVLS off. LSA teams, windows and GIN contexts do not use either.
    ``NCCL_MIN_CTAS`` and ``NCCL_MAX_CTAS`` in the environment take precedence over this
    configuration, as they do for every communicator.
    """
    fields = {"min_ctas": 1, "max_ctas": 1, "comm_name": "ENGRAM_SYMMETRIC_TRANSPORT"}
    if nccl.get_version().libnccl.version >= Version("2.32.0"):
        fields["nvls_host_mode"] = (
            nccl.NcclNvlsHostMode.DISABLE_TRANSPORT
            | nccl.NcclNvlsHostMode.DISABLE_SYMMETRIC_MULTIMEM
        )
    return nccl.NCCLConfig(**fields)


def _rows_tensor(tensor):
    # Unique row counts change every microbatch. Keep them out of CuTe's JIT
    # specialization key. Flatten first: DLPack normalizes singleton/empty row
    # strides, which cannot reliably describe a dynamic compact matrix layout.
    return from_dlpack(tensor.reshape(-1)).mark_layout_dynamic(leading_dim=0)


@dsl_user_op
def _load_acquire(address, *, loc=None, ip=None):
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [cutlass.Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _signal_release(address, *, loc=None, ip=None):
    # One writer per peer/slot. Atomic increment also avoids publishing a stale
    # epoch when a different invocation reuses the same receive slot.
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [cutlass.Int64(address).ir_value(loc=loc, ip=ip)],
            "atom.release.sys.global.add.u64 $0, [$1], 1;",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _wait_lsa(window: nc.Window, index, least, flags_offset):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        pointer = cute.make_ptr(cutlass.Int64, window.local_pointer(flags_offset + index * 8))
        value = _load_acquire(pointer.toint())
        while value < least:
            value = _load_acquire(pointer.toint())
    cute.arch.sync_threads()


@cute.jit
def _signal_lsa(window: nc.Window, peer, index, flags_offset):
    # The release is by thread 0, after it joins every payload writer in the CTA.
    cute.arch.sync_threads()
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        pointer = cute.make_ptr(cutlass.Int64, window.lsa_pointer(flags_offset + index * 8, peer))
        _signal_release(pointer.toint())
    cute.arch.sync_threads()


@cute.jit
def _copy_words(src, dst, count):
    tid, _, _ = cute.arch.thread_idx()
    for index in range(tid, count, 128):
        dst[index] = src[index]


@cute.kernel
def _exchange_kernel(
    comm: nc.DevComm,
    window: nc.Window,
    source: cute.Tensor,
    destination: cute.Tensor,
    splits: cute.Tensor,
    sequence: cute.Tensor,
    slot_bytes: cutlass.Int64,
    flags_offset: cutlass.Int64,
    row_words: cutlass.Int64,
    channels: cutlass.Constexpr,
    stripes: cutlass.Constexpr,
    has_gin: cutlass.Constexpr,
    remote_only: cutlass.Constexpr,
):
    """A ring schedule with two receive slots per sender and GPU-side credits.

    Each peer transfer is split into independent stripes, with one CTA and two
    credit-controlled slots per stripe. This exposes memory parallelism without
    increasing the workspace budget or requiring a grid-wide spin barrier.
    """
    block, _, _ = cute.arch.block_idx()
    channel, stripe = block // stripes, block % stripes
    tid, _, _ = cute.arch.thread_idx()
    team, lsa = comm.team_world, comm.team_lsa
    # NCCL's default LSA team has stride one. Translate world ranks using its
    # origin and check both bounds; rank translation alone is not membership.
    lsa_start = team.rank - lsa.rank
    coop = nc.cta()
    if cutlass.const_expr(has_gin):
        gin = comm.gin(nc.GinBackendMask.ALL, 0)
    word_bytes = source.element_type.width // 8
    capacity = slot_bytes // (row_words * word_bytes)
    for offset in range(channel, team.nRanks, channels):
        send_peer = (team.rank + offset) % team.nRanks
        recv_peer = (team.rank + team.nRanks - offset) % team.nRanks
        send_lsa = send_peer - lsa_start
        recv_lsa = recv_peer - lsa_start
        send_local = (send_lsa >= 0) & (send_lsa < lsa.nRanks)
        recv_local = (recv_lsa >= 0) & (recv_lsa < lsa.nRanks)
        send_begin, send_count = splits[send_peer, 0], splits[send_peer, 1]
        recv_begin, recv_count = splits[recv_peer, 2], splits[recv_peer, 3]
        if cutlass.const_expr(remote_only):
            if send_local:
                send_count = cutlass.Int64(0)
            if recv_local:
                recv_count = cutlass.Int64(0)
        send_start = send_count * stripe // stripes
        recv_start = recv_count * stripe // stripes
        send_begin += send_start
        recv_begin += recv_start
        send_count = send_count * (stripe + 1) // stripes - send_start
        recv_count = recv_count * (stripe + 1) // stripes - recv_start
        if offset == 0:
            _copy_words(
                cute.make_tensor(source.iterator + send_begin * row_words, source.layout),
                cute.make_tensor(destination.iterator + recv_begin * row_words, destination.layout),
                send_count * row_words,
            )
        else:
            send_chunks = (send_count + capacity - 1) // capacity
            recv_chunks = (recv_count + capacity - 1) // capacity
            rounds = max(send_chunks, recv_chunks)
            # Send n while consuming n-1. Two slots let the next put overlap the
            # previous receive copy without a per-chunk host launch or collective.
            for chunk in range(rounds + 1):
                slot = chunk % 2
                if chunk < send_chunks:
                    ack = (send_peer * stripes + stripe) * 4 + 2 + slot
                    expected_ack = sequence[ack]
                    cute.arch.sync_threads()
                    if send_local:
                        _wait_lsa(window, ack, expected_ack, flags_offset)
                    elif cutlass.const_expr(has_gin):
                        gin.wait_signal(coop, signal=ack, least=expected_ack)
                    count = min(capacity, send_count - chunk * capacity) * row_words
                    src = cute.make_tensor(
                        source.iterator + (send_begin + chunk * capacity) * row_words,
                        cute.make_layout(count),
                    )
                    recv_offset = (
                        2 * channels * stripes + 2 * (team.rank * stripes + stripe) + slot
                    ) * slot_bytes
                    ready = (team.rank * stripes + stripe) * 4 + slot
                    if send_local:
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                source.element_type, window.lsa_pointer(recv_offset, send_lsa)
                            ),
                            cute.make_layout(count),
                        )
                        _copy_words(src, dst, count)
                        _signal_lsa(window, send_lsa, ready, flags_offset)
                    elif cutlass.const_expr(has_gin):
                        staging = window.tensor(
                            source.element_type,
                            cute.make_layout(count),
                            offset=(2 * (channel * stripes + stripe) + slot) * slot_bytes,
                        )
                        _copy_words(src, staging, count)
                        cute.arch.sync_threads()
                        dst = window.tensor(
                            source.element_type, cute.make_layout(count), offset=recv_offset
                        )
                        gin.put(
                            team,
                            send_peer,
                            window,
                            dst,
                            window,
                            staging,
                            coop,
                            is_signal=True,
                            signal_id=ready,
                            signal_op=0,
                        )
                    if tid == 0:
                        sequence[ack] = expected_ack + 1
                    cute.arch.sync_threads()
                received_chunk = chunk - 1
                if (received_chunk >= 0) & (received_chunk < recv_chunks):
                    recv_slot = received_chunk % 2
                    ready = (recv_peer * stripes + stripe) * 4 + recv_slot
                    expected_ready = sequence[ready] + 1
                    cute.arch.sync_threads()
                    if recv_local:
                        _wait_lsa(window, ready, expected_ready, flags_offset)
                    elif cutlass.const_expr(has_gin):
                        gin.wait_signal(coop, signal=ready, least=expected_ready)
                    count = min(capacity, recv_count - received_chunk * capacity) * row_words
                    src = window.tensor(
                        source.element_type,
                        cute.make_layout(count),
                        offset=(
                            2 * channels * stripes + 2 * (recv_peer * stripes + stripe) + recv_slot
                        )
                        * slot_bytes,
                    )
                    dst = cute.make_tensor(
                        destination.iterator + (recv_begin + received_chunk * capacity) * row_words,
                        cute.make_layout(count),
                    )
                    _copy_words(src, dst, count)
                    cute.arch.sync_threads()
                    ack = (team.rank * stripes + stripe) * 4 + 2 + recv_slot
                    if recv_local:
                        _signal_lsa(window, recv_lsa, ack, flags_offset)
                    elif cutlass.const_expr(has_gin):
                        gin.signal(team, recv_peer, True, ack, 0, 0, coop)
                    if tid == 0:
                        sequence[ready] = expected_ready
                    cute.arch.sync_threads()
            # Drain both outgoing slots, including the final partially filled one.
            # The next exchange may change row width and peer counts.
            for slot in cutlass.range_constexpr(2):
                if slot < send_chunks:
                    ack = (send_peer * stripes + stripe) * 4 + 2 + slot
                    if send_local:
                        _wait_lsa(window, ack, sequence[ack], flags_offset)
                    elif cutlass.const_expr(has_gin):
                        gin.wait_signal(coop, signal=ack, least=sequence[ack])
    if cutlass.const_expr(has_gin):
        gin.flush(coop)


@cute.jit
def _exchange(
    comm: nc.DevComm,
    window: nc.Window,
    source: cute.Tensor,
    destination: cute.Tensor,
    splits: cute.Tensor,
    sequence: cute.Tensor,
    slot_bytes: cutlass.Int64,
    flags_offset: cutlass.Int64,
    row_words: cutlass.Int64,
    channels: cutlass.Constexpr,
    stripes: cutlass.Constexpr,
    has_gin: cutlass.Constexpr,
    remote_only: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _exchange_kernel(
        comm,
        window,
        source,
        destination,
        splits,
        sequence,
        slot_bytes,
        flags_offset,
        row_words,
        channels,
        stripes,
        has_gin,
        remote_only,
    ).launch(grid=(channels * stripes, 1, 1), block=(128, 1, 1), stream=stream)


@cute.kernel
def _table_barrier_kernel(comm: nc.DevComm, has_gin: cutlass.Constexpr):
    coop = nc.cta()
    if cutlass.const_expr(has_gin):
        gin = comm.gin(nc.GinBackendMask.ALL, 0)
        barrier = nc.barrier.world_hybrid(coop, gin, comm, 0)
        barrier.sync(coop, nc.MemoryOrder.ACQ_REL, nc.GinFenceLevel.NONE)
        barrier.destroy()
    else:
        barrier = nc.barrier.lsa_default(coop, comm, 0)
        barrier.sync(coop, nc.MemoryOrder.ACQ_REL)
        barrier.destroy()


@cute.jit
def _table_barrier(comm: nc.DevComm, has_gin: cutlass.Constexpr, stream: cuda.CUstream):
    _table_barrier_kernel(comm, has_gin).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)


@cute.kernel
def _lookup_kernel(
    comm: nc.DevComm,
    data_windows: tuple,
    scale_windows: tuple,
    metadata: cute.Tensor,
    requests: cute.Tensor,
    peers: cute.Tensor,
    output: cute.Tensor,
    rows: cutlass.Int64,
    data_words: cutlass.Constexpr,
    scale_words: cutlass.Constexpr,
    local_requests: cutlass.Constexpr,
):
    """Direct LSA reads, or owner-side packing of requests received over GIN."""
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    team, lsa = comm.team_world, comm.team_lsa
    lsa_start = team.rank - lsa.rank
    # Four warps per CTA, one row per warp; adjacent lanes read adjacent words.
    row = block * 4 + tid // 32
    lane = tid % 32
    if row < rows:
        peer = peers[row]
        lsa_peer = peer - lsa_start
        is_local = (lsa_peer >= 0) & (lsa_peer < lsa.nRanks)
        valid = is_local
        if cutlass.const_expr(not local_requests):
            valid = not is_local
            peer = team.rank
            lsa_peer = lsa.rank
        if valid:
            table_id, table_row = requests[row * 2], requests[row * 2 + 1]
            for table in cutlass.range_constexpr(len(data_windows)):
                if table_id == table:
                    data_offset = metadata[table, peer, 0] + table_row * metadata[table, peer, 1]
                    data = cute.make_tensor(
                        cute.make_ptr(
                            output.element_type,
                            data_windows[table].lsa_pointer(data_offset, lsa_peer),
                        ),
                        cute.make_layout(data_words),
                    )
                    for col in range(lane, data_words, 32):
                        output[row * (data_words + scale_words) + col] = data[col]
                    if cutlass.const_expr(scale_words > 0):
                        scale_offset = (
                            metadata[table, peer, 2] + table_row * metadata[table, peer, 3]
                        )
                        scales = cute.make_tensor(
                            cute.make_ptr(
                                output.element_type,
                                scale_windows[table].lsa_pointer(scale_offset, lsa_peer),
                            ),
                            cute.make_layout(scale_words),
                        )
                        for col in range(lane, scale_words, 32):
                            output[row * (data_words + scale_words) + data_words + col] = scales[
                                col
                            ]


@cute.jit
def _lookup(
    comm: nc.DevComm,
    data_windows: tuple,
    scale_windows: tuple,
    metadata: cute.Tensor,
    requests: cute.Tensor,
    peers: cute.Tensor,
    output: cute.Tensor,
    rows: cutlass.Int64,
    data_words: cutlass.Constexpr,
    scale_words: cutlass.Constexpr,
    local_requests: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _lookup_kernel(
        comm,
        data_windows,
        scale_windows,
        metadata,
        requests,
        peers,
        output,
        rows,
        data_words,
        scale_words,
        local_requests,
    ).launch(grid=((rows + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream)


class _Storage(NamedTuple):
    """One distinct table storage of this rank."""

    # (data pointer, bytes) of the untyped storage.
    key: tuple
    # Borrowed buffer from the CUDA allocation base to the storage end, as registered.
    view: object
    # Allocation base address: table offsets in the window are relative to it.
    base: int


class _TableDescription(NamedTuple):
    """This rank's view of the tables of one lookup, from ``describe_tables``."""

    signatures: tuple
    # Distinct storages of the tables, in first-use order.
    storages: tuple
    # Index into ``storages`` of each table.
    groups: tuple
    # (byte offset in its storage window, byte row stride) of each table.
    layout: tuple
    row_bytes: int


class _TableWindows(NamedTuple):
    """Registered windows of the tables of one lookup, valid until any storage changes."""

    description: _TableDescription
    # Transport window of each block of tables that share a storage on every rank.
    window_ids: tuple
    # Device window of each table; tables of one block repeat the same window.
    device_windows: tuple
    # Every rank's offsets and strides, and the row width, are multiples of four bytes.
    aligned: bool


class _Window(NamedTuple):
    """A registered table window, shared by every lookup whose tables it holds on all ranks."""

    window: object
    storage_key: tuple
    # Keeps the registered storage alive while the window exists.
    view: object
    device_window: object
    # Lookups using the window; it is closed, on every rank, when the last one lets go.
    keys: set


def _table_signatures(tables) -> tuple:
    """Storage identity of each table; any change needs a collective re-registration."""
    return tuple(
        (
            tensor.untyped_storage().data_ptr(),
            tensor.untyped_storage().nbytes(),
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )
        for tensor in tables
    )


class EngramDeviceTransport:
    """One LSA/GIN transport per row-shard group, shared by every Engram table of the rank.

    Construction and :meth:`prepare`, which allocates the workspace, are local.
    :meth:`initialize` is collective over the row group: it creates the NCCL communicator from
    ``unique_id`` (the same bytes on every rank), the device communicator and the workspace
    window, and compiles the barrier. Once every rank has initialized, every rank runs
    :meth:`barrier`, which publishes the zeroed workspace flags; only then may exchanges start.

    An exchange ``plan`` describes this rank's side of one exchange: ``send_splits`` and
    ``recv_splits`` (rows per peer, host integers) and ``splits``, a contiguous device
    ``[N, 4]`` int64 tensor of (send start, send rows, receive start, receive rows) per peer.
    Plans passed to :meth:`lookup` also provide ``reverse()``, the plan of the return direction.
    """

    def __init__(
        self,
        group,
        *,
        unique_id: bytes,
        workspace_bytes: int,
        expected_lsa_ranks: Optional[int] = None,
    ) -> None:
        check_runtime()
        self.group = group
        # Megatron uses None for a local, unsharded table, not WORLD.
        self.rank = torch.distributed.get_rank(group) if group is not None else 0
        self.size = torch.distributed.get_world_size(group) if group is not None else 1
        self.channels = min(4, self.size)
        self.unique_id = unique_id
        self.workspace_bytes = workspace_bytes
        # Row-group ranks in this rank's NVLink domain (its GPU fabric clique, or its host), for
        # the LSA-team check.
        self.expected_lsa_ranks = expected_lsa_ranks
        self.device = None
        # The workspace, its sequence counters and the registration plan, from prepare().
        self.buffer = self.sequence = self.peers = self.uniform_plan = None
        self.comm = None
        self.device_view = None
        self.tables = {}
        self.pending = {}
        # Registered table windows by id; ids are assigned in the same order on every rank.
        self.windows = {}
        self.next_window_id = 0
        self.lookup_kernels = {}
        self.exchange_kernels = {}
        self.barrier_kernel = None
        # Set by the first barrier: every rank has zeroed its workspace flags.
        self.published = False
        # Set when a registration failed on some rank: the ranks may hold different windows,
        # which only an abort releases.
        self.failed = False

    def prepare(self) -> None:
        """Lay out and allocate the workspace (local; a no-op once done).

        Everything of the start-up that can fail on one rank alone and needs no peer happens
        here: the slot layout, the ``ncclMemAlloc`` workspace with its zeroed completion flags,
        the sequence counters and the registration plan. The caller agrees on the outcome on
        every rank before any rank enters :meth:`initialize`, whose NCCL calls block until the
        whole row group joins them. :meth:`initialize` runs it first when the caller did not.
        """
        if self.buffer is not None:
            return
        self.device = torch.device("cuda", torch.cuda.current_device())
        workspace_bytes = self.workspace_bytes
        # Up to 128 CTAs with the default four peer channels. Keep at least 4 KiB
        # per slot, reducing stripes for small budgets without allocating more memory.
        stripe_bytes = self.size * 32 + 2 * (self.size + self.channels) * 4096
        self.stripes = 1 << min(5, max(0, (workspace_bytes // stripe_bytes).bit_length() - 1))
        flags_bytes = self.size * self.stripes * 4 * 8
        self.slot_bytes = (
            (workspace_bytes - flags_bytes)
            // (2 * (self.size + self.channels) * self.stripes)
            // 256
            * 256
        )
        if self.slot_bytes <= 0:
            raise ValueError("Engram workspace cannot hold two receive slots per shard")
        self.flags_offset = self.slot_bytes * 2 * (self.size + self.channels) * self.stripes
        with torch.cuda.device(self.device):
            # ncclMemAlloc needs no communicator.
            buffer = nccl_torch.empty(workspace_bytes, dtype=torch.uint8, device=self.device)
            buffer[self.flags_offset : self.flags_offset + flags_bytes].zero_()
            self.sequence = torch.zeros(
                self.size * self.stripes * 4, dtype=torch.int64, device=self.device
            )
            self.peers = torch.arange(self.size, dtype=torch.int64, device=self.device)
            ones = torch.ones_like(self.peers)
            # One row to and from every peer, for the exchanges of registration.
            self.uniform_plan = SimpleNamespace(
                send_splits=(1,) * self.size,
                recv_splits=(1,) * self.size,
                splits=torch.stack((self.peers, ones, self.peers, ones), dim=1),
            )
        # Set last: a failure above leaves the transport unprepared.
        self.buffer = buffer

    def initialize(self) -> None:
        """Create the communicator, device communicator and workspace window (collective).

        Its NCCL calls (communicator init, device communicator, window registration) block until
        every rank of the row group joins them, so the caller should agree on :meth:`prepare`
        on every rank first: a rank that fails between those calls leaves its peers waiting
        inside NCCL. It launches nothing that waits on a peer, so every rank returns and the
        caller can agree on the outcome before any spin-wait starts. The barrier and the
        exchange of 4-byte words (count, registration and gradient rows) are compiled here, so
        their compile failures surface at the same point. Every rank then runs :meth:`barrier`
        once before the first exchange.
        """
        if self.comm is not None:
            return
        self.prepare()
        with torch.cuda.device(self.device):
            self.comm = nccl.Communicator.init(
                nranks=self.size,
                rank=self.rank,
                unique_id=nccl.UniqueId.from_bytes(self.unique_id),
                config=_communicator_config(),
            )
            if not self.comm.device_api_support:
                raise RuntimeError("This NCCL communicator does not support the device API")
            self.has_gin = self.comm.team_lsa.n_ranks != self.size
            if self.has_gin and self.comm.gin_type == nccl.NcclGinType.NONE:
                raise RuntimeError("Engram shards cross LSA domains, but NCCL has no GIN transport")
            self.dev_comm = self.comm.create_dev_comm(
                nccl.NCCLDevCommRequirements(
                    barrier_count=1 if self.has_gin else 0,
                    lsa_barrier_count=1,
                    gin_connection_type=(
                        nccl.NcclGinConnectionType.FULL
                        if self.has_gin
                        else nccl.NcclGinConnectionType.NONE
                    ),
                    gin_signal_count=self.size * self.stripes * 4 if self.has_gin else 0,
                    gin_context_count=1 if self.has_gin else 0,
                )
            )
            self.window = self.comm.register_window(self.buffer)
            if not self.window.is_valid:
                raise RuntimeError("NCCL did not register the Engram workspace")
            self.device_view, self.window_view = nc.DevComm(self.dev_comm), nc.Window(self.window)
            self.barrier_kernel = cute.compile(
                _table_barrier, make_fake_dev_comm(), self.has_gin, make_fake_stream()
            )
            self._exchange_executable(torch.int32, False)
        self._log_initialization()

    def _log_initialization(self) -> None:
        lsa_ranks = self.comm.team_lsa.n_ranks
        if self.rank == 0:
            logger.info(
                "Engram symmetric transport: %d row ranks, LSA team of %d, has_gin=%s, "
                "gin_type=%s, %d channels x %d stripes, %d-byte slots, %d MiB workspace",
                self.size,
                lsa_ranks,
                self.has_gin,
                getattr(self.comm.gin_type, "name", self.comm.gin_type),
                self.channels,
                self.stripes,
                self.slot_bytes,
                self.workspace_bytes // 2**20,
            )
        if self.expected_lsa_ranks is not None and lsa_ranks < self.expected_lsa_ranks:
            logger.warning(
                "Engram row rank %d has an LSA team of %d ranks, but %d ranks of its row group "
                "share its NVLink domain (its GPU fabric clique, or its host), so the other "
                "domain-local peers are reached over GIN. Either the row group's rank order is "
                "not NVLink-contiguous (NCCL LSA teams are contiguous with stride one in it), or "
                "NCCL does not use multi-node NVLink (NCCL_MNNVL_ENABLE, IMEX channels).",
                self.rank,
                lsa_ranks,
                self.expected_lsa_ranks,
            )

    @staticmethod
    def _stream():
        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def barrier(self):
        """Publish parameter updates or finish all remote reads before reuse."""
        if self.barrier_kernel is None:
            self.barrier_kernel = cute.compile(
                _table_barrier, make_fake_dev_comm(), self.has_gin, make_fake_stream()
            )
        self.barrier_kernel(self.device_view, self._stream())
        self.published = True

    def _exchange_executable(self, word_dtype, remote_only):
        """The compiled exchange for one word type; row counts and widths stay runtime values."""
        signature = (word_dtype, self.channels, self.stripes, self.has_gin, remote_only)
        if signature not in self.exchange_kernels:
            word = cutlass.Int32 if word_dtype == torch.int32 else cutlass.Uint8
            self.exchange_kernels[signature] = cute.compile(
                _exchange,
                make_fake_dev_comm(),
                make_fake_window(),
                make_fake_tensor(word, (cute.sym_int64(),), (1,)),
                make_fake_tensor(word, (cute.sym_int64(),), (1,)),
                make_fake_tensor(cutlass.Int64, (self.size, 4), (4, 1)),
                make_fake_tensor(cutlass.Int64, (self.size * self.stripes * 4,), (1,)),
                cutlass.Int64(0),
                cutlass.Int64(0),
                cutlass.Int64(0),
                self.channels,
                self.stripes,
                self.has_gin,
                remote_only,
                make_fake_stream(),
            )
        return self.exchange_kernels[signature]

    @torch.no_grad()
    def exchange(self, source, plan, *, remote_only=False, output=None):
        """Exchange opaque rows with bounded slots and no host chunk loop.

        Every rank of the row group must launch every exchange, in the same order, even with
        no rows of its own: no rank knows the global count matrix, and a peer may still send
        to this rank or wait for its credits. ``remote_only`` skips LSA peers, and ``output``
        receives the rows in place, leaving the slices of skipped peers untouched.
        """
        if not source.is_cuda or source.ndim < 2:
            raise ValueError("Engram symmetric requires CUDA row tensors")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Engram communication must remain outside CUDA Graph capture")
        if self.device_view is None:
            raise RuntimeError("Initialize the Engram transport before exchanging rows")
        if not self.published:
            raise RuntimeError(
                "Run barrier() on every rank of the row group after initialize() and before "
                "the first exchange"
            )
        send_splits, recv_splits = tuple(plan.send_splits), tuple(plan.recv_splits)
        if (
            len(send_splits) != self.size
            or len(recv_splits) != self.size
            or tuple(plan.splits.shape) != (self.size, 4)
            or plan.splits.dtype != torch.int64
            or plan.splits.device != source.device
        ):
            raise ValueError("Engram exchange plan does not match its row group")
        if sum(send_splits) != source.shape[0]:
            raise ValueError("Engram input splits must cover every input row")
        source = source.contiguous()
        row_bytes = math.prod(source.shape[1:]) * source.element_size()
        if row_bytes <= 0 or row_bytes > self.slot_bytes:
            raise ValueError("Engram workspace cannot hold a complete row per peer/slot")
        if output is None:
            output = source.new_empty((sum(recv_splits), *source.shape[1:]))
        if (
            output.shape != (sum(recv_splits), *source.shape[1:])
            or output.dtype != source.dtype
            or not output.is_contiguous()
        ):
            raise ValueError("Engram output must match the receive shape and dtype")
        aligned = source.data_ptr() % 4 == output.data_ptr() % 4 == row_bytes % 4 == 0
        word_dtype = torch.int32 if aligned else torch.uint8
        word_bytes = torch.empty((), dtype=word_dtype).element_size()
        # Keep the executable, bypassing auto-JIT argument processing on every
        # transfer. Row counts and widths are runtime arguments, not specializations.
        self._exchange_executable(word_dtype, remote_only)(
            self.device_view,
            self.window_view,
            _rows_tensor(source.view(word_dtype).reshape(-1)),
            _rows_tensor(output.view(word_dtype).reshape(-1)),
            from_dlpack(plan.splits.contiguous()),
            from_dlpack(self.sequence),
            cutlass.Int64(self.slot_bytes),
            cutlass.Int64(self.flags_offset),
            cutlass.Int64(row_bytes // word_bytes),
            self._stream(),
        )
        return output

    @staticmethod
    def _check_vmm(tensor):
        """Reject storage that NCCL cannot register: windows need CUDA VMM mappings."""
        address = tensor.untyped_storage().data_ptr()
        with torch.cuda.device(tensor.device):
            status, handle = cuda.cuMemRetainAllocationHandle(address)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                f"Engram table storage at {address:#x} is not a CUDA VMM allocation ({status}); "
                "use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True or an NCCL allocator"
            )
        cuda.cuMemRelease(handle)

    @staticmethod
    def _storage_view(tensor):
        storage = tensor.untyped_storage()
        stride = tensor.stride(0) * tensor.element_size()
        if not storage.nbytes():
            # Empty owners must still participate in collective registration. nccl4py takes a
            # device ID (or a cuda.core Device), not a torch.device.
            return nccl.mem_alloc(256, device=tensor.device.index), 0, stride

        address = storage.data_ptr()
        with torch.cuda.device(tensor.device):
            status, allocation, size = cuda.cuMemGetAddressRange(address)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cannot query Engram CUDA allocation at {address:#x}: {status}")
        allocation, size = int(allocation), int(size)
        if not allocation <= address < allocation + size:
            raise RuntimeError(f"Invalid CUDA allocation range for Engram storage at {address:#x}")

        # A PyTorch storage can itself be an unaligned allocator suballocation.
        # Start at the CUDA allocation base: NCCL's LSA pointer calculation uses
        # the local window's allocation offset for every peer, so merely rounding
        # to 4 KiB is insufficient when that offset differs between ranks.
        # The storage may span multiple VMM segments; NCCL traverses those itself.
        view = Buffer.from_handle(
            allocation, address + storage.nbytes() - allocation, owner=storage
        )
        # The borrowed buffer keeps the original allocation alive, without copying
        # the table or taking ownership of the aligned address.
        return view, tensor.data_ptr() - allocation, stride

    def describe_tables(self, key, tables):
        """Describe this rank's table storage for the next count exchange; never communicates.

        Returns ``(changed, error, layout)``: whether the storage differs from the windows
        registered for ``key``, a description of a failure to prepare it (``None`` when it can
        be registered), and the (byte offset in its window, byte row stride) of every table.
        A failure is reported through the count exchange, so every rank raises together.
        """
        signatures = _table_signatures(tables)
        registered = self.tables.get(key)
        if registered is not None and registered.description.signatures == signatures:
            self.pending.pop(key, None)
            return False, None, registered.description.layout
        try:
            description = self._describe(tables, signatures)
        except Exception as exc:
            self.pending.pop(key, None)
            return True, f"{type(exc).__name__}: {exc}", ((0, 0),) * len(tables)
        self.pending[key] = description
        return True, None, description.layout

    def _describe(self, tables, signatures):
        if not tables:
            raise ValueError("An Engram lookup needs at least one table")
        first = tables[0]
        for tensor in tables:
            if tensor.device != self.device:
                raise ValueError(f"Engram table on {tensor.device}, transport on {self.device}")
            if tensor.ndim != 2 or tensor.stride(1) != 1:
                raise ValueError("Engram direct lookup requires row-major tables")
            if tensor.dtype != first.dtype or tensor.shape[1] != first.shape[1]:
                raise ValueError("The tables of one Engram lookup must share dtype and width")
        storages, groups, layout, index = [], [], [], {}
        for tensor in tables:
            storage = tensor.untyped_storage()
            storage_key = (storage.data_ptr(), storage.nbytes())
            if storage_key not in index:
                if storage.nbytes():
                    self._check_vmm(tensor)
                # Reuse allocation queries and borrowed buffers until storage changes.
                view, offset, _ = self._storage_view(tensor)
                index[storage_key] = len(storages)
                storages.append(_Storage(storage_key, view, tensor.data_ptr() - offset))
            group = index[storage_key]
            groups.append(group)
            offset = tensor.data_ptr() - storages[group].base if storage.nbytes() else 0
            layout.append((offset, tensor.stride(0) * tensor.element_size()))
        return _TableDescription(
            signatures,
            tuple(storages),
            tuple(groups),
            tuple(layout),
            first.shape[1] * first.element_size(),
        )

    def register_tables(self, key, metadata):
        """Bring the windows of ``key`` up to date once any rank reported a change (collective).

        Every rank of the row group calls this after the same count exchange, whose
        ``[T, N, 2]`` (offset, stride) ``metadata`` it receives. One exchange shares, per
        table, each rank's storage index and the id of a window that already holds that
        storage. Tables form one block only where every rank holds them in one storage, and a
        block reuses a window only where every rank names the same one. Every rank therefore
        registers and closes the same windows in the same order, and the tables of all
        lookups in one DDP parameter buffer share a single window. Each rank names only the
        newest window of a storage, so where ranks disagree on which tables share a storage,
        later lookups reuse fewer windows and register more.

        A rank that cannot register a window still issues every registration of its peers, and
        one more exchange shares each rank's outcome: on any failure every rank raises, and the
        transport can then only be aborted.
        """
        description = self.pending.pop(key, None)
        registered = self.tables.get(key)
        if description is None:
            if registered is None:
                raise RuntimeError("describe_tables() must precede the first register_tables()")
            # Another rank changed: re-register this rank's unchanged storage.
            description = registered.description
        # The newest window of each storage: after a peer's storage changed, the first lookup
        # to refresh registers a window that the next lookups over the same storage then reuse.
        held = {entry.storage_key: window_id for window_id, entry in self.windows.items()}
        num_tables = len(description.groups)
        row = torch.tensor(
            [
                value
                for group in description.groups
                for value in (group, held.get(description.storages[group].key, -1))
            ],
            dtype=torch.int64,
            device=self.device,
        )
        gathered = self.exchange(
            row.expand(self.size, 2 * num_tables).contiguous(), self.uniform_plan
        )
        # Per table: (storage index, held window) of rank 0, then of rank 1, and so on.
        columns = gathered.view(self.size, num_tables, 2).transpose(0, 1).reshape(num_tables, -1)
        blocks = {}
        table_blocks = [
            blocks.setdefault(tuple(column), len(blocks)) for column in columns.tolist()
        ]
        aligned = description.row_bytes % 4 == 0 and bool((metadata.cpu() % 4 == 0).all())
        # A collective decision: only one shard may have had its parameter storage replaced
        # by an optimizer/DDP buffer refresh. No peer may still read the old windows.
        self.barrier()
        torch.cuda.current_stream().synchronize()
        window_ids, problem = [], None
        for block, column in enumerate(blocks):
            window_id = column[1]
            if window_id >= 0 and set(column[1::2]) == {window_id}:
                window_ids.append(window_id)
                continue
            storage = description.storages[description.groups[table_blocks.index(block)]]
            window_id, self.next_window_id = self.next_window_id, self.next_window_id + 1
            window_ids.append(window_id)
            try:
                window = self.comm.register_window(storage.view)
                if not window.is_valid:
                    raise RuntimeError("NCCL returned an invalid table window")
                self.windows[window_id] = _Window(
                    window, storage.key, storage.view, nc.Window(window), set()
                )
            except Exception as exc:
                # Keep issuing the registrations of the peers; every rank raises together below.
                problem = problem or f"{type(exc).__name__}: {exc}"
        failed = self._failed_ranks(problem is not None)
        if failed:
            self.failed = True
            reason = f"; on this rank: {problem}" if problem is not None else ""
            raise RuntimeError(
                f"Engram direct lookup cannot register table windows on row ranks {failed}"
                f"{reason}. It requires registerable CUDA table storage: VMM allocations "
                "(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) or an NCCL allocator; the "
                "backend never copies the entire table."
            )
        for window_id in window_ids:
            self.windows[window_id].keys.add(key)
        if registered is not None:
            # Same order on every rank, in case closing a window is collective.
            for window_id in sorted(set(registered.window_ids) - set(window_ids)):
                self._release(window_id, key)
        self.tables[key] = _TableWindows(
            description,
            tuple(window_ids),
            tuple(self.windows[window_ids[block]].device_window for block in table_blocks),
            aligned,
        )

    def _failed_ranks(self, failed: bool) -> list:
        """Row ranks that report a failure: one exchange and one host copy (collective)."""
        flags = torch.full((self.size, 1), int(failed), dtype=torch.int64, device=self.device)
        return self.exchange(flags, self.uniform_plan)[:, 0].nonzero().flatten().tolist()

    def _release(self, window_id, key):
        entry = self.windows[window_id]
        entry.keys.discard(key)
        if not entry.keys:
            entry.window.close()
            del self.windows[window_id]

    @torch.no_grad()
    def lookup(self, key, tables, requests, plan, metadata):
        """Read LSA rows directly and exchange only cross-domain requests/results.

        ``requests`` is ``[sum(plan.send_splits), 2]`` int64 (table, owner-local row), grouped
        by owner in peer order, and ``metadata`` the ``[T, N, 2]`` int64 (offset, stride) of
        every peer's tables from the same count exchange. Returns the rows as uint8
        ``[len(requests), row_bytes]`` in request order.
        """
        if not requests.is_cuda:
            raise ValueError("Engram symmetric requires CUDA tensors")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Engram communication must remain outside CUDA Graph capture")
        registered = self.tables.get(key)
        if registered is None or registered.description.signatures != _table_signatures(tables):
            raise RuntimeError("Engram table windows are stale: run the count exchange first")
        rows = sum(plan.send_splits)
        if requests.dtype != torch.int64 or tuple(requests.shape) != (rows, 2):
            raise ValueError("Engram requests must be [rows, 2] int64 rows that match the plan")
        if tuple(metadata.shape) != (len(tables), self.size, 2):
            raise ValueError("Engram lookup metadata must be [tables, row ranks, 2]")
        requests, metadata = requests.contiguous(), metadata.contiguous()
        data_bytes = registered.description.row_bytes
        dtype = torch.int32 if registered.aligned else torch.uint8
        word_bytes = torch.empty((), dtype=dtype).element_size()
        output = torch.empty((rows, data_bytes), dtype=torch.uint8, device=requests.device)

        def gather(ids, counts, target, local_requests):
            # Unlike the barrier and exchange kernels, which every rank launches, the lookup
            # kernel never waits on a peer, so a rank without rows may skip it. It must stay
            # wait-free.
            if not ids.shape[0]:
                return
            # The peer of every row, expanded on the device from the plan's row counts.
            peers = torch.repeat_interleave(self.peers, counts, output_size=ids.shape[0])
            tensor_args = (
                from_dlpack(metadata),
                _rows_tensor(ids),
                _rows_tensor(peers),
                _rows_tensor(target.view(dtype).reshape(-1)),
                cutlass.Int64(ids.shape[0]),
            )
            signature = (len(tables), dtype, data_bytes, local_requests)
            if signature not in self.lookup_kernels:
                # Auto-JIT tries to extract device IR from host Window objects
                # nested in tuples (and rejects an empty scale tuple). Compile
                # with NCCL's type-only arguments, then pass live handles to the
                # executable. Row counts and table addresses remain dynamic.
                self.lookup_kernels[signature] = cute.compile(
                    _lookup,
                    make_fake_dev_comm(),
                    tuple(make_fake_window() for _ in tables),
                    (),
                    make_fake_tensor(cutlass.Int64, metadata.shape, metadata.stride()),
                    make_fake_tensor(cutlass.Int64, (cute.sym_int64(),), (1,)),
                    make_fake_tensor(cutlass.Int64, (cute.sym_int64(),), (1,)),
                    make_fake_tensor(
                        cutlass.Int32 if word_bytes == 4 else cutlass.Uint8,
                        (cute.sym_int64(),),
                        (1,),
                    ),
                    cutlass.Int64(0),
                    data_bytes // word_bytes,
                    0,
                    local_requests,
                    make_fake_stream(),
                )
            self.lookup_kernels[signature](
                self.device_view, registered.device_windows, (), *tensor_args, self._stream()
            )

        self.barrier()
        gather(requests, plan.splits[:, 1], output, True)
        if self.has_gin:
            remote_ids = self.exchange(requests, plan, remote_only=True)
            packed = output.new_empty((sum(plan.recv_splits), data_bytes))
            gather(remote_ids, plan.splits[:, 3], packed, False)
            self.exchange(packed, plan.reverse(), remote_only=True, output=output)
        # No optimizer may update/rebind this table until every remote reader has
        # finished. This barrier is per lookup, not per peer or per chunk.
        self.barrier()
        return output

    def close(self, clean: bool = False) -> None:
        """Release the communicator and its windows; a no-op once closed.

        ``clean=True`` is collective: a device barrier, a stream synchronization, then every
        window is closed and the communicator destroyed. Use it only where every rank of the
        row group arrives (the normal end of training). Otherwise, and always after a failed
        registration, the communicator is aborted without any device barrier, which cannot
        hang on a failed peer. A clean close whose barrier, synchronization or window close
        raises aborts the communicator before it re-raises, so nothing is left registered.
        """
        if self.comm is None:
            # At most prepare() ran: release its workspace.
            self.buffer = self.sequence = self.peers = self.uniform_plan = None
            return
        try:
            if clean and self.device_view is not None and not self.failed:
                try:
                    self.barrier()
                    torch.cuda.current_stream().synchronize()
                    for entry in self.windows.values():
                        entry.window.close()
                    self.window.close()
                except BaseException:
                    self._abort_after_failed_close()
                    raise
                self.comm.destroy()
            else:
                # Initialization may have failed before the device communicator existed.
                self.comm.abort()
        finally:
            self.tables.clear()
            self.pending.clear()
            self.windows.clear()
            self.lookup_kernels.clear()
            self.exchange_kernels.clear()
            self.comm = None
            self.buffer = self.sequence = self.window = self.dev_comm = None
            self.device_view = self.window_view = None
            self.peers = self.uniform_plan = None
            self.barrier_kernel = None
            self.published = self.failed = False

    def _abort_after_failed_close(self) -> None:
        """Abort after a failed clean close; the failure of the close is the one to report."""
        try:
            self.comm.abort()
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Engram row rank %d could not abort its communicator after a failed clean close",
                self.rank,
                exc_info=True,
            )
