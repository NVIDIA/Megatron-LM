# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CuTe/NCCL device kernels for bounded Engram row communication.

Imported only when selecting the ``symmetric`` backend. NCCL4Py owns the
communicator and registered windows; no CUDA C++ extension is built here.
All calls use the caller's current stream, serialized by EngramRuntime.
"""

import math

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


class EngramDeviceTransport:
    """One LSA/GIN transport per explicit row-shard group, shared by its tables."""

    def __init__(self, group):
        version = nccl.get_version()
        if version.nccl4py < Version("0.6.0") or version.libnccl is None:
            raise ImportError(
                "Engram symmetric requires nccl4py >= 0.6.0 and its NCCL device bitcode"
            )
        if version.libnccl.version != version.nccl_bindings:
            raise ImportError(
                "Engram symmetric requires matching NCCL runtime, NCCL4Py headers and device "
                f"bitcode; got runtime {version.libnccl.version}, headers {version.nccl_bindings}"
            )
        self.group = group
        # Megatron uses None for a local, unsharded table, not WORLD.
        self.rank = torch.distributed.get_rank(group) if group is not None else 0
        self.size = torch.distributed.get_world_size(group) if group is not None else 1
        self.channels = min(4, self.size)
        self.comm = None
        self.device_view = None
        self.tables = {}
        self.lookup_kernels = {}
        self.exchange_kernels = {}
        self.barrier_kernel = None

    def _initialize(self, device, workspace_bytes):
        if self.comm is not None:
            if device != self.buffer.device:
                raise ValueError("An Engram transport cannot span multiple local CUDA devices")
            return
        # Up to 128 CTAs with the default four peer channels. Keep at least 4 KiB
        # per slot, reducing stripes for small budgets without allocating more memory.
        stripe_bytes = self.size * 32 + 2 * (self.size + self.channels) * 4096
        self.stripes = 1 << min(5, max(0, (workspace_bytes // stripe_bytes).bit_length() - 1))
        unique_id = [nccl.get_unique_id() if self.rank == 0 else None]
        if self.size > 1:
            torch.distributed.broadcast_object_list(
                unique_id,
                src=torch.distributed.get_global_rank(self.group, 0),
                group=self.group,
                device=device,
            )
        with torch.cuda.device(device):
            self.comm = nccl.Communicator.init(
                nranks=self.size, rank=self.rank, unique_id=unique_id[0]
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
            self.buffer = nccl_torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
            self.buffer[self.flags_offset : self.flags_offset + flags_bytes].zero_()
            self.sequence = torch.zeros(
                self.size * self.stripes * 4, dtype=torch.int64, device=device
            )
            self.window = self.comm.register_window(self.buffer)
            if not self.window.is_valid:
                raise RuntimeError("NCCL did not register the Engram workspace")
            self.device_view, self.window_view = nc.DevComm(self.dev_comm), nc.Window(self.window)
        # Publish initialized flags before another GPU can touch this workspace.
        self.barrier()

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

    @torch.no_grad()
    def exchange(
        self,
        source,
        recv_splits,
        send_splits,
        plan,
        workspace_bytes,
        *,
        remote_only=False,
        output=None,
    ):
        """Exchange opaque rows with bounded slots and no host chunk loop."""
        if not source.is_cuda or source.ndim < 2:
            raise ValueError("Engram symmetric requires CUDA row tensors")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Engram communication must remain outside CUDA Graph capture")
        if (
            plan is None
            or plan.peer_counts[self.rank].tolist() != send_splits
            or plan.peer_counts[:, self.rank].tolist() != recv_splits
        ):
            raise ValueError("Engram exchange plan does not match its peer splits")
        if sum(send_splits) != source.shape[0]:
            raise ValueError("Engram input splits must cover every input row")
        self._initialize(source.device, workspace_bytes)
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
        if plan.max_rows == 0:
            return output
        aligned = source.data_ptr() % 4 == output.data_ptr() % 4 == row_bytes % 4 == 0
        word_dtype = torch.int32 if aligned else torch.uint8
        word_bytes = torch.empty((), dtype=word_dtype).element_size()
        counts = torch.tensor([send_splits, recv_splits], dtype=torch.int64)
        starts = counts.cumsum(1) - counts
        splits = torch.stack((starts[0], counts[0], starts[1], counts[1]), dim=1).to(source.device)
        # Keep the executable, bypassing auto-JIT argument processing on every
        # transfer. Row counts and widths are runtime arguments, not specializations.
        signature = (word_dtype, self.channels, self.stripes, self.has_gin, remote_only)
        if signature not in self.exchange_kernels:
            self.exchange_kernels[signature] = cute.compile(
                _exchange,
                make_fake_dev_comm(),
                make_fake_window(),
                make_fake_tensor(
                    cutlass.Int32 if aligned else cutlass.Uint8, (cute.sym_int64(),), (1,)
                ),
                make_fake_tensor(
                    cutlass.Int32 if aligned else cutlass.Uint8, (cute.sym_int64(),), (1,)
                ),
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
        self.exchange_kernels[signature](
            self.device_view,
            self.window_view,
            _rows_tensor(source.view(word_dtype).reshape(-1)),
            _rows_tensor(output.view(word_dtype).reshape(-1)),
            from_dlpack(splits),
            from_dlpack(self.sequence),
            cutlass.Int64(self.slot_bytes),
            cutlass.Int64(self.flags_offset),
            cutlass.Int64(row_bytes // word_bytes),
            self._stream(),
        )
        return output

    @staticmethod
    def _storage_view(tensor):
        storage = tensor.untyped_storage()
        stride = tensor.stride(0) * tensor.element_size()
        if not storage.nbytes():
            # Empty owners must still participate in collective registration.
            return nccl.mem_alloc(256, device=tensor.device), 0, stride

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

    def _register_tables(self, key, data, scales):
        signatures = tuple(
            (
                tensor.untyped_storage().data_ptr(),
                tensor.untyped_storage().nbytes(),
                tensor.data_ptr(),
                tensor.stride(0) * tensor.element_size(),
            )
            for tensor in (*data, *scales)
        )
        cached = self.tables.get(key)
        changed = cached is None or cached[0] != signatures
        descriptors, error = [], None
        try:
            # Reuse allocation queries and borrowed buffers until storage changes.
            descriptors = (
                [self._storage_view(tensor) for tensor in (*data, *scales)]
                if changed
                else cached[2]
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # A collective decision is necessary: only one shard may have had its
        # parameter storage replaced by an optimizer/DDP buffer refresh.
        info = (changed, [(offset, stride) for _, offset, stride in descriptors], error)
        gathered = [info]
        if self.size > 1:
            gathered = [None] * self.size
            torch.distributed.all_gather_object(gathered, info, group=self.group)
        errors = [f"row rank {rank}: {item[2]}" for rank, item in enumerate(gathered) if item[2]]
        if errors:
            raise RuntimeError("Cannot prepare Engram table windows: " + "; ".join(errors))
        if any(item[0] for item in gathered):
            self.barrier()
            torch.cuda.current_stream().synchronize()
            if cached is not None:
                for window in cached[1]:
                    window.close()
                self.tables.pop(key)
            windows = []
            try:
                for view, _, _ in descriptors:
                    window = self.comm.register_window(view)
                    if not window.is_valid:
                        raise RuntimeError("NCCL returned an invalid table window")
                    windows.append(window)
            except Exception as exc:
                for window in windows:
                    window.close()
                raise RuntimeError(
                    "Engram direct lookup requires registerable CUDA table storage. Use VMM "
                    "allocations (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) or an "
                    "NCCL allocator; the backend never copies the entire table."
                ) from exc
            metadata = []
            for table in range(len(data)):
                metadata.append(
                    [
                        [*item[1][table], *(item[1][table + len(data)] if scales else (0, 0))]
                        for item in gathered
                    ]
                )
            aligned = all(value % 4 == 0 for table in metadata for peer in table for value in peer)
            metadata = torch.tensor(metadata, dtype=torch.int64, device=data[0].device)
            # Keep registered allocations alive even when an optimizer replaces a view.
            cached = (signatures, windows, descriptors, metadata, aligned)
        self.tables[key] = cached
        return cached[1], cached[3], cached[4]

    @torch.no_grad()
    def lookup(self, key, data, scales, requests, send, recv, plans, workspace_bytes):
        """Read LSA rows directly and exchange only cross-domain requests/results."""
        if not requests.is_cuda:
            raise ValueError("Engram symmetric requires CUDA tensors")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Engram communication must remain outside CUDA Graph capture")
        self._initialize(requests.device, workspace_bytes)
        if any(tensor.ndim != 2 or tensor.stride(1) != 1 for tensor in (*data, *scales)):
            raise ValueError("Engram direct lookup requires row-major table data and scales")
        windows, metadata, aligned = self._register_tables(key, data, scales)
        data_bytes = data[0].shape[1] * data[0].element_size()
        scale_bytes = data[0].shape[1] // 32 if scales else 0
        row_bytes = data_bytes + scale_bytes
        dtype = torch.int32 if aligned and data_bytes % 4 == scale_bytes % 4 == 0 else torch.uint8
        word_bytes = torch.empty((), dtype=dtype).element_size()
        output = torch.empty((sum(send), row_bytes), dtype=torch.uint8, device=requests.device)
        data_windows = tuple(nc.Window(window) for window in windows[: len(data)])
        scale_windows = tuple(nc.Window(window) for window in windows[len(data) :])

        def gather(ids, splits, target, local_requests):
            if not ids.shape[0]:
                return
            peers = torch.repeat_interleave(
                torch.arange(self.size, device=ids.device),
                torch.tensor(splits, dtype=torch.int64, device=ids.device),
                output_size=ids.shape[0],
            )
            tensor_args = (
                from_dlpack(metadata),
                _rows_tensor(ids),
                _rows_tensor(peers),
                _rows_tensor(target.view(dtype).reshape(-1)),
                cutlass.Int64(ids.shape[0]),
            )
            signature = (len(data), dtype, data_bytes, scale_bytes, local_requests)
            if signature not in self.lookup_kernels:
                # Auto-JIT tries to extract device IR from host Window objects
                # nested in tuples (and rejects an empty scale tuple). Compile
                # with NCCL's type-only arguments, then pass live handles to the
                # executable. Row counts and table addresses remain dynamic.
                self.lookup_kernels[signature] = cute.compile(
                    _lookup,
                    make_fake_dev_comm(),
                    tuple(make_fake_window() for _ in data_windows),
                    tuple(make_fake_window() for _ in scale_windows),
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
                    scale_bytes // word_bytes,
                    local_requests,
                    make_fake_stream(),
                )
            self.lookup_kernels[signature](
                self.device_view, data_windows, scale_windows, *tensor_args, self._stream()
            )

        self.barrier()
        gather(requests, send, output, True)
        if self.has_gin:
            remote_ids = self.exchange(
                requests, recv, send, plans[0], workspace_bytes, remote_only=True
            )
            packed = output.new_empty((sum(recv), row_bytes))
            gather(remote_ids, recv, packed, False)
            self.exchange(
                packed, send, recv, plans[1], workspace_bytes, remote_only=True, output=output
            )
        # No optimizer may update/rebind this table until every remote reader has
        # finished. This barrier is per lookup, not per peer or per chunk.
        self.barrier()
        return output

    def close(self):
        """Collectively finish device work before releasing communicator resources."""
        if self.comm is not None:
            if self.device_view is None:
                # Initialization may fail before the device communicator exists.
                self.comm.abort()
            else:
                self.barrier()
                torch.cuda.current_stream().synchronize()
                self.comm.destroy()
            self.tables.clear()
            self.comm = None
            self.buffer = self.sequence = self.window = self.dev_comm = None
            self.device_view = self.window_view = None
            self.barrier_kernel = None
