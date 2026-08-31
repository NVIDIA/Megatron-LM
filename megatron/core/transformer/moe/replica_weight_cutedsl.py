# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CuTeDSL kernels for direct intra-node replica weight movement.

Only virtual weights and gradients occupy PyTorch native symmetric memory.
Source weights remain in parameter or GTP-gather storage: each owner pushes them
directly into destination virtual slots, and replica gradients are accumulated into
stable native-wgrad staging before autograd/DDP or GTP finalization. No activation
transport is involved.
"""

import functools
import math
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.cpasync as cpasync
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    from cutlass import BFloat16, Float32, Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op

    HAVE_CUTEDSL = True
except ImportError:
    cuda = MagicMock()
    cutlass = MagicMock()
    cute = MagicMock()
    cute.jit = null_decorator
    cute.kernel = null_decorator
    utils = MagicMock()
    BFloat16 = Float32 = Int32 = Int64 = MagicMock()
    llvm = MagicMock()
    make_ptr = MagicMock()
    T = MagicMock()
    dsl_user_op = null_decorator
    HAVE_CUTEDSL = False


_GRID_SYNC_TAG = 0x80000000
_BARRIER_TIMEOUT_CYCLES = 100 * 2_000_000_000
MAX_REPLICA_WEIGHT_SMS = 32


@cute.jit
def _tensor_1d(pointer, elements):
    return cute.make_tensor(pointer, cute.make_layout((elements,)))


def _inline_asm(result_type, operands, assembly, constraints, *, loc, ip):
    """Emit volatile PTX with the attributes shared by all transport primitives."""
    return llvm.inline_asm(
        result_type,
        operands,
        assembly,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _int32_asm(operands, assembly, constraints, *, loc, ip) -> Int32:
    return Int32(_inline_asm(T.i32(), operands, assembly, constraints, loc=loc, ip=ip))


@dsl_user_op
def _clock64(*, loc=None, ip=None) -> Int64:
    """Read the device clock for fail-fast barrier timeouts."""
    return Int64(_inline_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", loc=loc, ip=ip))


@dsl_user_op
def _device_trap(*, loc=None, ip=None) -> None:
    """Stop a mismatched launch rather than corrupting reusable barrier state."""
    _inline_asm(None, [], "trap;", "", loc=loc, ip=ip)


@dsl_user_op
def _atomic_add_release_gpu(address, value, *, loc=None, ip=None) -> Int32:
    """Issue a GPU-scope release atomic add."""
    return _int32_asm(
        [address, Int32(value).ir_value(loc=loc, ip=ip)],
        "atom.add.release.gpu.global.s32 $0, [$1], $2;",
        "=r,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_acquire_gpu(address, *, loc=None, ip=None) -> Int32:
    """Issue a GPU-scope acquire load."""
    return _int32_asm([address], "ld.acquire.gpu.global.s32 $0, [$1];", "=r,l", loc=loc, ip=ip)


@dsl_user_op
def _atomic_cas_release_sys(address, compare, value, *, loc=None, ip=None) -> Int32:
    """Issue a system-scope release compare-and-swap."""
    return _int32_asm(
        [address, Int32(compare).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "atom.global.release.sys.cas.b32 $0, [$1], $2, $3;",
        "=r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atomic_cas_acquire_sys(address, compare, value, *, loc=None, ip=None) -> Int32:
    """Issue a system-scope acquire compare-and-swap."""
    return _int32_asm(
        [address, Int32(compare).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "atom.global.acquire.sys.cas.b32 $0, [$1], $2, $3;",
        "=r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_reduce_async_bulk_add_f32(
    smem_ptr: cute.Pointer, gmem_ptr: cute.Pointer, store_bytes: int | Int32, *, loc=None, ip=None
) -> None:
    """Asynchronously add one shared-memory span into global FP32 memory."""
    smem_address = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    _inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_address, Int32(store_bytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@cute.jit
def _grid_sync(barrier, num_blocks: Int32, thread_idx: Int32):
    """Self-resetting cooperative-grid barrier."""
    cute.arch.sync_threads()
    if thread_idx == 0:
        address = barrier.toint().ir_value()
        increment = Int32(1)
        if cute.arch.block_idx()[0] == 0:
            increment = Int32(_GRID_SYNC_TAG) - (num_blocks - 1)
        previous = _atomic_add_release_gpu(address, increment)
        complete = cutlass.Boolean(False)
        while not complete:
            current = _load_acquire_gpu(address)
            complete = ((current ^ previous) & _GRID_SYNC_TAG) != 0
    cute.arch.sync_threads()


@cute.jit
def _cross_rank_barrier(
    signal_ptrs: cute.Tensor,
    grid_barrier,
    rank: Int32,
    num_ranks: cutlass.Constexpr[int],
    num_blocks: Int32,
    thread_idx: Int32,
):
    """Publish preceding stores and acquire peer stores entirely on device."""
    # Native symmetric memory exposes the same allocation through a local and
    # one or more peer VMM aliases. Publish stores through the alias proxy
    # before the system-scope release signals make them available to readers.
    cute.arch.fence_proxy("alias")
    _grid_sync(grid_barrier, num_blocks, thread_idx)
    if cute.arch.block_idx()[0] == 0 and thread_idx < num_ranks:
        # Give every rank pair an independent self-resetting signal. The old
        # implementation reduced every arrival into one system-scope atomic,
        # which serialized NVLink atomics and dominated FP8's smaller payload.
        peer_base = signal_ptrs[thread_idx]
        peer_signals = cute.make_ptr(Int32, peer_base, cute.AddressSpace.gmem, assumed_align=4)
        send_address = (peer_signals + rank).toint().ir_value()
        sent = cutlass.Boolean(False)
        start = _clock64()
        while not sent:
            previous = _atomic_cas_release_sys(send_address, Int32(0), Int32(1))
            sent = previous == Int32(0)
            if (_clock64() - start) >= Int64(_BARRIER_TIMEOUT_CYCLES):
                cute.printf(
                    "Replica CuTeDSL cross-rank send timed out: rank=%d peer=%d\n", rank, thread_idx
                )
                _device_trap()

        local_base = signal_ptrs[rank]
        local_signals = cute.make_ptr(Int32, local_base, cute.AddressSpace.gmem, assumed_align=4)
        receive_address = (local_signals + thread_idx).toint().ir_value()
        received = cutlass.Boolean(False)
        start = _clock64()
        while not received:
            previous = _atomic_cas_acquire_sys(receive_address, Int32(1), Int32(0))
            received = previous == Int32(1)
            if (_clock64() - start) >= Int64(_BARRIER_TIMEOUT_CYCLES):
                cute.printf(
                    "Replica CuTeDSL cross-rank receive timed out: rank=%d peer=%d\n",
                    rank,
                    thread_idx,
                )
                _device_trap()
    _grid_sync(grid_barrier, num_blocks, thread_idx)
    # The system-scope acquire above publishes peer writes through the generic
    # proxy. Bridge that visibility before a following asynchronous transaction.
    cute.arch.fence_proxy("async.global")


@cute.jit
def _bulk_load_copy(
    copy_atom: cute.CopyAtom,
    source,
    destination,
    barrier,
    bulk_elements: cutlass.Constexpr[int],
    bulks_per_chunk: cutlass.Constexpr[int],
):
    """Issue one warp-collective G2S chunk and arrive on its pipeline barrier."""
    for bulk in cutlass.range_constexpr(bulks_per_chunk):
        source_bulk = _tensor_1d(source + bulk * bulk_elements, bulk_elements)
        destination_bulk = _tensor_1d(destination + bulk * bulk_elements, bulk_elements)
        # CopyBulkG2SOp performs its own full-warp lane election.  Wrapping it
        # in elect_one leaves the compiler-generated election divergent and
        # can deadlock the pipeline.
        cute.copy(copy_atom, source_bulk, destination_bulk, mbar_ptr=barrier)


@cute.jit
def _bulk_store_copy(
    copy_atom: cute.CopyAtom,
    source,
    destination,
    bulk_elements: cutlass.Constexpr[int],
    bulks_per_chunk: cutlass.Constexpr[int],
):
    """Issue one warp-collective S2G chunk."""
    for bulk in cutlass.range_constexpr(bulks_per_chunk):
        source_bulk = _tensor_1d(source + bulk * bulk_elements, bulk_elements)
        destination_bulk = _tensor_1d(destination + bulk * bulk_elements, bulk_elements)
        # CopyBulkS2GOp performs its own full-warp lane election.
        cute.copy(copy_atom, source_bulk, destination_bulk)


class _ReplicaBulkKernel:
    """Configuration shared by the warp-specialized transport kernels."""

    STAGES = 3

    def __init__(
        self,
        *,
        world_size: int,
        num_local_experts: int,
        fc1_member_numel: int,
        fc2_member_numel: int,
        num_sms: int,
    ) -> None:
        self.world_size = world_size
        self.num_local_experts = num_local_experts
        self.fc1_member_numel = fc1_member_numel
        self.fc2_member_numel = fc2_member_numel
        self.fc1_member_chunks = fc1_member_numel // self.CHUNK_ELEMENTS
        self.fc2_member_chunks = fc2_member_numel // self.CHUNK_ELEMENTS
        self.num_sms = num_sms


class _ReplicaWeightPushKernel:
    """Push BF16 or MXFP8 owner-local weight components into replica slots."""

    NUM_THREADS = 64
    MAX_BULK_ELEMENTS = 8192
    MAX_DATA_CHUNK_BYTES = 65536
    MAX_SCALE_CHUNK_BYTES = 8192

    def __init__(
        self,
        *,
        world_size: int,
        num_local_experts: int,
        fc1_member_bytes: int,
        fc2_member_bytes: int,
        num_sms: int,
        fc1_scale_bytes: int = 0,
        fc2_scale_bytes: int = 0,
    ) -> None:
        self.world_size = world_size
        self.num_local_experts = num_local_experts
        self.fc1_member_bytes = fc1_member_bytes
        self.fc2_member_bytes = fc2_member_bytes
        self.fc1_scale_bytes = fc1_scale_bytes
        self.fc2_scale_bytes = fc2_scale_bytes
        self.has_scales = fc1_scale_bytes > 0
        if self.has_scales != (fc2_scale_bytes > 0):
            raise ValueError("Replica weight projections must either both have scales or neither.")
        self.num_sms = num_sms

        # BF16 uses 6x32-KiB stages. MXFP8's half-sized data payload uses
        # 3x64-KiB stages plus a short 2x8-KiB scale phase. Both retain the
        # same data-pipeline shared-memory footprint and generated schedule as
        # the formerly separate kernels.
        self.data_stages = 3 if self.has_scales else 6
        max_data_chunk_bytes = self.MAX_DATA_CHUNK_BYTES if self.has_scales else 32768
        self.chunk_bytes = math.gcd(
            max_data_chunk_bytes, math.gcd(fc1_member_bytes, fc2_member_bytes)
        )
        self.chunk_elements = self.chunk_bytes // 2
        self.bulk_elements = min(self.MAX_BULK_ELEMENTS, self.chunk_elements)
        self.bulks_per_chunk = self.chunk_elements // self.bulk_elements
        self.fc1_member_chunks = fc1_member_bytes // self.chunk_bytes
        self.fc2_member_chunks = fc2_member_bytes // self.chunk_bytes

        self.fc1_data_base_elements = 0
        self.fc1_scale_base_elements = num_local_experts * fc1_member_bytes // 2
        self.fc2_data_base_elements = num_local_experts * (fc1_member_bytes + fc1_scale_bytes) // 2
        self.fc2_scale_base_elements = (
            num_local_experts * (fc1_member_bytes + fc1_scale_bytes + fc2_member_bytes) // 2
        )
        if self.has_scales:
            self.scale_stages = 2
            self.scale_chunk_bytes = math.gcd(
                self.MAX_SCALE_CHUNK_BYTES, math.gcd(fc1_scale_bytes, fc2_scale_bytes)
            )
            self.scale_chunk_elements = self.scale_chunk_bytes // 2
            self.scale_bulk_elements = min(self.MAX_BULK_ELEMENTS, self.scale_chunk_elements)
            self.scale_bulks_per_chunk = self.scale_chunk_elements // self.scale_bulk_elements
            self.fc1_scale_chunks = fc1_scale_bytes // self.scale_chunk_bytes
            self.fc2_scale_chunks = fc2_scale_bytes // self.scale_chunk_bytes

    def _smem_bytes(self) -> int:
        scale_stages = self.scale_stages if self.has_scales else 0
        scale_chunk_bytes = self.scale_chunk_bytes if self.has_scales else 0
        stages = self.data_stages * self.chunk_bytes + scale_stages * scale_chunk_bytes
        barriers = (self.data_stages + scale_stages) * 2 * 8
        plan = (3 * self.world_size * self.num_local_experts + 1) * 4
        return stages + barriers + plan + 256

    @cute.jit
    def __call__(
        self,
        fc1_source_bases_ptr: cute.Pointer,
        fc2_source_bases_ptr: cute.Pointer,
        fc1_scale_bases_ptr: cute.Pointer,
        fc2_scale_bases_ptr: cute.Pointer,
        peer_base_ptr: cute.Pointer,
        signal_base_ptr: cute.Pointer,
        experts_ptr: cute.Pointer,
        grid_barrier_ptr: cute.Pointer,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        fc1_source_bases = _tensor_1d(fc1_source_bases_ptr, self.num_local_experts)
        fc2_source_bases = _tensor_1d(fc2_source_bases_ptr, self.num_local_experts)
        fc1_scale_bases = _tensor_1d(fc1_scale_bases_ptr, self.num_local_experts)
        fc2_scale_bases = _tensor_1d(fc2_scale_bases_ptr, self.num_local_experts)
        peer_bases = _tensor_1d(peer_base_ptr, self.world_size)
        signal_bases = _tensor_1d(signal_base_ptr, self.world_size)
        experts = _tensor_1d(experts_ptr, self.world_size * self.num_local_experts)
        self.kernel(
            fc1_source_bases,
            fc2_source_bases,
            fc1_scale_bases,
            fc2_scale_bases,
            peer_bases,
            signal_bases,
            experts,
            grid_barrier_ptr,
            rank,
        ).launch(
            grid=(self.num_sms, 1, 1),
            block=(self.NUM_THREADS, 1, 1),
            smem=self._smem_bytes(),
            stream=stream,
            cooperative=True,
        )

    @cute.kernel
    def kernel(
        self,
        fc1_source_bases: cute.Tensor,
        fc2_source_bases: cute.Tensor,
        fc1_scale_bases: cute.Tensor,
        fc2_scale_bases: cute.Tensor,
        peer_bases: cute.Tensor,
        signal_bases: cute.Tensor,
        experts: cute.Tensor,
        grid_barrier,
        rank: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        stages = cutlass.const_expr(self.data_stages)
        smem = utils.SmemAllocator()
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)
        owner_experts = smem.allocate_tensor(
            Int32, cute.make_layout((self.world_size * self.num_local_experts,)), byte_alignment=16
        )
        destinations = smem.allocate_tensor(
            Int32, cute.make_layout((self.world_size * self.num_local_experts,)), byte_alignment=16
        )
        destination_slots = smem.allocate_tensor(
            Int32, cute.make_layout((self.world_size * self.num_local_experts,)), byte_alignment=16
        )
        active_count = smem.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)
        stage_smem = smem.allocate_tensor(
            BFloat16,
            cute.make_ordered_layout((self.chunk_elements, stages), order=(0, 1)),
            byte_alignment=128,
        )
        if cutlass.const_expr(self.has_scales):
            scale_stages = cutlass.const_expr(self.scale_stages)
            scale_load_mbar = smem.allocate_array(Int64, num_elems=2 * scale_stages)
            scale_stage_smem = smem.allocate_tensor(
                BFloat16,
                cute.make_ordered_layout((self.scale_chunk_elements, scale_stages), order=(0, 1)),
                byte_alignment=128,
            )

        if tid == 0:
            count = Int32(0)
            plan_entries = cutlass.const_expr(self.world_size * self.num_local_experts)
            # Keep this latency-sensitive scan as one compact runtime loop.
            for entry in cutlass.range(0, plan_entries, 1, unroll=1):
                destination = entry // self.num_local_experts
                slot = entry - destination * self.num_local_experts
                expert = experts[entry]
                owner_expert = expert - rank * self.num_local_experts
                if owner_expert >= Int32(0) and owner_expert < Int32(self.num_local_experts):
                    owner_experts[count] = owner_expert
                    destinations[count] = destination
                    destination_slots[count] = slot
                    count += Int32(1)
            active_count[0] = count
        cute.arch.sync_threads()

        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar,
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=cutlass.const_expr(self.chunk_bytes),
        )
        if cutlass.const_expr(self.has_scales):
            scale_load_pipe = pipeline.PipelineTmaAsync.create(
                barrier_storage=scale_load_mbar,
                num_stages=scale_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=cutlass.const_expr(self.scale_chunk_bytes),
            )
        load_atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), BFloat16)
        store_atom = cute.make_copy_atom(cpasync.CopyBulkS2GOp(), BFloat16)
        chunks_per_replica = cutlass.const_expr(self.fc1_member_chunks + self.fc2_member_chunks)
        remote_work = active_count[0] * chunks_per_replica
        if warp == 0:
            load_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, stages)
            for work in cutlass.range(block, remote_work, self.num_sms, unroll=1):
                member_chunk = work // active_count[0]
                active = work - member_chunk * active_count[0]
                owner_expert = owner_experts[active]
                source = cute.make_ptr(
                    BFloat16,
                    fc1_source_bases[owner_expert],
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                projection_chunk = member_chunk
                if member_chunk >= self.fc1_member_chunks:
                    source = cute.make_ptr(
                        BFloat16,
                        fc2_source_bases[owner_expert],
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    projection_chunk = member_chunk - self.fc1_member_chunks
                source_offset = Int64(projection_chunk * self.chunk_elements)
                load_pipe.producer_acquire(load_state)
                stage = stage_smem[(None, load_state.index)]
                _bulk_load_copy(
                    load_atom,
                    source + source_offset,
                    stage.iterator,
                    load_pipe.producer_get_barrier(load_state),
                    cutlass.const_expr(self.bulk_elements),
                    cutlass.const_expr(self.bulks_per_chunk),
                )
                load_pipe.producer_commit(load_state)
                load_state.advance()
        elif warp == 1:
            consume_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, stages)
            for work in cutlass.range(block, remote_work, self.num_sms, unroll=1):
                member_chunk = work // active_count[0]
                active = work - member_chunk * active_count[0]
                destination = destinations[active]
                slot = destination_slots[active]
                member_elements = cutlass.const_expr(self.fc1_member_bytes // 2)
                projection_base = Int64(self.fc1_data_base_elements)
                projection_chunk = member_chunk
                if member_chunk >= self.fc1_member_chunks:
                    member_elements = cutlass.const_expr(self.fc2_member_bytes // 2)
                    projection_base = Int64(self.fc2_data_base_elements)
                    projection_chunk = member_chunk - self.fc1_member_chunks
                destination_offset = (
                    projection_base
                    + Int64(slot) * member_elements
                    + Int64(projection_chunk * self.chunk_elements)
                )
                peer = cute.make_ptr(
                    BFloat16, peer_bases[destination], cute.AddressSpace.gmem, assumed_align=16
                )
                load_pipe.consumer_wait(consume_state)
                stage = stage_smem[(None, consume_state.index)]
                _bulk_store_copy(
                    store_atom,
                    stage.iterator,
                    peer + destination_offset,
                    cutlass.const_expr(self.bulk_elements),
                    cutlass.const_expr(self.bulks_per_chunk),
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                load_pipe.consumer_release(consume_state)
                consume_state.advance()
        if cutlass.const_expr(self.has_scales):
            # One CTA bulk-copy engine serves both phases, so finish data before
            # issuing the much smaller scale transfer.
            cute.arch.sync_threads()
            scale_chunks_per_replica = cutlass.const_expr(
                self.fc1_scale_chunks + self.fc2_scale_chunks
            )
            scale_work = active_count[0] * scale_chunks_per_replica
            if warp == 0:
                scale_load_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, scale_stages
                )
                for work in cutlass.range(block, scale_work, self.num_sms, unroll=1):
                    member_chunk = work // active_count[0]
                    active = work - member_chunk * active_count[0]
                    owner_expert = owner_experts[active]
                    source = cute.make_ptr(
                        BFloat16,
                        fc1_scale_bases[owner_expert],
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    projection_chunk = member_chunk
                    if member_chunk >= self.fc1_scale_chunks:
                        source = cute.make_ptr(
                            BFloat16,
                            fc2_scale_bases[owner_expert],
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        projection_chunk = member_chunk - self.fc1_scale_chunks
                    source_offset = Int64(projection_chunk * self.scale_chunk_elements)
                    scale_load_pipe.producer_acquire(scale_load_state)
                    stage = scale_stage_smem[(None, scale_load_state.index)]
                    _bulk_load_copy(
                        load_atom,
                        source + source_offset,
                        stage.iterator,
                        scale_load_pipe.producer_get_barrier(scale_load_state),
                        cutlass.const_expr(self.scale_bulk_elements),
                        cutlass.const_expr(self.scale_bulks_per_chunk),
                    )
                    scale_load_pipe.producer_commit(scale_load_state)
                    scale_load_state.advance()
            elif warp == 1:
                scale_consume_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, scale_stages
                )
                for work in cutlass.range(block, scale_work, self.num_sms, unroll=1):
                    member_chunk = work // active_count[0]
                    active = work - member_chunk * active_count[0]
                    destination = destinations[active]
                    slot = destination_slots[active]
                    member_elements = cutlass.const_expr(self.fc1_scale_bytes // 2)
                    projection_base = Int64(self.fc1_scale_base_elements)
                    projection_chunk = member_chunk
                    if member_chunk >= self.fc1_scale_chunks:
                        member_elements = cutlass.const_expr(self.fc2_scale_bytes // 2)
                        projection_base = Int64(self.fc2_scale_base_elements)
                        projection_chunk = member_chunk - self.fc1_scale_chunks
                    destination_offset = (
                        projection_base
                        + Int64(slot) * member_elements
                        + Int64(projection_chunk * self.scale_chunk_elements)
                    )
                    peer = cute.make_ptr(
                        BFloat16, peer_bases[destination], cute.AddressSpace.gmem, assumed_align=16
                    )
                    scale_load_pipe.consumer_wait(scale_consume_state)
                    stage = scale_stage_smem[(None, scale_consume_state.index)]
                    _bulk_store_copy(
                        store_atom,
                        stage.iterator,
                        peer + destination_offset,
                        cutlass.const_expr(self.scale_bulk_elements),
                        cutlass.const_expr(self.scale_bulks_per_chunk),
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    scale_load_pipe.consumer_release(scale_consume_state)
                    scale_consume_state.advance()

        _cross_rank_barrier(
            signal_bases,
            grid_barrier,
            rank,
            cutlass.const_expr(self.world_size),
            Int32(self.num_sms),
            tid,
        )


class _ReplicaGradReduceKernel(_ReplicaBulkKernel):
    """Accumulate FP32 or BF16 replica gradients and clear used local slots."""

    BULK_ELEMENTS = 4096
    BULKS_PER_CHUNK = 4
    CHUNK_ELEMENTS = BULK_ELEMENTS * BULKS_PER_CHUNK
    BF16_CONSUMER_WARPS = 16
    # Transport the chunks in several passes over disjoint ranges. The
    # cross-rank barrier closing a pass proves every owner has read that range,
    # so the next pass can zero the local slots behind it while it waits on the
    # wire. Only the trailing range stays exposed. More passes hide more of the
    # zero fill but add a barrier each; four is the measured optimum.
    TRANSPORT_PASSES = 4
    # Zero the slots with ordinary vector stores instead of bulk copies: the
    # retiring warp runs beside the peer loads, and both would otherwise queue
    # behind the same asynchronous-copy engine.
    CLEAR_STORE_BITS = 128

    def __init__(self, *, grad_dtype: torch.dtype, **kwargs) -> None:
        if grad_dtype == torch.bfloat16:
            self.CHUNK_ELEMENTS = math.gcd(
                2 * self.CHUNK_ELEMENTS,
                math.gcd(kwargs["fc1_member_numel"], kwargs["fc2_member_numel"]),
            )
            self.BULKS_PER_CHUNK = self.CHUNK_ELEMENTS // self.BULK_ELEMENTS
        super().__init__(**kwargs)
        self.is_bf16 = grad_dtype == torch.bfloat16
        self.grad_type = BFloat16 if self.is_bf16 else Float32
        self.element_bytes = 2 if self.is_bf16 else 4
        self.clear_vector = self.CLEAR_STORE_BITS // (8 * self.element_bytes)
        # One producer warp, the reduction warps, and one warp that retires the
        # slots the previous pass drained.
        self.consumer_warps = self.BF16_CONSUMER_WARPS if self.is_bf16 else 1
        self.num_threads = 32 * (self.consumer_warps + 2)

    def _smem_bytes(self) -> int:
        stages = self.STAGES * self.CHUNK_ELEMENTS * self.element_bytes
        barriers = self.STAGES * 2 * 8
        plan = (self.num_local_experts * (self.world_size + 1) + 1) * 4
        return stages + barriers + plan + 256

    def _chunk_boundaries(self) -> list[int]:
        """Return the transport pass boundaries in per-expert chunk units."""
        chunks_per_expert = self.fc1_member_chunks + self.fc2_member_chunks
        return [
            chunks_per_expert * index // self.TRANSPORT_PASSES
            for index in range(self.TRANSPORT_PASSES + 1)
        ]

    @cute.jit
    def _clear_slot_bulk(
        self,
        arena: cute.Tensor,
        active_slots: cute.Tensor,
        zero_atom: cute.CopyAtom,
        zeros: cute.Tensor,
        lane: Int32,
        work: Int32,
        first_bulk: cutlass.Constexpr[int],
        range_bulks: cutlass.Constexpr[int],
    ):
        """Zero one bulk of one locally hosted replica slot with the calling warp.

        ``work`` enumerates ``(active slot, bulk within the retired range)``,
        where the range holds ``range_bulks`` bulks per slot and starts at
        whole-slot bulk ``first_bulk``.
        """
        active = work // range_bulks
        slot_bulk = first_bulk + work - active * range_bulks
        slot = Int64(active_slots[active])
        bulks_per_fc1 = cutlass.const_expr(self.fc1_member_numel // self.BULK_ELEMENTS)
        destination_offset = slot * self.fc1_member_numel + Int64(slot_bulk * self.BULK_ELEMENTS)
        if slot_bulk >= bulks_per_fc1:
            destination_offset = (
                Int64(self.num_local_experts * self.fc1_member_numel)
                + slot * self.fc2_member_numel
                + Int64((slot_bulk - bulks_per_fc1) * self.BULK_ELEMENTS)
            )
        vector = cutlass.const_expr(self.clear_vector)
        store_bytes = cutlass.const_expr(self.CLEAR_STORE_BITS // 8)
        for index in cutlass.range_constexpr(self.BULK_ELEMENTS // (32 * vector)):
            # Offsetting an iterator drops its alignment, and the vector store
            # atom requires the full width, so restate it on every destination.
            destination = (
                arena.iterator + destination_offset + (lane + index * 32) * vector
            ).align(store_bytes)
            cute.copy(zero_atom, zeros, _tensor_1d(destination, vector))

    @cute.jit
    def __call__(
        self,
        arena_ptr: cute.Pointer,
        fc1_native_grad_bases_ptr: cute.Pointer,
        fc2_native_grad_bases_ptr: cute.Pointer,
        peer_base_ptr: cute.Pointer,
        signal_base_ptr: cute.Pointer,
        experts_ptr: cute.Pointer,
        grid_barrier_ptr: cute.Pointer,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        fc1_numel = cutlass.const_expr(self.num_local_experts * self.fc1_member_numel)
        fc2_numel = cutlass.const_expr(self.num_local_experts * self.fc2_member_numel)
        arena = _tensor_1d(arena_ptr, fc1_numel + fc2_numel)
        fc1_native_grad_bases = _tensor_1d(fc1_native_grad_bases_ptr, self.num_local_experts)
        fc2_native_grad_bases = _tensor_1d(fc2_native_grad_bases_ptr, self.num_local_experts)
        peer_bases = _tensor_1d(peer_base_ptr, self.world_size)
        signal_bases = _tensor_1d(signal_base_ptr, self.world_size)
        experts = _tensor_1d(experts_ptr, self.world_size * self.num_local_experts)
        self.kernel(
            arena,
            fc1_native_grad_bases,
            fc2_native_grad_bases,
            peer_bases,
            signal_bases,
            experts,
            grid_barrier_ptr,
            rank,
        ).launch(
            grid=(self.num_sms, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=self._smem_bytes(),
            stream=stream,
            cooperative=True,
        )

    @cute.kernel
    def kernel(
        self,
        arena: cute.Tensor,
        fc1_native_grad_bases: cute.Tensor,
        fc2_native_grad_bases: cute.Tensor,
        peer_bases: cute.Tensor,
        signal_bases: cute.Tensor,
        experts: cute.Tensor,
        grid_barrier,
        rank: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        stages = cutlass.const_expr(self.STAGES)
        chunk_bytes = cutlass.const_expr(self.CHUNK_ELEMENTS * self.element_bytes)
        virtual_fc1_numel = cutlass.const_expr(self.num_local_experts * self.fc1_member_numel)

        smem = utils.SmemAllocator()
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)
        stage_smem = smem.allocate_tensor(
            self.grad_type,
            cute.make_ordered_layout((self.CHUNK_ELEMENTS, stages), order=(0, 1)),
            byte_alignment=128,
        )
        matches = smem.allocate_tensor(
            Int32, cute.make_layout((self.num_local_experts * self.world_size,)), byte_alignment=16
        )
        active_slots = smem.allocate_tensor(
            Int32, cute.make_layout((self.num_local_experts,)), byte_alignment=16
        )
        active_count = smem.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)

        for index in cutlass.range(tid, self.num_local_experts * self.world_size, self.num_threads):
            matches[index] = Int32(-1)
        cute.arch.sync_threads()
        for index in cutlass.range(tid, self.num_local_experts * self.world_size, self.num_threads):
            expert = experts[index]
            owner_expert = expert - rank * self.num_local_experts
            if owner_expert >= Int32(0) and owner_expert < Int32(self.num_local_experts):
                destination = index // self.num_local_experts
                slot = index - destination * self.num_local_experts
                matches[owner_expert * self.world_size + destination] = slot
        if tid == 0:
            count = Int32(0)
            for slot in cutlass.range_constexpr(self.num_local_experts):
                if experts[rank * self.num_local_experts + slot] >= Int32(0):
                    active_slots[count] = Int32(slot)
                    count += Int32(1)
            active_count[0] = count
        cute.arch.sync_threads()

        _cross_rank_barrier(
            signal_bases,
            grid_barrier,
            rank,
            cutlass.const_expr(self.world_size),
            Int32(self.num_sms),
            tid,
        )

        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar,
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cutlass.const_expr(self.consumer_warps)
            ),
            tx_count=chunk_bytes,
        )
        load_atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), self.grad_type)
        zero_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.grad_type,
            num_bits_per_copy=cutlass.const_expr(self.CLEAR_STORE_BITS),
        )
        zeros = cute.make_rmem_tensor((self.clear_vector,), self.grad_type)
        for index in cutlass.range_constexpr(self.clear_vector):
            zeros[index] = self.grad_type(0.0)
        load_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, stages)
        consume_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, stages)
        boundaries = self._chunk_boundaries()
        bulks_per_chunk = cutlass.const_expr(self.BULKS_PER_CHUNK)
        for transport_pass in cutlass.range_constexpr(self.TRANSPORT_PASSES):
            first_chunk = cutlass.const_expr(boundaries[transport_pass])
            retired_first_bulk = cutlass.const_expr(
                boundaries[transport_pass - 1] * bulks_per_chunk if transport_pass else 0
            )
            retired_bulks = cutlass.const_expr(
                first_chunk * bulks_per_chunk - retired_first_bulk if transport_pass else 0
            )
            retired_work = active_count[0] * retired_bulks
            if warp <= self.consumer_warps:
                # Sweep the local experts fastest. Consecutive experts come from
                # different peers, so this keeps the blocks running at any
                # instant spread over the peers instead of pointing all of them
                # at the one peer that hosts the expert currently being swept.
                for work in cutlass.range(
                    first_chunk * self.num_local_experts + block,
                    cutlass.const_expr(boundaries[transport_pass + 1] * self.num_local_experts),
                    self.num_sms,
                    unroll=1,
                ):
                    expert_chunk = work // self.num_local_experts
                    local_expert = work - expert_chunk * self.num_local_experts
                    member_chunk = expert_chunk
                    member_numel = cutlass.const_expr(self.fc1_member_numel)
                    virtual_projection_base = Int64(0)
                    native_grad_bases = fc1_native_grad_bases
                    if expert_chunk >= self.fc1_member_chunks:
                        member_chunk = expert_chunk - self.fc1_member_chunks
                        member_numel = cutlass.const_expr(self.fc2_member_numel)
                        virtual_projection_base = Int64(virtual_fc1_numel)
                        native_grad_bases = fc2_native_grad_bases
                    member_offset = Int64(member_chunk * self.CHUNK_ELEMENTS)

                    if warp == 0:
                        for destination in cutlass.range_constexpr(self.world_size):
                            slot = matches[local_expert * self.world_size + destination]
                            if slot >= Int32(0):
                                peer = cute.make_ptr(
                                    self.grad_type,
                                    peer_bases[destination],
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                )
                                peer_offset = (
                                    virtual_projection_base
                                    + Int64(slot) * member_numel
                                    + member_offset
                                )
                                load_pipe.producer_acquire(load_state)
                                stage = stage_smem[(None, load_state.index)]
                                _bulk_load_copy(
                                    load_atom,
                                    peer + peer_offset,
                                    stage.iterator,
                                    load_pipe.producer_get_barrier(load_state),
                                    cutlass.const_expr(self.BULK_ELEMENTS),
                                    cutlass.const_expr(self.BULKS_PER_CHUNK),
                                )
                                load_pipe.producer_commit(load_state)
                                load_state.advance()
                    else:
                        source_count = Int32(0)
                        for destination in cutlass.range_constexpr(self.world_size):
                            if matches[local_expert * self.world_size + destination] >= Int32(0):
                                source_count += Int32(1)
                        if source_count > Int32(0):
                            native_destination = cute.make_ptr(
                                self.grad_type,
                                native_grad_bases[local_expert],
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            native_destination = _tensor_1d(
                                native_destination + member_offset, self.CHUNK_ELEMENTS
                            )
                            if cutlass.const_expr(self.is_bf16):
                                consumer_thread = tid - Int32(32)
                                consumer_threads = cutlass.const_expr(32 * self.consumer_warps)
                                thread_elements = cutlass.const_expr(
                                    self.CHUNK_ELEMENTS // consumer_threads
                                )
                                accumulator = cute.make_rmem_tensor((thread_elements,), Float32)
                                for register in cutlass.range_constexpr(thread_elements):
                                    element = consumer_thread + register * consumer_threads
                                    accumulator[register] = Float32(native_destination[element])
                                for _source_index in cutlass.range(source_count, unroll=1):
                                    load_pipe.consumer_wait(consume_state)
                                    stage = stage_smem[(None, consume_state.index)]
                                    for register in cutlass.range_constexpr(thread_elements):
                                        element = consumer_thread + register * consumer_threads
                                        accumulator[register] += Float32(stage[element])
                                    load_pipe.consumer_release(consume_state)
                                    consume_state.advance()
                                # Round only on the final store; peer traffic and persistent
                                # storage remain BF16 while local additions use FP32.
                                for register in cutlass.range_constexpr(thread_elements):
                                    element = consumer_thread + register * consumer_threads
                                    native_destination[element] = BFloat16(accumulator[register])
                            else:
                                for _source_index in cutlass.range(source_count, unroll=1):
                                    load_pipe.consumer_wait(consume_state)
                                    with cute.arch.elect_one():
                                        for bulk in cutlass.range_constexpr(self.BULKS_PER_CHUNK):
                                            stage = stage_smem[(None, consume_state.index)]
                                            _cp_reduce_async_bulk_add_f32(
                                                stage.iterator + bulk * self.BULK_ELEMENTS,
                                                native_destination.iterator
                                                + bulk * self.BULK_ELEMENTS,
                                                self.BULK_ELEMENTS * 4,
                                            )
                                        cute.arch.cp_async_bulk_commit_group()
                                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                                    load_pipe.consumer_release(consume_state)
                                    consume_state.advance()
            else:
                if cutlass.const_expr(retired_bulks > 0):
                    # The barrier that closed the previous pass proved every owner
                    # has read that range, so this warp zeroes it concurrently with
                    # the transport warps above, which are waiting on the wire.
                    for work in cutlass.range(block, retired_work, self.num_sms, unroll=1):
                        self._clear_slot_bulk(
                            arena,
                            active_slots,
                            zero_atom,
                            zeros,
                            lane,
                            work,
                            retired_first_bulk,
                            retired_bulks,
                        )

            _cross_rank_barrier(
                signal_bases,
                grid_barrier,
                rank,
                cutlass.const_expr(self.world_size),
                Int32(self.num_sms),
                tid,
            )

        # Owners have now acquired every remote virtual gradient. Retire the
        # trailing range with all warps; planner slots need not form a dense
        # prefix, so address them through the active-slot table.
        trailing_first_bulk = cutlass.const_expr(boundaries[-2] * bulks_per_chunk)
        trailing_bulks = cutlass.const_expr(boundaries[-1] * bulks_per_chunk - trailing_first_bulk)
        warps = cutlass.const_expr(self.num_threads // 32)
        trailing_work = active_count[0] * trailing_bulks
        for work in cutlass.range(
            block * warps + warp, trailing_work, self.num_sms * warps, unroll=1
        ):
            self._clear_slot_bulk(
                arena,
                active_slots,
                zero_atom,
                zeros,
                lane,
                work,
                trailing_first_bulk,
                trailing_bulks,
            )


def _validate_compile_shape(
    world_size: int,
    num_local_experts: int,
    fc1_member_numel: int,
    fc2_member_numel: int,
    num_sms: int,
) -> None:
    if not HAVE_CUTEDSL:
        raise ImportError("Replica CuTeDSL weight transfer requires nvidia-cutlass-dsl.")
    if world_size <= 0 or num_local_experts <= 0 or num_sms <= 0:
        raise ValueError("Replica CuTeDSL launch dimensions must be positive.")
    if num_sms > MAX_REPLICA_WEIGHT_SMS:
        raise ValueError(
            "Replica CuTeDSL weight kernels are limited to "
            f"{MAX_REPLICA_WEIGHT_SMS} SMs, got {num_sms}."
        )
    # One thread of the first block signals each peer across the barrier, so no
    # transport block may be narrower than the rank count. The push kernel is
    # the narrowest of the two.
    max_ranks = _ReplicaWeightPushKernel.NUM_THREADS
    if world_size > max_ranks:
        raise ValueError(
            "Replica CuTeDSL supports at most " f"{max_ranks} EP ranks, got {world_size}."
        )
    tile_elements = _ReplicaGradReduceKernel.CHUNK_ELEMENTS
    if fc1_member_numel % tile_elements or fc2_member_numel % tile_elements:
        raise ValueError(
            "Replica CuTeDSL projections must contain a multiple of "
            f"{tile_elements} elements "
            f"per expert, got {(fc1_member_numel, fc2_member_numel)}."
        )


@functools.lru_cache(maxsize=None)
def _get_compiled_bf16_weight_push(
    world_size: int,
    num_local_experts: int,
    fc1_member_numel: int,
    fc2_member_numel: int,
    num_sms: int,
    device_index: int,
):
    _validate_compile_shape(
        world_size, num_local_experts, fc1_member_numel, fc2_member_numel, num_sms
    )
    kernel = _ReplicaWeightPushKernel(
        world_size=world_size,
        num_local_experts=num_local_experts,
        fc1_member_bytes=2 * fc1_member_numel,
        fc2_member_bytes=2 * fc2_member_numel,
        num_sms=num_sms,
    )
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    i64_ptr = make_ptr(Int64, 0, cute.AddressSpace.gmem, assumed_align=8)
    pointer_args = (i64_ptr, i64_ptr, i64_ptr, i64_ptr, i64_ptr, i64_ptr, i32_ptr, i32_ptr)
    stream = cuda.CUstream(0)
    with torch.cuda.device(device_index):
        return cute.compile(kernel, *pointer_args, Int32(0), stream)


@functools.lru_cache(maxsize=None)
def _get_compiled_grad_reduce(
    world_size: int,
    num_local_experts: int,
    fc1_member_numel: int,
    fc2_member_numel: int,
    num_sms: int,
    device_index: int,
    grad_dtype: torch.dtype = torch.float32,
):
    _validate_compile_shape(
        world_size, num_local_experts, fc1_member_numel, fc2_member_numel, num_sms
    )
    if grad_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "Replica CuTeDSL gradients must use torch.float32 or torch.bfloat16, "
            f"got {grad_dtype}."
        )
    pointer_type = BFloat16 if grad_dtype == torch.bfloat16 else Float32
    kernel = _ReplicaGradReduceKernel(
        grad_dtype=grad_dtype,
        world_size=world_size,
        num_local_experts=num_local_experts,
        fc1_member_numel=fc1_member_numel,
        fc2_member_numel=fc2_member_numel,
        num_sms=num_sms,
    )
    grad_ptr = make_ptr(pointer_type, 0, cute.AddressSpace.gmem, assumed_align=16)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    i64_ptr = make_ptr(Int64, 0, cute.AddressSpace.gmem, assumed_align=8)
    pointer_args = (grad_ptr, i64_ptr, i64_ptr, i64_ptr, i64_ptr, i32_ptr, i32_ptr)
    stream = cuda.CUstream(0)
    with torch.cuda.device(device_index):
        return cute.compile(kernel, *pointer_args, Int32(0), stream)


def compile_replica_weight_kernels(
    *,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
    device_index: int,
    grad_dtype: torch.dtype = torch.float32,
    rowwise_scale_numels: tuple[int, int] | None = None,
    columnwise_scale_numels: tuple[int, int] | None = None,
) -> None:
    """JIT compile the format-specific weight push and shared grad reduction."""
    _get_compiled_grad_reduce(
        world_size,
        num_local_experts,
        member_numels[0],
        member_numels[1],
        num_sms,
        device_index,
        grad_dtype,
    )
    if rowwise_scale_numels is None and columnwise_scale_numels is None:
        _get_compiled_bf16_weight_push(
            world_size, num_local_experts, member_numels[0], member_numels[1], num_sms, device_index
        )
        return
    if rowwise_scale_numels is None or columnwise_scale_numels is None:
        raise ValueError("MXFP8 compilation requires both rowwise and columnwise scale shapes.")
    _get_compiled_mxfp8_weight_kernels(
        world_size,
        num_local_experts,
        member_numels[0],
        member_numels[1],
        rowwise_scale_numels[0],
        rowwise_scale_numels[1],
        columnwise_scale_numels[0],
        columnwise_scale_numels[1],
        num_sms,
        device_index,
    )


def _validate_mxfp8_compile_shape(
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    rowwise_scale_numels: tuple[int, int],
    columnwise_scale_numels: tuple[int, int],
    num_sms: int,
) -> None:
    """Validate the aligned native MXFP8 byte layout used by the push kernels."""
    _validate_compile_shape(
        world_size, num_local_experts, member_numels[0], member_numels[1], num_sms
    )
    for projection, member_numel in enumerate(member_numels):
        for orientation, scale_numels in (
            ("rowwise", rowwise_scale_numels),
            ("columnwise", columnwise_scale_numels),
        ):
            scale_numel = scale_numels[projection]
            if scale_numel <= 0 or scale_numel % 2:
                raise ValueError(
                    "Replica CuTeDSL MXFP8 scales must contain a positive even number "
                    f"of bytes; {orientation} projection {projection} has {scale_numel}."
                )
            if scale_numel * 32 != member_numel:
                raise ValueError(
                    "Replica CuTeDSL MXFP8 requires one E8M0 scale byte per 32 "
                    f"weight bytes; {orientation} projection {projection} has "
                    f"weight_bytes={member_numel}, scale_bytes={scale_numel}."
                )


@functools.lru_cache(maxsize=None)
def _get_compiled_mxfp8_weight_kernels(
    world_size: int,
    num_local_experts: int,
    fc1_member_numel: int,
    fc2_member_numel: int,
    fc1_rowwise_scale_numel: int,
    fc2_rowwise_scale_numel: int,
    fc1_columnwise_scale_numel: int,
    fc2_columnwise_scale_numel: int,
    num_sms: int,
    device_index: int,
):
    member_numels = (fc1_member_numel, fc2_member_numel)
    rowwise_scale_numels = (fc1_rowwise_scale_numel, fc2_rowwise_scale_numel)
    columnwise_scale_numels = (fc1_columnwise_scale_numel, fc2_columnwise_scale_numel)
    _validate_mxfp8_compile_shape(
        world_size,
        num_local_experts,
        member_numels,
        rowwise_scale_numels,
        columnwise_scale_numels,
        num_sms,
    )
    common = dict(world_size=world_size, num_local_experts=num_local_experts, num_sms=num_sms)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    i64_ptr = make_ptr(Int64, 0, cute.AddressSpace.gmem, assumed_align=8)
    pointer_args = (i64_ptr, i64_ptr, i64_ptr, i64_ptr, i64_ptr, i64_ptr, i32_ptr, i32_ptr)

    def orientation_kernel(fc1_scale_numel: int, fc2_scale_numel: int):
        return _ReplicaWeightPushKernel(
            **common,
            fc1_member_bytes=fc1_member_numel,
            fc2_member_bytes=fc2_member_numel,
            fc1_scale_bytes=fc1_scale_numel,
            fc2_scale_bytes=fc2_scale_numel,
        )

    kernels = (
        orientation_kernel(fc1_rowwise_scale_numel, fc2_rowwise_scale_numel),
        orientation_kernel(fc1_columnwise_scale_numel, fc2_columnwise_scale_numel),
    )
    stream = cuda.CUstream(0)
    with torch.cuda.device(device_index):
        return tuple(cute.compile(kernel, *pointer_args, Int32(0), stream) for kernel in kernels)


def _runtime_ptr(dtype, tensor_or_address, *, assumed_align: int = 16):
    """Create a CuTe pointer from a tensor or a PyTorch symm-mem raw address."""
    address = (
        tensor_or_address.data_ptr()
        if isinstance(tensor_or_address, torch.Tensor)
        else int(tensor_or_address)
    )
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=assumed_align)


def _as_pointer_table(
    tensor: torch.Tensor, num_local_experts: int, *, dtype: torch.dtype
) -> torch.Tensor:
    """Return a stable device table containing one data pointer per local expert.

    The public kernel helpers historically accepted one contiguous ``[expert, ...]``
    tensor.  Replica bridges can now pass an ``int64`` pointer table instead, which
    also represents TE's independently allocated ``weight0..weightN`` parameters.
    """
    if tensor.dtype == torch.int64:
        if (
            tensor.device.type != "cuda"
            or tensor.ndim != 1
            or tensor.numel() != num_local_experts
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Replica CuTeDSL pointer tables must be contiguous CUDA int64 tensors "
                f"with {num_local_experts} entries."
            )
        return tensor
    if (
        tensor.device.type != "cuda"
        or tensor.dtype != dtype
        or tensor.ndim < 2
        or tensor.shape[0] != num_local_experts
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "Replica CuTeDSL sources and native grads must be contiguous CUDA tensors "
            f"with shape [{num_local_experts}, ...] and dtype {dtype}."
        )
    return torch.tensor(
        [tensor[index].data_ptr() for index in range(num_local_experts)],
        dtype=torch.int64,
        device=tensor.device,
    )


def launch_replica_weight_prefetch(
    *,
    sources: tuple[torch.Tensor, torch.Tensor],
    arena: torch.Tensor,
    peer_bases: torch.Tensor,
    signal_bases: torch.Tensor,
    experts_to_copy: torch.Tensor,
    grid_barrier: torch.Tensor,
    rank: int,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
    scale_sources: tuple[torch.Tensor, torch.Tensor] | None = None,
    rowwise_scale_numels: tuple[int, int] | None = None,
    columnwise_scale_numels: tuple[int, int] | None = None,
    orientation: str | None = None,
) -> None:
    """Launch a BF16 or MXFP8 owner-push into destination virtual slots."""
    device_index = arena.device.index
    if device_index is None:
        raise ValueError("Replica CuTeDSL arena must be a CUDA tensor.")
    if scale_sources is None:
        if any(
            value is not None
            for value in (rowwise_scale_numels, columnwise_scale_numels, orientation)
        ):
            raise ValueError("BF16 prefetch does not accept MXFP8 scale metadata.")
        if arena.dtype != torch.bfloat16:
            raise ValueError(
                f"Replica CuTeDSL BF16 arena must use torch.bfloat16, got {arena.dtype}."
            )
        kernel = _get_compiled_bf16_weight_push(
            world_size, num_local_experts, member_numels[0], member_numels[1], num_sms, device_index
        )
        source_dtype = torch.bfloat16
    else:
        if rowwise_scale_numels is None or columnwise_scale_numels is None:
            raise ValueError("MXFP8 prefetch requires rowwise and columnwise scale shapes.")
        if orientation not in ("rowwise", "columnwise"):
            raise ValueError(
                "Replica CuTeDSL MXFP8 orientation must be 'rowwise' or 'columnwise', "
                f"got {orientation!r}."
            )
        if arena.dtype != torch.uint8:
            raise ValueError(
                f"Replica CuTeDSL MXFP8 arena must use torch.uint8, got {arena.dtype}."
            )
        kernel = _get_compiled_mxfp8_weight_kernels(
            world_size,
            num_local_experts,
            member_numels[0],
            member_numels[1],
            rowwise_scale_numels[0],
            rowwise_scale_numels[1],
            columnwise_scale_numels[0],
            columnwise_scale_numels[1],
            num_sms,
            device_index,
        )[0 if orientation == "rowwise" else 1]
        source_dtype = torch.uint8
    stream = cuda.CUstream(torch.cuda.current_stream(arena.device).cuda_stream)
    source_bases = tuple(
        _as_pointer_table(source, num_local_experts, dtype=source_dtype) for source in sources
    )
    scale_bases = source_bases
    if scale_sources is not None:
        scale_bases = tuple(
            _as_pointer_table(source, num_local_experts, dtype=torch.uint8)
            for source in scale_sources
        )
    kernel(
        _runtime_ptr(Int64, source_bases[0], assumed_align=8),
        _runtime_ptr(Int64, source_bases[1], assumed_align=8),
        _runtime_ptr(Int64, scale_bases[0], assumed_align=8),
        _runtime_ptr(Int64, scale_bases[1], assumed_align=8),
        _runtime_ptr(Int64, peer_bases, assumed_align=8),
        _runtime_ptr(Int64, signal_bases, assumed_align=8),
        _runtime_ptr(Int32, experts_to_copy),
        _runtime_ptr(Int32, grid_barrier),
        Int32(rank),
        stream,
    )


def launch_replica_grad_reduce(
    *,
    arena: torch.Tensor,
    native_grads: tuple[torch.Tensor, torch.Tensor],
    peer_bases: torch.Tensor,
    signal_bases: torch.Tensor,
    experts_to_copy: torch.Tensor,
    grid_barrier: torch.Tensor,
    rank: int,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
) -> None:
    """Accumulate virtual gradients into native wgrad staging and clear used slots."""
    device_index = arena.device.index
    if device_index is None:
        raise ValueError("Replica CuTeDSL grad arena must be a CUDA tensor.")
    if arena.dtype == torch.float32:
        pointer_type = Float32
    elif arena.dtype == torch.bfloat16:
        pointer_type = BFloat16
    else:
        raise ValueError(
            "Replica CuTeDSL grad arena must use torch.float32 or torch.bfloat16, "
            f"got {arena.dtype}."
        )
    compiled = _get_compiled_grad_reduce(
        world_size,
        num_local_experts,
        member_numels[0],
        member_numels[1],
        num_sms,
        device_index,
        arena.dtype,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(arena.device).cuda_stream)
    native_grad_bases = tuple(
        _as_pointer_table(native_grad, num_local_experts, dtype=arena.dtype)
        for native_grad in native_grads
    )
    compiled(
        _runtime_ptr(pointer_type, arena),
        _runtime_ptr(Int64, native_grad_bases[0], assumed_align=8),
        _runtime_ptr(Int64, native_grad_bases[1], assumed_align=8),
        _runtime_ptr(Int64, peer_bases, assumed_align=8),
        _runtime_ptr(Int64, signal_bases, assumed_align=8),
        _runtime_ptr(Int32, experts_to_copy),
        _runtime_ptr(Int32, grid_barrier),
        Int32(rank),
        stream,
    )
