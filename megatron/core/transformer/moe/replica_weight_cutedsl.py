# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CuTeDSL kernels for direct intra-node replica weight movement.

Only virtual weights and gradients occupy PyTorch native symmetric memory.
Native weights remain in DDP-owned storage: each owner pushes them directly
into destination virtual slots, and replica gradients are accumulated directly
into the owner's existing ``main_grad``. No activation transport is involved.
"""

import functools
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
    return Int64(
        _inline_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", loc=loc, ip=ip)
    )


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
    return _int32_asm(
        [address], "ld.acquire.gpu.global.s32 $0, [$1];", "=r,l", loc=loc, ip=ip
    )


@dsl_user_op
def _reduce_add_release_sys(address, value, *, loc=None, ip=None) -> None:
    """Publish a system-scope barrier arrival to a peer signal pad."""
    _inline_asm(
        None,
        [address, Int32(value).ir_value(loc=loc, ip=ip)],
        "red.release.sys.global.add.s32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_sc_sys(*, loc=None, ip=None) -> None:
    """Wait for this thread's non-returning peer arrivals to become visible."""
    _inline_asm(None, [], "fence.sc.sys;", "", loc=loc, ip=ip)


@dsl_user_op
def _load_acquire_sys(address, *, loc=None, ip=None) -> Int32:
    """Acquire a peer's system-scope barrier arrival."""
    return _int32_asm(
        [address], "ld.acquire.sys.global.s32 $0, [$1];", "=r,l", loc=loc, ip=ip
    )


@dsl_user_op
def _cp_reduce_async_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
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
    if cute.arch.block_idx()[0] == 0:
        local_base = signal_ptrs[rank]
        local_signals = cute.make_ptr(
            Int32, local_base, cute.AddressSpace.gmem, assumed_align=4
        )
        local_signal_tensor = _tensor_1d(local_signals, 3)
        status = local_signal_tensor[2] & Int32(3)
        phase = status & Int32(1)
        sign = status >> Int32(1)
        if thread_idx < num_ranks:
            peer_base = signal_ptrs[thread_idx]
            peer_signals = cute.make_ptr(
                Int32, peer_base, cute.AddressSpace.gmem, assumed_align=4
            )
            delta = Int32(1)
            if sign != 0:
                delta = Int32(-1)
            _reduce_add_release_sys((peer_signals + phase).toint().ir_value(), delta)
            # ``red`` has no return value and may otherwise remain outstanding
            # after a fast rank leaves the kernel.  Every signaling thread must
            # drain its own remote arrival before thread 0 can observe local
            # completion and release the CTA.
            _fence_sc_sys()
        cute.arch.sync_threads()
        if thread_idx == 0:
            _atomic_add_release_gpu(
                (local_signals + Int32(2)).toint().ir_value(), Int32(1)
            )
            target = Int32(num_ranks)
            if sign != 0:
                target = Int32(0)
            complete = cutlass.Boolean(False)
            start = _clock64()
            while not complete:
                current = _load_acquire_sys((local_signals + phase).toint().ir_value())
                complete = current == target
                if (_clock64() - start) >= Int64(_BARRIER_TIMEOUT_CYCLES):
                    cute.printf(
                        "Replica CuTeDSL cross-rank barrier timed out: "
                        "rank=%d phase=%d sign=%d signal=%d target=%d\n",
                        rank,
                        phase,
                        sign,
                        current,
                        target,
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
    """Issue one G2S chunk and arrive on its pipeline barrier."""
    for bulk in cutlass.range_constexpr(bulks_per_chunk):
        source_bulk = _tensor_1d(source + bulk * bulk_elements, bulk_elements)
        destination_bulk = _tensor_1d(destination + bulk * bulk_elements, bulk_elements)
        with cute.arch.elect_one():
            cute.copy(copy_atom, source_bulk, destination_bulk, mbar_ptr=barrier)


@cute.jit
def _bulk_store_copy(
    copy_atom: cute.CopyAtom,
    source,
    destination,
    bulk_elements: cutlass.Constexpr[int],
    bulks_per_chunk: cutlass.Constexpr[int],
):
    """Issue one elected thread's S2G chunk."""
    for bulk in cutlass.range_constexpr(bulks_per_chunk):
        source_bulk = _tensor_1d(source + bulk * bulk_elements, bulk_elements)
        destination_bulk = _tensor_1d(destination + bulk * bulk_elements, bulk_elements)
        cute.copy(copy_atom, source_bulk, destination_bulk)


class _ReplicaBulkKernel:
    """Configuration shared by the warp-specialized transport kernels."""

    STAGES = 3
    NUM_THREADS = 64

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


class _ReplicaWeightPushKernel(_ReplicaBulkKernel):
    """Push owner-local native weights into destination virtual slots."""

    STAGES = 6
    BULK_ELEMENTS = 8192
    BULKS_PER_CHUNK = 2
    CHUNK_ELEMENTS = BULK_ELEMENTS * BULKS_PER_CHUNK

    def _smem_bytes(self) -> int:
        stages = self.STAGES * self.CHUNK_ELEMENTS * 2
        barriers = self.STAGES * 2 * 8
        plan = (3 * self.world_size * self.num_local_experts + 1) * 4
        return stages + barriers + plan + 256

    @cute.jit
    def __call__(
        self,
        fc1_source_bases_ptr: cute.Pointer,
        fc2_source_bases_ptr: cute.Pointer,
        peer_base_ptr: cute.Pointer,
        signal_base_ptr: cute.Pointer,
        experts_ptr: cute.Pointer,
        grid_barrier_ptr: cute.Pointer,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        fc1_source_bases = _tensor_1d(fc1_source_bases_ptr, self.num_local_experts)
        fc2_source_bases = _tensor_1d(fc2_source_bases_ptr, self.num_local_experts)
        peer_bases = _tensor_1d(peer_base_ptr, self.world_size)
        signal_bases = _tensor_1d(signal_base_ptr, self.world_size)
        experts = _tensor_1d(experts_ptr, self.world_size * self.num_local_experts)
        self.kernel(
            fc1_source_bases,
            fc2_source_bases,
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
        peer_bases: cute.Tensor,
        signal_bases: cute.Tensor,
        experts: cute.Tensor,
        grid_barrier,
        rank: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        stages = cutlass.const_expr(self.STAGES)
        chunk_bytes = cutlass.const_expr(self.CHUNK_ELEMENTS * 2)
        chunks_per_replica = cutlass.const_expr(
            self.fc1_member_chunks + self.fc2_member_chunks
        )
        smem = utils.SmemAllocator()
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)
        owner_experts = smem.allocate_tensor(
            Int32,
            cute.make_layout((self.world_size * self.num_local_experts,)),
            byte_alignment=16,
        )
        destinations = smem.allocate_tensor(
            Int32,
            cute.make_layout((self.world_size * self.num_local_experts,)),
            byte_alignment=16,
        )
        destination_slots = smem.allocate_tensor(
            Int32,
            cute.make_layout((self.world_size * self.num_local_experts,)),
            byte_alignment=16,
        )
        active_count = smem.allocate_tensor(
            Int32, cute.make_layout((1,)), byte_alignment=4
        )
        stage_smem = smem.allocate_tensor(
            BFloat16,
            cute.make_ordered_layout((self.CHUNK_ELEMENTS, stages), order=(0, 1)),
            byte_alignment=128,
        )

        if tid == 0:
            count = Int32(0)
            for destination in cutlass.range_constexpr(self.world_size):
                for slot in cutlass.range_constexpr(self.num_local_experts):
                    expert = experts[destination * self.num_local_experts + slot]
                    owner_expert = expert - rank * self.num_local_experts
                    if owner_expert >= Int32(0) and owner_expert < Int32(
                        self.num_local_experts
                    ):
                        owner_experts[count] = owner_expert
                        destinations[count] = Int32(destination)
                        destination_slots[count] = Int32(slot)
                        count += Int32(1)
            active_count[0] = count
        cute.arch.sync_threads()

        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar,
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=chunk_bytes,
        )
        load_atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), BFloat16)
        store_atom = cute.make_copy_atom(cpasync.CopyBulkS2GOp(), BFloat16)
        remote_work = active_count[0] * chunks_per_replica
        if warp == 0:
            load_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages
            )
            for work in cutlass.range(block, remote_work, self.num_sms, unroll=1):
                # Interleave owner-local replicas across blocks before advancing
                # their member chunks. In an all-peers plan this avoids having
                # every block (and every owner rank) push into one destination
                # at a time, which otherwise creates a transient 3-to-1 hotspot.
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
                source_offset = Int64(projection_chunk * self.CHUNK_ELEMENTS)
                load_pipe.producer_acquire(load_state)
                stage = stage_smem[(None, load_state.index)]
                _bulk_load_copy(
                    load_atom,
                    source + source_offset,
                    stage.iterator,
                    load_pipe.producer_get_barrier(load_state),
                    cutlass.const_expr(self.BULK_ELEMENTS),
                    cutlass.const_expr(self.BULKS_PER_CHUNK),
                )
                load_pipe.producer_commit(load_state)
                load_state.advance()
        elif warp == 1:
            consume_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages
            )
            for work in cutlass.range(block, remote_work, self.num_sms, unroll=1):
                member_chunk = work // active_count[0]
                active = work - member_chunk * active_count[0]
                destination = destinations[active]
                slot = destination_slots[active]
                member_numel = cutlass.const_expr(self.fc1_member_numel)
                projection_base = Int64(0)
                projection_chunk = member_chunk
                if member_chunk >= self.fc1_member_chunks:
                    member_numel = cutlass.const_expr(self.fc2_member_numel)
                    projection_base = Int64(
                        self.num_local_experts * self.fc1_member_numel
                    )
                    projection_chunk = member_chunk - self.fc1_member_chunks
                destination_offset = (
                    projection_base
                    + Int64(slot) * member_numel
                    + Int64(projection_chunk * self.CHUNK_ELEMENTS)
                )
                peer = cute.make_ptr(
                    BFloat16,
                    peer_bases[destination],
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                load_pipe.consumer_wait(consume_state)
                stage = stage_smem[(None, consume_state.index)]
                with cute.arch.elect_one():
                    _bulk_store_copy(
                        store_atom,
                        stage.iterator,
                        peer + destination_offset,
                        cutlass.const_expr(self.BULK_ELEMENTS),
                        cutlass.const_expr(self.BULKS_PER_CHUNK),
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                load_pipe.consumer_release(consume_state)
                consume_state.advance()

        _cross_rank_barrier(
            signal_bases,
            grid_barrier,
            rank,
            cutlass.const_expr(self.world_size),
            Int32(self.num_sms),
            tid,
        )


class _ReplicaGradReduceKernel(_ReplicaBulkKernel):
    """Accumulate remote virtual gradients, then clear every used local slot."""

    BULK_ELEMENTS = 4096
    BULKS_PER_CHUNK = 4
    CHUNK_ELEMENTS = BULK_ELEMENTS * BULKS_PER_CHUNK

    def _smem_bytes(self) -> int:
        stages = self.STAGES * self.CHUNK_ELEMENTS * 4
        zero_source = self.BULK_ELEMENTS * 4
        barriers = self.STAGES * 2 * 8
        plan = (self.num_local_experts * (self.world_size + 1) + 1) * 4
        return stages + zero_source + barriers + plan + 256

    @cute.jit
    def __call__(
        self,
        arena_ptr: cute.Pointer,
        fc1_main_grad_bases_ptr: cute.Pointer,
        fc2_main_grad_bases_ptr: cute.Pointer,
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
        fc1_main_grad_bases = _tensor_1d(
            fc1_main_grad_bases_ptr, self.num_local_experts
        )
        fc2_main_grad_bases = _tensor_1d(
            fc2_main_grad_bases_ptr, self.num_local_experts
        )
        peer_bases = _tensor_1d(peer_base_ptr, self.world_size)
        signal_bases = _tensor_1d(signal_base_ptr, self.world_size)
        experts = _tensor_1d(experts_ptr, self.world_size * self.num_local_experts)
        self.kernel(
            arena,
            fc1_main_grad_bases,
            fc2_main_grad_bases,
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
        arena: cute.Tensor,
        fc1_main_grad_bases: cute.Tensor,
        fc2_main_grad_bases: cute.Tensor,
        peer_bases: cute.Tensor,
        signal_bases: cute.Tensor,
        experts: cute.Tensor,
        grid_barrier,
        rank: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        stages = cutlass.const_expr(self.STAGES)
        chunk_bytes = cutlass.const_expr(self.CHUNK_ELEMENTS * 4)
        virtual_fc1_numel = cutlass.const_expr(
            self.num_local_experts * self.fc1_member_numel
        )
        fc1_chunks = cutlass.const_expr(self.num_local_experts * self.fc1_member_chunks)
        fc2_chunks = cutlass.const_expr(self.num_local_experts * self.fc2_member_chunks)

        smem = utils.SmemAllocator()
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)
        stage_smem = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout((self.CHUNK_ELEMENTS, stages), order=(0, 1)),
            byte_alignment=128,
        )
        zero_smem = smem.allocate_tensor(
            Float32, cute.make_layout((self.BULK_ELEMENTS,)), byte_alignment=128
        )
        matches = smem.allocate_tensor(
            Int32,
            cute.make_layout((self.num_local_experts * self.world_size,)),
            byte_alignment=16,
        )
        active_slots = smem.allocate_tensor(
            Int32, cute.make_layout((self.num_local_experts,)), byte_alignment=16
        )
        active_count = smem.allocate_tensor(
            Int32, cute.make_layout((1,)), byte_alignment=4
        )

        for index in cutlass.range(
            tid, self.num_local_experts * self.world_size, self.NUM_THREADS
        ):
            matches[index] = Int32(-1)
        cute.arch.sync_threads()
        for index in cutlass.range(
            tid, self.num_local_experts * self.world_size, self.NUM_THREADS
        ):
            expert = experts[index]
            owner_expert = expert - rank * self.num_local_experts
            if owner_expert >= Int32(0) and owner_expert < Int32(
                self.num_local_experts
            ):
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
        for element in cutlass.range(tid, self.BULK_ELEMENTS, self.NUM_THREADS):
            zero_smem[element] = 0.0
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
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=chunk_bytes,
        )
        load_atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
        store_atom = cute.make_copy_atom(cpasync.CopyBulkS2GOp(), Float32)
        load_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages
        )
        consume_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages
        )
        total_work = cutlass.const_expr(fc1_chunks + fc2_chunks)
        for work in cutlass.range(block, total_work, self.num_sms, unroll=1):
            is_fc2 = work >= fc1_chunks
            projection_work = work
            member_chunks = cutlass.const_expr(self.fc1_member_chunks)
            member_numel = cutlass.const_expr(self.fc1_member_numel)
            virtual_projection_base = Int64(0)
            if is_fc2:
                projection_work = work - fc1_chunks
                member_chunks = cutlass.const_expr(self.fc2_member_chunks)
                member_numel = cutlass.const_expr(self.fc2_member_numel)
                virtual_projection_base = Int64(virtual_fc1_numel)
            local_expert = projection_work // member_chunks
            member_chunk = projection_work - local_expert * member_chunks
            member_offset = Int64(member_chunk * self.CHUNK_ELEMENTS)

            if warp == 0:
                for destination in cutlass.range_constexpr(self.world_size):
                    slot = matches[local_expert * self.world_size + destination]
                    if slot >= Int32(0):
                        peer = cute.make_ptr(
                            Float32,
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
            elif warp == 1:
                main_destination = cute.make_ptr(
                    Float32,
                    fc1_main_grad_bases[local_expert],
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                if is_fc2:
                    main_destination = cute.make_ptr(
                        Float32,
                        fc2_main_grad_bases[local_expert],
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                main_destination = _tensor_1d(
                    main_destination + member_offset, self.CHUNK_ELEMENTS
                )
                source_count = Int32(0)
                for destination in cutlass.range_constexpr(self.world_size):
                    if matches[local_expert * self.world_size + destination] >= Int32(
                        0
                    ):
                        source_count += Int32(1)
                for _source_index in cutlass.range(source_count, unroll=1):
                    load_pipe.consumer_wait(consume_state)
                    with cute.arch.elect_one():
                        for bulk in cutlass.range_constexpr(self.BULKS_PER_CHUNK):
                            stage = stage_smem[(None, consume_state.index)]
                            _cp_reduce_async_bulk_add_f32(
                                stage.iterator + bulk * self.BULK_ELEMENTS,
                                main_destination.iterator + bulk * self.BULK_ELEMENTS,
                                self.BULK_ELEMENTS * 4,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    load_pipe.consumer_release(consume_state)
                    consume_state.advance()

        if warp == 1:
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_wait_group(0, read=True)

        _cross_rank_barrier(
            signal_bases,
            grid_barrier,
            rank,
            cutlass.const_expr(self.world_size),
            Int32(self.num_sms),
            tid,
        )

        # Owners have now acquired all remote virtual gradients. Clear exactly
        # the locally used slots; planner slots need not form a dense prefix.
        bulks_per_fc1 = cutlass.const_expr(self.fc1_member_numel // self.BULK_ELEMENTS)
        bulks_per_fc2 = cutlass.const_expr(self.fc2_member_numel // self.BULK_ELEMENTS)
        bulks_per_slot = cutlass.const_expr(bulks_per_fc1 + bulks_per_fc2)
        clear_work = active_count[0] * bulks_per_slot
        if warp == 0:
            # Amortize the bulk-group drain across the hardware's eight
            # outstanding groups while preserving sparse-slot addressing.
            for base in cutlass.range(
                block, clear_work, self.num_sms * 8, unroll=1
            ):
                with cute.arch.elect_one():
                    for batch in cutlass.range_constexpr(8):
                        work = base + batch * self.num_sms
                        if work < clear_work:
                            active = work // bulks_per_slot
                            slot_bulk = work - active * bulks_per_slot
                            slot = active_slots[active]
                            destination_offset = (
                                Int64(slot) * self.fc1_member_numel
                                + Int64(slot_bulk * self.BULK_ELEMENTS)
                            )
                            if slot_bulk >= bulks_per_fc1:
                                destination_offset = (
                                    Int64(virtual_fc1_numel)
                                    + Int64(slot) * self.fc2_member_numel
                                    + Int64(
                                        (slot_bulk - bulks_per_fc1)
                                        * self.BULK_ELEMENTS
                                    )
                                )
                            clear_destination = _tensor_1d(
                                arena.iterator + destination_offset,
                                self.BULK_ELEMENTS,
                            )
                            cute.copy(store_atom, zero_smem, clear_destination)
                            cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)


def _validate_compile_shape(
    world_size: int,
    num_local_experts: int,
    fc1_member_numel: int,
    fc2_member_numel: int,
    num_sms: int,
) -> None:
    if not HAVE_CUTEDSL:
        raise ImportError(
            "Replica CuTeDSL weight transfer requires nvidia-cutlass-dsl."
        )
    if world_size <= 0 or num_local_experts <= 0 or num_sms <= 0:
        raise ValueError("Replica CuTeDSL launch dimensions must be positive.")
    if num_sms > MAX_REPLICA_WEIGHT_SMS:
        raise ValueError(
            "Replica CuTeDSL weight kernels are limited to "
            f"{MAX_REPLICA_WEIGHT_SMS} SMs, got {num_sms}."
        )
    max_ranks = min(
        _ReplicaWeightPushKernel.NUM_THREADS,
        _ReplicaGradReduceKernel.NUM_THREADS,
    )
    if world_size > max_ranks:
        raise ValueError(
            "Replica CuTeDSL supports at most "
            f"{max_ranks} EP ranks, got {world_size}."
        )
    tile_elements = _ReplicaGradReduceKernel.CHUNK_ELEMENTS
    assert tile_elements == _ReplicaWeightPushKernel.CHUNK_ELEMENTS
    if fc1_member_numel % tile_elements or fc2_member_numel % tile_elements:
        raise ValueError(
            "Replica CuTeDSL projections must contain a multiple of "
            f"{tile_elements} elements "
            f"per expert, got {(fc1_member_numel, fc2_member_numel)}."
        )


@functools.lru_cache(maxsize=None)
def _get_compiled_kernels(
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
    kernel_args = dict(
        world_size=world_size,
        num_local_experts=num_local_experts,
        fc1_member_numel=fc1_member_numel,
        fc2_member_numel=fc2_member_numel,
        num_sms=num_sms,
    )
    f32_ptr = make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    i64_ptr = make_ptr(Int64, 0, cute.AddressSpace.gmem, assumed_align=8)
    specifications = (
        (
            _ReplicaWeightPushKernel(**kernel_args),
            (i64_ptr, i64_ptr, i64_ptr, i64_ptr, i32_ptr, i32_ptr),
        ),
        (
            _ReplicaGradReduceKernel(**kernel_args),
            (f32_ptr, i64_ptr, i64_ptr, i64_ptr, i64_ptr, i32_ptr, i32_ptr),
        ),
    )
    stream = cuda.CUstream(0)
    with torch.cuda.device(device_index):
        return tuple(
            cute.compile(kernel, *arguments, Int32(0), stream)
            for kernel, arguments in specifications
        )


def compile_replica_weight_kernels(
    *,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
    device_index: int,
) -> None:
    """JIT compile both fixed-shape kernels before entering the hot path."""
    _get_compiled_kernels(
        world_size,
        num_local_experts,
        member_numels[0],
        member_numels[1],
        num_sms,
        device_index,
    )


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
            "Replica CuTeDSL sources and main grads must be contiguous CUDA tensors "
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
) -> None:
    """Launch owner-push prefetch into destination virtual slots."""
    device_index = arena.device.index
    if device_index is None:
        raise ValueError("Replica CuTeDSL arena must be a CUDA tensor.")
    push, _ = _get_compiled_kernels(
        world_size,
        num_local_experts,
        member_numels[0],
        member_numels[1],
        num_sms,
        device_index,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(arena.device).cuda_stream)
    source_bases = tuple(
        _as_pointer_table(source, num_local_experts, dtype=torch.bfloat16)
        for source in sources
    )
    push(
        _runtime_ptr(Int64, source_bases[0], assumed_align=8),
        _runtime_ptr(Int64, source_bases[1], assumed_align=8),
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
    main_grads: tuple[torch.Tensor, torch.Tensor],
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
    """Accumulate virtual gradients into native main-grad and clear used slots."""
    device_index = arena.device.index
    if device_index is None:
        raise ValueError("Replica CuTeDSL grad arena must be a CUDA tensor.")
    _, compiled = _get_compiled_kernels(
        world_size,
        num_local_experts,
        member_numels[0],
        member_numels[1],
        num_sms,
        device_index,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(arena.device).cuda_stream)
    main_grad_bases = tuple(
        _as_pointer_table(main_grad, num_local_experts, dtype=torch.float32)
        for main_grad in main_grads
    )
    compiled(
        _runtime_ptr(Float32, arena),
        _runtime_ptr(Int64, main_grad_bases[0], assumed_align=8),
        _runtime_ptr(Int64, main_grad_bases[1], assumed_align=8),
        _runtime_ptr(Int64, peer_bases, assumed_align=8),
        _runtime_ptr(Int64, signal_bases, assumed_align=8),
        _runtime_ptr(Int32, experts_to_copy),
        _runtime_ptr(Int32, grid_barrier),
        Int32(rank),
        stream,
    )
