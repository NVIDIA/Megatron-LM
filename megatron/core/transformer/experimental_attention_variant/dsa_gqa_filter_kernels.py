# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CuTe DSL filter kernels for the fused DSA-over-GQA top-k selection.

A filter scores rows' indexer queries against their paged indexer keys on tensor cores and keeps
every key whose score reaches its row's threshold, without writing the scores anywhere. Each
program takes one block of consecutive rows of one request (``build_row_metadata``) and a strided
subset of the request's 128-key tiles (a tile lies inside one cache page, so it is one contiguous
``[128, Di]`` block). Kept keys take slots of the row's candidate buffer through an atomic on the
row's count, so the count ends as the row's total kept keys, past the capacity when it filled
(kept keys are a fraction of a percent of all keys, and cluster wherever the row attends).

``DecodeFilter`` uses warp MMAs and suits blocks of a few rows (decode). ``PrefillFilter`` uses
Blackwell tcgen05 MMAs into tensor memory for blocks of up to 128 rows: a TMA warp loads the
block's queries once and then key tiles, an MMA warp scores each tile into one of two tensor-memory
accumulators, and four epilogue warps compare them, thread ``t`` owning row ``t`` of the block so
its threshold, length and candidate count stay in registers.
"""

from __future__ import annotations

import math

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    HAVE_CUTE_DSL = True
except ImportError:
    HAVE_CUTE_DSL = False

import torch

# Keys per tile, and the indexer head dimension the kernels are built for.
BLOCK_KEYS = 128
INDEX_DIM = 128
# Rows per block of the decode filter, and of the prefill filter (one tcgen05 MMA tile).
DECODE_ROW_BLOCK = 16
PREFILL_ROW_BLOCK = 128


if HAVE_CUTE_DSL:

    def _k_major_smem_layout(rows, stages):
        # K-major ([rows, 128] row-major) swizzled layout, as CUTLASS's Ampere tensorop examples.
        major = 64
        swizzle_bits = min(int(math.log2(major * 16 // 128)), 3)
        atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, major), stride=(major, 1)),
        )
        shape = (rows, INDEX_DIM, stages) if stages else (rows, INDEX_DIM)
        return cute.tile_to_shape(atom, shape, (0, 1, 2) if stages else (0, 1))

    class DecodeFilter:
        """Warp-MMA filter: queries stay in registers, key tiles stream through a cp.async ring."""

        def __init__(self, page_size: int, stages: int = 2, warps: int = 4):
            self.page_size = page_size
            self.bm = DECODE_ROW_BLOCK
            self.stages = stages
            self.warps = warps
            self.threads = warps * 32

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mTable: cute.Tensor,
            mLen: cute.Tensor,
            mTau: cute.Tensor,
            mBStart: cute.Tensor,
            mBRows: cute.Tensor,
            mBReq: cute.Tensor,
            mCounts: cute.Tensor,
            mCandIdx: cute.Tensor,
            mCandVal: cute.Tensor,
            topk: cutlass.Int32,
            capacity: cutlass.Int32,
            workers: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            sA_layout = _k_major_smem_layout(self.bm, 0)
            sB_layout = _k_major_smem_layout(BLOCK_KEYS, self.stages)
            atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
                cutlass.BFloat16,
                num_bits_per_copy=128,
            )
            # 16 threads cover one 128-wide row (8 BF16 each).
            copy_threads = cute.make_layout((self.threads // 16, 16), stride=(16, 1))
            tiled_copy = cute.make_tiled_copy_tv(atom_async, copy_threads, cute.make_layout((1, 8)))
            op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
            warps_m = max(1, min(self.bm // 16, self.warps // 2))
            warps_n = self.warps // warps_m
            tiled_mma = cute.make_tiled_mma(
                op,
                cute.make_layout((warps_m, warps_n, 1)),
                permutation_mnk=(warps_m * 16, warps_n * 16, 16),
            )
            smem_bytes = (
                cute.size_in_bytes(cutlass.BFloat16, sA_layout)
                + cute.size_in_bytes(cutlass.BFloat16, sB_layout)
                + 3 * self.bm * 4
            )
            num_blocks = cute.size(mBStart, mode=[0])
            self.kernel(
                mQ,
                mK,
                mTable,
                mLen,
                mTau,
                mBStart,
                mBRows,
                mBReq,
                mCounts,
                mCandIdx,
                mCandVal,
                topk,
                capacity,
                sA_layout,
                sB_layout,
                tiled_copy,
                tiled_mma,
            ).launch(
                grid=[workers, num_blocks, 1],
                block=[self.threads, 1, 1],
                smem=smem_bytes,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mTable: cute.Tensor,
            mLen: cute.Tensor,
            mTau: cute.Tensor,
            mBStart: cute.Tensor,
            mBRows: cute.Tensor,
            mBReq: cute.Tensor,
            mCounts: cute.Tensor,
            mCandIdx: cute.Tensor,
            mCandVal: cute.Tensor,
            topk: cutlass.Int32,
            capacity: cutlass.Int32,
            sA_layout: cute.ComposedLayout,
            sB_layout: cute.ComposedLayout,
            tiled_copy: cute.TiledCopy,
            tiled_mma: cute.TiledMma,
        ):
            """Filter paged indexer candidates for the assigned query blocks."""
            tidx, _, _ = cute.arch.thread_idx()
            worker, block, _ = cute.arch.block_idx()
            workers, _, _ = cute.arch.grid_dim()
            num_rows = mBRows[block]
            row0 = mBStart[block]
            request = mBReq[block]
            # Rows of a block are consecutive tokens of one request: the last row is the longest.
            longest = cutlass.Int32(0)
            if num_rows > 0:
                longest = mLen[row0 + num_rows - 1]
            num_tiles = (longest + BLOCK_KEYS - 1) // BLOCK_KEYS
            my_tiles = cutlass.Int32(0)
            if worker < num_tiles:
                my_tiles = (num_tiles - worker + workers - 1) // workers
            # Rows with at most top-k keys take all of them and need no filtering.
            if longest > topk and my_tiles > 0:
                smem = cutlass.utils.SmemAllocator()
                sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
                sB = smem.allocate_tensor(cutlass.BFloat16, sB_layout, 16)
                sLen = smem.allocate_tensor(cutlass.Int32, cute.make_layout(self.bm), 16)
                sTau = smem.allocate_tensor(cutlass.Float32, cute.make_layout(self.bm), 16)
                # Rows that need no filtering (padding, or at most top-k keys) get an infinite
                # threshold, so no score passes it.
                if tidx < self.bm:
                    sLen[tidx] = cutlass.Int32(0)
                    sTau[tidx] = cutlass.Float32.inf
                    if tidx < num_rows:
                        row_length = mLen[row0 + tidx]
                        sLen[tidx] = row_length
                        if row_length > topk:
                            sTau[tidx] = mTau[row0 + tidx]

                thr_copy = tiled_copy.get_slice(tidx)
                # Queries: rows past the block are predicated off (zero-filled).
                gA = cute.make_tensor(
                    (mQ.iterator + cutlass.Int64(row0) * INDEX_DIM).align(16),
                    cute.make_layout((self.bm, INDEX_DIM), stride=(INDEX_DIM, 1)),
                )
                cA = cute.make_identity_tensor((self.bm, INDEX_DIM))
                tAgA = thr_copy.partition_S(gA)
                tAsA = thr_copy.partition_D(sA)
                tAcA = thr_copy.partition_S(cA)
                tAsA.fill(0)
                cute.arch.sync_threads()
                for m in cutlass.range_constexpr(cute.size(tAgA, mode=[1])):
                    if cute.elem_less(tAcA[0, m, 0][0], num_rows):
                        cute.copy(tiled_copy, tAgA[None, m, None], tAsA[None, m, None])

                # Prologue: the queries plus the first stages - 1 key tiles in flight.
                for s in cutlass.range_constexpr(self.stages - 1):
                    if s < my_tiles:
                        pro_start = (worker + s * workers) * BLOCK_KEYS
                        pro_page = mTable[request, pro_start // self.page_size]
                        pro_offset = (
                            cutlass.Int64(pro_page) * self.page_size + pro_start % self.page_size
                        ) * INDEX_DIM
                        gB = cute.make_tensor(
                            (mK.iterator + pro_offset).align(16),
                            cute.make_layout((BLOCK_KEYS, INDEX_DIM), stride=(INDEX_DIM, 1)),
                        )
                        cute.copy(
                            tiled_copy,
                            thr_copy.partition_S(gB),
                            thr_copy.partition_D(sB[None, None, s]),
                        )
                    cute.arch.cp_async_commit_group()

                thr_mma = tiled_mma.get_slice(tidx)
                tCsA = thr_mma.partition_A(sA)
                tCrA = tiled_mma.make_fragment_A(tCsA)
                tCsB0 = thr_mma.partition_B(sB[None, None, 0])
                tCrB = tiled_mma.make_fragment_B(tCsB0)
                cC = cute.make_identity_tensor((self.bm, BLOCK_KEYS))
                tCcC = thr_mma.partition_C(cC)
                acc = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((self.bm, BLOCK_KEYS)))
                atom_ld = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16
                )
                ldA = cute.make_tiled_copy_A(atom_ld, tiled_mma).get_slice(tidx)
                ldB = cute.make_tiled_copy_B(atom_ld, tiled_mma).get_slice(tidx)
                num_k = cute.size(tCrA, mode=[2])

                cute.arch.cp_async_wait_group(self.stages - 2)
                cute.arch.sync_threads()
                # The block's lowest threshold: most scores fall below it and skip the per-row
                # checks.
                tau_min = cutlass.Float32.inf
                for r in cutlass.range_constexpr(self.bm):
                    tau_min = cute.arch.fmin(tau_min, sTau[r])
                cute.copy(ldA, ldA.partition_S(sA), ldA.retile(tCrA))

                read_stage = cutlass.Int32(0)
                for i in cutlass.range(my_tiles):
                    cute.arch.cp_async_wait_group(self.stages - 2)
                    cute.arch.sync_threads()
                    sB_i = sB[None, None, read_stage]
                    cute.copy(ldB, ldB.partition_S(sB_i), ldB.retile(tCrB))
                    # Refill the stage consumed stages - 1 tiles ago.
                    nxt = i + self.stages - 1
                    write_stage = (read_stage + self.stages - 1) % self.stages
                    if nxt < my_tiles:
                        nxt_start = (worker + nxt * workers) * BLOCK_KEYS
                        nxt_page = mTable[request, nxt_start // self.page_size]
                        nxt_offset = (
                            cutlass.Int64(nxt_page) * self.page_size + nxt_start % self.page_size
                        ) * INDEX_DIM
                        gB = cute.make_tensor(
                            (mK.iterator + nxt_offset).align(16),
                            cute.make_layout((BLOCK_KEYS, INDEX_DIM), stride=(INDEX_DIM, 1)),
                        )
                        cute.copy(
                            tiled_copy,
                            thr_copy.partition_S(gB),
                            thr_copy.partition_D(sB[None, None, write_stage]),
                        )
                    cute.arch.cp_async_commit_group()
                    acc.fill(0.0)
                    for k in cutlass.range_constexpr(num_k):
                        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
                    start = (worker + i * workers) * BLOCK_KEYS
                    for e in cutlass.range_constexpr(cute.size(acc)):
                        if acc[e] >= tau_min:
                            r = tCcC[e][0]
                            position = start + tCcC[e][1]
                            if position < sLen[r]:
                                if acc[e] >= sTau[r]:
                                    row = row0 + r
                                    slot = cute.arch.atomic_add(
                                        mCounts.iterator + row, cutlass.Int32(1)
                                    )
                                    if slot < capacity:
                                        mCandIdx[row, slot] = position
                                        mCandVal[row, slot] = acc[e]
                    read_stage = (read_stage + 1) % self.stages
                cute.arch.cp_async_wait_group(0)

    # Prefill filter warp roles: four epilogue warps (TMEM lanes 0-127), then MMA, then TMA.
    _EPILOGUE_WARPS = 4
    _MMA_WARP = 4
    _TMA_WARP = 5
    _ACC_STAGES = 2

    class PrefillFilter:
        """tcgen05 filter: [128 rows x 128 keys] MMAs into tensor memory, one row per thread."""

        def __init__(self, page_size: int, stages: int = 4):
            if page_size % BLOCK_KEYS:
                raise ValueError(f"Page size {page_size} is not a multiple of {BLOCK_KEYS}.")
            self.tiles_per_page = page_size // BLOCK_KEYS
            self.stages = stages

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mTable: cute.Tensor,
            mLen: cute.Tensor,
            mTau: cute.Tensor,
            mBStart: cute.Tensor,
            mBRows: cute.Tensor,
            mBReq: cute.Tensor,
            mCounts: cute.Tensor,
            mCandIdx: cute.Tensor,
            mCandVal: cute.Tensor,
            topk: cutlass.Int32,
            capacity: cutlass.Int32,
            workers: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                cutlass.BFloat16,
                OperandMajorMode.K,
                OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (PREFILL_ROW_BLOCK, BLOCK_KEYS),
            )
            mma_tiler = (PREFILL_ROW_BLOCK, BLOCK_KEYS, INDEX_DIM)
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
            )
            a_smem_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, mma_tiler, cutlass.BFloat16, 1
            )
            b_smem_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, mma_tiler, cutlass.BFloat16, self.stages
            )
            a_smem = cute.slice_(a_smem_staged, (None, None, None, 0))
            b_smem = cute.slice_(b_smem_staged, (None, None, None, 0))
            tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
                sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id),
                mQ,
                a_smem,
                mma_tiler,
                tiled_mma,
                cluster_layout_vmnk.shape,
            )
            tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
                sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id),
                mK,
                b_smem,
                mma_tiler,
                tiled_mma,
                cluster_layout_vmnk.shape,
            )
            a_bytes = cute.size_in_bytes(cutlass.BFloat16, a_smem)
            b_bytes = cute.size_in_bytes(cutlass.BFloat16, b_smem)
            num_blocks = cute.size(mBStart, mode=[0])
            self.kernel(
                tiled_mma,
                tma_atom_a,
                tma_a,
                tma_atom_b,
                tma_b,
                cluster_layout_vmnk,
                a_smem_staged,
                b_smem_staged,
                a_bytes,
                b_bytes,
                mTable,
                mLen,
                mTau,
                mBStart,
                mBRows,
                mBReq,
                mCounts,
                mCandIdx,
                mCandVal,
                topk,
                capacity,
            ).launch(
                grid=[workers, num_blocks, 1],
                block=[32 * (_TMA_WARP + 1), 1, 1],
                cluster=(1, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            tiled_mma: cute.TiledMma,
            tma_atom_a: cute.CopyAtom,
            mA: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB: cute.Tensor,
            cluster_layout_vmnk: cute.Layout,
            a_smem_staged: cute.ComposedLayout,
            b_smem_staged: cute.ComposedLayout,
            a_bytes: cutlass.Constexpr,
            b_bytes: cutlass.Constexpr,
            mTable: cute.Tensor,
            mLen: cute.Tensor,
            mTau: cute.Tensor,
            mBStart: cute.Tensor,
            mBRows: cute.Tensor,
            mBReq: cute.Tensor,
            mCounts: cute.Tensor,
            mCandIdx: cute.Tensor,
            mCandVal: cute.Tensor,
            topk: cutlass.Int32,
            capacity: cutlass.Int32,
        ):
            """Filter paged indexer candidates for the assigned query blocks."""
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            worker, block, _ = cute.arch.block_idx()
            workers, _, _ = cute.arch.grid_dim()
            num_rows = mBRows[block]
            row0 = mBStart[block]
            request = mBReq[block]
            # Rows of a block are consecutive tokens of one request: the last row is the longest.
            longest = cutlass.Int32(0)
            if num_rows > 0:
                longest = mLen[row0 + num_rows - 1]
            num_tiles = (longest + BLOCK_KEYS - 1) // BLOCK_KEYS
            my_tiles = cutlass.Int32(0)
            if worker < num_tiles:
                my_tiles = (num_tiles - worker + workers - 1) // workers
            # Rows with at most top-k keys take all of them and need no filtering.
            if longest > topk and my_tiles > 0:
                if warp_idx == _TMA_WARP:
                    cpasync.prefetch_descriptor(tma_atom_a)
                    cpasync.prefetch_descriptor(tma_atom_b)

                @cute.struct
                class SharedStorage:
                    """Shared-memory barriers and TMEM ownership state for the selector."""

                    a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
                    b_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
                    acc_mbar: cute.struct.MemRange[cutlass.Int64, _ACC_STAGES * 2]
                    tmem_dealloc_mbar: cutlass.Int64
                    tmem_holding_buf: cutlass.Int32

                smem = utils.SmemAllocator()
                storage = smem.allocate(SharedStorage)
                a_producer, a_consumer = pipeline.PipelineTmaUmma.create(
                    barrier_storage=storage.a_mbar.data_ptr(),
                    num_stages=1,
                    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                    tx_count=a_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                ).make_participants()
                b_producer, b_consumer = pipeline.PipelineTmaUmma.create(
                    barrier_storage=storage.b_mbar.data_ptr(),
                    num_stages=self.stages,
                    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                    tx_count=b_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                ).make_participants()
                acc_pipeline = pipeline.PipelineUmmaAsync.create(
                    barrier_storage=storage.acc_mbar.data_ptr(),
                    num_stages=_ACC_STAGES,
                    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                    consumer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, 32 * _EPILOGUE_WARPS
                    ),
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
                tmem_alloc_barrier = pipeline.NamedBarrier(
                    barrier_id=1, num_threads=32 * (_EPILOGUE_WARPS + 1)
                )
                tmem_dealloc_barrier = pipeline.NamedBarrier(
                    barrier_id=2, num_threads=32 * _EPILOGUE_WARPS
                )
                tmem = utils.TmemAllocator(
                    storage.tmem_holding_buf.ptr,
                    barrier_for_retrieve=tmem_alloc_barrier,
                    allocator_warp_id=0,
                    is_two_cta=False,
                    two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
                )
                pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

                sA = smem.allocate_tensor(
                    element_type=cutlass.BFloat16,
                    layout=a_smem_staged.outer,
                    byte_alignment=128,
                    swizzle=a_smem_staged.inner,
                )
                sB = smem.allocate_tensor(
                    element_type=cutlass.BFloat16,
                    layout=b_smem_staged.outer,
                    byte_alignment=128,
                    swizzle=b_smem_staged.inner,
                )
                # Epilogue staging: 32 scores per thread (rows of 36 stay 16-byte aligned).
                sStage = smem.allocate_tensor(
                    cutlass.Float32, cute.make_layout((PREFILL_ROW_BLOCK, 36), stride=(36, 1)), 16
                )

                # Queries: the block's rows start at row0 (TMA zero-fills rows past the tensor;
                # rows past the block are scored but never kept).
                tiler = (PREFILL_ROW_BLOCK, BLOCK_KEYS, INDEX_DIM)
                gA = cute.local_tile(
                    cute.domain_offset((row0, 0, 0), mA),
                    cute.slice_(tiler, (None, 0, None)),
                    (None, None, None),
                )
                gB = cute.local_tile(mB, cute.slice_(tiler, (0, None, None)), (None, None, None))
                thr_mma = tiled_mma.get_slice(0)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(thr_mma.partition_A(gA), 0, 3),
                )
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(thr_mma.partition_B(gB), 0, 3),
                )
                tCrA = tiled_mma.make_fragment_A(sA)
                tCrB = tiled_mma.make_fragment_B(sB)
                acc_shape = tiled_mma.partition_shape_C((PREFILL_ROW_BLOCK, BLOCK_KEYS))
                tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, _ACC_STAGES))

                pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

                if warp_idx == _TMA_WARP:
                    handle = a_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, 0, 0, 0)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
                    for i in cutlass.range(my_tiles, unroll=1):
                        tile = worker + i * workers
                        page = mTable[request, tile // self.tiles_per_page]
                        slot_tile = page * self.tiles_per_page + tile % self.tiles_per_page
                        handle = b_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, slot_tile, 0, 0)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                    b_producer.tail()
                    a_producer.tail()

                if warp_idx == _MMA_WARP:
                    tmem.wait_for_alloc()
                    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
                    tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
                    acc_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, _ACC_STAGES
                    )
                    a_handle = a_consumer.wait_and_advance()
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for i in cutlass.range(my_tiles, unroll=1):
                        acc_pipeline.producer_acquire(acc_state)
                        tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                        handle = b_consumer.wait_and_advance()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kblk in cutlass.range(num_kblocks, unroll_full=True):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[(None, None, kblk, 0)],
                                tCrB[(None, None, kblk, handle.index)],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        handle.release()
                        acc_pipeline.producer_commit(acc_state)
                        acc_state.advance()
                    a_handle.release()
                    acc_pipeline.producer_tail(acc_state)

                if warp_idx < _EPILOGUE_WARPS:
                    tmem.allocate(_ACC_STAGES * BLOCK_KEYS)
                    tmem.wait_for_alloc()
                    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
                    tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
                    load_atom = cute.make_copy_atom(
                        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
                    )
                    thr_load = tcgen05.make_tmem_copy(
                        load_atom, tCtAcc_base[((None, None), 0, 0, 0)]
                    ).get_slice(tidx)
                    tCoord = thr_load.partition_D(
                        cute.make_identity_tensor((PREFILL_ROW_BLOCK, BLOCK_KEYS))
                    )
                    scores = cute.make_rmem_tensor(tCoord.shape, cutlass.Float32)
                    chunks = cute.size(tCoord, mode=[2])
                    # This thread's row: tensor-memory lane = row of the block.
                    row = tCoord[0][0]
                    length = cutlass.Int32(0)
                    tau = cutlass.Float32.inf
                    if row < num_rows:
                        length = mLen[row0 + row]
                        if length > topk:
                            tau = mTau[row0 + row]
                    acc_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, _ACC_STAGES
                    )
                    for i in cutlass.range(my_tiles, unroll=1):
                        acc_pipeline.consumer_wait(acc_state)
                        # The row's scores of the tile into registers; wait for the loads, then
                        # hand the accumulator back to the MMA warp.
                        cute.copy(
                            thr_load,
                            thr_load.partition_S(
                                tCtAcc_base[((None, None), 0, 0, acc_state.index)]
                            ),
                            scores,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        acc_pipeline.consumer_release(acc_state)
                        start = (worker + i * workers) * BLOCK_KEYS
                        for c in cutlass.range_constexpr(chunks):
                            # Chunk c holds columns c * 32 .. c * 32 + 31; bit e of the mask is
                            # whether column c * 32 + e is kept.
                            chunk = scores[(None, None, c)]
                            mask = cutlass.Int32(0)
                            for e in cutlass.range_constexpr(32):
                                mask = mask | (cutlass.Int32(chunk[e] >= tau) << e)
                            valid = length - (start + c * 32)
                            if valid < 32:
                                mask = mask & ((cutlass.Int32(1) << cutlass.max(valid, 0)) - 1)
                            # Kept keys are rare: reserve their slots with one atomic, stage the
                            # chunk in shared memory, and visit only the set bits.
                            if mask != 0:
                                slot = cute.arch.atomic_add(
                                    mCounts.iterator + (row0 + row), cute.arch.popc(mask)
                                )
                                for e in cutlass.range_constexpr(32):
                                    sStage[row, e] = chunk[e]
                                while mask != 0:
                                    low = mask & (-mask)
                                    e = cute.arch.popc(low - 1)
                                    if slot < capacity:
                                        mCandIdx[row0 + row, slot] = start + c * 32 + e
                                        mCandVal[row0 + row, slot] = sStage[row, e]
                                    slot += 1
                                    mask = mask ^ low
                        acc_state.advance()
                    tmem_dealloc_barrier.arrive_and_wait()
                    tmem.relinquish_alloc_permit()
                    tmem.free(tmem_ptr)


_COMPILED = {}


def _vector(tensor, alignment):
    return from_dlpack(tensor, assumed_align=alignment, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=tensor.dim() - 1
    )


def _k_major(tensor):
    # [rows, Di] as the (M or N, K, L=1) tensor the TMA atoms take, K-major.
    return from_dlpack(
        tensor.unsqueeze(-1), assumed_align=16, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=1)


def filter_candidates(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_lengths: torch.Tensor,
    thresholds: torch.Tensor,
    blocks: tuple,
    counts: torch.Tensor,
    cand_idx: torch.Tensor,
    cand_val: torch.Tensor,
    *,
    topk: int,
    workers: int,
    prefill: bool,
) -> None:
    """Append each row's keys scoring at least its threshold to its candidate buffer.

    Args:
        query: BF16 indexer queries ``[rows,Di]``, contiguous.
        key_cache: Paged BF16 indexer keys ``[pages,page_size,Di]``, contiguous.
        block_table: Int32 ``[requests,max_pages]``.
        row_lengths: Int32 ``[rows]`` causal key count of each row.
        thresholds: FP32 ``[rows]``; rows with at most ``topk`` keys are skipped.
        blocks: ``(block_starts, block_rows, block_requests)`` with at most
            ``PREFILL_ROW_BLOCK`` (prefill) or ``DECODE_ROW_BLOCK`` rows per block.
        counts: Int32 ``[rows]``, zeroed; receives each row's kept-key count, which exceeds the
            capacity when the buffer filled (the keys past it are dropped).
        cand_idx, cand_val: ``[rows,capacity]`` candidate positions and scores; a row's first
            ``min(count, capacity)`` slots are written.
        topk: Keys each row selects.
        workers: Programs per block.
        prefill: Use the tcgen05 prefill filter rather than the decode filter.
    """
    capacity = cand_idx.size(1)
    page_size = key_cache.size(1)
    keys = key_cache.view(-1, key_cache.size(-1))
    if prefill:
        args = (_k_major(query), _k_major(keys))
    else:
        args = (_vector(query, 16), from_dlpack(key_cache, assumed_align=16, enable_tvm_ffi=True))
    args += tuple(
        _vector(tensor, 4)
        for tensor in (block_table, row_lengths, thresholds, *blocks, counts, cand_idx, cand_val)
    )
    # The decode filter takes the cache with its static shape, fixed per model.
    key = (prefill, tuple(key_cache.shape), query.device.index)
    compiled = _COMPILED.get(key)
    if compiled is None:
        kernel = PrefillFilter(page_size) if prefill else DecodeFilter(page_size)
        compiled = cute.compile(
            kernel,
            *args,
            cutlass.Int32(topk),
            cutlass.Int32(capacity),
            cutlass.Int32(workers),
            stream=make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
    if prefill:
        tensors = (query.unsqueeze(-1), keys.unsqueeze(-1))
    else:
        tensors = (query, key_cache)
    compiled(
        *tensors,
        block_table,
        row_lengths,
        thresholds,
        *blocks,
        counts,
        cand_idx,
        cand_val,
        topk,
        capacity,
        workers,
    )
