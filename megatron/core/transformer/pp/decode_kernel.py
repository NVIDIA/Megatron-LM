# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


import math
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Float32, Int32, Int64
from flash_attn.cute.tile_scheduler import (
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
)
from flash_attn.cute.utils import ex2_emulation_2


class PPAttentionDecode:
    """Split decode over paged KV: one work item per (flat = b * h_kv + kvh, super-block)."""

    def __init__(
        self,
        super_block_tiles: int = 8,
        tokens_per_item: int = 1,
        rows_per_item: int = 16,
        head_dim: int = 256,
    ):
        assert super_block_tiles >= 1
        assert head_dim in (128, 256), "decode kernel supports head_dim 128 or 256"
        self.head_dim = int(head_dim)
        self.G = super_block_tiles
        self.tokens = int(tokens_per_item)
        self.rows_in = int(rows_per_item)
        if self.rows_in < 1 or self.rows_in % self.tokens:
            raise ValueError(
                f"rows_per_item {self.rows_in} must be a positive multiple "
                f"of tokens_per_item {self.tokens}"
            )
        self.rpt = self.rows_in // self.tokens
        for n in (16, 32, 64):
            if self.rows_in <= n:
                n_rows = n
                break
        else:
            raise ValueError(
                f"rows_per_item {self.rows_in} > 64 does not fit SMEM; "
                "chunk the verify token count"
            )
        self.n_rows = n_rows
        self.qk_acc_dtype = cutlass.Float32
        self.pv_acc_dtype = cutlass.Float32
        self.qk_mma_tiler = (128, self.n_rows, 128)
        self.pv_mma_tiler = (128, self.n_rows, 128)
        self.iterations_qk = self.head_dim // 128
        self.iterations_pv = self.head_dim // 128

        self.n_softmax_wg = max(1, cute.ceil_div(self.n_rows, 32))
        self.pingpong = self.n_rows <= 32 and self.n_softmax_wg == 2
        self.cols_wg = self.n_rows if self.pingpong else min(32, self.n_rows)
        self.softmax_warp_ids = tuple(range(4 * self.n_softmax_wg))
        self.correction_warp_ids = tuple(range(4 * self.n_softmax_wg, 4 * self.n_softmax_wg + 4))
        self.mma_warp_id = 4 * self.n_softmax_wg + 4
        self.load_warp_id = self.mma_warp_id + 1
        self.empty_warp_id = (self.load_warp_id + 1, self.load_warp_id + 2)
        self.split_pv_warp = self.n_rows > 32
        self.pv_warp_id = self.load_warp_id + 1
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_id,
            )
        )
        if self.n_softmax_wg == 1:
            self.num_regs_softmax = 256
            self.num_regs_correction = 160
        else:
            self.num_regs_softmax = 176
            self.num_regs_correction = 128
        self.num_regs_other = 32
        self.ex2_emu_freq = 14
        self.ex2_emu_res = 6
        self.ex2_emu_start_frg = 0

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.softmax_barrier_id = 2
        self.corr_barrier_id = self.softmax_barrier_id + self.n_softmax_wg
        self.ss_free_barrier_id = self.corr_barrier_id + 1
        self.ss_ready_barrier_id = self.corr_barrier_id + 2
        self.s_pair_barrier_id = self.ss_ready_barrier_id + 1
        self.ss_handoff_threads = 128 * (self.n_softmax_wg + 1)
        self.corr_stages_ss = self.n_rows > 32

        self.q_stage = self.iterations_qk
        if self.n_rows <= 32:
            self.kv_stage = 3
            self.kv_stage_v = 2
            self.qk_acc_stage = 2
        else:
            self.kv_stage = 2
            self.kv_stage_v = 2
            self.qk_acc_stage = 2
        self.smem_s_stage = self.qk_acc_stage if self.n_rows <= 32 else 1
        self.p_stage = max(self.qk_acc_stage, 2)
        self.mma_corr_stage = 1
        self.buffer_align_bytes = 1024

        self.tmem_s_offset = 0
        self.tmem_o_offset = self.qk_acc_stage * self.n_rows
        cols = (self.qk_acc_stage + self.iterations_pv) * self.n_rows
        self.tmem_alloc_cols = max(32, 1 << (cols - 1).bit_length())

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        softmax_scale: Float32,
        mSeqUsedK: cute.Tensor,
        mPageTable: cute.Tensor,
        mOPartial: cute.Tensor,
        mMLPartial: cute.Tensor,
        splits: Int32,
        h_kv: Int32,
        num_ctas: Int32,
        stream: cuda.CUstream = None,
    ):
        cta_group = tcgen05.CtaGroup.ONE
        k = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0]))
        v = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[3, 1, 2, 0]))
        flat = mQ.shape[0]
        rows = mQ.shape[1]
        q = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (rows, self.head_dim, flat), stride=(mQ.stride[1], mQ.stride[3], mQ.stride[0])
            ),
        )
        page_table = cute.make_tensor(
            mPageTable.iterator,
            cute.make_layout(
                (mPageTable.shape[0], mPageTable.shape[1]),
                stride=(mPageTable.stride[0], mPageTable.stride[1]),
            ),
        )
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type

        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.K,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (qk_tiled_mma.thr_id.shape,)
        )

        q_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.qk_mma_tiler, self.k_dtype, self.kv_stage
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.kv_stage_v
        )
        p_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, self.p_stage
        )
        s_smem_layout = cute.make_layout(
            (self.n_rows, 128, self.smem_s_stage), stride=(128, 1, self.n_rows * 128)
        )
        stats_smem_layout = cute.make_layout(
            (self.n_rows, 2, self.p_stage), stride=(2, 1, 2 * self.n_rows)
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        self.tma_copy_kv_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_v_bytes = cute.size_in_bytes(self.v_dtype, v_smem_layout)

        @cute.struct
        class SharedStorage:
            """Shared memory of one CTA."""

            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage_v * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.p_stage * 2]
            s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.p_stage * 2]
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            pingpong_mbar_ptr: cute.struct.MemRange[Int64, 4]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            tmem_dealloc_mbar: Int64
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage
        scale_softmax_log2 = softmax_scale * math.log2(math.exp(1.0))
        grid = (Int32(num_ctas), Int32(1), Int32(1))
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            mOPartial,
            mMLPartial,
            mSeqUsedK,
            page_table,
            scale_softmax_log2,
            splits,
            h_kv,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            v_smem_layout_staged,
            p_smem_layout_staged,
            s_smem_layout,
            stats_smem_layout,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=False,
        )

    @cute.kernel
    def kernel(  # pylint: disable=missing-function-docstring
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_nkl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mOPartial: cute.Tensor,
        mMLPartial: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mPageTable: cute.Tensor,
        scale_softmax_log2: Float32,
        splits: Int32,
        h_kv: Int32,
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        s_smem_layout: cute.Layout,
        stats_smem_layout: cute.Layout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage_v,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                4 * self.threads_per_warp
                if (self.corr_stages_ss or self.pingpong)
                else len(self.softmax_warp_ids) * self.threads_per_warp
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        mma_s_producer, mma_s_consumer = mma_s_pipe.make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_stage,
            producer_group=make_thread_cooperative_group(4 * self.n_rows),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.p_stage,
            producer_group=make_thread_cooperative_group(4 * self.n_rows),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                4 * self.threads_per_warp
                if self.pingpong
                else self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.correction_warp_ids) * self.threads_per_warp
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

        pingpong_mbar = storage.pingpong_mbar_ptr.data_ptr()
        if cutlass.const_expr(self.pingpong):
            if warp_idx == 0:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(4):
                        cute.arch.mbarrier_init(pingpong_mbar + i, 4 * self.cols_wg)
                cute.arch.mbarrier_init_fence()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            self.q_dtype,
            q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            self.k_dtype,
            k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV = smem.allocate_tensor(
            self.v_dtype,
            v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sP = smem.allocate_tensor(
            self.q_dtype,
            p_smem_layout_staged.outer,
            swizzle=p_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sS = smem.allocate_tensor(
            self.qk_acc_dtype, s_smem_layout, byte_alignment=128, swizzle=cute.make_swizzle(3, 4, 3)
        )
        sAlpha = smem.allocate_tensor(
            self.qk_acc_dtype, cute.make_layout(self.n_rows), byte_alignment=128
        )
        sCorr = smem.allocate_tensor(self.qk_acc_dtype, stats_smem_layout, byte_alignment=128)
        sSum = smem.allocate_tensor(
            self.qk_acc_dtype, cute.make_layout(self.n_rows), byte_alignment=128
        )
        sMax = smem.allocate_tensor(
            self.qk_acc_dtype, cute.make_layout(self.n_rows), byte_alignment=128
        )
        if cutlass.const_expr(self.pingpong):
            sM = smem.allocate_tensor(
                self.qk_acc_dtype, cute.make_layout((2, self.n_rows)), byte_alignment=128
            )
            sL = smem.allocate_tensor(
                self.qk_acc_dtype, cute.make_layout((2, self.n_rows)), byte_alignment=128
            )
        else:
            sM = sMax
            sL = sSum

        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        tSrK = qk_thr_mma.make_fragment_A(sK)
        tSrQ = qk_thr_mma.make_fragment_B(sQ)
        tOrV = pv_thr_mma.make_fragment_A(sV)
        tOrP = pv_thr_mma.make_fragment_B(sP)
        qk_acc_shape = qk_thr_mma.partition_shape_C(self.qk_mma_tiler[:2])
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        pv_acc_shape = pv_thr_mma.partition_shape_C(self.pv_mma_tiler[:2])
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_staged = cute.make_tensor(
            tOtO.iterator + self.tmem_o_offset,
            cute.append(tOtO.layout, cute.make_layout(self.iterations_pv, stride=self.n_rows)),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)

        p_stage_elems = self.n_rows * 128
        sP_view = cute.make_tensor(
            sP.iterator,
            cute.composition(
                sP.layout,
                cute.make_layout(
                    (self.n_rows, (16, 8), self.p_stage),
                    stride=(1, (self.n_rows, self.n_rows * 16), p_stage_elems),
                ),
            ),
        )

        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        cta_id, _, _ = cute.arch.block_idx()
        grid_x = cute.arch.grid_dim()[0]
        num_items = mQ_nkl.shape[2] * splits
        seqlen_q = Int32(mQ_nkl.shape[0])

        gQ_nkl = cute.flat_divide(mQ_nkl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgQ = qk_thr_mma.partition_B(gQ_nkl)
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ, 0, 3),
        )
        gK = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
        gV = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[0, 2]))
        tSgK = qk_thr_mma.partition_A(gK)
        tSgV = pv_thr_mma.partition_A(gV)
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK, 0, 3),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tSgV, 0, 3),
        )

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            ltidx, _, _ = cute.arch.thread_idx()
            lane0 = ltidx % 32 == 0
            for item in cutlass.range(cta_id, num_items, grid_x, unroll=1):
                (
                    flat_coord,
                    batch_coord,
                    kv_head_coord,
                    seqlen_k,
                    qlen,
                    n_tiles_k,
                    loop_start,
                    loop_steps,
                    loop_end,
                ) = self._work_item(item, splits, h_kv, mSeqUsedK)
                iloc = (item - cta_id) // grid_x
                if loop_steps > 0:
                    for half in cutlass.range_constexpr(self.iterations_qk):
                        q_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, 0, half, flat_coord],
                            tQsQ[None, q_handle.index],
                            tma_bar_ptr=q_handle.barrier,
                        )
                    kv_coord = loop_start
                    k_page = mPageTable[batch_coord, kv_coord]
                    for half in cutlass.range_constexpr(self.iterations_qk):
                        k_handle = load_k_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, 0, half, kv_head_coord, k_page],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                    kv_coord += 1
                    v_page_prev = k_page
                    if loop_steps > 1:
                        k_page = mPageTable[batch_coord, kv_coord]
                    for i in cutlass.range(1, loop_steps, 1, unroll=1):
                        for half in cutlass.range_constexpr(self.iterations_qk):
                            k_handle = load_k_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, 0, half, kv_head_coord, k_page],
                                tKsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                        for half in cutlass.range_constexpr(self.iterations_pv):
                            v_handle = load_v_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV[None, half, 0, kv_head_coord, v_page_prev],
                                tVsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                        v_page_prev = k_page
                        kv_coord += 1
                        if kv_coord < loop_end:
                            k_page = mPageTable[batch_coord, kv_coord]
                    for half in cutlass.range_constexpr(self.iterations_pv):
                        v_handle = load_v_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, half, 0, kv_head_coord, v_page_prev],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=v_handle.barrier,
                        )
            load_k_producer.tail()
            load_v_producer.tail()

        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            mtidx, _, _ = cute.arch.thread_idx()
            mlane0 = mtidx % 32 == 0
            for item in cutlass.range(cta_id, num_items, grid_x, unroll=1):
                (
                    flat_coord,
                    batch_coord,
                    kv_head_coord,
                    seqlen_k,
                    qlen,
                    n_tiles_k,
                    loop_start,
                    loop_steps,
                    loop_end,
                ) = self._work_item(item, splits, h_kv, mSeqUsedK)
                iloc = (item - cta_id) // grid_x
                load_q_releaser = load_q_consumer.clone()
                if loop_steps > 0:
                    s_handle = mma_s_producer.acquire_and_advance()
                    tStS_slice = tStS[None, None, None, s_handle.index]
                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for half in cutlass.range_constexpr(self.iterations_qk):
                        load_q_consumer.wait_and_advance()
                        tSrQ_slice = tSrQ[None, None, None, half]
                        k_handle = load_k_consumer.wait_and_advance()
                        tSrK_slice = tSrK[None, None, None, k_handle.index]
                        num_kphases = cute.size(tSrK_slice, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            cute.gemm(
                                qk_tiled_mma,
                                tStS_slice,
                                tSrK_slice[None, None, kphase_idx],
                                tSrQ_slice[None, None, kphase_idx],
                                tStS_slice,
                            )
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        k_handle.release()
                        if loop_steps == 1:
                            load_q_releaser.release()
                            load_q_releaser.advance()
                    s_handle.commit()
                    if loop_steps > 1:
                        for i in cutlass.range(1, loop_steps - 1, 1, unroll=1):
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for half in cutlass.range_constexpr(self.iterations_qk):
                                tSrQ_slice = tSrQ[None, None, None, half]
                                k_handle = load_k_consumer.wait_and_advance()
                                tSrK_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrK_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrK_slice[None, None, kphase_idx],
                                        tSrQ_slice[None, None, kphase_idx],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                            s_handle.commit()
                            if cutlass.const_expr(not self.split_pv_warp):
                                p_handle = p_mma_consumer.wait_and_advance()

                                o_handle = mma_corr_producer.acquire_and_advance()

                                pv_whether_acc = (loop_start + i - 1) % self.G != 0

                                for half in cutlass.range_constexpr(self.iterations_pv):

                                    v_handle = load_v_consumer.wait_and_advance()

                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)

                                    tOtO_slice = tOtO_staged[None, None, None, half]

                                    tOrV_slice = tOrV[None, None, None, v_handle.index]

                                    tOrP_slice = tOrP[None, None, None, p_handle.index]

                                    num_kphases = cute.size(tOrV_slice, mode=[2])

                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):

                                        cute.gemm(
                                            pv_tiled_mma,
                                            tOtO_slice,
                                            tOrV_slice[None, None, kphase_idx],
                                            tOrP_slice[None, None, kphase_idx],
                                            tOtO_slice,
                                        )

                                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                                    v_handle.release()

                                o_handle.commit()

                                p_handle.release()

                        s_handle = mma_s_producer.acquire_and_advance()
                        tStS_slice = tStS[None, None, None, s_handle.index]
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for half in cutlass.range_constexpr(self.iterations_qk):
                            tSrQ_slice = tSrQ[None, None, None, half]
                            k_handle = load_k_consumer.wait_and_advance()
                            tSrK_slice = tSrK[None, None, None, k_handle.index]
                            num_kphases = cute.size(tSrK_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                cute.gemm(
                                    qk_tiled_mma,
                                    tStS_slice,
                                    tSrK_slice[None, None, kphase_idx],
                                    tSrQ_slice[None, None, kphase_idx],
                                    tStS_slice,
                                )
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            k_handle.release()
                            load_q_releaser.release()
                            load_q_releaser.advance()
                        s_handle.commit()
                        if cutlass.const_expr(not self.split_pv_warp):
                            p_handle = p_mma_consumer.wait_and_advance()

                            o_handle = mma_corr_producer.acquire_and_advance()

                            pv_whether_acc = (loop_end - 2) % self.G != 0

                            for half in cutlass.range_constexpr(self.iterations_pv):

                                v_handle = load_v_consumer.wait_and_advance()

                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)

                                tOtO_slice = tOtO_staged[None, None, None, half]

                                tOrV_slice = tOrV[None, None, None, v_handle.index]

                                tOrP_slice = tOrP[None, None, None, p_handle.index]

                                num_kphases = cute.size(tOrV_slice, mode=[2])

                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):

                                    cute.gemm(
                                        pv_tiled_mma,
                                        tOtO_slice,
                                        tOrV_slice[None, None, kphase_idx],
                                        tOrP_slice[None, None, kphase_idx],
                                        tOtO_slice,
                                    )

                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                                v_handle.release()

                            o_handle.commit()

                            p_handle.release()

                    if cutlass.const_expr(not self.split_pv_warp):
                        p_handle = p_mma_consumer.wait_and_advance()

                        o_handle = mma_corr_producer.acquire_and_advance()

                        pv_whether_acc = (loop_end - 1) % self.G != 0

                        for half in cutlass.range_constexpr(self.iterations_pv):

                            v_handle = load_v_consumer.wait_and_advance()

                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)

                            tOtO_slice = tOtO_staged[None, None, None, half]

                            tOrV_slice = tOrV[None, None, None, v_handle.index]

                            tOrP_slice = tOrP[None, None, None, p_handle.index]

                            num_kphases = cute.size(tOrV_slice, mode=[2])

                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):

                                cute.gemm(
                                    pv_tiled_mma,
                                    tOtO_slice,
                                    tOrV_slice[None, None, kphase_idx],
                                    tOrP_slice[None, None, kphase_idx],
                                    tOtO_slice,
                                )

                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                            v_handle.release()

                        o_handle.commit()

                        p_handle.release()

            mma_s_producer.tail()
            if cutlass.const_expr(not self.split_pv_warp):
                mma_corr_producer.tail()

        if cutlass.const_expr(self.split_pv_warp):
            if warp_idx == self.pv_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                ptidx, _, _ = cute.arch.thread_idx()
                plane0 = ptidx % 32 == 0
                for item in cutlass.range(cta_id, num_items, grid_x, unroll=1):
                    (
                        flat_coord,
                        batch_coord,
                        kv_head_coord,
                        seqlen_k,
                        qlen,
                        n_tiles_k,
                        loop_start,
                        loop_steps,
                        loop_end,
                    ) = self._work_item(item, splits, h_kv, mSeqUsedK)
                    iloc = (item - cta_id) // grid_x
                    if loop_steps > 0:
                        for j in cutlass.range(0, loop_steps, 1, unroll=1):
                            p_handle = p_mma_consumer.wait_and_advance()

                            o_handle = mma_corr_producer.acquire_and_advance()

                            pv_whether_acc = (loop_start + j) % self.G != 0

                            for half in cutlass.range_constexpr(self.iterations_pv):

                                v_handle = load_v_consumer.wait_and_advance()

                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)

                                tOtO_slice = tOtO_staged[None, None, None, half]

                                tOrV_slice = tOrV[None, None, None, v_handle.index]

                                tOrP_slice = tOrP[None, None, None, p_handle.index]

                                num_kphases = cute.size(tOrV_slice, mode=[2])

                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):

                                    cute.gemm(
                                        pv_tiled_mma,
                                        tOtO_slice,
                                        tOrV_slice[None, None, kphase_idx],
                                        tOrP_slice[None, None, kphase_idx],
                                        tOtO_slice,
                                    )

                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                                v_handle.release()

                            o_handle.commit()

                            p_handle.release()

                mma_corr_producer.tail()

        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            tidx, _, _ = cute.arch.thread_idx()
            wg = tidx // 128
            s_ord_base = Int32(0)
            if cutlass.const_expr(self.pingpong):
                if wg == 1:
                    p_mma_producer.advance()
                    s_corr_producer.advance()
            for item in cutlass.range(cta_id, num_items, grid_x, unroll=1):
                (
                    flat_coord,
                    batch_coord,
                    kv_head_coord,
                    seqlen_k,
                    qlen,
                    n_tiles_k,
                    loop_start,
                    loop_steps,
                    loop_end,
                ) = self._work_item(item, splits, h_kv, mSeqUsedK)
                iloc = (item - cta_id) // grid_x
                if loop_steps > 0:
                    row_st = cute.make_rmem_tensor(cute.make_layout((1, 3)), Float32)
                    row_st[0, 0] = -Float32.inf
                    row_st[0, 1] = Float32(0.0)
                    row_st[0, 2] = -Float32.inf
                    s_ord0 = s_ord_base
                    s_ord_base += loop_steps
                    step_first = loop_start
                    step_stride = 1
                    if cutlass.const_expr(self.pingpong):
                        step_first = loop_start + (s_ord0 + wg) % 2
                        step_stride = 2
                    for step in cutlass.range(step_first, loop_end, step_stride, unroll=1):
                        is_sb_boundary = (step % self.G == 0) & (step > loop_start)
                        mma_s_consumer, p_mma_producer, s_corr_producer = self.softmax_step(
                            step,
                            (step + 1) * 128 > seqlen_k - self.tokens,
                            cutlass.Boolean((step == loop_start) | (not self.corr_stages_ss)),
                            cutlass.Boolean(self.corr_stages_ss & (step < loop_end - 1)),
                            s_ord0 + step - loop_start,
                            (
                                row_st,
                                is_sb_boundary,
                                cutlass.Boolean(step == loop_start),
                                seqlen_q,
                                seqlen_k,
                                qlen,
                                scale_softmax_log2,
                            ),
                            (tStS, sS, sP_view, sCorr, sM, sL, pingpong_mbar),
                            (mma_s_consumer, p_mma_producer, s_corr_producer, mma_s_pipe),
                        )
                        if cutlass.const_expr(self.pingpong):
                            p_mma_producer.advance()
                            s_corr_producer.advance()
                    if cutlass.const_expr(self.pingpong):
                        if (s_ord0 + loop_steps - 1) % 2 == wg:
                            sum_handle = sum_producer.acquire_and_advance()
                            tw = tidx % 128
                            if tw < 4 * self.cols_wg:
                                if tw % 4 == 0:
                                    r = tw // 4
                                    sSum[r] = sL[wg, r]
                                    sMax[r] = sM[wg, r]
                            cute.arch.fence_view_async_shared()
                            sum_handle.commit()
                        else:
                            sum_producer.advance()
                    else:
                        sum_handle = sum_producer.acquire_and_advance()
                        tw = tidx % 128
                        if tw < 4 * self.cols_wg:
                            if tw % 4 == 0:
                                r = self.cols_wg * (tidx // 128) + tw // 4
                                sSum[r] = row_st[0, 1]
                                sMax[r] = row_st[0, 0]
                        cute.arch.fence_view_async_shared()
                        sum_handle.commit()
            n_prod_warps = len(self.softmax_warp_ids) if self.pingpong else self.n_rows // 8
            if warp_idx < n_prod_warps:
                p_mma_producer.tail()
                s_corr_producer.tail()

        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
            s_ord_base = Int32(0)
            for item in cutlass.range(cta_id, num_items, grid_x, unroll=1):
                (
                    flat_coord,
                    batch_coord,
                    kv_head_coord,
                    seqlen_k,
                    qlen,
                    n_tiles_k,
                    loop_start,
                    loop_steps,
                    loop_end,
                ) = self._work_item(item, splits, h_kv, mSeqUsedK)
                iloc = (item - cta_id) // grid_x
                if loop_steps > 0:
                    stats_handle = s_corr_consumer.wait_and_advance()
                    stats_handle.release()
                    s_ord0 = s_ord_base
                    s_ord_base += loop_steps
                    if self.corr_stages_ss:
                        if loop_steps > 1:
                            self._corr_stage_ss(tStS, sS, mma_s_pipe, s_ord0 + 1, thread_idx)
                    for step in cutlass.range(1, loop_steps, 1, unroll=1):
                        if self.corr_stages_ss:
                            if step + 1 <= loop_steps - 1:
                                self._corr_stage_ss(
                                    tStS, sS, mma_s_pipe, s_ord0 + step + 1, thread_idx
                                )
                        is_sb_boundary = (loop_start + step) % self.G == 0
                        mma_corr_consumer, s_corr_consumer = self.correction_step(
                            is_sb_boundary,
                            loop_start + step,
                            scale_softmax_log2,
                            (s_corr_consumer, sCorr, sAlpha),
                            (mma_corr_consumer, tOtO_staged),
                            (mOPartial, mMLPartial, flat_coord, seqlen_q),
                            thread_idx,
                        )
                    mma_corr_consumer, sum_consumer = self.correction_epilog(
                        (seqlen_q,),
                        (sum_consumer, sSum, sMax),
                        (mma_corr_consumer, tOtO_staged),
                        (mOPartial, mMLPartial, flat_coord),
                        (loop_start + loop_steps - 1) // self.G,
                        thread_idx,
                    )

        if warp_idx > self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _work_item(self, item: Int32, splits: Int32, h_kv: Int32, mSeqUsedK: cute.Tensor):
        """Map persistent item id -> (flat, s) and that item's tile range."""
        flat_coord = item // splits
        s_idx = item - flat_coord * splits
        batch_coord = flat_coord // h_kv
        kv_head_coord = flat_coord - batch_coord * h_kv
        seqlen_k = mSeqUsedK[batch_coord]
        qlen = Int32(self.tokens)
        n_tiles_k = (seqlen_k + 127) // 128
        loop_start = s_idx * self.G
        loop_end = cutlass.min(loop_start + self.G, n_tiles_k)
        loop_steps = loop_end - loop_start
        return (
            flat_coord,
            batch_coord,
            kv_head_coord,
            seqlen_k,
            qlen,
            n_tiles_k,
            loop_start,
            loop_steps,
            loop_end,
        )

    @cute.jit
    def softmax_step(
        self,
        step: Int32,
        need_apply_mask: cutlass.Boolean,
        self_stage: cutlass.Boolean,
        next_staged: cutlass.Boolean,
        s_ord: Int32,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ):
        """S^T stage -> per-row softmax -> sP stage."""
        row_st, is_sb_boundary, first, seqlen_q, seqlen_k, qlen, scale = value_args
        tStS, sS, sP_view, sCorr, sM, sL, pingpong_mbar = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer, mma_s_pipe = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        wg = tidx // 128
        tw = tidx % 128
        other = 1 - wg
        col0 = Int32(0)
        if cutlass.const_expr(not self.pingpong):
            col0 = self.cols_wg * wg
        pingpong_phase = ((s_ord - 1) // 2) % 2
        s_stage = Int32(0)
        if self_stage:
            s_stage = s_ord % self.qk_acc_stage
            mma_s_pipe.consumer_wait(
                pipeline.PipelineState(
                    self.qk_acc_stage, Int32(0), s_stage, (s_ord // self.qk_acc_stage) % 2
                )
            )
            tStS_slice = tStS[(None, None), 0, 0, s_stage]
            tStS_wg = cute.make_tensor(
                tStS_slice.iterator + col0, cute.make_layout((128, self.cols_wg), stride=(65536, 1))
            )
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(1)), self.qk_acc_dtype
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_wg)
            thr_load = tmem_tiled_load.get_slice(tw)
            tTMEM_LOADtS = thr_load.partition_S(tStS_wg)
            cS = cute.make_identity_tensor(tStS_wg.shape)
            tTMEM_LOADrS = cute.make_rmem_tensor(thr_load.partition_D(cS).shape, self.qk_acc_dtype)
            cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            if self.corr_stages_ss:
                cute.arch.barrier(
                    barrier_id=self.s_pair_barrier_id, number_of_threads=128 * self.n_softmax_wg
                )
                if wg == 0:
                    mma_s_pipe.consumer_release(
                        pipeline.PipelineState(self.qk_acc_stage, Int32(0), s_stage, Int32(0))
                    )
            else:
                mma_s_pipe.consumer_release(
                    pipeline.PipelineState(self.qk_acc_stage, Int32(0), s_stage, Int32(0))
                )
            for e in cutlass.range_constexpr(cute.size(tTMEM_LOADrS)):
                sS[col0 + e, tw, s_stage % self.smem_s_stage] = tTMEM_LOADrS[e]
            cute.arch.barrier(
                barrier_id=self.softmax_barrier_id + wg, number_of_threads=self.threads_per_warp * 4
            )
        else:
            cute.arch.barrier(
                barrier_id=self.ss_ready_barrier_id, number_of_threads=self.ss_handoff_threads
            )
        if tw >= 4 * self.cols_wg:
            if next_staged:
                cute.arch.barrier_arrive(
                    barrier_id=self.ss_free_barrier_id, number_of_threads=self.ss_handoff_threads
                )
        if tw < 4 * self.cols_wg:
            r = col0 + tw // 4
            f = tw % 4
            stats_handle = s_corr_producer.acquire_and_advance()
            p_handle = p_mma_producer.acquire_and_advance()
            old_row_max = Float32(-Float32.inf)
            fold_m = Float32(-Float32.inf)
            fold_l = Float32(0.0)
            if cutlass.const_expr(not self.pingpong):
                fold_m = row_st[0, 2]
                fold_l = row_st[0, 1]
                if is_sb_boundary:
                    row_st[0, 0] = -Float32.inf
                    row_st[0, 1] = Float32(0.0)
                old_row_max = row_st[0, 0]
            frag = cute.make_rmem_tensor(cute.make_layout(32), self.qk_acc_dtype)
            sS_row = sS[(r, None, s_stage % self.smem_s_stage)]
            sS_frag = cute.local_tile(sS_row, (32,), (f,))
            cute.autovec_copy(sS_frag, frag)
            if next_staged:
                cute.arch.barrier_arrive(
                    barrier_id=self.ss_free_barrier_id, number_of_threads=self.ss_handoff_threads
                )
            if (r >= seqlen_q) | (r // self.rpt >= qlen):
                for j in cutlass.range_constexpr(32):
                    frag[j] = -Float32.inf
            if need_apply_mask:
                key_limit = seqlen_k - qlen + 1 + r // self.rpt
                for j in cutlass.range_constexpr(32):
                    if step * 128 + 32 * f + j >= key_limit:
                        frag[j] = -Float32.inf
            fmax = frag.load().reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
            m_other = cute.arch.shuffle_sync_bfly(fmax, offset=1)
            fmax = cute.arch.fmax(fmax, m_other)
            m_other = cute.arch.shuffle_sync_bfly(fmax, offset=2)
            fmax = cute.arch.fmax(fmax, m_other)
            if cutlass.const_expr(self.pingpong):
                if s_ord > 0:
                    cute.arch.mbarrier_wait(pingpong_mbar + other, pingpong_phase)
                if not first:
                    fold_m = sM[other, r]
                old_row_max = fold_m
                if is_sb_boundary:
                    old_row_max = -Float32.inf
                    cute.arch.mbarrier_wait(pingpong_mbar + 2 + other, pingpong_phase)
                    fold_l = sL[other, r]
            row_max_new = cute.arch.fmax(old_row_max, fmax)
            if cutlass.const_expr(self.pingpong):
                if f == 0:
                    sM[wg, r] = row_max_new
                cute.arch.mbarrier_arrive(pingpong_mbar + wg)
            row_max_safe = row_max_new
            if row_max_new == -cutlass.Float32.inf:
                row_max_safe = 0.0
            if f == 0:
                sCorr[r, 0, stats_handle.index] = fold_m if is_sb_boundary else old_row_max
                sCorr[r, 1, stats_handle.index] = fold_l if is_sb_boundary else row_max_safe
            stats_handle.commit()
            scale_neg = (0.0 - row_max_safe) * scale
            rP = cute.make_rmem_tensor(cute.make_layout(32), self.q_dtype)
            for k in cutlass.range_constexpr(0, 32, 2):
                frag[k], frag[k + 1] = cute.arch.fma_packed_f32x2(
                    (frag[k], frag[k + 1]), (scale, scale), (scale_neg, scale_neg)
                )
                if cutlass.const_expr(k % self.ex2_emu_freq < self.ex2_emu_freq - self.ex2_emu_res):
                    frag[k] = cute.math.exp2(frag[k], fastmath=True)
                    frag[k + 1] = cute.math.exp2(frag[k + 1], fastmath=True)
                else:
                    if f == 3:
                        frag[k] = cute.math.exp2(frag[k], fastmath=True)
                        frag[k + 1] = cute.math.exp2(frag[k + 1], fastmath=True)
                    else:
                        frag[k], frag[k + 1] = ex2_emulation_2(frag[k], frag[k + 1])
            rP.store(frag.load().to(self.q_dtype))
            for c in cutlass.range_constexpr(4):
                key = 32 * f + 8 * c
                sP_16 = sP_view[(r, (None, key // 16), p_handle.index)]
                sP_frag = cute.local_tile(sP_16, (8,), ((key % 16) // 8,))
                rP_frag = cute.local_tile(rP, (8,), (c,))
                cute.autovec_copy(rP_frag, sP_frag)
            cute.arch.fence_view_async_shared()
            p_handle.commit()
            acc_scale = cute.math.exp2(scale * (old_row_max - row_max_safe), fastmath=True) * 0.5
            l_prev = Float32(0.0)
            if cutlass.const_expr(self.pingpong):
                if not is_sb_boundary:
                    if s_ord > 0:
                        cute.arch.mbarrier_wait(pingpong_mbar + 2 + other, pingpong_phase)
                    if not first:
                        l_prev = sL[other, r]
            else:
                l_prev = row_st[0, 1]
            row_sum_scaled = l_prev * acc_scale
            a0_seed = row_sum_scaled if f == 0 else Float32(0.0)
            a = (a0_seed, a0_seed)
            for k in cutlass.range_constexpr(0, 32, 2):
                a = cute.arch.add_packed_f32x2(a, (frag[k], frag[k + 1]))
            a1 = (
                cute.arch.shuffle_sync_bfly(a[0], offset=1),
                cute.arch.shuffle_sync_bfly(a[1], offset=1),
            )
            a2 = (
                cute.arch.shuffle_sync_bfly(a[0], offset=2),
                cute.arch.shuffle_sync_bfly(a[1], offset=2),
            )
            a3 = (
                cute.arch.shuffle_sync_bfly(a[0], offset=3),
                cute.arch.shuffle_sync_bfly(a[1], offset=3),
            )
            b0 = cute.arch.add_packed_f32x2(a, a1)
            b1 = cute.arch.add_packed_f32x2(a2, a3)
            c = cute.arch.add_packed_f32x2(b0, b1)
            if cutlass.const_expr(self.pingpong):
                if f == 0:
                    sL[wg, r] = c[0] + c[1]
                cute.arch.mbarrier_arrive(pingpong_mbar + 2 + wg)
            else:
                row_st[0, 1] = c[0] + c[1]
                row_st[0, 0] = row_max_new
                row_st[0, 2] = row_max_new
        cute.arch.barrier(
            barrier_id=self.softmax_barrier_id + wg, number_of_threads=self.threads_per_warp * 4
        )
        return mma_s_consumer, p_mma_producer, s_corr_producer

    @cute.jit
    def _corr_stage_ss(
        self, tStS: cute.Tensor, sS: cute.Tensor, mma_s_pipe, s_ord: Int32, thread_idx: Int32
    ):
        """Stage sS for a future tile from the correction warpgroup."""
        cute.arch.barrier(
            barrier_id=self.ss_free_barrier_id, number_of_threads=self.ss_handoff_threads
        )
        s_stage = s_ord % self.qk_acc_stage
        mma_s_pipe.consumer_wait(
            pipeline.PipelineState(
                self.qk_acc_stage, Int32(0), s_stage, (s_ord // self.qk_acc_stage) % 2
            )
        )
        tStS_slice = tStS[(None, None), 0, 0, s_stage]
        for h in cutlass.range_constexpr(self.n_softmax_wg):
            tStS_h = cute.make_tensor(
                tStS_slice.iterator + self.cols_wg * h,
                cute.make_layout((128, self.cols_wg), stride=(65536, 1)),
            )
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(1)), self.qk_acc_dtype
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_h)
            thr_load = tmem_tiled_load.get_slice(thread_idx)
            tTMEM_LOADtS = thr_load.partition_S(tStS_h)
            cS = cute.make_identity_tensor(tStS_h.shape)
            tTMEM_LOADrS = cute.make_rmem_tensor(thr_load.partition_D(cS).shape, self.qk_acc_dtype)
            cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            for e in cutlass.range_constexpr(cute.size(tTMEM_LOADrS)):
                sS[self.cols_wg * h + e, thread_idx, s_stage % self.smem_s_stage] = tTMEM_LOADrS[e]
        mma_s_pipe.consumer_release(
            pipeline.PipelineState(self.qk_acc_stage, Int32(0), s_stage, Int32(0))
        )
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=self.ss_ready_barrier_id, number_of_threads=self.ss_handoff_threads
        )

    @cute.jit
    def correction_step(
        self,
        is_boundary,
        tile_idx: Int32,
        scale_softmax_log2: Float32,
        stats_args: tuple,
        o_args: tuple,
        dump_args: tuple,
        thread_idx: Int32,
    ):
        """Rescale O^T by alpha, or at a super-block boundary dump the partial and its (m, l)."""
        s_corr_consumer, sCorr, sAlpha = stats_args
        mma_o_consumer, tOtO_staged = o_args
        mOPartial, mMLPartial, flat_coord, seqlen_q = dump_args
        stats_handle = s_corr_consumer.wait_and_advance()
        st = stats_handle.index
        o_handle = mma_o_consumer.wait_and_advance()
        if is_boundary:
            s_done = tile_idx // self.G - 1
            if thread_idx < seqlen_q:
                mMLPartial[flat_coord, s_done, thread_idx, 0] = sCorr[thread_idx, 0, st]
                mMLPartial[flat_coord, s_done, thread_idx, 1] = sCorr[thread_idx, 1, st]
            self._dump_o(mOPartial, tOtO_staged, flat_coord, s_done, seqlen_q, thread_idx)
        else:
            if thread_idx < self.n_rows:
                sAlpha[thread_idx] = cute.math.exp2(
                    scale_softmax_log2 * (sCorr[thread_idx, 0, st] - sCorr[thread_idx, 1, st]),
                    fastmath=True,
                )
            cute.arch.barrier(
                barrier_id=self.corr_barrier_id,
                number_of_threads=self.threads_per_warp * len(self.correction_warp_ids),
            )
            for half in cutlass.range_constexpr(self.iterations_pv):
                tOtO = tOtO_staged[(None, None), 0, 0, half]
                for cc in cutlass.range_constexpr(cute.ceil_div(self.n_rows, 32)):
                    cols_cc = min(32, self.n_rows - 32 * cc)
                    tO = cute.make_tensor(
                        tOtO.iterator + 32 * cc, cute.make_layout((128, cols_cc), stride=(65536, 1))
                    )
                    tmem_load_atom = cute.make_copy_atom(
                        tcgen05.Ld32x32bOp(tcgen05.Repetition(1)), self.pv_acc_dtype
                    )
                    tiled_ld = tcgen05.make_tmem_copy(tmem_load_atom, tO)
                    thr_ld = tiled_ld.get_slice(thread_idx)
                    tmem_store_atom = cute.make_copy_atom(
                        tcgen05.St32x32bOp(tcgen05.Repetition(1)), self.pv_acc_dtype
                    )
                    tiled_st = tcgen05.make_tmem_copy(tmem_store_atom, tO)
                    thr_st = tiled_st.get_slice(thread_idx)
                    tTMEM_LOADtO = thr_ld.partition_S(tO)
                    tTMEM_STOREtO = thr_st.partition_D(tO)
                    cO = cute.make_identity_tensor(tO.shape)
                    tTMrO = cute.make_rmem_tensor(thr_ld.partition_D(cO).shape, self.pv_acc_dtype)
                    cute.copy(tiled_ld, tTMEM_LOADtO, tTMrO)
                    for j in cutlass.range_constexpr(0, cute.size(tTMrO), 2):
                        tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                            (tTMrO[j], tTMrO[j + 1]), (sAlpha[32 * cc + j], sAlpha[32 * cc + j + 1])
                        )
                    cute.copy(tiled_st, tTMrO, tTMEM_STOREtO)
            cute.arch.fence_view_async_tmem_store()
            cute.arch.barrier(
                barrier_id=self.corr_barrier_id,
                number_of_threads=self.threads_per_warp * len(self.correction_warp_ids),
            )
        cute.arch.fence_view_async_tmem_load()
        stats_handle.release()
        o_handle.release()
        return mma_o_consumer, s_corr_consumer

    @cute.jit
    def _dump_o(
        self,
        mOPartial: cute.Tensor,
        tOtO_staged: cute.Tensor,
        flat_coord: Int32,
        s_idx: Int32,
        seqlen_q: Int32,
        thread_idx: Int32,
    ):
        """Dump O^T[lane=d, col=r] -> mOPartial[flat, s, r, d], rows < seqlen_q."""
        for half in cutlass.range_constexpr(self.iterations_pv):
            tOtO = tOtO_staged[(None, None), 0, 0, half]
            for cc in cutlass.range_constexpr(cute.ceil_div(self.n_rows, 32)):
                cols_cc = min(32, self.n_rows - 32 * cc)
                tO = cute.make_tensor(
                    tOtO.iterator + 32 * cc, cute.make_layout((128, cols_cc), stride=(65536, 1))
                )
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(1)), self.pv_acc_dtype
                )
                tiled_ld = tcgen05.make_tmem_copy(tmem_load_atom, tO)
                thr_ld = tiled_ld.get_slice(thread_idx)
                tTMEM_LOADtO = thr_ld.partition_S(tO)
                cO = cute.make_identity_tensor(tO.shape)
                tTMrO = cute.make_rmem_tensor(thr_ld.partition_D(cO).shape, self.pv_acc_dtype)
                cute.copy(tiled_ld, tTMEM_LOADtO, tTMrO)
                lane = thread_idx
                for j in cutlass.range_constexpr(cols_cc):
                    r = 32 * cc + j
                    if r < seqlen_q:
                        mOPartial[flat_coord, s_idx, r, half * 128 + lane] = tTMrO[j]

    @cute.jit
    def correction_epilog(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: tuple,
        dump_args: tuple,
        s_last: Int32,
        thread_idx: Int32,
    ):
        """Dump the last super-block's partial; (m, l) from sSum/sMax."""
        (seqlen_q,) = value_args
        sum_consumer, sSum, sMax = sum_args
        mma_o_consumer, tOtO_staged = o_args
        mOPartial, mMLPartial, flat_coord = dump_args
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx % self.n_rows]
        row_max = sMax[thread_idx % self.n_rows]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        o_handle = mma_o_consumer.wait_and_advance()
        if thread_idx < seqlen_q:
            mMLPartial[flat_coord, s_last, thread_idx, 0] = row_max
            mMLPartial[flat_coord, s_last, thread_idx, 1] = row_sum
        self._dump_o(mOPartial, tOtO_staged, flat_coord, s_last, seqlen_q, thread_idx)
        o_handle.release()
        return mma_o_consumer, sum_consumer
