# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import logging
from typing import Optional, Tuple, Type

try:
    import cuda.bindings.driver as cuda  # type: ignore
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline  # type: ignore
    import cutlass.utils as utils  # type: ignore
    import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
    from cutlass.cute.nvgpu import cpasync, tcgen05

    SM100_TMEM_CAPACITY_COLUMNS: int = 512

    def make_thread_cooperative_group(size: int, alignment: Optional[int] = None):
        """
        Create a thread cooperative group.
        """
        return pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size, alignment=alignment if alignment is not None else size
        )

    class BwdPartialDlogits:
        """
        This class implements the backward kernel for partial d_logits.
        """

        def __init__(
            self,
            reduction: int,
            acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
            use_2cta_instrs: bool = False,
            mma_tiler_mn: Tuple[int, int] = (128, 256),
            vocab_per_split: int = 512,
        ):
            self.REDUCTION: cutlass.Constexpr[cutlass.Int32] = cutlass.const_expr(reduction)
            self.acc_dtype = acc_dtype
            self.use_2cta_instrs = use_2cta_instrs
            self.mma_tiler = (*mma_tiler_mn, 1)
            self.vocab_per_split = vocab_per_split

            self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)

            self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

            self.threads_per_warp: int = 32

            self.epi_warp_ids = (0, 1, 2, 3)
            self.load_warp_ids = 4
            self.mma_warp_ids = 5
            self.empty_warp_ids = (6, 7)

            self.threads_per_cta: int = self.threads_per_warp * len(
                (*self.epi_warp_ids, self.load_warp_ids, self.mma_warp_ids, *self.empty_warp_ids)
            )
            self.cta_sync_barrier = pipeline.NamedBarrier(
                barrier_id=1, num_threads=self.threads_per_cta
            )

            self.buffer_align_bytes: int = 1024
            self.num_regs_other: int = 32
            self.num_regs_epi: int = 192

        def _compute_grid(
            self,
            problem_mnk: Tuple[int, int, int],
            cluster_shape_mn: Tuple[int, int],
            cta_tiler: Tuple[int, int, int],
        ) -> Tuple[int, int, int]:
            cluster_shape_mnk = (*cluster_shape_mn, 1)

            grid = cute.round_up(
                (
                    cute.ceil_div(problem_mnk[0], cta_tiler[0]),
                    cute.ceil_div(self.vocab_per_split, cta_tiler[1]),
                    1,
                ),
                cluster_shape_mnk,
            )
            return grid

        def _compute_stages(
            self,
            tiled_mma: cute.TiledMma,
            mma_tiler: Tuple[int, int, int],
            a_dtype: Type[cutlass.Numeric],
            b_dtype: Type[cutlass.Numeric],
        ):
            num_acc_stage = 1
            num_ab_stage = 4
            num_epi_stage_per_tile = 4
            return num_acc_stage, num_ab_stage, num_epi_stage_per_tile

        def _setup_attributes(
            self,
            tiled_mma: cute.TiledMma,
            a_dtype: Type[cutlass.Numeric],
            b_dtype: Type[cutlass.Numeric],
        ):
            self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,)
            )

            mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            # it requires k-mode to be 128B aligned
            mma_inst_tile_k: int = 4
            self.mma_tiler = (
                self.mma_tiler[0],
                self.mma_tiler[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )

            self.num_acc_stage, self.num_ab_stage, self.num_epi_stage_per_tile = (
                self._compute_stages(tiled_mma, self.mma_tiler, a_dtype, b_dtype)
            )
            self.tmem_alloc_cols = self.num_acc_stage * self.mma_tiler[1]
            assert self.tmem_alloc_cols <= SM100_TMEM_CAPACITY_COLUMNS

            self.cta_tile_shape_mnk = (
                self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
                self.mma_tiler[1],
                self.mma_tiler[2],
            )

        @cute.kernel
        def kernel(
            self,
            split_idx: cutlass.Int32,
            tiled_mma: cute.TiledMma,
            tma_atom_a: cute.CopyAtom,
            mA: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB: cute.Tensor,
            mLabels: cute.Tensor,
            mDlogprobs: cute.Tensor,
            mMaximum: cute.Tensor,
            mAccu: cute.Tensor,
            mDlogits_partial: cute.Tensor,
            scalarNumValidTokens: cute.Pointer,
            ignore_index: cutlass.Int64,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            cluster_layout_vmnk: cute.Layout,
            problem_mnk: Tuple[int, int, int],
            rank: cutlass.Int32,
        ) -> None:
            """
            The backward kernel for partial d_logits.
            """
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            # FIXME: block swizzling applied here
            pidm, pidn = bidx, bidy

            # FIXME: if 2 CTAs, modify here
            cta_rank_in_cluster = 0
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

            # prefetch tma descriptors
            if warp_idx == self.load_warp_ids:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)

            ab_pipeline = pipeline.PipelineTmaUmma.create(
                num_stages=self.num_ab_stage,
                producer_group=make_thread_cooperative_group(len([self.load_warp_ids])),
                consumer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
                tx_count=self.tma_copy_ab_bytes,
                barrier_storage=storage.load_ab_mbar_ptr.data_ptr(),
            )
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )

            mma_pipeline = pipeline.PipelineUmmaAsync.create(
                num_stages=self.num_acc_stage,
                producer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
                consumer_group=make_thread_cooperative_group(
                    self.threads_per_warp * len(self.epi_warp_ids)
                ),
                barrier_storage=storage.mma_mbar_ptr.data_ptr(),
            )
            mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
            if warp_idx == self.empty_warp_ids[0]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, self.threads_per_warp * len(self.epi_warp_ids)
                    )
                    cute.arch.mbarrier_init_fence()

            # -------- tensor partition ------------ #
            # swizzle o [(tileM, tileK), loopM, loopK, stage]
            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            # swizzle o [(tileN, tileK), loopN, loopK, stage]
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )

            # FIXME: if 2 CTAs, modify here
            thr_mma = tiled_mma.get_slice(0)
            # [MMA, loopM, loopK, stage]
            tCsA = thr_mma.make_fragment_A(sA)
            # [MMA, loopN, loopK, stage]
            tCsB = thr_mma.make_fragment_B(sB)

            # [tileM, tileK, loopK]
            gA = cute.local_tile(
                mA, (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]), (pidm, None)
            )
            # [vocab_per_split, dim]
            mB_n = cute.local_tile(
                mB, (self.vocab_per_split, cute.size(mB.layout.shape, mode=[1])), (split_idx, 0)
            )
            # [tileN, tileK, loopK]
            gB = cute.local_tile(
                mB_n, (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]), (pidn, None)
            )

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            # just to make sure SMEM and GMEM tensor has the same size in the first rank
            tCgA = thr_mma.partition_A(gA)
            tCgB = thr_mma.partition_B(gB)
            # [CPY, stage] & [CPY, loopK]
            tTMAsA, tTMAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],  # cta_coord,
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            # [CPY, stage] & [CPY, loopK]
            tTMAsB, tTMAgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],  # cta_coord
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )

            # ------ Allocate TMEM ------ #
            tmem_holding_buf = storage.tmem_holding_buf
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.alloc_tmem(
                    self.tmem_alloc_cols, tmem_holding_buf, is_two_cta=self.use_2cta_instrs
                )
            self.cta_sync_barrier.arrive_and_wait()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )

            tmem_shape = (128, self.tmem_alloc_cols)
            acc_shape = thr_mma.partition_shape_C(tmem_shape)
            tCtC_fake = thr_mma.make_fragment_C(acc_shape)
            # [(tileM, tileN), loopM, loopN]
            tCtC = cute.make_tensor(tmem_ptr, tCtC_fake.layout)

            # ------ Empty ------ #
            if warp_idx in self.empty_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # ------ Load ------ #
            if warp_idx == self.load_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                for k in cutlass.range(cute.size(gA, mode=[2])):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tTMAgA[(None, k)],
                        tTMAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )
                    cute.copy(
                        tma_atom_b,
                        tTMAgB[(None, k)],
                        tTMAsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )
                    ab_pipeline.producer_commit(ab_producer_state)
                    ab_producer_state.advance()

            # ------ MMA ------ #
            if warp_idx == self.mma_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                mma_pipeline.producer_acquire(mma_producer_state)

                for k in cutlass.range(cute.size(gA, mode=[2])):
                    ab_pipeline.consumer_wait(ab_consumer_state)

                    for kblock_idx in cutlass.range(cute.size(tCsA, mode=[2]), unroll_full=True):
                        cute.gemm(
                            tiled_mma,
                            cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                            tCsA[(None, None, kblock_idx, ab_consumer_state.index)],
                            tCsB[(None, None, kblock_idx, ab_consumer_state.index)],
                            cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()

                mma_pipeline.producer_commit(mma_producer_state)
                mma_producer_state.advance()

            # ------ EPI ------ #
            if warp_idx in self.epi_warp_ids:
                cute.arch.warpgroup_reg_alloc(self.num_regs_epi)

                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk,
                    utils.LayoutEnum.ROW_MAJOR,
                    self.acc_dtype,
                    self.acc_dtype,
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                    self.use_2cta_instrs,
                )
                # [tileM, subTileN, loopM, CntSubTileN, loopN]
                tAcc_epi = cute.flat_divide(
                    tCtC[((None, None), 0, None)],
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                )
                tiled_copy_t2r = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
                )
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTMEM_load_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
                tTMEM_load_tAcc = cute.group_modes(
                    tTMEM_load_tAcc, 3, cute.rank(tTMEM_load_tAcc) - 1
                )

                # predicates
                cAcc = cute.make_identity_tensor(self.mma_tiler[:2])
                tCcAcc = thr_mma.partition_C(cAcc)
                tCcAcc_epi = cute.flat_divide(
                    tCcAcc[((None, None), 0, None)],
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                )
                tTMEM_load_cAcc = thr_copy_t2r.partition_D(tCcAcc_epi)
                tTMEM_load_cAcc_shape = cute.select(tTMEM_load_cAcc.shape, mode=[0, 1, 2])
                tTMEM_load_rAcc = cute.make_fragment(tTMEM_load_cAcc_shape, self.acc_dtype)

                copy_atom_g2r_int64 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), mLabels.element_type
                )
                copy_atom_g2r_fp32 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), mDlogprobs.element_type
                )
                epilogue_thread_layout = cute.make_layout((128, 1), stride=(1, 1))
                tiled_copy_g2r_int64 = cute.make_tiled_copy_tv(
                    copy_atom_g2r_int64, epilogue_thread_layout, cute.make_layout((1, 1))
                )
                tiled_copy_g2r_fp32 = cute.make_tiled_copy_tv(
                    copy_atom_g2r_fp32, epilogue_thread_layout, cute.make_layout((1, 1))
                )
                thr_copy_g2r_int64 = tiled_copy_g2r_int64.get_slice(tidx)
                thr_copy_g2r_fp32 = tiled_copy_g2r_fp32.get_slice(tidx)

                # [tileM]
                gLabels = cute.local_tile(mLabels, (self.epi_tile[0],), (pidm,))
                gMaximum = cute.local_tile(mMaximum, (self.epi_tile[0],), (pidm,))
                gAccu = cute.local_tile(mAccu, (self.epi_tile[0],), (pidm,))

                # slice along M direction
                tMCAcc = thr_copy_g2r_int64.partition_S(cAcc)[(None, None, 0)]
                # [(1, 1), 1]
                tMCAcc_mask = cute.make_fragment(tMCAcc.shape, cutlass.Boolean)
                # to align shape with gMax and gAccu
                tMCAcc_mask = cute.append_ones(tMCAcc_mask)
                tMCAcc_mask[0] = cute.elem_less(
                    pidm * self.epi_tile[0] + tidx, cute.size(mA, mode=[0])
                )
                # [(1, 1), 1, 1]
                tMgLabels = thr_copy_g2r_int64.partition_S(cute.append_ones(gLabels))
                tMrLabels = cute.make_fragment(tMgLabels.shape, tMgLabels.element_type)
                cute.copy(tiled_copy_g2r_int64, tMgLabels, tMrLabels, pred=tMCAcc_mask)
                tMgMaximum = thr_copy_g2r_fp32.partition_S(cute.append_ones(gMaximum))
                tMrMaximum = cute.make_fragment(tMgMaximum.layout, tMgMaximum.element_type)
                cute.copy(tiled_copy_g2r_fp32, tMgMaximum, tMrMaximum, pred=tMCAcc_mask)
                tMgAccu = thr_copy_g2r_fp32.partition_S(cute.append_ones(gAccu))
                tMrAccu = cute.make_fragment(tMgAccu.layout, tMgAccu.element_type)
                cute.copy(tiled_copy_g2r_fp32, tMgAccu, tMrAccu, pred=tMCAcc_mask)

                tMrDlogprobs = cute.make_fragment(tMgAccu.layout, mDlogprobs.element_type)
                if cutlass.const_expr(self.REDUCTION == 2):
                    # mean reduction
                    num_valid_tokens = cute.make_tensor(scalarNumValidTokens, layout=(1,))
                    tMrDlogprobs[0] = mDlogprobs[0] / num_valid_tokens[0].to(cutlass.Float32)
                elif cutlass.const_expr(self.REDUCTION == 1):
                    # sum reduction
                    tMrDlogprobs[0] = mDlogprobs[0]
                else:
                    # no reduction
                    gDlogprobs = cute.local_tile(mDlogprobs, (self.epi_tile[0],), (pidm,))
                    tMgDlogprobs = thr_copy_g2r_fp32.partition_S(cute.append_ones(gDlogprobs))
                    cute.copy(tiled_copy_g2r_fp32, tMgDlogprobs, tMrDlogprobs, pred=tMCAcc_mask)

                tMrAccu[0] = cute.arch.rcp_approx(tMrAccu[0])
                tMrDlogprobs[0] *= tMrLabels[0] != ignore_index
                tMr_d_acc_exp_logits = tMrDlogprobs[0] * tMrAccu[0]

                # ------ Partial output ------ #
                # [tileM, tileN]
                gDlogits_partial = cute.local_tile(
                    mDlogits_partial, (self.epi_tile[0], self.epi_tile[1]), (pidm, pidn)
                )
                # blackwell supports STG.256
                copy_atom_r2g = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    gDlogits_partial.element_type,
                    num_bits_per_copy=256,
                )
                tiled_copy_r2g = cute.make_tiled_copy_tv(
                    copy_atom_r2g, epilogue_thread_layout, copy_atom_r2g.layout_dst_tv
                )
                thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

                # [CPY, loopM, loopN]
                tR2GCAcc = thr_copy_r2g.partition_S(cAcc)
                tR2GCAcc_pred = cute.make_fragment(tR2GCAcc.shape, cutlass.Boolean)
                for elem in cutlass.range(cute.size(tR2GCAcc_pred, mode=[0])):
                    for row in cutlass.range(cute.size(tR2GCAcc_pred, mode=[1])):
                        for col in cutlass.range(cute.size(tR2GCAcc_pred, mode=[2])):
                            tR2GCAcc_pred[elem, row, col] = cute.elem_less(
                                pidm * self.epi_tile[0] + tR2GCAcc[elem, row, col][0],
                                problem_mnk[0],
                            ) and cute.elem_less(
                                split_idx * self.vocab_per_split
                                + pidn * self.epi_tile[1]
                                + tR2GCAcc[elem, row, col][1],
                                problem_mnk[1],
                            )

                tR2GgDlogits = thr_copy_r2g.partition_D(gDlogits_partial)

                # for type conversion
                dLogits_half = cute.make_fragment(tTMEM_load_rAcc.shape, tR2GgDlogits.element_type)
                dLogits_half = cute.tiled_divide(
                    dLogits_half, (cute.size(tR2GgDlogits, mode=[0]), 1)
                )
                dLogits_half = cute.group_modes(dLogits_half, 2, cute.rank(dLogits_half))

                mma_pipeline.consumer_wait(mma_consumer_state)

                block_vocab_left_idx: cutlass.Int64 = (
                    split_idx * self.vocab_per_split + pidn * self.epi_tile[1]
                )
                block_vocab_right_idx: cutlass.Int64 = min(
                    split_idx * self.vocab_per_split + (pidn + 1) * self.epi_tile[1],
                    min((split_idx + 1) * self.vocab_per_split, problem_mnk[1]),
                )
                num_n_subtiles: cutlass.Int64 = cute.ceil_div(
                    (block_vocab_right_idx - block_vocab_left_idx),
                    cute.size(tTMEM_load_rAcc, mode=[0]),
                )
                for n_subtile in cutlass.range(num_n_subtiles):
                    cute.copy(
                        tiled_copy_t2r,
                        tTMEM_load_tAcc[(None, None, None, n_subtile, mma_consumer_state.index)],
                        tTMEM_load_rAcc,
                    )

                    for idx in cutlass.range(
                        cute.size(tTMEM_load_rAcc, mode=[0]), unroll_full=True
                    ):
                        # exp_logits
                        tTMEM_load_rAcc[idx] = cute.exp(tTMEM_load_rAcc[idx] - tMrMaximum[0])

                        position: cutlass.Int64 = (
                            rank * problem_mnk[1]
                            + split_idx * self.vocab_per_split
                            + pidn * self.epi_tile[1]
                            + n_subtile * cute.size(tTMEM_load_rAcc, mode=[0])
                            + idx
                        )
                        mask: cutlass.Boolean = (
                            position == tMrLabels[0] and tMrLabels[0] != ignore_index
                        )
                        # d_logits
                        tTMEM_load_rAcc[idx] *= tMr_d_acc_exp_logits
                        tTMEM_load_rAcc[idx] += mask * -tMrDlogprobs[0]
                        dLogits_half[idx] = tTMEM_load_rAcc[idx].to(dLogits_half.element_type)

                    for idx in cutlass.range(cute.size(dLogits_half, mode=[1]), unroll_full=True):
                        copy_id = n_subtile * cute.size(dLogits_half, mode=[1]) + idx
                        cute.copy(
                            tiled_copy_r2g,
                            dLogits_half[(None, idx, None)],
                            tR2GgDlogits[(None, None, copy_id)],
                            pred=tR2GCAcc_pred[((0, None), None, copy_id)],
                        )

                mma_pipeline.consumer_release(mma_consumer_state)
                mma_consumer_state.advance()

            # ------ Deallocate TMEM ------ #
            self.cta_sync_barrier.arrive_and_wait()
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(
                    tmem_ptr, self.tmem_alloc_cols, is_two_cta=self.use_2cta_instrs
                )

        @cute.jit
        def __call__(
            self,
            split_idx: cutlass.Int32,
            hidden: cute.Tensor,
            weight: cute.Tensor,
            labels: cute.Tensor,
            dlogprobs: cute.Tensor,
            maximum: cute.Tensor,
            accu: cute.Tensor,
            dlogits_partial: cute.Tensor,
            scalarNumValidTokens: cute.Pointer,
            ignore_index: cutlass.Int64,
            rank: cutlass.Int32,
            stream: cuda.CUstream,
        ) -> None:
            a_dtype: Type[cutlass.Numeric] = hidden.element_type
            b_dtype: Type[cutlass.Numeric] = weight.element_type

            if cutlass.const_expr(hidden.element_type != weight.element_type):
                raise RuntimeError(
                    f"data type don't match: {hidden.element_type} v.s. {weight.element_type}"
                )
            if cutlass.const_expr(hidden.element_type not in [cutlass.Float16, cutlass.BFloat16]):
                raise RuntimeError("hidden can only be FP16 or BF16")
            if cutlass.const_expr(hidden.layout.shape[1] != weight.layout.shape[1]):
                raise RuntimeError("K dimension doesn't match")

            problem_mnk = (hidden.layout.shape[0], weight.layout.shape[0], hidden.layout.shape[1])
            if cutlass.const_expr((problem_mnk[2] * a_dtype.width // 8) % 16 != 0):
                raise RuntimeError(f"K dimension is not 16B aligned: {problem_mnk[2]}")
            if cutlass.const_expr((problem_mnk[2] * b_dtype.width // 8) % 128 != 0):
                raise RuntimeError(f"N dimension is not 128B aligned: {problem_mnk[1]}")

            grid = self._compute_grid(
                problem_mnk=problem_mnk,
                cluster_shape_mn=self.cluster_shape_mn,
                cta_tiler=self.mma_tiler,
            )

            a_major_mode = utils.LayoutEnum.from_tensor(hidden).mma_major_mode()
            b_major_mode = utils.LayoutEnum.from_tensor(weight).mma_major_mode()

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                a_dtype,
                a_major_mode,
                b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler[:2],
            )
            self._setup_attributes(tiled_mma, a_dtype, b_dtype)

            self.epi_tile = self.cta_tile_shape_mnk[:2]

            # Swizzle o [(tileM, tileK), loopM, loopK, stage]
            a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, a_dtype, self.num_ab_stage
            )
            # Swizzle o [(tileN, tileK), loopN, loopK, stage]
            b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, b_dtype, self.num_ab_stage
            )
            tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
            tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

            # Swizzle o [(tileM, tileK), loopM, loopK]
            a_smem_layout = cute.select(a_smem_layout_staged, mode=[0, 1, 2])
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                hidden,
                a_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            # Swizzle o [(tileN, tileK), loopN, loopK]
            b_smem_layout = cute.select(b_smem_layout_staged, mode=[0, 1, 2])
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                weight,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            a_copy_size = cute.size_in_bytes(a_dtype, a_smem_layout)
            b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
            self.tma_copy_ab_bytes = a_copy_size + b_copy_size

            @cute.struct
            class SharedStorage:
                """
                The shared storage for the backward kernel.
                """

                load_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
                mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]

                tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
                tmem_holding_buf: cutlass.Int32

                sA: cute.struct.Align[
                    cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]

            self.shared_storage = SharedStorage

            self.kernel(
                split_idx,
                tiled_mma,
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                labels,
                dlogprobs,
                maximum,
                accu,
                dlogits_partial,
                scalarNumValidTokens,
                ignore_index,
                a_smem_layout_staged,
                b_smem_layout_staged,
                self.cluster_layout_vmnk,
                problem_mnk,
                rank,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
            )

except ImportError:
    logging.warning("Cutlass or CUDA bindings not found. BwdPartialDlogits will not be available.")
