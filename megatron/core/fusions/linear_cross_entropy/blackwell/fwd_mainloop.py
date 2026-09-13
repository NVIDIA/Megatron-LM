# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Implementations of the fusion lm_head(Linear) + Cross-Entropy kernel
"""

import logging
from typing import Tuple, Type

try:
    import cuda.bindings.driver as cuda  # type: ignore
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline  # type: ignore
    import cutlass.utils as utils  # type: ignore
    import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
    from cutlass.cute.nvgpu import cpasync, tcgen05

    SM100_TMEM_CAPACITY_COLUMNS: int = 512

    def make_thread_cooperative_group(size: int):
        """
        Create a thread cooperative group.
        """
        return pipeline.CooperativeGroup(pipeline.Agent.Thread, size, alignment=size)

    class FwdMainLoop:
        """
        This class implements the mainloop for forward process.

        Traits stored as attributes.

        :param acc_dtype:
        """

        def __init__(
            self,
            acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
            use_2cta_instrs: bool = False,
            mma_tiler_mn: Tuple[int, int] = (128, 256),
            vocab_per_split: int = 512,
        ):
            """
            Configuration including:
                - MMA instruction settings
                - Cluster Shape
            """
            self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
            self.use_2cta_instrs = use_2cta_instrs
            # This is the shape covered by tiledMMA, not just single MMA instruction
            self.mma_tiler = (*mma_tiler_mn, 1)
            self.cta_tiler = (self.mma_tiler[0], vocab_per_split, self.mma_tiler[2])
            self.vocab_per_split = vocab_per_split

            self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)

            self.occupancy = 1
            # query SMEM capacity
            self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

            # the maximum columns per MMA is 256, and there is only one GEMM, so we can fully
            # assign TMEM for that GEMM of different tiles.
            # so 512 = 2 * 256

            self.threads_per_warp: int = 32
            # 1 warp for loading, 1 warp for issuing MMA, 1 WG for storing
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
            self.tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=2, num_threads=self.threads_per_cta
            )

            self.buffer_align_bytes: int = 1024
            self.num_regs_other: int = 32
            self.num_regs_epi: int = 192

        def _compute_stages(
            self,
            tiled_mma: cute.TiledMma,
            mma_tiler: Tuple[int, int, int],
            a_dtype: Type[cutlass.Numeric],
            b_dtype: Type[cutlass.Numeric],
        ):
            a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
                tiled_mma, mma_tiler, a_dtype, 1  # only single stage
            )
            b_smem_layout_stage_one = sm100_utils.make_smem_layout_b(
                tiled_mma, mma_tiler, b_dtype, 1
            )
            a_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            b_bytes_per_stage = cute.size_in_bytes(b_dtype, b_smem_layout_stage_one)
            num_acc_stage = 2
            num_a_stage = 4
            num_b_stage = 4
            num_epi_stage_per_tile = 4

            return num_acc_stage, num_a_stage, num_b_stage, num_epi_stage_per_tile

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

            # this is fixed for dense MMA, k=16
            mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            # 16*4 = 64; 64 * sizeof(FP16) = 128Bytes
            mma_inst_tile_k: int = 4
            self.mma_tiler = (
                self.mma_tiler[0],
                self.mma_tiler[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )

            self.num_acc_stage, self.num_a_stage, self.num_b_stage, self.num_epi_stage_per_tile = (
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
            tiled_mma: cute.TiledMma,
            tma_atom_a: cute.CopyAtom,
            mA: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB: cute.Tensor,
            mLabels: cute.Tensor,
            mMax: cute.Tensor,
            mAccu: cute.Tensor,
            mLogprobs: cute.Tensor,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            cluster_layout_vmnk: cute.Layout,
            problem_mnk: Tuple[int, int, int],
            ignore_index: cutlass.Int64,
            rank: cutlass.Int32,
        ):
            """
            The forward kernel for the mainloop.
            """
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            # FIXME: block swizzling applied here
            pidm, pidn = bidx, bidy

            # prefetch tma descriptors
            if warp_idx == self.load_warp_ids:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

            # declare SMEM
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)

            ab_pipeline = pipeline.PipelineTmaUmma.create(
                num_stages=self.num_a_stage,
                producer_group=make_thread_cooperative_group(len([self.load_warp_ids])),
                consumer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
                tx_count=self.tma_copy_a_bytes + self.tma_copy_b_bytes,
                barrier_storage=storage.load_ab_mbar_ptr.data_ptr(),
            )
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_a_stage
            )
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_a_stage
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

            # -------- SMEM partition ------------ #
            # swizzle o [(tileM, tileK), loopM, loopK, Stage]
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

            # ---------- GMEM partition ----------- #
            # [tileM, tileK, loopK]
            gA = cute.local_tile(mA, (self.mma_tiler[0], self.mma_tiler[2]), (pidm, None))

            # [vocab_size_per_split, dim]
            mB_n = cute.local_tile(
                mB, (self.vocab_per_split, cute.size(mB.layout.shape, mode=[1])), (pidn, 0)
            )

            # [tileN, tileK, loopN, loopK]
            gB = cute.local_tile(mB_n, (self.mma_tiler[1], self.mma_tiler[2]), (None, None))

            # [MMA, tileCntM, tileCntK, loopK]
            tCgA = thr_mma.partition_A(gA)
            # [MMA, tileCntN, tileCntK, loopN, loopK]
            tCgB = thr_mma.partition_B(gB)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            # FIXME: if 2 CTAs, modify here
            cta_rank_in_cluster = 0
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
            tTMAsA, tTMAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],  # cta_coord,
                a_cta_layout,
                cute.group_modes(sA, 0, 3),  # SMEM tensor
                cute.group_modes(tCgA, 0, 3),  # GMEM tensor
            )
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            tTMAsB, tTMAgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],  # cta_coord
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )

            # Allocate TMEM
            tmem_holding_buf = storage.tmem_holding_buf
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.alloc_tmem(
                    self.tmem_alloc_cols, tmem_holding_buf, is_two_cta=self.use_2cta_instrs
                )
            self.cta_sync_barrier.arrive_and_wait()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )

            # [(tileM, tileN), loopM, loopN]
            tmem_shape = (128, self.tmem_alloc_cols)
            acc_shape = thr_mma.partition_shape_C(tmem_shape)
            tCtC_fake = thr_mma.make_fragment_C(acc_shape)
            tCtC = cute.make_tensor(tmem_ptr, tCtC_fake.layout)

            block_vocab_left_idx: cutlass.Int64 = pidn * self.vocab_per_split
            block_vocab_right_idx: cutlass.Int64 = min(
                (pidn + 1) * self.vocab_per_split, problem_mnk[1]
            )
            num_n_tiles: cutlass.Int64 = cute.ceil_div(
                (block_vocab_right_idx - block_vocab_left_idx), self.mma_tiler[1]
            )

            # ///////
            # empty
            # ///////
            if warp_idx in self.empty_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # ///////
            # load
            # ///////
            if warp_idx == self.load_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                for n in cutlass.range(num_n_tiles):
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
                            tTMAgB[(None, n, k)],
                            tTMAsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        )
                        ab_pipeline.producer_commit(ab_producer_state)
                        ab_producer_state.advance()

            # ///////
            # mma
            # ///////
            if warp_idx == self.mma_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                for n in cutlass.range(num_n_tiles):
                    # disable accumulate for the first tile
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    mma_pipeline.producer_acquire(mma_producer_state)

                    for k in cutlass.range(cute.size(gA, mode=[2])):
                        ab_pipeline.consumer_wait(ab_consumer_state)

                        for kblock_idx in cutlass.range(
                            cute.size(tCsA, mode=[2]), unroll_full=True
                        ):
                            cute.gemm(
                                tiled_mma,
                                cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                                tCsA[(None, None, kblock_idx, ab_consumer_state.index)],
                                tCsB[(None, None, kblock_idx, ab_consumer_state.index)],
                                cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                            )
                            # enable accumulate for the next tile
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()

                    mma_pipeline.producer_commit(mma_producer_state)
                    mma_producer_state.advance()

            # //////////
            # epilogue
            # //////////
            if warp_idx in self.epi_warp_ids:
                cute.arch.warpgroup_reg_alloc(self.num_regs_epi)

                # epilog TMEM copy and partition
                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk,
                    utils.LayoutEnum.ROW_MAJOR,  # This is hard-coded
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
                # [(pattern), loopM, loopN, CntTileM, CntTileN]
                tTMEM_load_tAcc = cute.group_modes(
                    tTMEM_load_tAcc, 3, cute.rank(tTMEM_load_tAcc) - 1
                )

                cAcc = cute.make_identity_tensor(self.mma_tiler[:2])
                tCcAcc = thr_mma.partition_C(cAcc)
                # [tileM, subTileN, loopM, CntSubTileN, CntTileN]
                tCcAcc_epi = cute.flat_divide(
                    tCcAcc[((None, None), 0, None)],
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                )
                tTMEM_load_cAcc = thr_copy_t2r.partition_D(tCcAcc_epi)
                tTMEM_load_cAcc_shape = cute.select(tTMEM_load_cAcc.shape, mode=[0, 1, 2])

                # epilogue layouts
                epilogue_thread_layout = cute.make_layout((128, 1))
                copy_atom_g2r = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), mLabels.element_type
                )
                tiled_copy_g2r = cute.make_tiled_copy(
                    copy_atom_g2r, epilogue_thread_layout, (128, 1)
                )
                thr_copy_g2r = tiled_copy_g2r.get_slice(tidx)

                copy_atom_r2g = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
                tiled_copy_r2g = cute.make_tiled_copy(
                    copy_atom_r2g, epilogue_thread_layout, (128, 1)
                )
                thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

                # auxiliary tensors
                # [tileM]
                gLabels = cute.local_tile(mLabels, (self.epi_tile[0],), (pidm,))

                tLabelsCAcc = thr_copy_g2r.partition_S(cAcc)[(None, None, 0)]
                tLabelsCAcc_mask = cute.make_fragment(tLabelsCAcc.shape, cutlass.Boolean)
                # [(1, 1), 1]
                tLabelsCAcc_mask[0] = cute.elem_less(pidm * self.epi_tile[0] + tidx, problem_mnk[0])
                # to align shape with gMax and gAccu
                tLabelsCAcc_mask = cute.append_ones(tLabelsCAcc_mask)

                # [(1, 1), 1, 1]
                tLabelsgLabels = thr_copy_g2r.partition_S(cute.append_ones(gLabels))
                tLabelsrLabels = cute.make_fragment(
                    tLabelsgLabels.shape, tLabelsgLabels.element_type
                )
                cute.copy(tiled_copy_g2r, tLabelsgLabels, tLabelsrLabels, pred=tLabelsCAcc_mask)
                valid_mask: cutlass.Boolean = (
                    tLabelsrLabels[0] != ignore_index
                ) and tLabelsCAcc_mask[0]

                # [tileM, 1]
                gMax = cute.local_tile(mMax, (self.epi_tile[0], 1), (pidm, pidn))
                # [(CPYM, CPYN), loopM, loopN]
                tR2GgMax = thr_copy_r2g.partition_D(gMax)
                tR2GrMax = cute.make_fragment(tR2GgMax.shape, tR2GgMax.element_type)
                tR2GrMax.fill(-1e30)

                # [tileM, 1]
                gAccu = cute.local_tile(mAccu, (self.epi_tile[0], 1), (pidm, pidn))
                # [(CPYM, CPYN), loopM, loopN]
                tR2GgAccu = thr_copy_r2g.partition_D(gAccu)
                tR2GrAccu = cute.make_fragment(tR2GgAccu.shape, tR2GgAccu.element_type)
                tR2GrAccu.fill(0.0)

                # [tileM, 1]
                gLogprobs = cute.append_ones(
                    cute.local_tile(mLogprobs, (self.epi_tile[0],), (pidm,))
                )
                # [(CPYM, CPYN), loopM, loopN]
                tR2GgLogprobs = thr_copy_r2g.partition_D(gLogprobs)
                tR2GrLogprobs = cute.make_fragment(tR2GgLogprobs.shape, tR2GgLogprobs.element_type)
                tR2GrLogprobs.fill(0.0)

                # [(tileN // num_epi_stage_per_tile, 1), 1, 1]
                tTMEM_load_rAcc = cute.make_fragment(tTMEM_load_cAcc_shape, self.acc_dtype)

                for n in cutlass.range(num_n_tiles):
                    mma_pipeline.consumer_wait(mma_consumer_state)

                    left: cutlass.Int64 = block_vocab_left_idx + n * self.epi_tile[1]
                    right: cutlass.Int64 = min(
                        (n + 1) * self.epi_tile[1] + block_vocab_left_idx, block_vocab_right_idx
                    )
                    num_n_subtiles: cutlass.Int64 = cute.ceil_div(
                        (right - left), cute.size(tTMEM_load_rAcc, mode=[0])
                    )
                    for n_subtile in cutlass.range(num_n_subtiles):
                        cute.copy(
                            tiled_copy_t2r,
                            tTMEM_load_tAcc[
                                (None, None, None, n_subtile, mma_consumer_state.index)
                            ],
                            tTMEM_load_rAcc,
                        )

                        for idx in cutlass.range(
                            cute.size(tTMEM_load_rAcc, mode=[0]), unroll_full=True
                        ):
                            local_position: cutlass.Int64 = (
                                n * self.epi_tile[1]
                                + n_subtile * cute.size(tTMEM_load_rAcc, mode=[0])
                                + idx
                            )
                            if (block_vocab_left_idx + local_position) < block_vocab_right_idx:
                                _max_old = tR2GrMax[0]
                                tR2GrMax[0] = cute.arch.fmax(tR2GrMax[0], tTMEM_load_rAcc[idx])
                                exp_logits = cute.exp(tTMEM_load_rAcc[idx] - tR2GrMax[0])
                                coeff = cute.exp(_max_old - tR2GrMax[0])
                                tR2GrAccu[0] = coeff * tR2GrAccu[0] + exp_logits

                                position: cutlass.Int64 = (
                                    rank * problem_mnk[1]
                                    + pidn * self.vocab_per_split
                                    + local_position
                                )
                                mask: cutlass.Boolean = valid_mask and (
                                    position == tLabelsrLabels[0]
                                )
                                tR2GrLogprobs[0] += mask * tTMEM_load_rAcc[idx]

                    mma_pipeline.consumer_release(mma_consumer_state)
                    mma_consumer_state.advance()

                cute.copy(tiled_copy_r2g, tR2GrMax, tR2GgMax, pred=tLabelsCAcc_mask)
                cute.copy(tiled_copy_r2g, tR2GrAccu, tR2GgAccu, pred=tLabelsCAcc_mask)

                vocab_left_idx: cutlass.Int64 = rank * problem_mnk[1] + pidn * self.vocab_per_split
                vocab_right_idx: cutlass.Int64 = rank * problem_mnk[1] + min(
                    (pidn + 1) * self.vocab_per_split, problem_mnk[1]
                )
                valid: cutlass.Boolean = (
                    tLabelsrLabels[0] >= vocab_left_idx and tLabelsrLabels[0] < vocab_right_idx
                )
                tLabelsCAcc_mask[0] &= valid

                cute.copy(tiled_copy_r2g, tR2GrLogprobs, tR2GgLogprobs, pred=tLabelsCAcc_mask)

            # Dealloc TMEM
            self.cta_sync_barrier.arrive_and_wait()
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(
                    tmem_ptr, self.tmem_alloc_cols, is_two_cta=self.use_2cta_instrs
                )

        @staticmethod
        def _compute_grid(
            problem_mnk: Tuple[int, int, int],
            cluster_shape_mn: Tuple[int, int],
            cta_tiler: Tuple[int, int, int],
            num_splits: int,
        ) -> Tuple[int, int, int]:

            cluster_shape = (*cluster_shape_mn, 1)

            grid = cute.round_up(
                (cute.ceil_div(problem_mnk[0], cta_tiler[0]), num_splits, 1), cluster_shape
            )
            return grid

        @cute.jit
        def __call__(
            self,
            hidden: cute.Tensor,
            weight: cute.Tensor,
            labels: cute.Tensor,
            _logprobs: cute.Tensor,
            _max: cute.Tensor,
            _accu: cute.Tensor,
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

            num_splits = cute.ceil_div(problem_mnk[1], self.vocab_per_split)

            grid = self._compute_grid(
                problem_mnk=problem_mnk,
                cluster_shape_mn=self.cluster_shape_mn,
                cta_tiler=self.cta_tiler,
                num_splits=num_splits,
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
            if cutlass.const_expr((problem_mnk[2] * a_dtype.width // 8) % 128 != 0):
                raise RuntimeError(f"K dimension is not 128B aligned: {problem_mnk[2]}")

            self.epi_tile = self.mma_tiler[:2]

            # Swizzle o [(tileM, tileK), loopM, loopK, stage]
            a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, a_dtype, self.num_a_stage
            )
            # Swizzle o [(tileN, tileK), loopN, loopK, stage]
            b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, b_dtype, self.num_b_stage
            )

            # TMA loading
            tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
            tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

            # Swizzle o [(tileM, tileK), loopM, loopK]
            a_smem_layout = cute.select(a_smem_layout_staged, mode=[0, 1, 2])
            # create tma copy atom for hidden,
            # and the cooresponding tma descriptor tensor
            tma_atom_a, tma_desc_a = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                hidden,  # gmem_tensor
                a_smem_layout,  # SMEM layout
                self.mma_tiler,  # MMA tiler
                tiled_mma,  # TiledMMA
                self.cluster_layout_vmnk.shape,  # cluster_shape_vmnk
            )
            # Swizzle o [(tileN, tileK), loopN, loopK]
            b_smem_layout = cute.select(b_smem_layout_staged, mode=[0, 1, 2])
            tma_atom_b, tma_desc_b = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                weight,  # gmem_tensor
                b_smem_layout,  # SMEM layout
                self.mma_tiler,  # MMA tiler
                tiled_mma,  # TiledMMA
                self.cluster_layout_vmnk.shape,  # cluster_shape_vmnk
            )
            a_copy_size = cute.size_in_bytes(a_dtype, a_smem_layout)
            b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
            self.tma_copy_a_bytes = a_copy_size
            self.tma_copy_b_bytes = b_copy_size

            assert self.num_a_stage == self.num_b_stage

            @cute.struct
            class SharedStorage:
                """
                The shared storage for the forward kernel.
                """

                # pipeline barriers, 2 = producer + consumer
                load_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage * 2]
                mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
                tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
                # tmem holding buffer
                tmem_holding_buf: cutlass.Int32
                # SMEM tensors
                sA: cute.struct.Align[
                    cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]

            self.shared_storage = SharedStorage

            # launch kernel
            self.kernel(
                tiled_mma,
                tma_atom_a,
                tma_desc_a,
                tma_atom_b,
                tma_desc_b,
                labels,
                _max,
                _accu,
                _logprobs,
                a_smem_layout_staged,
                b_smem_layout_staged,
                self.cluster_layout_vmnk,
                problem_mnk,
                ignore_index,
                rank,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
            )
            return None

except ImportError:
    logging.warning("Cutlass or CUDA Python bindings not found. FwdMainLoop will not be available.")
