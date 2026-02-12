# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import logging
from typing import Tuple, Type

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.hopper_helpers as sm90_utils
    import math
    from megatron.core.fusions.linear_cross_entropy.hopper.utils import make_acc_tensor_mn_view
    
    class BwdPartialDlogits:
        """
        Hopper (SM90) implementation of backward kernel for partial d_logits.
        
        Computes: d_logits = (softmax(logits) - one_hot(labels)) * dlogprobs
        where logits = hidden @ weight.T
        
        Similar to Blackwell version but uses:
        - SM90 WGMMA instead of TCGEN05
        - Register accumulator instead of TMEM
        """
        
        def __init__(
            self,
            reduction: int,
            acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
            use_tma_multicast: bool = False,
            mma_tiler_mn: tuple = (128, 256),
            vocab_per_split: int = 512,
        ):
            self.REDUCTION: cutlass.Constexpr[cutlass.Int32] = cutlass.const_expr(reduction)
            self.acc_dtype = acc_dtype
            self.use_tma_multicast = use_tma_multicast
            # MMA tiler (K dimension will be set in _setup_attributes)
            self.mma_tiler = (*mma_tiler_mn, 1)
            self.vocab_per_split = vocab_per_split
            self.cta_tiler = (self.mma_tiler[0], vocab_per_split, self.mma_tiler[2])
            
            self.cluster_shape_mn = (2, 1) if self.use_tma_multicast else (1, 1)
            
            self.occupancy = 1
            self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
            
            self.threads_per_warp = 32
            # For large tile, use two warp groups
            self.atom_layout_mnk = (
                (2, 1, 1) if self.mma_tiler[0] > 64 and self.mma_tiler[1] > 128 else (1, 1, 1)
            )
            
            self.mma_warp_groups = math.prod(self.atom_layout_mnk)
            self.num_threads_per_warp_group = 128
            self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
            
            self.buffer_align_bytes = 1024

        def _compute_grid(
            self,
            problem_mnk: Tuple[int, int, int],
            cluster_shape_mn: Tuple[int, int],
            cta_tiler: Tuple[int, int, int],
        ) -> Tuple[int, int, int]:
            cluster_shape = (*cluster_shape_mn, 1)
            
            grid = cute.round_up(
                (
                    cute.ceil_div(problem_mnk[0], cta_tiler[0]),
                    cute.ceil_div(self.vocab_per_split, cta_tiler[1]),
                    1,
                ),
                cluster_shape,
            )
            return grid

        def _compute_stages(
            self,
            tile_shape_mnk: tuple,
            a_dtype: Type[cutlass.Numeric],
            b_dtype: Type[cutlass.Numeric],
            smem_capacity: int,
            occupancy: int,
        ) -> int:
            epi_stage = 4
            epi_bytes = 0 # epi_smem will reuse smem ab.

            a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
            b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
            
            ab_bytes_per_stage = (
                cute.size(a_shape) * a_dtype.width // 8
                + cute.size(b_shape) * b_dtype.width // 8
            )
            
            mbar_helpers_bytes = 1024
            
            ab_stage = (
                smem_capacity // occupancy - mbar_helpers_bytes
            ) // ab_bytes_per_stage
            
            return ab_stage, epi_stage


        def _setup_attributes(
            self,
            tiled_mma: cute.TiledMma,
            a_dtype: Type[cutlass.Numeric],
            b_dtype: Type[cutlass.Numeric],
        ):
            self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)

            # this is fixed for dense MMA, k=16
            mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])

            # 16*4 = 64; 64 * sizeof(FP16) = 128Bytes
            mma_inst_tile_k = 4
            self.mma_tiler = (
                self.mma_tiler[0],
                self.mma_tiler[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )
            
            # Cluster layout
            self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
            self.num_mcast_ctas_a = self.cluster_shape_mn[1]
            self.num_mcast_ctas_b = self.cluster_shape_mn[0]
            self.is_a_mcast = self.num_mcast_ctas_a > 1
            self.is_b_mcast = self.num_mcast_ctas_b > 1
            
            is_cooperative = self.atom_layout_mnk == (2, 1, 1)
            self.epi_tile = sm90_utils.compute_tile_shape_or_override(
                self.mma_tiler, self.acc_dtype, is_cooperative=is_cooperative
            )
        
            # Compute pipeline stages
            self.ab_stage, self.epi_stage = self._compute_stages(
                self.mma_tiler,
                a_dtype,
                b_dtype,
                self.smem_capacity,
                self.occupancy,
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
            tma_atom_c: cute.CopyAtom,
            mC: cute.Tensor,
            mLabels: cute.Tensor,
            mDlogprobs: cute.Tensor,
            mMaximum: cute.Tensor,
            mAccu: cute.Tensor,
            scalarNumValidTokens: cute.Pointer,
            ignore_index: cutlass.Int64,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            epi_smem_layout_staged: cute.ComposedLayout,
            cta_layout_mnk: cute.Layout,
            problem_mnk: tuple,
            rank: cutlass.Int32,
        ) -> None:
            """
            The backward kernel for partial d_logits.
            """
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            cidx, cidy, _ = cute.arch.cluster_idx()
            cdimx, cdimy, _ = cute.arch.cluster_dim()
            
            # Prefetch TMA descriptors
            if warp_idx == 0:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            
            # CTA Swizzle for L2 reuse
            cluster_id = cidx + cdimx * cidy 
            group_size_m = 8
            
            s_shape = ((group_size_m, cdimx // group_size_m), cdimy) 
            s_stride = ((1, cdimy * group_size_m), group_size_m)
            s_layout = cute.make_layout(s_shape, stride=s_stride)

            num_reg_cids = cute.size(s_shape)
            cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)

            if cluster_id >= num_reg_cids:
                tail_size_m = cdimx % group_size_m
                tail_layout = cute.make_layout(
                    (tail_size_m, cdimy), stride=(1, tail_size_m)
                )
                tail_cid = cluster_id - num_reg_cids    
                tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
                cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
                cid_n = tail_cid_n

            # Get pid from cluster id
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            pid_m = cid_m * self.cluster_shape_mn[0] + bidx_in_cluster[0]
            pid_n = cid_n * self.cluster_shape_mn[1] + bidx_in_cluster[1]

            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )

            cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
            
            # Get multicast mask
            a_mcast_mask = cute.make_layout_image_mask(
                cta_layout_mnk, cluster_coord_mnk, mode=1
            )
            b_mcast_mask = cute.make_layout_image_mask(
                cta_layout_mnk, cluster_coord_mnk, mode=0
            )
            a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
            b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
            
            # Allocate shared memory
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            
            # Setup pipeline
            mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
            
            mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
            num_warps = self.threads_per_cta // 32
            consumer_arrive_cnt = mcast_size * num_warps
            mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, consumer_arrive_cnt
            )
            
            cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
            mainloop_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=mainloop_pipeline_array_ptr,
                num_stages=self.ab_stage,
                producer_group=mainloop_pipeline_producer_group,
                consumer_group=mainloop_pipeline_consumer_group,
                tx_count=self.tma_copy_bytes_a + self.tma_copy_bytes_b,
                cta_layout_vmnk=cta_layout_vmnk,
            )
            
            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            # -------- GMEM partition -------- #
            gA = cute.local_tile(mA, (self.mma_tiler[0], self.mma_tiler[2]), (pid_m, None))

            mB_split = cute.local_tile(mB, (self.vocab_per_split, cute.size(mB.layout.shape, mode=[1])), (split_idx, 0))

            gB = cute.local_tile(mB_split, (self.mma_tiler[1], self.mma_tiler[2]), (pid_n, None)) 

            gC = cute.local_tile(mC, (self.mma_tiler[0], self.mma_tiler[1]), (pid_m, pid_n))

            # -------- SMEM partition ------------ #
            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            sC_ptr = cute.recast_ptr(
                sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
            )
            sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)

            warp_group_idx = cute.arch.make_warp_uniform(
                tidx // self.num_threads_per_warp_group
            )
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups, stride=self.num_threads_per_warp_group
            )

            # Warpgroup-level MMA for TMA
            wg_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

            # Make fragments for SMEM -> Register
            tCsA = wg_mma.partition_A(sA)
            tCsB = wg_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)  

            thr_mma = tiled_mma.get_slice(tidx)
            tCcAcc = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler[:2]))
            accumulators = cute.make_rmem_tensor(tCcAcc.shape, self.acc_dtype)

            a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
            tTMAsA, tTMAgA = cute.nvgpu.cpasync.tma_partition(
                tma_atom_a,
                cluster_coord_mnk[1],
                a_cta_layout,
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2),
            )

            b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
            tTMAsB, tTMAgB = cute.nvgpu.cpasync.tma_partition(
                tma_atom_b,
                cluster_coord_mnk[0],
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB, 0, 2),
            )

            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_wait()
            else:
                cute.arch.sync_threads()

            # ///////
            # tma prefetch
            # ///////
            k_tile_cnt = cute.size(gA, mode=[2])
            prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0) 
            if warp_idx == 0:
                for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state) 
                    cute.copy(
                        tma_atom_a,
                        tTMAgA[(None, mainloop_producer_state.count)],
                        tTMAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tTMAgB[(None, mainloop_producer_state.count)],
                        tTMAsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

            tCcAcc_mn = make_acc_tensor_mn_view(tCcAcc)
            acc_mn_view = make_acc_tensor_mn_view(accumulators)

            num_rows = cute.size(tCcAcc_mn, mode=[0])
            num_cols = cute.size(tCcAcc_mn, mode=[1])

            gMaximum = cute.local_tile(mMaximum, (self.mma_tiler[0],), (pid_m,))
            gAccu = cute.local_tile(mAccu, (self.mma_tiler[0],), (pid_m,))
            gLabels = cute.local_tile(mLabels, (self.mma_tiler[0],), (pid_m,))
            gDlogprobs = cute.local_tile(mDlogprobs, (self.mma_tiler[0],), (pid_m,))


            # ///////
            # prologue mma
            # ///////
            k_pipe_mmas = 1
                    
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            num_k_blocks = cute.size(tCrA, mode=[2])

            for k_tile in cutlass.range_constexpr(k_pipe_mmas):
                # Wait for A/B buffer to be ready
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_read_state, peek_ab_full_status
                )

                cute.nvgpu.warpgroup.fence()
                for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_block_coord = (
                        None,
                        None,
                        k_block_idx,
                        mainloop_consumer_read_state.index,
                    )
                    tCrA_1phase = tCrA[k_block_coord]
                    tCrB_1phase = tCrB[k_block_coord]

                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA_1phase,
                        tCrB_1phase,
                        accumulators,
                    )
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                cute.nvgpu.warpgroup.commit_group()
                mainloop_consumer_read_state.advance()
                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_read_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_read_state
                    )

            # ///////
            # main mma
            # ///////
            for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1,unroll=1):
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_read_state, peek_ab_full_status
                )
                cute.nvgpu.warpgroup.fence()
                for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_block_coord = (
                        None,
                        None,
                        k_block_idx,
                        mainloop_consumer_read_state.index,
                    )
                    tCrA_1phase = tCrA[k_block_coord]
                    tCrB_1phase = tCrB[k_block_coord]
                    
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA_1phase,
                        tCrB_1phase,
                        accumulators,
                    )
                
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                
                mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                
                mainloop_consumer_read_state.advance()
                mainloop_consumer_release_state.advance()
                
                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_read_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_read_state
                    )

                if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)                    
                
                    cute.copy(
                        tma_atom_a,
                        tTMAgA[(None, mainloop_producer_state.count)],
                        tTMAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tTMAgB[(None, mainloop_producer_state.count)],
                        tTMAsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=b_mcast_mask,
                    )

                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
            
            # Synchronization
            cute.nvgpu.warpgroup.wait_group(0)

            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            else:
                cute.arch.sync_threads()


            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )

            copy_atom_C = cute.make_copy_atom( # 
                cute.nvgpu.warp.StMatrix8x8x16bOp( #stmatrix
                    self.c_layout.is_m_major_c(),
                    4,
                ),
                self.c_dtype,
            )

            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s,
                tiled_copy_C_Atom,
            )

            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
            tCgC_for_tma_partition = cute.zipped_divide(gC, self.epi_tile)
            bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                tma_atom_c,
                0,
                cute.make_layout(1),
                sepi_for_tma_partition,
                tCgC_for_tma_partition,
            )
                
            epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
            epi_tile_shape = tCgC_for_tma_partition.shape[1]
            epi_tile_layout = cute.make_layout(
                epi_tile_shape, stride=(epi_tile_shape[1], 1)
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=c_producer_group,
            )

            # Local vocab offset within this rank's weight shard
            local_vocab_offset = split_idx * self.vocab_per_split
            # Global vocab offset for TP: rank * local_vocab_size + local offset
            global_vocab_offset = rank * problem_mnk[1] + local_vocab_offset
            
            for r in cutlass.range(num_rows, unroll_full=True):
                coord_m = tCcAcc_mn[r, 0][0]
                coord_m_global: cutlass.Int32 = pid_m * self.mma_tiler[0] + coord_m
                
                label = gLabels[coord_m]
                maximum = gMaximum[coord_m]
                accu_inv = cute.arch.rcp_approx(gAccu[coord_m])
                
                dlogprobs = cutlass.Float32.zero
                
                if cutlass.const_expr(self.REDUCTION == 2):
                    # mean reduction
                    num_valid_tokens_tensor = cute.make_tensor(scalarNumValidTokens, layout=(1,))
                    dlogprobs = mDlogprobs[0] / num_valid_tokens_tensor[0].to(cutlass.Float32)
                elif cutlass.const_expr(self.REDUCTION == 1):
                    # sum reduction
                    dlogprobs = mDlogprobs[0]
                else:
                    # no reduction: per-token
                    dlogprobs = gDlogprobs[coord_m]
                
                d_over_accu = dlogprobs * accu_inv
                
                if coord_m_global < problem_mnk[0] and label != ignore_index:
                    # Valid token: compute gradient
                    for v in cutlass.range(num_cols, unroll_full=True):
                        coord_n = tCcAcc_mn[r, v][1]
                        # Use global position for label comparison (TP-aware)
                        vocab_position = global_vocab_offset + pid_n * self.mma_tiler[1] + coord_n
                        logit_val = acc_mn_view[r, v]
                    
                        exp_logit = cute.exp(logit_val - maximum)
                        acc_mn_view[r, v] = exp_logit * d_over_accu
                        if vocab_position == label:
                            acc_mn_view[r, v] -= dlogprobs
                else:
                    for v in cutlass.range(num_cols, unroll_full=True):
                        acc_mn_view[r, v] = cutlass.Float32.zero
                

            for epi_idx in cutlass.range_constexpr(epi_tile_num):
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
                acc_vec = tRS_rD.load()
                tRS_rD_out.store(acc_vec.to(self.c_dtype))

                # Copy from D registers to shared memory
                epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                cute.copy(
                    tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
                )

                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                # barrier for sync
                pipeline.sync(barrier_id=1)

                gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                # Copy from shared memory to global memory
                if warp_idx == 0:
                    cute.copy(
                        tma_atom_c,
                        bSG_sD[(None, epi_buffer)],
                        bSG_gD[(None, gmem_coord)],
                    )
                    c_pipeline.producer_commit()
                    c_pipeline.producer_acquire()

                pipeline.sync(barrier_id=1)

            if warp_idx == 0:
                c_pipeline.producer_tail()    

            return


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
            """Host function to launch the kernel."""
            a_dtype = hidden.element_type
            b_dtype = weight.element_type
            self.c_dtype = dlogits_partial.element_type
            self.c_layout = utils.LayoutEnum.from_tensor(dlogits_partial)
            
            if cutlass.const_expr(hidden.element_type != weight.element_type):
                raise RuntimeError(
                    f"Data type mismatch: {hidden.element_type} vs {weight.element_type}"
                )
            if cutlass.const_expr(hidden.element_type not in [cutlass.Float16, cutlass.BFloat16]):
                raise RuntimeError("Hidden must be FP16 or BF16")
            if cutlass.const_expr(hidden.layout.shape[1] != weight.layout.shape[1]):
                raise RuntimeError("K dimension doesn't match")
            
            problem_mnk = (hidden.layout.shape[0], weight.layout.shape[0], hidden.layout.shape[1])
            if cutlass.const_expr((problem_mnk[2] * a_dtype.width // 8) % 16 != 0):
                raise RuntimeError(f"K dimension not 16B aligned: {problem_mnk[2]}")
            
            grid = self._compute_grid(
                problem_mnk=problem_mnk,
                cluster_shape_mn=self.cluster_shape_mn,
                cta_tiler=self.mma_tiler,
            )
            
            a_major_mode = utils.LayoutEnum.from_tensor(hidden).sm90_mma_major_mode()
            b_major_mode = utils.LayoutEnum.from_tensor(weight).sm90_mma_major_mode()
            
            tiled_mma = sm90_utils.make_trivial_tiled_mma(
                a_dtype,
                b_dtype,
                a_major_mode,
                b_major_mode,
                self.acc_dtype,
                self.atom_layout_mnk,
                tiler_mn=(64, self.mma_tiler[1]),
            )
            self._setup_attributes(tiled_mma, a_dtype, b_dtype)
            
            # Create SMEM layouts
            a_smem_layout_staged = sm90_utils.make_smem_layout_a(
                utils.LayoutEnum.from_tensor(hidden),
                self.mma_tiler,
                a_dtype,
                self.ab_stage,
            )
            b_smem_layout_staged = sm90_utils.make_smem_layout_b(
                utils.LayoutEnum.from_tensor(weight),
                self.mma_tiler,
                b_dtype,
                self.ab_stage,
            )
                
            epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                self.epi_stage,
            )
            # TMA atoms            
            a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
            tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
                if self.is_a_mcast
                else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                hidden,
                a_smem_layout,
                (self.mma_tiler[0], self.mma_tiler[2]),
                num_multicast=self.num_mcast_ctas_a,
            )
            
            b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
            tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
                if self.is_b_mcast
                else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                weight,
                b_smem_layout,
                (self.mma_tiler[1], self.mma_tiler[2]),
                num_multicast=self.num_mcast_ctas_b,
            )
            epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
                dlogits_partial,
                epi_smem_layout,
                self.epi_tile,
            )

            a_copy_size = cute.size_in_bytes(a_dtype, a_smem_layout)
            b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
            self.tma_copy_bytes_a = a_copy_size
            self.tma_copy_bytes_b = b_copy_size
            
            @cute.struct
            class SharedStorage:
                """Shared storage for the backward kernel."""
                mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
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
                tma_atom_c,
                tma_tensor_c,
                labels,
                dlogprobs,
                maximum,
                accu,
                scalarNumValidTokens,
                ignore_index,
                a_smem_layout_staged,
                b_smem_layout_staged,
                epi_smem_layout_staged,
                self.cta_layout_mnk,
                problem_mnk,
                rank,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
            )


except ImportError:
    logging.warning(
        "PyTorch not found. Backward functions will not be available."
    )
