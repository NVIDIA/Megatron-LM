# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Implementations of the fusion lm_head(Linear) + Cross-Entropy kernel
"""

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
    import os
    import sys
    sys.path.append("/nfsdata/languageAI/users/hoyoun/fused_linear_cross")
    from megatron.core.fusions.linear_cross_entropy.hopper.utils import OnlineCrossEntropy, make_acc_tensor_mn_view
    
    class FwdMainLoop:
        """
        This class implements the mainloop for forward process.
        """
        def __init__(
            self,
            acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
            use_tma_multicast: bool = True,
            mma_tiler_mn: Tuple[int, int] = (128, 256),
            vocab_per_split: int = 512,
        ):
            self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
            self.use_tma_multicast = use_tma_multicast
            # MMA tiler (K dimension will be set in _setup_attributes)
            self.mma_tiler = (*mma_tiler_mn, 1)
            self.cta_tiler = (self.mma_tiler[0], vocab_per_split, self.mma_tiler[2])
            self.vocab_per_split = vocab_per_split
            
            self.cluster_shape_mn = (2, 1) if self.use_tma_multicast else (1, 1)
            
            self.occupancy = 1
            self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
            
            self.threads_per_warp = 32
            # For large tile size, using two warp groups
            self.atom_layout_mnk = (
                (2, 1, 1)
                if self.mma_tiler[0] > 64 and self.mma_tiler[1] > 128
                else (1, 1, 1)
            )

            self.mma_warp_groups = math.prod(self.atom_layout_mnk)
            self.num_threads_per_warp_group = 128
            self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
            
            self.buffer_align_bytes: int = 1024

        def _compute_stages(
            self,
            tile_shape_mnk: Tuple[int, int, int],
            a_dtype: Type[cutlass.Numeric],
            b_dtype: Type[cutlass.Numeric],
            smem_capacity: int,
            occupancy: int,
        ) -> int:
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
            
            return ab_stage


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
            
            # Compute pipeline stages
            self.ab_stage = self._compute_stages(
                self.mma_tiler,
                a_dtype,
                b_dtype,
                self.smem_capacity,
                self.occupancy,
            )

        @cute.kernel
        def kernel(
            self,
            tma_atom_a: cute.CopyAtom,
            mA: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB: cute.Tensor,
            mLabels: cute.Tensor,
            mMax: cute.Tensor,
            mAccu: cute.Tensor,
            mLogprobs: cute.Tensor,
            tiled_mma: cute.TiledMma,
            cta_layout_mnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            problem_mnk: Tuple[int, int, int],
            ignore_index: cutlass.Int64,
            rank: cutlass.Int32,
        ):
            """
            The forward kernel for the mainloop.
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

            # -------- SMEM partition ------------ #
            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )

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
        
            # -------- GMEM partition -------- #
            gA = cute.local_tile(mA, (self.mma_tiler[0], self.mma_tiler[2]), (pid_m, None))

            mB_n = cute.local_tile(mB, (self.vocab_per_split, cute.size(mB.layout.shape, mode=[1])), (pid_n, 0))

            gB = cute.local_tile(mB_n, (self.mma_tiler[1], self.mma_tiler[2]), (None, None)) 

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

            k_tile_cnt = cute.size(gA, mode=[2])
            prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0) 

            block_vocab_left_idx: cutlass.Int32 = pid_n * self.vocab_per_split
            block_vocab_right_idx: cutlass.Int32 = min(
                (pid_n + 1) * self.vocab_per_split, problem_mnk[1]
            )
            num_n_tiles: cutlass.Int32 = cute.ceil_div(
                (block_vocab_right_idx - block_vocab_left_idx), self.mma_tiler[1]
            )

            # ///////
            # tma prefetch
            # ///////
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
                        tTMAgB[(None, 0, mainloop_producer_state.count)],
                        tTMAsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                    
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < num_n_tiles * k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

            # -------- Online Cross Entropy Partition-------- #
            
            # MMA and Accumulator Setup
            thr_mma = tiled_mma.get_slice(tidx)
            tCcAcc = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler[:2]))
            tCcAcc_mn = make_acc_tensor_mn_view(tCcAcc)
            accumulators = cute.make_rmem_tensor(tCcAcc.shape, self.acc_dtype)
            acc_mn_view = make_acc_tensor_mn_view(accumulators)
                
            # Compute Loop Dimensions
            num_rows = cute.size(tCcAcc_mn, mode=[0])
            num_cols = cute.size(tCcAcc_mn, mode=[1])
            num_k_blocks = cute.size(tCrA, mode=[2])
            
            # Initialize Online Cross-Entropy Computation
            online_ce = OnlineCrossEntropy.create(num_rows, num_cols)
            online_ce.reset()

            # Partition Global Memory Tiles
            gMax = cute.local_tile(mMax, (self.mma_tiler[0], 1), (pid_m, pid_n))
            gAccu = cute.local_tile(mAccu, (self.mma_tiler[0], 1), (pid_m, pid_n))
            gLogprobs = cute.local_tile(mLogprobs, (self.mma_tiler[0],), (pid_m,))
            gLabels = cute.local_tile(mLabels, (self.mma_tiler[0],), (pid_m,))
            
            # Copy Labels to Thread-Local Memory
            labels_per_thread = cute.make_rmem_tensor((num_rows,), mLabels.element_type)
            for r in cutlass.range(num_rows, unroll_full=True):
                coord_m = tCcAcc_mn[r, 0][0]
                labels_per_thread[r] = gLabels[coord_m]
            
            # M-dimension row offset for this block
            row_block_start: cutlass.Int32 = pid_m * self.mma_tiler[0]
            # Global vocab offset for TP: rank * local_vocab_size
            # For DP (rank=0), this is 0 and has no effect
            vocab_offset: cutlass.Int32 = rank * problem_mnk[1]
                
            # ///////
            # mma
            # ///////
            for n in cutlass.range(num_n_tiles):
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for k_tile_idx in cutlass.range(k_tile_cnt, unroll=1):
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
                    cute.nvgpu.warpgroup.wait_group(0)
                    
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    
                    mainloop_consumer_read_state.advance()
                    mainloop_consumer_release_state.advance()
                    
                    peek_ab_full_status = cutlass.Boolean(1)
                    if mainloop_consumer_read_state.count < num_n_tiles * k_tile_cnt:
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            mainloop_consumer_read_state
                        )
                    if warp_idx == 0 and mainloop_producer_state.count < num_n_tiles * k_tile_cnt:
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)

                        n_to_load = mainloop_producer_state.count // k_tile_cnt
                        k_to_load = mainloop_producer_state.count % k_tile_cnt
                        
                        cute.copy(
                            tma_atom_a,
                            tTMAgA[(None, k_to_load)],
                            tTMAsA[(None, mainloop_producer_state.index)],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tTMAgB[(None, n_to_load, k_to_load)],
                            tTMAsB[(None, mainloop_producer_state.index)],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                            mcast_mask=b_mcast_mask,
                        )

                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()
                
                cute.arch.sync_threads()

                # //////////
                # epilogue
                # //////////
                vocab_tile_start: cutlass.Int32 = block_vocab_left_idx + n * self.mma_tiler[1]
                vocab_tile_size: cutlass.Int32 = block_vocab_right_idx - vocab_tile_start
                
                online_ce.update(
                    acc_mn_view, 
                    labels_per_thread, 
                    tCcAcc_mn, 
                    gLogprobs,
                    row_block_start, 
                    problem_mnk[0],
                    vocab_tile_start, 
                    vocab_tile_size, 
                    block_vocab_left_idx,
                    vocab_offset,
                    ignore_index, 
                    is_first=(n == 0)
                )

            # check both block validity and edge tile boundaries
            num_m_blocks: cutlass.Int32 = cute.ceil_div(problem_mnk[0], self.mma_tiler[0])
            block_is_valid: cutlass.Boolean = pid_m < num_m_blocks
            
            if tidx % 4 == 0 and block_is_valid:
                for r in cutlass.range(num_rows, unroll_full=True):
                    coord_m = tCcAcc_mn[r, 0][0]
                    coord_m_global: cutlass.Int32 = pid_m * self.mma_tiler[0] + coord_m
                    if coord_m_global < problem_mnk[0]:
                        gMax[coord_m, 0] = online_ce.row_max[r]
                        gAccu[coord_m, 0] = online_ce.row_sum[r]
            
            # Synchronization
            cute.nvgpu.warpgroup.wait_group(0)

            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            else:
                cute.arch.sync_threads()
            return

        @staticmethod
        def _compute_grid(
            problem_mnk: Tuple[int, int, int],
            cluster_shape_mn: Tuple[int, int],
            cta_tiler: Tuple[int, int, int],
            num_splits: int,
        ) -> Tuple[int, int, int]:
            """
            Compute grid dimensions for kernel launch.
            """
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
            if cutlass.const_expr((problem_mnk[2] * a_dtype.width // 8) % 128 != 0):
                raise RuntimeError(f"K dimension is not 128B aligned: {problem_mnk[2]}")

            # Create shared memory layouts
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

            # Create TMA atoms
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
            self.tma_copy_bytes_a = cute.size_in_bytes(a_dtype, a_smem_layout)
            self.tma_copy_bytes_b = cute.size_in_bytes(b_dtype, b_smem_layout)

            # Define shared storage
            @cute.struct
            class SharedStorage:
                mainloop_pipeline_array_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.ab_stage * 2
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[
                        a_dtype, cute.cosize(a_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[
                        b_dtype, cute.cosize(b_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
            
            self.shared_storage = SharedStorage
            
            # Launch kernel
            self.kernel(
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                labels,
                _max,
                _accu,
                _logprobs,
                tiled_mma,
                self.cta_layout_mnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
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
    logging.warning(
        "Cutlass or CUDA Python bindings not found. "
        "FwdMainLoop (Hopper) will not be available."
    )

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    
    print("=" * 80)
    print("Testing Hopper FwdMainLoop with TP")
    print("=" * 80)
    
    # Test configuration
    num_tokens = 80
    vocab_size_total = 152064
    dim = 64
    dtype = torch.bfloat16
    ignore_index = -100
    tp_world_size = 2
    vocab_size_per_rank = vocab_size_total // tp_world_size
    
    torch.manual_seed(42)
    
    # Create test data
    hidden = torch.randn(num_tokens, dim, dtype=dtype, device='cuda')
    weight_full = torch.randn(vocab_size_total, dim, dtype=dtype, device='cuda')
    labels = torch.randint(0, vocab_size_total, (num_tokens,), dtype=torch.long, device='cuda')
    
    print(f"\nTest setup:")
    print(f"  Tokens: {num_tokens}")
    print(f"  Total vocab: {vocab_size_total}")
    print(f"  Vocab per rank: {vocab_size_per_rank}")
    print(f"  Dim: {dim}")
    print(f"  TP world size: {tp_world_size}")
    
    # Reference: compute full logits and cross entropy
    logits_ref = hidden.float() @ weight_full.T.float()
    logprobs_ref = F.cross_entropy(logits_ref, labels, reduction='none', ignore_index=ignore_index)
    
    print(f"\nReference logprobs (first 10):")
    print(logprobs_ref[:10])
    
    # Store results from all ranks
    all_max = []
    all_accu = []
    all_logprobs = []
    
    # Test each TP rank separately
    for rank in range(tp_world_size):
        print(f"\n{'=' * 80}")
        print(f"Testing TP Rank {rank}")
        print(f"{'=' * 80}")
        
        # Get this rank's weight partition
        weight_rank = weight_full[rank * vocab_size_per_rank:(rank + 1) * vocab_size_per_rank, :].contiguous()
        
        print(f"  Weight shape: {weight_rank.shape}")
        print(f"  Global vocab range: [{rank * vocab_size_per_rank}, {(rank + 1) * vocab_size_per_rank})")
        
        # Count labels in this rank's range
        labels_in_range = ((labels >= rank * vocab_size_per_rank) & 
                          (labels < (rank + 1) * vocab_size_per_rank)).sum().item()
        print(f"  Labels in this rank: {labels_in_range}/{num_tokens}")
        
        # Prepare outputs
        _max = torch.full((num_tokens, 25), float('-inf'), device='cuda', dtype=torch.float32)
        _accu = torch.zeros((num_tokens, 25), device='cuda', dtype=torch.float32)
        _logprobs = torch.zeros(num_tokens, device='cuda', dtype=torch.float32)
        
        # Call kernel
        try:
            from cutlass.cute.runtime import from_dlpack
            import cuda.bindings.driver as cuda
            
            fwd_kernel = FwdMainLoop(vocab_per_split=3072)
            
            hidden_packed = from_dlpack(hidden.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0)
            weight_packed = from_dlpack(weight_rank.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0)
            labels_packed = from_dlpack(labels.detach(), assumed_align=8).mark_compact_shape_dynamic(mode=0)
            logprobs_packed = from_dlpack(_logprobs, assumed_align=16).mark_compact_shape_dynamic(mode=0)
            _max_packed = from_dlpack(_max, assumed_align=8).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
            _accu_packed = from_dlpack(_accu, assumed_align=8).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
            cuda_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            
            fwd_kernel(
                hidden_packed,
                weight_packed,
                labels_packed,
                logprobs_packed,
                _max_packed,
                _accu_packed,
                ignore_index,
                rank,
                cuda_stream
            )
            torch.cuda.synchronize()
            
            print(f"\n  Kernel executed successfully")
            print(f"  Partial _logprobs (first 10): {_logprobs[:10]}")
            
            # Verify this rank's results against reference
            print(f"\n  Verifying rank {rank} results:")
            
            # Compute reference for this rank only
            logits_rank_ref = hidden.float() @ weight_rank.T.float()
            
            # 1. Verify _logprobs (label logits)
            print(f"\n  1. Checking _logprobs (label logits):")
            max_diff_logprobs = 0.0
            count_in_range = 0
            
            for i in range(num_tokens):
                label_val = labels[i].item()
                in_range = (rank * vocab_size_per_rank <= label_val < (rank + 1) * vocab_size_per_rank)
                
                if in_range:
                    # This rank should have computed the logit for this label
                    local_label_idx = label_val - rank * vocab_size_per_rank
                    ref_logit = logits_rank_ref[i, local_label_idx].item()
                    kernel_logit = _logprobs[i].item()
                    diff = abs(ref_logit - kernel_logit)
                    max_diff_logprobs = max(max_diff_logprobs, diff)
                    count_in_range += 1
                    
                    if i < 3:  # Show first 3
                        print(f"      Token {i}: ref={ref_logit:.6f}, kernel={kernel_logit:.6f}, diff={diff:.6e}")
                else:
                    # This rank should have output 0.0 for this token
                    kernel_logit = _logprobs[i].item()
                    if abs(kernel_logit) > 1e-6:
                        print(f"      WARNING: Token {i} not in range but got non-zero: {kernel_logit}")
            
            print(f"      Max diff: {max_diff_logprobs:.6e}")
            
            # 2. Verify _max and _accu (per split)
            print(f"\n  2. Checking _max and _accu (per split):")
            
            # Compute reference _max and _accu using the same logic as the kernel
            # The kernel processes vocabulary in splits
            vocab_per_split = 3072
            num_splits = (vocab_size_per_rank + vocab_per_split - 1) // vocab_per_split
            
            _max_ref = torch.full((num_tokens, num_splits), float('-inf'), device='cuda', dtype=torch.float32)
            _accu_ref = torch.zeros((num_tokens, num_splits), device='cuda', dtype=torch.float32)
            
            for split_idx in range(num_splits):
                start_idx = split_idx * vocab_per_split
                end_idx = min((split_idx + 1) * vocab_per_split, vocab_size_per_rank)
                
                # Get logits for this split
                logits_split = logits_rank_ref[:, start_idx:end_idx]
                
                # Compute max for this split
                _max_ref[:, split_idx] = logits_split.max(dim=1)[0]
                
                # Compute sum(exp(logits - max)) for this split
                _accu_ref[:, split_idx] = torch.exp(logits_split - _max_ref[:, split_idx].unsqueeze(1)).sum(dim=1)
            
            # Compare _max
            max_diff_max = (_max - _max_ref).abs().max().item()
            mean_diff_max = (_max - _max_ref).abs().mean().item()
            print(f"      _max: max_diff={max_diff_max:.6e}, mean_diff={mean_diff_max:.6e}")
            
            # Compare _accu
            max_diff_accu = (_accu - _accu_ref).abs().max().item()
            mean_diff_accu = (_accu - _accu_ref).abs().mean().item()
            print(f"      _accu: max_diff={max_diff_accu:.6e}, mean_diff={mean_diff_accu:.6e}")
            
            # Show a few examples
            print(f"      Example (token 0, first 3 splits):")
            for split_idx in range(min(3, num_splits)):
                print(f"        Split {split_idx}: _max ref={_max_ref[0, split_idx]:.6f}, kernel={_max[0, split_idx]:.6f}")
                print(f"                  _accu ref={_accu_ref[0, split_idx]:.6f}, kernel={_accu[0, split_idx]:.6f}")
            
            # Overall summary
            print(f"\n  Summary for rank {rank}:")
            print(f"    Tokens in range: {count_in_range}/{num_tokens}")
            print(f"    Max diff (_logprobs): {max_diff_logprobs:.6e}")
            print(f"    Max diff (_max):      {max_diff_max:.6e}")
            print(f"    Max diff (_accu):     {max_diff_accu:.6e}")
            
            tolerance = 1e-3
            all_pass = max_diff_logprobs < tolerance and max_diff_max < tolerance and max_diff_accu < tolerance
            
            if all_pass:
                print(f"    ✅ PASS: All differences < tolerance {tolerance:.6e}")
            else:
                print(f"    ❌ FAIL: Some differences >= tolerance {tolerance:.6e}")
            
            # Store results
            all_max.append(_max.clone())
            all_accu.append(_accu.clone())
            all_logprobs.append(_logprobs.clone())
                
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine results from all ranks (simulating all-reduce)
    print(f"\n{'=' * 80}")
    print("Combining results from all ranks (simulating TP epilogue)")
    print(f"{'=' * 80}")
    
    # Step 1: all-reduce MAX on _max
    reduced_max = torch.stack(all_max, dim=0).max(dim=0)[0]
    print(f"  Step 1: Computed reduced_max (max across ranks)")
    
    # Step 2: all-reduce SUM on _logprobs (label logits)
    combined_label_logits = torch.stack(all_logprobs, dim=0).sum(dim=0)
    print(f"  Step 2: Computed combined_label_logits (sum across ranks)")
    
    # Step 3: forward_tp_epilogue - rescale _accu and reduce across splits
    # This mimics triton kernel: forward_tp_epilogue
    maximum = torch.zeros(num_tokens, device='cuda', dtype=torch.float32)
    accumulate = torch.zeros(num_tokens, device='cuda', dtype=torch.float32)
    
    for i in range(num_tokens):
        global_max = -float('inf')
        global_accu = 0.0
        
        # Process each split
        for split_idx in range(reduced_max.shape[1]):
            # Update global_max
            max_old = global_max
            local_max = reduced_max[i, split_idx].item()
            global_max = max(global_max, local_max)
            
            # Rescale and accumulate _accu from all ranks
            coeff = math.exp(max_old - global_max) if max_old > -float('inf') else 0.0
            global_accu = coeff * global_accu
            
            for rank in range(tp_world_size):
                original_max = all_max[rank][i, split_idx].item()
                accu_val = all_accu[rank][i, split_idx].item()
                scale = math.exp(original_max - global_max)
                global_accu += scale * accu_val
        
        maximum[i] = global_max
        accumulate[i] = global_accu
    
    print(f"  Step 3: Computed maximum and accumulate (TP epilogue)")
    
    # Step 4: all-reduce SUM on accumulate (in real code, but we already combined above)
    print(f"  Step 4: accumulate already combined (would be all-reduce SUM)")
    
    # Step 5: forward_tp_epilogue_update_logprobs
    # CE = maximum + log(accumulate) - logprobs
    import math
    final_logprobs = maximum + torch.log(accumulate) - combined_label_logits
    print(f"  Step 5: Computed final logprobs")
    
    print(f"\nCombined final logprobs (first 10):")
    print(final_logprobs[:10])
    print(f"\nReference logprobs (first 10):")
    print(logprobs_ref[:10])
    
    # Compare with reference
    print(f"\n{'=' * 80}")
    print("Comparing with reference")
    print(f"{'=' * 80}")
    
    diff = (final_logprobs - logprobs_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nMax absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    
    # Check if within tolerance
    tolerance = 1e-3
    if max_diff < tolerance:
        print(f"\n✅ PASS: Max diff {max_diff:.6e} < tolerance {tolerance:.6e}")
    else:
        print(f"\n❌ FAIL: Max diff {max_diff:.6e} >= tolerance {tolerance:.6e}")
        
        # Show worst cases
        print(f"\nWorst 5 differences:")
        _, worst_indices = diff.topk(5)
        for idx in worst_indices:
            print(f"  Token {idx}: kernel={final_logprobs[idx]:.6f}, ref={logprobs_ref[idx]:.6f}, diff={diff[idx]:.6e}")
    
    print(f"\n{'=' * 80}")
    print("Test complete")
    print(f"{'=' * 80}")