# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng,
# Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.
# Modified by NVIDIA CORPORATION, 2026: super-block fold for split-invariant bytes.

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Float32, Int32, Int64
from flash_attn.cute.mask import Sm100FusedMask as FusedMask
from flash_attn.cute.tile_scheduler import SM100_TMEM_CAPACITY_COLUMNS
from flash_attn.cute.tile_scheduler import Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler
from flash_attn.cute.tile_scheduler import (
    Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
)
from flash_attn.cute.tile_scheduler import compute_sm100_fmha_grid as compute_grid
from flash_attn.cute.tile_scheduler import (
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
)
from flash_attn.cute.utils import as_bshkrd_tensor, ex2_emulation_2, smid

from .fold import fold_pair, fold_stats, normalize_scale, vcopy, view16, zero_key_rows

_TUNE_HD256_CAUSAL = {
    "ex2_emu_freq": 14,
    "ex2_emu_res": 6,
    "ex2_emu_start_frg": 0,
    "num_regs_softmax": 256,
    "num_regs_correction": 160,
}


class PPAttentionForward:
    """Forward kernel for prefill and chunked paged prefill."""

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        super_block_tiles: int = 8,
        num_regs_other: int = 32,
    ):
        assert head_dim in (128, 256), "kernel only supports head_dim 128 or 256"
        self.head_dim = head_dim
        assert super_block_tiles >= 1, "super_block_tiles must be >= 1"
        self.cta_group_size = 2
        self.G = super_block_tiles
        self.fold_chunks = (head_dim // 128) * 4
        self.fold_chunk_bytes = 32 * 128 * 4
        self.fold_barrier_id = 2
        self.fold_worker_barrier_id = 3
        self.fold_worker_threads = 64
        self.fold_args_len = 2 * 128 + 2

        qk_acc_dtype = cutlass.Float32
        pv_acc_dtype = cutlass.Float32
        mma_tiler = (128, 128, head_dim)
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.qhead_per_kvhead = qhead_per_kvhead
        self.mma_tiler = mma_tiler
        assert mma_tiler[0] == 128 and mma_tiler[1] == 128, "Only 128x128 tile impl is supported"
        self.cta_tiler = (mma_tiler[0], mma_tiler[1], mma_tiler[2])
        self.qk_mma_tiler = (
            self.cta_group_size * mma_tiler[0],
            mma_tiler[1],
            min(self.cta_tiler[2], 128),
        )
        self.pv_mma_tiler = self.qk_mma_tiler
        self.pv_block_tiler = (
            self.pv_mma_tiler[0] // self.cta_group_size,
            self.pv_mma_tiler[1],
            self.pv_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_pv = self.cta_tiler[2] // self.pv_mma_tiler[1]
        self.cluster_shape_mn = (self.cta_group_size, 1)
        self.tmem_warp_shape_mn = (4, 1)
        self.is_causal = is_causal
        self.use_semantic_trip_range = is_causal

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = (10, 11)
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

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

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )

        self.tmem_s_offset = 0
        self.tmem_o_offset = 256
        self.tmem_p_offset = self.tmem_s_offset

        _tune = _TUNE_HD256_CAUSAL
        self.num_regs_softmax = _tune.get("num_regs_softmax", 256)
        self.num_regs_correction = _tune.get("num_regs_correction", 160)
        assert 24 <= num_regs_other <= 80 and num_regs_other % 8 == 0
        self.num_regs_other = num_regs_other
        self.ex2_emu_freq = _tune.get("ex2_emu_freq", 4)
        self.ex2_emu_res = _tune.get("ex2_emu_res", 3)
        self.ex2_emu_start_frg = _tune.get("ex2_emu_start_frg", 0)

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = self.iterations_qk
        self.kv_stage = 4
        self.qk_acc_stage = 2
        self.mma_corr_stage = 1

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        mScratch: Optional[cute.Tensor] = None,
        mLSE: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        assert mScratch is not None

        q_tensor, k_tensor, v_tensor = mQ, mK, mV
        o_tensor = mO
        lse_tensor = mLSE
        cum_seqlen_q = mCuSeqlensQ

        q_rank = len(mQ.shape)
        k_rank = len(mK.shape)
        if cutlass.const_expr(cum_seqlen_q is not None):
            if cutlass.const_expr(q_rank == 3):
                s_q = mQ.shape[0]
                h_q = mQ.shape[1]
                d = mQ.shape[2]
            else:
                raise RuntimeError(f"hd256 forward varlen expects q rank 3, got rank {q_rank}")
        else:
            if cutlass.const_expr(q_rank == 4):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2]
                d = mQ.shape[3]
            else:
                raise RuntimeError(f"hd256 forward non-varlen expects q rank 4, got rank {q_rank}")

        if cutlass.const_expr(k_rank == 4):
            s_k = mK.shape[1]
            h_k = mK.shape[2]
        else:
            raise RuntimeError(f"hd256 forward expects k rank 4, got rank {k_rank}")
        if cutlass.const_expr(cum_seqlen_q is not None):
            b = mCuSeqlensQ.shape[0] - 1
        else:
            b = mQ.shape[0]

        scale_softmax = softmax_scale
        scale_softmax_log2 = softmax_scale * math.log2(math.exp(1.0))
        scale_output = 1.0
        s_lse = s_q
        h_r = h_q // h_k
        s_q64 = Int64(s_q)
        s_k64 = Int64(s_k)
        s_lse64 = Int64(s_lse)
        h_r64 = Int64(h_r)
        h_k64 = Int64(h_k)
        b64 = Int64(b)
        s_q_total = q_tensor.shape[0] if cum_seqlen_q is not None else s_q64
        s_k_total = s_k64
        b_lse = b64 if cum_seqlen_q is None else 1
        stride_b_lse = h_r64 * h_k64 * s_lse64 if cum_seqlen_q is None else 0
        varlen_q = cum_seqlen_q is not None
        q_norm = as_bshkrd_tensor(q_tensor, h_k, h_r, varlen_q)

        q = cute.make_tensor(
            q_norm.iterator,
            cute.make_layout(
                (s_q_total, d, ((h_r, h_k), b)),
                stride=(
                    q_norm.stride[1],
                    q_norm.stride[4],
                    ((q_norm.stride[3], q_norm.stride[2]), q_norm.stride[0]),
                ),
            ),
        )
        if cutlass.const_expr(mPageTable is not None):
            page_size = k_tensor.shape[1]
            max_seqlen_k_paged = Int32(mPageTable.shape[1] * page_size)
            k = cute.make_tensor(k_tensor.iterator, cute.select(k_tensor.layout, mode=[1, 3, 2, 0]))
            v = cute.make_tensor(v_tensor.iterator, cute.select(v_tensor.layout, mode=[3, 1, 2, 0]))
            page_table = cute.make_tensor(
                mPageTable.iterator,
                cute.make_layout(
                    (mPageTable.shape[0], mPageTable.shape[1]),
                    stride=(mPageTable.stride[0], mPageTable.stride[1]),
                ),
            )
        else:
            k_norm = as_bshkrd_tensor(k_tensor, h_k, 1, False)
            v_norm = as_bshkrd_tensor(v_tensor, h_k, 1, False)
            k = cute.make_tensor(
                k_norm.iterator,
                cute.make_layout(
                    (s_k_total, d, ((h_r, h_k), b)),
                    stride=(
                        k_norm.stride[1],
                        k_norm.stride[4],
                        ((0, k_norm.stride[2]), k_norm.stride[0]),
                    ),
                ),
            )
            v = cute.make_tensor(
                v_norm.iterator,
                cute.make_layout(
                    (d, s_k_total, ((h_r, h_k), b)),
                    stride=(
                        v_norm.stride[4],
                        v_norm.stride[1],
                        ((0, v_norm.stride[2]), v_norm.stride[0]),
                    ),
                ),
            )
            page_table = None
            max_seqlen_k_paged = None
        o_norm = as_bshkrd_tensor(o_tensor, h_k, h_r, varlen_q)
        o = cute.make_tensor(
            o_norm.iterator,
            cute.make_layout(
                (s_q_total, d, ((h_r, h_k), b)),
                stride=(
                    o_norm.stride[1],
                    o_norm.stride[4],
                    ((o_norm.stride[3], o_norm.stride[2]), o_norm.stride[0]),
                ),
            ),
        )
        o_res = None
        if cutlass.const_expr(lse_tensor is not None):
            lse_layout = cute.make_layout(
                (s_lse64, ((h_r, h_k), b_lse)),
                stride=(1, ((s_lse64, h_r64 * s_lse64), stride_b_lse)),
            )
            lse = cute.make_tensor(lse_tensor.iterator, lse_layout)
        else:
            lse = None

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.q_dtype.width

        o_shape = (s_q, o.shape[1], o.shape[2]) if cum_seqlen_q is not None else o.shape
        self.tile_sched_params, grid = compute_grid(o_shape, self.cta_tiler, False)

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (qk_tiled_mma.thr_id.shape,)
        )

        self.epi_tile = self.pv_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, self.qk_mma_tiler, self.k_dtype, self.kv_stage
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, self.qk_acc_stage
        )
        p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.kv_stage
        )
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_kv_bytes = k_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.v_stage_elems = cute.cosize(v_smem_layout)

        @cute.struct
        class SharedStorage:
            """Shared memory layout of one CTA."""

            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 4]
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            fold_mbar_ptr: cute.struct.MemRange[Int64, 1]
            fold_req_mbar_ptr: cute.struct.MemRange[Int64, 1]
            fold_done_mbar_ptr: cute.struct.MemRange[Int64, 1]
            fold_args: cute.struct.MemRange[Float32, self.fold_args_len]
            tmem_dealloc_mbar: Int64
            v_tail_mbar: cute.struct.MemRange[Int64, 2]
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            o,
            o_res,
            cum_seqlen_q,
            mSeqUsedK,
            lse,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            page_table,
            max_seqlen_k_paged,
            mScratch,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout,
            v_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=False,
        )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        mRes_qdl: Optional[cute.Tensor],
        cum_seqlen_q: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        mPageTable: Optional[cute.Tensor],
        max_seqlen_k: Optional[Int32],
        mScratch: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        """Device entry point."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

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
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=make_thread_cooperative_group(
                3 if cutlass.const_expr(mPageTable is not None) else 1
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0]
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0]
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
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
                len(self.correction_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0]
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        if warp_idx == self.correction_warp_ids[0]:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.fold_mbar_ptr.data_ptr(), 1)
                cute.arch.mbarrier_init(
                    storage.fold_req_mbar_ptr.data_ptr(),
                    len(self.correction_warp_ids) * self.threads_per_warp,
                )
                cute.arch.mbarrier_init(
                    storage.fold_done_mbar_ptr.data_ptr(), self.fold_worker_threads
                )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=self.cta_group_size == 2,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

        v_tail_mbar = storage.v_tail_mbar.data_ptr()
        if cutlass.const_expr(mPageTable is not None):
            if warp_idx == self.load_warp_id:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(v_tail_mbar, 1)
                    cute.arch.mbarrier_init(v_tail_mbar + 1, 1)
                cute.arch.mbarrier_init_fence()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV = smem.allocate_tensor(
            element_type=self.v_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )

        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )
        sMax = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )
        sFoldObar = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=cute.make_layout(32 * 128), byte_alignment=1024
        )
        sFoldPart = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=cute.make_layout(32 * 128), byte_alignment=1024
        )
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv, stride=self.pv_mma_tiler[1] // self.tmem_warp_shape_mn[1]
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)

        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        blk_idx = cute.arch.block_idx()
        tile_sched = FmhaStaticTileScheduler(
            tile_sched_params, blk_idx[0], blk_idx, cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        if warp_idx == self.empty_warp_id[0] or warp_idx == self.empty_warp_id[1]:
            self._fold_worker_loop(
                mScratch,
                sFoldObar,
                sFoldPart,
                storage.fold_mbar_ptr.data_ptr(),
                storage.fold_req_mbar_ptr.data_ptr(),
                storage.fold_done_mbar_ptr.data_ptr(),
                cute.make_tensor(
                    storage.fold_args.data_ptr(), cute.make_layout(self.fold_args_len)
                ),
            )

        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            load_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader_load = load_rank % self.cta_group_size == 0
            load_peer_rank = load_rank ^ 1
            load_lane = cute.arch.lane_idx()
            tail_phase = Int32(0)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                _npairs = cute.ceil_div(mQ_qdl.shape[0], self.qk_mma_tiler[0])
                _pair = curr_block_coord[0] // self.cta_group_size
                _rank = curr_block_coord[0] % self.cta_group_size
                curr_block_coord = (
                    (_npairs - 1 - _pair) * self.cta_group_size + _rank,
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                batch_coord = curr_block_coord[2][1]
                seqlen_q = cute.size(mQ_qdl.shape[0])
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                cuseqlen_q = Int32(0)
                cuseqlen_k = Int32(0)
                block_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    block_offset = (
                        cuseqlen_q,
                        cuseqlen_k,
                        Int32(0),
                        ((Int32(0), Int32(0)), Int32(0)),
                    )
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0], mma_block_coord[0], seqlen_q
                    )
                if cutlass.const_expr(mSeqUsedK is not None):
                    seqlen_k = mSeqUsedK[batch_coord]
                if not continue_cond:
                    mQ_qdl_ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), mQ_qdl)
                    q_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                    )
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        block_in_cluster_coord_vmnk[2],
                        q_cta_layout,
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    kv_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                    )
                    if cutlass.const_expr(mPageTable is None):
                        mK_kdl_ = cute.domain_offset(
                            cute.select(block_offset, mode=[1, 2, 3]), mK_kdl
                        )
                        mV_dkl_ = cute.domain_offset(
                            cute.select(block_offset, mode=[2, 1, 3]), mV_dkl
                        )
                        gK_kdl = cute.flat_divide(
                            mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                        )
                        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_k,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(tSgK_kdl, 0, 3),
                        )
                        gV_dkl = cute.flat_divide(
                            mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                        )
                        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_v,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sV, 0, 3),
                            cute.group_modes(tSgV_dkl, 0, 3),
                        )
                        tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                        tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
                    else:
                        head_kv_coord = curr_block_coord[2][0] // self.qhead_per_kvhead
                        mK_kdl_ = mK_kdl[None, None, head_kv_coord, None]
                        mV_dkl_ = mV_dkl[None, None, head_kv_coord, None]
                        gK_kdl = cute.flat_divide(
                            mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                        )
                        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_k,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(tSgK_kdl, 0, 3),
                        )
                        gV_dkl = cute.flat_divide(
                            mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                        )
                        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_v,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sV, 0, 3),
                            cute.group_modes(tSgV_dkl, 0, 3),
                        )
                        tKgK = tKgK_kdl
                        tVgV = tVgV_dkl
                    q_hb_coord = mma_block_coord[2]
                    tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, q_hb_coord]

                    seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                        FusedMask.get_trip_start_count_via_block_info(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            False,
                            None,
                            None,
                        )
                    )
                    seqlen_kv_loop_end = seqlen_kv_loop_start + seqlen_kv_loop_steps
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        q_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, iter],
                            tQsQ[None, q_handle.index],
                            tma_bar_ptr=q_handle.barrier,
                        )

                    kv_coord = seqlen_kv_loop_start
                    k_page_idx = (
                        mPageTable[batch_coord, kv_coord]
                        if cutlass.const_expr(mPageTable is not None)
                        else None
                    )
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            (
                                tKgK[None, kv_coord, iter]
                                if cutlass.const_expr(mPageTable is None)
                                else tKgK[None, 0, iter, k_page_idx]
                            ),
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        self._paged_arrive(k_handle.barrier, is_leader_load, mPageTable)
                    kv_coord += 1
                    v_page_idx_prev = (
                        k_page_idx if cutlass.const_expr(mPageTable is not None) else None
                    )
                    if cutlass.const_expr(mPageTable is not None):
                        if seqlen_kv_loop_steps > 1:
                            k_page_idx = mPageTable[batch_coord, kv_coord]

                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            k_handle = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                (
                                    tKgK[None, kv_coord, iter]
                                    if cutlass.const_expr(mPageTable is None)
                                    else tKgK[None, 0, iter, k_page_idx]
                                ),
                                tKsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                            self._paged_arrive(k_handle.barrier, is_leader_load, mPageTable)
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                (
                                    tVgV[None, iter, kv_coord - 1]
                                    if cutlass.const_expr(mPageTable is None)
                                    else tVgV[None, iter, 0, v_page_idx_prev]
                                ),
                                tVsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                            self._paged_arrive(v_handle.barrier, is_leader_load, mPageTable)
                        v_page_idx_prev = (
                            k_page_idx if cutlass.const_expr(mPageTable is not None) else None
                        )
                        kv_coord += 1
                        if cutlass.const_expr(mPageTable is not None):
                            if kv_coord < seqlen_kv_loop_end:
                                k_page_idx = mPageTable[batch_coord, kv_coord]
                    if cutlass.const_expr(mPageTable is None):
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV[None, iter, seqlen_kv_loop_end - 1],
                                tVsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                    else:
                        tail_keys = seqlen_k - (seqlen_kv_loop_end - 1) * 128
                        tail_tx = Int32(self.tma_copy_kv_bytes)
                        if tail_keys < 128:
                            tail_tx = Int32(0)
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_kv_producer.acquire_and_advance(expected_tx=tail_tx)
                            if tail_keys < 128:
                                if is_leader_load:
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            v_tail_mbar, self.tma_copy_kv_bytes
                                        )
                                cute.copy(
                                    tma_atom_v,
                                    tVgV[None, iter, 0, v_page_idx_prev],
                                    tVsV[None, v_handle.index],
                                    tma_bar_ptr=v_tail_mbar,
                                )
                                if is_leader_load:
                                    cute.arch.mbarrier_wait(v_tail_mbar, tail_phase)
                                    cute.arch.fence_acq_rel_cluster()
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive(
                                            v_tail_mbar + 1, peer_cta_rank_in_cluster=load_peer_rank
                                        )
                                else:
                                    cute.arch.mbarrier_wait(v_tail_mbar + 1, tail_phase)
                                tail_phase = tail_phase ^ 1
                                zero_key_rows(
                                    sV,
                                    v_handle.index * self.v_stage_elems,
                                    tail_keys,
                                    load_lane,
                                    self.v_stage_elems,
                                )
                                cute.arch.fence_proxy("async.shared", space="cluster")
                                cute.arch.fence_acq_rel_cluster()
                                cute.arch.sync_warp()
                                with cute.arch.elect_one():
                                    if is_leader_load:
                                        cute.arch.mbarrier_arrive(v_handle.barrier)
                                    else:
                                        cute.arch.mbarrier_arrive(
                                            v_handle.barrier,
                                            peer_cta_rank_in_cluster=load_peer_rank,
                                        )
                            else:
                                cute.copy(
                                    tma_atom_v,
                                    tVgV[None, iter, 0, v_page_idx_prev],
                                    tVsV[None, v_handle.index],
                                    tma_bar_ptr=v_handle.barrier,
                                )
                                self._paged_arrive(v_handle.barrier, is_leader_load, mPageTable)

                work_tile = tile_sched.advance_to_next_work()
            load_kv_producer.tail()
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader_cta = cta_rank_in_cluster % self.cta_group_size == 0

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                _npairs = cute.ceil_div(mQ_qdl.shape[0], self.qk_mma_tiler[0])
                _pair = curr_block_coord[0] // self.cta_group_size
                _rank = curr_block_coord[0] % self.cta_group_size
                curr_block_coord = (
                    (_npairs - 1 - _pair) * self.cta_group_size + _rank,
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                seqlen_q = cute.size(mQ_qdl.shape[0])
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                batch_coord = curr_block_coord[2][1]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0], mma_block_coord[0], seqlen_q
                    )

                if not continue_cond:
                    if cutlass.const_expr(mSeqUsedK is not None):
                        seqlen_k = mSeqUsedK[batch_coord]

                    seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                        FusedMask.get_trip_start_count_via_block_info(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            False,
                            None,
                            None,
                        )
                    )
                    seqlen_kv_loop_end = seqlen_kv_loop_start + seqlen_kv_loop_steps

                    load_q_releaser = load_q_consumer.clone()
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if seqlen_kv_loop_steps > 1:
                        if is_leader_cta:
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter % self.q_stage]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                            s_handle.commit()
                        for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                            if is_leader_cta:
                                s_handle = mma_s_producer.acquire_and_advance()
                                tStS_slice = tStS[None, None, None, s_handle.index]
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for iter in cutlass.range(self.iterations_qk, unroll=1):
                                    tSrQ_slice = tSrQ[None, None, None, iter % self.q_stage]
                                    k_handle = load_kv_consumer.wait_and_advance()
                                    tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                    num_kphases = cute.size(tSrQ_slice, mode=[2])
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    k_handle.release()
                                s_handle.commit()

                                p_handle = p_mma_consumer.wait_and_advance()
                                o_handle = mma_corr_producer.acquire_and_advance()
                                pv_tile_idx = seqlen_kv_loop_start + i - 1
                                pv_whether_acc = pv_tile_idx % self.G != 0
                                for iter in cutlass.range(self.iterations_pv, unroll=1):
                                    v_handle = load_kv_consumer.wait_and_advance()
                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                                    tOtO_slice = tOtO_staged[None, None, None, iter]
                                    tStS_slice = tStS[None, None, None, p_handle.index]
                                    tP = cute.make_tensor(
                                        tStS_slice.iterator, p_tmem_layout_staged.outer
                                    )
                                    tOrP = pv_thr_mma.make_fragment_A(tP)
                                    tOrP_slice = cute.make_tensor(
                                        cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                        tOrP.layout,
                                    )
                                    tOrV_slice = tOrV[None, None, None, v_handle.index]
                                    num_kphases = cute.size(tOrV_slice, mode=[2])
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            pv_tiled_mma,
                                            tOtO_slice,
                                            tOrP_slice[kphase_coord],
                                            tOrV_slice[kphase_coord],
                                            tOtO_slice,
                                        )
                                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    v_handle.release()
                                o_handle.commit()
                                p_handle.release()
                        if is_leader_cta:
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                tSrQ_slice = tSrQ[None, None, None, iter % self.q_stage]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                            p_handle = p_mma_consumer.wait_and_advance()
                            o_handle = mma_corr_producer.acquire_and_advance()
                            pv_whether_acc = (seqlen_kv_loop_end - 2) % self.G != 0
                            for iter in cutlass.range(self.iterations_pv, unroll=1):
                                v_handle = load_kv_consumer.wait_and_advance()
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                                tOtO_slice = tOtO_staged[None, None, None, iter]
                                tStS_slice = tStS[None, None, None, p_handle.index]
                                tP = cute.make_tensor(
                                    tStS_slice.iterator, p_tmem_layout_staged.outer
                                )
                                tOrP = pv_thr_mma.make_fragment_A(tP)
                                tOrP_slice = cute.make_tensor(
                                    cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                    tOrP.layout,
                                )
                                tOrV_slice = tOrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tOrV_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        pv_tiled_mma,
                                        tOtO_slice,
                                        tOrP_slice[kphase_coord],
                                        tOrV_slice[kphase_coord],
                                        tOtO_slice,
                                    )
                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                            o_handle.commit()
                            p_handle.release()
                    else:
                        if is_leader_cta:
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter % self.q_stage]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                    if is_leader_cta:
                        p_handle = p_mma_consumer.wait_and_advance()
                        o_handle = mma_corr_producer.acquire_and_advance()
                        pv_whether_acc = (seqlen_kv_loop_end - 1) % self.G != 0
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_kv_consumer.wait_and_advance()
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            tOtO_slice = tOtO_staged[None, None, None, iter]
                            tStS_slice = tStS[None, None, None, p_handle.index]
                            tP = cute.make_tensor(tStS_slice.iterator, p_tmem_layout_staged.outer)
                            tOrP = pv_thr_mma.make_fragment_A(tP)
                            tOrP_slice = cute.make_tensor(
                                cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                tOrP.layout,
                            )
                            tOrV_slice = tOrV[None, None, None, v_handle.index]
                            num_kphases = cute.size(tOrV_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    pv_tiled_mma,
                                    tOtO_slice,
                                    tOrP_slice[kphase_coord],
                                    tOrV_slice[kphase_coord],
                                    tOtO_slice,
                                )
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            v_handle.release()
                        o_handle.commit()
                        p_handle.release()
                work_tile = tile_sched.advance_to_next_work()
            mma_s_producer.tail()
            mma_corr_producer.tail()

        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                _npairs = cute.ceil_div(mQ_qdl.shape[0], self.qk_mma_tiler[0])
                _pair = curr_block_coord[0] // self.cta_group_size
                _rank = curr_block_coord[0] % self.cta_group_size
                curr_block_coord = (
                    (_npairs - 1 - _pair) * self.cta_group_size + _rank,
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                seqlen_q = cute.size(mQ_qdl.shape[0])
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0], mma_block_coord[0], seqlen_q
                    )
                if not continue_cond:
                    if cutlass.const_expr(mSeqUsedK is not None):
                        seqlen_k = mSeqUsedK[batch_coord]

                    row_max = -Float32.inf
                    row_max_prev = -Float32.inf
                    row_sum = 0.0

                    start_count, trip_count = FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        False,
                        None,
                        None,
                    )
                    end_count = start_count + trip_count
                    if end_count <= start_count:
                        start_count = 0
                        end_count = 1
                    if cutlass.const_expr(self.use_semantic_trip_range):
                        n_block_min_causal_local_mask, n_block_min_before_local_mask = (
                            FusedMask.get_trip_mask_bounds_via_block_info(
                                mma_block_coord,
                                self.qk_mma_tiler,
                                seqlen_q,
                                seqlen_k,
                                self.is_causal,
                                False,
                                None,
                                None,
                            )
                        )
                    cS_base = cute.make_identity_tensor(
                        (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
                    )
                    cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)
                    tScS = qk_thr_mma.partition_C(cS)

                    for step in cutlass.range(start_count, end_count, 1, unroll=1):
                        cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                        tScS_iter = qk_thr_mma.partition_C(cS_iter)
                        if cutlass.const_expr(self.use_semantic_trip_range):
                            need_apply_mask = (
                                step >= n_block_min_causal_local_mask
                                or step < n_block_min_before_local_mask
                                or step == end_count - 1
                            )
                        else:
                            need_apply_mask = step == end_count - 1
                        is_sb_boundary = (step % self.G == 0) & (step > start_count)
                        fold_m = row_max_prev
                        fold_l = row_sum
                        if is_sb_boundary:
                            row_max_prev = -Float32.inf
                            row_sum = 0.0
                        row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer = (
                            self.softmax_step(
                                need_apply_mask,
                                (row_max_prev, row_sum, seqlen_q, seqlen_k, scale_softmax_log2),
                                (is_sb_boundary, fold_m, fold_l),
                                (tStS, tScS_iter),
                                (mma_s_consumer, p_mma_producer, s_corr_producer),
                            )
                        )
                        row_max_prev = row_max
                    sum_producer = self.store_sum_max(
                        row_max,
                        mLSE,
                        row_sum,
                        sSum,
                        sMax,
                        sum_producer,
                        curr_block_coord,
                        seqlen_q,
                        cum_seqlen_q,
                        cuseqlen_q,
                        scale_softmax,
                    )
                work_tile = tile_sched.advance_to_next_work()
            p_mma_producer.tail()
            s_corr_producer.tail()

        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)

            fold_phase = Int32(0)
            bulk_atom_fold = cute.make_copy_atom(cute.nvgpu.cpasync.CopyBulkG2SOp(), Float32)
            fold_mbar = storage.fold_mbar_ptr.data_ptr()
            fold_worker_args = ()
            fold_worker_args = (
                storage.fold_req_mbar_ptr.data_ptr(),
                storage.fold_done_mbar_ptr.data_ptr(),
                cute.make_tensor(
                    storage.fold_args.data_ptr(), cute.make_layout(self.fold_args_len)
                ),
            )

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                _npairs = cute.ceil_div(mQ_qdl.shape[0], self.qk_mma_tiler[0])
                _pair = curr_block_coord[0] // self.cta_group_size
                _rank = curr_block_coord[0] % self.cta_group_size
                curr_block_coord = (
                    (_npairs - 1 - _pair) * self.cta_group_size + _rank,
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                seqlen_q = cute.size(mQ_qdl.shape[0])
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                continue_cond = False
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0], mma_block_coord[0], seqlen_q
                    )

                if not continue_cond:
                    if cutlass.const_expr(mSeqUsedK is not None):
                        seqlen_k = mSeqUsedK[batch_coord]

                    mO_qdl_eff = mO_qdl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        block_offset_o = (
                            cuseqlen_q,
                            Int32(0),
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )
                        mO_qdl_eff = cute.domain_offset(
                            cute.select(block_offset_o, mode=[0, 2, 3]), mO_qdl
                        )

                    gO_qdl = cute.flat_divide(
                        mO_qdl_eff, cute.select(self.pv_block_tiler, mode=[0, 1])
                    )
                    cO_qdl = cute.flat_divide(
                        cute.make_identity_tensor(mO_qdl_eff.shape),
                        cute.select(self.pv_block_tiler, mode=[0, 1]),
                    )

                    seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                        FusedMask.get_trip_start_count_via_block_info(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            False,
                            None,
                            None,
                        )
                    )
                    o_l_coord = curr_block_coord[2]
                    gO_staged = gO_qdl[None, None, curr_block_coord[0], None, o_l_coord]
                    gRes_staged = None
                    if cutlass.const_expr(mRes_qdl is not None):
                        mRes_qdl_eff = mRes_qdl
                        if cutlass.const_expr(cum_seqlen_q is not None):
                            mRes_qdl_eff = cute.domain_offset(
                                cute.select(block_offset_o, mode=[0, 2, 3]), mRes_qdl
                            )
                        gRes_qdl = cute.flat_divide(
                            mRes_qdl_eff, cute.select(self.pv_block_tiler, mode=[0, 1])
                        )
                        gRes_staged = gRes_qdl[None, None, curr_block_coord[0], None, o_l_coord]
                    cO_staged = cO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                    tScS = qk_thr_mma.partition_C(cS)

                    scratch_smid = smid()
                    fold_M = -Float32.inf
                    fold_L = Float32(0.0)
                    fold_a = Float32(0.0)
                    fold_b = Float32(0.0)
                    obar_slot = Int32(0)
                    fold_next = Int32(self.fold_chunks)
                    staged = Int32(-1)
                    has_fold = cutlass.Boolean(False)

                    stats_handle = s_corr_consumer.wait_and_advance()
                    stats_handle.release()
                    for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        is_sb_boundary = (seqlen_kv_loop_start + step) % self.G == 0
                        (
                            fold_M,
                            fold_L,
                            fold_a,
                            fold_b,
                            obar_slot,
                            fold_next,
                            staged,
                            fold_phase,
                            has_fold,
                            mma_corr_consumer,
                            s_corr_consumer,
                        ) = self.correction_step(
                            is_sb_boundary,
                            (
                                has_fold,
                                fold_M,
                                fold_L,
                                fold_a,
                                fold_b,
                                obar_slot,
                                fold_next,
                                staged,
                                fold_phase,
                            ),
                            scale_softmax_log2,
                            (s_corr_consumer, tStS, tScS),
                            (mma_corr_consumer, tOtO_staged, cO_staged),
                            self.epi_tile,
                            mScratch,
                            scratch_smid,
                            curr_block_coord[0] * self.pv_block_tiler[0],
                            (sFoldObar, sFoldPart, fold_mbar, bulk_atom_fold) + fold_worker_args,
                        )
                    mma_corr_consumer, sum_consumer, fold_phase = self.correction_epilog(
                        (seqlen_q, scale_output),
                        (sum_consumer, sSum, sMax),
                        (has_fold, fold_M, fold_L, fold_a, fold_b, fold_next, staged, fold_phase),
                        (mma_corr_consumer, gO_staged, cO_staged, tOtO_staged),
                        self.epi_tile,
                        scale_softmax_log2,
                        mScratch,
                        scratch_smid,
                        curr_block_coord[0] * self.pv_block_tiler[0],
                        obar_slot,
                        (sFoldObar, sFoldPart, fold_mbar) + fold_worker_args[1:2],
                        (
                            (
                                mLSE,
                                (
                                    (curr_block_coord[2][0], Int32(0))
                                    if cutlass.const_expr(cum_seqlen_q is not None)
                                    else curr_block_coord[2]
                                ),
                                cuseqlen_q,
                                scale_softmax,
                            )
                            if cutlass.const_expr(mLSE is not None)
                            else ()
                        ),
                        ((gRes_staged,) if cutlass.const_expr(mRes_qdl is not None) else ()),
                    )
                work_tile = tile_sched.advance_to_next_work()
            if warp_idx == self.correction_warp_ids[0]:
                with cute.arch.elect_one():
                    fold_worker_args[2][257] = Float32(1.0)
            cute.arch.mbarrier_arrive(fold_worker_args[0])

        if warp_idx > self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

        return

    @cute.jit
    def _paged_arrive(self, barrier, is_leader, mPageTable: Optional[cute.Tensor]):
        """Two extra full-barrier arrivals for a paged load stage whose tail is not zeroed."""
        if cutlass.const_expr(mPageTable is not None):
            if is_leader:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(barrier, arrive_count=2)

    @cute.jit
    def softmax_step(
        self,
        need_apply_mask: cutlass.Boolean,
        value_args: Tuple,
        fold_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        """Softmax over one key tile."""
        row_max, row_sum, seqlen_q, seqlen_k, scale_softmax_log2 = value_args
        is_sb_boundary, fold_m, fold_l = fold_args
        tStS, tScS = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)

        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        if need_apply_mask:
            FusedMask.apply_mask_via_causal_local(
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                seqlen_q,
                seqlen_k,
                self.use_semantic_trip_range,
                self.is_causal,
                False,
                None,
                None,
            )
        old_row_max = row_max
        row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0

        stats_handle = s_corr_producer.acquire_and_advance()
        stats_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
        )
        stats_c_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
        )
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_store_stats_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)), self.qk_acc_dtype
        )
        tiled_tmem_store_stats = tcgen05.make_tmem_copy(tmem_store_stats_atom, tOtStats)
        thr_tmem_store_stats = tiled_tmem_store_stats.get_slice(thread_idx)
        tTMEM_STOREcStats = thr_tmem_store_stats.partition_S(tOcStats)
        tTMEM_STORErStats = cute.make_rmem_tensor(tTMEM_STOREcStats.shape, self.qk_acc_dtype)
        tTMEM_STORErStats[0] = old_row_max
        tTMEM_STORErStats[1] = row_max_safe
        if is_sb_boundary:
            tTMEM_STORErStats[0] = fold_m
            tTMEM_STORErStats[1] = fold_l
        tTMEM_STOREtStats = thr_tmem_store_stats.partition_D(tOtStats)
        cute.copy(tiled_tmem_store_stats, tTMEM_STORErStats, tTMEM_STOREtStats)
        cute.arch.fence_view_async_tmem_store()
        stats_handle.commit()

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale
        p_handle = p_mma_producer.acquire_and_advance()
        ex2_frg_tile = 32
        ex2_frg_cnt = cute.size(tTMEM_LOADrS) // ex2_frg_tile
        tTMEM_LOADrS_ex2 = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(ex2_frg_tile))
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.q_dtype)
        tTMEM_STORErP_ex2 = cute.logical_divide(tTMEM_STORErP, cute.make_layout(ex2_frg_tile))
        for j in cutlass.range_constexpr(ex2_frg_cnt):
            for k in cutlass.range_constexpr(0, ex2_frg_tile, 2):
                tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = cute.arch.fma_packed_f32x2(
                    (tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
                if cutlass.const_expr(self.ex2_emu_freq == 0):
                    tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(tTMEM_LOADrS_ex2[k, j], fastmath=True)
                    tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                        tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                    )
                else:
                    if cutlass.const_expr(
                        k % self.ex2_emu_freq < self.ex2_emu_freq - self.ex2_emu_res
                        or j >= ex2_frg_cnt - 1
                        or j < self.ex2_emu_start_frg
                    ):
                        tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k, j], fastmath=True
                        )
                        tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                        )
                    else:
                        tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = ex2_emulation_2(
                            tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]
                        )
            tTMEM_STORErP_ex2[None, j].store(tTMEM_LOADrS_ex2[None, j].load().to(self.q_dtype))
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tilePlikeFP32 = tStS_slice.shape[1] // Float32.width * self.q_dtype.width
        tStS_P_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], tilePlikeFP32))
        )
        tStS_P = cute.make_tensor(tStS_slice.iterator, tStS_P_layout)
        tScS_P_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], tilePlikeFP32))
        )
        tScS_P = cute.make_tensor(tScS_slice.iterator, tScS_P_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtP = thr_store.partition_D(tStS_P)
        tTMEM_STOREcS = thr_store.partition_S(tScS_P)
        tTMEM_STORErP_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErP.iterator, dtype=self.qk_acc_dtype), tTMEM_STOREcS.shape
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErP_, tTMEM_STOREtP)
        cute.arch.fence_view_async_tmem_store()

        p_handle.commit()
        acc_scale_ = scale * (old_row_max - row_max_safe)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        row_sum *= acc_scale
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)
        reduction_unroll = 4
        frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
            )
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
            )
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
            )
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
            )
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]
        return row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer

    @cute.jit
    def correction_step(
        self,
        is_boundary,
        fold_state: Tuple,
        scale_softmax_log2: Float32,
        stats_args: tuple,
        o_args: tuple,
        epi_tile: cute.Tile,
        mScratch: Optional[cute.Tensor],
        scratch_smid: Int32,
        tile_q_base: Int32,
        pf_args: Tuple = (),
    ):
        """One correction step: either the FA4 online rescale, or (at a super-block boundary)
        Phase 1
        of a fold — dump the finished partial to the free scratch slot and release the O handle
        immediately; the RMW `Obar = fold_pair(Obar, partial, a, b)` is deferred to Phase 2 chunks
        whose operands are prefetched one step ahead.
        """
        has_fold, fold_M, fold_L, fold_a, fold_b, obar_slot, fold_next, staged, phase = fold_state
        fold_req_mbar, fold_done_mbar, sFoldArgs = pf_args[4:]
        s_corr_consumer, tStS, tScS = stats_args
        mma_o_consumer, tOtO_staged, cO_staged = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        stats_handle = s_corr_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, stats_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        stats_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
        )
        stats_c_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
        )
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_load_stats_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)), self.qk_acc_dtype
        )
        tiled_tmem_load_stats = tcgen05.make_tmem_copy(tmem_load_stats_atom, tOtStats)
        thr_tmem_load_stats = tiled_tmem_load_stats.get_slice(thread_idx)
        tTMEM_LOADtStats = thr_tmem_load_stats.partition_S(tOtStats)
        tTMEM_LOADcStats = thr_tmem_load_stats.partition_D(tOcStats)
        tTMEM_LOADrStats = cute.make_rmem_tensor(tTMEM_LOADcStats.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load_stats, tTMEM_LOADtStats, tTMEM_LOADrStats)

        stat0 = tTMEM_LOADrStats[0]
        stat1 = tTMEM_LOADrStats[1]
        stats_handle.release()
        o_handle = mma_o_consumer.wait_and_advance()

        if is_boundary:
            staged, phase = self._fold_worker_wait(staged, phase, fold_done_mbar)
            dump_slot = obar_slot
            if has_fold:
                dump_slot = Int32(1) - obar_slot
            for iter in cutlass.range(self.iterations_pv, unroll_full=True):
                tOtO = tOtO_staged[(None, None), 0, 0, iter]
                cO = cO_staged[None, None, iter]
                tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
                cO_epi = cute.zipped_divide(cO, epi_tile)
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.pv_acc_dtype
                )
                tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
                thr_load = tmem_tiled_load.get_slice(thread_idx)
                tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
                tTMEM_LOADcO = thr_load.partition_D(cO_epi)
                n_chunks = cute.size(tTMEM_LOADtO, mode=[1])
                frag_shape = tTMEM_LOADcO[None, 0, 0].shape
                tTMrO = cute.make_rmem_tensor_like(
                    cute.append(
                        cute.make_layout(frag_shape),
                        cute.make_layout(2, stride=cute.size(frag_shape)),
                    ),
                    self.pv_acc_dtype,
                )
                cute.copy(tmem_tiled_load, tTMEM_LOADtO[None, 0, 0], tTMrO[None, 0])
                for i in cutlass.range(1, n_chunks, unroll_full=True):
                    cute.copy(tmem_tiled_load, tTMEM_LOADtO[None, i, 0], tTMrO[None, i % 2])
                    self._dump_frag(
                        tTMrO[None, (i - 1) % 2],
                        tTMEM_LOADcO[None, i - 1, 0],
                        mScratch,
                        scratch_smid,
                        dump_slot,
                        tile_q_base,
                    )
                self._dump_frag(
                    tTMrO[None, (n_chunks - 1) % 2],
                    tTMEM_LOADcO[None, n_chunks - 1, 0],
                    mScratch,
                    scratch_smid,
                    dump_slot,
                    tile_q_base,
                )
            if has_fold:
                fold_M, fold_L, fold_a, fold_b = fold_stats(
                    fold_M, fold_L, stat0, stat1, scale_softmax_log2
                )
                sFoldArgs[thread_idx] = fold_a
                sFoldArgs[128 + thread_idx] = fold_b
                if thread_idx == 0:
                    slot_f = Float32(0.0)
                    if obar_slot != 0:
                        slot_f = Float32(1.0)
                    sFoldArgs[256] = slot_f
                    sFoldArgs[257] = Float32(0.0)
                cute.arch.fence_proxy("async")
                cute.arch.mbarrier_arrive(fold_req_mbar)
                staged = Int32(1)
            else:
                fold_M = stat0
                fold_L = stat1
                has_fold = cutlass.Boolean(True)
        else:
            scale = scale_softmax_log2 * (stat0 - stat1)
            scale = cute.math.exp2(scale, fastmath=True)
            for iter in cutlass.range(self.iterations_pv, unroll_full=True):
                tOtO = tOtO_staged[(None, None), 0, 0, iter]
                cO = cO_staged[None, None, iter]
                tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
                cO_epi = cute.zipped_divide(cO, epi_tile)
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), self.pv_acc_dtype
                )
                tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
                thr_load = tmem_tiled_load.get_slice(thread_idx)
                tmem_store_atom = cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(16)), self.pv_acc_dtype
                )
                tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi)
                thr_store = tmem_store_atom.get_slice(thread_idx)
                tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
                tTMEM_LOADcO = thr_load.partition_D(cO_epi)
                tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
                tTMrO = cute.make_rmem_tensor_like(
                    cute.append(
                        cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
                        cute.make_layout(2, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)),
                    ),
                    self.pv_acc_dtype,
                )
                tTMEM_LOADtO_0 = tTMEM_LOADtO[None, 0, 0]
                cute.copy(tmem_tiled_load, tTMEM_LOADtO_0, tTMrO[None, 0])
                iter_num = cute.size(tTMEM_LOADtO, mode=[1])
                for i in cutlass.range(1, iter_num, unroll_full=True):
                    tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                    cute.copy(tmem_tiled_load, tTMEM_LOADtO_i, tTMrO[None, i % 2])
                    for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                        tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2] = (
                            cute.arch.mul_packed_f32x2(
                                (tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2]), (scale, scale)
                            )
                        )
                    tTMEM_STOREtO_prev_i = tTMEM_STOREtO[None, i - 1, 0]
                    cute.copy(tmem_store_atom, tTMrO[None, (i - 1) % 2], tTMEM_STOREtO_prev_i)

                for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                    tTMrO[j, (iter_num - 1) % 2], tTMrO[j + 1, (iter_num - 1) % 2] = (
                        cute.arch.mul_packed_f32x2(
                            (tTMrO[j, (iter_num - 1) % 2], tTMrO[j + 1, (iter_num - 1) % 2]),
                            (scale, scale),
                        )
                    )
                cute.copy(
                    tmem_store_atom,
                    tTMrO[None, (iter_num - 1) % 2],
                    tTMEM_STOREtO[None, iter_num - 1, 0],
                )
            cute.arch.fence_view_async_tmem_store()
        cute.arch.fence_view_async_tmem_load()
        o_handle.release()
        return (
            fold_M,
            fold_L,
            fold_a,
            fold_b,
            obar_slot,
            fold_next,
            staged,
            phase,
            has_fold,
            mma_o_consumer,
            s_corr_consumer,
        )

    @cute.jit
    def _fold_chunk_coords(
        self, c: cutlass.Constexpr, staged_args: tuple, epi_tile: cute.Tile, thread_idx: Int32
    ):
        """Coordinate partition for fold chunk c (0..fold_chunks-1)."""
        tOtO_staged, cO_staged = staged_args
        iter = c // 4
        i = c % 4
        tOtO = tOtO_staged[(None, None), 0, 0, iter]
        cO = cO_staged[None, None, iter]
        tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
        cO_epi = cute.zipped_divide(cO, epi_tile)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.pv_acc_dtype
        )
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        return thr_load.partition_D(cO_epi)[None, i, 0]

    @cute.jit
    def _fold_worker_wait(self, staged: Int32, phase: Int32, fold_done_mbar: cute.Pointer):
        """Correction side of the worker handshake: block until the worker has finished the fold in
        flight (staged >= 0), then flip the parity.
        """
        if staged >= 0:
            cute.arch.mbarrier_wait(fold_done_mbar, phase)
            phase = phase ^ Int32(1)
            staged = Int32(-1)
        return staged, phase

    @cute.jit
    def _fold_worker_loop(
        self,
        mScratch: cute.Tensor,
        sFoldObar: cute.Tensor,
        sFoldPart: cute.Tensor,
        fold_mbar: cute.Pointer,
        fold_req_mbar: cute.Pointer,
        fold_done_mbar: cute.Pointer,
        sFoldArgs: cute.Tensor,
    ):
        """Fold worker (64 threads, the two spare warps)."""
        bulk_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyBulkG2SOp(), Float32)
        tidx, _, _ = cute.arch.thread_idx()
        wtid = tidx % self.fold_worker_threads
        scratch_smid = smid()
        req_phase = Int32(0)
        tx_phase = Int32(0)
        running = cutlass.Boolean(True)
        while running:
            cute.arch.mbarrier_wait(fold_req_mbar, req_phase)
            req_phase = req_phase ^ Int32(1)
            obar_slot = Int32(0)
            if sFoldArgs[256] != 0.0:
                obar_slot = Int32(1)
            if sFoldArgs[257] != 0.0:
                running = cutlass.Boolean(False)
            else:
                partial_slot = Int32(1) - obar_slot
                for c in cutlass.range(self.fold_chunks, unroll=1):
                    self._fold_stage_issue(
                        c,
                        obar_slot,
                        partial_slot,
                        bulk_atom,
                        mScratch,
                        scratch_smid,
                        sFoldObar,
                        sFoldPart,
                        fold_mbar,
                        issuer_warp=self.empty_warp_id[0],
                    )
                    cute.arch.mbarrier_wait(fold_mbar, tx_phase)
                    tx_phase = tx_phase ^ Int32(1)
                    for r in cutlass.range_constexpr(2):
                        row = wtid + Int32(r * self.fold_worker_threads)
                        self._fold_row_chunk(
                            c,
                            sFoldArgs[row],
                            sFoldArgs[128 + row],
                            obar_slot,
                            mScratch,
                            scratch_smid,
                            sFoldObar,
                            sFoldPart,
                            row,
                        )
                    cute.arch.fence_proxy("async")
                    cute.arch.barrier(
                        barrier_id=self.fold_worker_barrier_id,
                        number_of_threads=self.fold_worker_threads,
                    )
                cute.arch.mbarrier_arrive(fold_done_mbar)

    @cute.jit
    def _dump_frag(
        self,
        frag: cute.Tensor,
        cO_i: cute.Tensor,
        mScratch: cute.Tensor,
        scratch_smid: Int32,
        dump_slot: Int32,
        tile_q_base: Int32,
    ):
        """Store one 32-column TMEM fragment of the partial O (one row per thread, columns
        col0..col0+31) to the dump slot as 8 x 16 B.
        """
        row = cO_i[0][0] - tile_q_base
        col0 = cO_i[0][1]
        base = (scratch_smid * 2 + dump_slot) * (self.head_dim * 128) + (col0 // 4) * 512 + row * 4
        v4 = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        for q in cutlass.range_constexpr(8):
            for k in cutlass.range_constexpr(4):
                v4[k] = frag[4 * q + k]
            vcopy(v4, view16(mScratch, base + q * 512, 4))

    @cute.jit
    def _fold_stage_issue(
        self,
        c: Int32,
        obar_slot: Int32,
        partial_slot: Int32,
        bulk_atom: cute.CopyAtom,
        mScratch: cute.Tensor,
        scratch_smid: Int32,
        sFoldObar: cute.Tensor,
        sFoldPart: cute.Tensor,
        fold_mbar: cute.Pointer,
        issuer_warp: cutlass.Constexpr[int] = 4,
    ):
        """One elected thread stages chunk c: two 16 KB cp.async.bulk copies (Obar chunk and partial
        chunk, each a contiguous gmem range) into the SMEM staging buffers, tracked by fold_mbar's
        tx count.
        """
        gObar = cute.make_tensor(
            mScratch.iterator
            + (scratch_smid * 2 + obar_slot) * (self.head_dim * 128)
            + c * (32 * 128),
            cute.make_layout(32 * 128),
        )
        gPart = cute.make_tensor(
            mScratch.iterator
            + (scratch_smid * 2 + partial_slot) * (self.head_dim * 128)
            + c * (32 * 128),
            cute.make_layout(32 * 128),
        )
        if cute.arch.warp_idx() == issuer_warp:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(fold_mbar, 2 * self.fold_chunk_bytes)
                cute.copy(bulk_atom, gObar, sFoldObar, mbar_ptr=fold_mbar)
                cute.copy(bulk_atom, gPart, sFoldPart, mbar_ptr=fold_mbar)

    @cute.jit
    def _fold_row_chunk(
        self,
        c: Int32,
        fold_a: Float32,
        fold_b: Float32,
        obar_slot: Int32,
        mScratch: cute.Tensor,
        scratch_smid: Int32,
        sFoldObar: cute.Tensor,
        sFoldPart: cute.Tensor,
        row: Int32,
    ):
        """Fold one row of staged chunk c into the Obar slot: 8 x (16 B Obar + 16 B partial) SMEM
        loads, 16 fold_pair, 8 x 16 B gmem stores.
        """
        obar_base = (
            (scratch_smid * 2 + obar_slot) * (self.head_dim * 128) + c * (32 * 128) + row * 4
        )
        vO = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        vP = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        for q in cutlass.range_constexpr(8):
            vcopy(view16(sFoldObar, q * 512 + row * 4, 4), vO)
            vcopy(view16(sFoldPart, q * 512 + row * 4, 4), vP)
            vO[0], vO[1] = fold_pair((vO[0], vO[1]), (vP[0], vP[1]), fold_a, fold_b)
            vO[2], vO[3] = fold_pair((vO[2], vO[3]), (vP[2], vP[3]), fold_a, fold_b)
            vcopy(vO, view16(mScratch, obar_base + q * 512, 4))

    @cute.jit
    def _fold_consume_smem(
        self,
        c: Int32,
        fold_a: Float32,
        fold_b: Float32,
        obar_slot: Int32,
        mScratch: cute.Tensor,
        scratch_smid: Int32,
        sFoldObar: cute.Tensor,
        sFoldPart: cute.Tensor,
        thread_idx: Int32,
    ):
        """Fold staged chunk c into the Obar slot; thread owns row=thread_idx."""
        self._fold_row_chunk(
            c, fold_a, fold_b, obar_slot, mScratch, scratch_smid, sFoldObar, sFoldPart, thread_idx
        )

    @cute.jit
    def _fold_rmw_chunk(
        self,
        c: cutlass.Constexpr,
        fold_a: Float32,
        fold_b: Float32,
        obar_slot: Int32,
        partial_slot: Int32,
        staged_args: tuple,
        epi_tile: cute.Tile,
        mScratch: cute.Tensor,
        scratch_smid: Int32,
        tile_q_base: Int32,
        thread_idx: Int32,
    ):
        """Unprefetched chunk c (thread owns row=thread_idx): all 16 x 16 B loads into fragments
        first,
        then fold_pair + 8 x 16 B stores into the Obar slot.
        """
        row = thread_idx
        obar_base = (
            (scratch_smid * 2 + obar_slot) * (self.head_dim * 128) + c * (32 * 128) + row * 4
        )
        part_base = (
            (scratch_smid * 2 + partial_slot) * (self.head_dim * 128) + c * (32 * 128) + row * 4
        )
        tRrObar = cute.make_rmem_tensor(cute.make_layout((4, 8)), self.pv_acc_dtype)
        tRrPart = cute.make_rmem_tensor(cute.make_layout((4, 8)), self.pv_acc_dtype)
        for q in cutlass.range_constexpr(8):
            vcopy(view16(mScratch, obar_base + q * 512, 4), tRrObar[None, q])
            vcopy(view16(mScratch, part_base + q * 512, 4), tRrPart[None, q])
        for q in cutlass.range_constexpr(8):
            tRrObar[0, q], tRrObar[1, q] = fold_pair(
                (tRrObar[0, q], tRrObar[1, q]), (tRrPart[0, q], tRrPart[1, q]), fold_a, fold_b
            )
            tRrObar[2, q], tRrObar[3, q] = fold_pair(
                (tRrObar[2, q], tRrObar[3, q]), (tRrPart[2, q], tRrPart[3, q]), fold_a, fold_b
            )
            vcopy(tRrObar[None, q], view16(mScratch, obar_base + q * 512, 4))

    @cute.jit
    def _fold_consume(
        self,
        fold_next: Int32,
        staged: Int32,
        phase: Int32,
        fold_a: Float32,
        fold_b: Float32,
        obar_slot: Int32,
        bulk_atom: cute.CopyAtom,
        staged_args: tuple,
        epi_tile: cute.Tile,
        mScratch: cute.Tensor,
        scratch_smid: Int32,
        tile_q_base: Int32,
        thread_idx: Int32,
        sFoldObar: cute.Tensor,
        sFoldPart: cute.Tensor,
        fold_mbar: cute.Pointer,
    ):
        """Phase 2 of a deferred fold, run after the O handle release in a non-boundary step: wait
        for
        the staged chunk, fold it into Obar, then (after a fence + named barrier so the staging
        buffers and the Obar stores are settled) issue the next chunk's bulk copies — they land
        during the next tile's work.
        """
        if staged >= 0:
            cute.arch.mbarrier_wait(fold_mbar, phase)
            phase = phase ^ Int32(1)
            self._fold_consume_smem(
                staged,
                fold_a,
                fold_b,
                obar_slot,
                mScratch,
                scratch_smid,
                sFoldObar,
                sFoldPart,
                thread_idx,
            )
            fold_next = staged + Int32(1)
            staged = Int32(-1)
            cute.arch.fence_proxy("async")
            cute.arch.barrier(
                barrier_id=self.fold_barrier_id,
                number_of_threads=len(self.correction_warp_ids) * self.threads_per_warp,
            )
            if fold_next < self.fold_chunks:
                self._fold_stage_issue(
                    fold_next,
                    obar_slot,
                    Int32(1) - obar_slot,
                    bulk_atom,
                    mScratch,
                    scratch_smid,
                    sFoldObar,
                    sFoldPart,
                    fold_mbar,
                )
                staged = fold_next
        return fold_next, staged, phase

    @cute.jit
    def _fold_drain(
        self,
        fold_next: Int32,
        staged: Int32,
        phase: Int32,
        fold_a: Float32,
        fold_b: Float32,
        obar_slot: Int32,
        staged_args: tuple,
        epi_tile: cute.Tile,
        mScratch: cute.Tensor,
        scratch_smid: Int32,
        tile_q_base: Int32,
        thread_idx: Int32,
        sFoldObar: cute.Tensor,
        sFoldPart: cute.Tensor,
        fold_mbar: cute.Pointer,
    ):
        """Finish a fold: consume the staged chunk (if in flight), then load-first RMW the remaining
        chunks.
        """
        partial_slot = Int32(1) - obar_slot
        if staged >= 0:
            cute.arch.mbarrier_wait(fold_mbar, phase)
            phase = phase ^ Int32(1)
            self._fold_consume_smem(
                staged,
                fold_a,
                fold_b,
                obar_slot,
                mScratch,
                scratch_smid,
                sFoldObar,
                sFoldPart,
                thread_idx,
            )
            fold_next = staged + Int32(1)
            staged = Int32(-1)
        for c in cutlass.range_constexpr(self.fold_chunks):
            if c >= fold_next:
                self._fold_rmw_chunk(
                    c,
                    fold_a,
                    fold_b,
                    obar_slot,
                    partial_slot,
                    staged_args,
                    epi_tile,
                    mScratch,
                    scratch_smid,
                    tile_q_base,
                    thread_idx,
                )
        return Int32(self.fold_chunks), Int32(-1), phase

    @cute.jit
    def correction_epilog(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        fold_state: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
        scale_softmax_log2: Float32,
        mScratch: Optional[cute.Tensor],
        scratch_smid: Int32,
        tile_q_base: Int32,
        obar_slot: Int32,
        pf_args: Tuple = (),
        lse_args: Tuple = (),
        res_args: Tuple = (),
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer, Int32]:
        """Correction epilog."""
        seqlen_q, _ = value_args
        sum_consumer, sSum, sMax = sum_args
        has_fold, fold_M, fold_L, fold_a, fold_b, _, staged, phase = fold_state
        mma_o_consumer, gO_staged, cO_staged, tOtO_staged = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        staged, phase = self._fold_worker_wait(staged, phase, pf_args[3])
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        row_max_last = sMax[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        fold_a = Float32(0.0)
        fold_b = Float32(0.0)
        m_final = row_max_last
        l_final = row_sum
        scale = normalize_scale(row_sum)
        if has_fold:
            m_final, l_final, fold_a, fold_b = fold_stats(
                fold_M, fold_L, row_max_last, row_sum, scale_softmax_log2
            )
            scale = normalize_scale(l_final)
        if cutlass.const_expr(len(lse_args) > 0):
            mLSE, hb_idx, q_offset, scale_softmax_v = lse_args
            l_bad = l_final == 0.0 or l_final != l_final
            lse_value = (
                scale_softmax_v * m_final + cute.math.log(l_final, fastmath=True)
                if not l_bad
                else -Float32.inf
            )
            q_idx = tile_q_base + thread_idx
            if cute.elem_less(q_idx, seqlen_q):
                mLSE[q_idx + q_offset, hb_idx] = lse_value
        gRes_staged = res_args[0] if cutlass.const_expr(len(res_args) > 0) else None
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range_constexpr(self.iterations_pv):
            gO = gO_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            if cutlass.const_expr(gRes_staged is not None):
                gRes_epi = cute.zipped_divide(gRes_staged[None, None, iter], epi_tile)
                tTMEM_LOADgRes = thr_tmem_load.partition_D(gRes_epi)
            for i in cutlass.range_constexpr(cute.size(tTMEM_LOADtO, mode=[1])):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                if has_fold:
                    tRrObar = cute.make_rmem_tensor(tTMrO.shape, self.pv_acc_dtype)
                    for j in cutlass.range(cute.size(tRrObar), unroll_full=True):
                        row = tTMEM_LOADcO_i[j][0] - tile_q_base
                        col = tTMEM_LOADcO_i[j][1]
                        tRrObar[j] = mScratch[scratch_smid, obar_slot, col // 4, row, col % 4]
                    for j in cutlass.range(0, cute.size(tTMrO), 2, unroll_full=True):
                        r0, r1 = fold_pair(
                            (tRrObar[j], tRrObar[j + 1]), (tTMrO[j], tTMrO[j + 1]), fold_a, fold_b
                        )
                        tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                            (r0, r1), (scale, scale)
                        )
                else:
                    for j in cutlass.range(0, cute.size(tTMrO), 2, unroll_full=True):
                        tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                            (tTMrO[j], tTMrO[j + 1]), (scale, scale)
                        )
                tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tSMrO.store(o_vec.to(self.o_dtype))
                if cute.elem_less(tTMEM_LOADcO_i[0][0], seqlen_q):
                    cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
                    if cutlass.const_expr(gRes_staged is not None):
                        tSMrRes = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                        res_vec = o_vec - tSMrO.load().to(self.pv_acc_dtype)
                        tSMrRes.store(res_vec.to(self.o_dtype))
                        cute.autovec_copy(tSMrRes, tTMEM_LOADgRes[None, i, 0])
        o_handle.release()
        return mma_o_consumer, sum_consumer, phase

    @cute.jit
    def store_sum_max(
        self,
        row_max,
        mLSE,
        row_sum,
        sSum,
        sMax,
        sum_producer,
        current_block_coord,
        seqlen_q,
        cum_seqlen_q,
        cuseqlen_q,
        scale_softmax,
    ):
        """Store sum max."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_producer.acquire_and_advance()
        sSum[thread_idx] = row_sum
        sMax[thread_idx] = row_max
        cute.arch.fence_view_async_shared()
        sum_handle.commit()
        return sum_producer
