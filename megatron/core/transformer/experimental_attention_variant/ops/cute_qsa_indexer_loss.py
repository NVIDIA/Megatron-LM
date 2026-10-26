# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Parts of the cp.async row-gather and the TMEM epilogue pattern follow the cuDNN frontend
# sparse score-recompute kernel (python/cudnn/deepseek_sparse_attention/score_recompute/
# sparse_score_recompute_sm100.py, SPDX-License-Identifier: Apache-2.0, Copyright (c) 2026
# NVIDIA CORPORATION & AFFILIATES).


"""CuTe DSL (SM100) kernels of the QSA indexer distillation loss
(``qsa_indexer_loss_impl='fused'``).

Teacher (:func:`qsa_indexer_teacher`): per query row and head, the softmax of the bf16-rounded
logits ``bf16(q . k) * scale`` over the tokens of the row's picked blocks, summed over heads
(and KV groups), then max-pooled to blocks (``pool=True``) or returned per token
(``pool=False``, tensor parallelism). ``q`` / ``k`` are the main attention path's (detached)
query and key, so the teacher needs no projection or RoPE of its own.

Tile kernel (:class:`_TeacherTileSm100`), for documents of at most ``TILE_MAX_BLOCKS`` blocks:
``BQ = 128 // heads-per-group`` consecutive rows of one document share the gathered keys (TMA
boxes over the contiguous key range); a heads-major pass computes the online max / sum-exp and
a keys-major pass the probabilities, with bit-identical logits.

Row kernel (:class:`_TeacherRowSm100`), one persistent CTA per SM walking the remaining query
rows (long documents, or rows whose tile spans two documents):

* a 4-warp load group gathers each picked block's ``R`` key rows with ``cp.async`` into
  128-token x 64-dim K-major SMEM stages (the slot order of the row: token ``i`` of a 128-token
  chunk is slot ``i // R``, token ``i % R``), and the row's query heads into SMEM;
* one MMA warp issues keys-major ``tcgen05`` MMAs (A = 128 gathered keys, B = the group's
  query heads padded to 16, K = head dim, fp32 accumulator in TMEM). The accumulation is the
  one cuBLAS uses for the reference einsum (sequential K16 steps), so the bf16-rounded
  logits match it bit for bit;
* every logit of the row stays resident in TMEM (``n_sel * R`` tokens x groups x 16 columns,
  512 columns for the proxy), so the keys are gathered once: the 4 epilogue warps (thread =
  token) keep a per-column online max / sum-exp while chunks arrive, reduce it over the CTA,
  then re-read TMEM for the probabilities, sum heads and groups per token, max-pool over the
  ``R`` lanes of a block and store.

Student (:func:`qsa_indexer_student_sparse`, K8): the KL against the teacher at the selected
blocks only. The row kernel (:class:`_StudentRowSm100`) gathers each selected block key once,
forms the indexer scores, the student logit and the KL, and computes the query gradient in the
same pass; the block-key gradient is one GEMM per query chunk over a collision-free dense
scatter of the per-slot gradients. The loss's backward only scales the saved gradients.

Deterministic (fixed reduction trees, no atomics) and free of host synchronization; all
compile-time parameters are model constants. Compiled kernels are cached per
(dtype, head counts, head dim, ratio, n_sel, pool, token offsets) and compiled on the first
(eager) call, so CUDA-graph capture only replays launches. The supported shapes are validated
once when the QSA layer is built (``qsa.py``).
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

_TOK = 128  # tokens per MMA chunk (UMMA M)
_KB = 64  # head-dim elements per SMEM stage (one 128 B swizzle atom row)
_TMEM_COLS = 512
_REGS_LOW, _REGS_HIGH = 96, 232  # tile kernel: warp-specialized register split
_SUB = 32  # TMEM load sub-tile width (columns) of the tile kernel epilogue

# cp.async (16 B, global -> shared) is the NVVM op behind cute.arch's auto-converting nvvm
# proxy. nvidia-cutlass-dsl <= 4.7 re-exported the op as cute.arch.cp_async_shared_global;
# 4.8 dropped that re-export, so fall back to the proxy, which takes the same arguments.
_cp_async_shared_global = getattr(cute.arch, "cp_async_shared_global", None)
if _cp_async_shared_global is None:
    _cp_async_shared_global = cute.arch.nvvm_wrappers.nvvm.cp_async_shared_global


def _pad16(n: int) -> int:
    return (n + 15) // 16 * 16


class _TeacherRowSm100:
    """Persistent row kernel; see the module docstring."""

    def __init__(self, ng, hpg, d, r, n_sel, pool, has_tokoff, kv_stages=10, lag=8):
        assert d % _KB == 0 and _TOK % r == 0
        self.ng, self.hpg, self.d, self.r, self.n_sel = ng, hpg, d, r, n_sel
        self.np_ = _pad16(hpg)
        self.nkb = d // _KB
        self.spc = _TOK // r  # slots per chunk
        self.max_chunks = (n_sel + self.spc - 1) // self.spc
        self.cols_per_chunk = ng * self.np_
        assert self.max_chunks * self.cols_per_chunk <= _TMEM_COLS
        assert ng * self.nkb >= 2
        self.pool = pool
        self.has_tokoff = has_tokoff
        self.kv_stages = kv_stages
        # cp.async groups kept in flight per loader thread; < kv_stages, and <= the stages of
        # one chunk (NG * NKB) so a row's Q buffer is committed before the next-but-one row
        self.lag = min(lag, kv_stages - 2, ng * (d // _KB))
        self.mma_tiler = (_TOK, self.np_, _KB)
        self.ncol = ng * hpg
        # warps 0-3 epilogue, 4-7 loader, 8 MMA
        self.num_threads = 9 * 32

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # [b, s, ng*hpg, d] bf16
        mK: cute.Tensor,  # [b, s_kv, ng, d] bf16
        mOrder: cute.Tensor,  # [b, s, n_sel] int64
        mPicked: cute.Tensor,  # [b, s, n_sel] uint8
        mNhi: cute.Tensor,  # [b, s] int32
        mTokoff: cute.Tensor,  # [s] int64 (dummy when not has_tokoff)
        mOut: cute.Tensor,  # [b, s, n_sel] fp32 (pool) or [b, s, n_sel * r] fp32
        scale: Float32,
        n_ctas: Int32,
        stream,
    ):
        op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            Float32,
            (_TOK, self.np_, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        sK_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.kv_stages
        )
        sQ_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, 2 * self.ng * self.nkb
        )
        self.kernel(
            tiled_mma, mQ, mK, mOrder, mPicked, mNhi, mTokoff, mOut, scale, sK_layout, sQ_layout
        ).launch(grid=(n_ctas, 1, 1), block=(self.num_threads, 1, 1), stream=stream)

    # ------------------------------------------------------------------ helpers
    @cute.jit
    def _copy16(self, atom, thr_copy, g_ptr, s_row, j):
        """cp.async of the j-th 16 B piece of a 64-element row (gmem pointer -> smem row)."""
        g_row = cute.make_tensor(
            cute.make_ptr(
                cutlass.BFloat16, g_ptr.llvm_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
            cute.make_layout((_KB,)),
        )
        g_ch = cute.flat_divide(g_row, (8,))
        s_ch = cute.flat_divide(s_row, (8,))
        cute.copy(atom, thr_copy.partition_S(g_ch[None, j]), thr_copy.partition_D(s_ch[None, j]))

    @cute.jit
    def _chunk_raw(self, mOrder, mPicked, bi, t, c, grp, dst_ord, dst_pk):
        """Issue the loads of a chunk's block ids / picked flags (consumed one chunk later, so
        their latency overlaps the current chunk's copies). Slots past n_sel are clamped."""
        R, SPC = self.r, self.spc
        for r0 in cutlass.range_constexpr(_TOK // 16):
            trow = r0 * 16 + grp
            slot = cutlass.min(c * SPC + trow // R, self.n_sel - 1)
            dst_ord[r0] = mOrder[bi, t, slot]
            dst_pk[r0] = mPicked[bi, t, slot]

    @cute.jit
    def _load_q(self, atom, thr_copy, mQ, sQ, bi, t, qbase, ltid):
        """cp.async the row's query heads into the B-operand buffers ``qbase + g * NKB + kb``."""
        NG, HPG, NKB = self.ng, self.hpg, self.nkb
        total = NG * HPG * NKB * 8
        for e in cutlass.range_constexpr((total + 127) // 128):
            idx = e * 128 + ltid
            if idx < total:
                qg = idx // (HPG * NKB * 8)
                rem = idx - qg * (HPG * NKB * 8)
                qh = rem // (NKB * 8)
                rem2 = rem - qh * (NKB * 8)
                qkb = rem2 // 8
                qj = rem2 % 8
                sq_dst = cute.composition(
                    sQ[None, None, None, qbase + qg * NKB + qkb], cute.make_layout((self.np_, _KB))
                )
                gq_ptr = mQ[bi, t, qg * HPG + qh, None].iterator + qkb * _KB
                self._copy16(atom, thr_copy, gq_ptr, sq_dst[qh, None], qj)

    # ------------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(  # pylint: disable=missing-function-docstring
        self,
        tiled_mma: cute.TiledMma,
        mQ,
        mK,
        mOrder,
        mPicked,
        mNhi,
        mTokoff,
        mOut,
        scale: Float32,
        sK_layout: cute.ComposedLayout,
        sQ_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bid, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()
        b_sz = mOrder.shape[0]
        s_sz = mOrder.shape[1]
        n_rows = b_sz * s_sz

        NG = const_expr(self.ng)
        HPG = const_expr(self.hpg)
        NP = const_expr(self.np_)
        NKB = const_expr(self.nkb)
        R = const_expr(self.r)
        SPC = const_expr(self.spc)
        NCOL = const_expr(self.ncol)

        @cute.struct
        class SharedStorage:  # pylint: disable=missing-class-docstring
            k_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.kv_stages]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.max_chunks]
            tmem_holding_buf: Int32
            red: cute.struct.MemRange[Float32, 4 * 2 * self.ncol]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sK = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=sK_layout.outer,
            byte_alignment=1024,
            swizzle=sK_layout.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=sQ_layout.outer,
            byte_alignment=1024,
            swizzle=sQ_layout.inner,
        )
        red = storage.red.get_tensor(
            cute.make_layout((4, 2, self.ncol), stride=(2 * self.ncol, self.ncol, 1))
        )

        pK = pipeline.PipelineAsyncUmma.create(
            num_stages=self.kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=storage.k_mbar.data_ptr(),
            defer_sync=True,
        )
        pQ = pipeline.PipelineAsyncUmma.create(
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=storage.q_mbar.data_ptr(),
            defer_sync=True,
        )
        pS = pipeline.PipelineUmmaAsync.create(
            num_stages=self.max_chunks,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            barrier_storage=storage.s_mbar.data_ptr(),
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_threads)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=8
        )
        tmem.allocate(_TMEM_COLS)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        # ================================================================ loader
        if warp_idx >= 4 and warp_idx < 8:
            ltid = tidx - 128
            grp = ltid // 8
            j8 = ltid % 8
            atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                cutlass.BFloat16,
                num_bits_per_copy=128,
            )
            thr_copy = cute.make_tiled_copy_tv(
                atom, cute.make_layout((1,)), cute.make_layout((8,))
            ).get_slice(0)
            # zero the padded query rows of both Q buffers once
            if const_expr(NP > HPG):
                for st in cutlass.range_constexpr(2 * NG * NKB):
                    sq_st = cute.composition(sQ[None, None, None, st], cute.make_layout((NP, _KB)))
                    for e in cutlass.range_constexpr(((NP - HPG) * _KB + 127) // 128):
                        idx = e * 128 + ltid
                        if idx < (NP - HPG) * _KB:
                            sq_st[HPG + idx // _KB, idx % _KB] = cutlass.BFloat16(0.0)
            k_issue = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.kv_stages
            )
            k_commit = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.kv_stages
            )
            q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)  # acquire
            q_commit = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)  # commit
            n_ord = cute.make_rmem_tensor((_TOK // 16,), Int64)
            n_pk = cute.make_rmem_tensor((_TOK // 16,), cutlass.Uint8)
            ctok = cute.make_rmem_tensor((_TOK // 16,), Int32)
            pending = Int32(0)  # issued cp.async groups not yet committed to the pipeline
            firsts = Int32(0)  # bit i: the group issued i groups ago also carried the row's Q
            LAG = const_expr(self.lag)
            rr = bid
            while rr < n_rows:
                row = n_rows - 1 - rr  # longest rows (largest t) first
                bi = row // s_sz
                t = row - bi * s_sz
                nhi = mNhi[bi, t]
                n_chunks = (nhi + SPC - 1) // SPC
                tok0 = Int32(0)
                if const_expr(self.has_tokoff):
                    tok0 = Int32(mTokoff[t])
                # token ids of this thread's 8 rows of a chunk (row = r0 * 16 + grp), -1 when
                # the slot is not picked; the next chunk's ids are loaded one chunk ahead
                if n_chunks > 0:
                    self._chunk_raw(mOrder, mPicked, bi, t, Int32(0), grp, n_ord, n_pk)
                for c in range(n_chunks):
                    for r0 in cutlass.range_constexpr(_TOK // 16):
                        trow = r0 * 16 + grp
                        slot = c * SPC + trow // R
                        tk = Int32(n_ord[r0]) * R + (trow % R) + tok0
                        ok = (slot < nhi) & (n_pk[r0] != cutlass.Uint8(0))
                        ctok[r0] = tk if ok else Int32(-1)
                    if c + 1 < n_chunks:
                        self._chunk_raw(mOrder, mPicked, bi, t, c + 1, grp, n_ord, n_pk)
                    for it in cutlass.range_constexpr(NG * NKB):
                        g = it // NKB
                        kb = it % NKB
                        pK.producer_acquire(k_issue)
                        if c == 0 and it == 0:
                            pQ.producer_acquire(q_state)
                            self._load_q(
                                atom, thr_copy, mQ, sQ, bi, t, q_state.index * (NG * NKB), ltid
                            )
                            q_state.advance()
                        sk_st = cute.composition(
                            sK[None, None, None, k_issue.index], cute.make_layout((_TOK, _KB))
                        )
                        for r0 in cutlass.range_constexpr(_TOK // 16):
                            if ctok[r0] >= 0:
                                g_ptr = mK[bi, ctok[r0], g, None].iterator + kb * _KB
                                self._copy16(atom, thr_copy, g_ptr, sk_st[r0 * 16 + grp, None], j8)
                        cute.arch.cp_async_commit_group()
                        is_first = Int32(0)
                        if c == 0 and it == 0:
                            is_first = Int32(1)
                        firsts = (firsts << 1) | is_first
                        pending = pending + 1
                        if pending > LAG:
                            # keep LAG groups in flight; commit the oldest one
                            cute.arch.cp_async_wait_group(LAG)
                            cute.arch.fence_view_async_shared()
                            pK.producer_commit(k_commit)
                            k_commit.advance()
                            if ((firsts >> LAG) & 1) != 0:
                                pQ.producer_commit(q_commit)
                                q_commit.advance()
                            pending = pending - 1
                        k_issue.advance()
                rr += nctas
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            while pending > 0:
                pK.producer_commit(k_commit)
                k_commit.advance()
                if ((firsts >> (pending - 1)) & 1) != 0:
                    pQ.producer_commit(q_commit)
                    q_commit.advance()
                pending = pending - 1

        # ================================================================ MMA
        if warp_idx == 8:
            tCrA = tiled_mma.make_fragment_A(sK)
            tCrB = tiled_mma.make_fragment_B(sQ)
            k_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kv_stages
            )
            q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            s_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.max_chunks
            )
            rr = bid
            while rr < n_rows:
                row = n_rows - 1 - rr  # longest rows (largest t) first
                bi = row // s_sz
                t = row - bi * s_sz
                nhi = mNhi[bi, t]
                n_chunks = (nhi + SPC - 1) // SPC
                if n_chunks > 0:
                    pQ.consumer_wait(q_cons)
                    qbase = q_cons.index * (NG * NKB)
                    for c in range(n_chunks):
                        pS.producer_acquire(s_prod)
                        for g in cutlass.range_constexpr(NG):
                            acc = cute.make_tensor(
                                tmem_ptr + (s_prod.index * NG + g) * NP, tCtAcc_fake.layout
                            )
                            for kb in cutlass.range_constexpr(NKB):
                                pK.consumer_wait(k_cons)
                                for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, kb != 0 or kk != 0)
                                    cute.gemm(
                                        tiled_mma,
                                        acc,
                                        tCrA[None, None, kk, k_cons.index],
                                        tCrB[None, None, kk, qbase + g * NKB + kb],
                                        acc,
                                    )
                                pK.consumer_release(k_cons)
                                k_cons.advance()
                        pS.producer_commit(s_prod)
                        s_prod.advance()
                    pQ.consumer_release(q_cons)
                    q_cons.advance()
                rr += nctas

        # ================================================================ epilogue
        if warp_idx < 4:
            ewarp = warp_idx
            lane = tidx % 32
            epi_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
            thr_mma = tiled_mma.get_slice(0)
            tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(NP)), Float32)
            tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_fake)
            thr_t = tmem_copy.get_slice(tidx)
            cC = cute.make_identity_tensor((_TOK, NP))
            tTR_c = thr_t.partition_D(thr_mma.partition_C(cC))
            tTR_r = cute.make_rmem_tensor(tTR_c.shape, Float32)
            r_in_chunk = tTR_c[0][0]
            st_m = cute.make_rmem_tensor((NCOL,), Float32)
            st_l = cute.make_rmem_tensor((NCOL,), Float32)
            LOG2E = Float32(math.log2(math.e))
            s_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.max_chunks
            )
            rr = bid
            while rr < n_rows:
                row = n_rows - 1 - rr  # longest rows (largest t) first
                bi = row // s_sz
                t = row - bi * s_sz
                nhi = mNhi[bi, t]
                n_chunks = (nhi + SPC - 1) // SPC
                if n_chunks > 0:
                    s_rel = s_cons.clone()
                    for i in cutlass.range_constexpr(NCOL):
                        st_m[i] = -Float32.inf
                        st_l[i] = Float32(0.0)
                    # ---- pass 1: per-thread online max / sum-exp per column
                    for c in range(n_chunks):
                        slot = c * SPC + r_in_chunk // R
                        ok = slot < nhi
                        if ok:
                            ok = mPicked[bi, t, slot] != cutlass.Uint8(0)
                        pS.consumer_wait(s_cons)
                        for g in cutlass.range_constexpr(NG):
                            acc = cute.make_tensor(
                                tmem_ptr + (s_cons.index * NG + g) * NP, tCtAcc_fake.layout
                            )
                            cute.copy(tmem_copy, thr_t.partition_S(acc), tTR_r)
                            cute.arch.fence_view_async_tmem_load()
                            for h in cutlass.range_constexpr(HPG):
                                x = Float32(tTR_r[h].to(cutlass.BFloat16)) * scale
                                ci = g * HPG + h
                                m_old = st_m[ci]
                                m_new = cute.arch.fmax(m_old, x)
                                l_new = st_l[ci] * cute.math.exp2(
                                    (m_old - m_new) * LOG2E, fastmath=True
                                ) + cute.math.exp2((x - m_new) * LOG2E, fastmath=True)
                                if ok:
                                    st_m[ci] = m_new
                                    st_l[ci] = l_new
                        s_cons.advance()
                    # ---- reduce (m, l) over the 128 threads (fixed butterfly + 4-warp tree)
                    for ci in cutlass.range_constexpr(NCOL):
                        m = st_m[ci]
                        l = st_l[ci]
                        for sh in cutlass.range_constexpr(5):
                            off = 16 >> sh
                            m_o = cute.arch.shuffle_sync_bfly(m, offset=off)
                            l_o = cute.arch.shuffle_sync_bfly(l, offset=off)
                            m_n = cute.arch.fmax(m, m_o)
                            m_s = m_n if m_n != -Float32.inf else Float32(0.0)
                            l = l * cute.math.exp2(
                                (m - m_s) * LOG2E, fastmath=True
                            ) + l_o * cute.math.exp2((m_o - m_s) * LOG2E, fastmath=True)
                            m = m_n
                        if lane == 0:
                            red[ewarp, 0, ci] = m
                            red[ewarp, 1, ci] = l
                    epi_bar.arrive_and_wait()
                    for ci in cutlass.range_constexpr(NCOL):
                        m = red[0, 0, ci]
                        l = red[0, 1, ci]
                        for w in cutlass.range_constexpr(1, 4):
                            m_o = red[w, 0, ci]
                            l_o = red[w, 1, ci]
                            m_n = cute.arch.fmax(m, m_o)
                            m_s = m_n if m_n != -Float32.inf else Float32(0.0)
                            l = l * cute.math.exp2(
                                (m - m_s) * LOG2E, fastmath=True
                            ) + l_o * cute.math.exp2((m_o - m_s) * LOG2E, fastmath=True)
                            m = m_n
                        st_m[ci] = m if m != -Float32.inf else Float32(0.0)
                        st_l[ci] = Float32(1.0) / cute.arch.fmax(l, Float32(1e-10))
                    epi_bar.arrive_and_wait()
                    # ---- pass 2: probabilities, head/group sum, block max-pool, store
                    for c in range(n_chunks):
                        slot = c * SPC + r_in_chunk // R
                        ok = slot < nhi
                        if ok:
                            ok = mPicked[bi, t, slot] != cutlass.Uint8(0)
                        mass = Float32(0.0)
                        for g in cutlass.range_constexpr(NG):
                            acc = cute.make_tensor(
                                tmem_ptr + (s_rel.index * NG + g) * NP, tCtAcc_fake.layout
                            )
                            cute.copy(tmem_copy, thr_t.partition_S(acc), tTR_r)
                            cute.arch.fence_view_async_tmem_load()
                            for h in cutlass.range_constexpr(HPG):
                                x = Float32(tTR_r[h].to(cutlass.BFloat16)) * scale
                                ci = g * HPG + h
                                mass = (
                                    mass
                                    + cute.math.exp2((x - st_m[ci]) * LOG2E, fastmath=True)
                                    * st_l[ci]
                                )
                        pS.consumer_release(s_rel)
                        s_rel.advance()
                        if not ok:
                            mass = Float32(0.0)
                        if const_expr(self.pool):
                            pooled = mass
                            for sh in cutlass.range_constexpr(int(math.log2(R))):
                                pooled = cute.arch.fmax(
                                    pooled, cute.arch.shuffle_sync_bfly(pooled, offset=1 << sh)
                                )
                            if r_in_chunk % R == 0 and slot < nhi:
                                mOut[bi, t, slot] = pooled
                        else:
                            if slot < nhi:
                                mOut[bi, t, slot * R + r_in_chunk % R] = mass
                rr += nctas

        # ================================================================ teardown
        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


class _TeacherTileSm100:
    """Query-tile kernel for short key ranges (the 4K proxy): ``BQ`` consecutive rows of one
    document share the gathered keys.

    A tile walks the dense block range ``[0, nb)`` of its rows (``nb`` = largest picked
    block + 1; at 4K nearly every visible block is picked by some row of a 10-row tile) in
    128-token chunks, twice:

    * pass 1, heads-major (A = the tile's ``BQ * HPG`` query heads of a group, padded to 128
      rows; B = 128 keys): thread = (query, head) row, flash-style online max / sum-exp
      over the tokens of the row's picked blocks (membership from an SMEM block -> slot map);
    * pass 2, keys-major (A = 128 keys, B = the 128 query-head rows): thread = token,
      probabilities with the pass-1 statistics, summed over heads and groups per query,
      max-pooled over the ``R`` lanes of a block and stored at the query's slot.

    Both MMAs are the same tcgen05 128x128x16 op with the operands swapped, so the logits
    of the two passes are bit-identical (and equal to the keys-major/cuBLAS rounding).
    """

    def __init__(self, ng, hpg, d, r, n_sel, nbp, pool, has_tokoff, kv_stages=4, use_tma=False):
        assert d % _KB == 0 and _TOK % r == 0 and hpg <= _TOK
        self.ng, self.hpg, self.d, self.r, self.n_sel = ng, hpg, d, r, n_sel
        self.bq = _TOK // hpg
        self.nkb = d // _KB
        self.spc = _TOK // r
        self.nbp = nbp  # block-map width (multiple of spc, >= document blocks)
        assert nbp % self.spc == 0
        self.pool = pool
        self.has_tokoff = has_tokoff
        self.kv_stages = kv_stages
        self.use_tma = use_tma  # contiguous key chunks via TMA instead of cp.async
        self.mma_tiler = (_TOK, _TOK, _KB)
        self.num_threads = 9 * 32
        assert 2 * ng * _TOK <= _TMEM_COLS

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # [b, s, ng*hpg, d] bf16
        mK: cute.Tensor,  # [b, s_kv, ng, d] bf16
        mOrder: cute.Tensor,  # [b, s, n_sel] int64
        mPicked: cute.Tensor,  # [b, s, n_sel] uint8
        mTileNb: cute.Tensor,  # [b, n_tiles] int32 (0 = tile not handled here)
        mTokoff: cute.Tensor,  # [s] int64 (dummy [1] when absent)
        mOut: cute.Tensor,  # [b, s, n_sel] or [b, s, n_sel * r] fp32
        scale: Float32,
        n_ctas: Int32,
        stream,
    ):
        op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            Float32,
            (_TOK, _TOK, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        sK_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.kv_stages
        )
        sQ_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.ng * self.nkb
        )
        tma_atom_k, tma_k = None, mK
        if const_expr(self.use_tma):
            # (s_kv, d, ng, b) view; one 128-token x 64-dim box per stage, OOB rows zero-filled
            mK_t = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0]))
            tma_atom_k, tma_k = cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                mK_t,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler,
                tiled_mma,
            )
        self.kernel(
            tiled_mma,
            mQ,
            tma_k,
            tma_atom_k,
            mOrder,
            mPicked,
            mTileNb,
            mTokoff,
            mOut,
            scale,
            sK_layout,
            sQ_layout,
        ).launch(grid=(n_ctas, 1, 1), block=(self.num_threads, 1, 1), stream=stream)

    @cute.jit
    def _copy16(self, atom, thr_copy, g_ptr, s_row, j):
        g_row = cute.make_tensor(
            cute.make_ptr(
                cutlass.BFloat16, g_ptr.llvm_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
            cute.make_layout((_KB,)),
        )
        g_ch = cute.flat_divide(g_row, (8,))
        s_ch = cute.flat_divide(s_row, (8,))
        cute.copy(atom, thr_copy.partition_S(g_ch[None, j]), thr_copy.partition_D(s_ch[None, j]))

    @cute.jit
    def _load_q_tile(self, atom, thr_copy, mQ, sQ, bi, t0, s_sz, ltid):
        """cp.async the tile's query heads: row ``qi * HPG + h`` of buffer ``g * NKB + kb``."""
        NG, HPG, NKB, BQ = self.ng, self.hpg, self.nkb, self.bq
        total = BQ * NG * HPG * NKB * 8
        for e in cutlass.range((total + 127) // 128, unroll=4):
            idx = e * 128 + ltid
            if idx < total:
                qi = idx // (NG * HPG * NKB * 8)
                rem = idx - qi * (NG * HPG * NKB * 8)
                qg = rem // (HPG * NKB * 8)
                rem = rem - qg * (HPG * NKB * 8)
                qh = rem // (NKB * 8)
                rem = rem - qh * (NKB * 8)
                qkb = rem // 8
                qj = rem % 8
                if t0 + qi < s_sz:
                    sq_dst = cute.composition(
                        sQ[None, None, None, qg * NKB + qkb], cute.make_layout((_TOK, _KB))
                    )
                    gq_ptr = mQ[bi, t0 + qi, qg * HPG + qh, None].iterator + qkb * _KB
                    self._copy16(atom, thr_copy, gq_ptr, sq_dst[qi * HPG + qh, None], qj)

    @cute.jit
    def _loader_tma(
        self,
        atom,
        thr_copy,
        mQ,
        tma_k,
        tma_atom_k,
        sQ,
        sK,
        pK,
        pQ,
        mTileNb,
        mTokoff,
        tiled_mma,
        bid,
        nctas,
        n_work,
        n_tiles_b,
        s_sz,
        ltid,
        warp_idx,
    ):
        """TMA variant of the loader: the 4 load warps cp.async the tile's queries; warp 4
        streams the contiguous key chunks with TMA (one 128x64 box per stage)."""
        NG, NKB, SPC, R, BQ = self.ng, self.nkb, self.spc, self.r, self.bq
        thr_mma = tiled_mma.get_slice(0)
        k_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
        q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        ww = bid
        while ww < n_work:
            work = n_work - 1 - ww
            bi = work // n_tiles_b
            tile = work - bi * n_tiles_b
            t0 = tile * BQ
            nb = mTileNb[bi, tile]
            n_chunks = (nb + SPC - 1) // SPC
            if n_chunks > 0:
                tok0 = Int32(0)
                if const_expr(self.has_tokoff):
                    tok0 = Int32(mTokoff[t0])
                pQ.producer_acquire(q_state)
                self._load_q_tile(atom, thr_copy, mQ, sQ, bi, t0, s_sz, ltid)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                pQ.producer_commit(q_state)
                q_state.advance()
                if warp_idx == 4:
                    for pc in range(2 * n_chunks):
                        c = pc % n_chunks
                        for it in cutlass.range_constexpr(NG * NKB):
                            g = it // NKB
                            kb = it % NKB
                            pK.producer_acquire(k_state)
                            gK = cute.domain_offset((tok0 + c * _TOK, 0), tma_k[None, None, g, bi])
                            gK_t = cute.local_tile(gK, (_TOK, _KB), (0, kb))
                            tCgK = thr_mma.partition_A(gK_t)
                            tKsK, tKgK = cpasync.tma_partition(
                                tma_atom_k,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(sK, 0, 3),
                                cute.group_modes(tCgK, 0, 3),
                            )
                            cute.copy(
                                tma_atom_k,
                                tKgK,
                                tKsK[(None, k_state.index)],
                                tma_bar_ptr=pK.producer_get_barrier(k_state),
                            )
                            k_state.advance()
            ww += nctas

    @cute.kernel
    def kernel(  # pylint: disable=missing-function-docstring
        self,
        tiled_mma: cute.TiledMma,
        mQ,
        mK,
        tma_atom_k,
        mOrder,
        mPicked,
        mTileNb,
        mTokoff,
        mOut,
        scale: Float32,
        sK_layout: cute.ComposedLayout,
        sQ_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bid, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()
        s_sz = mOrder.shape[1]
        n_tiles_b = mTileNb.shape[1]
        n_work = mTileNb.shape[0] * n_tiles_b

        NG = const_expr(self.ng)
        HPG = const_expr(self.hpg)
        BQ = const_expr(self.bq)
        NKB = const_expr(self.nkb)
        R = const_expr(self.r)
        SPC = const_expr(self.spc)
        NBP = const_expr(self.nbp)
        NSEL = const_expr(self.n_sel)
        NQH = const_expr(self.bq * self.hpg)

        @cute.struct
        class SharedStorage:  # pylint: disable=missing-class-docstring
            k_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.kv_stages]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            tmem_holding_buf: Int32
            stats: cute.struct.MemRange[Float32, 2 * self.ng * _TOK]
            inv: cute.struct.Align[cute.struct.MemRange[cutlass.Int16, self.bq * self.nbp], 16]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sK = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=sK_layout.outer,
            byte_alignment=1024,
            swizzle=sK_layout.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=sQ_layout.outer,
            byte_alignment=1024,
            swizzle=sQ_layout.inner,
        )
        # per (group, query-head row): (max, 1 / sum) of the row's softmax
        stats = storage.stats.get_tensor(cute.make_layout((NG, _TOK, 2), stride=(2 * _TOK, 2, 1)))
        inv = storage.inv.get_tensor(cute.make_layout((BQ, NBP), stride=(NBP, 1)))
        inv32 = cute.recast_tensor(storage.inv.get_tensor(cute.make_layout((BQ * NBP,))), Int32)

        if const_expr(self.use_tma):
            pK = pipeline.PipelineTmaUmma.create(
                num_stages=self.kv_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=_TOK * _KB * 2,
                barrier_storage=storage.k_mbar.data_ptr(),
                defer_sync=True,
            )
        else:
            pK = pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                barrier_storage=storage.k_mbar.data_ptr(),
                defer_sync=True,
            )
        pQ = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=storage.q_mbar.data_ptr(),
            defer_sync=True,
        )
        pS = pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            barrier_storage=storage.s_mbar.data_ptr(),
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_threads)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=8
        )
        tmem.allocate(_TMEM_COLS)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        # ================================================================ loader
        if warp_idx >= 4 and warp_idx < 8:
            cute.arch.setmaxregister_decrease(_REGS_LOW)
            ltid = tidx - 128
            grp = ltid // 8
            j8 = ltid % 8
            atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                cutlass.BFloat16,
                num_bits_per_copy=128,
            )
            thr_copy = cute.make_tiled_copy_tv(
                atom, cute.make_layout((1,)), cute.make_layout((8,))
            ).get_slice(0)
            if const_expr(NQH < _TOK):
                for st in cutlass.range_constexpr(NG * NKB):
                    sq_st = cute.composition(
                        sQ[None, None, None, st], cute.make_layout((_TOK, _KB))
                    )
                    for e in cutlass.range_constexpr(((_TOK - NQH) * _KB + 127) // 128):
                        idx = e * 128 + ltid
                        if idx < (_TOK - NQH) * _KB:
                            sq_st[NQH + idx // _KB, idx % _KB] = cutlass.BFloat16(0.0)
            if const_expr(self.use_tma):
                self._loader_tma(
                    atom,
                    thr_copy,
                    mQ,
                    mK,
                    tma_atom_k,
                    sQ,
                    sK,
                    pK,
                    pQ,
                    mTileNb,
                    mTokoff,
                    tiled_mma,
                    bid,
                    nctas,
                    n_work,
                    n_tiles_b,
                    s_sz,
                    ltid,
                    warp_idx,
                )
            if const_expr(not self.use_tma):
                k_issue = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.kv_stages
                )
                k_commit = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.kv_stages
                )
                q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                pending = Int32(0)
                ww = bid
                while ww < n_work:
                    work = n_work - 1 - ww
                    bi = work // n_tiles_b
                    tile = work - bi * n_tiles_b
                    t0 = tile * BQ
                    nb = mTileNb[bi, tile]
                    n_chunks = (nb + SPC - 1) // SPC
                    tok0 = Int32(0)
                    if const_expr(self.has_tokoff):
                        tok0 = Int32(mTokoff[t0])
                    n_tok = nb * R
                    if n_chunks > 0 and pending != 0:
                        # Q has a single buffer: the MMA releases it only after the previous
                        # tile's last key stage, so commit that stage before waiting for Q.
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        pK.producer_commit(k_commit)
                        k_commit.advance()
                        pending = Int32(0)
                    for pc in range(2 * n_chunks):
                        c = pc % n_chunks
                        for it in cutlass.range_constexpr(NG * NKB):
                            g = it // NKB
                            kb = it % NKB
                            pK.producer_acquire(k_issue)
                            if pc == 0 and it == 0:
                                pQ.producer_acquire(q_state)
                                self._load_q_tile(atom, thr_copy, mQ, sQ, bi, t0, s_sz, ltid)
                            sk_st = cute.composition(
                                sK[None, None, None, k_issue.index], cute.make_layout((_TOK, _KB))
                            )
                            for r0 in cutlass.range_constexpr(_TOK // 16):
                                trow = r0 * 16 + grp
                                if c * _TOK + trow < n_tok:
                                    g_ptr = (
                                        mK[bi, tok0 + c * _TOK + trow, g, None].iterator + kb * _KB
                                    )
                                    self._copy16(atom, thr_copy, g_ptr, sk_st[trow, None], j8)
                            cute.arch.cp_async_commit_group()
                            if pending != 0:
                                cute.arch.cp_async_wait_group(1)
                                cute.arch.fence_view_async_shared()
                                pK.producer_commit(k_commit)
                                k_commit.advance()
                            pending = Int32(1)
                            if pc == 0 and it == 1:
                                pQ.producer_commit(q_state)
                                q_state.advance()
                            k_issue.advance()
                    ww += nctas
                if pending != 0:
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    pK.producer_commit(k_commit)
                    k_commit.advance()

        # ================================================================ MMA
        if warp_idx == 8:
            frA_Q = tiled_mma.make_fragment_A(sQ)
            frB_K = tiled_mma.make_fragment_B(sK)
            frA_K = tiled_mma.make_fragment_A(sK)
            frB_Q = tiled_mma.make_fragment_B(sQ)
            k_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kv_stages
            )
            q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            s_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            ww = bid
            while ww < n_work:
                work = n_work - 1 - ww
                bi = work // n_tiles_b
                tile = work - bi * n_tiles_b
                nb = mTileNb[bi, tile]
                n_chunks = (nb + SPC - 1) // SPC
                if n_chunks > 0:
                    pQ.consumer_wait(q_cons)
                    for pc in range(2 * n_chunks):
                        pS.producer_acquire(s_prod)
                        for g in cutlass.range_constexpr(NG):
                            acc = cute.make_tensor(
                                tmem_ptr + (s_prod.index * NG + g) * _TOK, tCtAcc_fake.layout
                            )
                            for kb in cutlass.range_constexpr(NKB):
                                pK.consumer_wait(k_cons)
                                for kk in cutlass.range_constexpr(cute.size(frA_K, mode=[2])):
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, kb != 0 or kk != 0)
                                    if pc < n_chunks:
                                        cute.gemm(
                                            tiled_mma,
                                            acc,
                                            frA_Q[None, None, kk, g * NKB + kb],
                                            frB_K[None, None, kk, k_cons.index],
                                            acc,
                                        )
                                    else:
                                        cute.gemm(
                                            tiled_mma,
                                            acc,
                                            frA_K[None, None, kk, k_cons.index],
                                            frB_Q[None, None, kk, g * NKB + kb],
                                            acc,
                                        )
                                pK.consumer_release(k_cons)
                                k_cons.advance()
                        pS.producer_commit(s_prod)
                        s_prod.advance()
                    pQ.consumer_release(q_cons)
                    q_cons.advance()
                ww += nctas

        # ================================================================ epilogue
        if warp_idx < 4:
            cute.arch.setmaxregister_increase(_REGS_HIGH)
            epi_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
            thr_mma = tiled_mma.get_slice(0)
            # the epilogue reads the 128x128 accumulator in 128x32 column sub-tiles
            sub_layout = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((_TOK, _SUB))).layout
            tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(_SUB)), Float32)
            tmem_copy = tcgen05.make_tmem_copy(tmem_atom, cute.make_tensor(tmem_ptr, sub_layout))
            thr_t = tmem_copy.get_slice(tidx)
            cC = cute.make_identity_tensor((_TOK, _SUB))
            tTR_c = thr_t.partition_D(thr_mma.partition_C(cC))
            tTR_r = cute.make_rmem_tensor(tTR_c.shape, Float32)
            my_row = tTR_c[0][0]
            LOG2E = Float32(math.log2(math.e))
            NEG_INF = -Float32.inf
            mass = cute.make_rmem_tensor((BQ,), Float32)
            slots = cute.make_rmem_tensor((BQ,), Int32)
            st_m = cute.make_rmem_tensor((NG,), Float32)
            st_l = cute.make_rmem_tensor((NG,), Float32)
            s_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            ww = bid
            while ww < n_work:
                work = n_work - 1 - ww
                bi = work // n_tiles_b
                tile = work - bi * n_tiles_b
                t0 = tile * BQ
                nb = mTileNb[bi, tile]
                n_chunks = (nb + SPC - 1) // SPC
                if n_chunks > 0:
                    # ---- block -> slot map of the tile's rows (-1 = not picked)
                    epi_bar.arrive_and_wait()
                    for e in cutlass.range_constexpr((BQ * NBP // 2 + 127) // 128):
                        idx = e * 128 + tidx
                        if idx < BQ * NBP // 2:
                            inv32[idx] = Int32(-1)
                    epi_bar.arrive_and_wait()
                    for e in cutlass.range_constexpr((BQ * NSEL + 127) // 128):
                        idx = e * 128 + tidx
                        if idx < BQ * NSEL:
                            qi = idx // NSEL
                            sl = idx - qi * NSEL
                            if t0 + qi < s_sz:
                                if mPicked[bi, t0 + qi, sl] != cutlass.Uint8(0):
                                    inv[qi, Int32(mOrder[bi, t0 + qi, sl])] = cutlass.Int16(sl)
                    epi_bar.arrive_and_wait()
                    # ---- pass 1 (heads-major): thread = query-head row
                    qi_r = cutlass.min(my_row // HPG, BQ - 1)
                    for g in cutlass.range_constexpr(NG):
                        st_m[g] = NEG_INF
                        st_l[g] = Float32(0.0)
                    for c in range(n_chunks):
                        pS.consumer_wait(s_cons)
                        for g in cutlass.range_constexpr(NG):
                            for sub in cutlass.range_constexpr(_TOK // _SUB):
                                acc = cute.make_tensor(
                                    tmem_ptr + (s_cons.index * NG + g) * _TOK + sub * _SUB,
                                    sub_layout,
                                )
                                cute.copy(tmem_copy, thr_t.partition_S(acc), tTR_r)
                                cute.arch.fence_view_async_tmem_load()
                                cmax = NEG_INF
                                for jb in cutlass.range_constexpr(_SUB // R):
                                    okb = inv[
                                        qi_r, c * SPC + sub * (_SUB // R) + jb
                                    ] >= cutlass.Int16(0)
                                    for rr in cutlass.range_constexpr(R):
                                        j = jb * R + rr
                                        x = Float32(tTR_r[j].to(cutlass.BFloat16)) * scale
                                        x = x if okb else NEG_INF
                                        tTR_r[j] = x
                                        cmax = cute.arch.fmax(cmax, x)
                                m_old = st_m[g]
                                m_new = cute.arch.fmax(m_old, cmax)
                                m_s = m_new if m_new != NEG_INF else Float32(0.0)
                                acc_l = st_l[g] * cute.math.exp2(
                                    (m_old - m_s) * LOG2E, fastmath=True
                                )
                                for j in cutlass.range_constexpr(_SUB):
                                    acc_l = acc_l + cute.math.exp2(
                                        (tTR_r[j] - m_s) * LOG2E, fastmath=True
                                    )
                                st_m[g] = m_new
                                st_l[g] = acc_l
                        pS.consumer_release(s_cons)
                        s_cons.advance()
                    for g in cutlass.range_constexpr(NG):
                        stats[g, my_row, 0] = st_m[g] if st_m[g] != NEG_INF else Float32(0.0)
                        stats[g, my_row, 1] = Float32(1.0) / cute.arch.fmax(st_l[g], Float32(1e-10))
                    epi_bar.arrive_and_wait()
                    # ---- pass 2 (keys-major): thread = token of the chunk
                    for c in range(n_chunks):
                        blk = c * SPC + my_row // R
                        for qi in cutlass.range_constexpr(BQ):
                            slots[qi] = Int32(inv[qi, blk])
                            mass[qi] = Float32(0.0)
                        pS.consumer_wait(s_cons)
                        for g in cutlass.range_constexpr(NG):
                            for sub in cutlass.range_constexpr((NQH + _SUB - 1) // _SUB):
                                acc = cute.make_tensor(
                                    tmem_ptr + (s_cons.index * NG + g) * _TOK + sub * _SUB,
                                    sub_layout,
                                )
                                cute.copy(tmem_copy, thr_t.partition_S(acc), tTR_r)
                                cute.arch.fence_view_async_tmem_load()
                                for cc in cutlass.range_constexpr(_SUB):
                                    col = sub * _SUB + cc
                                    if const_expr(col < NQH):
                                        x = Float32(tTR_r[cc].to(cutlass.BFloat16)) * scale
                                        p = (
                                            cute.math.exp2(
                                                (x - stats[g, col, 0]) * LOG2E, fastmath=True
                                            )
                                            * stats[g, col, 1]
                                        )
                                        mass[col // HPG] = mass[col // HPG] + p
                        pS.consumer_release(s_cons)
                        s_cons.advance()
                        for qi in cutlass.range_constexpr(BQ):
                            v = mass[qi] if slots[qi] >= 0 else Float32(0.0)
                            if const_expr(self.pool):
                                for sh in cutlass.range_constexpr(int(math.log2(R))):
                                    v = cute.arch.fmax(
                                        v, cute.arch.shuffle_sync_bfly(v, offset=1 << sh)
                                    )
                                if my_row % R == 0 and slots[qi] >= 0:
                                    mOut[bi, t0 + qi, slots[qi]] = v
                            else:
                                if slots[qi] >= 0:
                                    mOut[bi, t0 + qi, slots[qi] * R + my_row % R] = v
                ww += nctas

        # ================================================================ teardown
        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


class _StudentRowSm100:
    """Sparse student (K8), one 128-thread CTA per query row (persistent, several per SM).

    For each 128-slot chunk of the row's selection, thread ``j`` (slot) loads its block key,
    computes the indexer scores ``z_h = bf16(q_h . k)`` (fp32 FMA), the student logit
    ``st = sum_h relu(z_h) / sqrt(d)`` and its KL partials; the chunk's keys are staged in
    SMEM, and thread ``d`` (head dim) accumulates the gradient sums
    ``u_h = sum_j exp(st_j - m) [z_hj > 0] k_jd`` (online-rescaled) and
    ``v_h = sum_j T_j [z_hj > 0] k_jd``. Then ``dq = w / sqrt(d) * (C u / l - v)`` with
    ``C = sum_j T_j``, ``l = sum_j exp(st_j - m)``. Outputs per row: ``dq`` (fp32), the
    student logits and positivity bits (for the block-key gradient), LSE, ``C`` and the row KL.
    """

    def __init__(self, nh, d, k, has_boff):
        assert d % 8 == 0 and d <= 256 and nh <= 8 and k <= 2048
        self.nh, self.d, self.k, self.has_boff = nh, d, k, has_boff
        self.dpt = (d + 127) // 128  # head-dim elements per thread in the gradient phase
        self.ld = d + 8  # SMEM row stride (bf16 elements) of the staged keys

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # [b, s, nh, d] bf16
        mBK: cute.Tensor,  # [b, nb, d] bf16
        mOrder: cute.Tensor,  # [b, s, k] int64
        mPicked: cute.Tensor,  # [b, s, k] uint8
        mBoff: cute.Tensor,  # [s] int64 (dummy when absent)
        mNhi: cute.Tensor,  # [b, s] int32
        mTgt: cute.Tensor,  # [b, s, k] fp32
        mW: cute.Tensor,  # [1] fp32  (loss_coeff / n_valid_rows)
        mDQ: cute.Tensor,  # [b, s, nh, d] fp32 out
        mGH: cute.Tensor,  # [b, s, nh, k] bf16 out: d loss / d score_h at each slot
        mRow: cute.Tensor,  # [b, s, 4] fp32 out: lse, csum, row KL, -
        scale: Float32,  # 1 / sqrt(d)
        n_ctas: Int32,
        stream,
    ):
        self.kernel(mQ, mBK, mOrder, mPicked, mBoff, mNhi, mTgt, mW, mDQ, mGH, mRow, scale).launch(
            grid=(n_ctas, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.jit
    def _block_reduce(self, val, red, slot, is_max: cutlass.Constexpr):
        """CTA-wide (128 threads) max / sum; ``red`` holds 4 floats per ``slot``."""
        if const_expr(is_max):
            wv = cute.arch.warp_reduction_max(val)
        else:
            wv = cute.arch.warp_reduction_sum(val)
        tidx, _, _ = cute.arch.thread_idx()
        if tidx % 32 == 0:
            red[slot, tidx // 32] = wv
        cute.arch.sync_threads()
        r = red[slot, 0]
        for w in cutlass.range_constexpr(1, 4):
            if const_expr(is_max):
                r = cute.arch.fmax(r, red[slot, w])
            else:
                r = r + red[slot, w]
        return r

    @cute.jit
    def _issue_rows(self, mBK, mOrder, mPicked, sbk, bi, t, c, nhi, boff, nb, tidx):
        # cp.async this thread's key row of chunk c (16 B pieces; zero-filled if unpicked)
        j = c * 128 + tidx
        ok = j < nhi
        blk = Int32(0)
        if ok:
            ok = mPicked[bi, t, j] != cutlass.Uint8(0)
            blk = cutlass.min(Int32(mOrder[bi, t, j]) + boff, nb - 1)
        nbytes = Int32(16) if ok else Int32(0)
        g_base = mBK[bi, blk, None].iterator
        s_base = sbk[c % 2, tidx, None].iterator
        for i in cutlass.range_constexpr(self.d // 8):
            _cp_async_shared_global(
                s_base + i * 8, g_base + i * 8, 16, "cg", cp_size=nbytes
            )
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(  # pylint: disable=missing-function-docstring
        self, mQ, mBK, mOrder, mPicked, mBoff, mNhi, mTgt, mW, mDQ, mGH, mRow, scale: Float32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()
        b_sz = mOrder.shape[0]
        s_sz = mOrder.shape[1]
        n_rows = b_sz * s_sz
        nb = mBK.shape[1]
        NH = const_expr(self.nh)
        D = const_expr(self.d)
        DPT = const_expr(self.dpt)
        LD = const_expr(self.ld)
        LOG2E = Float32(math.log2(math.e))
        LN2 = Float32(math.log(2.0))
        NEG_INF = -Float32.inf

        @cute.struct
        class SharedStorage:  # pylint: disable=missing-class-docstring
            sq: cute.struct.MemRange[Float32, self.nh * self.d]
            wgt: cute.struct.MemRange[Float32, 128 * 2 * self.nh]
            red: cute.struct.MemRange[Float32, 8 * 4]
            sst: cute.struct.MemRange[Float32, self.k]
            spos: cute.struct.MemRange[cutlass.Uint8, self.k]
            sbk: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, 2 * 128 * self.ld], 16]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sq = storage.sq.get_tensor(cute.make_layout((NH, D), stride=(D, 1)))
        wgt = storage.wgt.get_tensor(cute.make_layout((128, 2, NH), stride=(2 * NH, NH, 1)))
        red = storage.red.get_tensor(cute.make_layout((8, 4), stride=(4, 1)))
        sbk = storage.sbk.get_tensor(cute.make_layout((2, 128, LD), stride=(128 * LD, LD, 1)))
        sst = storage.sst.get_tensor(cute.make_layout((self.k,)))
        spos = storage.spos.get_tensor(cute.make_layout((self.k,)))
        KS = const_expr(self.k)
        z = cute.make_rmem_tensor((NH,), Float32)
        u = cute.make_rmem_tensor((NH * DPT,), Float32)
        v = cute.make_rmem_tensor((NH * DPT,), Float32)
        w_loss = mW[0]
        inv_sqrt_d = scale

        rr = bid
        while rr < n_rows:
            row = n_rows - 1 - rr
            bi = row // s_sz
            t = row - bi * s_sz
            nhi = mNhi[bi, t]
            boff = Int32(0)
            if const_expr(self.has_boff):
                boff = Int32(mBoff[t])
            for e in cutlass.range_constexpr((NH * D + 127) // 128):
                idx = e * 128 + tidx
                if idx < NH * D:
                    sq[idx // D, idx % D] = Float32(mQ[bi, t, idx // D, idx % D])
            for i in cutlass.range_constexpr(NH * DPT):
                u[i] = Float32(0.0)
                v[i] = Float32(0.0)
            m_run = NEG_INF
            l_t = Float32(0.0)
            a_t = Float32(0.0)
            b_t = Float32(0.0)
            c_t = Float32(0.0)
            cute.arch.sync_threads()
            n_chunks = (nhi + 127) // 128
            # key rows of chunk c are cp.async'ed into sbk[c % 2] one chunk ahead (zero-filled
            # for unpicked slots, so the gradient phase multiplies exact zeros)
            if n_chunks > 0:
                self._issue_rows(mBK, mOrder, mPicked, sbk, bi, t, Int32(0), nhi, boff, nb, tidx)
            for c in range(n_chunks):
                buf = c % 2
                if c + 1 < n_chunks:
                    self._issue_rows(mBK, mOrder, mPicked, sbk, bi, t, c + 1, nhi, boff, nb, tidx)
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                j = c * 128 + tidx
                ok = j < nhi
                if ok:
                    ok = mPicked[bi, t, j] != cutlass.Uint8(0)
                st = Float32(0.0)
                tg = Float32(0.0)
                posb = Int32(0)
                if ok:
                    tg = mTgt[bi, t, j]
                    for h in cutlass.range_constexpr(NH):
                        z[h] = Float32(0.0)
                    for dd in cutlass.range_constexpr(D):
                        kv = Float32(sbk[buf, tidx, dd])
                        for h in cutlass.range_constexpr(NH):
                            z[h] = z[h] + sq[h, dd] * kv
                    for h in cutlass.range_constexpr(NH):
                        zb = Float32(z[h].to(cutlass.BFloat16))
                        if zb > Float32(0.0):
                            st = st + zb
                            posb = posb | Int32(1 << h)
                    st = st * inv_sqrt_d
                if j < nhi:
                    sst[j] = st
                    spos[j] = cutlass.Uint8(posb)
                a_t = a_t + tg * cute.math.log(cute.arch.fmax(tg, Float32(1e-10)))
                b_t = b_t + (tg * st if ok else Float32(0.0))
                c_t = c_t + tg
                cm = self._block_reduce(st if ok else NEG_INF, red, c % 2, True)
                m_new = cute.arch.fmax(m_run, cm)
                m_s = m_new if m_new != NEG_INF else Float32(0.0)
                alpha = cute.math.exp2((m_run - m_s) * LOG2E, fastmath=True)
                ev = cute.math.exp2((st - m_s) * LOG2E, fastmath=True) if ok else Float32(0.0)
                l_t = l_t * alpha + ev
                for h in cutlass.range_constexpr(NH):
                    hit = (posb >> h) & 1
                    wgt[tidx, 0, h] = ev if hit != 0 else Float32(0.0)
                    wgt[tidx, 1, h] = tg if hit != 0 else Float32(0.0)
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(NH * DPT):
                    u[i] = u[i] * alpha
                for jj in cutlass.range(128, unroll=4):
                    for p in cutlass.range_constexpr(DPT):
                        dd = p * 128 + tidx
                        if dd < D:
                            kv = Float32(sbk[buf, jj, dd])
                            for h in cutlass.range_constexpr(NH):
                                u[p * NH + h] = u[p * NH + h] + wgt[jj, 0, h] * kv
                                v[p * NH + h] = v[p * NH + h] + wgt[jj, 1, h] * kv
                cute.arch.sync_threads()
                m_run = m_new
            L = self._block_reduce(l_t, red, 2, False)
            A = self._block_reduce(a_t, red, 3, False)
            B = self._block_reduce(b_t, red, 4, False)
            C = self._block_reduce(c_t, red, 5, False)
            has = L > Float32(0.0)
            m_s = m_run if m_run != NEG_INF else Float32(0.0)
            lse = m_s + cute.math.log2(cute.arch.fmax(L, Float32(1e-30)), fastmath=False) * LN2
            lse = lse if has else Float32(0.0)
            inv_l = Float32(1.0) / cute.arch.fmax(L, Float32(1e-30))
            gsc = w_loss * inv_sqrt_d
            for p in cutlass.range_constexpr(DPT):
                dd = p * 128 + tidx
                if dd < D:
                    for h in cutlass.range_constexpr(NH):
                        g = gsc * (C * u[p * NH + h] * inv_l - v[p * NH + h])
                        mDQ[bi, t, h, dd] = g if has else Float32(0.0)
            if tidx == 0:
                mRow[bi, t, 0] = lse
                mRow[bi, t, 1] = C
                mRow[bi, t, 2] = (A - B + C * lse) if has else Float32(0.0)
            # per-slot score gradient (bf16, the block-key GEMM operand):
            # d loss / d s_hj = w / sqrt(d) * (p_j * C - T_j) * [z_hj > 0]
            for jj in cutlass.range_constexpr((KS + 127) // 128):
                j = jj * 128 + tidx
                if j < KS:
                    gv = Float32(0.0)
                    pb = Int32(0)
                    if j < nhi:
                        if mPicked[bi, t, j] != cutlass.Uint8(0):
                            pj = cute.math.exp2((sst[j] - lse) * LOG2E, fastmath=True)
                            gv = gsc * (pj * C - mTgt[bi, t, j])
                            pb = Int32(spos[j])
                    if not has:
                        gv = Float32(0.0)
                    for h in cutlass.range_constexpr(NH):
                        mGH[bi, t, h, j] = (gv if ((pb >> h) & 1) != 0 else Float32(0.0)).to(
                            cutlass.BFloat16
                        )
            cute.arch.sync_threads()
            rr += nctas


# ---------------------------------------------------------------------------- host side
_COMPILED = {}
_NUM_SMS = {}
TILE_MAX_BLOCKS = 1024  # tile kernel only when a document has at most this many blocks
ROW_KV_STAGES, ROW_LAG = 10, 8  # row kernel: key stages, cp.async groups in flight
_SMEM_BYTES = 227 * 1024


def _num_sms(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _NUM_SMS:
        _NUM_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _NUM_SMS[idx]


def _to_cute(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=t.dim() - 1
    )


def teacher_tmem_columns(n_sel: int, compress_ratio: int, ng: int, hpg: int) -> int:
    """TMEM columns the row kernel needs to keep every logit of a row resident (<= 512)."""
    return -(-n_sel // (_TOK // compress_ratio)) * ng * _pad16(hpg)


def _tile_plan(q, k, n_sel, r):
    """``(bq, nbp, kv_stages)`` of the tile kernel, or None when it does not apply."""
    d = q.shape[-1]
    ng = k.shape[2]
    hpg = q.shape[2] // ng
    spc = _TOK // r
    nbp = -(-(-(-k.shape[1] // r)) // spc) * spc
    if hpg > _TOK or 2 * ng * _TOK > _TMEM_COLS or nbp > TILE_MAX_BLOCKS:
        return None
    bq = _TOK // hpg
    for kv_stages in (4, 3, 2):
        smem = (
            (ng * (d // _KB) + kv_stages) * _TOK * _KB * 2 + bq * nbp * 2 + 2 * ng * _TOK * 4 + 1024
        )
        if smem <= _SMEM_BYTES:
            return bq, nbp, kv_stages
    return None


def _launch(key, make_kernel, args, scale, n_ctas, stream):
    if key not in _COMPILED:
        _COMPILED[key] = cute.compile(
            make_kernel(),
            *[_to_cute(a) for a in args],
            Float32(scale),
            Int32(n_ctas),
            stream,
            options="--enable-tvm-ffi",
        )
    _COMPILED[key](*args, Float32(scale), Int32(n_ctas), stream)


def qsa_indexer_teacher(
    q: torch.Tensor,
    k: torch.Tensor,
    order: torch.Tensor,
    picked: torch.Tensor,
    compress_ratio: int,
    scale: float,
    token_offsets: Optional[torch.Tensor] = None,
    pool: bool = True,
    kernel: str = "auto",
) -> torch.Tensor:
    """Teacher targets of the indexer loss.

    Args:
        q: ``[b, s, n_heads, d]`` bf16 query (main path, after RoPE / norm).
        k: ``[b, t, n_groups, d]`` bf16 key.
        order / picked: ``[b, s, n_sel]`` the indexer's selected block ids and validity.
        compress_ratio: tokens per block ``R``.
        scale: softmax scale.
        token_offsets: ``[s]`` first token of each row's document (packed THD), or None.
        pool: max-pool the token mass to blocks (``[b, s, n_sel]``); else return it per
            token (``[b, s, n_sel, R]``, tensor parallelism reduces heads before pooling).
        kernel: ``"auto"`` (tile kernel where it applies, row kernel for the other rows) or
            ``"row"`` (row kernel only; tests).
    """
    import cuda.bindings.driver as cuda_drv

    assert q.dtype == k.dtype == torch.bfloat16, "the teacher kernels take bf16 q / k"
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "q / k must be contiguous in the head dim"
    assert kernel in ("auto", "row"), kernel
    b, s, n_heads, d = q.shape
    ng = k.shape[2]
    hpg = n_heads // ng
    n_sel = order.shape[-1]
    r = int(compress_ratio)
    device = q.device
    order = order.contiguous()
    picked_u8 = picked.contiguous().view(torch.uint8)
    slot = torch.arange(1, n_sel + 1, device=device, dtype=torch.int32)
    n_hi = torch.where(picked, slot, 0).amax(dim=-1).to(torch.int32).contiguous()
    if pool:
        out = torch.zeros(b, s, n_sel, dtype=torch.float32, device=device)
    else:
        out = torch.zeros(b, s, n_sel * r, dtype=torch.float32, device=device)
    has_tokoff = token_offsets is not None
    tokoff = (
        token_offsets.to(torch.int64).contiguous()
        if has_tokoff
        else torch.zeros(1, dtype=torch.int64, device=device)
    )
    if b * s > 0 and n_sel > 0:
        stream = cuda_drv.CUstream(torch.cuda.current_stream(device).cuda_stream)
        sms = _num_sms(device)
        plan = _tile_plan(q, k, n_sel, r) if kernel == "auto" else None
        if plan is not None:
            bq, nbp, kv_stages = plan
            n_tiles = -(-s // bq)
            pad = n_tiles * bq - s
            row_max = torch.where(picked, order, -1).amax(dim=-1)  # [b, s] last picked block
            if pad:
                row_max = torch.nn.functional.pad(row_max, (0, pad), value=-1)
            tile_nb = (row_max.view(b, n_tiles, bq).amax(dim=-1) + 1).to(torch.int32)
            if has_tokoff:
                # a tile runs here only if all its rows are in one document
                tk = tokoff
                if pad:
                    tk = torch.cat([tk, tk[-1:].expand(pad)])
                tk = tk.view(n_tiles, bq)
                uniform = tk.amin(dim=-1) == tk.amax(dim=-1)
                tile_nb = torch.where(uniform.view(1, n_tiles), tile_nb, 0)
                n_hi = torch.where(uniform.repeat_interleave(bq)[:s].view(1, s), 0, n_hi)
            tile_nb = tile_nb.contiguous()
            _launch(
                ("tile", ng, hpg, d, r, n_sel, nbp, bool(pool), has_tokoff, kv_stages),
                lambda: _TeacherTileSm100(
                    ng, hpg, d, r, n_sel, nbp, bool(pool), has_tokoff, kv_stages, True
                ),
                (q, k, order, picked_u8, tile_nb, tokoff, out),
                scale,
                min(b * n_tiles, sms),
                stream,
            )
        if plan is None or has_tokoff:
            # rows not covered by a tile (all rows without the tile kernel)
            _launch(
                ("row", ng, hpg, d, r, n_sel, bool(pool), has_tokoff),
                lambda: _TeacherRowSm100(
                    ng, hpg, d, r, n_sel, bool(pool), has_tokoff, ROW_KV_STAGES, ROW_LAG
                ),
                (q, k, order, picked_u8, n_hi, tokoff, out),
                scale,
                min(b * s, sms),
                stream,
            )
    if pool:
        return out
    return out.view(b, s, n_sel, r)


class QSASparseStudentKLCute(torch.autograd.Function):
    """KL(targets || softmax(student)) at the selected blocks with the CuTe row kernel;
    gradients computed in forward (deterministic, no atomics), scaled in backward."""

    @staticmethod
    def forward(ctx, q_idx, bk, order, picked, block_offsets, targets, loss_coeff, chunk, nb_hi_fn):
        """Run the student kernel: the loss, and the q / block-key gradients for backward."""
        import cuda.bindings.driver as cuda_drv

        b, s, nh, d = q_idx.shape
        nb = bk.shape[1]
        k = order.shape[-1]
        device = q_idx.device
        order = order.contiguous()
        picked_u8 = picked.contiguous().view(torch.uint8)
        slot = torch.arange(1, k + 1, device=device, dtype=torch.int32)
        n_hi = torch.where(picked, slot, 0).amax(dim=-1).to(torch.int32).contiguous()
        valid_rows = picked.any(dim=-1)
        w = (loss_coeff / valid_rows.sum().clamp_min(1).float()).reshape(1).contiguous()
        targets = targets.contiguous()
        dq = torch.empty(b, s, nh, d, dtype=torch.float32, device=device)
        gh = torch.empty(b, s, nh, k, dtype=q_idx.dtype, device=device)
        rowv = torch.empty(b, s, 4, dtype=torch.float32, device=device)
        has_boff = block_offsets is not None
        boff = (
            block_offsets.to(torch.int64).contiguous()
            if has_boff
            else torch.zeros(1, dtype=torch.int64, device=device)
        )
        sqrt_d = math.sqrt(d)
        if s > 0 and b > 0 and k > 0 and nb > 0:
            stream = cuda_drv.CUstream(torch.cuda.current_stream(device).cuda_stream)
            key = ("student", nh, d, k, has_boff)
            args = (q_idx, bk, order, picked_u8, boff, n_hi, targets, w, dq, gh, rowv)
            n_ctas = min(b * s, _num_sms(device) * STUDENT_CTAS_PER_SM)
            if key not in _COMPILED:
                _COMPILED[key] = cute.compile(
                    _StudentRowSm100(nh, d, k, has_boff),
                    *[_to_cute(a) for a in args],
                    Float32(1.0 / sqrt_d),
                    Int32(n_ctas),
                    stream,
                    options="--enable-tvm-ffi",
                )
            _COMPILED[key](*args, Float32(1.0 / sqrt_d), Int32(n_ctas), stream)
        else:
            dq.zero_()
            gh.zero_()
            rowv.zero_()
        loss = rowv[..., 2].sum() * w[0]

        # d loss / d block keys: per query chunk, the kernel's per-(row, head, slot) gradient is
        # scattered into a dense [rows*heads, blocks] matrix of the chunk's block range (a row
        # picks a block once; unpicked slots carry exact zeros, so the scatter-add is exact and
        # order-independent), then one GEMM against the queries; chunk results accumulate in
        # fp32. The dense buffer is allocated once and cleared by scattering zeros back.
        blk_all = order
        if has_boff:
            blk_all = blk_all + block_offsets.view(1, s, 1)
        blk_all = torch.where(picked, blk_all, 0)
        dbk = torch.zeros(b, nb, d, dtype=torch.float32, device=device)
        c_max = min(chunk, s)
        dense_buf = torch.zeros(b, c_max * nh * nb, dtype=q_idx.dtype, device=device)
        for c0 in range(0, s, chunk):
            c1 = min(c0 + chunk, s)
            c = c1 - c0
            nb_hi = nb_hi_fn(c1)
            if nb_hi <= 0:
                continue
            dense = dense_buf[:, : c * nh * nb_hi].view(b, c, nh, nb_hi)
            idx = blk_all[:, c0:c1, None, :].clamp_max(nb_hi - 1).expand(b, c, nh, k)
            dense.scatter_add_(3, idx, gh[:, c0:c1])
            dbk[:, :nb_hi] += torch.matmul(
                dense.view(b, c * nh, nb_hi).transpose(1, 2), q_idx[:, c0:c1].reshape(b, c * nh, d)
            ).float()
            dense.scatter_(3, idx, 0.0)
        ctx.save_for_backward(dq, dbk)
        ctx.dtypes = (q_idx.dtype, bk.dtype)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        """Scale the gradients computed in the forward by the incoming loss gradient."""
        dq, dbk = ctx.saved_tensors
        dq = (dq * grad_loss).to(ctx.dtypes[0])
        dbk = (dbk * grad_loss).to(ctx.dtypes[1])
        return dq, dbk, None, None, None, None, None, None, None


STUDENT_CTAS_PER_SM = 3  # 70 KB of double-buffered key rows per CTA


def qsa_indexer_student_sparse(
    q_idx: torch.Tensor,
    block_keys: torch.Tensor,
    order: torch.Tensor,
    picked: torch.Tensor,
    targets: torch.Tensor,
    loss_coeff: float,
    block_offsets: Optional[torch.Tensor] = None,
    chunk: int = 1024,
    nb_hi_fn=None,
) -> torch.Tensor:
    """Sparse-KL student loss of the indexer: ``KL(targets || softmax(student))`` at the
    selected blocks, mean over rows with a selection, times ``loss_coeff``.

    Args:
        q_idx: ``[b, s, nh, d]`` bf16 indexer queries; block_keys: ``[b, n_blocks, d]`` bf16.
        order / picked: ``[b, s, k]`` selected block ids (document-local when
            ``block_offsets`` ``[s]`` is given) and validity; targets: ``[b, s, k]`` teacher.
        chunk / nb_hi_fn: query chunk of the block-key gradient GEMM, and the number of
            blocks a chunk ending at row ``c1`` can read (default: all).
    """
    assert q_idx.dtype == block_keys.dtype == torch.bfloat16, "the student kernel takes bf16"
    assert block_keys.stride(-1) == 1 and block_keys.stride(1) % 8 == 0
    nb = block_keys.shape[1]
    if nb_hi_fn is None:
        nb_hi_fn = lambda c1: nb  # noqa: E731
    return QSASparseStudentKLCute.apply(
        q_idx, block_keys, order, picked, block_offsets, targets, loss_coeff, chunk, nb_hi_fn
    )
