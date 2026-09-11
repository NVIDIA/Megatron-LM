# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""Warp-specialized tcgen05 (Blackwell) fused CP backward ``dhu`` pre-process.

Reverse-direction sibling of :mod:`fused_gdr_cp_cute.fused_ws`.  The shipped
non-warp-specialized backward (:mod:`fused_gdr_cp_cute.bwd_fused`) inherits the
SM80 warp-MMA engine; on the current stack it measures 1027.70 us against
1020.94 us for exact FLA at the canonical CP2 shape -- i.e. no speedup at all.
The forward received the Blackwell rewrite; the backward never did.

Design provenance
-----------------
``the CuTe C++ reference kernel`` (CuTe C++) already proved the schedule.  Its
iteration 3b -- a naive transliteration of the forward WS kernel -- scored no
better than the SM80 engine: NCU showed 8.01% tensor utilization, five
serialized TMA waits, three UMMA waits and 13 CTA barriers per reverse chunk,
with 47.3% of PC samples parked on barriers.  Backward does only 1.25x the
tensor work and 1.60x the TMA bytes of the forward, so the gap was a
*scheduling* gap.  Iterations 3c/3d fixed it and reached 342.40 us.  This
module implements that schedule in CuTeDSL.

MATH (per value head hv; reverse scan over NT = ceil(T/64) chunks of BT=64)
--------------------------------------------------------------------------
State ``X = [H_tilde | P]`` in R^{K x (V+K)} fp32, held in TENSOR MEMORY as
the accumulator of gemm2/gemm3.  ``H_tilde(0) = 0``, ``P(0) = I``.  Per
reverse chunk, with log2-domain chunk-local cumulative gates ``g``::

    lambda_i = 2^(g_last - g_i)                 row decays (also the tail mask)
    gamma    = 2^(g_last)                       chunk decay (gk: per-K-row)
    src_i    = scale * 2^(g_i)                  gemm3 source scale

    A^T      = X^T @ K_t^T                                          (gemm1)
    y^T      = lambda (*) A^T  + [dV^T | 0]     (the P half has no dV term)
    X       <- gamma * X + W_t^T @ (-y)                              (gemm2)
    X_H     += Q_t^T @ (src (*) dO)^T                    (gemm3, H half only)

After NT reverse chunks, ``Z_left = P @ Z + H_tilde`` for any gradient ``Z``
entering the shard's right boundary -- exactly fla's backward pre-process.
The P half carries neither the dV term nor gemm3, mirroring the forward's M
half having no v term.

SMEM (why the overlay is mandatory, not an optimization)
-------------------------------------------------------
B200 allows 232,448 B of dynamic SMEM per CTA.  The backward needs five
operand tiles where the forward needs three: K maps onto the forward's W
slot, W^T onto its K^T slot and dV onto its V slot, but Q^T and dO are new
and cost +32 KB two-staged, against ~32 KB of headroom that is already spoken
for by the third x^T stage.  ``sX``'s pair-0 half is DEAD once gemm1(H) has
consumed it, so Q^T and dO land there and cost zero extra bytes.  The write
is gated by a dedicated gemm1(H)-retired UMMA edge (``p_g1b``).

Buffer renaming vs the forward (byte-identical shapes, so every layout,
TMA atom and fragment of the forward transfers verbatim):

===========  ==========================  =========================
forward      backward                    shape
===========  ==========================  =========================
sW           K tile        (gemm1 B)     [BT, K] k-major  x2 stages
sKt          W^T tile      (gemm2 A)     [K, BT] mn-major x2 stages
sV           dV tile       (elementwise) [BT, V]          x2 stages
sX           published X^T (gemm1 A)     [K, V+K], 4 strips
  strip 0    + Q^T overlay (gemm3 A)     [K, BT] mn-major x1
  strip 1    + dO overlay  (elementwise) [BT, V]          x1
sx           -y^T          (gemm2 B)     [NS*SPP, BT] k-major, stages 0/1
  stage 2    + (src*dO)^T  (gemm3 B)     [V, BT]      k-major
===========  ==========================  =========================

Total ~212 KB, inside the 227 KB cap with ~15 KB spare.

Warp roles (384 threads = 12 warps, K=V=128)
--------------------------------------------
=====  ================================================================
warps  role
=====  ================================================================
0-3    H pair (H_tilde strips): TMEM drain-publish, y, (src*dO)^T
4-7    P pair (P strips):       TMEM drain-publish, y
8      front TMA producer (K, W^T, dV, g/gk) -- two stages
9      sole tcgen05 issuer (gemm1 x2, gemm2 x2, gemm3)
10     back TMA producer (Q^T, dO) into the dead pair-0 sX overlay
11     lambda / src gate tables
=====  ================================================================

The reverse loop uses only pipeline (phase) barriers; the sole full-CTA
rendezvous are the item boundary and the epilogue.  Every MMA is issued by
one warp (tcgen05 issue and tcgen05.commit are elect-one-per-warp; multi-warp
issue duplicates MMAs and over-arrives the 1-count producer mbarriers).

Restrictions: sm_100a, K == V == 128, B = 1, BT = 64, gate modes
none/g/gk, no DPLR.  Other shapes fall back to the non-WS backward.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .bwd_fused import _FLA_WRAPPER_CACHE  # noqa: F401
from .bwd_fused import BT, CuteDSLFusedCPBwdPreProcess, PreProcessBwdFused
from .bwd_fused import chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl_fused as _bwd_entry
from .fused_ws import PreProcessFwdFusedWS, tmem_alloc_ptr

# _FLA_WRAPPER_CACHE is re-exported, not used here: benchmark harnesses find
# the live wrapper through `candidate.__module__` to inspect the published
# [Htilde|P] wire payload.

NS = 64                # strip width
VGS = 2                # front dV/gate pipeline stages
FRONT_STAGES = 2       # K / W^T pipeline stages

# named barrier ids (0 is reserved for sync_threads)
NB_MERGE = 1           # merge role work group (128 threads)
NB_EPILOG = 2          # compute item epilogue (all threads)
NB_TMEM_FREE = 3
NB_TMEM_ALLOC = 4
NB_ITEM = 5            # item boundaries (all threads)


class PreProcessBwdFusedWS(PreProcessBwdFused, PreProcessFwdFusedWS):
    """Warp-specialized reverse producer with peer push and suffix fold."""

    def __init__(self, H, HV, K, V, gate_mode, R, rank, n_sm=None,
                 full_chunks=False):
        # Backward protocol/geometry first (peer counts, NSTRIPS, NB, ...).
        PreProcessBwdFused.__init__(self, H, HV, K, V, gate_mode, R, rank)
        assert K == V == 128, "WS backward is specialized for K == V == 128"
        assert gate_mode in ("none", "g", "gk"), gate_mode
        if self.use_g:
            assert HV % 4 == 0, "WS g-mode needs HV % 4 == 0 (TMA box)"

        self.NG = self.NSTRIPS               # 4
        self.SPP = max(self.NSTRIPS // 2, 1)  # 2 strips per MMA pair
        self.num_threads = 384
        self.tma_warp = 8
        self.mma_warp = 9
        self.bk_warp = 10
        self.lam_warp = 11
        self.full_chunks = full_chunks
        # lambda doubles as the ragged-tail validity mask, so an aligned
        # ungated call compiles the whole table away.
        self.need_lam = self.use_g or not full_chunks
        # dV is NOT lambda-scaled, so unlike the forward's v term it needs an
        # explicit tail mask -- but only when a partial chunk can exist.  On
        # an aligned call the compare/select pair disappears from the hot
        # pass-2 loop (the same full-chunk specialization the forward uses).
        self.need_mask = not full_chunks
        # Does any rank read this rank's transition (P) half?  A consumer folds
        # P_j only for j <= source_hi - 1 and source_hi never exceeds R-1, so
        # the top rank's is dead for every world size and every chain.  See
        # `_pair_item`.
        self.push_transition = rank != R - 1
        # ... and an unread transition need not be computed either.  Warp
        # groups of 4 map to MMA pairs (pair 0 = the Htilde strips, pair 1 =
        # the P strips) and the WS backward is K == V == 128, so the
        # transition is exactly the upper half.  Dropping it at the top rank
        # removes gemm1(P) and gemm2(P) from every chunk of the reverse scan.
        self.n_groups = 2
        self.n_groups_live = 2 if self.push_transition else 1
        self.need_decay = self.use_g or self.use_gk
        # gemm3's per-token source scale is a second table only under
        # scalar g; otherwise src == scale * lambda exactly (lambda is 1/0).
        self.need_src_tab = self.use_g

        if n_sm is None:
            n_sm = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        self.n_sm = n_sm
        # Reverse direction: rank r produces for consumers 0..r-1 and merges
        # sources R-1..r+1 -- the mirror of the forward's counts.
        self.n_citems = HV if rank > 0 else 0
        self.n_mitems = HV * self.NB if rank < R - 1 else 0
        self.n_items = self.n_citems + self.n_mitems
        # A rank whose local sequence neither enters nor leaves this CP shard
        # still participates in symmetric-buffer init; one idle CTA keeps the
        # launch legal (both role blocks compile away).  Mirrors
        # PreProcessBwdFused.n_grid = max(n_work, 1).
        self.n_ctas = max(min(n_sm, self.n_items), 1)
        nwork = max(1, self.n_citems)
        eff0 = max(1, min(self.n_ctas, nwork))
        self.NSLOT = -(-nwork // eff0)

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        pQ: cute.Pointer,      # [1,T,H,K]  bf16
        pK: cute.Pointer,      # [1,T,H,K]  bf16
        pW: cute.Pointer,      # [1,T,HV,K] bf16
        pDO: cute.Pointer,     # [1,T,HV,V] bf16
        pDV: cute.Pointer,     # [1,T,HV,V] bf16
        pG: cute.Pointer,      # [1,T,HV] fp32 or dummy
        pGK: cute.Pointer,     # [1,T,HV,K] fp32 or dummy
        pHMloc: cute.Pointer,  # local symmetric [2,R,1,HV,NG,K,64] fp32
        pPtrs: cute.Pointer,   # int64 [3,R] rows hm/flags/acks
        pDHT: cute.Pointer,    # [HV,K,V] fp32 canonical output
        scale: cutlass.Float32,
        T: cutlass.Int32,
        step: cutlass.Int32,
        emit: cutlass.Int32,   # bit 0 = push Htilde half, bit 1 = push P half
        stream: cuda.CUstream,
    ):
        H, HV, K, V = self.H, self.HV, self.K, self.V
        NG, SPP = self.NG, self.SPP
        dtype = self.io_dtype

        # gmem views shaped for the TMA atoms.  K and Q are [T,H,K]; the
        # mn-major views transpose in the descriptor (stride 1 on the K mode)
        # exactly as the forward does for K^T.
        mQt3 = cute.make_tensor(
            pQ, cute.make_layout((K, T, H), stride=(1, H * K, K)))
        mWt3 = cute.make_tensor(
            pW, cute.make_layout((K, T, HV), stride=(1, HV * K, K)))
        mK3 = cute.make_tensor(
            pK, cute.make_layout((T, K, H), stride=(H * K, 1, K)))
        mDV3 = cute.make_tensor(
            pDV, cute.make_layout((T, V, HV), stride=(HV * V, 1, V)))
        mDO3 = cute.make_tensor(
            pDO, cute.make_layout((T, V, HV), stride=(HV * V, 1, V)))
        mG2 = cute.make_tensor(pG, cute.make_layout((T, HV), stride=(HV, 1)))
        mGk2 = cute.make_tensor(
            pGK, cute.make_layout((T * HV, K), stride=(K, 1)))
        mPtrs = cute.make_tensor(
            pPtrs, cute.make_layout((3, self.R), stride=(self.R, 1)))
        mDHT = cute.make_tensor(
            pDHT, cute.make_layout((HV, K, V), stride=(K * V, V, 1)))

        # ---- tcgen05 tiled MMAs (strip-width N=64), flipped "v5" shapes
        def mk_mma(amaj, bmaj, shape):
            return sm100_utils.make_trivial_tiled_mma(
                dtype, tcgen05.OperandMajorMode(amaj),
                tcgen05.OperandMajorMode(bmaj), cutlass.Float32,
                tcgen05.CtaGroup.ONE, shape, tcgen05.OperandSource.SMEM,
            )

        # Base atoms, only used to build the operand smem layouts (they carry
        # the canonical A/B tile shapes that the TMA descriptors need).
        tiled_mma1 = mk_mma("k", "mn", (BT, NS))    # A = [BT,K] k-major
        tiled_mma2 = mk_mma("mn", "mn", (K, NS))    # A = [K,BT] mn-major
        tiler1 = (BT, NS, K)
        tiler2 = (K, NS, BT)

        # gemm1: acc1^T = X^T @ K_t^T   A = X^T [NS*SPP,K] mn-major,
        #                               B = K_t [BT,K] k-major
        tiled_mma1f = mk_mma("mn", "k", (NS * SPP, BT))
        # gemm2: X += W_t^T @ (-y)      A = W^T [K,BT] mn-major,
        #                               B = -y^T [NS*SPP,BT] k-major
        # gemm3 shares gemm2's shape:   A = Q^T,  B = (src*dO)^T
        tiled_mma2f = mk_mma("mn", "k", (K, NS * SPP))
        tiler1f = (NS * SPP, BT, K)
        tiler2f = (K, NS * SPP, BT)

        # ---- operand smem layouts (identical to the forward's; only the
        # tensor that lands in each buffer changes)
        sK_lay = sm100_utils.make_smem_layout_a(
            tiled_mma1, tiler1, dtype, FRONT_STAGES)      # K tile
        sWt_lay = sm100_utils.make_smem_layout_a(
            tiled_mma2, tiler2, dtype, FRONT_STAGES)      # W^T tile
        sDV_lay = sm100_utils.make_smem_layout_a(
            tiled_mma1, tiler1, dtype, VGS)               # dV tile
        # 1-stage siblings for the dead-sX overlay
        sQt_lay = sm100_utils.make_smem_layout_a(tiled_mma2, tiler2, dtype, 1)
        sDO_lay = sm100_utils.make_smem_layout_a(tiled_mma1, tiler1, dtype, 1)
        # published state: strip index rides the "stage" mode of the B layout
        sX_lay = sm100_utils.make_smem_layout_b(tiled_mma1, tiler1, dtype, NG)
        # MMA-side views over the same bytes
        sXf_lay = sm100_utils.make_smem_layout_a(tiled_mma1f, tiler1f, dtype, 2)
        sKb_lay = sm100_utils.make_smem_layout_b(tiled_mma1f, tiler1f, dtype,
                                                 FRONT_STAGES)
        # three -y^T / (src*dO)^T stages: 0 = H pair, 1 = P pair, 2 = gemm3 B
        syt_lay = sm100_utils.make_smem_layout_b(tiled_mma2f, tiler2f, dtype, 3)
        sG_lay = cute.make_layout((BT, 4, VGS), stride=(4, 1, BT * 4))
        sGk_lay = cute.make_layout((1, K, VGS), stride=(K, 1, K))

        # ---- TMA atoms (raw-ptr + dynamic-T)
        def mk_tma_a(op, m, slay, tiler, mma):
            return cute.nvgpu.make_tiled_tma_atom_A(
                op, m, cute.slice_(slay, (None, None, None, 0)),
                tiler, mma, cute.make_layout((1, 1, 1, 1)).shape,
            )

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            (1, 1, 1), tiled_mma1.thr_id)
        atom_k, tKtile = mk_tma_a(a_op, mK3, sK_lay, tiler1, tiled_mma1)
        atom_wt, tWt = mk_tma_a(a_op, mWt3, sWt_lay, tiler2, tiled_mma2)
        atom_dv, tDV = mk_tma_a(a_op, mDV3, sDV_lay, tiler1, tiled_mma1)
        atom_qt, tQt = mk_tma_a(a_op, mQt3, sQt_lay, tiler2, tiled_mma2)
        atom_do, tDO = mk_tma_a(a_op, mDO3, sDO_lay, tiler1, tiled_mma1)
        if cutlass.const_expr(self.use_g):
            atom_g, tG = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mG2,
                cute.make_layout((BT, 4), stride=(4, 1)), (BT, 4),
            )
        else:
            atom_g, tG = atom_dv, tDV
        if cutlass.const_expr(self.use_gk):
            atom_gk, tGk = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mGk2,
                cute.make_layout((1, K), stride=(K, 1)), (1, K),
            )
        else:
            atom_gk, tGk = atom_dv, tDV

        self.fk_bytes = (
            cute.size_in_bytes(dtype, cute.slice_(sK_lay, (None, None, None, 0)))
            + cute.size_in_bytes(dtype,
                                 cute.slice_(sWt_lay, (None, None, None, 0)))
        )
        self.vg_bytes = (BT * V * dtype.width // 8
                         + (BT * 4 * 4 if self.use_g else 0)
                         + (K * 4 if self.use_gk else 0))
        self.bk_bytes = (
            cute.size_in_bytes(dtype, cute.slice_(sQt_lay, (None, None, None, 0)))
            + cute.size_in_bytes(dtype,
                                 cute.slice_(sDO_lay, (None, None, None, 0)))
        )

        # ---- merge-role machinery (SM80 hi/lo bf16 fold, as in the parent)
        sM_layout = cute.make_layout((K, K), stride=(K + 8, 1))
        atom_g2s_f32 = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32, num_bits_per_copy=128,
        )
        tiled_copy_M = cute.make_tiled_copy_tv(
            atom_g2s_f32, cute.make_layout(128), cute.make_layout(4))
        tiled_mma_sm80 = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
            (4, 1, 1), permutation_mnk=(64, 16, 16),
        )

        sy_size = cute.cosize(syt_lay)
        sX_size = cute.cosize(sX_lay)
        sK_size = cute.cosize(sK_lay)
        sWt_size = cute.cosize(sWt_lay)
        n_lam = BT * VGS * (2 if self.need_src_tab else 1)

        @cute.struct
        class SharedStorage:
            # mbarriers: full+empty pairs per stage
            fk_mbar: cute.struct.MemRange[cutlass.Int64, 2 * FRONT_STAGES]
            vg_mbar: cute.struct.MemRange[cutlass.Int64, 2 * VGS]
            bk_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            g1_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            g1b_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            g2_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            pX_mbar: cute.struct.MemRange[cutlass.Int64, 24]
            py_mbar: cute.struct.MemRange[cutlass.Int64, 12]
            tmem_holding_buf: cutlass.Int32
            # ---- big buffers: 1024-aligned and contiguous (the merge role
            # overlays its fp32 staging across them)
            sX: cute.struct.Align[cute.struct.MemRange[dtype, sX_size], 1024]
            sy: cute.struct.Align[cute.struct.MemRange[dtype, sy_size], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[dtype, sK_size], 1024]
            sWt: cute.struct.Align[cute.struct.MemRange[dtype, sWt_size], 1024]
            sDV: cute.struct.Align[
                cute.struct.MemRange[dtype, BT * V * VGS], 1024]
            sG: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, BT * 4 * VGS if self.use_g else 8], 128]
            sGk: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, K * VGS if self.use_gk else 8], 128]
            sItem: cute.struct.MemRange[cutlass.Int32, 8]
            # per-chunk gate tables: [BT, VGS] lambda, then [BT, VGS] src
            sLam: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, n_lam], 128]

        self.shared_storage = SharedStorage

        self.ws_bwd_kernel(
            atom_k, tKtile, atom_wt, tWt, atom_dv, tDV,
            atom_qt, tQt, atom_do, tDO, atom_g, tG, atom_gk, tGk,
            tiled_mma1, tiled_mma2, tiled_mma1f, tiled_mma2f,
            sK_lay, sWt_lay, sDV_lay, sQt_lay, sDO_lay,
            sX_lay, sXf_lay, sKb_lay, syt_lay, sG_lay, sGk_lay,
            tiled_mma_sm80, tiled_copy_M, sM_layout,
            mPtrs, mDHT, scale, T, step, emit, pHMloc,
        ).launch(
            grid=[self.n_ctas, 1, 1],
            block=[self.num_threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def ws_bwd_kernel(
        self,
        atom_k: cute.CopyAtom, tKtile: cute.Tensor,
        atom_wt: cute.CopyAtom, tWt: cute.Tensor,
        atom_dv: cute.CopyAtom, tDV: cute.Tensor,
        atom_qt: cute.CopyAtom, tQt: cute.Tensor,
        atom_do: cute.CopyAtom, tDO: cute.Tensor,
        atom_g: cute.CopyAtom, tG: cute.Tensor,
        atom_gk: cute.CopyAtom, tGk: cute.Tensor,
        tiled_mma1: cute.TiledMma, tiled_mma2: cute.TiledMma,
        tiled_mma1f: cute.TiledMma, tiled_mma2f: cute.TiledMma,
        sK_lay: cute.ComposedLayout, sWt_lay: cute.ComposedLayout,
        sDV_lay: cute.ComposedLayout, sQt_lay: cute.ComposedLayout,
        sDO_lay: cute.ComposedLayout, sX_lay: cute.ComposedLayout,
        sXf_lay: cute.ComposedLayout, sKb_lay: cute.ComposedLayout,
        syt_lay: cute.ComposedLayout, sG_lay: cute.Layout,
        sGk_lay: cute.Layout,
        tiled_mma_sm80: cute.TiledMma, tiled_copy_M: cute.TiledCopy,
        sM_layout: cute.Layout,
        mPtrs: cute.Tensor, mDHT: cute.Tensor,
        scale: cutlass.Float32,
        T: cutlass.Int32, step: cutlass.Int32,
        emit: cutlass.Int32,
        pHMloc: cute.Pointer,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        K, NG, SPP = self.K, self.NG, self.SPP
        rank = self.rank
        par = step % 2
        stepu = step.to(cutlass.Uint32)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        nb_item = pipeline.NamedBarrier(
            barrier_id=NB_ITEM, num_threads=self.num_threads)
        nb_epilog = pipeline.NamedBarrier(
            barrier_id=NB_EPILOG, num_threads=self.num_threads)

        has_compute = self.n_citems > 0
        if cutlass.const_expr(has_compute):
            cg_thr = pipeline.CooperativeGroup(pipeline.Agent.Thread)

            def cg(count=None):
                return (cg_thr if count is None else
                        pipeline.CooperativeGroup(pipeline.Agent.Thread, count))

            def mk_pipe(cls, stages, prod, cons, mbar, tx=None):
                if cutlass.const_expr(tx is not None):
                    return cls.create(
                        num_stages=stages, producer_group=prod,
                        consumer_group=cons, tx_count=tx,
                        barrier_storage=mbar, defer_sync=True)
                return cls.create(
                    num_stages=stages, producer_group=prod,
                    consumer_group=cons, barrier_storage=mbar,
                    defer_sync=True)

            # front TMA: K tile + W^T tile, consumed by the MMA warp
            p_fk = mk_pipe(pipeline.PipelineTmaUmma, FRONT_STAGES, cg_thr,
                           cg_thr, storage.fk_mbar.data_ptr(),
                           tx=self.fk_bytes)
            # dV + gates: consumed by the pair warps (the lambda warp waits
            # on full but deliberately does not arrive on empty; the pair
            # warps release the stage after their table reads).  4 arrivals
            # per LIVE group -- 8 normally, 4 where the transition pair is
            # compiled out.
            p_vg = mk_pipe(pipeline.PipelineTmaAsync, VGS, cg_thr,
                           cg(4 * self.n_groups_live),
                           storage.vg_mbar.data_ptr(), tx=self.vg_bytes)
            # back TMA: Q^T + dO into the dead pair-0 sX half, consumed by
            # the H pair (gemm3's A operand is read by the MMA warp, but it
            # is ordered behind the same p_bk full barrier via the H pair's
            # (src*dO)^T publish).
            # NOTE the consumer count is in WARPS, not threads:
            # PipelineTmaAsync.consumer_release only arrives from
            # `is_signaling_thread` == (tidx % 32 == 0), so a 4-warp consumer
            # contributes exactly 4 arrivals.  Declaring 128 here makes the
            # producer's second acquire wait forever (this is the same reason
            # the forward's p_vg declares cg(8) for its 8 pair warps).
            p_bk = mk_pipe(pipeline.PipelineTmaAsync, 1, cg_thr, cg(4),
                           storage.bk_mbar.data_ptr(), tx=self.bk_bytes)
            # gemm1(H) retired -> the back TMA warp may overwrite sX 0/1
            p_g1b = mk_pipe(pipeline.PipelineUmmaAsync, 1, cg_thr, cg(32),
                            storage.g1b_mbar.data_ptr())
            p_g1s, p_g2s, p_ys = [], [], []
            for pp in cutlass.range_constexpr(2):
                for lst, mbar in ((p_g1s, storage.g1_mbar),
                                  (p_g2s, storage.g2_mbar)):
                    lst.append(mk_pipe(pipeline.PipelineUmmaAsync, 1, cg_thr,
                                       cg(128), mbar.data_ptr() + 2 * pp))
                # 2 stages = the two token halves of y^T per chunk
                p_ys.append(mk_pipe(pipeline.PipelineAsyncUmma, 2,
                                    cg(128), cg_thr,
                                    storage.py_mbar.data_ptr() + 4 * pp))
            # (src*dO)^T publish -> gemm3, two token halves
            p_do = mk_pipe(pipeline.PipelineAsyncUmma, 2, cg(128), cg_thr,
                           storage.py_mbar.data_ptr() + 8)
            # X-publish staircase: 4 per-warp-quarter edges per pair
            NPUB = 4
            p_pubs = []
            for pp in cutlass.range_constexpr(2):
                qs = []
                for w in cutlass.range_constexpr(NPUB):
                    qs.append(mk_pipe(
                        pipeline.PipelineAsyncUmma, 1, cg(32), cg_thr,
                        storage.pX_mbar.data_ptr() + 2 * (4 + NPUB * pp + w)))
                p_pubs.append(qs)
            # gate tables: one 32-thread writer warp, 256 pair readers
            p_lam = mk_pipe(pipeline.PipelineAsync, VGS, cg(32), cg(256),
                            storage.pX_mbar.data_ptr() + 4)
            pipeline_init_arrive(cluster_shape_mn=(1, 1, 1), is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=(1, 1, 1))

            # tmem: NG acc1 strips + NG X strips, 64 cols each
            tmem_bar = pipeline.NamedBarrier(
                barrier_id=NB_TMEM_ALLOC, num_threads=self.num_threads)
            tmem = utils.TmemAllocator(
                tmem_alloc_ptr(storage.tmem_holding_buf),
                barrier_for_retrieve=tmem_bar,
                allocator_warp_id=0,
            )
            tmem.allocate(2 * NS * NG)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

            # TMEM map: acc1^T pair p = BT cols at [BT*p, BT*p+BT);
            # X strips at [2*BT + NS*SPP*p).  Identical to the forward.
            acc1f_lay = tiled_mma1f.make_fragment_C(
                tiled_mma1f.partition_shape_C((NS * SPP, BT))).layout
            stateq_lay = tiled_mma2f.make_fragment_C(
                tiled_mma2f.partition_shape_C((K, NS * SPP))).layout
            tAcc1ps, tXps = [], []
            for pp in cutlass.range_constexpr(2):
                tAcc1ps.append(cute.make_tensor(tmem_ptr + BT * pp, acc1f_lay))
                tXps.append(cute.make_tensor(
                    tmem_ptr + 2 * BT + NS * SPP * pp, stateq_lay))

            sK_t = storage.sK.get_tensor(sK_lay.outer, swizzle=sK_lay.inner)
            sWt_t = storage.sWt.get_tensor(sWt_lay.outer, swizzle=sWt_lay.inner)
            sX_t = storage.sX.get_tensor(sX_lay.outer, swizzle=sX_lay.inner)
            # Cold epilogue-only fp32 view over the same 64-KiB allocation
            # (16 warp-private 32x32 tiles for the coalesced peer push).
            sPush_t = cute.make_tensor(
                cute.recast_ptr(storage.sX.data_ptr(), dtype=cutlass.Float32),
                cute.make_layout(16 * 32 * 32))
            sXf_t = storage.sX.get_tensor(sXf_lay.outer, swizzle=sXf_lay.inner)
            sKb_t = storage.sK.get_tensor(sKb_lay.outer, swizzle=sKb_lay.inner)
            syt_t = storage.sy.get_tensor(syt_lay.outer, swizzle=syt_lay.inner)
            sDV_t = storage.sDV.get_tensor(sDV_lay.outer, swizzle=sDV_lay.inner)
            # ---- the dead pair-0 sX overlay: Q^T on strip 0, dO on strip 1.
            # Both are 16-KiB (NS*K bf16) 1024-aligned sub-blocks, so the
            # SW128 swizzle period divides the strip-1 offset exactly.  The
            # swizzle has to ride the POINTER (as `get_tensor(..., swizzle=)`
            # does for the named buffers) or make_fragment_A rejects the
            # composed layout, and the ldmatrix composition below would not
            # match sDV's addressing.
            sXbase = storage.sX.get_tensor(cute.make_layout(1)).iterator
            sQt_t = cute.make_tensor(
                cute.recast_ptr(sXbase, sQt_lay.inner,
                                dtype=self.io_dtype),
                sQt_lay.outer)
            sDO_t = cute.make_tensor(
                cute.recast_ptr(sXbase + NS * K, sDO_lay.inner,
                                dtype=self.io_dtype),
                sDO_lay.outer)
            lam_lay = cute.make_layout((BT, VGS), stride=(1, BT))
            sLam_t = storage.sLam.get_tensor(lam_lay)
            if cutlass.const_expr(self.need_src_tab):
                sSrc_t = cute.make_tensor(
                    storage.sLam.get_tensor(cute.make_layout(1)).iterator
                    + BT * VGS, lam_lay)
            else:
                sSrc_t = sLam_t
            sI_t = storage.sItem.get_tensor(cute.make_layout(8))
            if cutlass.const_expr(self.use_g):
                sG_t = storage.sG.get_tensor(sG_lay)
            else:
                sG_t = storage.sG.get_tensor(
                    cute.make_layout((8, 1, 1), stride=(1, 0, 0)))
            if cutlass.const_expr(self.use_gk):
                sGk_t = storage.sGk.get_tensor(sGk_lay)
            else:
                sGk_t = storage.sGk.get_tensor(
                    cute.make_layout((1, 8, 1), stride=(8, 1, 0)))

            Prod = pipeline.PipelineUserType.Producer
            Cons = pipeline.PipelineUserType.Consumer
            mps = pipeline.make_pipeline_state
            st_one = mps(Prod, 1)
            st_fk_c = mps(Cons, FRONT_STAGES)
            st_fk_p = mps(Prod, FRONT_STAGES)
            st_vg_c = mps(Cons, VGS)
            st_vg_p = mps(Prod, VGS)
            st_vg_l = mps(Cons, VGS)
            st_bk_p = mps(Prod, 1)
            st_bk_c = mps(Cons, 1)
            st_g1b_c = mps(Cons, 1)
            st_g1_c = mps(Cons, 1)
            st_g2_c = mps(Cons, 1)
            st_pub_c = mps(Cons, 1)
            st_y_p = mps(Prod, 2)
            st_y_c = mps(Cons, 2)
            st_y_cb = mps(Cons, 2)
            st_do_p = mps(Prod, 2)
            st_do_c = mps(Cons, 2)

        # ==================== persistent item loop =====================
        if cutlass.const_expr(has_compute):
            for slot in cutlass.range_constexpr(self.NSLOT):
                nb_item.arrive_and_wait()
                if tidx == 0:
                    wid = cutlass.Int32(bidx) + slot * self.n_ctas
                    sI_t[0] = wid                       # hv
                    sI_t[4] = (1 if wid < self.n_citems else 0)
                    # slot 7 = this launch's emit mask; see the forward twin in
                    # `fused_ws.py`.  Read only by the cold push epilogue, so it
                    # never lives in a register across the reverse scan.
                    sI_t[7] = emit
                nb_item.arrive_and_wait()
                if sI_t[4] != 0:
                    hv = sI_t[0]
                    nc = (T + (BT - 1)) // BT           # NT chunks
                    if warp_idx == self.tma_warp:
                        st_fk_p, st_vg_p = self._front_tma(
                            atom_k, tKtile, atom_wt, tWt, atom_dv, tDV,
                            atom_g, tG, atom_gk, tGk,
                            tiled_mma1, tiled_mma2,
                            sK_t, sWt_t, sDV_t, sG_t, sGk_t,
                            T, nc, hv, p_fk, p_vg, st_fk_p, st_vg_p,
                        )
                        # ack throttle before anyone pushes
                        ackv = cutlass.max(step - 2, 0).to(cutlass.Uint32)
                        ak_loc = mPtrs[2, rank]
                        if tidx == self.tma_warp * 32:
                            for cc in cutlass.range_constexpr(
                                    self.consumer_lo, rank):
                                for b in cutlass.range_constexpr(self.NB):
                                    self._spin_ge_sys(
                                        ak_loc, (cc * self.HV + hv) * self.NB + b,
                                        ackv)
                        nb_epilog.arrive_and_wait()
                    elif warp_idx == self.mma_warp:
                        (st_fk_c, st_pub_c, st_y_c, st_y_cb,
                         st_do_c) = self._mma_item(
                            nc, tiled_mma1f, tiled_mma2f,
                            sKb_t, sWt_t, sQt_t, sXf_t, syt_t,
                            tAcc1ps, tXps,
                            p_fk, p_g1s, p_g1b, p_g2s, p_pubs, p_ys, p_do,
                            p_lam, st_fk_c, st_pub_c, st_y_c, st_y_cb,
                            st_do_c, st_one,
                        )
                        nb_epilog.arrive_and_wait()
                    elif warp_idx == self.bk_warp:
                        st_bk_p, st_g1b_c = self._back_tma(
                            atom_qt, tQt, atom_do, tDO,
                            tiled_mma1, tiled_mma2, sQt_t, sDO_t,
                            T, nc, hv, p_bk, p_g1b, st_bk_p, st_g1b_c,
                        )
                        nb_epilog.arrive_and_wait()
                    elif warp_idx == self.lam_warp:
                        st_vg_l = self._gate_item(
                            tidx - self.lam_warp * 32, hv, T, nc, scale,
                            sG_t, sLam_t, sSrc_t, p_vg, p_lam, st_vg_l,
                        )
                        nb_epilog.arrive_and_wait()
                    else:
                        # Groups at or above `n_groups_live` are the dead
                        # transition half: they keep their single NB_EPILOG
                        # arrival and do nothing else.
                        for pp in cutlass.range_constexpr(
                                self.n_groups_live, 2):
                            if (warp_idx >= 4 * pp) and (warp_idx < 4 * (pp + 1)):
                                nb_epilog.arrive_and_wait()
                        for pp in cutlass.range_constexpr(self.n_groups_live):
                            if (warp_idx >= 4 * pp) and (warp_idx < 4 * (pp + 1)):
                                (st_vg_c, st_g1_c, st_g2_c, st_y_p,
                                 st_do_p, st_bk_c) = self._pair_item(
                                    pp, tidx % 128, hv, T, nc, par, scale,
                                    sI_t, mPtrs, tAcc1ps[pp], tXps[pp],
                                    sX_t, sPush_t, syt_t, sDV_t, sDO_t,
                                    sG_t, sGk_t, sLam_t, sSrc_t,
                                    p_vg, p_g1s[pp], p_g2s[pp], p_pubs[pp],
                                    p_ys[pp], p_do, p_lam, p_bk,
                                    st_vg_c, st_g1_c, st_g2_c, st_y_p,
                                    st_do_p, st_bk_c, st_one,
                                )
                    nb_epilog.arrive_and_wait()
                    cute.arch.fence_acq_rel_sys()
                    if tidx == 0:
                        self._flag_bwd_consumers(mPtrs, hv, stepu)

        if cutlass.const_expr(self.n_mitems > 0):
            wi = cutlass.Int32(bidx)
            while wi < self.n_mitems:
                nb_item.arrive_and_wait()
                mb = wi % self.NB
                mhv = wi // self.NB
                if warp_idx <= 3:
                    self._bwd_merge_role(
                        tidx, mb, mhv, par, stepu, mPtrs, mDHT, pHMloc,
                        storage, tiled_mma_sm80, tiled_copy_M, sM_layout,
                    )
                nb_item.arrive_and_wait()
                wi += self.n_ctas

        if cutlass.const_expr(has_compute):
            tmem.relinquish_alloc_permit()
            pipeline.NamedBarrier(
                barrier_id=NB_TMEM_FREE, num_threads=self.num_threads
            ).arrive_and_wait()
            tmem.free(tmem_ptr)

    # --------------------------------------------------------- reverse index
    @cute.jit
    def _rev(self, tq, nc):
        """Reverse-scan chunk index: the scan runs NT-1 .. 0."""
        return nc - 1 - tq

    # ------------------------------------------------------------- gemm util
    @cute.jit
    def _exec_gemm(self, tiled_mma, tAcc, tCrA, tCrB, a_stage, b_stage,
                   acc_init: cutlass.Constexpr = False,
                   kp_lo: cutlass.Constexpr = 0,
                   kp_hi: cutlass.Constexpr = -1):
        n_kp = cute.size(tCrB, mode=[2])
        hi = n_kp if kp_hi < 0 else kp_hi
        for kpi in cutlass.range(hi - kp_lo, unroll_full=True):
            kp = kp_lo + kpi
            tiled_mma.set(tcgen05.Field.ACCUMULATE,
                          cutlass.Boolean(acc_init or kp != 0))
            cute.gemm(tiled_mma, tAcc,
                      tCrA[None, None, kp, a_stage],
                      tCrB[None, None, kp, b_stage], tAcc)
        return tiled_mma

    # ----------------------------------------------------------- front TMA
    @cute.jit
    def _front_tma(
        self, atom_k, tKtile, atom_wt, tWt, atom_dv, tDV, atom_g, tG,
        atom_gk, tGk, tiled_mma1, tiled_mma2,
        sK_t, sWt_t, sDV_t, sG_t, sGk_t,
        T, nc, hv, p_fk, p_vg, st_fk, st_vg,
    ):
        hk = hv // (self.HV // self.H)
        thr_mma1 = tiled_mma1.get_slice(0)
        thr_mma2 = tiled_mma2.get_slice(0)
        K, V = self.K, self.V

        def tma_ld(atom, gt, off, tiler, thr, s_t, stage, bar):
            g = cute.local_tile(cute.domain_offset(off, gt), tiler,
                                (None, None, None), proj=(1, None, 1))
            tAs, tAg = cpasync.tma_partition(
                atom, 0, cute.make_layout(1),
                cute.group_modes(s_t, 0, 3),
                cute.group_modes(thr.partition_A(g), 0, 3),
            )
            cute.copy(atom, tAg[(None, 0, 0)], tAs[(None, stage)],
                      tma_bar_ptr=bar)

        def gate_ld(atom, gt, off, tiler, s_t, stage, bar):
            g = cute.local_tile(cute.domain_offset(off, gt), tiler, (0, 0))
            tGs, tGg = cpasync.tma_partition(
                atom, 0, cute.make_layout(1),
                cute.group_modes(s_t[None, None, stage], 0, 2),
                cute.group_modes(g, 0, 2),
            )
            cute.copy(atom, tGg, tGs, tma_bar_ptr=bar)

        for tq in cutlass.range(nc, unroll=1):
            cq = self._rev(tq, nc)
            t0 = cq * BT
            t_ofs = cutlass.max(cutlass.min(t0, T - BT), 0)

            p_fk.producer_acquire(st_fk)
            bar_fk = p_fk.producer_get_barrier(st_fk)
            tma_ld(atom_k, tKtile[None, None, hk], (t_ofs, 0), (BT, NS, K),
                   thr_mma1, sK_t, st_fk.index, bar_fk)
            tma_ld(atom_wt, tWt[None, None, hv], (0, t_ofs), (K, NS, BT),
                   thr_mma2, sWt_t, st_fk.index, bar_fk)

            p_vg.producer_acquire(st_vg)
            bar_vg = p_vg.producer_get_barrier(st_vg)
            tma_ld(atom_dv, tDV[None, None, hv], (t_ofs, 0), (BT, NS, V),
                   thr_mma1, sDV_t, st_vg.index, bar_vg)
            if cutlass.const_expr(self.use_g):
                gate_ld(atom_g, tG, (t_ofs, (hv // 4) * 4), (BT, 4),
                        sG_t, st_vg.index, bar_vg)
            if cutlass.const_expr(self.use_gk):
                last_idx = t0 + cutlass.min(T - t0, BT) - 1
                gate_ld(atom_gk, tGk, (last_idx * self.HV + hv, 0),
                        (1, self.K), sGk_t, st_vg.index, bar_vg)

            st_fk.advance()
            st_vg.advance()
        return st_fk, st_vg

    # ------------------------------------------------------------ back TMA
    @cute.jit
    def _back_tma(
        self, atom_qt, tQt, atom_do, tDO, tiled_mma1, tiled_mma2,
        sQt_t, sDO_t, T, nc, hv, p_bk, p_g1b, st_bk, st_g1b,
    ):
        """Q^T and dO land in the pair-0 sX half, which is dead only after
        gemm1(H) of the SAME chunk has consumed the published X_H^T."""
        hk = hv // (self.HV // self.H)
        thr_mma1 = tiled_mma1.get_slice(0)
        thr_mma2 = tiled_mma2.get_slice(0)
        K, V = self.K, self.V

        def tma_ld(atom, gt, off, tiler, thr, s_t, bar):
            g = cute.local_tile(cute.domain_offset(off, gt), tiler,
                                (None, None, None), proj=(1, None, 1))
            tAs, tAg = cpasync.tma_partition(
                atom, 0, cute.make_layout(1),
                cute.group_modes(s_t, 0, 3),
                cute.group_modes(thr.partition_A(g), 0, 3),
            )
            cute.copy(atom, tAg[(None, 0, 0)], tAs[(None, 0)],
                      tma_bar_ptr=bar)

        for tq in cutlass.range(nc, unroll=1):
            cq = self._rev(tq, nc)
            t0 = cq * BT
            t_ofs = cutlass.max(cutlass.min(t0, T - BT), 0)
            # gemm1(H) retired: the overlay bytes are free.  This edge also
            # transitively orders the H pair's previous-chunk dO reads (its
            # own chain runs pass1a -> pub -> gemm1, and pass1a waits gemm3).
            p_g1b.consumer_wait(st_g1b)
            st_g1b.advance()
            p_bk.producer_acquire(st_bk)
            bar_bk = p_bk.producer_get_barrier(st_bk)
            tma_ld(atom_qt, tQt[None, None, hk], (0, t_ofs), (K, NS, BT),
                   thr_mma2, sQt_t, bar_bk)
            tma_ld(atom_do, tDO[None, None, hv], (t_ofs, 0), (BT, NS, V),
                   thr_mma1, sDO_t, bar_bk)
            st_bk.advance()
        return st_bk, st_g1b

    # ---------------------------------------------------------- gate tables
    @cute.jit
    def _gate_item(
        self, ltid, hv, T, nc, scale, sG_t, sLam_t, sSrc_t, p_vg, p_lam, st_vg,
    ):
        """One warp publishes both per-token gate tables for the chunk.

        ``lam`` is the row decay 2^(g_last-g_i) and doubles as the tail
        validity mask; ``src`` is gemm3's scale*2^(g_i).  The warp waits on
        vg_full but does not arrive on vg_empty -- the pair warps release the
        stage after consuming the tables.
        """
        if cutlass.const_expr(self.need_lam):
            for tq in cutlass.range(nc, unroll=1):
                p_vg.consumer_wait(st_vg)
                cq = self._rev(tq, nc)
                t0 = cq * BT
                n_valid = T - t0
                shift = t0 - cutlass.max(cutlass.min(t0, T - BT), 0)
                g_last = cutlass.Float32(0.0)
                if cutlass.const_expr(self.use_g):
                    g_last = sG_t[cutlass.min(n_valid, BT) - 1 + shift,
                                  hv % 4, st_vg.index]
                for rep in cutlass.range_constexpr(BT // 32):
                    row = ltid + 32 * rep
                    idxl = row - shift
                    lam = cutlass.Float32(0.0)
                    src = cutlass.Float32(0.0)
                    if (idxl >= 0) & (idxl < n_valid):
                        lam = cutlass.Float32(1.0)
                        src = scale
                        if cutlass.const_expr(self.use_g):
                            gi = sG_t[row, hv % 4, st_vg.index]
                            lam = cute.math.exp2(g_last - gi, fastmath=True)
                            src = scale * cute.math.exp2(gi, fastmath=True)
                    sLam_t[row, st_vg.index] = lam
                    if cutlass.const_expr(self.need_src_tab):
                        sSrc_t[row, st_vg.index] = src
                p_lam.producer_commit(st_vg)
                st_vg.advance()
        return st_vg

    # -------------------------------------------------------------- MMA warp
    @cute.jit
    def _mma_item(
        self, nc, tiled_mma1f, tiled_mma2f,
        sKb_t, sWt_t, sQt_t, sXf_t, syt_t, tAcc1ps, tXps,
        p_fk, p_g1s, p_g1b, p_g2s, p_pubs, p_ys, p_do, p_lam,
        st_fk, st_pub, st_yc, st_ycb, st_doc, st_one,
    ):
        tCrX = tiled_mma1f.make_fragment_A(sXf_t)     # A = X^T   (stage=pair)
        tCrK = tiled_mma1f.make_fragment_B(sKb_t)     # B = K     (stage=fk)
        tCrWt = tiled_mma2f.make_fragment_A(sWt_t)    # A = W^T   (stage=fk)
        tCrQt = tiled_mma2f.make_fragment_A(sQt_t)    # A = Q^T   (1 stage)
        tCry = tiled_mma2f.make_fragment_B(syt_t)     # B = -y^T / (src*dO)^T
        mma1 = tiled_mma1f
        mma2 = tiled_mma2f
        NKH = cute.size(tCry, mode=[2]) // 2          # k-phases per token half

        # arm every LIVE pair's loop-top gemm2 wait for the first chunk (no
        # MMAs are in flight yet); balanced by the epilogue waits.  A dead
        # transition pair has no warps waiting on it.
        for pp in cutlass.range_constexpr(self.n_groups_live):
            p_g2s[pp].producer_commit(st_one)
        for _t in cutlass.range(nc, unroll=1):
            p_fk.consumer_wait(st_fk)
            # ---- gemm1(H): acc1_H^T = X_H^T @ K^T, published-quarter staircase
            for w in cutlass.range_constexpr(4):
                p_pubs[0][w].consumer_wait(st_pub)
                mma1 = self._exec_gemm(mma1, tAcc1ps[0], tCrX, tCrK,
                                       0, st_fk.index,
                                       kp_lo=2 * w, kp_hi=2 * w + 2)
            p_g1s[0].producer_commit(st_one)
            # sX strips 0/1 are dead from here: release them to the back TMA
            p_g1b.producer_commit(st_one)
            # ---- gemm1(P) and gemm2(P).  Both are dropped, together with the
            # edges that order them, on a rank whose transition half no other
            # rank can read: pair 1 has no warps there to publish X or wait on
            # the commits.  gemm2(P) is FIRST when it does run -- the P pair
            # carries neither dV nor gemm3, so its y^T lands earlier and its
            # recurrence gate can be released before the H chain runs, letting
            # pair 1 start the next chunk's drain/publish under
            # gemm2(H)+gemm3.  Separate TMEM accumulators, so this is
            # bit-exact either way.
            if cutlass.const_expr(self.n_groups_live > 1):
                for w in cutlass.range_constexpr(4):
                    p_pubs[1][w].consumer_wait(st_pub)
                    mma1 = self._exec_gemm(mma1, tAcc1ps[1], tCrX, tCrK,
                                           1, st_fk.index,
                                           kp_lo=2 * w, kp_hi=2 * w + 2)
                if cutlass.const_expr(self.need_lam):
                    p_lam.consumer_wait(st_fk)
                p_g1s[1].producer_commit(st_one)

                p_ys[1].consumer_wait(st_ycb)
                mma2 = self._exec_gemm(mma2, tXps[1], tCrWt, tCry,
                                       st_fk.index, 1, acc_init=True,
                                       kp_lo=0, kp_hi=NKH)
                st_ycb.advance()
                p_ys[1].consumer_wait(st_ycb)
                mma2 = self._exec_gemm(mma2, tXps[1], tCrWt, tCry,
                                       st_fk.index, 1, acc_init=True,
                                       kp_lo=NKH)
                p_g2s[1].producer_commit(st_one)

            # ---- gemm2(H): X_H += W^T @ (-y_H), two token halves
            p_ys[0].consumer_wait(st_yc)
            mma2 = self._exec_gemm(mma2, tXps[0], tCrWt, tCry,
                                   st_fk.index, 0, acc_init=True,
                                   kp_lo=0, kp_hi=NKH)
            st_yc.advance()
            p_ys[0].consumer_wait(st_yc)
            mma2 = self._exec_gemm(mma2, tXps[0], tCrWt, tCry,
                                   st_fk.index, 0, acc_init=True, kp_lo=NKH)

            # ---- gemm3: X_H += Q^T @ (src*dO)^T.  Pair 0's "gemm2 done"
            # edge is deferred to here so the next chunk's pass 1a (which
            # overwrites sX strips 0/1, i.e. Q^T) cannot run before gemm3
            # retires.
            p_do.consumer_wait(st_doc)
            mma2 = self._exec_gemm(mma2, tXps[0], tCrQt, tCry,
                                   0, 2, acc_init=True, kp_lo=0, kp_hi=NKH)
            st_doc.advance()
            p_do.consumer_wait(st_doc)
            mma2 = self._exec_gemm(mma2, tXps[0], tCrQt, tCry,
                                   0, 2, acc_init=True, kp_lo=NKH)
            p_g2s[0].producer_commit(st_one)

            p_fk.consumer_release(st_fk)
            st_fk.advance()
            st_pub.advance()
            st_yc.advance()
            st_ycb.advance()
            st_doc.advance()
        return st_fk, st_pub, st_yc, st_ycb, st_doc

    # ------------------------------------------------------------ pair group
    @cute.jit
    def _pair_item(
        self, pair: cutlass.Constexpr, ltid, hv, T, nc, par, scale,
        sI_t, mPtrs, tAcc1, tX,
        sX_t, sPush_t, syt_t, sDV_t, sDO_t, sG_t, sGk_t, sLam_t, sSrc_t,
        p_vg, p_g1, p_g2, p_pub, p_y, p_do, p_lam, p_bk,
        st_vg, st_g1c, st_g2c, st_yp, st_dop, st_bkc, st_one,
    ):
        """One warpgroup owns both strips of an MMA pair.

        pair 0 = H_tilde (carries dV and gemm3); pair 1 = P (neither).
        """
        K, V, SPP = self.K, self.V, self.SPP
        dtype = self.io_dtype
        strip0 = 2 * pair
        is_h = pair == 0
        BTH = BT // 2

        # The pair's two X strips are adjacent TMEM columns; thread ltid owns
        # one K row and drains/stores all 128 values with one x128 operation.
        x_2d = tX[((None, None), 0, 0)]
        atom_ld_x = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(2 * NS), tcgen05.Pack.NONE),
            cutlass.Float32)
        atom_st_x = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(2 * NS),
                               tcgen05.Unpack.NONE),
            cutlass.Float32)
        t2r_x = tcgen05.make_tmem_copy(atom_ld_x, x_2d)
        r2t_x = tcgen05.make_tmem_copy(atom_st_x, x_2d)
        thr_x = t2r_x.get_slice(ltid)
        tXt = thr_x.partition_S(x_2d)
        tXst = r2t_x.get_slice(ltid).partition_D(x_2d)
        cX = thr_x.partition_D(cute.make_identity_tensor((K, 2 * NS)))
        rX = cute.make_rmem_tensor(
            cute.slice_(cX.shape, (None, 0, 0)), cutlass.Float32)

        # Full-pair acc1^T, split into two 32-token halves.
        acc1_2d = tAcc1[((None, None), 0, 0)]
        pan_acc = cute.make_layout(
            (NS * SPP, BTH, 2), stride=(1, NS * SPP, NS * SPP * BTH))
        acc1_3h = cute.make_tensor(
            acc1_2d.iterator, cute.composition(acc1_2d.layout, pan_acc))
        acc1h = []
        for hh in cutlass.range_constexpr(2):
            acc1h.append(acc1_3h[(None, None, hh)])
        atom_ld_a1 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(4), tcgen05.Pack.NONE),
            cutlass.Float32)
        t2r_a1 = tcgen05.make_tmem_copy(atom_ld_a1, acc1h[0])
        thr_a1 = t2r_a1.get_slice(ltid)
        tA1tH = []
        for hh in cutlass.range_constexpr(2):
            tA1tH.append(thr_a1.partition_S(acc1h[hh]))
        cA1 = thr_a1.partition_D(cute.make_identity_tensor((NS * SPP, BTH)))

        # -y^T publication tile (stage = pair) and, for pair 0, the gemm3
        # (src*dO)^T tile in stage 2.
        pan_yt = cute.make_layout(
            (NS * SPP, (16, BT // 32), 2),
            stride=(1, (NS * SPP, 16 * NS * SPP), (BT // 2) * NS * SPP))
        st_atom_a = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR, dtype, cutlass.Float32, t2r_a1)
        r2s_a = cute.make_tiled_copy_D(st_atom_a, t2r_a1)

        sy1 = syt_t[(None, None, None, pair)]
        sy3 = cute.make_tensor(sy1.iterator,
                               cute.composition(sy1.layout, pan_yt))
        tRS_syH = []
        for hh in cutlass.range_constexpr(2):
            tRS_syH.append(
                r2s_a.get_slice(ltid).partition_D(sy3[(None, None, hh)]))
        # gemm3's B operand rides stage 2 of the same buffer (H pair only)
        sdo1 = syt_t[(None, None, None, 2)]
        sdo3 = cute.make_tensor(sdo1.iterator,
                                cute.composition(sdo1.layout, pan_yt))
        tRS_doH = []
        for hh in cutlass.range_constexpr(2):
            tRS_doH.append(
                r2s_a.get_slice(ltid).partition_D(sdo3[(None, None, hh)]))

        # dV^T / dO^T use the same full-pair fragment mapping as acc1^T.
        ld_atom_v = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            dtype)
        s2r_v = cute.make_tiled_copy_D(ld_atom_v, t2r_a1)
        thr_v = s2r_v.get_slice(ltid)
        pan_vt = cute.make_layout(
            (V, BTH, 2, VGS), stride=(BT, 1, BTH, BT * K))
        sDVt4 = cute.make_tensor(
            sDV_t.iterator, cute.composition(sDV_t.layout, pan_vt))
        pan_ot = cute.make_layout(
            (V, BTH, 2, 1), stride=(BT, 1, BTH, BT * K))
        sDOt4 = cute.make_tensor(
            sDO_t.iterator, cute.composition(sDO_t.layout, pan_ot))

        rA1a = cute.make_rmem_tensor(cA1.shape, cutlass.Float32)
        rA1b = cute.make_rmem_tensor(cA1.shape, cutlass.Float32)
        tVsm0 = thr_v.partition_S(sDVt4[(None, None, 0, 0)])
        rVsA = cute.make_rmem_tensor(tVsm0.shape, dtype)
        rVsB = cute.make_rmem_tensor(tVsm0.shape, dtype)
        if cutlass.const_expr(self.need_lam):
            rLam = cute.make_rmem_tensor(32, cutlass.Float32)
        if cutlass.const_expr(is_h):
            # pass 3 runs one token half at a time, so a single dO fragment
            # and a half-width src table suffice (see the pass-3 comment)
            rOsA = cute.make_rmem_tensor(tVsm0.shape, dtype)
            rSrc = cute.make_rmem_tensor(16, cutlass.Float32)
        ryc = cute.make_rmem_tensor(tRS_syH[0].shape, dtype)
        sub_a1 = cute.size(rA1a)
        lam_half = sub_a1 // 2

        # X publish destinations: two strips x two column halves.
        atom_pub = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)
        pubX = cute.make_tiled_copy_tv(
            atom_pub, cute.make_layout((K, 1)), cute.make_layout((1, NS // 2)))
        pan_xh = cute.make_layout(
            ((16, K // 16), NS // 2, 2), stride=((NS, 16 * NS), 1, NS // 2))
        tPubX = []
        for ss in cutlass.range_constexpr(2):
            sX1 = sX_t[(None, None, None, strip0 + ss)]
            sX3h = cute.make_tensor(sX1.iterator,
                                    cute.composition(sX1.layout, pan_xh))
            halves = []
            for hh in cutlass.range_constexpr(2):
                halves.append(pubX.get_slice(ltid).partition_D(
                    sX3h[(None, None, hh)]))
            tPubX.append(halves)
        rXcv = cute.make_fragment_like(tPubX[0][0], dtype)
        atom_f32x4 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32,
            num_bits_per_copy=128)

        # X(0): H_tilde = 0; P = I (the two diagonal blocks of the P pair).
        row0 = cX[0][0]
        for i in cutlass.range_constexpr(2 * NS):
            val = cutlass.Float32(0.0)
            if cutlass.const_expr(not is_h):
                if row0 == cX[i][1]:
                    val = cutlass.Float32(1.0)
            rX[i] = val
        cute.copy(r2t_x, rX, tXst[(None, 0, 0)])
        cute.arch.fence_view_async_tmem_store()

        for tq in cutlass.range(nc, unroll=1):
            # gemm2/gemm3 of the previous reverse chunk have retired: X is
            # final and its publish buffer is free.
            p_g2.consumer_wait(st_g2c)
            st_g2c.advance()

            # ---- pass 1a: one x128 TMEM drain, then publish both strips.
            cute.copy(t2r_x, tXt[(None, 0, 0)], rX)
            cute.arch.fence_view_async_tmem_load()
            for ss in cutlass.range_constexpr(2):
                for hh in cutlass.range_constexpr(2):
                    for i in cutlass.range_constexpr(NS // 2):
                        rXcv[i] = rX[ss * NS + hh * (NS // 2) + i].to(dtype)
                    cute.copy(pubX, rXcv, tPubX[ss][hh])
            cute.arch.fence_proxy("async.shared", space="cta")
            wig = ltid // 32
            for w in cutlass.range_constexpr(4):
                if wig == w:
                    p_pub[w].producer_commit(st_one)

            # ---- dV/gate stage
            p_vg.consumer_wait(st_vg)
            cq = self._rev(tq, nc)
            t0 = cq * BT
            n_valid = T - t0
            shift = t0 - cutlass.max(cutlass.min(t0, T - BT), 0)
            g_last = cutlass.Float32(0.0)
            if cutlass.const_expr(self.use_g):
                g_last = sG_t[cutlass.min(n_valid, BT) - 1 + shift,
                              hv % 4, st_vg.index]
            gamma = cutlass.Float32(1.0)
            if cutlass.const_expr(self.use_g):
                gamma = cute.math.exp2(g_last, fastmath=True)

            # ---- pass 1b: gamma decay while X is resident in rX.
            if cutlass.const_expr(self.need_decay):
                if cutlass.const_expr(self.use_gk):
                    gscale = cute.math.exp2(sGk_t[0, row0, st_vg.index],
                                            fastmath=True)
                else:
                    gscale = gamma
                # Vector store, NOT a scalar `for i: rX[i] = gscale * rX[i]`.
                # The scalar loop compiled to 128 static FMUL executing
                # 8,388,608 times per pair; this emits the packed
                # `FMUL2 Rn, Rn.F32x2.HI_LO, Rgamma.F32` the CuTe C++
                # reference uses -- 64 static, 4,194,304 executed. Bit-exact
                # (FMUL2 is two independent IEEE fp32 multiplies).
                #
                # Worth -8,388,608 instructions (-9% of the kernel), but only
                # -3,969 elapsed cycles (-0.6%) and NO measurable wall-clock
                # change: at 0.86 waves and 28% issue-slot occupancy this
                # kernel is bound by its per-chunk dependency chain, not by
                # instruction throughput.  Kept because it is free and
                # strictly fewer instructions; do not expect it to show up in
                # a benchmark.  See OPT_LOG.md "SASS root cause".
                rX.store(rX.load() * gscale)
                cute.copy(r2t_x, rX, tXst[(None, 0, 0)])

            # ---- pass 2: y^T = lambda (*) acc1^T + dV^T, published negated.
            p_g1.consumer_wait(st_g1c)
            if cutlass.const_expr(self.need_lam and pair == 0):
                p_lam.consumer_wait(st_vg)

            cute.copy(t2r_a1, tA1tH[0], rA1a)
            if cutlass.const_expr(is_h):
                cute.copy(s2r_v,
                          thr_v.partition_S(sDVt4[(None, None, 0, st_vg.index)]),
                          rVsA)
            if cutlass.const_expr(self.need_lam):
                for j in cutlass.range_constexpr(sub_a1 // 4):
                    rLam[2 * j] = sLam_t[cA1[4 * j][1], st_vg.index]
                    rLam[2 * j + 1] = sLam_t[cA1[4 * j + 1][1], st_vg.index]
            cute.arch.fence_view_async_tmem_load()

            cute.copy(t2r_a1, tA1tH[1], rA1b)
            if cutlass.const_expr(is_h):
                cute.copy(s2r_v,
                          thr_v.partition_S(sDVt4[(None, None, 1, st_vg.index)]),
                          rVsB)
            if cutlass.const_expr(self.need_lam):
                for j in cutlass.range_constexpr(sub_a1 // 4):
                    rLam[lam_half + 2 * j] = sLam_t[
                        BTH + cA1[4 * j][1], st_vg.index]
                    rLam[lam_half + 2 * j + 1] = sLam_t[
                        BTH + cA1[4 * j + 1][1], st_vg.index]

            for j in cutlass.range_constexpr(sub_a1 // 4):
                for q in cutlass.range_constexpr(4):
                    i = 4 * j + q
                    yv = rA1a[i]
                    if cutlass.const_expr(self.need_lam):
                        yv = yv * rLam[2 * j + (q & 1)]
                    if cutlass.const_expr(is_h):
                        # dV is NOT lambda-scaled; invalid rows contribute
                        # exactly zero via the lambda mask.
                        dv = rVsA[i].to(cutlass.Float32)
                        if cutlass.const_expr(self.need_mask):
                            if rLam[2 * j + (q & 1)] == cutlass.Float32(0.0):
                                dv = cutlass.Float32(0.0)
                        yv = yv + dv
                    ryc[i] = (cutlass.Float32(0.0) - yv).to(dtype)
            cute.copy(r2s_a, ryc, tRS_syH[0])
            if cutlass.const_expr(self.need_decay):
                cute.arch.fence_view_async_tmem_store()
            cute.arch.fence_proxy("async.shared", space="cta")
            p_y.producer_commit(st_yp)
            st_yp.advance()

            cute.arch.fence_view_async_tmem_load()
            for j in cutlass.range_constexpr(sub_a1 // 4):
                for q in cutlass.range_constexpr(4):
                    i = 4 * j + q
                    yv = rA1b[i]
                    if cutlass.const_expr(self.need_lam):
                        yv = yv * rLam[lam_half + 2 * j + (q & 1)]
                    if cutlass.const_expr(is_h):
                        dv = rVsB[i].to(cutlass.Float32)
                        if cutlass.const_expr(self.need_mask):
                            if rLam[lam_half + 2 * j
                                    + (q & 1)] == cutlass.Float32(0.0):
                                dv = cutlass.Float32(0.0)
                        yv = yv + dv
                    ryc[i] = (cutlass.Float32(0.0) - yv).to(dtype)
            cute.copy(r2s_a, ryc, tRS_syH[1])
            cute.arch.fence_proxy("async.shared", space="cta")
            p_y.producer_commit(st_yp)
            st_yp.advance()

            # ---- pass 3 (H pair only): (src (*) dO)^T for gemm3.
            #
            # One token half at a time.  Unlike pass 2 -- where staging both
            # halves is deliberate, to hide TMEM load latency -- pass 3 reads
            # only shared memory and needs no cross-half data, so a single dO
            # fragment and a half-width src table suffice.
            #
            # This was written expecting it to cut the kernel's register
            # spill (ncu: 794,624 spill instructions vs 2,048 for the CuTe
            # C++ reference).  It did not: the rewrite compiles to a
            # BYTE-IDENTICAL instruction count (98,907,904 both ways), so
            # ptxas was already sinking the second half's loads.  Kept only
            # because it states the dependency structure honestly; do not
            # expect it to buy registers.  The real spill source has not been
            # localized -- see profile/dhu-bwd-preprocess-opt/OPT_LOG.md.
            if cutlass.const_expr(is_h):
                p_bk.consumer_wait(st_bkc)
                for hh in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(sub_a1 // 4):
                        if cutlass.const_expr(self.need_src_tab):
                            rSrc[2 * j] = sSrc_t[
                                BTH * hh + cA1[4 * j][1], st_vg.index]
                            rSrc[2 * j + 1] = sSrc_t[
                                BTH * hh + cA1[4 * j + 1][1], st_vg.index]
                        else:
                            # src == scale * lambda exactly (lambda is 1 or 0)
                            base = scale
                            if cutlass.const_expr(self.need_lam):
                                rSrc[2 * j] = base * rLam[
                                    lam_half * hh + 2 * j]
                                rSrc[2 * j + 1] = base * rLam[
                                    lam_half * hh + 2 * j + 1]
                            else:
                                rSrc[2 * j] = base
                                rSrc[2 * j + 1] = base
                    cute.copy(s2r_v,
                              thr_v.partition_S(sDOt4[(None, None, hh, 0)]),
                              rOsA)
                    for j in cutlass.range_constexpr(sub_a1 // 4):
                        for q in cutlass.range_constexpr(4):
                            i = 4 * j + q
                            ryc[i] = (rOsA[i].to(cutlass.Float32)
                                      * rSrc[2 * j + (q & 1)]).to(dtype)
                    cute.copy(r2s_a, ryc, tRS_doH[hh])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    p_do.producer_commit(st_dop)
                    st_dop.advance()
                p_bk.consumer_release(st_bkc)
                st_bkc.advance()

            p_vg.consumer_release(st_vg)
            st_vg.advance()
            st_g1c.advance()

        # ---- pair-side push: drain once, then emit the two 64-column strips
        p_g2.consumer_wait(st_g2c)
        st_g2c.advance()
        pipeline.NamedBarrier(
            barrier_id=NB_EPILOG, num_threads=self.num_threads
        ).arrive_and_wait()

        # Mirror of the forward's dead-transition skip.  A consumer folds P_j
        # only for j in [rank+1, R-2] (`_bwd_merge_role` seeds the accumulator
        # from source R-1's Htilde and folds transitions below it), so the top
        # rank's transition is unreadable for every world size and every
        # packing -- unlike the *emit mask* below, which suppresses a
        # transition that is readable but must not act.  Pair 1 owns those
        # strips; at R == 2 rank R-1 is the only backward producer, so this
        # halves the wire.  The epilogue barrier above has already been
        # arrived at.
        if cutlass.const_expr(not (is_h or self.push_transition)):
            return st_vg, st_g1c, st_g2c, st_yp, st_dop, st_bkc

        cute.copy(t2r_x, tXt[(None, 0, 0)], rX)
        cute.arch.fence_view_async_tmem_load()
        # Emit mask: a rank whose summary must not reach its consumers still
        # has to push and flag -- they spin on those flags -- so the half is
        # zeroed rather than skipped.  Zeroed, not scaled by 0.0, because
        # 0.0 * inf is NaN.  See `launch_validated` and the forward twin.
        if (sI_t[7] & (1 if is_h else 2)) == 0:
            for i in cutlass.range_constexpr(2 * NS):
                rX[i] = cutlass.Float32(0.0)
        lane = ltid & 31
        warp_row0 = row0 - lane
        for ss in cutlass.range_constexpr(2):
            strip = strip0 + ss
            xbase = ss * NS
            off0 = self._hm_off(par, self.rank, hv, strip)
            warp_slot = (4 * strip + ltid // 32) * (32 * 32)
            for half in cutlass.range_constexpr(2):
                for v4 in cutlass.range_constexpr(8):
                    src4 = cute.make_tensor(
                        (rX.iterator + xbase + 32 * half + 4 * v4).align(16),
                        cute.make_layout(4))
                    phys = lane * 32 + 4 * (v4 ^ (lane & 7))
                    dst4 = cute.make_tensor(
                        (sPush_t.iterator + warp_slot + phys).align(16),
                        cute.make_layout(4))
                    cute.copy(atom_f32x4, src4, dst4)
                cute.arch.sync_warp()
                for q in cutlass.range_constexpr(8):
                    lr = 4 * q + lane // 8
                    v4 = lane & 7
                    phys = lr * 32 + 4 * (v4 ^ (lr & 7))
                    src4 = cute.make_tensor(
                        (sPush_t.iterator + warp_slot + phys).align(16),
                        cute.make_layout(4))
                    col = 32 * half + 4 * v4
                    for j in cutlass.range_constexpr(self.consumer_lo,
                                                     self.rank):
                        dst4 = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32,
                                mPtrs[0, j] + 4 * (off0
                                                   + (warp_row0 + lr) * NS
                                                   + col),
                                cute.AddressSpace.gmem, assumed_align=16),
                            cute.make_layout(4))
                        cute.copy(atom_f32x4, src4, dst4)
                cute.arch.sync_warp()

        return st_vg, st_g1c, st_g2c, st_yp, st_dop, st_bkc

    # ------------------------------------------------------------- protocol
    @cute.jit
    def _flag_bwd_consumers(self, mPtrs, hv, stepu):
        """Reverse direction: rank r flags every EARLIER consumer."""
        rb = self.rank * self.HV + hv
        for j in cutlass.range_constexpr(self.consumer_lo, self.rank):
            for s in cutlass.range_constexpr(self.NSTRIPS):
                self._signal_relaxed_sys(
                    mPtrs[1, j], rb * self.NSTRIPS + s, stepu)

    # ------------------------------------------------------------ merge role
    @cute.jit
    def _bwd_merge_role(self, tidx, b, hv, par, stepu, mPtrs, mDHT, pHMloc,
                        storage, tiled_mma, tiled_copy_M, sM_layout):
        """Descending suffix fold z <- P_j @ z + Htilde_j on warps 0-3.

        The incoming gradient to the global right boundary is zero, so the
        first live suffix term is exactly Htilde from the last source rank.
        """
        bar = pipeline.NamedBarrier(barrier_id=NB_MERGE, num_threads=128)
        dtype = self.io_dtype
        K = self.K
        rank = self.rank

        pRaw = cute.recast_ptr(
            storage.sX.get_tensor(cute.make_layout(1)).iterator,
            dtype=cutlass.Float32)
        mRaw = cute.make_tensor(pRaw, cute.make_layout(self.NK * self.STRIP))
        mRawHe = cute.make_tensor(pRaw + self.NK * self.STRIP,
                                  cute.make_layout(self.STRIP))
        pHiLo = cute.recast_ptr(pRaw + (self.NK + 1) * self.STRIP, dtype=dtype)
        sMhi = cute.make_tensor(pHiLo, sM_layout)
        sMlo = cute.make_tensor(pHiLo + K * (K + 8), sM_layout)
        stripT_layout = cute.make_layout((64, K), stride=(1, 64))
        mAddT_smem = cute.make_tensor(mRawHe.iterator, stripT_layout)

        fl_loc = mPtrs[1, rank]
        thr_mma = tiled_mma.get_slice(tidx)
        macc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((64, K)), cutlass.Float32)
        rAhi = cute.make_fragment_like(macc, dtype)
        rAlo = cute.make_fragment_like(macc, dtype)
        last = self.source_hi

        if tidx == 0:
            self._spin_ge_sys(
                fl_loc, (last * self.HV + hv) * self.NSTRIPS + b, stepu)
        bar.arrive_and_wait()
        self._stage_he_issue(pHMloc + self._hm_off(par, last, hv, b),
                             mRawHe, tiled_copy_M, tidx)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        bar.arrive_and_wait()
        tHlast = thr_mma.partition_C(mAddT_smem)
        for i in cutlass.range_constexpr(cute.size(macc)):
            macc[i] = tHlast[i]
        bar.arrive_and_wait()

        if cutlass.const_expr(rank + 1 < self.source_hi):
            j0 = self.source_hi - 1
            if tidx == 0:
                self._spin_producer_sys(fl_loc, j0, hv, b, stepu)
            bar.arrive_and_wait()
            self._stage_issue(pHMloc + self._hm_off(par, j0, hv, self.NV),
                              mRaw, tiled_copy_M, tidx)
            self._stage_he_issue(pHMloc + self._hm_off(par, j0, hv, b),
                                 mRawHe, tiled_copy_M, tidx)
            cute.arch.cp_async_commit_group()

        for j in cutlass.range_constexpr(self.source_hi - 1, rank, -1):
            cute.arch.cp_async_wait_group(0)
            bar.arrive_and_wait()
            self._convert_raw(mRaw, sMhi, sMlo, tidx)
            self._snapshot_and_init(thr_mma, macc, rAhi, rAlo,
                                    mAddT_smem, True)
            if cutlass.const_expr(j - 1 > rank):
                if tidx == 0:
                    self._spin_producer_sys(fl_loc, j - 1, hv, b, stepu)
            bar.arrive_and_wait()
            if cutlass.const_expr(j - 1 > rank):
                self._stage_issue(
                    pHMloc + self._hm_off(par, j - 1, hv, self.NV),
                    mRaw, tiled_copy_M, tidx)
                self._stage_he_issue(
                    pHMloc + self._hm_off(par, j - 1, hv, b),
                    mRawHe, tiled_copy_M, tidx)
                cute.arch.cp_async_commit_group()
            self._fold_gemms(thr_mma, tiled_mma, macc, rAhi, rAlo,
                             sMhi, sMlo, tidx)

        gDHT = cute.make_tensor(
            mDHT.iterator + (hv * self.K * self.V + b * 64),
            cute.make_layout((64, self.K), stride=(1, self.V)))
        tDHT = thr_mma.partition_C(gDHT)
        for i in cutlass.range_constexpr(cute.size(macc)):
            tDHT[i] = macc[i]
        bar.arrive_and_wait()
        if tidx == 0:
            cute.arch.fence_acq_rel_sys()
            for j in cutlass.range_constexpr(rank + 1, self.source_hi + 1):
                self._signal_relaxed_sys(
                    mPtrs[2, j],
                    (rank * self.HV + hv) * self.NB + b, stepu)


# ---------------------------------------------------------------------- host
class CuteDSLFusedCPBwdPreProcessWS(CuteDSLFusedCPBwdPreProcess):
    """Same host wrapper, warp-specialized reverse compute engine.

    Aligned calls (``T % BT == 0``) compile the ragged-tail machinery away,
    so chunk alignment joins the compile-cache key.  Both specializations
    coexist in the cache; a workload that alternates lengths pays one extra
    compile, not one per call.
    """

    OP_CLASS = PreProcessBwdFusedWS

    def _op_key(self, T):
        return (int(T) % BT == 0,)

    def _make_op(self, key):
        (full_chunks,) = key
        return PreProcessBwdFusedWS(
            self.H, self.HV, self.DK, self.DV, self.gate_mode,
            self.R, self.rank, full_chunks=full_chunks,
        )


def ws_supported(K: int, V: int, gate_mode: str) -> bool:
    """Shape gate for the warp-specialized backward.

    Everything outside this envelope keeps the non-WS backward, which
    supports K in {64,128}, any 64-multiple V and bounded CP neighborhoods.
    """
    return K == V == 128 and gate_mode in ("none", "g", "gk")


@torch.no_grad()
def chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl_ws(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context=None,
    chunk_size: int = BT,
):
    """FLA-signature adapter selecting the warp-specialized engine.

    Shapes outside :func:`ws_supported` fall back to the non-WS backward
    rather than failing, so this entry point is a drop-in replacement.
    """
    K = q.shape[-1]
    V = do.shape[-1]
    mode = "g" if g is not None else ("gk" if gk is not None else "none")
    cls = (CuteDSLFusedCPBwdPreProcessWS if ws_supported(K, V, mode)
           else None)
    return _bwd_entry(
        q, k, w, do, dv, g=g, gk=gk, bg=bg, scale=scale,
        state_v_first=state_v_first, cu_seqlens=cu_seqlens, dht=dht,
        initial_state=initial_state, context=context, chunk_size=chunk_size,
        wrapper_cls=cls,
    )


chunk_gated_delta_rule_bwd_dhu_pre_process = (
    chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl_ws
)
