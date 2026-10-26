# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""Warp-specialized tcgen05 (Blackwell) fused CP pre-process — v4 "WS".

v3 engine (pair ping-pong). Wire protocol (parity-buffered hm_sym, monotonic
flags, credit acks, merge role) and SM80 fold from `protocol.CPProtocolMixin`;
compute engine built on warp specialization + TMA + tensor memory. Design
history: docs/ws_kernel_design.md; API notes: refs/ws_*.md.

MATH (per value-head hv; T tokens in NT = ceil(T/64) chunks of BT=64).
Inputs per chunk t (all bf16): W_t = w[t0:t0+64] in R^{64xK} (beta-scaled
decay keys), K_t = k[t0:t0+64] in R^{64xK} (L2-normalized keys),
V_t = u[t0:t0+64] in R^{64xV} (values); gates g are log2-domain,
chunk-local-cumsummed (g_i = sum of log2-decays within the chunk up to
token i; g_last = g at the chunk's final valid row).

State X = [h | M] in R^{Kx(V+K)} fp32, split into NSTRIPS = V/64 + K/64
column strips X_s in R^{Kx64}; h(0) = 0, M(0) = I. Each strip lives in
TENSOR MEMORY as its gemm2 accumulator. Per chunk:

  lambda_i = 2^(g_last - g_i)                    (row decays, fp32; i < 64)
  gamma    = 2^(g_last)                          (chunk decay; gk: per-K-row)
  x_t      = lambda ⊙ ([V_t | 0] - W_t @ X)      in R^{64x(V+K)}   (gemm1 +
                                                  elementwise pass 2)
  X       <- gamma * X + K_t^T @ x_t             (decay pass 1b + gemm2,
                                                  ACCUMULATE=1 in TMEM)

After NT chunks X = [h | M] satisfies h = sum_i decay_i k_i^T x_i (the
chunk-parallel delta-rule prefix state) and M = prod of per-token decays
projected through (I - w k^T) factors — exactly fla's pre_process output;
strips are pushed to every consumer rank's hm_sym buffer for the CP merge
fold h <- M_j @ h + he_j.

Compute CTA layout (K=V=128: 384 threads = 12 warps):
  warps 0..3        pair group 0 — owns both 64-col h strips
  warps 4..7        pair group 1 — owns both 64-col M strips
  warp 8            TMA warp — bulk-tensor W/K/V/gate loads
  warp 9            MMA warp — issues all tcgen05 gemms
  warps 10..11      lambda helpers — publish the 64-row lambda table

Each pair thread owns one K row across both adjacent strips. The pair's
X[K,128] state stays in TENSOR MEMORY as gemm2's accumulator and drains as
one x128 fragment. K=64 retains the existing two one-strip groups (10 warps).

Strips are grouped into 2 MMA PAIRS (h strips / M strips). Per chunk t,
per pair (all bf16 in / fp32 acc, CtaGroup.ONE; v5 "flip" shapes —
transposed gemm1 for the full M=128 datapath):
  pair pass 1a: publish X_bf16 -> smem       (TMEM ld, cvt, coalesced STS;
               K=128: per-warp-quarter arrives feed the gemm1 staircase)
  MMA warp:    gemm1_p: acc1_p^T = X_p^T @ W_t^T  M=64*SPP N=BT  Kdim=K
  pair pass 1b (under gemm1): X *= gamma decay TMEM ld->st
  pair pass 2 (after acc1 wait): x^T = lam*([v|0]^T - acc1^T) -> bf16 smem,
               two token halves (V^T ldmatrix.trans + lambda in the acc1
               TMEM-load shadow — conflict- and spill-free)
  MMA warp:    gemm2_p: X_p += K_t^T @ x_p^T       M=K  N=64*SPP  (ACC=1)

The two pairs ping-pong on the MMA pipe: pair A's TMEM/elementwise phase
runs under pair B's gemm execution and vice versa (FA4-style). All MMA
issue/commit is single-warp (tcgen05 issue + tcgen05.commit are
elect_one-per-warp; multi-warp issue duplicates MMAs and over-arrives the
1-count producer mbarriers — the v2 deadlock). Only full-side mbarrier
edges are used: acc reuse is ordered by program order + the rendezvous.
gemm2(t-1)'s wait sits at the top of chunk t so the V/gate wait and exp2
overlap gemm2 flight; the decay TMEM store hides under gemm1.

PERSISTENT: grid = min(#SM, work items); each CTA loops work items
(item i < HV: compute head i; else merge pair (hv, b)). TMEM alloc and
pipeline init happen once per CTA. Merge items run the SM80 fold on
warps 0-3 between full-CTA item barriers (their staging overlays the
compute smem).

Restrictions kept: K == V in {64, 128}; g-mode needs HV % 4 == 0;
modes none/g/gk; no DPLR; B = 1 (packed sequences are windowed by the
caller); one whole-head scan per item (no T-split).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ._host import CPHostCaches, _normalize_gate
from .protocol import BT, CPProtocolMixin, tmem_alloc_ptr

NS = 64  # strip width
VGS = 2  # V/gate pipeline stages (3 measured 5us WORSE: smem carve-out shrinks L1)

# (a pair-B FMA-chain start-offset probe — "SKEW" — lived here; probed
# 200/400/800 steps, all ~= +1us: phase-collision hypothesis dead)

# named barrier ids (0 is reserved for sync_threads)
NB_MERGE = 1  # merge role work group (128 threads)
NB_EPILOG = 2  # compute item epilogue (all threads)
NB_TMEM_FREE = 3  # tmem dealloc (all threads)
NB_TMEM_ALLOC = 4  # tmem alloc retrieve (all threads)
NB_ITEM = 5  # merge item boundaries (all threads)
# (deadlock-triage printf tracing: see dbg/cutedsl_fused_ws7.py, WS_DEBUG)

# K=128 peer push: the native row-owner epilogue would emit one 16B vector
# from each 256B-separated row per warp instruction, consuming only half of
# every 32B global-store sector.  The epilogue instead stages through the
# now-dead sX storage as one 32x32 fp32 tile per compute warp and
# redistributes each vector instruction across four adjacent rows so every
# sector is populated (see `_pair_item` / `_wg_item`).


class PreProcessFwdFusedWS(CPProtocolMixin):
    @staticmethod
    def build_key(gate_mode, full_chunks):
        """Compile-cache key: what `full_chunks` actually changes in codegen.

        It reaches the emitted program only through
        `need_lam = use_g or not full_chunks`, so under scalar `g` both values
        produce the same binary and must share one cache entry.

        Under per-key `gk` the two programs really do differ, and the wrapper
        deliberately runs the ragged (tail-masked) one for every `T`: the key
        is `True` regardless of `full_chunks`, and the wrapper constructs that
        single entry with `full_chunks=False`.  One `cute.compile` per
        wrapper, at the first launch (which every rank reaches in lockstep
        right after the rank-consistent handshake), instead of a second
        multi-second compile at the first non-64-aligned boundary window --
        which under THD lands at a random mid-training step on one rank while
        its peers spin in symmetric memory.  The cost is the lambda 1/0 table
        on aligned calls, the same work scalar `g` already does for lambda.

        `True` therefore means "the generic specialisation" in every mode.
        """
        return gate_mode in ("g", "gk") or not full_chunks

    def __init__(self, H, HV, K, V, gate_mode, R, rank, n_sm=None, full_chunks=False):
        CPProtocolMixin.__init__(self, H, HV, K, V, gate_mode, R, rank)
        if self.use_g:
            assert HV % 4 == 0, "WS g-mode needs HV % 4 == 0 (TMA box)"
        self.NG = self.NSTRIPS
        self.SPP = max(self.NSTRIPS // 2, 1)  # strips per MMA pair
        # K=128 combines the two strips of each MMA pair into one warpgroup:
        # 8 pair warps + TMA + MMA + 2 lambda helpers. K=64 already has one
        # strip per pair and keeps its original 8 worker + 2 aux warps.
        self.num_threads = 384 if K == 128 else 128 * self.NG + 64
        self.tma_warp = 8
        self.mma_warp = 9
        self.lam_warp = 10
        # Aligned calls have lambda == 1 for every token in non-g modes and
        # compile out the lambda publish/wait/load/multiply chain. Ragged calls
        # use the generic specialization so lambda also serves as the tail
        # validity mask.
        self.full_chunks = full_chunks
        self.need_lam = self.use_g or not full_chunks
        # Does any rank read this rank's transition (M) half?  A consumer folds
        # M_j only for j >= 1, so rank 0's is dead for every world size and
        # every chain.  See `_pair_item` / `_wg_item`.
        self.push_transition = rank != 0
        # ... and an unread transition need not be *computed* either.  The
        # compute item is organised as warp groups of 4: MMA pairs at K=128
        # (pair 0 = the h strips, pair 1 = the M strips) and single strips at
        # K=64, and because the WS engine requires K == V the transition is
        # always exactly the upper half of them.  Dropping that half at rank 0
        # removes both of its per-chunk gemms from the serial scan.  The
        # remaining groups still load the same w/k tiles, so this is not a
        # halving -- it takes the scan from compute-bound toward its load
        # floor.  Under THD that matters more than under BSHD in absolute
        # terms per token, because the same fixed costs sit on a shorter scan.
        self.n_groups = 2 if K == 128 else self.NG
        self.n_groups_live = self.n_groups if self.push_transition else self.n_groups // 2
        self.need_decay = self.use_g or self.use_gk
        if n_sm is None:
            n_sm = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        self.n_citems = HV if rank < R - 1 else 0
        self.n_mitems = HV * self.NB if rank > 0 else 0
        self.n_items = self.n_citems + self.n_mitems
        assert self.n_items > 0
        self.n_ctas = min(n_sm, self.n_items)
        self.n_sm = n_sm
        # compute items per CTA: whole-head walk over HV heads, so a CTA may
        # need ceil(HV/eff) slots
        nwork = HV
        eff0 = max(1, min(self.n_ctas, nwork))
        self.NSLOT = -(-nwork // eff0)

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        pK: cute.Pointer,  # [1, T, H,  K] bf16
        pV: cute.Pointer,  # [1, T, HV, V] bf16 (u)
        pW: cute.Pointer,  # [1, T, HV, K] bf16
        pG: cute.Pointer,  # [1, T, HV] fp32 or dummy
        pGK: cute.Pointer,  # [1, T, HV, K] fp32 or dummy
        pHMloc: cute.Pointer,  # local hm_sym base, fp32 [2,R,HV,NSTRIPS,K,64]
        pPtrs: cute.Pointer,  # int64 [3, R]
        pH0: cute.Pointer,  # fp32 [HV, K, V] out
        pSegFl: cute.Pointer,  # walk descriptors (per-(CTA, slot) items)
        T: cutlass.Int32,
        step: cutlass.Int32,
        emit: cutlass.Int32,  # bit 0 = push h half, bit 1 = push M half
        stream: cuda.CUstream,
    ):
        H, HV, K = self.H, self.HV, self.K
        NG = self.NG
        dtype = self.io_dtype

        # ---- gmem views shaped for the TMA atoms (B = 1: the batch mode is
        # dropped and the descriptor box is T rows tall)
        mW3 = cute.make_tensor(pW, cute.make_layout((T, K, HV), stride=(HV * K, 1, K)))
        mK3 = cute.make_tensor(pK, cute.make_layout((K, T, H), stride=(1, H * K, K)))
        mV3 = cute.make_tensor(
            pV, cute.make_layout((T, self.V, HV), stride=(HV * self.V, 1, self.V))
        )
        mG2 = cute.make_tensor(pG, cute.make_layout((T, HV), stride=(HV, 1)))
        mGk2 = cute.make_tensor(pGK, cute.make_layout((T * HV, K), stride=(K, 1)))
        mPtrs = cute.make_tensor(pPtrs, cute.make_layout((3, self.R), stride=(self.R, 1)))
        mH0 = cute.make_tensor(
            pH0, cute.make_layout((HV, K, self.V), stride=(K * self.V, self.V, 1))
        )

        # ---- tcgen05 tiled MMAs (strip-width N=64)
        def mk_mma(amaj, bmaj, shape):
            return sm100_utils.make_trivial_tiled_mma(
                dtype,
                tcgen05.OperandMajorMode(amaj),
                tcgen05.OperandMajorMode(bmaj),
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                shape,
                tcgen05.OperandSource.SMEM,
            )

        # A = W tile [64, K] k-major; B = X_bf16 (64, K) N-contig
        tiled_mma1 = mk_mma("k", "mn", (BT, NS))
        # A = K_t^T [K, 64] M-contig; B = x (64, 64) N-contig
        tiled_mma2 = mk_mma("mn", "mn", (K, NS))
        tiler1 = (BT, NS, K)
        tiler2 = (K, NS, BT)

        # ---- pair MMAs, TRANSPOSED gemm1 (v5 "flip"): per pair
        #   acc1^T = X_pair^T @ W^T   M = NS*SPP (=128)  N = BT  Kdim = K
        # instead of acc1 = W @ X (M=64). M=64 runs the 128-lane tensor
        # datapath at half rate AND doubles the TMEM C-phase cycles (ncu:
        # tensor pipe 49% active, mem_tensor 70% — the real "long
        # scoreboard" source). Flipping runs gemm1 full-rate:
        #   A = X^T [NS*SPP, K] mn-major — probed BYTE-IDENTICAL to the
        #       concatenated per-strip B tiles, so the pass-1a publish
        #       path is unchanged (dbg/probe_flip_layouts.py);
        #   B = W [BT, K] k-major — probed byte-identical to the k-major
        #       A layout, so the W TMA/buffer is unchanged;
        #   C = acc1^T [NS*SPP, BT] — fragment layout == the X state
        #       fragment, so pass 2 reuses the [128,64] TMEM machinery
        #       (one M-subtile per 64-column half of the pair).
        # gemm2 keeps C = X (M=K) but its B becomes x^T k-major (pass 2
        # publishes x transposed). Same products, same k-phase order in
        # both gemms -> bit-exact (same argument as the N-widening
        # merges; verified by ws_iter's max|diff|=0 gate).
        SPP = self.SPP
        # A = X^T (strip-col contig); B = W tile [BT, K] k-major
        tiled_mma1f = mk_mma("mn", "k", (NS * SPP, BT))
        # A = K_t^T [K, 64] M-contig; B = x^T (token contig)
        tiled_mma2f = mk_mma("mn", "k", (K, NS * SPP))
        tiler1f = (NS * SPP, BT, K)
        tiler2f = (K, NS * SPP, BT)

        sW_lay = sm100_utils.make_smem_layout_a(tiled_mma1, tiler1, dtype, 2)
        sKt_lay = sm100_utils.make_smem_layout_a(tiled_mma2, tiler2, dtype, 2)
        # strip index rides the "stage" mode of the B layouts (publish view)
        sX_lay = sm100_utils.make_smem_layout_b(tiled_mma1, tiler1, dtype, NG)
        # MMA-side views for the flipped gemms:
        #  sXf: pair A operand (X^T mn-major) over the SAME sX bytes
        #  sWb: W as gemm1f's B (k-major) over the SAME sW bytes
        #  sxt: x^T k-major B of gemm2f, stage = pair (this one is a real
        #       layout change — pass 2 publishes transposed)
        sXf_lay = sm100_utils.make_smem_layout_a(tiled_mma1f, tiler1f, dtype, 2)
        sWb_lay = sm100_utils.make_smem_layout_b(tiled_mma1f, tiler1f, dtype, 2)
        sxt_lay = sm100_utils.make_smem_layout_b(tiled_mma2f, tiler2f, dtype, 2)
        # V gets the canonical SW128 A-operand layout (same (BT, K) shape as
        # W): a plain (BT, V) layout has 256B rows, so a fixed 16B column
        # chunk aliases to the same banks on every row and the per-chunk V
        # reads were 8-way bank-conflicted (ncu: 84% of shared-load
        # wavefronts). The swizzle XORs the 16B chunk with row%8.
        # NOTE: folding v into gemm1 via -I ext k-phases (roadmap lever) was
        # tried 2026-07-16 and REVERTED: tcgen05's accumulator add is not
        # IEEE-RN-equivalent once |acc| >> |v| (deterministic ~3e-3 rel
        # drift past ~20 chunks), which breaks the bit-exactness contract.
        sV_lay = sm100_utils.make_smem_layout_a(tiled_mma1, tiler1, dtype, VGS)
        sG_lay = cute.make_layout((BT, 4, VGS), stride=(4, 1, BT * 4))
        sGk_lay = cute.make_layout((1, K, VGS), stride=(K, 1, K))

        # ---- TMA atoms (raw-ptr + dynamic-T; refs/ws_tma-dynamic.md)
        def mk_tma_a(op, m, slay, tiler, mma):
            return cute.nvgpu.make_tiled_tma_atom_A(
                op,
                m,
                cute.slice_(slay, (None, None, None, 0)),
                tiler,
                mma,
                cute.make_layout((1, 1, 1, 1)).shape,
            )

        # both MMAs are CtaGroup.ONE with thr_id size 1, so the TMA-A op
        # (mcast/CtaGroup selection only) is identical for all three loads
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1, 1), tiled_mma1.thr_id)
        atom_w, tW = mk_tma_a(a_op, mW3, sW_lay, tiler1, tiled_mma1)
        atom_kt, tKt = mk_tma_a(a_op, mK3, sKt_lay, tiler2, tiled_mma2)
        atom_v, tV = mk_tma_a(a_op, mV3, sV_lay, tiler1, tiled_mma1)
        if cutlass.const_expr(self.use_g):
            atom_g, tG = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mG2,
                cute.make_layout((BT, 4), stride=(4, 1)),
                (BT, 4),
            )
        else:
            atom_g, tG = atom_v, tV
        if cutlass.const_expr(self.use_gk):
            atom_gk, tGk = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mGk2,
                cute.make_layout((1, K), stride=(K, 1)),
                (1, K),
            )
        else:
            atom_gk, tGk = atom_v, tV

        self.wk_bytes = cute.size_in_bytes(
            dtype, cute.slice_(sW_lay, (None, None, None, 0))
        ) + cute.size_in_bytes(dtype, cute.slice_(sKt_lay, (None, None, None, 0)))
        self.vg_bytes = (
            BT * self.V * dtype.width // 8
            + (BT * 4 * 4 if self.use_g else 0)
            + (K * 4 if self.use_gk else 0)
        )

        # ---- merge-role machinery (SM80 fold, as in the parent kernel)
        sM_layout = cute.make_layout((K, K), stride=(K + 8, 1))
        atom_g2s_f32 = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        tiled_copy_M = cute.make_tiled_copy_tv(
            atom_g2s_f32, cute.make_layout(128), cute.make_layout(4)
        )
        tiled_mma_sm80 = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
            (4, 1, 1),
            permutation_mnk=(64, 16, 16),
        )

        sx_size = cute.cosize(sxt_lay)
        sX_size = cute.cosize(sX_lay)
        sW_size = cute.cosize(sW_lay)
        sKt_size = cute.cosize(sKt_lay)

        @cute.struct
        class SharedStorage:
            # mbarriers: full+empty pairs per stage
            wk_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            vg_mbar: cute.struct.MemRange[cutlass.Int64, 2 * VGS]
            g1_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            g2_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            pX_mbar: cute.struct.MemRange[cutlass.Int64, 24]
            px_mbar: cute.struct.MemRange[cutlass.Int64, 8]
            tmem_holding_buf: cutlass.Int32
            # ---- big buffers: 1024-aligned, 1024-multiple sizes, contiguous
            # (the merge role overlays its staging here)
            sX: cute.struct.Align[cute.struct.MemRange[dtype, sX_size], 1024]
            sx: cute.struct.Align[cute.struct.MemRange[dtype, sx_size], 1024]
            sW: cute.struct.Align[cute.struct.MemRange[dtype, sW_size], 1024]
            sKt: cute.struct.Align[cute.struct.MemRange[dtype, sKt_size], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[dtype, BT * self.V * VGS], 1024]
            sG: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, BT * 4 * VGS if self.use_g else 8], 128
            ]
            sGk: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, K * VGS if self.use_gk else 8], 128
            ]
            # per-item work descriptor (hv, c0, c1, n_later, valid),
            # loaded from the host-precomputed table by thread 0
            sItem: cute.struct.MemRange[cutlass.Int32, 8]
            # per-chunk lambda table (one value per token row), written
            # once by 2 warps instead of redundantly by every strip
            # thread; rides the vg stage lifecycle
            sLam: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, BT * VGS], 128]

        self.shared_storage = SharedStorage

        self.ws_kernel(
            atom_w,
            tW,
            atom_kt,
            tKt,
            atom_v,
            tV,
            atom_g,
            tG,
            atom_gk,
            tGk,
            tiled_mma1,
            tiled_mma2,
            tiled_mma1f,
            tiled_mma2f,
            sW_lay,
            sX_lay,
            sKt_lay,
            sXf_lay,
            sWb_lay,
            sxt_lay,
            sV_lay,
            sG_lay,
            sGk_lay,
            tiled_mma_sm80,
            tiled_copy_M,
            sM_layout,
            mPtrs,
            mH0,
            T,
            step,
            emit,
            pHMloc,
            pSegFl,
        ).launch(
            grid=[self.n_ctas, 1, 1],
            block=[self.num_threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def ws_kernel(
        self,
        atom_w: cute.CopyAtom,
        tW: cute.Tensor,
        atom_kt: cute.CopyAtom,
        tKt: cute.Tensor,
        atom_v: cute.CopyAtom,
        tV: cute.Tensor,
        atom_g: cute.CopyAtom,
        tG: cute.Tensor,
        atom_gk: cute.CopyAtom,
        tGk: cute.Tensor,
        tiled_mma1: cute.TiledMma,
        tiled_mma2: cute.TiledMma,
        tiled_mma1f: cute.TiledMma,
        tiled_mma2f: cute.TiledMma,
        sW_lay: cute.ComposedLayout,
        sX_lay: cute.ComposedLayout,
        sKt_lay: cute.ComposedLayout,
        sXf_lay: cute.ComposedLayout,
        sWb_lay: cute.ComposedLayout,
        sxt_lay: cute.ComposedLayout,
        sV_lay: cute.ComposedLayout,
        sG_lay: cute.Layout,
        sGk_lay: cute.Layout,
        tiled_mma_sm80: cute.TiledMma,
        tiled_copy_M: cute.TiledCopy,
        sM_layout: cute.Layout,
        mPtrs: cute.Tensor,
        mH0: cute.Tensor,
        T: cutlass.Int32,
        step: cutlass.Int32,
        emit: cutlass.Int32,
        pHMloc: cute.Pointer,
        pSegFl: cute.Pointer,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        K, NG = self.K, self.NG
        R, rank = self.R, self.rank
        par = step % 2
        stepu = step.to(cutlass.Uint32)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # frozen trace-time descriptors; ops emit at the call sites
        nb_item = pipeline.NamedBarrier(barrier_id=NB_ITEM, num_threads=self.num_threads)
        nb_epilog = pipeline.NamedBarrier(barrier_id=NB_EPILOG, num_threads=self.num_threads)

        has_compute = self.n_citems > 0
        if cutlass.const_expr(has_compute):
            # trace-time pipeline builders (inlined -> objects identical to
            # the unrolled .create calls). cg(): count-less single-thread
            # groups all alias one shared immutable descriptor, cg(n): a fresh
            # n-thread group. mk_pipe(): the shared create(defer_sync=True)
            # shape; tx_count is passed only for the TMA-fed pipelines.
            cg_thr = pipeline.CooperativeGroup(pipeline.Agent.Thread)

            def cg(count=None):
                return (
                    cg_thr
                    if count is None
                    else pipeline.CooperativeGroup(pipeline.Agent.Thread, count)
                )

            def mk_pipe(cls, stages, prod, cons, mbar, tx=None):
                if cutlass.const_expr(tx is not None):
                    return cls.create(
                        num_stages=stages,
                        producer_group=prod,
                        consumer_group=cons,
                        tx_count=tx,
                        barrier_storage=mbar,
                        defer_sync=True,
                    )
                return cls.create(
                    num_stages=stages,
                    producer_group=prod,
                    consumer_group=cons,
                    barrier_storage=mbar,
                    defer_sync=True,
                )

            p_wk = mk_pipe(
                pipeline.PipelineTmaUmma,
                2,
                cg_thr,
                cg_thr,
                storage.wk_mbar.data_ptr(),
                tx=self.wk_bytes,
            )
            # Only the pair/strip warps release V/gate stages. The two K=128
            # lambda helpers wait on vg_full but deliberately do not arrive on
            # vg_empty; pair release happens after their table read.
            # `PipelineTmaAsync` has lane 0 of each consumer warp arrive on the
            # empty barrier, so the count is 4 warps per LIVE group -- 8
            # normally, 4 on a rank whose transition group is compiled out.
            p_vg = mk_pipe(
                pipeline.PipelineTmaAsync,
                VGS,
                cg_thr,
                cg(4 * self.n_groups_live),
                storage.vg_mbar.data_ptr(),
                tx=self.vg_bytes,
            )
            # per-pair "gemm done" + operand-ready pipelines: the MMA warp
            # commits/waits, WG threads of the pair wait/arrive (full side
            # only; TMEM/operand reuse ordering comes from program order)
            SPP = self.SPP
            p_g1s, p_g2s, p_xs = [], [], []
            for pp in cutlass.range_constexpr(2):
                # per-pair "gemm done" edges (identical but for the mbar
                # storage; plain-python inner loop, trace-time only)
                for lst, mbar in ((p_g1s, storage.g1_mbar), (p_g2s, storage.g2_mbar)):
                    lst.append(
                        mk_pipe(
                            pipeline.PipelineUmmaAsync, 1, cg_thr, cg(128), mbar.data_ptr() + 2 * pp
                        )
                    )
                # 2 stages = the two token halves of x^T per chunk:
                # gemm2's first k-phases run under pass 2's second half
                p_xs.append(
                    mk_pipe(
                        pipeline.PipelineAsyncUmma,
                        2,
                        cg(128),
                        cg_thr,
                        storage.px_mbar.data_ptr() + 4 * pp,
                    )
                )
            # X-publish edges (per pair). K=128: 4 per-warp-quarter
            # staircase pipelines — "X rows 32w..32w+31 published"; the
            # MMA warp starts gemm1f's k-phase pair w as soon as those
            # rows land (one pair warp publishes both strips; 32
            # threads). K=64: the M64 lane interleave cannot align rows
            # to k-phases, so it keeps ONE whole-pair edge (128*SPP
            # threads). Only the used set is created.
            NPUB = 4 if K == 128 else 1
            p_pubs = []
            for pp in cutlass.range_constexpr(2):
                qs = []
                for w in cutlass.range_constexpr(NPUB):
                    qs.append(
                        mk_pipe(
                            pipeline.PipelineAsyncUmma,
                            1,
                            cg(32 if K == 128 else 128 * SPP),
                            cg_thr,
                            storage.pX_mbar.data_ptr() + 2 * (4 + NPUB * pp + w),
                        )
                    )
                p_pubs.append(qs)
            # Lambda edge (K=128: dedicated helper warps 10-11; K=64:
            # strip-0, v6 scheme). Pair-1 readers gate on the MMA warp's
            # p_lam wait before the g1 pair-1 commit (non-binding); pair-0
            # readers wait the full barrier directly after their g1 wake.
            # This is a 2-stage (VGS) full-only pipeline on slots 4..7 of
            # pX_mbar; nobody explicitly releases it. sLam stage reuse is
            # ordered by the writers' own vg consumer_wait(t+2), which
            # implies every reader released vg(t) after its last lambda(t)
            # read. The (index, phase) sequence matches p_vg's / p_wk's,
            # so their states are reused without another loop-carried pair.
            p_lam = mk_pipe(
                pipeline.PipelineAsync, VGS, cg(64), cg(128), storage.pX_mbar.data_ptr() + 4
            )
            # (pair-skew gate tried here and REVERTED, +13 us: pass-1a
            # drains are TMEM-ld latency, not port bandwidth — PROGRESS
            # #42)
            pipeline_init_arrive(cluster_shape_mn=(1, 1, 1), is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=(1, 1, 1))

            # tmem: NG acc1 strips + NG X strips, 64 cols each
            tmem_bar = pipeline.NamedBarrier(barrier_id=NB_TMEM_ALLOC, num_threads=self.num_threads)
            tmem = utils.TmemAllocator(
                tmem_alloc_ptr(storage.tmem_holding_buf),
                barrier_for_retrieve=tmem_bar,
                allocator_warp_id=0,
            )
            tmem.allocate(2 * NS * NG)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

            # TMEM map (flip): acc1^T pair p = BT cols at [BT*p, BT*p+BT)
            # (M = NS*SPP rows), X strips at [2*BT + NS*s). acc1 shrank
            # from NS*NG to 2*BT cols; alloc size keeps the legal
            # 2*NS*NG (512 for K=128 covers the 384 used; 256 == used
            # for K=64).
            state_lay = tiled_mma2.make_fragment_C(tiled_mma2.partition_shape_C((K, NS))).layout
            acc1f_lay = tiled_mma1f.make_fragment_C(
                tiled_mma1f.partition_shape_C((NS * SPP, BT))
            ).layout
            stateq_lay = tiled_mma2f.make_fragment_C(
                tiled_mma2f.partition_shape_C((K, NS * SPP))
            ).layout
            tXts = []
            for s in cutlass.range_constexpr(NG):
                tXts.append(cute.make_tensor(tmem_ptr + 2 * BT + NS * s, state_lay))
            tAcc1ps = []
            tXps = []
            for pp in cutlass.range_constexpr(2):
                tAcc1ps.append(cute.make_tensor(tmem_ptr + BT * pp, acc1f_lay))
                tXps.append(cute.make_tensor(tmem_ptr + 2 * BT + NS * SPP * pp, stateq_lay))

            sW_t = storage.sW.get_tensor(sW_lay.outer, swizzle=sW_lay.inner)
            sKt_t = storage.sKt.get_tensor(sKt_lay.outer, swizzle=sKt_lay.inner)
            sX_t = storage.sX.get_tensor(sX_lay.outer, swizzle=sX_lay.inner)
            # Cold epilogue-only fp32 view over the same 64-KiB allocation.
            # At K=128 this is exactly 16 warp-private 32x32 tiles; all hot
            # sX users are finished before NB_EPILOG lets the remap run.
            sPush_t = cute.make_tensor(
                cute.recast_ptr(storage.sX.data_ptr(), dtype=cutlass.Float32),
                cute.make_layout(16 * 32 * 32),
            )
            sXf_t = storage.sX.get_tensor(sXf_lay.outer, swizzle=sXf_lay.inner)
            sWb_t = storage.sW.get_tensor(sWb_lay.outer, swizzle=sWb_lay.inner)
            sxt_t = storage.sx.get_tensor(sxt_lay.outer, swizzle=sxt_lay.inner)
            sV_t = storage.sV.get_tensor(sV_lay.outer, swizzle=sV_lay.inner)
            sLam_t = storage.sLam.get_tensor(cute.make_layout((BT, VGS), stride=(1, BT)))
            sI_t = storage.sItem.get_tensor(cute.make_layout(8))
            if cutlass.const_expr(self.use_g):
                sG_t = storage.sG.get_tensor(sG_lay)
            else:
                sG_t = storage.sG.get_tensor(cute.make_layout((8, 1, 1), stride=(1, 0, 0)))
            if cutlass.const_expr(self.use_gk):
                sGk_t = storage.sGk.get_tensor(sGk_lay)
            else:
                sGk_t = storage.sGk.get_tensor(cute.make_layout((1, 8, 1), stride=(8, 1, 0)))

        # pipeline states persist across items (mbarrier phases carry
        # over); single shared state per pipeline kind — all strips advance
        # in lockstep (lists of states break MLIR SSA dominance in dynamic
        # loops). _c = consumer view, _p = producer view.
        # v9 consolidation: producer_commit reads ONLY state.index, which
        # is constant 0 for every 1-stage pipeline — so ONE frozen,
        # never-advanced state (st_one) serves ALL 1-stage producer
        # commits (g1/g2 of both pairs on the MMA warp, the X publish on
        # the WG side). The two pairs' pX consumer states had identical
        # (index, phase) sequences and are merged into st_pub_c. p_lam
        # carries no states at all (rides st_vg / st_wk). The px pair-B
        # consumer keeps its own state: the within-chunk wait order is
        # p0h0, p0h1, p1h0, p1h1 = stage 0,1,0,1, which one 2-stage
        # state cannot emit.
        if cutlass.const_expr(has_compute):
            Prod = pipeline.PipelineUserType.Producer
            Cons = pipeline.PipelineUserType.Consumer
            mps = pipeline.make_pipeline_state
            st_one = mps(Prod, 1)
            st_wk_c = mps(Cons, 2)
            st_wk_p = mps(Prod, 2)
            st_vg_c = mps(Cons, VGS)
            st_vg_p = mps(Prod, VGS)
            st_g1_c = mps(Cons, 1)
            st_g2_c = mps(Cons, 1)
            st_pub_c = mps(Cons, 1)
            st_x_p = mps(Prod, 2)
            st_x_c = mps(Cons, 2)
            st_x_cb = mps(Cons, 2)

        # ==================== persistent item loop =====================
        # compute items come from a host-precomputed per-(CTA, slot) table
        # (hv, 0, 0, 0, valid): constexpr slot loop + one dynamic validity
        # branch — the same shape as the plain per-head loop (walk arithmetic
        # in registers, or device integer division anywhere in this function,
        # can tip register allocation into spills)
        if cutlass.const_expr(has_compute):
            for slot in cutlass.range_constexpr(self.NSLOT):
                nb_item.arrive_and_wait()
                if tidx == 0:
                    dsc = cute.make_tensor(
                        pSegFl
                        + (
                            cutlass.Int32(self.n_sm) + (cutlass.Int32(bidx) * self.NSLOT + slot) * 8
                        ),
                        cute.make_layout(8),
                    )
                    for q in cutlass.range_constexpr(5):
                        sI_t[q] = dsc[q].to(cutlass.Int32)
                    # slot 7 = this launch's emit mask.  It rides the item
                    # descriptor rather than a threaded argument because the
                    # push epilogue is the only reader, and a kernel argument
                    # would otherwise stay live in a register across the whole
                    # chunk loop of a function that is already at its spill
                    # cliff.  Not part of the host walk table -- that is
                    # uploaded once per wrapper, and the mask changes with the
                    # packing.
                    sI_t[7] = emit
                nb_item.arrive_and_wait()
                if sI_t[4] != 0:
                    # ------------- compute item: head hv, chunks [0, nc) ---
                    hv = sI_t[0]
                    c0 = sI_t[1]
                    # Whole-head walk: every live row has c0 == 0, so the
                    # chunk count is a function of the dynamic `T` alone.
                    # Reading it from `T` instead of from the table is what
                    # lets the resident walk table stay untouched when the
                    # length changes -- under THD `nt` changes every step,
                    # and a per-length retune was a ~5.7 us host
                    # `index_fill_` dispatch plus its own kernel launch,
                    # per direction per layer per step.
                    nc = (T + (BT - 1)) // BT
                    if warp_idx == self.tma_warp:
                        st_wk_p, st_vg_p = self._tma_item(
                            atom_w,
                            tW,
                            atom_kt,
                            tKt,
                            atom_v,
                            tV,
                            atom_g,
                            tG,
                            atom_gk,
                            tGk,
                            tiled_mma1,
                            tiled_mma2,
                            sW_t,
                            sKt_t,
                            sV_t,
                            sG_t,
                            sGk_t,
                            T,
                            c0,
                            nc,
                            hv,
                            p_wk,
                            p_vg,
                            st_wk_p,
                            st_vg_p,
                        )
                        # ack throttle before anyone pushes
                        ackv = cutlass.max(step - 2, 0).to(cutlass.Uint32)
                        ak_loc = mPtrs[2, rank]
                        if tidx == self.tma_warp * 32:
                            for cc in cutlass.range_constexpr(rank + 1, R):
                                for b in cutlass.range_constexpr(self.NB):
                                    self._spin_ge_sys(
                                        ak_loc, (cc * self.HV + hv) * self.NB + b, ackv
                                    )
                        nb_epilog.arrive_and_wait()
                    elif warp_idx == self.mma_warp:
                        st_wk_c, st_pub_c, st_x_c, st_x_cb = self._mma_item(
                            nc,
                            tiled_mma1f,
                            tiled_mma2f,
                            sWb_t,
                            sKt_t,
                            sXf_t,
                            sxt_t,
                            tAcc1ps,
                            tXps,
                            p_wk,
                            p_g1s,
                            p_g2s,
                            p_pubs,
                            p_xs,
                            p_lam,
                            st_wk_c,
                            st_pub_c,
                            st_x_c,
                            st_x_cb,
                            st_one,
                        )
                        nb_epilog.arrive_and_wait()
                    else:
                        if cutlass.const_expr(K == 128):
                            if warp_idx >= self.lam_warp:
                                st_vg_c = self._lambda_item(
                                    tidx - self.lam_warp * 32,
                                    hv,
                                    T,
                                    c0,
                                    nc,
                                    sG_t,
                                    sLam_t,
                                    p_vg,
                                    p_lam,
                                    st_vg_c,
                                )
                                nb_epilog.arrive_and_wait()
                            else:
                                # One 4-warp group owns both strips of a pair.
                                # Groups at or above `n_groups_live` are the
                                # dead transition half: they keep their single
                                # NB_EPILOG arrival (the barrier counts all
                                # `num_threads`) and do nothing else.
                                for pp in cutlass.range_constexpr(self.n_groups_live, 2):
                                    if warp_idx >= 4 * pp and warp_idx < 4 * (pp + 1):
                                        nb_epilog.arrive_and_wait()
                                for pp in cutlass.range_constexpr(self.n_groups_live):
                                    if warp_idx >= 4 * pp and warp_idx < 4 * (pp + 1):
                                        st_vg_c, st_g1_c, st_g2_c, st_x_p = self._pair_item(
                                            pp,
                                            tidx % 128,
                                            hv,
                                            T,
                                            c0,
                                            nc,
                                            par,
                                            sI_t,
                                            mPtrs,
                                            tAcc1ps[pp],
                                            tXps[pp],
                                            sX_t,
                                            sPush_t,
                                            sxt_t,
                                            sV_t,
                                            sG_t,
                                            sGk_t,
                                            sLam_t,
                                            p_vg,
                                            p_g1s[pp],
                                            p_g2s[pp],
                                            p_pubs[pp],
                                            p_xs[pp],
                                            p_lam,
                                            st_vg_c,
                                            st_g1_c,
                                            st_g2_c,
                                            st_x_p,
                                            st_one,
                                        )
                        else:
                            # K=64: one existing strip group per MMA pair.
                            # See the K=128 branch for the dead-group arrival.
                            for s in cutlass.range_constexpr(self.n_groups_live, NG):
                                if warp_idx >= 4 * s and warp_idx < 4 * (s + 1):
                                    nb_epilog.arrive_and_wait()
                            for s in cutlass.range_constexpr(self.n_groups_live):
                                if warp_idx >= 4 * s and warp_idx < 4 * (s + 1):
                                    st_vg_c, st_g1_c, st_g2_c, st_x_p = self._wg_item(
                                        s,
                                        tidx % 128,
                                        hv,
                                        T,
                                        c0,
                                        nc,
                                        par,
                                        sI_t,
                                        mPtrs,
                                        tAcc1ps[s // self.SPP],
                                        tXts[s],
                                        sX_t,
                                        sPush_t,
                                        sxt_t,
                                        sV_t,
                                        sG_t,
                                        sGk_t,
                                        sLam_t,
                                        p_vg,
                                        p_g1s[s // self.SPP],
                                        p_g2s[s // self.SPP],
                                        p_pubs[s // self.SPP],
                                        p_xs[s // self.SPP],
                                        p_lam,
                                        st_vg_c,
                                        st_g1_c,
                                        st_g2_c,
                                        st_x_p,
                                        st_one,
                                    )
                    # item tail: flags for the pushed head
                    nb_epilog.arrive_and_wait()
                    cute.arch.fence_acq_rel_sys()
                    if tidx == 0:
                        self._flag_consumers(mPtrs, hv, stepu)

        if cutlass.const_expr(self.n_mitems > 0):
            wi = cutlass.Int32(bidx)
            while wi < self.n_mitems:
                # ---------------- merge item: (hv, b) -------------------
                # full-CTA fences around the item: its staging overlays the
                # compute smem, and the TMA warp of a preceding compute item
                # may still be prefetching into it
                nb_item.arrive_and_wait()
                # flat merge index over (hv, V-subtile): wi in [0, HV*NB)
                mb = wi % self.NB
                mhv = wi // self.NB
                if warp_idx <= 3:
                    self._merge_role(
                        tidx,
                        mb,
                        mhv,
                        par,
                        stepu,
                        mPtrs,
                        mH0,
                        pHMloc,
                        storage,
                        tiled_mma_sm80,
                        tiled_copy_M,
                        sM_layout,
                    )
                nb_item.arrive_and_wait()
                wi += self.n_ctas

        if cutlass.const_expr(has_compute):
            tmem.relinquish_alloc_permit()
            pipeline.NamedBarrier(
                barrier_id=NB_TMEM_FREE, num_threads=self.num_threads
            ).arrive_and_wait()
            tmem.free(tmem_ptr)

    # ----------------------------------------------------------- TMA warp
    @cute.jit
    def _tma_item(
        self,
        atom_w,
        tW,
        atom_kt,
        tKt,
        atom_v,
        tV,
        atom_g,
        tG,
        atom_gk,
        tGk,
        tiled_mma1,
        tiled_mma2,
        sW_t,
        sKt_t,
        sV_t,
        sG_t,
        sGk_t,
        T,
        c0,
        nc,
        hv,
        p_wk,
        p_vg,
        st_wk,
        st_vg,
    ):
        hk = hv // (self.HV // self.H)
        thr_mma1 = tiled_mma1.get_slice(0)
        thr_mma2 = tiled_mma2.get_slice(0)
        K = self.K

        # one bulk-tensor load: offset the gmem view to the (clamped)
        # chunk window, tile it through the MMA-A partition shape, and
        # issue the TMA into the pipeline stage (trace-time helper —
        # inlined, zero codegen difference vs the unrolled form)
        def tma_ld(atom, gt, off, tiler, thr, s_t, stage, bar):
            g = cute.local_tile(
                cute.domain_offset(off, gt), tiler, (None, None, None), proj=(1, None, 1)
            )
            tAs, tAg = cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(s_t, 0, 3),
                cute.group_modes(thr.partition_A(g), 0, 3),
            )
            cute.copy(atom, tAg[(None, 0, 0)], tAs[(None, stage)], tma_bar_ptr=bar)

        # gate sibling of tma_ld: the g/gk boxes are 2-mode (no MMA-A
        # partition), so they group 2 modes instead of 3 — otherwise the same
        # trace-time-inlined domain_offset -> local_tile -> tma_partition ->
        # copy sequence (byte-identical to the unrolled form)
        def gate_ld(atom, gt, off, tiler, s_t, stage, bar):
            g = cute.local_tile(cute.domain_offset(off, gt), tiler, (0, 0))
            tGs, tGg = cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(s_t[None, None, stage], 0, 2),
                cute.group_modes(g, 0, 2),
            )
            cute.copy(atom, tGg, tGs, tma_bar_ptr=bar)

        for tq in cutlass.range(nc, unroll=1):
            t0 = (c0 + tq) * BT
            t_ofs = cutlass.max(cutlass.min(t0, T - BT), 0)

            p_wk.producer_acquire(st_wk)
            bar_wk = p_wk.producer_get_barrier(st_wk)
            tma_ld(
                atom_w,
                tW[None, None, hv],
                (t_ofs, 0),
                (BT, NS, K),
                thr_mma1,
                sW_t,
                st_wk.index,
                bar_wk,
            )
            tma_ld(
                atom_kt,
                tKt[None, None, hk],
                (0, t_ofs),
                (K, NS, BT),
                thr_mma2,
                sKt_t,
                st_wk.index,
                bar_wk,
            )

            p_vg.producer_acquire(st_vg)
            bar_vg = p_vg.producer_get_barrier(st_vg)
            tma_ld(
                atom_v,
                tV[None, None, hv],
                (t_ofs, 0),
                (BT, NS, K),
                thr_mma1,
                sV_t,
                st_vg.index,
                bar_vg,
            )
            if cutlass.const_expr(self.use_g):
                gate_ld(atom_g, tG, (t_ofs, (hv // 4) * 4), (BT, 4), sG_t, st_vg.index, bar_vg)
            if cutlass.const_expr(self.use_gk):
                last_idx = t0 + cutlass.min(T - t0, BT) - 1
                gate_ld(
                    atom_gk,
                    tGk,
                    (last_idx * self.HV + hv, 0),
                    (1, self.K),
                    sGk_t,
                    st_vg.index,
                    bar_vg,
                )

            st_wk.advance()
            st_vg.advance()
        return st_wk, st_vg

    # ------------------------------------------------------ lambda helpers
    @cute.jit
    def _lambda_item(self, ltid, hv, T, c0, nc, sG_t, sLam_t, p_vg, p_lam, st_vg):
        """Publish one lambda row per helper thread for the K=128 path.

        The helpers wait on vg_full but do not release vg_empty. The pair
        warps release the stage only after pass 2 has consumed this table.
        """
        if cutlass.const_expr(self.need_lam):
            for tq in cutlass.range(nc, unroll=1):
                p_vg.consumer_wait(st_vg)
                t0 = (c0 + tq) * BT
                n_valid = T - t0
                shift = t0 - cutlass.max(cutlass.min(t0, T - BT), 0)
                g_last = cutlass.Float32(0.0)
                if cutlass.const_expr(self.use_g):
                    g_last = sG_t[cutlass.min(n_valid, BT) - 1 + shift, hv % 4, st_vg.index]
                idxl = ltid - shift
                lam = cutlass.Float32(0.0)
                if (idxl >= 0) & (idxl < n_valid):
                    lam = cutlass.Float32(1.0)
                    if cutlass.const_expr(self.use_g):
                        lam = cute.math.exp2(
                            g_last - sG_t[ltid, hv % 4, st_vg.index], fastmath=True
                        )
                sLam_t[ltid, st_vg.index] = lam
                p_lam.producer_commit(st_vg)
                st_vg.advance()
        return st_vg

    # ---------------------------------------------------------- pair groups
    @cute.jit
    def _pair_item(
        self,
        pair: cutlass.Constexpr,
        ltid,
        hv,
        T,
        c0,
        nc,
        par,
        sI_t,
        mPtrs,
        tAcc1,
        tX,
        sX_t,
        sPush_t,
        sxt_t,
        sV_t,
        sG_t,
        sGk_t,
        sLam_t,
        p_vg,
        p_g1,
        p_g2,
        p_pub,
        p_x,
        p_lam,
        st_vg,
        st_g1c,
        st_g2c,
        st_xp,
        st_one,
    ):
        """K=128 worker: one warpgroup owns both strips of an MMA pair."""
        K = self.K
        SPP = self.SPP
        dtype = self.io_dtype
        strip0 = 2 * pair
        is_h = pair == 0
        BTH = BT // 2

        # The pair's two X strips are adjacent TMEM columns. Thread ltid owns
        # one K row and drains/stores all 128 values with one x128 operation.
        x_2d = tX[((None, None), 0, 0)]
        atom_ld_x = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(2 * NS), tcgen05.Pack.NONE), cutlass.Float32
        )
        atom_st_x = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(2 * NS), tcgen05.Unpack.NONE), cutlass.Float32
        )
        t2r_x = tcgen05.make_tmem_copy(atom_ld_x, x_2d)
        r2t_x = tcgen05.make_tmem_copy(atom_st_x, x_2d)
        thr_x = t2r_x.get_slice(ltid)
        tXt = thr_x.partition_S(x_2d)
        tXst = r2t_x.get_slice(ltid).partition_D(x_2d)
        cX = thr_x.partition_D(cute.make_identity_tensor((K, 2 * NS)))
        rX = cute.make_rmem_tensor(cute.slice_(cX.shape, (None, 0, 0)), cutlass.Float32)

        # Full-pair acc1^T, split into two 32-token halves.
        acc1_2d = tAcc1[((None, None), 0, 0)]
        pan_acc = cute.make_layout((NS * SPP, BTH, 2), stride=(1, NS * SPP, NS * SPP * BTH))
        acc1_3h = cute.make_tensor(acc1_2d.iterator, cute.composition(acc1_2d.layout, pan_acc))
        acc1h = []
        for hh in cutlass.range_constexpr(2):
            acc1h.append(acc1_3h[(None, None, hh)])
        atom_ld_a1 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(4), tcgen05.Pack.NONE), cutlass.Float32
        )
        t2r_a1 = tcgen05.make_tmem_copy(atom_ld_a1, acc1h[0])
        thr_a1 = t2r_a1.get_slice(ltid)
        tA1tH = []
        for hh in cutlass.range_constexpr(2):
            tA1tH.append(thr_a1.partition_S(acc1h[hh]))
        cA1 = thr_a1.partition_D(cute.make_identity_tensor((NS * SPP, BTH)))

        # Pair x^T publication tile.
        sxt1 = sxt_t[(None, None, None, pair)]
        pan_xt = cute.make_layout(
            (NS * SPP, (16, BT // 32), 2),
            stride=(1, (NS * SPP, 16 * NS * SPP), (BT // 2) * NS * SPP),
        )
        sxt3 = cute.make_tensor(sxt1.iterator, cute.composition(sxt1.layout, pan_xt))
        st_atom_a = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR, dtype, cutlass.Float32, t2r_a1
        )
        r2s_a = cute.make_tiled_copy_D(st_atom_a, t2r_a1)
        tRS_sxH = []
        for hh in cutlass.range_constexpr(2):
            tRS_sxH.append(r2s_a.get_slice(ltid).partition_D(sxt3[(None, None, hh)]))

        # V^T uses the same full-pair fragment mapping as acc1^T.
        pan_vt = cute.make_layout((self.V, BTH, 2, VGS), stride=(BT, 1, BTH, BT * K))
        sVt4 = cute.make_tensor(sV_t.iterator, cute.composition(sV_t.layout, pan_vt))
        ld_atom_v = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), dtype
        )
        s2r_v = cute.make_tiled_copy_D(ld_atom_v, t2r_a1)
        thr_v = s2r_v.get_slice(ltid)

        rA1a = cute.make_rmem_tensor(cA1.shape, cutlass.Float32)
        rA1b = cute.make_rmem_tensor(cA1.shape, cutlass.Float32)
        tVsm0 = thr_v.partition_S(sVt4[(None, None, 0, 0)])
        rVsA = cute.make_rmem_tensor(tVsm0.shape, dtype)
        rVsB = cute.make_rmem_tensor(tVsm0.shape, dtype)
        if cutlass.const_expr(self.need_lam):
            rLam = cute.make_rmem_tensor(32, cutlass.Float32)
        rxc = cute.make_rmem_tensor(tRS_sxH[0].shape, dtype)
        sub_a1 = cute.size(rA1a)
        lam_half = sub_a1 // 2

        # Build four publish destinations: two strips x two column halves.
        atom_pub = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)
        pubX = cute.make_tiled_copy_tv(
            atom_pub, cute.make_layout((K, 1)), cute.make_layout((1, NS // 2))
        )
        pan_xh = cute.make_layout(((16, K // 16), NS // 2, 2), stride=((NS, 16 * NS), 1, NS // 2))
        tPubX = []
        for ss in cutlass.range_constexpr(2):
            sX1 = sX_t[(None, None, None, strip0 + ss)]
            sX3h = cute.make_tensor(sX1.iterator, cute.composition(sX1.layout, pan_xh))
            halves = []
            for hh in cutlass.range_constexpr(2):
                halves.append(pubX.get_slice(ltid).partition_D(sX3h[(None, None, hh)]))
            tPubX.append(halves)
        rXcv = cute.make_fragment_like(tPubX[0][0], dtype)

        atom_f32x4 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
        )

        # X(0): pair 0 is h=0; pair 1 owns the two diagonal M blocks.
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
            p_g2.consumer_wait(st_g2c)
            st_g2c.advance()

            # pass 1a: one x128 TMEM drain, then publish both strips.
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

            # V/gate stage is independent of the W/K stage and normally
            # becomes ready while gemm1 consumes the just-published X.
            p_vg.consumer_wait(st_vg)

            t0 = (c0 + tq) * BT
            n_valid = T - t0
            shift = t0 - cutlass.max(cutlass.min(t0, T - BT), 0)
            g_last = cutlass.Float32(0.0)
            if cutlass.const_expr(self.use_g):
                g_last = sG_t[cutlass.min(n_valid, BT) - 1 + shift, hv % 4, st_vg.index]
            gamma = cutlass.Float32(1.0)
            if cutlass.const_expr(self.use_g):
                gamma = cute.math.exp2(g_last, fastmath=True)

            # pass 1b: decay both strips while they are resident in rX.
            if cutlass.const_expr(self.need_decay):
                if cutlass.const_expr(self.use_gk):
                    gscale = cute.math.exp2(sGk_t[0, row0, st_vg.index], fastmath=True)
                else:
                    gscale = gamma
                for i in cutlass.range_constexpr(2 * NS):
                    rX[i] = gscale * rX[i]
                cute.copy(r2t_x, rX, tXst[(None, 0, 0)])

            p_g1.consumer_wait(st_g1c)
            if cutlass.const_expr(self.need_lam and pair == 0):
                p_lam.consumer_wait(st_vg)

            # pass 2, half 0. Full-pair fragments remove the former `gip`
            # selection; one arrival now publishes all 128 x rows.
            cute.copy(t2r_a1, tA1tH[0], rA1a)
            if cutlass.const_expr(is_h):
                tVsmA = thr_v.partition_S(sVt4[(None, None, 0, st_vg.index)])
                cute.copy(s2r_v, tVsmA, rVsA)
            if cutlass.const_expr(self.need_lam):
                for j in cutlass.range_constexpr(sub_a1 // 4):
                    rLam[2 * j] = sLam_t[cA1[4 * j][1], st_vg.index]
                    rLam[2 * j + 1] = sLam_t[cA1[4 * j + 1][1], st_vg.index]
            cute.arch.fence_view_async_tmem_load()

            cute.copy(t2r_a1, tA1tH[1], rA1b)
            if cutlass.const_expr(is_h):
                tVsmB = thr_v.partition_S(sVt4[(None, None, 1, st_vg.index)])
                cute.copy(s2r_v, tVsmB, rVsB)
            if cutlass.const_expr(self.need_lam):
                for j in cutlass.range_constexpr(sub_a1 // 4):
                    rLam[lam_half + 2 * j] = sLam_t[BTH + cA1[4 * j][1], st_vg.index]
                    rLam[lam_half + 2 * j + 1] = sLam_t[BTH + cA1[4 * j + 1][1], st_vg.index]

            for j in cutlass.range_constexpr(sub_a1 // 4):
                for q in cutlass.range_constexpr(4):
                    i = 4 * j + q
                    xv = cutlass.Float32(0.0)
                    if cutlass.const_expr(is_h):
                        xv = rVsA[i].to(cutlass.Float32)
                    xv = xv - rA1a[i]
                    if cutlass.const_expr(self.need_lam):
                        xv = xv * rLam[2 * j + (q & 1)]
                    rxc[i] = xv.to(dtype)
            cute.copy(r2s_a, rxc, tRS_sxH[0])
            if cutlass.const_expr(self.need_decay):
                cute.arch.fence_view_async_tmem_store()
            cute.arch.fence_proxy("async.shared", space="cta")
            p_x.producer_commit(st_xp)
            st_xp.advance()

            cute.arch.fence_view_async_tmem_load()
            for j in cutlass.range_constexpr(sub_a1 // 4):
                for q in cutlass.range_constexpr(4):
                    i = 4 * j + q
                    xv = cutlass.Float32(0.0)
                    if cutlass.const_expr(is_h):
                        xv = rVsB[i].to(cutlass.Float32)
                    xv = xv - rA1b[i]
                    if cutlass.const_expr(self.need_lam):
                        xv = xv * rLam[lam_half + 2 * j + (q & 1)]
                    rxc[i] = xv.to(dtype)
            cute.copy(r2s_a, rxc, tRS_sxH[1])
            cute.arch.fence_proxy("async.shared", space="cta")
            p_x.producer_commit(st_xp)
            st_xp.advance()

            p_vg.consumer_release(st_vg)
            st_vg.advance()
            st_g1c.advance()

        # Pair-side push: drain once, then emit the two 64-column strips.
        p_g2.consumer_wait(st_g2c)
        st_g2c.advance()
        pipeline.NamedBarrier(barrier_id=NB_EPILOG, num_threads=self.num_threads).arrive_and_wait()

        # Rank 0's transition half is dead wire.  A consumer folds M_j only for
        # j >= 1 (`_merge_role` iterates `range_constexpr(1, rank)`): nothing
        # precedes rank 0, so its M has no earlier state to act on and no rank
        # can ever read it.  Pair 1 owns exactly those strips, so at rank 0 its
        # whole epilogue -- the TMEM read-out and NK*STRIP fp32 of peer writes,
        # 8 MB at HV=K=V=128 -- is skipped.  At R == 2 rank 0 is the only
        # forward producer, which halves the wire outright.  The barrier above
        # has already been arrived at, so the arrival counts are untouched, and
        # `_flag_consumers` still signals every strip (a consumer that reads M
        # is by construction reading a rank that still pushes it).
        if cutlass.const_expr(not (is_h or self.push_transition)):
            return st_vg, st_g1c, st_g2c, st_xp

        cute.copy(t2r_x, tXt[(None, 0, 0)], rX)
        cute.arch.fence_view_async_tmem_load()
        # Emit mask (see `CuteDSLFusedCPPreProcessWS.launch_validated`): a rank
        # whose summary must not reach its consumers still has to push and flag
        # -- they spin on those flags -- so the half is zeroed here instead of
        # skipped.  Zeroed, not scaled by 0.0: the value being suppressed is a
        # scan over tokens this rank's consumers do not belong to, and 0.0 * inf
        # is NaN.
        if (sI_t[7] & (1 if is_h else 2)) == 0:
            for i in cutlass.range_constexpr(2 * NS):
                rX[i] = cutlass.Float32(0.0)
        lane = ltid & 31
        warp_row0 = row0 - lane

        # Coalesced peer push: stage one 32-column half at a time through the
        # warp-private 32x32 fp32 tile with a vector-XOR row layout, then have
        # each lane write a float4 from eight rows, so every STG instruction
        # covers four adjacent rows and fills all 32B sectors.
        for ss in cutlass.range_constexpr(2):
            strip = strip0 + ss
            xbase = ss * NS
            off0 = self._hm_off(par, self.rank, hv, strip)

            warp_slot = (4 * strip + ltid // 32) * (32 * 32)
            for half in cutlass.range_constexpr(2):
                for v4 in cutlass.range_constexpr(8):
                    src4 = cute.make_tensor(
                        (rX.iterator + xbase + 32 * half + 4 * v4).align(16), cute.make_layout(4)
                    )
                    phys = lane * 32 + 4 * (v4 ^ (lane & 7))
                    dst4 = cute.make_tensor(
                        (sPush_t.iterator + warp_slot + phys).align(16), cute.make_layout(4)
                    )
                    cute.copy(atom_f32x4, src4, dst4)
                cute.arch.sync_warp()
                for q in cutlass.range_constexpr(8):
                    lr = 4 * q + lane // 8
                    v4 = lane & 7
                    phys = lr * 32 + 4 * (v4 ^ (lr & 7))
                    src4 = cute.make_tensor(
                        (sPush_t.iterator + warp_slot + phys).align(16), cute.make_layout(4)
                    )
                    col = 32 * half + 4 * v4
                    for j in cutlass.range_constexpr(self.rank + 1, self.R):
                        dst4 = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32,
                                mPtrs[0, j] + 4 * (off0 + (warp_row0 + lr) * NS + col),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(4),
                        )
                        cute.copy(atom_f32x4, src4, dst4)
                cute.arch.sync_warp()

        return st_vg, st_g1c, st_g2c, st_xp

    # -------------------------------------------------------- strip groups
    @cute.jit
    def _wg_item(
        self,
        strip: cutlass.Constexpr,
        ltid,
        hv,
        T,
        c0,
        nc,
        par,
        sI_t,
        mPtrs,
        tAcc1,
        tX,
        sX_t,
        sPush_t,
        sxt_t,
        sV_t,
        sG_t,
        sGk_t,
        sLam_t,
        p_vg,
        p_g1,
        p_g2,
        p_pub,
        p_x,
        p_lam,
        st_vg,
        st_g1c,
        st_g2c,
        st_xp,
        st_one,
    ):
        K = self.K
        SPP = self.SPP
        dtype = self.io_dtype
        is_h = strip < self.NV
        cc0 = 0 if is_h else NS * (strip - self.NV)
        pair = strip // SPP  # which MMA pair this strip belongs to
        gip = strip % SPP  # group-in-pair: which acc1^T M-subtile

        # tAcc1 is the PAIR's acc1^T [NS*SPP, BT] (flip: strips are M-rows,
        # tokens are columns); this group owns M-subtile `gip`. Pass 2 is
        # TOKEN-HALF split (BTH columns at a time): half 0's publish lets
        # the MMA warp start gemm2's first k-phases under half 1's work.
        BTH = BT // 2
        x_2d = tX[((None, None), 0, 0)]
        atom_ld = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), cutlass.Float32
        )
        atom_ld_h = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(4), tcgen05.Pack.NONE), cutlass.Float32
        )
        atom_st = cute.make_copy_atom(
            tcgen05.St16x256bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE), cutlass.Float32
        )
        # acc1^T token-half views: composed through the pair fragment's
        # 2-D layout so the M=64 interleaved lane mapping (K=64: rows
        # 16-31 live at lanes 32-47) survives — a flat (rows):(65536)
        # hardcode silently reads the wrong lanes there
        acc1_2d = tAcc1[((None, None), 0, 0)]
        pan_acc = cute.make_layout((NS * SPP, BTH, 2), stride=(1, NS * SPP, NS * SPP * BTH))
        acc1_3h = cute.make_tensor(acc1_2d.iterator, cute.composition(acc1_2d.layout, pan_acc))
        acc1h = []
        for hh in cutlass.range_constexpr(2):
            acc1h.append(acc1_3h[(None, None, hh)])
        t2r_a1 = tcgen05.make_tmem_copy(atom_ld_h, acc1h[0])
        t2r_x = tcgen05.make_tmem_copy(atom_ld, x_2d)
        r2t_x = tcgen05.make_tmem_copy(atom_st, x_2d)
        thr_a1 = t2r_a1.get_slice(ltid)
        thr_x = t2r_x.get_slice(ltid)
        thr_xs = r2t_x.get_slice(ltid)
        tA1tH = []
        for hh in cutlass.range_constexpr(2):
            tA1tH.append(thr_a1.partition_S(acc1h[hh]))
        tXt = thr_x.partition_S(x_2d)
        tXst = thr_xs.partition_D(x_2d)
        cA1 = thr_a1.partition_D(cute.make_identity_tensor((NS * SPP, BTH)))
        cX = thr_x.partition_D(cute.make_identity_tensor((K, NS)))

        sX1 = sX_t[(None, None, None, strip)]
        sxt1 = sxt_t[(None, None, None, pair)]

        # 2-D logical views of the smem publish tiles (swizzle preserved)
        # so the bf16 publishes go through stmatrix instead of scalar
        # stores. sX: (K rows, NS) 16-row panels, N contiguous. sxt: the
        # pair's x^T k-major B tile — (strip-row, token) with tokens
        # contiguous (flat (NS*SPP, BT):(BT, 1) under the swizzle).
        pan_x = cute.make_layout(((16, K // 16), NS), stride=((NS, 16 * NS), 1))
        pan_xt = cute.make_layout(
            (NS * SPP, (16, BT // 32), 2),
            stride=(1, (NS * SPP, 16 * NS * SPP), (BT // 2) * NS * SPP),
        )
        sX2d = cute.make_tensor(sX1.iterator, cute.composition(sX1.layout, pan_x))
        sxt3 = cute.make_tensor(sxt1.iterator, cute.composition(sxt1.layout, pan_xt))

        # transposed logical (V-col, token, stage) view of the swizzled
        # sV (in the canonical A layout's flat domain the token mode has
        # stride 1 and the k-submodes coalesce to vcol*BT). Pass 2 reads
        # V^T fragments via ldmatrix.trans paired to the acc1^T tmem
        # load: 16B lane chunks run along the physically contiguous
        # vcol dim while registers receive token-major values in the
        # acc fragment order — per-element LDS here cost ~30us (32
        # scattered 2B loads/thread, 128B token stride).
        pan_vt = cute.make_layout((self.V, BTH, 2, VGS), stride=(BT, 1, BTH, BT * K))
        sVt4 = cute.make_tensor(sV_t.iterator, cute.composition(sV_t.layout, pan_vt))
        ld_atom_v = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), dtype
        )
        s2r_v = cute.make_tiled_copy_D(ld_atom_v, t2r_a1)
        thr_v = s2r_v.get_slice(ltid)

        rA1a = cute.make_rmem_tensor(cute.slice_(cA1.shape, (None, 0, 0)), cutlass.Float32)
        rA1b = cute.make_rmem_tensor(cute.slice_(cA1.shape, (None, 0, 0)), cutlass.Float32)
        tVsm0 = thr_v.partition_S(sVt4[(None, None, 0, 0)])
        rVsA = cute.make_rmem_tensor(cute.slice_(tVsm0.shape, (None, 0, 0)), dtype)
        rVsB = cute.make_rmem_tensor(cute.slice_(tVsm0.shape, (None, 0, 0)), dtype)
        if cutlass.const_expr(self.need_lam):
            rLam = cute.make_rmem_tensor(16, cutlass.Float32)
        rXs = cute.make_rmem_tensor(cute.slice_(cX.shape, (None, 0, 0)), cutlass.Float32)
        n_x_m = cute.size(cX, mode=[1])
        n_x_n = cute.size(cX, mode=[2])
        sub_a1 = cute.size(rA1a)
        sub_x = cute.size(rXs)

        # stmatrix store path paired to the tmem loads (same thread-value
        # order, so flat register index i lines up across t2r and r2s)
        st_atom_x = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR, dtype, cutlass.Float32, t2r_x
        )
        st_atom_a = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR, dtype, cutlass.Float32, t2r_a1
        )
        r2s_x = cute.make_tiled_copy_D(st_atom_x, t2r_x)
        r2s_a = cute.make_tiled_copy_D(st_atom_a, t2r_a1)
        tRS_sX = r2s_x.get_slice(ltid).partition_D(sX2d)
        # ---- K=128: row-per-thread X path (Ld32x32b / St32x32b): thread
        # ltid owns K-row ltid, all NS cols. The bf16 publish becomes a
        # plain coalesced 128b STS tiled copy (no stmatrix) and the
        # epilogue push becomes coalesced 128b STG. (K=64's M=64 state
        # fragment has the 16-lanes-per-quarter interleave, which
        # Ld32x32b cannot address — it keeps the 16x256b path.)
        if cutlass.const_expr(K == 128):
            atom_ld32 = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(NS), tcgen05.Pack.NONE), cutlass.Float32
            )
            atom_st32 = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(NS), tcgen05.Unpack.NONE), cutlass.Float32
            )
            atom_f32x4 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
            )
            t2r_x32 = tcgen05.make_tmem_copy(atom_ld32, x_2d)
            r2t_x32 = tcgen05.make_tmem_copy(atom_st32, x_2d)
            thr_x32 = t2r_x32.get_slice(ltid)
            tXt32 = thr_x32.partition_S(x_2d)
            tXst32 = r2t_x32.get_slice(ltid).partition_D(x_2d)
            cX32 = thr_x32.partition_D(cute.make_identity_tensor((K, NS)))
            rX = cute.make_rmem_tensor(cute.slice_(cX32.shape, (None, 0, 0)), cutlass.Float32)
            # (splitting this drain into two Rep(32) pipelined halves
            # measured +6.4 us — PROGRESS #42; keep the monolithic op)
            # publish copy: thread t -> row t (matches the 32x32b lane
            # mapping), value v -> col; done in two 32-col halves to
            # keep the bf16 staging at 16 regs
            atom_pub = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128
            )
            pubX = cute.make_tiled_copy_tv(
                atom_pub, cute.make_layout((K, 1)), cute.make_layout((1, NS // 2))
            )
            pan_xh = cute.make_layout(
                ((16, K // 16), NS // 2, 2), stride=((NS, 16 * NS), 1, NS // 2)
            )
            sX3h = cute.make_tensor(sX1.iterator, cute.composition(sX1.layout, pan_xh))
            tPubXH = []
            for hh in cutlass.range_constexpr(2):
                tPubXH.append(pubX.get_slice(ltid).partition_D(sX3h[(None, None, hh)]))
            rXcv = cute.make_fragment_like(tPubXH[0], dtype)
        tRS_sxH = []
        for hh in cutlass.range_constexpr(2):
            tRS_sxH.append(r2s_a.get_slice(ltid).partition_D(sxt3[(None, None, hh)]))
        # per-subtile bf16 staging (8 regs) instead of whole-tile buffers:
        # a full-tile rXc (32 regs) live on top of rXs+rXb was part of the
        # 96-reg-cap spill peak
        rXc = cute.make_rmem_tensor(cute.slice_(tRS_sX.shape, (None, 0, 0)), dtype)
        rxc = cute.make_rmem_tensor(cute.slice_(tRS_sxH[0].shape, (None, 0, 0)), dtype)

        # ---- init X(0) in TMEM: 0 (h) / identity block (M)
        # (previous item's gemm2 completion was consumed in-thread)
        if cutlass.const_expr(K == 128):
            row0 = cX32[0][0]
            for i in cutlass.range_constexpr(NS):
                val = cutlass.Float32(0.0)
                if cutlass.const_expr(not is_h):
                    if row0 == cc0 + cX32[i][1]:
                        val = cutlass.Float32(1.0)
                rX[i] = val
            cute.copy(r2t_x32, rX, tXst32[(None, 0, 0)])
        else:
            for sm in cutlass.range_constexpr(n_x_m):
                for sn in cutlass.range_constexpr(n_x_n):
                    base = sub_x * (sm + n_x_m * sn)
                    for i in cutlass.range_constexpr(sub_x):
                        crd = cX[base + i]
                        val = cutlass.Float32(0.0)
                        if cutlass.const_expr(not is_h):
                            if crd[0] == cc0 + crd[1]:
                                val = cutlass.Float32(1.0)
                        rXs[i] = val
                    cute.copy(r2t_x, rXs, tXst[(None, sm, sn)])
        cute.arch.fence_view_async_tmem_store()

        base = sub_a1 * gip
        for tq in cutlass.range(nc, unroll=1):
            # ---- loop order (v6): pass 1a runs FIRST (right after the
            # g2 wait) — X(t) is its only input, so the V/gate wait and
            # gate math hide under gemm1's flight instead of sitting on
            # the serial X -> gemm1 critical path (the v5 order cost 553
            # cyc/chunk of dead pre-g2 loop head). λ-table ordering is
            # the v8 scheme — see the p_lam creation comment and the
            # λ block below. t0/n_valid/shift are computed in the
            # vg block, not here: keeping them live across pass 1a is
            # exactly the extra state the 96-reg cliff punishes (the
            # per-thread-lambda variant of this reorder measured pass 2
            # at +500..+900 cyc from the spills it caused).
            # gemm2(t-1) done -> X(t) final (t=0: the pre-loop dummy commit)
            p_g2.consumer_wait(st_g2c)
            st_g2c.advance()

            # ---- pass 1a: publish X_bf16 (gemm1 operand). K=128:
            # row-per-thread Ld32x32b + coalesced 128b STS in two
            # 32-col halves; K=64 keeps the 16x256b+stmatrix path.
            if cutlass.const_expr(K == 128):
                cute.copy(t2r_x32, tXt32[(None, 0, 0)], rX)
                cute.arch.fence_view_async_tmem_load()
                for hh in cutlass.range_constexpr(2):
                    for i in cutlass.range_constexpr(NS // 2):
                        rXcv[i] = rX[hh * (NS // 2) + i].to(dtype)
                    cute.copy(pubX, rXcv, tPubXH[hh])
            else:
                cute.copy(t2r_x, tXt[(None, 0, 0)], rXs)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(sub_x):
                    rXc[i] = rXs[i].to(dtype)
                cute.copy(r2s_x, rXc, tRS_sX[(None, 0, 0)])
            cute.arch.fence_proxy("async.shared", space="cta")
            if cutlass.const_expr(K == 128):
                # per-warp quarter arrive: this warp published X rows
                # [32w, 32w+32) of its strip
                wig = ltid // 32
                for w in cutlass.range_constexpr(4):
                    if wig == w:
                        p_pub[w].producer_commit(st_one)
            else:
                # per-thread arrive = "all strips' X_bf16 published"
                p_pub[0].producer_commit(st_one)

            # ---- V/gate wait + gate math (v6: under gemm1's flight).
            # gates are log2-domain chunk-local cumsums: g_last = g[last
            # valid row]; gamma = 2^g_last is the whole-chunk decay applied
            # to the carried state X in pass 1b (ragged tail: the TMA
            # window is clamped, `shift` re-indexes so g_last still lands
            # on token T-1)
            p_vg.consumer_wait(st_vg)
            t0 = (c0 + tq) * BT
            n_valid = T - t0
            shift = t0 - cutlass.max(cutlass.min(t0, T - BT), 0)
            g_last = cutlass.Float32(0.0)
            if cutlass.const_expr(self.use_g):
                g_last = sG_t[cutlass.min(n_valid, BT) - 1 + shift, hv % 4, st_vg.index]
            gamma = cutlass.Float32(1.0)
            if cutlass.const_expr(self.use_g):
                gamma = cute.math.exp2(g_last, fastmath=True)

            # ---- lambda table: lambda_i = 2^(g_last - g_i) per token row
            # (0 outside the valid window). Written by the first 64
            # threads of strip 1 (K=128; strip 0 for K=64) — at K=128,
            # s1's pass 1b ends ~540 cyc before s0's, so the λ work rides
            # its slack instead of overrunning the g1 wake by ~240 cyc on
            # the pair-0 serial chain. Reader ordering: see the p_lam
            # creation comment (readers' waits sit after their g1 wake /
            # before the g1 pair-1 commit).
            if cutlass.const_expr(self.need_lam):
                if cutlass.const_expr(strip == (1 if K == 128 else 0)):
                    if ltid < BT:
                        idxl = ltid - shift
                        lam = cutlass.Float32(0.0)
                        if (idxl >= 0) & (idxl < n_valid):
                            lam = cutlass.Float32(1.0)
                            if cutlass.const_expr(self.use_g):
                                lam = cute.math.exp2(
                                    g_last - sG_t[ltid, hv % 4, st_vg.index], fastmath=True
                                )
                        sLam_t[ltid, st_vg.index] = lam
                        p_lam.producer_commit(st_vg)

            # ---- pass 1b (hidden under gemm1's flight): X <- gamma * X
            # in TMEM (the decay half of X <- gamma*X + K_t^T @ x_t; gemm2
            # adds the second term). gemm2 needs it only after the pass-2
            # barrier. gk mode: gamma is per-K-row (2^gk[row]); the decay
            # pair is per SUBTILE (each subtile covers two different K
            # rows — rows from cX, elements paired (i>>1)&1)
            if cutlass.const_expr(self.need_decay):
                if cutlass.const_expr(K == 128):
                    if cutlass.const_expr(self.use_gk):
                        gker = cute.math.exp2(sGk_t[0, cX32[0][0], st_vg.index], fastmath=True)
                        for i in cutlass.range_constexpr(NS):
                            rX[i] = gker * rX[i]
                    else:
                        for i in cutlass.range_constexpr(NS):
                            rX[i] = gamma * rX[i]
                    cute.copy(r2t_x32, rX, tXst32[(None, 0, 0)])
                else:
                    for sm in cutlass.range_constexpr(n_x_m):
                        for sn in cutlass.range_constexpr(n_x_n):
                            rr = rXs
                            base = sub_x * (sm + n_x_m * sn)
                            if cutlass.const_expr(self.use_gk):
                                gke_pair = [
                                    cute.math.exp2(
                                        sGk_t[0, cX[base + 0][0], st_vg.index], fastmath=True
                                    ),
                                    cute.math.exp2(
                                        sGk_t[0, cX[base + 2][0], st_vg.index], fastmath=True
                                    ),
                                ]
                                for i in cutlass.range_constexpr(sub_x):
                                    rr[i] = gke_pair[(i >> 1) & 1] * rr[i]
                            else:
                                for i in cutlass.range_constexpr(sub_x):
                                    rr[i] = gamma * rr[i]
                            cute.copy(r2t_x, rr, tXst[(None, sm, sn)])
                if cutlass.const_expr(K != 128):
                    cute.arch.fence_view_async_tmem_store()

            p_g1.consumer_wait(st_g1c)  # gemm1f done -> acc1^T readable
            # v8 (K=128): pair-0 λ visibility (strip-1 written, commits
            # ~700-900; this point is ~1100+ — always satisfied, ~20 cyc
            # check). Pair-1 readers ride the MMA warp's p_lam wait
            # before the g1 pair-1 commit. Register pressure here
            # (post-pass-1b) tolerates the extra basic block; in the vg
            # block this wait spilled one value per thread, before the
            # g1 wait it measured +2 us, and applying it to pair 1 too
            # (dropping the MMA-side wait) measured +3.7 us. K=64 keeps
            # the v6 scheme: strip-0 writer + MMA wait before the g1
            # pair-0 commit — no direct wait at all.
            if cutlass.const_expr(self.need_lam and K == 128 and strip < SPP):
                p_lam.consumer_wait(st_vg)

            # ---- pass 2 (flip): x^T = lam ⊙ ([v|0]^T - acc1^T) -> bf16
            # -> smem (k-major B of gemm2). This group handles M-subtile
            # `gip` of the pair's acc1^T: fragment rows are strip
            # columns (pair-local == global v columns for the h pair),
            # fragment cols are tokens. lambda comes from the shared
            # per-token table; V reads issue between the acc1 tmem-ld
            # and its fence so their smem latency hides under the
            # ~400cyc TMEM-load wait (fixed sV column, varying token row
            # — conflict-free through the SW128 swizzle).
            # half 0: TMEM ld issues first; its V^T (ldmatrix.trans) and
            # lambda quad-pairs load in the TMEM shadow. Fragment quads
            # are (r0:c, r0:c+1, r1:c, r1:c+1) — same column pair for
            # both rows — so 2 lambda loads cover 4 elements (the
            # bit-exact gate guards this structural assumption).
            cute.copy(t2r_a1, tA1tH[0][(None, gip, 0)], rA1a)
            if cutlass.const_expr(is_h):
                tVsmA = thr_v.partition_S(sVt4[(None, None, 0, st_vg.index)])
                cute.copy(s2r_v, tVsmA[(None, gip, 0)], rVsA)
            if cutlass.const_expr(self.need_lam):
                for j in cutlass.range_constexpr(sub_a1 // 4):
                    rLam[2 * j] = sLam_t[cA1[base + 4 * j][1], st_vg.index]
                    rLam[2 * j + 1] = sLam_t[cA1[base + 4 * j + 1][1], st_vg.index]
            cute.arch.fence_view_async_tmem_load()
            # half 1's TMEM ld + V/lambda fly under half 0's math+publish
            cute.copy(t2r_a1, tA1tH[1][(None, gip, 0)], rA1b)
            if cutlass.const_expr(is_h):
                tVsmB = thr_v.partition_S(sVt4[(None, None, 1, st_vg.index)])
                cute.copy(s2r_v, tVsmB[(None, gip, 0)], rVsB)
            if cutlass.const_expr(self.need_lam):
                for j in cutlass.range_constexpr(sub_a1 // 4):
                    rLam[8 + 2 * j] = sLam_t[BTH + cA1[base + 4 * j][1], st_vg.index]
                    rLam[8 + 2 * j + 1] = sLam_t[BTH + cA1[base + 4 * j + 1][1], st_vg.index]
            for j in cutlass.range_constexpr(sub_a1 // 4):
                for q in cutlass.range_constexpr(4):
                    i = 4 * j + q
                    xv = cutlass.Float32(0.0)
                    if cutlass.const_expr(is_h):
                        xv = rVsA[i].to(cutlass.Float32)
                    xv = xv - rA1a[i]
                    if cutlass.const_expr(self.need_lam):
                        xv = xv * rLam[2 * j + (q & 1)]
                    rxc[i] = xv.to(dtype)
            cute.copy(r2s_a, rxc, tRS_sxH[0][(None, gip, 0)])
            if cutlass.const_expr(self.need_decay and K == 128):
                # the pass-1b decay store must be visible before gemm2
                # accumulates onto X — this edge (first px arrive) is the
                # latest legal point, so the store drains under the
                # g1-wait + acc1 load + half-0 math instead of blocking
                cute.arch.fence_view_async_tmem_store()
            cute.arch.fence_proxy("async.shared", space="cta")
            # half-0 arrive: the MMA warp starts gemm2's first k-phases
            # while this group works half 1
            p_x.producer_commit(st_xp)
            st_xp.advance()
            cute.arch.fence_view_async_tmem_load()
            for j in cutlass.range_constexpr(sub_a1 // 4):
                for q in cutlass.range_constexpr(4):
                    i = 4 * j + q
                    xv = cutlass.Float32(0.0)
                    if cutlass.const_expr(is_h):
                        xv = rVsB[i].to(cutlass.Float32)
                    xv = xv - rA1b[i]
                    if cutlass.const_expr(self.need_lam):
                        xv = xv * rLam[8 + 2 * j + (q & 1)]
                    rxc[i] = xv.to(dtype)
            cute.copy(r2s_a, rxc, tRS_sxH[1][(None, gip, 0)])
            cute.arch.fence_proxy("async.shared", space="cta")
            p_x.producer_commit(st_xp)
            st_xp.advance()
            p_vg.consumer_release(st_vg)

            st_vg.advance()
            st_g1c.advance()

        # ---- push epilogue: rendezvous with the ack throttle, read the
        # final state out of TMEM, store to every consumer's hm strip slot
        p_g2.consumer_wait(st_g2c)  # last gemm2 -> final X readable
        st_g2c.advance()
        pipeline.NamedBarrier(barrier_id=NB_EPILOG, num_threads=self.num_threads).arrive_and_wait()
        # See `_pair_item`: rank 0's transition strips are read by nobody.
        if cutlass.const_expr(not (is_h or self.push_transition)):
            return st_vg, st_g1c, st_g2c, st_xp
        if cutlass.const_expr(K == 128):
            cute.copy(t2r_x32, tXt32[(None, 0, 0)], rX)
            cute.arch.fence_view_async_tmem_load()
            # See `_pair_item`: suppressed halves are zeroed, not skipped.
            if (sI_t[7] & (1 if is_h else 2)) == 0:
                for i in cutlass.range_constexpr(cute.size(rX)):
                    rX[i] = cutlass.Float32(0.0)
            off0 = self._hm_off(par, self.rank, hv, strip)
            # ltid owns one complete state row.  Stage one 32-column
            # half at a time with a vector-XOR row layout, then have
            # each lane write a float4 from eight rows.  Across the
            # warp each STG instruction therefore covers four
            # adjacent rows and fills all 32B sectors.
            lane = ltid & 31
            warp_slot = (4 * strip + ltid // 32) * (32 * 32)
            warp_row0 = cX32[0][0] - lane
            for half in cutlass.range_constexpr(2):
                for v4 in cutlass.range_constexpr(8):
                    src4 = cute.make_tensor(
                        (rX.iterator + 32 * half + 4 * v4).align(16), cute.make_layout(4)
                    )
                    phys = lane * 32 + 4 * (v4 ^ (lane & 7))
                    dst4 = cute.make_tensor(
                        (sPush_t.iterator + warp_slot + phys).align(16), cute.make_layout(4)
                    )
                    cute.copy(atom_f32x4, src4, dst4)
                cute.arch.sync_warp()
                for q in cutlass.range_constexpr(8):
                    lr = 4 * q + lane // 8
                    v4 = lane & 7
                    phys = lr * 32 + 4 * (v4 ^ (lr & 7))
                    src4 = cute.make_tensor(
                        (sPush_t.iterator + warp_slot + phys).align(16), cute.make_layout(4)
                    )
                    col = 32 * half + 4 * v4
                    for j in cutlass.range_constexpr(self.rank + 1, self.R):
                        dst4 = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32,
                                mPtrs[0, j] + 4 * (off0 + (warp_row0 + lr) * 64 + col),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(4),
                        )
                        cute.copy(atom_f32x4, src4, dst4)
                cute.arch.sync_warp()
        else:
            for sm in cutlass.range_constexpr(n_x_m):
                for sn in cutlass.range_constexpr(n_x_n):
                    cute.copy(t2r_x, tXt[(None, sm, sn)], rXs)
                    cute.arch.fence_view_async_tmem_load()
                    # See `_pair_item`: suppressed halves are zeroed, not
                    # skipped.
                    if (sI_t[7] & (1 if is_h else 2)) == 0:
                        for i in cutlass.range_constexpr(sub_x):
                            rXs[i] = cutlass.Float32(0.0)
                    base = sub_x * (sm + n_x_m * sn)
                    off0 = self._hm_off(par, self.rank, hv, strip)
                    for j in cutlass.range_constexpr(self.rank + 1, self.R):
                        hm_j = mPtrs[0, j]
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32, hm_j, cute.AddressSpace.gmem, assumed_align=4
                            ),
                            cute.make_layout(2 * self.R * self.HV * self.NSTRIPS * self.STRIP),
                        )
                        for i in cutlass.range_constexpr(sub_x):
                            crd = cX[base + i]
                            dst[off0 + crd[0] * 64 + crd[1]] = rXs[i]
        return st_vg, st_g1c, st_g2c, st_xp

    # ----------------------------------------------------------- MMA warp
    @cute.jit
    def _mma_item(
        self,
        nc,
        tiled_mma1f,
        tiled_mma2f,
        sWb_t,
        sKt_t,
        sXf_t,
        sxt_t,
        tAcc1ps,
        tXps,
        p_wk,
        p_g1s,
        p_g2s,
        p_pubs,
        p_xs,
        p_lam,
        st_wk,
        st_pub,
        st_xc,
        st_xcb,
        st_one,
    ):
        # flip: gemm1 is acc1^T = X^T @ W^T -> A = X^T (stage = pair),
        # B = W (stage = wk); gemm2 keeps A = K^T (stage = wk) but B is
        # the k-major x^T (stage = pair)
        tCrX = tiled_mma1f.make_fragment_A(sXf_t)
        tCrW = tiled_mma1f.make_fragment_B(sWb_t)
        tCrKt = tiled_mma2f.make_fragment_A(sKt_t)
        tCrx = tiled_mma2f.make_fragment_B(sxt_t)
        mma1 = tiled_mma1f
        mma2 = tiled_mma2f
        NKH = cute.size(tCrx, mode=[2]) // 2  # k-phases per token half
        # arm every LIVE pair's loop-top gemm2 wait for t=0 (no prior MMAs in
        # flight -> arrive immediately); balanced by the epilogue waits.  A
        # dead transition pair has no warps waiting on it, so arming it would
        # be a signal nobody consumes.
        for pp in cutlass.range_constexpr(self.n_groups_live):
            p_g2s[pp].producer_commit(st_one)
        # ping-pong: pair A's gemm executes while pair B publishes and
        # vice versa; each pair's chain only waits on its own commits
        for t in cutlass.range(nc, unroll=1):
            p_wk.consumer_wait(st_wk)
            if cutlass.const_expr(self.K == 128):
                # staircase: k-phase pair w starts once X rows
                # [32w, 32w+32) of both strips are published
                for w in cutlass.range_constexpr(4):
                    p_pubs[0][w].consumer_wait(st_pub)
                    mma1 = self._exec_gemm(
                        mma1, tAcc1ps[0], tCrX, tCrW, 0, st_wk.index, kp_lo=2 * w, kp_hi=2 * w + 2
                    )
                # Waiting p_lam here before this commit (the historical
                # 18-warp s1-writer scheme) gated the commit on the lambda
                # tail; the direct pair-0 reader wait remains preferable.
                p_g1s[0].producer_commit(st_one)
                # Pair 1 is the transition half; on a rank whose transition is
                # unreadable it has no warps to feed and its gemms are dropped
                # along with the edges that would order them.
                if cutlass.const_expr(self.n_groups_live > 1):
                    for w in cutlass.range_constexpr(4):
                        p_pubs[1][w].consumer_wait(st_pub)
                        mma1 = self._exec_gemm(
                            mma1,
                            tAcc1ps[1],
                            tCrX,
                            tCrW,
                            1,
                            st_wk.index,
                            kp_lo=2 * w,
                            kp_hi=2 * w + 2,
                        )
                    # The dedicated helpers' lambda publication orders pair-1's
                    # pass-2 readers through this commit; pair-0 readers wait
                    # p_lam directly after their g1 wake. st_wk carries the
                    # matching 2-stage (index, phase) sequence.
                    if cutlass.const_expr(self.need_lam):
                        p_lam.consumer_wait(st_wk)
                    p_g1s[1].producer_commit(st_one)
            else:
                p_pubs[0][0].consumer_wait(st_pub)  # pair A X published
                mma1 = self._exec_gemm(mma1, tAcc1ps[0], tCrX, tCrW, 0, st_wk.index)
                # K=64 keeps the v6 λ edge: strip-0 writer, waited before
                # the g1 pair-0 commit (readers gate on their g1)
                if cutlass.const_expr(self.need_lam):
                    p_lam.consumer_wait(st_wk)
                p_g1s[0].producer_commit(st_one)
                # See the K=128 branch: pair B is the transition half.
                if cutlass.const_expr(self.n_groups_live > 1):
                    p_pubs[1][0].consumer_wait(st_pub)  # pair B X published
                    mma1 = self._exec_gemm(mma1, tAcc1ps[1], tCrX, tCrW, 1, st_wk.index)
                    p_g1s[1].producer_commit(st_one)
            p_xs[0].consumer_wait(st_xc)  # pair A x^T half 0
            mma2 = self._exec_gemm(
                mma2, tXps[0], tCrKt, tCrx, st_wk.index, 0, acc_init=True, kp_lo=0, kp_hi=NKH
            )
            st_xc.advance()
            p_xs[0].consumer_wait(st_xc)  # pair A x^T half 1
            mma2 = self._exec_gemm(
                mma2, tXps[0], tCrKt, tCrx, st_wk.index, 0, acc_init=True, kp_lo=NKH
            )
            p_g2s[0].producer_commit(st_one)
            # Pair B's gemm2. Dropped with the rest of the transition pair when
            # it is dead; `st_xcb` then advances once per chunk instead of
            # twice, which is unobservable because nothing waits on p_xs[1].
            if cutlass.const_expr(self.n_groups_live > 1):
                p_xs[1].consumer_wait(st_xcb)  # pair B x^T half 0
                mma2 = self._exec_gemm(
                    mma2, tXps[1], tCrKt, tCrx, st_wk.index, 1, acc_init=True, kp_lo=0, kp_hi=NKH
                )
                st_xcb.advance()
                p_xs[1].consumer_wait(st_xcb)  # pair B x^T half 1
                mma2 = self._exec_gemm(
                    mma2, tXps[1], tCrKt, tCrx, st_wk.index, 1, acc_init=True, kp_lo=NKH
                )
                p_g2s[1].producer_commit(st_one)
            p_wk.consumer_release(st_wk)  # frees W/K once all gemms drain
            st_wk.advance()
            st_pub.advance()
            st_xc.advance()
            st_xcb.advance()
        return st_wk, st_pub, st_xc, st_xcb

    # one monotonic step flag per (strip, consumer rank)
    @cute.jit
    def _flag_consumers(self, mPtrs, hv, stepu):
        rb = self.rank * self.HV + hv
        for j in cutlass.range_constexpr(self.rank + 1, self.R):
            for s in cutlass.range_constexpr(self.NSTRIPS):
                self._signal_relaxed_sys(mPtrs[1, j], rb * self.NSTRIPS + s, stepu)

    # ------------------------------------------------------------ merge role
    @cute.jit
    def _merge_role(
        self,
        tidx,
        b,
        hv,
        par,
        stepu,
        mPtrs,
        mH0,
        pHMloc,
        storage,
        tiled_mma,
        tiled_copy_M,
        sM_layout,
    ):
        """SM80 warp-MMA fold chain (`CPProtocolMixin` helpers) on warps
        0-3, NamedBarrier(NB_MERGE, 128) instead of CTA barriers.

        `b` is the V-subtile (0..NB)."""
        bar = pipeline.NamedBarrier(barrier_id=NB_MERGE, num_threads=128)
        dtype = self.io_dtype
        K = self.K
        rank = self.rank

        pRaw = cute.recast_ptr(
            storage.sX.get_tensor(cute.make_layout(1)).iterator, dtype=cutlass.Float32
        )
        mRaw = cute.make_tensor(pRaw, cute.make_layout(self.NK * self.STRIP))
        mRawHe = cute.make_tensor(pRaw + self.NK * self.STRIP, cute.make_layout(self.STRIP))
        pHiLo = cute.recast_ptr(pRaw + (self.NK + 1) * self.STRIP, dtype=dtype)
        sMhi = cute.make_tensor(pHiLo, sM_layout)
        sMlo = cute.make_tensor(pHiLo + K * (K + 8), sM_layout)
        stripT_layout = cute.make_layout((64, K), stride=(1, 64))
        mAddT_smem = cute.make_tensor(mRawHe.iterator, stripT_layout)

        fl_loc = mPtrs[1, rank]
        thr_mma = tiled_mma.get_slice(tidx)
        macc = cute.make_rmem_tensor(thr_mma.partition_shape_C((64, K)), cutlass.Float32)

        if tidx == 0:
            self._spin_ge_sys(fl_loc, (0 * self.HV + hv) * self.NSTRIPS + b, stepu)
        bar.arrive_and_wait()
        self._stage_he_issue(pHMloc + self._hm_off(par, 0, hv, b), mRawHe, tiled_copy_M, tidx)
        cute.arch.cp_async_commit_group()
        rAhi = cute.make_fragment_like(macc, dtype)
        rAlo = cute.make_fragment_like(macc, dtype)
        cute.arch.cp_async_wait_group(0)
        bar.arrive_and_wait()
        tHe0 = thr_mma.partition_C(mAddT_smem)
        for i in cutlass.range_constexpr(cute.size(macc)):
            macc[i] = tHe0[i]
        bar.arrive_and_wait()  # he_0 reads done before he_1 restages

        if cutlass.const_expr(rank > 1):
            if tidx == 0:
                self._spin_producer_sys(fl_loc, 1, hv, b, stepu)
            bar.arrive_and_wait()
            self._stage_issue(pHMloc + self._hm_off(par, 1, hv, self.NV), mRaw, tiled_copy_M, tidx)
            self._stage_he_issue(pHMloc + self._hm_off(par, 1, hv, b), mRawHe, tiled_copy_M, tidx)
            cute.arch.cp_async_commit_group()
        for j in cutlass.range_constexpr(1, rank):
            cute.arch.cp_async_wait_group(0)
            bar.arrive_and_wait()
            self._convert_raw(mRaw, sMhi, sMlo, tidx)
            self._snapshot_and_init(thr_mma, macc, rAhi, rAlo, mAddT_smem, True)
            if cutlass.const_expr(j + 1 < rank):
                if tidx == 0:
                    self._spin_producer_sys(fl_loc, j + 1, hv, b, stepu)
            bar.arrive_and_wait()
            if cutlass.const_expr(j + 1 < rank):
                self._stage_issue(
                    pHMloc + self._hm_off(par, j + 1, hv, self.NV), mRaw, tiled_copy_M, tidx
                )
                self._stage_he_issue(
                    pHMloc + self._hm_off(par, j + 1, hv, b), mRawHe, tiled_copy_M, tidx
                )
                cute.arch.cp_async_commit_group()
            self._fold_gemms(thr_mma, tiled_mma, macc, rAhi, rAlo, sMhi, sMlo, tidx)

        # h0[hv, k, 64b + c] transposed block view (64, K):(1, V)
        gH0 = cute.make_tensor(
            mH0.iterator + (hv * self.K * self.V + b * 64),
            cute.make_layout((64, self.K), stride=(1, self.V)),
        )
        tH0 = thr_mma.partition_C(gH0)
        for i in cutlass.range_constexpr(cute.size(macc)):
            tH0[i] = macc[i]
        bar.arrive_and_wait()
        if tidx == 0:
            cute.arch.fence_acq_rel_sys()
            for j in cutlass.range_constexpr(rank):
                self._signal_relaxed_sys(mPtrs[2, j], (rank * self.HV + hv) * self.NB + b, stepu)


# ---------------------------------------------------------------------- host
def walk_descriptor(HV, n_ctas, nslot):
    """Host-precomputed per-(CTA, slot) compute items of the whole-head walk:
    int32 rows (hv, 0, 0, 0, valid, 0, 0, 0).

    Each item is a full [0, NT) scan of one head; CTA i owns heads
    {i, i+eff0, i+2*eff0, ...}.  The table is a pure function of the geometry
    (the kernel derives its chunk count from the dynamic `T`), so a wrapper
    builds and uploads it once.

    Vectorized over CTAs: every field is an affine or floor-div function of
    the CTA index, so the scalar `n_ctas x nslot` walk collapses to a handful
    of integer tensor ops."""
    d = torch.zeros(n_ctas, nslot, 8, dtype=torch.int32)
    idx = torch.arange(n_ctas, dtype=torch.int64)
    eff0 = min(n_ctas, HV)
    assert nslot * eff0 >= HV, "NSLOT too small for the walk"
    slots = torch.arange(nslot, dtype=torch.int64)
    wh = idx[:, None] + slots[None, :] * eff0  # [n_ctas, nslot]
    live = wh < HV
    d[..., 0] = torch.where(live, wh, 0).to(torch.int32)  # hv
    d[..., 4] = live.to(torch.int32)  # valid
    return d.flatten()


class CuteDSLFusedCPPreProcessWS(CPHostCaches):
    """Host wrapper for whole-head WS execution."""

    def __init__(self, group, H, HV, DK, DV, gate_mode="g", device=None):
        import torch.distributed as dist

        from .symm import CPSymmBuffers

        self.group = group
        self.R = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.H, self.HV, self.DK, self.DV = H, HV, DK, DV
        self.gate_mode = gate_mode
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.bufs = CPSymmBuffers(group, HV, DK, DV, self.device)
        self.ptrs = torch.stack(
            [self.bufs.hm_ptrs, self.bufs.fl_ptrs, self.bufs.ak_ptrs]
        ).contiguous()
        # Keep the canonical full-chunk specialization byte-for-byte: in
        # none mode it compiles out the lambda publication chain.  Ragged
        # calls lazily compile the generic tail-masked specialization instead
        # of slowing every T%64==0 production launch.  Under scalar `g` the two
        # are the same program (see `build_key`), so this one entry serves both
        # and no ragged call ever triggers a second compile.  Under per-key
        # `gk` the key is the same for every `T` and this one entry IS the
        # ragged specialization (`full_chunks=False`), so again exactly one
        # compile; see `build_key` for why that is preferred over a lazy
        # mid-training compile of the aligned program.
        aligned_key = PreProcessFwdFusedWS.build_key(gate_mode, True)
        self._ops = {
            aligned_key: PreProcessFwdFusedWS(
                H, HV, DK, DV, gate_mode, self.R, self.rank, full_chunks=gate_mode != "gk"
            )
        }
        self.op = self._ops[aligned_key]
        # [n_sm reserved words][n_ctas x NSLOT x 8 int32 walk descriptors].
        # Only producer ranks own a walk table.  It is a pure function of the
        # geometry -- the kernel derives its chunk count from the dynamic `T`
        # (see `ws_kernel`) -- so it is uploaded once here and never touched
        # again; under THD `nt` changes every step, and a per-length retune
        # used to be a ~5.7 us host dispatch plus its own kernel launch per
        # direction per layer per step.
        self.seg_flags = torch.zeros(
            self.op.n_sm + self.op.n_ctas * self.op.NSLOT * 8,
            dtype=torch.uint32,
            device=self.device,
        )
        if self.rank < self.R - 1:
            self.seg_flags[self.op.n_sm :].copy_(
                walk_descriptor(self.HV, self.op.n_ctas, self.op.NSLOT).view(torch.uint32)
            )
        self.step = 0
        self._compiled = {}
        self._init_host_caches()
        self._h0_zero = None
        # The inactive gate still occupies a positional kernel argument.
        # Keep one stable device allocation per wrapper instead of allocating
        # and zeroing a tiny CUDA tensor on every steady-state call.
        self._dummy_g = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        self._dummy_gk = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def launch_validated(
        self, k, v, w, g=None, gk=None, h0_out=None, emit_h=True, emit_m=True, _stream_handle=None
    ):
        """``__call__`` without the argument validation.

        For callers that have already established every condition ``__call__``
        checks -- in-tree that is ``cutedsl/backend.py``, whose dispatch
        predicate has to derive the same shapes, dtypes, devices and contiguity
        flags anyway in order to decide whether to run at all.  Re-deriving them
        here costs ~15 us of host time, and the CP pre-process runs twice per
        layer per step.  That is invisible under BSHD, where the kernel scans a
        whole 8192-token shard; under THD the window averages half a shard, so
        the flat host term is a third of the whole pre-process.

        The contract is exactly the validation block of ``__call__``: `k`,
        `v`(=u) and `w` contiguous bf16 CUDA tensors on ``self.device`` with
        shapes ``[1,T,H,K] / [1,T,HV,V] / [1,T,HV,K]``, ``T > 0``, the gate
        matching ``self.gate_mode`` (contiguous, fp32 or bf16, shape
        ``[1,T,HV]`` for `g`), and `h0_out` a contiguous fp32 ``[HV,K,V]``
        tensor on ``self.device``.

        ``emit_h`` / ``emit_m`` select which halves of this rank's summary
        ``h_out = M . h_in + X`` actually reach its consumers.  The kernel
        always pushes to ranks ``rank+1 .. R-1`` and always folds ranks
        ``0 .. rank-1`` -- one chain shape, no per-packing recompile -- and a
        packed batch is expressed by suppressing halves instead of by shrinking
        the chain.  Clearing ``emit_m`` makes the fold ``h <- 0 . h + X``, i.e.
        it TRUNCATES every consumer's sum at this rank, which is what a
        sequence boundary at or inside this rank's window means; clearing
        ``emit_h`` as well makes this rank contribute nothing at all.  The
        caller sets them from the CP context (see ``cutedsl/backend.py``):

            emit_h = post_num_ranks > 0
            emit_m = emit_h and pre_num_ranks > 0 and one local sequence

        The default ``True, True`` is the single-sequence case, where the whole
        group is one chain.
        """
        return self._launch(k, v, w, g, gk, h0_out, emit_h, emit_m, _stream_handle)

    @torch.no_grad()
    def __call__(self, k, v, w, g=None, gk=None, h0_out=None, emit_h=True, emit_m=True):
        self._validate(k, v, w, g, gk, h0_out)
        return self._launch(k, v, w, g, gk, h0_out, emit_h, emit_m)

    def _validate(self, k, v, w, g, gk, h0_out):
        if k.ndim != 4:
            raise ValueError("k must have shape [1,T,H,K]")
        B, T, H, K = k.shape
        if T <= 0:
            raise ValueError("T must be positive")
        if (B, H, K) != (1, self.H, self.DK):
            raise ValueError("k shape does not match the wrapper geometry")
        if tuple(v.shape) != (1, T, self.HV, self.DV):
            raise ValueError("v must have shape [1,T,HV,V]")
        if tuple(w.shape) != (1, T, self.HV, self.DK):
            raise ValueError("w must have shape [1,T,HV,K]")
        for tensor, name in ((k, "k"), (v, "v"), (w, "w")):
            if tensor.dtype != torch.bfloat16:
                raise TypeError(f"{name} must be bfloat16")
            if not tensor.is_cuda or not tensor.is_contiguous() or tensor.device != self.device:
                raise ValueError(f"{name} must be contiguous on {self.device}")
        if g is not None and gk is not None:
            raise ValueError("g and gk are mutually exclusive")
        supplied_mode = "g" if g is not None else ("gk" if gk is not None else "none")
        if supplied_mode != self.gate_mode:
            raise ValueError(
                f"wrapper gate_mode={self.gate_mode!r}, " f"got {supplied_mode!r} inputs"
            )
        if g is not None:
            if tuple(g.shape) != (1, T, self.HV):
                raise ValueError("g must have shape [1,T,HV]")
            if g.dtype not in (torch.float32, torch.bfloat16):
                raise TypeError("g must be fp32 or bfloat16")
            if not g.is_cuda or g.device != self.device:
                raise ValueError(f"g must be on {self.device}")
        if gk is not None:
            if tuple(gk.shape) != (1, T, self.HV, self.DK):
                raise ValueError("gk must have shape [1,T,HV,K]")
            if gk.dtype not in (torch.float32, torch.bfloat16):
                raise TypeError("gk must be fp32 or bfloat16")
            if not gk.is_cuda or gk.device != self.device:
                raise ValueError(f"gk must be on {self.device}")
        if h0_out is not None and (
            tuple(h0_out.shape) != (self.HV, self.DK, self.DV)
            or h0_out.dtype != torch.float32
            or not h0_out.is_cuda
            or not h0_out.is_contiguous()
            or h0_out.device != self.device
        ):
            raise ValueError("h0_out must be contiguous fp32 [HV,K,V] on " f"{self.device}")

    def _launch(self, k, v, w, g, gk, h0_out, emit_h=True, emit_m=True, _stream_handle=None):
        T = k.shape[1]
        if h0_out is None:
            if self.rank == 0:
                if self._h0_zero is None:
                    self._h0_zero = torch.zeros(
                        self.HV, self.DK, self.DV, dtype=torch.float32, device=self.device
                    )
                h0_out = self._h0_zero
            else:
                h0_out = torch.empty(
                    self.HV, self.DK, self.DV, dtype=torch.float32, device=self.device
                )
        g = _normalize_gate(g, self._dummy_g)
        gk = _normalize_gate(gk, self._dummy_gk)

        self.step += 1
        args = (
            self._p(k, cutlass.BFloat16),
            self._p(v, cutlass.BFloat16),
            self._p(w, cutlass.BFloat16),
            self._p(g, cutlass.Float32),
            self._p(gk, cutlass.Float32),
            self._p(self.bufs.hm_sym, cutlass.Float32),
            self._p(self.ptrs, cutlass.Int64),
            self._p(h0_out, cutlass.Float32),
            self._p(self.seg_flags, cutlass.Uint32),
            cutlass.Int32(T),
            cutlass.Int32(self.step),
            cutlass.Int32((1 if emit_h else 0) | (2 if emit_m else 0)),
            self._cu_stream(_stream_handle),
        )
        # `full_chunks` reaches codegen through exactly one flag,
        # `need_lam = use_g or not full_chunks`, so in g-mode the aligned and
        # ragged specializations are the same program.  (Verified by compiling
        # both and diffing the artifacts: identical sm_100a PTX and cubin.)
        # Keying the caches on `need_lam` instead of on `full_chunks`
        # therefore costs nothing in none mode, where the two really do
        # differ, and in g-mode it stops the first non-64-aligned call from
        # paying a multi-second `cute.compile` for a binary that is already
        # loaded.  That mattered under THD specifically: the window is
        # 64-aligned about one step in 64, so the ragged compile landed at a
        # random mid-training step, with the peer rank spinning on 148 SMs for
        # the duration.  gk-mode pins the ragged program for every `T` for the
        # same reason (see `build_key`), so its key never misses after the
        # constructor's entry either.
        key = self.op.__class__.build_key(self.gate_mode, T % BT == 0)
        op = self._ops.get(key)
        if op is None:
            op = PreProcessFwdFusedWS(
                self.H,
                self.HV,
                self.DK,
                self.DV,
                self.gate_mode,
                self.R,
                self.rank,
                full_chunks=False,
            )
            self._ops[key] = op
            # Geometry and scratch sizing are protocol-visible and must not
            # depend on the tail specialization.
            assert (op.n_ctas, op.NSLOT, op.n_sm, op.n_citems, op.n_mitems) == (
                self.op.n_ctas,
                self.op.NSLOT,
                self.op.n_sm,
                self.op.n_citems,
                self.op.n_mitems,
            )
        compiled = self._compiled.get(key)
        if compiled is None:
            compiled = cute.compile(op, *args)
            self._compiled[key] = compiled
        compiled(*args)
        return h0_out
