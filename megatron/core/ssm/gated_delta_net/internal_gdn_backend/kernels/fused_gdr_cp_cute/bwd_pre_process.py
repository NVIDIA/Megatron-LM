# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""CuTeDSL backward context-parallel summary producer for gated delta rule.

For every value head this kernel computes the fp32 affine summary

    dhm[hv] = [H_tilde | P]  in R^{K x (V + K)}

used by ``chunk_gated_delta_rule_bwd_dhu_pre_process``.  If ``Z`` is the
gradient entering the right side of a local shard, its left-boundary gradient
is

    Z_left = P @ Z + H_tilde.

The local recurrence is evaluated one 64-wide output-column strip per
four-warp CTA.  The CTA keeps the transposed state in registers and scans
64-token chunks from right to left:

    y^T = dv^T + Z^T K^T D             (H_tilde strips)
        =        P^T K^T D             (P strips)
    Z^T = Z^T Gamma - y^T W + scale * do^T E Q
    P^T = P^T Gamma - y^T W

where scalar ``g`` supplies D/E/Gamma, ``gk`` supplies a per-key Gamma, and
the ungated mode uses identities.  Q/K/W/dO/dV are bf16 MMA operands and all
persistent accumulators are fp32.

The implementation deliberately mirrors ``cutedsl_pre_process.py``:

* register C-fragments are regrouped as A operands between chained MMAs;
* Q/K/W/dO/dV use a two-stage reverse cp.async pipeline;
* a clamped in-bounds tile window plus an explicit validity mask handles a
  ragged first/last-sized chunk without a large predicated-copy path.

Supported production shapes are K in {64, 128}, V divisible by 64, BT=64,
B=1, and gate modes none/scalar-g/per-key-gk.  DPLR ``bg`` is intentionally
out of scope.  The promoted long-sequence path is scalar-g GDN.  The ungated
mode is useful for short/reference cases, but an uncontracted T=8192 scan
needs a TF32/hi-lo transition sidecar because repeatedly feeding P through a
bf16 MMA operand does not reproduce FLA's fp32 matrix chain.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_ptr

from .pre_process import PreProcessFwdMerged


class PreProcessBwdMerged(PreProcessFwdMerged):
    """One-CTA-per-strip local backward affine-summary producer."""

    def __init__(
        self,
        H: int,
        HV: int,
        K: int,
        V: int,
        gate_mode: str = "g",
        BT: int = 64,
    ):
        assert K in (64, 128), "backward producer currently supports K=64 or K=128"
        super().__init__(H, HV, K, V, gate_mode=gate_mode, BT=BT)
        # Backward needs five operand tiles.  Pairing strips would exceed the
        # useful register/smem budget and is not enabled.
        self.paired = False
        self.num_threads = 128

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        pQ: cute.Pointer,    # [1, T, H,  K] bf16
        pK: cute.Pointer,    # [1, T, H,  K] bf16
        pW: cute.Pointer,    # [1, T, HV, K] bf16
        pDO: cute.Pointer,   # [1, T, HV, V] bf16
        pDV: cute.Pointer,   # [1, T, HV, V] bf16
        pG: cute.Pointer,    # [1, T, HV] fp32 or dummy
        pGK: cute.Pointer,   # [1, T, HV, K] fp32 or dummy
        pDHM: cute.Pointer,  # [HV, K, V+K] fp32
        scale: cutlass.Float32,
        T: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        H, HV, K, V = self.H, self.HV, self.K, self.V
        mQ = cute.make_tensor(
            pQ, cute.make_layout((1, T, H, K), stride=(0, H * K, K, 1))
        )
        mK = cute.make_tensor(
            pK, cute.make_layout((1, T, H, K), stride=(0, H * K, K, 1))
        )
        mW = cute.make_tensor(
            pW, cute.make_layout((1, T, HV, K), stride=(0, HV * K, K, 1))
        )
        mDO = cute.make_tensor(
            pDO, cute.make_layout((1, T, HV, V), stride=(0, HV * V, V, 1))
        )
        mDV = cute.make_tensor(
            pDV, cute.make_layout((1, T, HV, V), stride=(0, HV * V, V, 1))
        )
        mG = cute.make_tensor(
            pG, cute.make_layout((1, T, HV), stride=(0, HV, 1))
        )
        mGK = cute.make_tensor(
            pGK, cute.make_layout((1, T, HV, K), stride=(0, HV * K, K, 1))
        )
        mDHM = cute.make_tensor(
            pDHM,
            cute.make_layout(
                (HV, K, V + K), stride=(K * (V + K), V + K, 1)
            ),
        )
        dtype = mQ.element_type

        sw_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        sQKW_layout = cute.tile_to_shape(sw_atom, (self.BT, self.K), (0, 1))
        sV_layout = cute.tile_to_shape(sw_atom, (self.BT, 64), (0, 1))

        @cute.struct
        class SharedStorage:
            # Two stages are sufficient to overlap reverse-scan loads and keep
            # K=128 below the 227 KiB SM100 dynamic-smem limit.
            sQ0: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sQKW_layout)], 1024
            ]
            sQ1: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sQKW_layout)], 1024
            ]
            sK0: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sQKW_layout)], 1024
            ]
            sK1: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sQKW_layout)], 1024
            ]
            sW0: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sQKW_layout)], 1024
            ]
            sW1: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sQKW_layout)], 1024
            ]
            sDO0: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sV_layout)], 1024
            ]
            sDO1: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sV_layout)], 1024
            ]
            sDV0: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sV_layout)], 1024
            ]
            sDV1: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sV_layout)], 1024
            ]
            sG0: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.BT * self.g_slab_hv if self.use_g else 8,
                ],
                16,
            ]
            sG1: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.BT * self.g_slab_hv if self.use_g else 8,
                ],
                16,
            ]
            sGk0: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, self.K if self.use_gk else 8
                ],
                16,
            ]
            sGk1: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, self.K if self.use_gk else 8
                ],
                16,
            ]

        atom_g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=128,
        )
        elems = 128 // dtype.width
        thr_k = self.K // elems
        tiled_copy_QKW = cute.make_tiled_copy_tv(
            atom_g2s,
            cute.make_layout((128 // thr_k, thr_k), stride=(thr_k, 1)),
            cute.make_layout((1, elems)),
        )
        thr_v = 64 // elems
        tiled_copy_V = cute.make_tiled_copy_tv(
            atom_g2s,
            cute.make_layout((128 // thr_v, thr_v), stride=(thr_v, 1)),
            cute.make_layout((1, elems)),
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
            (4, 1, 1),
            permutation_mnk=(64, 16, 16),
        )

        self.kernel(
            mQ,
            mK,
            mW,
            mDO,
            mDV,
            mG,
            mGK,
            mDHM,
            scale,
            T,
            sQKW_layout,
            sV_layout,
            tiled_copy_QKW,
            tiled_copy_V,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=[self.NSTRIPS, self.HV, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    # -------------------------------------------------- pipeline stage bodies
    @cute.jit
    def _issue_bwd_chunk(
        self,
        i_t,
        is_h,
        strip,
        T,
        mQ_h,
        mK_h,
        mW_h,
        mDO_h,
        mDV_h,
        tiled_copy_QKW,
        tiled_copy_V,
        thr_copy_QKW,
        thr_copy_V,
        dQ,
        dK,
        dW,
        dDO,
        dDV,
    ):
        """Issue one reverse-scan chunk into a selected smem stage."""
        t_ofs = cutlass.max(
            cutlass.min(i_t * self.BT, T - self.BT), cutlass.Int32(0)
        )
        gK = cute.local_tile(
            cute.domain_offset((t_ofs, 0), mK_h),
            (self.BT, self.K),
            (0, 0),
        )
        gW = cute.local_tile(
            cute.domain_offset((t_ofs, 0), mW_h),
            (self.BT, self.K),
            (0, 0),
        )
        cute.copy(tiled_copy_QKW, thr_copy_QKW.partition_S(gK), dK)
        cute.copy(tiled_copy_QKW, thr_copy_QKW.partition_S(gW), dW)
        if is_h:
            gQ = cute.local_tile(
                cute.domain_offset((t_ofs, 0), mQ_h),
                (self.BT, self.K),
                (0, 0),
            )
            gDO = cute.local_tile(
                cute.domain_offset((t_ofs, 0), mDO_h),
                (self.BT, 64),
                (0, strip),
            )
            gDV = cute.local_tile(
                cute.domain_offset((t_ofs, 0), mDV_h),
                (self.BT, 64),
                (0, strip),
            )
            cute.copy(tiled_copy_QKW, thr_copy_QKW.partition_S(gQ), dQ)
            cute.copy(tiled_copy_V, thr_copy_V.partition_S(gDO), dDO)
            cute.copy(tiled_copy_V, thr_copy_V.partition_S(gDV), dDV)

    @cute.jit
    def _step_bwd(
        self,
        i_t,
        is_h,
        strip,
        T,
        NT,
        tidx,
        hv,
        scale,
        nrows: cutlass.Constexpr,
        ncols: cutlass.Constexpr,
        ncols_k: cutlass.Constexpr,
        mG,
        mGK,
        mQ_h,
        mK_h,
        mW_h,
        mDO_h,
        mDV_h,
        tiled_copy_QKW,
        tiled_copy_V,
        thr_copy_QKW,
        thr_copy_V,
        tiled_mma,
        s_tiled_K,
        s_tiled_Wt,
        s_tiled_Qt,
        # current smem stage
        tSsK,
        tSsWt,
        tSsQt,
        sDOb,
        sDVb,
        sG,
        sGkLast,
        # destinations for the chunk two positions earlier
        nQsQ,
        nKsK,
        nWsW,
        nDOsDO,
        nDVsDV,
        nG,
        nGk,
        # register fragments
        tBrK,
        tBrK_cv,
        tBrWt,
        tBrWt_cv,
        tBrQt,
        tBrQt_cv,
        state,
        state_g2,
        state_mn,
        tmp,
        tmp_mn,
        row_scale,
        tCcC_mn,
        tCcS_mn,
    ):
        """Consume one chunk and refill its stage for the reverse pipeline."""
        dtype = self.io_dtype
        t0 = i_t * self.BT
        n_valid = T - t0

        # At most the newer refill group remains outstanding.
        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier()

        shift = t0 - cutlass.max(
            cutlass.min(t0, T - self.BT), cutlass.Int32(0)
        )
        g_last = cutlass.Float32(0.0)
        if cutlass.const_expr(self.use_g):
            g_last = sG[cutlass.min(n_valid, self.BT) - 1 + shift]

        # tmp = state(bf16) @ K^T
        tmp.fill(0.0)
        for kb in cutlass.range_constexpr(self.K // 16):
            state_sl = state_g2[None, None, (None, kb)]
            rA = cute.make_fragment_like(state_sl, dtype)
            rA.store(state_sl.load().to(dtype))
            rAv = self._a_view(rA)
            cute.copy(
                s_tiled_K,
                tSsK[None, None, kb],
                tBrK_cv[None, None, 0],
            )
            cute.gemm(
                tiled_mma,
                tmp,
                rAv[None, None, 0],
                tBrK[None, None, 0],
                tmp,
            )

        # row_scale is D's time diagonal and doubles as the validity mask.
        for c in cutlass.range_constexpr(ncols):
            n = tCcC_mn[0, c][1]
            idx = n - shift
            row_scale[c] = cutlass.Float32(0.0)
            if idx >= 0:
                if idx < n_valid:
                    if cutlass.const_expr(self.use_g):
                        row_scale[c] = cute.math.exp2(
                            g_last - sG[n], fastmath=True
                        )
                    else:
                        row_scale[c] = cutlass.Float32(1.0)

        # tmp becomes -y^T, ready to be regrouped as the A operand of
        # -y^T @ W.  Invalid positions are exactly zero.
        if is_h:
            for r in cutlass.range_constexpr(nrows):
                for c in cutlass.range_constexpr(ncols):
                    crd = tCcC_mn[r, c]
                    valid = row_scale[c] != cutlass.Float32(0.0)
                    y = tmp_mn[r, c] * row_scale[c]
                    if valid:
                        y = y + sDVb[crd[1], crd[0]].to(cutlass.Float32)
                    tmp_mn[r, c] = cutlass.Float32(0.0) - y
        else:
            for r in cutlass.range_constexpr(nrows):
                for c in cutlass.range_constexpr(ncols):
                    tmp_mn[r, c] = (
                        cutlass.Float32(0.0)
                        - tmp_mn[r, c] * row_scale[c]
                    )

        rY = cute.make_fragment_like(tmp, dtype)
        rY.store(tmp.load().to(dtype))
        rYv = self._a_view(rY)

        # Gamma decay of the persistent state.
        if cutlass.const_expr(self.use_g):
            gamma = cute.math.exp2(g_last, fastmath=True)
            state.store(state.load() * gamma)
        if cutlass.const_expr(self.use_gk):
            for c in cutlass.range_constexpr(ncols_k):
                n = tCcS_mn[0, c][1]
                gamma_k = cute.math.exp2(sGkLast[n], fastmath=True)
                for r in cutlass.range_constexpr(nrows):
                    state_mn[r, c] = state_mn[r, c] * gamma_k

        # state += -y^T @ W
        for kb in cutlass.range_constexpr(self.BT // 16):
            cute.copy(
                s_tiled_Wt,
                tSsWt[None, None, kb],
                tBrWt_cv[None, None, 0],
            )
            cute.gemm(
                tiled_mma,
                state,
                rYv[None, None, kb],
                tBrWt[None, None, 0],
                state,
            )

        if is_h:
            # Reuse tmp as the source A fragment.  Moving the scalar time
            # factor to dO preserves do^T diag(E) Q algebraically and lets the
            # source MMA accumulate directly into the persistent fp32 state.
            for r in cutlass.range_constexpr(nrows):
                for c in cutlass.range_constexpr(ncols):
                    crd = tCcC_mn[r, c]
                    n = crd[1]
                    idx = n - shift
                    src_scale = cutlass.Float32(0.0)
                    if idx >= 0:
                        if idx < n_valid:
                            src_scale = scale
                            if cutlass.const_expr(self.use_g):
                                src_scale = src_scale * cute.math.exp2(
                                    sG[n], fastmath=True
                                )
                    tmp_mn[r, c] = (
                        sDOb[crd[1], crd[0]].to(cutlass.Float32) * src_scale
                    )
            rDO = cute.make_fragment_like(tmp, dtype)
            rDO.store(tmp.load().to(dtype))
            rDOv = self._a_view(rDO)
            for kb in cutlass.range_constexpr(self.BT // 16):
                cute.copy(
                    s_tiled_Qt,
                    tSsQt[None, None, kb],
                    tBrQt_cv[None, None, 0],
                )
                cute.gemm(
                    tiled_mma,
                    state,
                    rDOv[None, None, kb],
                    tBrQt[None, None, 0],
                    state,
                )

        # Every thread is done reading this stage.  Refill it with chunk i-2
        # while the other stage is consumed by the next iteration.
        cute.arch.barrier()
        if i_t >= 2:
            self._issue_bwd_chunk(
                i_t - 2,
                is_h,
                strip,
                T,
                mQ_h,
                mK_h,
                mW_h,
                mDO_h,
                mDV_h,
                tiled_copy_QKW,
                tiled_copy_V,
                thr_copy_QKW,
                thr_copy_V,
                nQsQ,
                nKsK,
                nWsW,
                nDOsDO,
                nDVsDV,
            )
            self._load_gates(
                i_t - 2, T, tidx, hv, mG, mGK, nG, nGk
            )
        cute.arch.cp_async_commit_group()

    # --------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mW: cute.Tensor,
        mDO: cute.Tensor,
        mDV: cute.Tensor,
        mG: cute.Tensor,
        mGK: cute.Tensor,
        mDHM: cute.Tensor,
        scale: cutlass.Float32,
        T: cutlass.Int32,
        sQKW_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tiled_copy_QKW: cute.TiledCopy,
        tiled_copy_V: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        strip, hv, _ = cute.arch.block_idx()
        dtype = mQ.element_type
        is_h = strip < self.NV
        hk = hv // (self.HV // self.H)
        NT = cute.ceil_div(T, self.BT)

        mQ_h = mQ[0, None, hk, None]
        mK_h = mK[0, None, hk, None]
        mW_h = mW[0, None, hv, None]
        mDO_h = mDO[0, None, hv, None]
        mDV_h = mDV[0, None, hv, None]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = [
            storage.sQ0.get_tensor(sQKW_layout),
            storage.sQ1.get_tensor(sQKW_layout),
        ]
        sK = [
            storage.sK0.get_tensor(sQKW_layout),
            storage.sK1.get_tensor(sQKW_layout),
        ]
        sW = [
            storage.sW0.get_tensor(sQKW_layout),
            storage.sW1.get_tensor(sQKW_layout),
        ]
        sDO = [
            storage.sDO0.get_tensor(sV_layout),
            storage.sDO1.get_tensor(sV_layout),
        ]
        sDV = [
            storage.sDV0.get_tensor(sV_layout),
            storage.sDV1.get_tensor(sV_layout),
        ]

        g_layout = cute.make_layout(
            self.BT * self.g_slab_hv if self.use_g else 8
        )
        gk_layout = cute.make_layout(self.K if self.use_gk else 8)
        sGs = [
            storage.sG0.get_tensor(g_layout),
            storage.sG1.get_tensor(g_layout),
        ]
        if cutlass.const_expr(self.use_g):
            g_view = cute.make_layout(self.BT, stride=self.g_slab_hv)
            sG = [
                cute.make_tensor(b.iterator + hv % self.g_slab_hv, g_view)
                for b in sGs
            ]
        else:
            sG = sGs
        sGkLast = [
            storage.sGk0.get_tensor(gk_layout),
            storage.sGk1.get_tensor(gk_layout),
        ]

        # B operand views: direct K gives K^T in gemm1; transposed W/Q views
        # give W/Q as the reduction-major operands of gemm2/gemm3.
        t_layout = cute.make_layout(
            (self.K, self.BT), stride=(self.BT, 1)
        )
        sWt = [
            cute.composition(sW[0], t_layout),
            cute.composition(sW[1], t_layout),
        ]
        sQt = [
            cute.composition(sQ[0], t_layout),
            cute.composition(sQ[1], t_layout),
        ]

        thr_copy_QKW = tiled_copy_QKW.get_slice(tidx)
        thr_copy_V = tiled_copy_V.get_slice(tidx)
        tQsQ = [thr_copy_QKW.partition_D(b) for b in sQ]
        tKsK = [thr_copy_QKW.partition_D(b) for b in sK]
        tWsW = [thr_copy_QKW.partition_D(b) for b in sW]
        tDOsDO = [thr_copy_V.partition_D(b) for b in sDO]
        tDVsDV = [thr_copy_V.partition_D(b) for b in sDV]

        thr_mma = tiled_mma.get_slice(tidx)
        state = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((64, self.K)), cutlass.Float32
        )
        state_g2 = cute.make_tensor(
            state.iterator,
            cute.logical_divide(state.layout, (None, None, 2)),
        )
        state_mn = self._mn_view(state)

        tCcC_mn = self._mn_view(
            thr_mma.partition_C(cute.make_identity_tensor((64, 64)))
        )
        tCcS_mn = self._mn_view(
            thr_mma.partition_C(cute.make_identity_tensor((64, self.K)))
        )
        nrows = cute.size(tCcC_mn, mode=[0])
        ncols = cute.size(tCcC_mn, mode=[1])
        ncols_k = cute.size(tCcS_mn, mode=[1])

        state.fill(0.0)
        if not is_h:
            c0 = (strip - self.NV) * 64
            for r in cutlass.range_constexpr(nrows):
                for c in cutlass.range_constexpr(ncols_k):
                    crd = tCcS_mn[r, c]
                    if crd[1] == c0 + crd[0]:
                        state_mn[r, c] = cutlass.Float32(1.0)

        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            dtype,
        )
        smem_copy_atom_t = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            dtype,
        )
        s_tiled_K = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
        s_tiled_Wt = cute.make_tiled_copy_B(smem_copy_atom_t, tiled_mma)
        s_tiled_Qt = cute.make_tiled_copy_B(smem_copy_atom_t, tiled_mma)
        s_thr_K = s_tiled_K.get_slice(tidx)
        s_thr_Wt = s_tiled_Wt.get_slice(tidx)
        s_thr_Qt = s_tiled_Qt.get_slice(tidx)
        tSsK = [s_thr_K.partition_S(b) for b in sK]
        tSsWt = [s_thr_Wt.partition_S(b) for b in sWt]
        tSsQt = [s_thr_Qt.partition_S(b) for b in sQt]

        def _one_b_fragment(thr_copy, s_tile):
            part = thr_mma.partition_B(s_tile)
            grouped = cute.make_tensor(
                part.iterator,
                cute.logical_divide(part.layout, (None, None, 1)),
            )
            frag = thr_mma.make_fragment_B(
                grouped[None, None, (None, 0)]
            )
            return frag, thr_copy.retile(frag)

        tBrK, tBrK_cv = _one_b_fragment(s_thr_K, sK[0])
        tBrWt, tBrWt_cv = _one_b_fragment(s_thr_Wt, sWt[0])
        tBrQt, tBrQt_cv = _one_b_fragment(s_thr_Qt, sQt[0])

        tmp = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((64, 64)), cutlass.Float32
        )
        tmp_mn = self._mn_view(tmp)
        row_scale = cute.make_rmem_tensor(ncols, cutlass.Float32)

        # Seed the two reverse-pipeline stages with chunks NT-1 and NT-2.
        self._issue_bwd_chunk(
            NT - 1,
            is_h,
            strip,
            T,
            mQ_h,
            mK_h,
            mW_h,
            mDO_h,
            mDV_h,
            tiled_copy_QKW,
            tiled_copy_V,
            thr_copy_QKW,
            thr_copy_V,
            tQsQ[0],
            tKsK[0],
            tWsW[0],
            tDOsDO[0],
            tDVsDV[0],
        )
        self._load_gates(
            NT - 1, T, tidx, hv, mG, mGK, sGs[0], sGkLast[0]
        )
        cute.arch.cp_async_commit_group()
        if NT > 1:
            self._issue_bwd_chunk(
                NT - 2,
                is_h,
                strip,
                T,
                mQ_h,
                mK_h,
                mW_h,
                mDO_h,
                mDV_h,
                tiled_copy_QKW,
                tiled_copy_V,
                thr_copy_QKW,
                thr_copy_V,
                tQsQ[1],
                tKsK[1],
                tWsW[1],
                tDOsDO[1],
                tDVsDV[1],
            )
            self._load_gates(
                NT - 2, T, tidx, hv, mG, mGK, sGs[1], sGkLast[1]
            )
        cute.arch.cp_async_commit_group()

        for i2 in cutlass.range(cute.ceil_div(NT, 2)):
            c = NT - 1 - i2 * 2
            self._step_bwd(
                c,
                is_h,
                strip,
                T,
                NT,
                tidx,
                hv,
                scale,
                nrows,
                ncols,
                ncols_k,
                mG,
                mGK,
                mQ_h,
                mK_h,
                mW_h,
                mDO_h,
                mDV_h,
                tiled_copy_QKW,
                tiled_copy_V,
                thr_copy_QKW,
                thr_copy_V,
                tiled_mma,
                s_tiled_K,
                s_tiled_Wt,
                s_tiled_Qt,
                tSsK[0],
                tSsWt[0],
                tSsQt[0],
                sDO[0],
                sDV[0],
                sG[0],
                sGkLast[0],
                tQsQ[0],
                tKsK[0],
                tWsW[0],
                tDOsDO[0],
                tDVsDV[0],
                sGs[0],
                sGkLast[0],
                tBrK,
                tBrK_cv,
                tBrWt,
                tBrWt_cv,
                tBrQt,
                tBrQt_cv,
                state,
                state_g2,
                state_mn,
                tmp,
                tmp_mn,
                row_scale,
                tCcC_mn,
                tCcS_mn,
            )
            if c >= 1:
                self._step_bwd(
                    c - 1,
                    is_h,
                    strip,
                    T,
                    NT,
                    tidx,
                    hv,
                    scale,
                    nrows,
                    ncols,
                    ncols_k,
                    mG,
                    mGK,
                    mQ_h,
                    mK_h,
                    mW_h,
                    mDO_h,
                    mDV_h,
                    tiled_copy_QKW,
                    tiled_copy_V,
                    thr_copy_QKW,
                    thr_copy_V,
                    tiled_mma,
                    s_tiled_K,
                    s_tiled_Wt,
                    s_tiled_Qt,
                    tSsK[1],
                    tSsWt[1],
                    tSsQt[1],
                    sDO[1],
                    sDV[1],
                    sG[1],
                    sGkLast[1],
                    tQsQ[1],
                    tKsK[1],
                    tWsW[1],
                    tDOsDO[1],
                    tDVsDV[1],
                    sGs[1],
                    sGkLast[1],
                    tBrK,
                    tBrK_cv,
                    tBrWt,
                    tBrWt_cv,
                    tBrQt,
                    tBrQt_cv,
                    state,
                    state_g2,
                    state_mn,
                    tmp,
                    tmp_mn,
                    row_scale,
                    tCcC_mn,
                    tCcS_mn,
                )

        # Store through a transposed view so each strip's register fragment is
        # contiguous in its logical column dimension.
        mDHM_h = mDHM[hv, None, None]
        dhmT = cute.composition(
            mDHM_h,
            cute.make_layout(
                (self.V + self.K, self.K), stride=(self.K, 1)
            ),
        )
        gDHM = cute.local_tile(dhmT, (64, self.K), (strip, 0))
        tDHMg = thr_mma.partition_C(gDHM)
        for i in cutlass.range_constexpr(cute.size(state)):
            tDHMg[i] = state[i]


_BWD_COMPILE_CACHE: dict = {}


def _ptr(t: torch.Tensor, dtype):
    return make_ptr(
        dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )


def _pad_time_to_bt(t: torch.Tensor, BT: int) -> torch.Tensor:
    pad = BT - t.shape[1]
    if pad <= 0:
        return t
    # torch.nn.functional.pad lists dimensions from the innermost outward.
    spec = (0, 0) * (t.dim() - 2) + (0, pad, 0, 0)
    return torch.nn.functional.pad(t, spec)


@torch.no_grad()
def pre_process_bwd_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float | None = None,
    BT: int = 64,
) -> torch.Tensor:
    """Return local ``[H_tilde | P]`` summaries in FLA's packed layout."""
    B, T, H, K = q.shape
    HV, V = do.shape[2], do.shape[-1]
    assert B == 1 and T >= 1
    assert BT == 64
    assert q.shape == k.shape
    assert w.shape == (B, T, HV, K)
    assert dv.shape == do.shape
    assert HV % H == 0
    assert q.dtype == k.dtype == w.dtype == do.dtype == dv.dtype == torch.bfloat16
    assert all(x.is_contiguous() for x in (q, k, w, do, dv))
    assert not (g is not None and gk is not None)
    scale = K ** -0.5 if scale is None else float(scale)

    original_T = T
    if T < BT:
        q = _pad_time_to_bt(q, BT)
        k = _pad_time_to_bt(k, BT)
        w = _pad_time_to_bt(w, BT)
        do = _pad_time_to_bt(do, BT)
        dv = _pad_time_to_bt(dv, BT)
        if g is not None:
            g = _pad_time_to_bt(g.unsqueeze(-1), BT).squeeze(-1)
        if gk is not None:
            gk = _pad_time_to_bt(gk, BT)

    mode = "g" if g is not None else ("gk" if gk is not None else "none")
    if g is None:
        g = torch.zeros(1, 1, 1, dtype=torch.float32, device=q.device)
    if gk is None:
        gk = torch.zeros(
            1, 1, 1, 1, dtype=torch.float32, device=q.device
        )
    g = g.float().contiguous()
    gk = gk.float().contiguous()
    dhm = torch.empty(
        HV, K, V + K, dtype=torch.float32, device=q.device
    )

    args = (
        _ptr(q, cutlass.BFloat16),
        _ptr(k, cutlass.BFloat16),
        _ptr(w, cutlass.BFloat16),
        _ptr(do, cutlass.BFloat16),
        _ptr(dv, cutlass.BFloat16),
        _ptr(g, cutlass.Float32),
        _ptr(gk, cutlass.Float32),
        _ptr(dhm, cutlass.Float32),
        cutlass.Float32(scale),
        cutlass.Int32(original_T),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    key = (H, HV, K, V, mode)
    compiled = _BWD_COMPILE_CACHE.get(key)
    if compiled is None:
        op = PreProcessBwdMerged(
            H, HV, K, V, gate_mode=mode, BT=BT
        )
        compiled = cute.compile(op, *args)
        _BWD_COMPILE_CACHE[key] = compiled
    compiled(*args)
    return dhm
