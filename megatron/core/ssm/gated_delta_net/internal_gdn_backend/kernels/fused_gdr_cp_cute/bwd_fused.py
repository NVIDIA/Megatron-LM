# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""Fused CuTeDSL CP backward ``dhu`` pre-process.

This is the reverse-direction sibling of :mod:`fused_gdr_cp_cute.fused`.
One kernel launch per rank performs all three parts of FLA's backward CP
pre-process:

* local reverse scan, producing ``dhm = [Htilde | P]``;
* peer writes plus system-scope ready flags;
* descending suffix composition into the current rank's ``dht``.

The local arithmetic is the proven :class:`PreProcessBwdMerged` recurrence.
The peer protocol and the high/low bf16 warp-MMA fold are inherited from the
non-warp-specialized forward fused kernel.  Only the direction changes:

* rank ``r > 0`` produces and pushes to consumers ``0 .. r-1``;
* rank ``r < R-1`` consumes sources ``R-1 .. r+1``;
* acknowledgements travel from an earlier consumer to a later producer.

The first production path intentionally keeps one CTA per 64-column summary
strip (no T-split), B=1, BT=64, K in {64,128}, and V divisible by 64.  Gate
modes none/scalar-g/per-key-gk share the standalone backward implementation.
DPLR ``bg`` is not supported.
"""

from __future__ import annotations

from collections import OrderedDict

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from .bwd_pre_process import PreProcessBwdMerged, _pad_time_to_bt
from .fused import PreProcessFwdFused

BT = 64
# CuTeDSL pointer objects memoized by device address. THD gives most arguments
# a fresh address every step (the producer window starts at a moving `bos`), so
# this must evict rather than fill and stop -- see `_p`.
PTR_CACHE_SIZE = 4096


class PreProcessBwdFused(PreProcessBwdMerged, PreProcessFwdFused):
    """One-launch local-summary producer, reverse peer push, and suffix fold."""

    def __init__(
        self,
        H: int,
        HV: int,
        K: int,
        V: int,
        gate_mode: str,
        R: int,
        rank: int,
    ):
        # Call the protocol-bearing base explicitly.  A normal super() would
        # enter PreProcessBwdMerged.__init__, whose next MRO class is the fused
        # forward class and therefore has a different constructor signature.
        PreProcessFwdFused.__init__(
            self,
            H,
            HV,
            K,
            V,
            gate_mode,
            R,
            rank,
            split=1,
            B=1,
        )
        assert K in (64, 128), "fused backward supports K in {64,128}"
        assert V % 64 == 0
        # The chain is always the whole group.  It used to be configurable
        # (`pre_count` / `post_count`), which let a caller build an engine whose
        # pushes covered fewer ranks than the emit rules in
        # `cutedsl/backend.py` assume they cover -- those rules truncate a
        # consumer's fold by zeroing a transition, which only bounds the sum
        # correctly if every earlier rank is in it.  A shortened chain would
        # drop terms twice.  It is fixed here so that cannot be expressed.
        self.consumer_lo = 0
        self.source_hi = R - 1
        # Backward producers are all non-first ranks.  Backward consumers are
        # all non-last ranks and own one merge CTA per 64 output columns.
        self.n_compute = self.NSTRIPS if rank > 0 else 0
        self.n_merge = self.NV if rank < R - 1 else 0
        self.n_work = self.n_compute + self.n_merge
        # A rank whose complete local sequence neither enters nor leaves this
        # CP shard still participates in symmetric-buffer initialization.  A
        # single idle CTA keeps its launch legal without touching the wire.
        self.n_grid = max(self.n_work, 1)

    @cute.jit
    def __call__(
        self,
        pQ: cute.Pointer,      # [1,T,H,K] bf16
        pK: cute.Pointer,      # [1,T,H,K] bf16
        pW: cute.Pointer,      # [1,T,HV,K] bf16
        pDO: cute.Pointer,     # [1,T,HV,V] bf16
        pDV: cute.Pointer,     # [1,T,HV,V] bf16
        pG: cute.Pointer,      # [1,T,HV] fp32 or dummy
        pGK: cute.Pointer,     # [1,T,HV,K] fp32 or dummy
        pHMloc: cute.Pointer,  # local symmetric [2,R,1,HV,NG,K,64]
        pPtrs: cute.Pointer,   # int64 [3,R], rows hm/flags/acks
        pDHT: cute.Pointer,    # [HV,K,V] fp32 canonical output
        scale: cutlass.Float32,
        T: cutlass.Int32,
        step: cutlass.Int32,
        emit: cutlass.Int32,   # bit 0 = push Htilde half, bit 1 = push P half
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
        mPtrs = cute.make_tensor(
            pPtrs, cute.make_layout((3, self.R), stride=(self.R, 1))
        )
        mDHT = cute.make_tensor(
            pDHT, cute.make_layout((HV, K, V), stride=(K * V, V, 1))
        )
        dtype = mQ.element_type

        sw_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        sQKW_layout = cute.tile_to_shape(sw_atom, (BT, K), (0, 1))
        sV_layout = cute.tile_to_shape(sw_atom, (BT, 64), (0, 1))
        # Same fold layout as the forward non-WS path: rows are 16-byte
        # aligned and shifted by four banks to keep ldmatrix traffic spread.
        sM_layout = cute.make_layout((K, K), stride=(K + 8, 1))

        @cute.struct
        class SharedStorage:
            # The six Q/K/W slabs are deliberately contiguous.  Once the
            # reverse scan finishes (or in a merge-only CTA), they are
            # reinterpreted as fp32 [P | Htilde] fold staging:
            # (NK+1)*K*64 floats.  At K=128 both views occupy exactly 96 KiB.
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
                    BT * self.g_slab_hv if self.use_g else 8,
                ],
                16,
            ]
            sG1: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    BT * self.g_slab_hv if self.use_g else 8,
                ],
                16,
            ]
            sGk0: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, K if self.use_gk else 8
                ],
                16,
            ]
            sGk1: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, K if self.use_gk else 8
                ],
                16,
            ]
            sMhi: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sM_layout)], 1024
            ]
            sMlo: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sM_layout)], 1024
            ]

        atom_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            dtype,
            num_bits_per_copy=128,
        )
        elems = 128 // dtype.width
        thr_k = K // elems
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
        atom_g2s_f32 = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        tiled_copy_M = cute.make_tiled_copy_tv(
            atom_g2s_f32, cute.make_layout(128), cute.make_layout(4)
        )
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(
                dtype, cutlass.Float32, (16, 8, 16)
            ),
            (4, 1, 1),
            permutation_mnk=(64, 16, 16),
        )

        self.fused_bwd_kernel(
            mQ,
            mK,
            mW,
            mDO,
            mDV,
            mG,
            mGK,
            mPtrs,
            mDHT,
            scale,
            T,
            step,
            emit,
            pHMloc,
            sQKW_layout,
            sV_layout,
            sM_layout,
            tiled_copy_QKW,
            tiled_copy_V,
            tiled_copy_M,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=[self.n_grid, HV, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def fused_bwd_kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mW: cute.Tensor,
        mDO: cute.Tensor,
        mDV: cute.Tensor,
        mG: cute.Tensor,
        mGK: cute.Tensor,
        mPtrs: cute.Tensor,
        mDHT: cute.Tensor,
        scale: cutlass.Float32,
        T: cutlass.Int32,
        step: cutlass.Int32,
        emit: cutlass.Int32,
        pHMloc: cute.Pointer,
        sQKW_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sM_layout: cute.Layout,
        tiled_copy_QKW: cute.TiledCopy,
        tiled_copy_V: cute.TiledCopy,
        tiled_copy_M: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, hv, _ = cute.arch.block_idx()
        dtype = mQ.element_type
        rank = self.rank
        par = step % 2
        stepu = step.to(cutlass.Uint32)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sMhi = storage.sMhi.get_tensor(sM_layout)
        sMlo = storage.sMlo.get_tensor(sM_layout)
        # Fold raw storage overlays exactly the six Q/K/W scan stages.
        pRaw = cute.recast_ptr(
            storage.sQ0.get_tensor(cute.make_layout(1)).iterator,
            dtype=cutlass.Float32,
        )
        mRaw = cute.make_tensor(
            pRaw, cute.make_layout(self.NK * self.STRIP)
        )
        mRawH = cute.make_tensor(
            pRaw + self.NK * self.STRIP,
            cute.make_layout(self.STRIP),
        )
        stripT_layout = cute.make_layout(
            (64, self.K), stride=(1, 64)
        )
        mAddT_smem = cute.make_tensor(mRawH.iterator, stripT_layout)

        if bidx < self.n_compute:
            # ===================== local reverse-scan producer ===============
            strip = bidx
            is_h = strip < self.NV
            hk = hv // (self.HV // self.H)
            NT = cute.ceil_div(T, BT)

            mQ_h = mQ[0, None, hk, None]
            mK_h = mK[0, None, hk, None]
            mW_h = mW[0, None, hv, None]
            mDO_h = mDO[0, None, hv, None]
            mDV_h = mDV[0, None, hv, None]

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
                BT * self.g_slab_hv if self.use_g else 8
            )
            gk_layout = cute.make_layout(
                self.K if self.use_gk else 8
            )
            sGs = [
                storage.sG0.get_tensor(g_layout),
                storage.sG1.get_tensor(g_layout),
            ]
            if cutlass.const_expr(self.use_g):
                g_view = cute.make_layout(
                    BT, stride=self.g_slab_hv
                )
                sG = [
                    cute.make_tensor(
                        slab.iterator + hv % self.g_slab_hv, g_view
                    )
                    for slab in sGs
                ]
            else:
                sG = sGs
            sGkLast = [
                storage.sGk0.get_tensor(gk_layout),
                storage.sGk1.get_tensor(gk_layout),
            ]

            t_layout = cute.make_layout(
                (self.K, BT), stride=(BT, 1)
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
            tQsQ = [thr_copy_QKW.partition_D(slab) for slab in sQ]
            tKsK = [thr_copy_QKW.partition_D(slab) for slab in sK]
            tWsW = [thr_copy_QKW.partition_D(slab) for slab in sW]
            tDOsDO = [thr_copy_V.partition_D(slab) for slab in sDO]
            tDVsDV = [thr_copy_V.partition_D(slab) for slab in sDV]

            thr_mma = tiled_mma.get_slice(tidx)
            state = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((64, self.K)),
                cutlass.Float32,
            )
            state_g2 = cute.make_tensor(
                state.iterator,
                cute.logical_divide(
                    state.layout, (None, None, 2)
                ),
            )
            state_mn = self._mn_view(state)
            tCcC_mn = self._mn_view(
                thr_mma.partition_C(
                    cute.make_identity_tensor((64, 64))
                )
            )
            tCcS_mn = self._mn_view(
                thr_mma.partition_C(
                    cute.make_identity_tensor((64, self.K))
                )
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
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    transpose=False, num_matrices=4
                ),
                dtype,
            )
            smem_copy_atom_t = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    transpose=True, num_matrices=4
                ),
                dtype,
            )
            s_tiled_K = cute.make_tiled_copy_B(
                smem_copy_atom, tiled_mma
            )
            s_tiled_Wt = cute.make_tiled_copy_B(
                smem_copy_atom_t, tiled_mma
            )
            s_tiled_Qt = cute.make_tiled_copy_B(
                smem_copy_atom_t, tiled_mma
            )
            s_thr_K = s_tiled_K.get_slice(tidx)
            s_thr_Wt = s_tiled_Wt.get_slice(tidx)
            s_thr_Qt = s_tiled_Qt.get_slice(tidx)
            tSsK = [s_thr_K.partition_S(slab) for slab in sK]
            tSsWt = [s_thr_Wt.partition_S(slab) for slab in sWt]
            tSsQt = [s_thr_Qt.partition_S(slab) for slab in sQt]

            def _one_b_fragment(thr_copy, s_tile):
                part = thr_mma.partition_B(s_tile)
                grouped = cute.make_tensor(
                    part.iterator,
                    cute.logical_divide(
                        part.layout, (None, None, 1)
                    ),
                )
                frag = thr_mma.make_fragment_B(
                    grouped[None, None, (None, 0)]
                )
                return frag, thr_copy.retile(frag)

            tBrK, tBrK_cv = _one_b_fragment(s_thr_K, sK[0])
            tBrWt, tBrWt_cv = _one_b_fragment(s_thr_Wt, sWt[0])
            tBrQt, tBrQt_cv = _one_b_fragment(s_thr_Qt, sQt[0])
            tmp = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((64, 64)),
                cutlass.Float32,
            )
            tmp_mn = self._mn_view(tmp)
            row_scale = cute.make_rmem_tensor(
                ncols, cutlass.Float32
            )

            # Reverse double-buffered prologue: newest chunk first.
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
                NT - 1,
                T,
                tidx,
                hv,
                mG,
                mGK,
                sGs[0],
                sGkLast[0],
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
                    NT - 2,
                    T,
                    tidx,
                    hv,
                    mG,
                    mGK,
                    sGs[1],
                    sGkLast[1],
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

            # Parity credit: every earlier consumer must have retired the
            # previous user of this parity slot before it is overwritten.
            ackv = cutlass.max(step - 2, 0).to(cutlass.Uint32)
            ak_loc = mPtrs[2, rank]
            if tidx == 0:
                for cc in cutlass.range_constexpr(
                    self.consumer_lo, rank
                ):
                    for b in cutlass.range_constexpr(self.NB):
                        self._spin_ge_sys(
                            ak_loc,
                            (cc * self.HV + hv) * self.NB + b,
                            ackv,
                        )
            cute.arch.barrier()

            # Emit mask: a rank whose summary must not reach its consumers
            # still has to push and flag -- they spin on those flags -- so the
            # half is zeroed rather than skipped.  `state` is dead after the
            # push, so this is in place.  See `launch_validated`.
            if (emit & (1 if is_h else 2)) == 0:
                for i in cutlass.range_constexpr(cute.size(state)):
                    state[i] = cutlass.Float32(0.0)

            # Push the final register fragment directly into every earlier
            # consumer's symmetric allocation at source slot ``rank``.
            for j in cutlass.range_constexpr(
                self.consumer_lo, rank
            ):
                dptr = cute.make_ptr(
                    cutlass.Float32,
                    mPtrs[0, j],
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ) + self._hm_off(par, rank, hv, strip)
                gBlk = cute.make_tensor(dptr, stripT_layout)
                tBlk = thr_mma.partition_C(gBlk)
                for i in cutlass.range_constexpr(cute.size(state)):
                    tBlk[i] = state[i]
            cute.arch.barrier()
            cute.arch.fence_acq_rel_sys()
            if tidx == 0:
                for j in cutlass.range_constexpr(
                    self.consumer_lo, rank
                ):
                    self._signal_relaxed_sys(
                        mPtrs[1, j],
                        (rank * self.HV + hv) * self.NSTRIPS + strip,
                        stepu,
                    )

        elif bidx < self.n_work:
            # ======================= descending suffix merge =================
            b = bidx - self.n_compute
            fl_loc = mPtrs[1, rank]
            thr_mma = tiled_mma.get_slice(tidx)
            macc = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((64, self.K)),
                cutlass.Float32,
            )
            rAhi = cute.make_fragment_like(macc, self.io_dtype)
            rAlo = cute.make_fragment_like(macc, self.io_dtype)
            last = self.source_hi

            # The incoming gradient to the global right boundary is zero, so
            # the first live suffix term is exactly Htilde from the last rank.
            if tidx == 0:
                self._spin_ge_sys(
                    fl_loc,
                    (last * self.HV + hv) * self.NSTRIPS + b,
                    stepu,
                )
            cute.arch.barrier()
            self._stage_he_issue(
                pHMloc + self._hm_off(par, last, hv, b),
                mRawH,
                tiled_copy_M,
                tidx,
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            tHlast = thr_mma.partition_C(mAddT_smem)
            for i in cutlass.range_constexpr(cute.size(macc)):
                macc[i] = tHlast[i]
            cute.arch.barrier()

            # Prime the next-earlier source when at least two ranks follow us.
            if cutlass.const_expr(rank + 1 < self.source_hi):
                j0 = self.source_hi - 1
                if tidx == 0:
                    self._spin_producer_sys(
                        fl_loc, j0, hv, b, stepu
                    )
                cute.arch.barrier()
                self._stage_issue(
                    pHMloc + self._hm_off(
                        par, j0, hv, self.NV
                    ),
                    mRaw,
                    tiled_copy_M,
                    tidx,
                )
                self._stage_he_issue(
                    pHMloc + self._hm_off(par, j0, hv, b),
                    mRawH,
                    tiled_copy_M,
                    tidx,
                )
                cute.arch.cp_async_commit_group()

            # z <- P_j @ z + Htilde_j for j=R-2,...,rank+1.
            for j in cutlass.range_constexpr(
                self.source_hi - 1, rank, -1
            ):
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                self._convert_raw(
                    mRaw, sMhi, sMlo, tidx
                )
                self._snapshot_and_init(
                    thr_mma,
                    macc,
                    rAhi,
                    rAlo,
                    mAddT_smem,
                    True,
                )
                if cutlass.const_expr(j - 1 > rank):
                    if tidx == 0:
                        self._spin_producer_sys(
                            fl_loc, j - 1, hv, b, stepu
                        )
                cute.arch.barrier()
                if cutlass.const_expr(j - 1 > rank):
                    self._stage_issue(
                        pHMloc + self._hm_off(
                            par, j - 1, hv, self.NV
                        ),
                        mRaw,
                        tiled_copy_M,
                        tidx,
                    )
                    self._stage_he_issue(
                        pHMloc + self._hm_off(
                            par, j - 1, hv, b
                        ),
                        mRawH,
                        tiled_copy_M,
                        tidx,
                    )
                    cute.arch.cp_async_commit_group()
                self._fold_gemms(
                    thr_mma,
                    tiled_mma,
                    macc,
                    rAhi,
                    rAlo,
                    sMhi,
                    sMlo,
                    tidx,
                )

            gDHT = cute.make_tensor(
                mDHT.iterator
                + (hv * self.K * self.V + b * 64),
                cute.make_layout(
                    (64, self.K), stride=(1, self.V)
                ),
            )
            tDHT = thr_mma.partition_C(gDHT)
            for i in cutlass.range_constexpr(cute.size(macc)):
                tDHT[i] = macc[i]
            cute.arch.barrier()
            if tidx == 0:
                # Output-before-credit ordering, then acknowledge every later
                # producer whose parity slot this consumer has retired.
                cute.arch.fence_acq_rel_sys()
                for j in cutlass.range_constexpr(
                    rank + 1, self.source_hi + 1
                ):
                    self._signal_relaxed_sys(
                        mPtrs[2, j],
                        (rank * self.HV + hv) * self.NB + b,
                        stepu,
                    )


def _ptr(tensor: torch.Tensor, dtype):
    return make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )


def _make_raw_stream():
    """See `fused_ws._make_raw_stream`."""
    raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if raw is None:
        def raw(_index):
            return torch.cuda.current_stream().cuda_stream
    return raw


_raw_stream = _make_raw_stream()


def _normalize_gate(gate, dummy):
    """fp32-contiguous view of a gate slot, without a no-op ATen call."""
    if gate is None:
        return dummy
    if gate.dtype is not torch.float32:
        gate = gate.float()
    return gate if gate.is_contiguous() else gate.contiguous()


class CuteDSLFusedCPBwdPreProcess:
    """Persistent host wrapper for the fused reverse-direction kernel."""

    # Compute engine.  Subclasses (e.g. the warp-specialized Blackwell
    # backward) swap this without duplicating the host wrapper.
    OP_CLASS = None

    def _op_key(self, T):
        """Compile-cache key.  Engines that specialize on chunk alignment
        (the warp-specialized backward) widen this.

        The chain shape is deliberately NOT part of it.  The engine is always
        built on the full chain -- rank r pushes to ``r-1 .. 0`` and folds
        ``r+1 .. R-1`` -- and a packed batch is expressed by suppressing halves
        of the pushed summary at runtime (``emit_h`` / ``emit_m``) rather than
        by shortening the chain.  Keying on ``(pre_count, post_count)`` used to
        mean a fresh multi-second `cute.compile` for every chain shape the
        packing produced, at a random mid-training step, with every peer rank
        spinning for the duration -- and under THD the shape changes every step.
        """
        return ()

    def _make_op(self, key):
        del key
        cls = type(self).OP_CLASS or PreProcessBwdFused
        return cls(
            self.H,
            self.HV,
            self.DK,
            self.DV,
            self.gate_mode,
            self.R,
            self.rank,
        )

    def __init__(
        self,
        group,
        H: int,
        HV: int,
        DK: int,
        DV: int,
        gate_mode: str = "g",
        device=None,
    ):
        import torch.distributed as dist

        from .symm import CPSymmBuffers

        assert DK in (64, 128)
        assert DV % 64 == 0
        assert gate_mode in ("none", "g", "gk")
        self.group = group
        self.R = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        assert self.R >= 2
        self.H, self.HV, self.DK, self.DV = H, HV, DK, DV
        self.gate_mode = gate_mode
        self.device = device or torch.device(
            "cuda", torch.cuda.current_device()
        )
        self.bufs = CPSymmBuffers(
            group, HV, DK, DV, self.device, smax=1, B=1
        )
        self.ptrs = torch.stack(
            [
                self.bufs.hm_ptrs,
                self.bufs.fl_ptrs,
                self.bufs.ak_ptrs,
            ]
        ).contiguous()
        self.step = 0
        self._ops: dict[tuple, PreProcessBwdFused] = {}
        self._compiled: dict[tuple, object] = {}
        self._ptr_cache: OrderedDict[tuple[int, object], object] = OrderedDict()
        self._stream_cache: dict[int, object] = {}
        self._last_zero = None
        self._dummy_g = torch.zeros(
            1, 1, 1, dtype=torch.float32, device=self.device
        )
        self._dummy_gk = torch.zeros(
            1, 1, 1, 1, dtype=torch.float32, device=self.device
        )

    def _cu_stream(self):
        """`cuda.CUstream` for the current torch stream, memoized by handle.

        See the forward wrapper: the raw accessor avoids building a throwaway
        `torch.cuda.Stream` (0.09 us against 3.1 us) and the ctypes wrapper is
        cached on the handle.
        """
        handle = _raw_stream(self.device.index)
        stream = self._stream_cache.get(handle)
        if stream is None:
            stream = cuda.CUstream(handle)
            self._stream_cache[handle] = stream
        return stream

    def _p(self, tensor: torch.Tensor, dtype):
        """Memoize the CuTeDSL pointer object for a device address.

        LRU, not fill-and-stop.  Under THD the producer window starts at a
        different `bos` every step, so most arguments arrive with an address
        never seen before; a cache that stops inserting once full ends up
        holding only the addresses from the first thousand steps and misses on
        every call thereafter.  Evicting keeps the recurring entries (the
        symmetric buffers, the dummy gates) resident and bounds the retained
        pointer objects at the same time.
        """
        key = (tensor.data_ptr(), dtype)
        pointer = self._ptr_cache.get(key)
        if pointer is not None:
            self._ptr_cache.move_to_end(key)
            return pointer
        pointer = _ptr(tensor, dtype)
        self._ptr_cache[key] = pointer
        if len(self._ptr_cache) > PTR_CACHE_SIZE:
            self._ptr_cache.popitem(last=False)
        return pointer

    @torch.no_grad()
    def launch_validated(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        do: torch.Tensor,
        dv: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        scale: float | None = None,
        dht_out: torch.Tensor | None = None,
        emit_h: bool = True,
        emit_m: bool = True,
    ) -> torch.Tensor:
        """``__call__`` without the argument assertions.

        Backward twin of the forward wrapper's `launch_validated`; see there.
        The contract is the assertion block of ``__call__``: `q`/`k`/`w`/`do`/
        `dv` contiguous bf16 with `B == 1`, `T >= 1`, the wrapper's own
        `(H, HV, K, V)`, a gate matching `self.gate_mode`, and `dht_out` (when
        given) a contiguous fp32 `[HV,K,V]` tensor.

        ``emit_h`` / ``emit_m`` are the mirror of the forward's: the reverse
        chain is always the full ``rank-1 .. 0`` push and ``rank+1 .. R-1``
        fold, and a packed batch suppresses halves of the pushed
        ``dh_out = P . dh_in + Htilde`` instead of shortening it.  The caller
        sets them from the CP context (see ``cutedsl/backend.py``):

            emit_h = pre_num_ranks > 0
            emit_m = emit_h and post_num_ranks > 0 and one local sequence
        """
        return self._launch(q, k, w, do, dv, g, gk, scale, dht_out,
                            emit_h, emit_m)

    @torch.no_grad()
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        do: torch.Tensor,
        dv: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        scale: float | None = None,
        dht_out: torch.Tensor | None = None,
        emit_h: bool = True,
        emit_m: bool = True,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        HV, V = do.shape[2], do.shape[-1]
        assert B == 1 and T >= 1
        assert (H, HV, K, V) == (
            self.H,
            self.HV,
            self.DK,
            self.DV,
        )
        assert q.shape == k.shape
        assert w.shape == (B, T, HV, K)
        assert do.shape == dv.shape == (B, T, HV, V)
        assert HV % H == 0
        assert not (g is not None and gk is not None)
        assert all(
            tensor.dtype == torch.bfloat16
            for tensor in (q, k, w, do, dv)
        )
        assert all(
            tensor.is_contiguous()
            for tensor in (q, k, w, do, dv)
        )
        mode = (
            "g"
            if g is not None
            else ("gk" if gk is not None else "none")
        )
        assert mode == self.gate_mode
        return self._launch(q, k, w, do, dv, g, gk, scale, dht_out,
                            emit_h, emit_m)

    def _launch(self, q, k, w, do, dv, g, gk, scale, dht_out,
                emit_h=True, emit_m=True) -> torch.Tensor:
        T, K = q.shape[1], q.shape[3]
        HV, V = do.shape[2], do.shape[-1]
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

        # See the forward wrapper: the unconditional `.float().contiguous()`
        # was four ATen dispatches per call that could not do anything, since
        # the dummies are fp32-contiguous and the live gate arrives fp32 from
        # `chunk_local_cumsum`.
        g = _normalize_gate(g, self._dummy_g)
        gk = _normalize_gate(gk, self._dummy_gk)

        if dht_out is None:
            # Only the top rank has no source to fold, so only it leaves the
            # output untouched.  (This used to key on `post_count == 0`; with
            # the full chain every other rank runs the merge, and a rank whose
            # sources contribute nothing gets zeros through their emit masks.)
            if self.rank == self.R - 1:
                if self._last_zero is None:
                    self._last_zero = torch.zeros(
                        HV,
                        K,
                        V,
                        dtype=torch.float32,
                        device=self.device,
                    )
                dht_out = self._last_zero
            else:
                dht_out = torch.empty(
                    HV,
                    K,
                    V,
                    dtype=torch.float32,
                    device=self.device,
                )
        else:
            assert dht_out.shape == (HV, K, V)
            assert dht_out.dtype == torch.float32
            assert dht_out.is_contiguous()

        self.step += 1
        args = (
            self._p(q, cutlass.BFloat16),
            self._p(k, cutlass.BFloat16),
            self._p(w, cutlass.BFloat16),
            self._p(do, cutlass.BFloat16),
            self._p(dv, cutlass.BFloat16),
            self._p(g, cutlass.Float32),
            self._p(gk, cutlass.Float32),
            self._p(self.bufs.hm_sym, cutlass.Float32),
            self._p(self.ptrs, cutlass.Int64),
            self._p(dht_out, cutlass.Float32),
            cutlass.Float32(scale),
            cutlass.Int32(original_T),
            cutlass.Int32(self.step),
            cutlass.Int32((1 if emit_h else 0) | (2 if emit_m else 0)),
            self._cu_stream(),
        )
        geometry = self._op_key(original_T)
        op = self._ops.get(geometry)
        if op is None:
            op = self._make_op(geometry)
            self._ops[geometry] = op
        compiled = self._compiled.get(geometry)
        if compiled is None:
            compiled = cute.compile(op, *args)
            self._compiled[geometry] = compiled
        compiled(*args)
        return dht_out


_FLA_WRAPPER_CACHE: dict[tuple, CuteDSLFusedCPBwdPreProcess] = {}


def _producer_slice(
    tensor: torch.Tensor | None,
    bos: int,
    eos: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[:, bos:eos].contiguous()


def _first_local_sequence_bounds(
    cu_seqlens: torch.Tensor | None,
    context,
    T: int,
) -> tuple[int, int]:
    if cu_seqlens is None:
        return 0, T
    cpu = getattr(context, "cu_seqlens_cpu", None)
    if cpu is None:
        cpu = cu_seqlens.detach().cpu()
    if len(cpu) < 2:
        raise ValueError("cu_seqlens must contain at least two offsets")
    bos, eos = int(cpu[0]), int(cpu[1])
    if not (0 <= bos < eos <= T):
        raise ValueError(
            f"invalid first local sequence [{bos}, {eos}) for T={T}"
        )
    return bos, eos


@torch.no_grad()
def chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl_fused(
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
    wrapper_cls: type | None = None,
) -> tuple[torch.Tensor, None] | tuple[
    torch.Tensor | None, torch.Tensor | None
]:
    """FLA-signature adapter around :class:`CuteDSLFusedCPBwdPreProcess`.

    ``wrapper_cls`` selects the compute engine; it defaults to the
    non-warp-specialized wrapper.  The warp-specialized Blackwell backward
    passes :class:`~fused_gdr_cp_cute.bwd_ws.CuteDSLFusedCPBwdPreProcessWS`.
    """
    if wrapper_cls is None:
        wrapper_cls = CuteDSLFusedCPBwdPreProcess
    if context is None or context.group is None:
        return dht, initial_state
    if dht is not None:
        raise AssertionError("When CP is enabled, dht must be None")
    if initial_state is not None:
        raise AssertionError("CP backward expects initial_state=None")
    if bg is not None:
        raise NotImplementedError(
            "fused CuTeDSL backward does not support DPLR bg"
        )
    if chunk_size != BT:
        raise NotImplementedError(
            "fused CuTeDSL backward requires BT=64"
        )
    if g is not None and gk is not None:
        raise ValueError("g and gk are mutually exclusive")

    B, T, H, K = q.shape
    HV, V = do.shape[2], do.shape[-1]
    if B != 1:
        raise NotImplementedError(
            "fused CuTeDSL backward currently requires B=1"
        )
    if scale is None:
        scale = K ** -0.5
    mode = (
        "g"
        if g is not None
        else ("gk" if gk is not None else "none")
    )
    from .backend import _emit_flags

    emit_h, emit_m = _emit_flags(context, forward=False)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    canonical = torch.zeros(
        N,
        HV,
        K,
        V,
        dtype=torch.float32,
        device=q.device,
    )

    bos, eos = _first_local_sequence_bounds(cu_seqlens, context, T)
    group = context.group
    device_index = q.device.index
    key = (
        id(group),
        H,
        HV,
        K,
        V,
        mode,
        device_index,
        wrapper_cls,
    )
    fused = _FLA_WRAPPER_CACHE.get(key)
    if fused is None:
        fused = wrapper_cls(
            group,
            H,
            HV,
            K,
            V,
            gate_mode=mode,
            device=q.device,
        )
        _FLA_WRAPPER_CACHE[key] = fused

    fused(
        _producer_slice(q, bos, eos),
        _producer_slice(k, bos, eos),
        _producer_slice(w, bos, eos),
        _producer_slice(do, bos, eos),
        _producer_slice(dv, bos, eos),
        g=_producer_slice(g, bos, eos),
        gk=_producer_slice(gk, bos, eos),
        scale=scale,
        dht_out=canonical[-1],
        emit_h=emit_h,
        emit_m=emit_m,
    )

    if state_v_first:
        return canonical.transpose(-1, -2).contiguous(), None
    return canonical, None


# Candidate harnesses expect the frozen FLA entry-point name.
chunk_gated_delta_rule_bwd_dhu_pre_process = (
    chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl_fused
)
