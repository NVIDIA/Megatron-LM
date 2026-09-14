# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""CuteDSL fused CP pre-process: compute + P2P push + signal-driven merge.

Same architecture and wire protocol as the original fused CP prototype (see its
docstring for the full design: strip-major parity-buffered `hm_sym`, monotonic
uint32 `flags`, credit `acks`), with the verified CuteDSL compute kernel
(`pre_process.PreProcessFwdMerged`, 1.35x Triton at K=V=128) as the
compute role.

v2 additions (profile-guided, see docs/cutedsl_kernel_logic.md §7):

* **T-split compute** (`split=S`): the serial chunk scan of each (head, strip)
  is cut into S independent segments (each the affine characterization of its
  chunk sub-range; gates are chunk-local so boundaries need no adjustment).
  Grid becomes S*NSTRIPS compute CTAs per head. Segment s>0 stores its strip
  to a local scratch (`hm_seg`) and gpu-scope release-signals `seg_flags`;
  the segment-0 CTA keeps its result in the MMA accumulator, folds the later
  segments on top in time order
      he <- M_s @ he + he_s        (h-part strips)
      M  <- M_s @ M                (M-part strips)
  and then runs the unchanged push epilogue. hm_seg needs no parity: it is
  produced and consumed within one kernel launch (same-stream launches never
  overlap).

* **Warp-MMA fold** for both the compose above and the merge role, replacing
  the per-thread FMA loop (255 reg/thread + local spills, ~24 us/fold at
  K=128). The fold `out = M @ acc + add` is computed transposed,
  `out^T = acc^T @ M^T` — gemm1's exact shape: A = the fp32 C-fragment
  re-fed through the FA2 register relayout, B = M loaded straight from gmem
  fp32 via `thr_mma.partition_B` (no smem, no ldmatrix). Precision: bf16
  hi/lo 3-pass emulation (drop lo@lo), ~4e-6 rel vs fp64 (fla's TF32 merge:
  ~2.6e-4). Validated in dbg/probe_fold_mma.py.

grid.x layout per head:  [0, n_compute)                    compute segments
                          bx = seg*NSTRIPS + strip; seg 0 CTAs also compose+push
                         [n_compute, n_compute+NB)          merge CTAs

Restrictions: as parent (K in {64,128,256} single-GPU), fused path K <= 128;
gate modes none/g/gk; R <= 8; split*NSTRIPS*HV + NB*HV should stay <= #SMs
for full overlap (not a correctness requirement: producers retire).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from .pre_process import PreProcessFwdMerged
from .ragged import NeutralTailPadder

BT = 64


class PreProcessFwdFused(PreProcessFwdMerged):
    def __init__(self, H, HV, K, V, gate_mode, R, rank, split=1, B=1):
        super().__init__(H, HV, K, V, gate_mode=gate_mode, BT=BT)
        assert K <= 128, "fused merge stages full M_j via strips; K<=128 for now"
        assert 0 <= rank < R
        assert split >= 1
        assert B >= 1
        self.R, self.rank, self.S = R, rank, split
        # BS>1: independent sequences carried as a leading batch mode on the
        # symmetric payload/flag/ack tensors. B == 1 leaves every offset
        # byte-identical to the pre-batch kernel.
        self.B = B
        self.NB = self.NV
        self.n_compute = split * self.NSTRIPS if rank < R - 1 else 0
        self.n_merge = self.NB if rank > 0 else 0
        assert self.n_compute + self.n_merge > 0
        self.STRIP = self.K * BT  # elements per hm strip

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        pK: cute.Pointer,     # [1, T, H,  K] bf16
        pV: cute.Pointer,     # [1, T, HV, V] bf16 (u)
        pW: cute.Pointer,     # [1, T, HV, K] bf16
        pG: cute.Pointer,     # [1, T, HV] fp32 or dummy
        pGK: cute.Pointer,    # [1, T, HV, K] fp32 or dummy
        pHMloc: cute.Pointer,  # local hm_sym base, fp32 [2,R,HV,NSTRIPS,K,64]
        pPtrs: cute.Pointer,   # int64 [3, R]: rows = hm/fl/ak peer base addrs
        pH0: cute.Pointer,     # fp32 [HV, K, V] out
        pHMseg: cute.Pointer,  # fp32 [S-1, HV, NSTRIPS, K, 64] scratch (S>1)
        pSegFl: cute.Pointer,  # uint32 [S-1, HV, NSTRIPS] gpu-scope flags (S>1)
        T: cutlass.Int32,
        step: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        H, HV, K, V = self.H, self.HV, self.K, self.V
        mK = cute.make_tensor(
            pK, cute.make_layout((1, T, H, K), stride=(0, H * K, K, 1))
        )
        mV = cute.make_tensor(
            pV, cute.make_layout((1, T, HV, V), stride=(0, HV * V, V, 1))
        )
        mW = cute.make_tensor(
            pW, cute.make_layout((1, T, HV, K), stride=(0, HV * K, K, 1))
        )
        mG = cute.make_tensor(pG, cute.make_layout((1, T, HV), stride=(0, HV, 1)))
        mGK = cute.make_tensor(
            pGK, cute.make_layout((1, T, HV, K), stride=(0, HV * K, K, 1))
        )
        mPtrs = cute.make_tensor(
            pPtrs, cute.make_layout((3, self.R), stride=(self.R, 1))
        )
        mH0 = cute.make_tensor(
            pH0, cute.make_layout((HV, K, V), stride=(K * V, V, 1))
        )
        dtype = mK.element_type

        sw_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        sWK_layout = cute.tile_to_shape(sw_atom, (self.BT, self.K), (0, 1))
        sV_layout = cute.tile_to_shape(sw_atom, (self.BT, 64), (0, 1))
        # fold M tiles: plain row-padded (not swizzled) — the convert loop's
        # per-element swizzle math dominated the latency-bound fold otherwise.
        # Row stride K+8 bf16 = 272 B: 16B-aligned rows, and the 4-bank shift
        # per row spreads the 8-row ldmatrix groups across all banks.
        sM_layout = cute.make_layout((self.K, self.K), stride=(self.K + 8, 1))
        sv_size = cute.cosize(sV_layout)

        @cute.struct
        class SharedStorage:
            sW0: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sWK_layout)], 1024
            ]
            sW1: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sWK_layout)], 1024
            ]
            sW2: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sWK_layout)], 1024
            ]
            sK0: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sWK_layout)], 1024
            ]
            sK1: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sWK_layout)], 1024
            ]
            sK2: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sWK_layout)], 1024
            ]
            sVa0: cute.struct.Align[cute.struct.MemRange[dtype, sv_size], 1024]
            sVa1: cute.struct.Align[cute.struct.MemRange[dtype, sv_size], 1024]
            sVa2: cute.struct.Align[cute.struct.MemRange[dtype, sv_size], 1024]
            # g gates staged as a [BT, g_slab_hv] slab (cp.async atoms are
            # 128-bit); the per-head column is read via a strided view
            sG0: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.BT * self.g_slab_hv if self.use_g else 8
                ], 16
            ]
            sG1: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.BT * self.g_slab_hv if self.use_g else 8
                ], 16
            ]
            sG2: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.BT * self.g_slab_hv if self.use_g else 8
                ], 16
            ]
            sGk0: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, self.K if self.use_gk else 8
                ], 16
            ]
            sGk1: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, self.K if self.use_gk else 8
                ], 16
            ]
            sGk2: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, self.K if self.use_gk else 8
                ], 16
            ]
            # fold staging: bf16 hi/lo swizzled [K, K] tiles of the multiplier
            # M (merge role and seg-0 compose; compute tiles are dead by then)
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
            dtype, num_bits_per_copy=128,
        )
        elems = 128 // dtype.width
        thr_k = self.K // elems
        tiled_copy_WK = cute.make_tiled_copy_tv(
            atom_g2s,
            cute.make_layout((self.num_threads // thr_k, thr_k), stride=(thr_k, 1)),
            cute.make_layout((1, elems)),
        )
        thr_v = 64 // elems
        tiled_copy_V = cute.make_tiled_copy_tv(
            atom_g2s,
            cute.make_layout((128 // thr_v, thr_v), stride=(thr_v, 1)),
            cute.make_layout((1, elems)),
        )
        # flat fp32 cp.async copy for staging a fold multiplier M into smem
        atom_g2s_f32 = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            cutlass.Float32, num_bits_per_copy=128,
        )
        tiled_copy_M = cute.make_tiled_copy_tv(
            atom_g2s_f32, cute.make_layout(128), cute.make_layout(4)
        )
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
            (4, 1, 1),
            permutation_mnk=(64, 16, 16),
        )

        self.fused_kernel(
            mK, mV, mW, mG, mGK, mPtrs, mH0, T, step,
            pHMloc, pHMseg, pSegFl,
            sWK_layout, sV_layout, sM_layout, tiled_copy_WK, tiled_copy_V,
            tiled_copy_M, tiled_mma, SharedStorage,
        ).launch(
            grid=[self.n_compute + self.n_merge, self.HV, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    # ------------------------------------------------------- device helpers
    @cute.jit
    def _spin_ge(self, base_addr: cutlass.Int64, elem_off, expected,
                 scope: cutlass.Constexpr):
        """Spin ld.acquire.<scope> on a uint32 flag until it is >= expected."""
        p = cute.make_ptr(cutlass.Uint32, base_addr, cute.AddressSpace.gmem,
                          assumed_align=4) + elem_off
        v = cute.arch.load(p.llvm_ptr, cutlass.Uint32, sem="acquire", scope=scope)
        while v < expected:
            v = cute.arch.load(p.llvm_ptr, cutlass.Uint32, sem="acquire",
                               scope=scope)

    @cute.jit
    def _spin_ge_sys(self, base_addr: cutlass.Int64, elem_off, expected):
        self._spin_ge(base_addr, elem_off, expected, "sys")

    @cute.jit
    def _signal_relaxed_sys(self, base_addr: cutlass.Int64, elem_off, val):
        """Relaxed sys-scope flag store. Every st.release.sys lowers to its
        own MEMBAR.ALL.SYS (~3.7 us serialized, ncu-measured); a signal burst
        instead uses ONE preceding fence_acq_rel_sys + relaxed stores (PTX
        release pattern: fence;st.relaxed pairs with the spins' ld.acquire).
        Callers own the fence."""
        p = cute.make_ptr(cutlass.Uint32, base_addr, cute.AddressSpace.gmem,
                          assumed_align=4) + elem_off
        cute.arch.store(p.llvm_ptr, val, sem="relaxed", scope="sys")

    @cute.jit
    def _spin_ge_gpu_ptr(self, pbase: cute.Pointer, elem_off, expected):
        """Spin ld.acquire.gpu on a uint32 flag reached via a cute.Pointer."""
        p = pbase + elem_off
        v = cute.arch.load(p.llvm_ptr, cutlass.Uint32, sem="acquire", scope="gpu")
        while v < expected:
            v = cute.arch.load(p.llvm_ptr, cutlass.Uint32, sem="acquire",
                               scope="gpu")

    @cute.jit
    def _signal_gpu_ptr(self, pbase: cute.Pointer, elem_off, val):
        p = pbase + elem_off
        cute.arch.store(p.llvm_ptr, val, sem="release", scope="gpu")

    @cute.jit
    def _spin_producer_sys(self, fl_loc: cutlass.Int64, j, hv, b, stepu,
                           batch=0):
        """Wait for producer rank j's strip b + all its M strips (sys scope)."""
        jb = (j * self.B + batch) * self.HV + hv
        self._spin_ge_sys(fl_loc, jb * self.NSTRIPS + b, stepu)
        for s in cutlass.range_constexpr(self.NK):
            self._spin_ge_sys(
                fl_loc, jb * self.NSTRIPS + self.NV + s, stepu
            )

    @cute.jit
    def _spin_segment_gpu(self, pSegFl: cute.Pointer, s, hv, strip, stepu):
        """Wait for segment s's own strip + all its M strips (gpu scope)."""
        self._spin_ge_gpu_ptr(
            pSegFl, ((s - 1) * self.HV + hv) * self.NSTRIPS + strip, stepu
        )
        for kb in cutlass.range_constexpr(self.NK):
            self._spin_ge_gpu_ptr(
                pSegFl, ((s - 1) * self.HV + hv) * self.NSTRIPS + self.NV + kb,
                stepu,
            )

    def _hm_off(self, par, r, hv, strip, batch=0):
        """Element offset of strip (par, r, batch, hv, strip) in hm_sym.

        hm_sym is [2, RV, B, HV, NSTRIPS, K, 64]; batch == 0 with B == 1 gives
        the pre-batch offset exactly."""
        rb = ((par * self.R + r) * self.B + batch) * self.HV + hv
        return (rb * self.NSTRIPS + strip) * self.STRIP

    def _seg_off(self, s, hv, strip):
        """Element offset of strip (s, hv, strip) in hm_seg (s in [1, S))."""
        return (((s - 1) * self.HV + hv) * self.NSTRIPS + strip) * self.STRIP

    @cute.jit
    def _stage_issue(self, pBsrc, mRaw, tiled_copy_M, tidx):
        """cp.async the NK contiguous fp32 M strips gmem -> smem raw tile.

        Strip offsets are 4*STRIP-byte multiples, so the 16B alignment lost to
        pointer arithmetic is re-asserted for the 128-bit cp.async atom.
        Caller commits the group."""
        mSrc = cute.make_tensor(pBsrc.align(16),
                                cute.make_layout(self.NK * self.STRIP))
        thr = tiled_copy_M.get_slice(tidx)
        cute.copy(tiled_copy_M, thr.partition_S(mSrc), thr.partition_D(mRaw))

    @cute.jit
    def _stage_he_issue(self, pHe, mRawHe, tiled_copy_M, tidx):
        """cp.async one fp32 he strip ([K, 64] row-major) gmem -> smem.

        The fold's additive init otherwise costs 64 scattered gmem loads per
        thread on the critical path. Caller commits the group."""
        mSrc = cute.make_tensor(pHe.align(16), cute.make_layout(self.STRIP))
        thr = tiled_copy_M.get_slice(tidx)
        cute.copy(tiled_copy_M, thr.partition_S(mSrc), thr.partition_D(mRawHe))

    @cute.jit
    def _convert_raw(self, mRaw, sMhi, sMlo, tidx):
        """smem raw fp32 [NK*STRIP] -> bf16 hi/lo row-padded [K, K] tiles.

        flat = s*STRIP + n*64 + j; v4 runs never cross a row or strip. The
        padded tiles are flat-indexed (off = n*(K+8) + k) so the store address
        is one mad + increment — no per-element layout math."""
        dtype = self.io_dtype
        NVEC = self.NK * self.STRIP // (128 * 4)
        mRaw4 = cute.make_tensor(
            mRaw.iterator, cute.make_layout((4, 128 * NVEC), stride=(1, 4))
        )
        sMhiF = cute.make_tensor(
            sMhi.iterator, cute.make_layout(self.K * (self.K + 8))
        )
        sMloF = cute.make_tensor(
            sMlo.iterator, cute.make_layout(self.K * (self.K + 8))
        )
        for e in cutlass.range_constexpr(NVEC):
            i4 = tidx + e * 128
            src = mRaw4[None, i4]
            rv = cute.make_fragment_like(src, cutlass.Float32)
            rv.store(src.load())
            f0 = i4 * 4
            n = (f0 % self.STRIP) // 64
            kbase = (f0 // self.STRIP) * 64 + f0 % 64
            off = n * (self.K + 8) + kbase
            for q in cutlass.range_constexpr(4):
                val = rv[q]
                hi = val.to(dtype)
                sMhiF[off + q] = hi
                sMloF[off + q] = (val - hi.to(cutlass.Float32)).to(dtype)

    @cute.jit
    def _snapshot_and_init(self, thr_mma, acc, rAhi, rAlo, mAddT, use_add):
        """Snapshot acc into bf16 hi/lo A fragments, then re-init acc with the
        additive strip (a cheap smem view) or zero. use_add may be runtime."""
        dtype = self.io_dtype
        rAhi.store(acc.load().to(dtype))
        rAlo.store((acc.load() - rAhi.load().to(cutlass.Float32)).to(dtype))
        tCa = thr_mma.partition_C(mAddT)
        for i in cutlass.range_constexpr(cute.size(acc)):
            acc[i] = cutlass.Float32(0.0)
            if use_add:
                acc[i] = tCa[i]

    @cute.jit
    def _fold_gemms(self, thr_mma, tiled_mma, acc, rAhi, rAlo, sMhi, sMlo,
                    tidx):
        """acc += (M @ old_acc_strip)^T where old_acc is the rAhi/rAlo bf16
        hi/lo snapshot and M is staged in the sMhi/sMlo swizzled tiles
        (B-operand layout: n=K rows, k=K cols; barrier-published).

        gemm1's exact B path (ldmatrix per 16-wide k-block); bf16 hi/lo 3-pass
        emulation (Ahi@Bhi, Alo@Bhi, Ahi@Blo), ~4e-6 rel err vs fp64
        (dbg/probe_fold_mma.py). No barriers inside — callers own the
        stage/convert/fold pipeline synchronization."""
        dtype = self.io_dtype
        rAhi_g2 = cute.make_tensor(
            rAhi.iterator, cute.logical_divide(rAhi.layout, (None, None, 2))
        )
        rAlo_g2 = cute.make_tensor(
            rAlo.iterator, cute.logical_divide(rAlo.layout, (None, None, 2))
        )
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            dtype,
        )
        s_tiled_B = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
        s_thr_B = s_tiled_B.get_slice(tidx)
        tSsBhi = s_thr_B.partition_S(sMhi)
        tSsBlo = s_thr_B.partition_S(sMlo)
        tPB = thr_mma.partition_B(sMhi)
        tPB_g = cute.make_tensor(
            tPB.iterator, cute.logical_divide(tPB.layout, (None, None, 1))
        )
        tBrBh = thr_mma.make_fragment_B(tPB_g[None, None, (None, 0)])
        tBrBh_cv = s_thr_B.retile(tBrBh)
        tBrBl = thr_mma.make_fragment_B(tPB_g[None, None, (None, 0)])
        tBrBl_cv = s_thr_B.retile(tBrBl)
        for kb in cutlass.range_constexpr(self.K // 16):
            aHi_src = rAhi_g2[None, None, (None, kb)]
            aLo_src = rAlo_g2[None, None, (None, kb)]
            rAh = cute.make_fragment_like(aHi_src, dtype)
            rAl = cute.make_fragment_like(aLo_src, dtype)
            rAh.store(aHi_src.load())
            rAl.store(aLo_src.load())
            rAhv = self._a_view(rAh)
            rAlv = self._a_view(rAl)
            cute.copy(s_tiled_B, tSsBhi[None, None, kb], tBrBh_cv[None, None, 0])
            cute.copy(s_tiled_B, tSsBlo[None, None, kb], tBrBl_cv[None, None, 0])
            cute.gemm(tiled_mma, acc, rAhv[None, None, 0], tBrBh[None, None, 0], acc)
            cute.gemm(tiled_mma, acc, rAlv[None, None, 0], tBrBh[None, None, 0], acc)
            cute.gemm(tiled_mma, acc, rAhv[None, None, 0], tBrBl[None, None, 0], acc)

    # --------------------------------------------------------------- kernel
    @cute.kernel
    def fused_kernel(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mW: cute.Tensor,
        mG: cute.Tensor,
        mGK: cute.Tensor,
        mPtrs: cute.Tensor,     # int64 (3, R): hm/fl/ak peer base addrs
        mH0: cute.Tensor,       # (HV, K, V) fp32
        T: cutlass.Int32,
        step: cutlass.Int32,
        pHMloc: cute.Pointer,
        pHMseg: cute.Pointer,
        pSegFl: cute.Pointer,
        sWK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sM_layout: cute.Layout,
        tiled_copy_WK: cute.TiledCopy,
        tiled_copy_V: cute.TiledCopy,
        tiled_copy_M: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, hv, _ = cute.arch.block_idx()
        dtype = mK.element_type
        R, rank, S = self.R, self.rank, self.S
        par = step % 2
        stepu = step.to(cutlass.Uint32)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sMhi = storage.sMhi.get_tensor(sM_layout)
        sMlo = storage.sMlo.get_tensor(sM_layout)
        # raw fp32 staging tiles for the fold pipeline, overlaid on the compute
        # W/K smem tiles (dead in the merge role; dead after the scan in the
        # seg-0 compose). M strips + he strip = (NK+1)*STRIP fp32 (96 KB at
        # K=128) == the six W/K tiles exactly.
        pRaw = cute.recast_ptr(
            storage.sW0.get_tensor(cute.make_layout(1)).iterator,
            dtype=cutlass.Float32,
        )
        mRaw = cute.make_tensor(pRaw, cute.make_layout(self.NK * self.STRIP))
        mRawHe = cute.make_tensor(
            pRaw + self.NK * self.STRIP, cute.make_layout(self.STRIP)
        )

        # transposed strip view layout (64, K):(1, 64)
        stripT_layout = cute.make_layout((64, self.K), stride=(1, 64))
        # fold additive operand: the staged he strip, viewed transposed
        mAddT_smem = cute.make_tensor(mRawHe.iterator, stripT_layout)

        if bidx < self.n_compute:
            # ============ compute role: one segment of one hm strip ==========
            seg = bidx // self.NSTRIPS
            strip = bidx % self.NSTRIPS
            is_h = strip < self.NV
            hk = hv // (self.HV // self.H)
            NT = cute.ceil_div(T, self.BT)
            # segment chunk range [c0, c1) (empty segments fall through to an
            # identity affine map: zero he / identity M from the init)
            NT_per = cute.ceil_div(NT, S)
            c0 = seg * NT_per
            c1 = cutlass.min(NT, c0 + NT_per)

            mW_h = mW[0, None, hv, None]
            mK_h = mK[0, None, hk, None]
            mV_h = mV[0, None, hv, None]

            sW = [storage.sW0.get_tensor(sWK_layout),
                  storage.sW1.get_tensor(sWK_layout),
                  storage.sW2.get_tensor(sWK_layout)]
            sK = [storage.sK0.get_tensor(sWK_layout),
                  storage.sK1.get_tensor(sWK_layout),
                  storage.sK2.get_tensor(sWK_layout)]
            sV0 = [storage.sVa0.get_tensor(sV_layout),
                   storage.sVa1.get_tensor(sV_layout),
                   storage.sVa2.get_tensor(sV_layout)]
            sV1 = sV0
            # g slabs ([BT, g_slab_hv] flat, cp.async dst) + per-head views
            g_layout = cute.make_layout(
                self.BT * self.g_slab_hv if self.use_g else 8
            )
            gk_layout = cute.make_layout(self.K if self.use_gk else 8)
            sGs = [storage.sG0.get_tensor(g_layout),
                   storage.sG1.get_tensor(g_layout),
                   storage.sG2.get_tensor(g_layout)]
            if cutlass.const_expr(self.use_g):
                g_view = cute.make_layout(self.BT, stride=self.g_slab_hv)
                sG = [cute.make_tensor(b.iterator + hv % self.g_slab_hv,
                                       g_view) for b in sGs]
            else:
                sG = sGs
            sGkLast = [storage.sGk0.get_tensor(gk_layout),
                       storage.sGk1.get_tensor(gk_layout),
                       storage.sGk2.get_tensor(gk_layout)]
            kt_layout = cute.make_layout((self.K, self.BT), stride=(self.BT, 1))
            sKt = [cute.composition(sK[0], kt_layout),
                   cute.composition(sK[1], kt_layout),
                   cute.composition(sK[2], kt_layout)]

            thr_copy_WK = tiled_copy_WK.get_slice(tidx)
            thr_copy_V = tiled_copy_V.get_slice(tidx)
            tWsW = [thr_copy_WK.partition_D(b) for b in sW]
            tKsK = [thr_copy_WK.partition_D(b) for b in sK]
            tV0sV = [thr_copy_V.partition_D(b) for b in sV0]
            tV1sV = tV0sV

            thr_mma = tiled_mma.get_slice(tidx)
            hT = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((64, self.K)), cutlass.Float32
            )
            hT_g2 = cute.make_tensor(
                hT.iterator, cute.logical_divide(hT.layout, (None, None, 2))
            )
            hT_mn = self._mn_view(hT)
            tCcC_mn = self._mn_view(
                thr_mma.partition_C(cute.make_identity_tensor((64, 64)))
            )
            tCcS_mn = self._mn_view(
                thr_mma.partition_C(cute.make_identity_tensor((64, self.K)))
            )
            nrows = cute.size(tCcC_mn, mode=[0])
            ncols = cute.size(tCcC_mn, mode=[1])
            ncols_k = cute.size(tCcS_mn, mode=[1])

            hT.fill(0.0)
            if not is_h:
                cc0 = (strip - self.NV) * 64
                for r in cutlass.range_constexpr(nrows):
                    for c in cutlass.range_constexpr(ncols_k):
                        crd = tCcS_mn[r, c]
                        if crd[1] == cc0 + crd[0]:
                            hT_mn[r, c] = cutlass.Float32(1.0)

            smem_copy_atom = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                dtype,
            )
            smem_copy_atom_t = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                dtype,
            )
            s_tiled_W = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
            s_tiled_Kt = cute.make_tiled_copy_B(smem_copy_atom_t, tiled_mma)
            s_thr_W = s_tiled_W.get_slice(tidx)
            s_thr_Kt = s_tiled_Kt.get_slice(tidx)
            tSsW = [s_thr_W.partition_S(b) for b in sW]
            tSsKt = [s_thr_Kt.partition_S(b) for b in sKt]
            tPBW = thr_mma.partition_B(sW[0])
            tPBW_g = cute.make_tensor(
                tPBW.iterator, cute.logical_divide(tPBW.layout, (None, None, 1))
            )
            tBrW = thr_mma.make_fragment_B(tPBW_g[None, None, (None, 0)])
            tBrW_cv = s_thr_W.retile(tBrW)
            tPBK = thr_mma.partition_B(sKt[0])
            tPBK_g = cute.make_tensor(
                tPBK.iterator, cute.logical_divide(tPBK.layout, (None, None, 1))
            )
            tBrKt = thr_mma.make_fragment_B(tPBK_g[None, None, (None, 0)])
            tBrKt_cv = s_thr_Kt.retile(tBrKt)

            acc1 = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((64, 64)), cutlass.Float32
            )
            acc1_mn = self._mn_view(acc1)
            lam = cute.make_rmem_tensor(ncols, cutlass.Float32)

            # -------- sequential scan over this segment's chunks ------------
            # (empty segments — NT < S — must not leave cp.async in flight)
            if c0 < c1:
                self._issue_chunk(c0, is_h, 0, strip, T, mW_h, mK_h, mV_h,
                                  tiled_copy_WK, tiled_copy_V,
                                  thr_copy_WK, thr_copy_V,
                                  tWsW[0], tKsK[0], tV0sV[0], tV1sV[0])
                self._load_gates(c0, T, tidx, hv, mG, mGK, sGs[0], sGkLast[0])
            cute.arch.cp_async_commit_group()
            if c0 + 1 < c1:
                self._issue_chunk(c0 + 1, is_h, 0, strip, T, mW_h, mK_h, mV_h,
                                  tiled_copy_WK, tiled_copy_V,
                                  thr_copy_WK, thr_copy_V,
                                  tWsW[1], tKsK[1], tV0sV[1], tV1sV[1])
                self._load_gates(c0 + 1, T, tidx, hv, mG, mGK, sGs[1],
                                 sGkLast[1])
            cute.arch.cp_async_commit_group()
            nt_local = cutlass.max(c1 - c0, 0)
            for i3 in cutlass.range(cute.ceil_div(nt_local, 3)):
                c = c0 + i3 * 3
                self._step(c, is_h, 0, strip, T, c1, tidx, hv,
                           nrows, ncols, ncols_k,
                           mG, mGK, mW_h, mK_h, mV_h,
                           tiled_copy_WK, tiled_copy_V, thr_copy_WK, thr_copy_V,
                           tiled_mma, s_tiled_W, s_tiled_Kt,
                           tSsW[0], tSsKt[0], sV0[0], sV1[0], sG[0], sGkLast[0],
                           tWsW[2], tKsK[2], tV0sV[2], tV1sV[2], sGs[2],
                           sGkLast[2],
                           tBrW, tBrW_cv, tBrKt, tBrKt_cv,
                           hT, hT_g2, hT_mn, acc1, acc1_mn, lam, tCcC_mn, tCcS_mn)
                if c + 1 < c1:
                    self._step(c + 1, is_h, 0, strip, T, c1, tidx, hv,
                               nrows, ncols, ncols_k,
                               mG, mGK, mW_h, mK_h, mV_h,
                               tiled_copy_WK, tiled_copy_V, thr_copy_WK, thr_copy_V,
                               tiled_mma, s_tiled_W, s_tiled_Kt,
                               tSsW[1], tSsKt[1], sV0[1], sV1[1], sG[1], sGkLast[1],
                               tWsW[0], tKsK[0], tV0sV[0], tV1sV[0], sGs[0],
                               sGkLast[0],
                               tBrW, tBrW_cv, tBrKt, tBrKt_cv,
                               hT, hT_g2, hT_mn, acc1, acc1_mn, lam,
                               tCcC_mn, tCcS_mn)
                    if c + 2 < c1:
                        self._step(c + 2, is_h, 0, strip, T, c1, tidx, hv,
                                   nrows, ncols, ncols_k,
                                   mG, mGK, mW_h, mK_h, mV_h,
                                   tiled_copy_WK, tiled_copy_V,
                                   thr_copy_WK, thr_copy_V,
                                   tiled_mma, s_tiled_W, s_tiled_Kt,
                                   tSsW[2], tSsKt[2], sV0[2], sV1[2], sG[2],
                                   sGkLast[2],
                                   tWsW[1], tKsK[1], tV0sV[1], tV1sV[1], sGs[1],
                                   sGkLast[1],
                                   tBrW, tBrW_cv, tBrKt, tBrKt_cv,
                                   hT, hT_g2, hT_mn, acc1, acc1_mn, lam,
                                   tCcC_mn, tCcS_mn)

            if cutlass.const_expr(S > 1):
                if seg > 0:
                    # ==== segment s>0: store strip locally, gpu-release flag ==
                    dseg = pHMseg + self._seg_off(seg, hv, strip)
                    gSeg = cute.make_tensor(dseg, stripT_layout)
                    tSegC = thr_mma.partition_C(gSeg)
                    for i in cutlass.range_constexpr(cute.size(hT)):
                        tSegC[i] = hT[i]
                    cute.arch.barrier()
                    cute.arch.fence_acq_rel_gpu()
                    if tidx == 0:
                        self._signal_gpu_ptr(
                            pSegFl,
                            ((seg - 1) * self.HV + hv) * self.NSTRIPS + strip,
                            stepu,
                        )
                else:
                    # ==== segment 0: compose the later segments on top =======
                    # stage(s+1) is issued under fold(s)'s gemms; the raw tile
                    # overlays the scan's W/K smem, so drain scan prefetches.
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()
                    if tidx == 0:
                        self._spin_segment_gpu(pSegFl, 1, hv, strip, stepu)
                    cute.arch.barrier()
                    self._stage_issue(pHMseg + self._seg_off(1, hv, self.NV),
                                      mRaw, tiled_copy_M, tidx)
                    self._stage_he_issue(
                        pHMseg + self._seg_off(1, hv, strip),
                        mRawHe, tiled_copy_M, tidx)
                    cute.arch.cp_async_commit_group()
                    rAhi = cute.make_fragment_like(hT, self.io_dtype)
                    rAlo = cute.make_fragment_like(hT, self.io_dtype)
                    for s in cutlass.range_constexpr(1, S):
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.barrier()  # raw M+he (s) landed, visible
                        self._convert_raw(mRaw, sMhi, sMlo, tidx)
                        self._snapshot_and_init(thr_mma, hT, rAhi, rAlo,
                                                mAddT_smem, is_h)
                        if cutlass.const_expr(s + 1 < S):
                            if tidx == 0:
                                self._spin_segment_gpu(pSegFl, s + 1, hv,
                                                       strip, stepu)
                        cute.arch.barrier()  # hi/lo published; raw reads done
                        if cutlass.const_expr(s + 1 < S):
                            self._stage_issue(
                                pHMseg + self._seg_off(s + 1, hv, self.NV),
                                mRaw, tiled_copy_M, tidx)
                            self._stage_he_issue(
                                pHMseg + self._seg_off(s + 1, hv, strip),
                                mRawHe, tiled_copy_M, tidx)
                            cute.arch.cp_async_commit_group()
                        self._fold_gemms(thr_mma, tiled_mma, hT, rAhi, rAlo,
                                         sMhi, sMlo, tidx)

            if seg == 0:
                # ====== epilogue: ack-throttle -> push to consumers -> signal
                ackv = cutlass.max(step - 2, 0).to(cutlass.Uint32)
                ak_loc = mPtrs[2, rank]
                if tidx == 0:
                    for cc in cutlass.range_constexpr(rank + 1, R):
                        for b in cutlass.range_constexpr(self.NB):
                            self._spin_ge_sys(
                                ak_loc, (cc * self.HV + hv) * self.NB + b, ackv
                            )
                cute.arch.barrier()
                for j in cutlass.range_constexpr(rank + 1, R):
                    hm_j = mPtrs[0, j]
                    dptr = cute.make_ptr(
                        cutlass.Float32, hm_j, cute.AddressSpace.gmem,
                        assumed_align=16,
                    ) + self._hm_off(par, rank, hv, strip)
                    gBlk = cute.make_tensor(dptr, stripT_layout)
                    tBlk = thr_mma.partition_C(gBlk)
                    for i in cutlass.range_constexpr(cute.size(hT)):
                        tBlk[i] = hT[i]
                cute.arch.barrier()
                cute.arch.fence_acq_rel_sys()
                if tidx == 0:
                    # relaxed stores: the single fence above forms the release
                    # pattern for the whole flag burst (vs one implicit
                    # MEMBAR.ALL.SYS per st.release.sys)
                    for j in cutlass.range_constexpr(rank + 1, R):
                        self._signal_relaxed_sys(
                            mPtrs[1, j],
                            (rank * self.HV + hv) * self.NSTRIPS + strip, stepu,
                        )
        else:
            # ========== merge role: h0[hv][:, 64b:64b+64], warp-MMA fold =====
            b = bidx - self.n_compute
            fl_loc = mPtrs[1, rank]
            thr_mma = tiled_mma.get_slice(tidx)
            macc = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((64, self.K)), cutlass.Float32
            )

            # j = 0: h = he_0 (rank 0's M is never needed), staged through smem
            if tidx == 0:
                self._spin_ge_sys(fl_loc, (0 * self.HV + hv) * self.NSTRIPS + b,
                                  stepu)
            cute.arch.barrier()  # publish the acquire to the whole CTA
            self._stage_he_issue(pHMloc + self._hm_off(par, 0, hv, b),
                                 mRawHe, tiled_copy_M, tidx)
            cute.arch.cp_async_commit_group()
            rAhi = cute.make_fragment_like(macc, self.io_dtype)
            rAlo = cute.make_fragment_like(macc, self.io_dtype)
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            tHe0 = thr_mma.partition_C(mAddT_smem)
            for i in cutlass.range_constexpr(cute.size(macc)):
                macc[i] = tHe0[i]
            cute.arch.barrier()  # he_0 reads done before he_1 restages

            if cutlass.const_expr(rank > 1):
                if tidx == 0:
                    self._spin_producer_sys(fl_loc, 1, hv, b, stepu)
                cute.arch.barrier()
                self._stage_issue(pHMloc + self._hm_off(par, 1, hv, self.NV),
                                  mRaw, tiled_copy_M, tidx)
                self._stage_he_issue(pHMloc + self._hm_off(par, 1, hv, b),
                                     mRawHe, tiled_copy_M, tidx)
                cute.arch.cp_async_commit_group()
            for j in cutlass.range_constexpr(1, rank):
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()  # raw M+he (j) landed, visible CTA-wide
                self._convert_raw(mRaw, sMhi, sMlo, tidx)
                self._snapshot_and_init(thr_mma, macc, rAhi, rAlo,
                                        mAddT_smem, True)
                if cutlass.const_expr(j + 1 < rank):
                    if tidx == 0:
                        self._spin_producer_sys(fl_loc, j + 1, hv, b, stepu)
                cute.arch.barrier()  # hi/lo published; raw reads done
                if cutlass.const_expr(j + 1 < rank):
                    self._stage_issue(
                        pHMloc + self._hm_off(par, j + 1, hv, self.NV),
                        mRaw, tiled_copy_M, tidx)
                    self._stage_he_issue(
                        pHMloc + self._hm_off(par, j + 1, hv, b),
                        mRawHe, tiled_copy_M, tidx)
                    cute.arch.cp_async_commit_group()
                self._fold_gemms(thr_mma, tiled_mma, macc, rAhi, rAlo,
                                 sMhi, sMlo, tidx)

            # h0[hv, k, 64b + c] transposed block view (64, K):(1, V)
            gH0 = cute.make_tensor(
                mH0.iterator + (hv * self.K * self.V + b * 64),
                cute.make_layout((64, self.K), stride=(1, self.V)),
            )
            tH0 = thr_mma.partition_C(gH0)
            for i in cutlass.range_constexpr(cute.size(macc)):
                tH0[i] = macc[i]
            cute.arch.barrier()
            if tidx == 0:
                # one fence + relaxed ack burst (the old per-ack
                # st.release.sys membars were ~26% of the merge kernel)
                cute.arch.fence_acq_rel_sys()
                for j in cutlass.range_constexpr(rank):
                    self._signal_relaxed_sys(
                        mPtrs[2, j], (rank * self.HV + hv) * self.NB + b, stepu
                    )


# ---------------------------------------------------------------------- host
def _ptr(t: torch.Tensor, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


class CuteDSLFusedCPPreProcess:
    """Host wrapper mirroring tilelang_fused.FusedCPPreProcess."""

    def __init__(self, group, H, HV, DK, DV, gate_mode="g", device=None,
                 split=1, auto_split=True):
        import torch.distributed as dist

        from .symm import CPSymmBuffers

        self.group = group
        self.R = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.H, self.HV, self.DK, self.DV = H, HV, DK, DV
        self.gate_mode = gate_mode
        self.split = split          # max S; buffers are sized for this
        # dispatch S by T: the compose tail costs ~10 us per exposed fold
        # (T-independent), the scan ~1.3 us/chunk, so small shards want a
        # smaller S (profile model: crossovers at NT~16 for S=2, NT~64 for
        # S=4; see docs/ncu_optimization_roadmap.md item 2)
        self.auto_split = auto_split
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.bufs = CPSymmBuffers(group, HV, DK, DV, self.device)
        # int64 [3, R]: peer base addresses for hm/flags/acks
        self.ptrs = torch.stack(
            [self.bufs.hm_ptrs, self.bufs.fl_ptrs, self.bufs.ak_ptrs]
        ).contiguous()
        NSTRIPS = DK // 64 + DV // 64
        if split > 1:
            self.hm_seg = torch.empty(
                split - 1, HV, NSTRIPS, DK, BT,
                dtype=torch.float32, device=self.device,
            )
            self.seg_flags = torch.zeros(
                split - 1, HV, NSTRIPS, dtype=torch.uint32, device=self.device
            )
        else:  # dummies (never dereferenced when S == 1)
            self.hm_seg = torch.zeros(1, dtype=torch.float32, device=self.device)
            self.seg_flags = torch.zeros(1, dtype=torch.uint32, device=self.device)
        self.step = 0
        self._compiled = {}          # effective S -> compiled kernel
        # host-overhead caches: make_ptr costs ~2 us/arg; the zeroed rank-0 h0
        # is never written by the kernel so one shared buffer is safe to alias
        self._ptr_cache = {}
        self._h0_zero = None
        self._tail_padder = NeutralTailPadder(BT)
        # The inactive gate still occupies a positional kernel argument.
        # Keep stable device allocations instead of allocating and zeroing a
        # tiny CUDA tensor on every steady-state call.
        self._dummy_g = torch.zeros(
            1, 1, 1, dtype=torch.float32, device=self.device
        )
        self._dummy_gk = torch.zeros(
            1, 1, 1, 1, dtype=torch.float32, device=self.device
        )

    def _pick_split(self, T):
        """Effective S for this shard length (seg buffers fit any S <= max;
        seg_flags stay monotonic across S switches since unused slots simply
        keep stale, smaller step values). Also capped so the grid stays
        within one wave: T-split only converts idle SMs into scan
        parallelism — once (S*NSTRIPS + NB)*HV exceeds the SM count it adds
        compose work (and intra-launch seg-flag spins across waves) for
        nothing."""
        if not self.auto_split or self.split == 1:
            return self.split
        NSTRIPS = self.DK // 64 + self.DV // 64
        NB = self.DV // 64
        n_sm = torch.cuda.get_device_properties(self.device).multi_processor_count
        s_cap = max((n_sm // self.HV - NB) // NSTRIPS, 1)
        NT = (T + BT - 1) // BT
        if NT < 16:
            s_auto = 1
        elif NT < 64:
            s_auto = 2
        else:
            s_auto = self.split
        return min(s_auto, s_cap, self.split)

    def _p(self, t, dtype):
        key = (t.data_ptr(), dtype)
        p = self._ptr_cache.get(key)
        if p is None:
            p = make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem,
                         assumed_align=16)
            if len(self._ptr_cache) < 4096:
                self._ptr_cache[key] = p
        return p

    @torch.no_grad()
    def __call__(self, k, v, w, g=None, gk=None, h0_out=None):
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
            if (
                not tensor.is_cuda
                or not tensor.is_contiguous()
                or tensor.device != self.device
            ):
                raise ValueError(
                    f"{name} must be contiguous on {self.device}"
                )
        if g is not None and gk is not None:
            raise ValueError("g and gk are mutually exclusive")
        supplied_mode = (
            "g" if g is not None else ("gk" if gk is not None else "none")
        )
        if supplied_mode != self.gate_mode:
            raise ValueError(
                f"wrapper gate_mode={self.gate_mode!r}, "
                f"got {supplied_mode!r} inputs"
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
        if h0_out is None:
            if self.rank == 0:
                # never written by the kernel (no merge CTAs): alias one buffer
                if self._h0_zero is None:
                    self._h0_zero = torch.zeros(
                        self.HV, self.DK, self.DV,
                        dtype=torch.float32, device=self.device)
                h0_out = self._h0_zero
            else:
                # the merge role overwrites every element: skip the zero-fill
                h0_out = torch.empty(self.HV, self.DK, self.DV,
                                     dtype=torch.float32, device=self.device)
        elif (
            tuple(h0_out.shape) != (self.HV, self.DK, self.DV)
            or h0_out.dtype != torch.float32
            or not h0_out.is_cuda
            or not h0_out.is_contiguous()
            or h0_out.device != self.device
        ):
            raise ValueError(
                "h0_out must be contiguous fp32 [HV,K,V] on "
                f"{self.device}"
            )
        if g is not None:
            g = g.float().contiguous()
        if gk is not None:
            gk = gk.float().contiguous()
        # The inherited clamped-window tail path is direct for T>=64. Tiny
        # sequences still issue one 64-row cp.async tile, so provide physical
        # backing while retaining the original logical T for its validity mask.
        if T < BT:
            k = self._tail_padder.pad(k, name="k")
            v = self._tail_padder.pad(v, name="v")
            w = self._tail_padder.pad(w, name="w")
            if g is not None:
                g = self._tail_padder.pad(g, name="g", edge=True)
            if gk is not None:
                gk = self._tail_padder.pad(gk, name="gk", edge=True)
        if g is None:
            g = self._dummy_g
        if gk is None:
            gk = self._dummy_gk

        self.step += 1
        args = (
            self._p(k, cutlass.BFloat16), self._p(v, cutlass.BFloat16),
            self._p(w, cutlass.BFloat16), self._p(g, cutlass.Float32),
            self._p(gk, cutlass.Float32),
            self._p(self.bufs.hm_sym, cutlass.Float32),
            self._p(self.ptrs, cutlass.Int64),
            self._p(h0_out, cutlass.Float32),
            self._p(self.hm_seg, cutlass.Float32),
            self._p(self.seg_flags, cutlass.Uint32),
            cutlass.Int32(T), cutlass.Int32(self.step),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        S = self._pick_split(T)
        fn = self._compiled.get(S)
        if fn is None:
            op = PreProcessFwdFused(self.H, self.HV, self.DK, self.DV,
                                    self.gate_mode, self.R, self.rank,
                                    split=S)
            fn = cute.compile(op, *args)
            self._compiled[S] = fn
        fn(*args)
        return h0_out
