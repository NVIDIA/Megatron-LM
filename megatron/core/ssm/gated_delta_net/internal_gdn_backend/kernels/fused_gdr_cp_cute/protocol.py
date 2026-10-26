# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""Wire protocol and fold helpers shared by the two warp-specialized CP engines.

The forward (:mod:`.fused_ws`) and backward (:mod:`.bwd_ws`) boundary
pre-process kernels differ in their compute role but share everything that
touches the peer protocol: the symmetric-memory strip layout (`_hm_off`), the
sys-scope flag/ack spins and signals, the cp.async staging of a producer's
summary into shared memory, the fp32 -> bf16 hi/lo split and the SM80
warp-MMA fold ``acc += (M @ acc_strip)^T`` (3-pass hi/lo emulation, ~4e-6 rel
vs fp64), the FA2 accumulator->A-operand register relayout, and the tcgen05
gemm issue helper.  They used to inherit these from the non-warp-specialized
SM80 engines; :class:`CPProtocolMixin` is that inheritance made explicit, with
the geometry fields both engines read.

Payload layout (see :mod:`.symm`): ``hm_sym [2, R, HV, NSTRIPS, K, 64]`` fp32,
strip-major and parity-buffered on ``step % 2``; ``flags [R, HV, NSTRIPS]`` and
``acks [R, HV, NV]`` are monotonic uint32 step ids.  Strips ``< NV`` carry the
``h`` (forward) / ``Htilde`` (backward) half of a rank's summary, strips
``>= NV`` the transition ``M`` / ``P``.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05

BT = 64


def tmem_alloc_ptr(field):
    """SMEM pointer for ``TmemAllocator``'s ``alloc_result_dst_smem_ptr``.

    The shape of a scalar ``@cute.struct`` field changed across CuTeDSL
    releases: 4.6.0.dev0 hands back a ``_ScalarData`` wrapper that exposes the
    pointer as ``.ptr``, while newer releases hand back the ``_Pointer``
    itself, where ``.ptr`` raises
    ``AttributeError: '_Pointer' object has no attribute 'ptr'``.
    Both are accepted here so one source tree runs on either toolchain --
    hardcoding either form breaks the other.
    """
    return getattr(field, "ptr", field)


class CPProtocolMixin:
    """Geometry fields and device helpers of the CP boundary wire protocol.

    ``gate_mode`` selects the decay the compute role applies per chunk:
    ``"none"``, scalar ``"g"`` (GDN) or per-key ``"gk"`` (KDA); the engines
    branch on ``use_g`` / ``use_gk``.  The fold helpers below are gate-agnostic.
    """

    def __init__(self, H, HV, K, V, gate_mode, R, rank):
        assert K == V and K in (64, 128), "CP engines need K == V in {64, 128}"
        assert HV % H == 0
        assert gate_mode in ("none", "g", "gk")
        assert 0 <= rank < R
        self.H, self.HV, self.K, self.V = H, HV, K, V
        self.use_g = gate_mode == "g"
        self.use_gk = gate_mode == "gk"
        self.NV = V // 64  # h-part column strips
        self.NK = K // 64  # M-part column strips
        self.NSTRIPS = self.NV + self.NK
        self.NB = self.NV  # merge CTAs: one per 64 output columns
        self.STRIP = K * BT  # elements per hm strip
        self.io_dtype = cutlass.BFloat16
        self.R, self.rank = R, rank

    # ---------------------------------------------------------- device utils
    def _a_view(self, rX: cute.Tensor) -> cute.Tensor:
        """Re-group a C-fragment-shaped register tensor into an A-operand view:
        (V, MMA_M, MMA_N) -> ((V, 2), MMA_M, MMA_N / 2). FA2:880-896."""
        ld = cute.logical_divide(rX.layout, (None, None, 2))
        view = cute.make_layout(
            ((ld.shape[0], ld.shape[2][0]), ld.shape[1], ld.shape[2][1]),
            stride=((ld.stride[0], ld.stride[2][0]), ld.stride[1], ld.stride[2][1]),
        )
        return cute.make_tensor(rX.iterator, view)

    # ------------------------------------------------------------- gemm util
    @cute.jit
    def _exec_gemm(
        self,
        tiled_mma,
        tAcc,
        tCrA,
        tCrB,
        a_stage,
        b_stage,
        acc_init: cutlass.Constexpr = False,
        kp_lo: cutlass.Constexpr = 0,
        kp_hi: cutlass.Constexpr = -1,
    ):
        n_kp = cute.size(tCrB, mode=[2])
        hi = n_kp if kp_hi < 0 else kp_hi
        for kpi in cutlass.range(hi - kp_lo, unroll_full=True):
            kp = kp_lo + kpi
            tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(acc_init or kp != 0))
            cute.gemm(
                tiled_mma, tAcc, tCrA[None, None, kp, a_stage], tCrB[None, None, kp, b_stage], tAcc
            )
        return tiled_mma

    # ------------------------------------------------------- flags and acks
    @cute.jit
    def _spin_ge(self, base_addr: cutlass.Int64, elem_off, expected, scope: cutlass.Constexpr):
        """Spin ld.acquire.<scope> on a uint32 flag until it is >= expected."""
        p = (
            cute.make_ptr(cutlass.Uint32, base_addr, cute.AddressSpace.gmem, assumed_align=4)
            + elem_off
        )
        v = cute.arch.load(p.llvm_ptr, cutlass.Uint32, sem="acquire", scope=scope)
        while v < expected:
            v = cute.arch.load(p.llvm_ptr, cutlass.Uint32, sem="acquire", scope=scope)

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
        p = (
            cute.make_ptr(cutlass.Uint32, base_addr, cute.AddressSpace.gmem, assumed_align=4)
            + elem_off
        )
        cute.arch.store(p.llvm_ptr, val, sem="relaxed", scope="sys")

    @cute.jit
    def _spin_producer_sys(self, fl_loc: cutlass.Int64, j, hv, b, stepu):
        """Wait for producer rank j's strip b + all its M strips (sys scope)."""
        jb = j * self.HV + hv
        self._spin_ge_sys(fl_loc, jb * self.NSTRIPS + b, stepu)
        for s in cutlass.range_constexpr(self.NK):
            self._spin_ge_sys(fl_loc, jb * self.NSTRIPS + self.NV + s, stepu)

    def _hm_off(self, par, r, hv, strip):
        """Element offset of strip (par, r, hv, strip) in hm_sym
        ([2, R, HV, NSTRIPS, K, 64])."""
        rb = (par * self.R + r) * self.HV + hv
        return (rb * self.NSTRIPS + strip) * self.STRIP

    # ------------------------------------------------------------ fold chain
    @cute.jit
    def _stage_issue(self, pBsrc, mRaw, tiled_copy_M, tidx):
        """cp.async the NK contiguous fp32 M strips gmem -> smem raw tile.

        Strip offsets are 4*STRIP-byte multiples, so the 16B alignment lost to
        pointer arithmetic is re-asserted for the 128-bit cp.async atom.
        Caller commits the group."""
        mSrc = cute.make_tensor(pBsrc.align(16), cute.make_layout(self.NK * self.STRIP))
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
        mRaw4 = cute.make_tensor(mRaw.iterator, cute.make_layout((4, 128 * NVEC), stride=(1, 4)))
        sMhiF = cute.make_tensor(sMhi.iterator, cute.make_layout(self.K * (self.K + 8)))
        sMloF = cute.make_tensor(sMlo.iterator, cute.make_layout(self.K * (self.K + 8)))
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
    def _fold_gemms(self, thr_mma, tiled_mma, acc, rAhi, rAlo, sMhi, sMlo, tidx):
        """acc += (M @ old_acc_strip)^T where old_acc is the rAhi/rAlo bf16
        hi/lo snapshot and M is staged in the sMhi/sMlo swizzled tiles
        (B-operand layout: n=K rows, k=K cols; barrier-published).

        gemm1's exact B path (ldmatrix per 16-wide k-block); bf16 hi/lo 3-pass
        emulation (Ahi@Bhi, Alo@Bhi, Ahi@Blo), ~4e-6 rel err vs fp64
        (dbg/probe_fold_mma.py). No barriers inside — callers own the
        stage/convert/fold pipeline synchronization."""
        dtype = self.io_dtype
        rAhi_g2 = cute.make_tensor(rAhi.iterator, cute.logical_divide(rAhi.layout, (None, None, 2)))
        rAlo_g2 = cute.make_tensor(rAlo.iterator, cute.logical_divide(rAlo.layout, (None, None, 2)))
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), dtype
        )
        s_tiled_B = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
        s_thr_B = s_tiled_B.get_slice(tidx)
        tSsBhi = s_thr_B.partition_S(sMhi)
        tSsBlo = s_thr_B.partition_S(sMlo)
        tPB = thr_mma.partition_B(sMhi)
        tPB_g = cute.make_tensor(tPB.iterator, cute.logical_divide(tPB.layout, (None, None, 1)))
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
