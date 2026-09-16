# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""CuteDSL (CUTLASS Python DSL) rewrite of fla's `pre_process_fwd_kernel_merged`.

Computes, per value-head, the affine characterization (h_e, M) of the gated
delta rule recurrence over a sequence shard (see docs/pre_process_fwd_math.md):

    h_out = M @ h_in + h_e,     hm[hv] = [h_e | M]  in R^{K x (V+K)}, fp32.

Implementation notes
--------------------
The kernel keeps the state *transposed* relative to the Triton original: a
64-wide column strip of hm is owned by one 4-warp group, which holds the fp32
accumulator hT = (strip x K) in registers across the sequential chunk loop.
Per chunk t (BT = 64 timesteps) it runs two chained SM80-style warp MMAs
(bf16 in, fp32 accum; `mma.sync.m16n8k16`, valid on sm_80..sm_100+):

    gemm1: acc1 = hT_bf16 @ W_t^T      (A = converted state accumulator,
                                        B = W tile from smem, ldmatrix)
    x     = (V_t^T - acc1) * lambda    (h-part)      [elementwise, fp32]
          = (    - acc1) * lambda      (M-part)
    hT   *= gamma (USE_G) / diag(2^gk_last) (USE_GK)
    gemm2: hT += x_bf16 @ K_t          (A = converted acc1,
                                        B = K tile from smem, ldmatrix.trans)

h-part strips (strip < V/64) and M-part strips share this skeleton; the
M-part initializes hT to the identity strip and skips the V term. The
accumulator->A-operand register relayout is the FlashAttention-2 example trick
(cutlass/examples/python/CuTeDSL/cute/ampere/kernel/attention/
flash_attention_v2.py:875-919); the C-fragment of one m16n8k16 MMA is bitwise
re-groupable into the A-fragment of the next.

Performance shape (the kernel is a serial scan, so per-chunk *latency* is
everything):
  * w/k/v tiles are triple-buffered in smem with prefetch distance two
    (the equivalent of Triton num_stages=3), one barrier per chunk; the loop
    is peeled by three so the buffer index stays compile-time.
  * The ragged last chunk reuses the unpredicated copy path through a clamped
    tile window [T-BT, T); its dead columns are zeroed by the lambda mask
    (a large predicated copy path would make the loop i-cache-bound).
  * A "paired" mode (two strips per 256-thread CTA sharing the head's w/k
    tiles) is implemented but disabled — measured slower on B200; see
    __init__.

Restrictions vs the Triton kernel: K in {64,128,256}, V % 64 == 0, BT=64,
B=1 (single shard, MULTI_SEQS/varlen-slice handled by the caller passing T),
DPLR (USE_BG) not implemented. Gating modes: none / g (GDN) / gk (KDA).
(K=192 currently trips a CuteDSL compiler crash and is rejected.)
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_ptr


class PreProcessFwdMerged:
    def __init__(self, H: int, HV: int, K: int, V: int, gate_mode: str = "g",
                 BT: int = 64):
        assert K in (64, 128, 256), "K must be in {64, 128, 256}"
        assert V % 64 == 0, "V must be a multiple of 64"
        assert HV % H == 0
        assert BT == 64
        assert gate_mode in ("none", "g", "gk")
        self.H, self.HV, self.K, self.V = H, HV, K, V
        self.BT = BT
        self.use_g = gate_mode == "g"
        self.use_gk = gate_mode == "gk"
        self.NV = V // 64            # h-part column strips
        self.NK = K // 64            # M-part column strips
        self.NSTRIPS = self.NV + self.NK
        # Two strips per CTA (256 threads) is supported but measured slower
        # than one strip per CTA on B200 (the barrier-lockstepped halves
        # contend on smem/LSU instead of overlapping), so it stays off.
        self.paired = False
        self.num_threads = 256 if self.paired else 128
        self.io_dtype = cutlass.BFloat16
        # g gates are cp.async-staged (128-bit atoms): when HV % 4 == 0 only
        # the 4-head group holding this CTA's head is staged ([BT, 4] tile,
        # one 16B row-atom, HV-independent 1 KB); otherwise the full [BT, HV]
        # slab (contiguous; small HV only, or smem would blow up at HV=64)
        self.g_slab_hv = 4 if HV % 4 == 0 else HV

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        pK: cute.Pointer,   # [1, T, H,  K] bf16
        pV: cute.Pointer,   # [1, T, HV, V] bf16 (GDN/KDA: this is u)
        pW: cute.Pointer,   # [1, T, HV, K] bf16
        pG: cute.Pointer,   # [1, T, HV] fp32 (log2, chunk-local cumsum) or dummy
        pGK: cute.Pointer,  # [1, T, HV, K] fp32 or dummy
        pHM: cute.Pointer,  # [HV, K, V+K] fp32 out
        T: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # Tensors are built from raw pointers with static strides (B = 1, so
        # the batch stride is irrelevant) — raw pointers keep the per-call
        # host overhead ~2us/arg instead of ~19us/arg for from_dlpack.
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
        mG = cute.make_tensor(
            pG, cute.make_layout((1, T, HV), stride=(0, HV, 1))
        )
        mGK = cute.make_tensor(
            pGK, cute.make_layout((1, T, HV, K), stride=(0, HV * K, K, 1))
        )
        mHM = cute.make_tensor(
            pHM, cute.make_layout((HV, K, V + K), stride=(K * (V + K), V + K, 1))
        )
        dtype = mK.element_type

        # Swizzled smem layouts for bf16 (rows, cols) tiles; 64-col atom,
        # identical to the FA2 example (flash_attention_v2.py:226-244).
        sw_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        sWK_layout = cute.tile_to_shape(sw_atom, (self.BT, self.K), (0, 1))
        sV_layout = cute.tile_to_shape(sw_atom, (self.BT, 64), (0, 1))
        sv_size = cute.cosize(sV_layout)
        sv1_size = sv_size if self.paired else 8  # dummy when unpaired

        @cute.struct
        class SharedStorage:
            # three pipeline stages per operand tile (prefetch distance 2);
            # w/k are shared by both strips of a paired CTA, v is per strip.
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
            sVb0: cute.struct.Align[cute.struct.MemRange[dtype, sv1_size], 1024]
            sVb1: cute.struct.Align[cute.struct.MemRange[dtype, sv1_size], 1024]
            sVb2: cute.struct.Align[cute.struct.MemRange[dtype, sv1_size], 1024]
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

        # gmem -> smem cp.async tiled copies, 128-bit per thread. w/k tiles
        # use all threads of the CTA; v tiles use one 128-thread strip group.
        atom_g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype, num_bits_per_copy=128,
        )
        elems = 128 // dtype.width  # 8 for bf16
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

        # 4 warps stacked on M (FA2 config): M tile 64, ldmatrix-friendly perm.
        # NOTE: the atom layout must keep N un-partitioned (N-atom = 1), or the
        # accumulator->A-operand relayout would not hold.
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
            (4, 1, 1),
            permutation_mnk=(64, 16, 16),
        )

        ctas = (self.NSTRIPS + 1) // 2 if self.paired else self.NSTRIPS
        self.kernel(
            mK, mV, mW, mG, mGK, mHM, T,
            sWK_layout, sV_layout, tiled_copy_WK, tiled_copy_V, tiled_mma,
            SharedStorage,
        ).launch(
            grid=[ctas, self.HV, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    # ---------------------------------------------------------- device utils
    def _mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """(V, MMA_M, MMA_N) accumulator fragment -> (M, N) logical view.

        Verbatim from the FA2 example (_make_acc_tensor_mn_view)."""
        col = cute.make_layout(acc.layout.shape)
        mn = cute.make_layout(
            ((col.shape[0][1], col.shape[1]), (col.shape[0][0], col.shape[2])),
            stride=(
                (col.stride[0][1], col.stride[1]),
                (col.stride[0][0], col.stride[2]),
            ),
        )
        return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))

    def _a_view(self, rX: cute.Tensor) -> cute.Tensor:
        """Re-group a C-fragment-shaped register tensor into an A-operand view:
        (V, MMA_M, MMA_N) -> ((V, 2), MMA_M, MMA_N / 2). FA2:880-896."""
        ld = cute.logical_divide(rX.layout, (None, None, 2))
        view = cute.make_layout(
            ((ld.shape[0], ld.shape[2][0]), ld.shape[1], ld.shape[2][1]),
            stride=((ld.stride[0], ld.stride[2][0]), ld.stride[1], ld.stride[2][1]),
        )
        return cute.make_tensor(rX.iterator, view)

    # -------------------------------------------------- pipeline stage bodies
    @cute.jit
    def _issue_chunk(self, i_t, is_h, half, strip, T,
                     mW_h, mK_h, mV_h,
                     tiled_copy_WK, tiled_copy_V, thr_copy_WK, thr_copy_V,
                     dW, dK, dV0, dV1):
        """cp.async loads of chunk i_t's w/k(/v) tiles into the smem dst
        partitions dW/dK/dV{0,1}.

        The last (ragged) chunk's tile window is shifted up to [T-BT, T) so it
        stays in bounds; the duplicated rows it reads are dead data — every
        consumer masks them out (see the `lam` mask in _step). The host
        zero-pads inputs to at least BT rows so the clamp is always legal.
        Keeping a single unpredicated copy path keeps the hot loop's code
        small (the kernel is i-cache-bound otherwise)."""
        t_ofs = cutlass.max(cutlass.min(i_t * self.BT, T - self.BT), 0)
        gW = cute.local_tile(
            cute.domain_offset((t_ofs, 0), mW_h), (self.BT, self.K), (0, 0)
        )
        gK = cute.local_tile(
            cute.domain_offset((t_ofs, 0), mK_h), (self.BT, self.K), (0, 0)
        )
        cute.copy(tiled_copy_WK, thr_copy_WK.partition_S(gW), dW)
        cute.copy(tiled_copy_WK, thr_copy_WK.partition_S(gK), dK)
        if is_h:
            gV = cute.local_tile(
                cute.domain_offset((t_ofs, 0), mV_h), (self.BT, 64), (0, strip)
            )
            sSrc = thr_copy_V.partition_S(gV)
            if cutlass.const_expr(self.paired):
                if half == 0:
                    cute.copy(tiled_copy_V, sSrc, dV0)
                else:
                    cute.copy(tiled_copy_V, sSrc, dV1)
            else:
                cute.copy(tiled_copy_V, sSrc, dV0)

    @cute.jit
    def _load_gates(self, i_t, T, tidx, hv, mG, mGK, sGb, sGkb):
        """cp.async the g / gk_last gate values of chunk i_t into the smem
        gate buffers. Called two chunks ahead, inside the same commit group as
        the w/k/v tiles, so the gmem latency rides the tile pipeline (the old
        plain LDG->STS pair was the hottest stall in the scan: the in-warp
        dependent store defeated the distance-2 prefetch).

        The g row uses the same clamped window [t_ofs, t_ofs+BT) as the tiles
        (always in bounds given the host's T >= BT padding); the read sites in
        _step undo the shift. Rows past T land in smem but are never read."""
        t0 = i_t * self.BT
        atom_f32 = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32, num_bits_per_copy=128,
        )
        if cutlass.const_expr(self.use_g):
            # cp.async atoms are 128-bit, so a whole [BT, g_slab_hv] slab is
            # staged (chunk starts are 64-row multiples -> row atoms stay
            # 16B-aligned); _step reads head hv's column through a strided
            # view of it.
            t_ofs = cutlass.max(cutlass.min(t0, T - self.BT), 0)
            if cutlass.const_expr(self.g_slab_hv == 4 and self.HV != 4):
                # [BT, 4] tile of this CTA's 4-head group, 1 atom per row
                tiled_copy_G = cute.make_tiled_copy_tv(
                    atom_f32,
                    cute.make_layout((self.BT, 1), stride=(1, 0)),
                    cute.make_layout((1, 4)),
                )
                hv0 = (hv // 4) * 4
                pSlab = (mG.iterator + t_ofs * self.HV + hv0).align(16)
                mSrc = cute.make_tensor(
                    pSlab, cute.make_layout((self.BT, 4), stride=(self.HV, 1))
                )
                mDst = cute.make_tensor(
                    sGb.iterator, cute.make_layout((self.BT, 4), stride=(4, 1))
                )
                if tidx < self.BT:
                    thr_cg = tiled_copy_G.get_slice(tidx)
                    cute.copy(tiled_copy_G, thr_cg.partition_S(mSrc),
                              thr_cg.partition_D(mDst))
            else:
                # full contiguous [BT, HV] slab (small HV / HV % 4 != 0)
                n_el = self.BT * self.HV
                thr_g = min(n_el // 4, 128)
                tiled_copy_G = cute.make_tiled_copy_tv(
                    atom_f32, cute.make_layout(thr_g),
                    cute.make_layout(n_el // thr_g),
                )
                pSlab = (mG.iterator + t_ofs * self.HV).align(16)
                mSrc = cute.make_tensor(pSlab, cute.make_layout(n_el))
                if tidx < thr_g:
                    thr_cg = tiled_copy_G.get_slice(tidx)
                    cute.copy(tiled_copy_G, thr_cg.partition_S(mSrc),
                              thr_cg.partition_D(sGb))
        if cutlass.const_expr(self.use_gk):
            # one contiguous K-row (the chunk's last valid gk row)
            thr_gk = min(self.K // 4, 128)
            tiled_copy_GK = cute.make_tiled_copy_tv(
                atom_f32, cute.make_layout(thr_gk),
                cute.make_layout(self.K // thr_gk),
            )
            last_idx = t0 + cutlass.min(T - t0, self.BT) - 1
            pRow = (mGK.iterator
                    + (last_idx * self.HV + hv) * self.K).align(16)
            mSrcK = cute.make_tensor(pRow, cute.make_layout(self.K))
            if tidx < thr_gk:
                thr_ck = tiled_copy_GK.get_slice(tidx)
                cute.copy(tiled_copy_GK, thr_ck.partition_S(mSrcK),
                          thr_ck.partition_D(sGkb))

    @cute.jit
    def _step(self, i_t, is_h, half, strip, T, NT, tidx, hv,
              nrows: cutlass.Constexpr, ncols: cutlass.Constexpr,
              ncols_k: cutlass.Constexpr,
              mG, mGK, mW_h, mK_h, mV_h,
              tiled_copy_WK, tiled_copy_V, thr_copy_WK, thr_copy_V,
              tiled_mma, s_tiled_W, s_tiled_Kt,
              # current buffer (i_t % 3)
              tSsW, tSsKt, sV0b, sV1b, sG, sGkLast,
              # prefetch buffer ((i_t + 2) % 3)
              nWsW, nKsK, nV0sV, nV1sV, nG, nGk,
              # register-resident tensors
              tBrW, tBrW_cv, tBrKt, tBrKt_cv,
              hT, hT_g2, hT_mn, acc1, acc1_mn, lam, tCcC_mn, tCcS_mn):
        """One chunk of the recurrence, computed from smem buffer i_t % 3;
        prefetches chunk i_t+2 into buffer (i_t+2) % 3."""
        dtype = self.io_dtype
        t0 = i_t * self.BT
        n_valid = T - t0

        # Chunk i_t's loads must be complete; chunk i_t+1's group (committed
        # one step earlier) may stay in flight.
        cute.arch.cp_async_wait_group(1)
        # One barrier per step: it both publishes chunk i_t's smem tiles and
        # gates to all threads, and guarantees every thread has finished
        # reading buffer (i_t+2)%3 (last used by chunk i_t-1) before the
        # prefetch below overwrites it.
        cute.arch.barrier()

        # ---- prefetch chunk i_t+2 (distance-2 pipeline)
        if i_t + 2 < NT:
            self._issue_chunk(i_t + 2, is_h, half, strip, T, mW_h, mK_h, mV_h,
                              tiled_copy_WK, tiled_copy_V,
                              thr_copy_WK, thr_copy_V,
                              nWsW, nKsK, nV0sV, nV1sV)
            self._load_gates(i_t + 2, T, tidx, hv, mG, mGK, nG, nGk)
        cute.arch.cp_async_commit_group()

        # sG holds the clamped window [t_ofs, t_ofs+BT): window position n is
        # global timestep t_ofs+n, so chunk-local index idx lives at
        # sG[idx + shift] with shift = t0 - t_ofs (nonzero only for the ragged
        # last chunk, where the window is pulled back to stay in bounds).
        shift = t0 - cutlass.max(cutlass.min(t0, T - self.BT), 0)
        # last valid gate of the chunk, from smem (kept off the critical
        # gmem path)
        g_last = cutlass.Float32(0.0)
        if cutlass.const_expr(self.use_g):
            g_last = sG[cutlass.min(n_valid, self.BT) - 1 + shift]

        # ---- gemm1: acc1 = hT(bf16) @ W_t^T
        acc1.fill(0.0)
        for kb in cutlass.range_constexpr(self.K // 16):
            hT_sl = hT_g2[None, None, (None, kb)]          # (V, M, 2)
            rA1 = cute.make_fragment_like(hT_sl, dtype)
            rA1.store(hT_sl.load().to(dtype))
            rA1v = self._a_view(rA1)                       # ((V,2), M, 1)
            cute.copy(s_tiled_W, tSsW[None, None, kb], tBrW_cv[None, None, 0])
            cute.gemm(
                tiled_mma, acc1,
                rA1v[None, None, 0], tBrW[None, None, 0], acc1,
            )

        # ---- middle elementwise: x = ([v^T] - acc1) * lambda
        # lam doubles as the time-validity mask. The loaded tile window is
        # [t_ofs, t_ofs+BT) with t_ofs = clamp(t0, 0, T-BT): window column n
        # holds global timestep t_ofs+n, so positions n < shift (rows
        # duplicated from the previous chunk by the ragged-window clamp) and
        # n >= shift + n_valid (zero-padded tail when T < BT) are dead and
        # multiplied by 0 here; the gate of window column n lives at sG[n]
        # (sG is loaded through the same clamped window as the tiles).
        for c in cutlass.range_constexpr(ncols):
            n = tCcC_mn[0, c][1]
            idx = n - shift
            lam[c] = cutlass.Float32(0.0)
            if idx >= 0:
                if idx < n_valid:
                    if cutlass.const_expr(self.use_g):
                        lam[c] = cute.math.exp2(g_last - sG[n], fastmath=True)
                    else:
                        lam[c] = cutlass.Float32(1.0)
        if is_h:
            if cutlass.const_expr(self.paired):
                if half == 0:
                    for r in cutlass.range_constexpr(nrows):
                        for c in cutlass.range_constexpr(ncols):
                            crd = tCcC_mn[r, c]
                            xv = sV0b[crd[1], crd[0]].to(cutlass.Float32) \
                                - acc1_mn[r, c]
                            acc1_mn[r, c] = xv * lam[c]
                else:
                    for r in cutlass.range_constexpr(nrows):
                        for c in cutlass.range_constexpr(ncols):
                            crd = tCcC_mn[r, c]
                            xv = sV1b[crd[1], crd[0]].to(cutlass.Float32) \
                                - acc1_mn[r, c]
                            acc1_mn[r, c] = xv * lam[c]
            else:
                for r in cutlass.range_constexpr(nrows):
                    for c in cutlass.range_constexpr(ncols):
                        crd = tCcC_mn[r, c]
                        xv = sV0b[crd[1], crd[0]].to(cutlass.Float32) \
                            - acc1_mn[r, c]
                        acc1_mn[r, c] = xv * lam[c]
        else:
            for r in cutlass.range_constexpr(nrows):
                for c in cutlass.range_constexpr(ncols):
                    xv = cutlass.Float32(0.0) - acc1_mn[r, c]
                    acc1_mn[r, c] = xv * lam[c]

        rA2 = cute.make_fragment_like(acc1, dtype)
        rA2.store(acc1.load().to(dtype))
        rA2v = self._a_view(rA2)                           # ((V,2), M, BT/16)

        # ---- decay the state
        if cutlass.const_expr(self.use_g):
            gamma = cute.math.exp2(g_last, fastmath=True)
            hT.store(hT.load() * gamma)
        if cutlass.const_expr(self.use_gk):
            for c in cutlass.range_constexpr(ncols_k):
                n = tCcS_mn[0, c][1]
                e = cute.math.exp2(sGkLast[n], fastmath=True)
                for r in cutlass.range_constexpr(nrows):
                    hT_mn[r, c] = hT_mn[r, c] * e

        # ---- gemm2: hT += x(bf16) @ K_t
        for kb in cutlass.range_constexpr(self.BT // 16):
            cute.copy(s_tiled_Kt, tSsKt[None, None, kb], tBrKt_cv[None, None, 0])
            cute.gemm(
                tiled_mma, hT,
                rA2v[None, None, kb], tBrKt[None, None, 0], hT,
            )

    # --------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mW: cute.Tensor,
        mG: cute.Tensor,
        mGK: cute.Tensor,
        mHM: cute.Tensor,
        T: cutlass.Int32,
        sWK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tiled_copy_WK: cute.TiledCopy,
        tiled_copy_V: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, hv, _ = cute.arch.block_idx()
        dtype = mK.element_type

        # Paired mode: two 4-warp groups per CTA, each owning one strip. An
        # odd trailing strip is duplicated by both halves (identical stores).
        if cutlass.const_expr(self.paired):
            lidx = tidx % 128
            half = tidx // 128
            strip = cutlass.min(bidx * 2 + half, self.NSTRIPS - 1)
        else:
            lidx = tidx
            half = tidx // 128  # always 0
            strip = bidx

        is_h = strip < self.NV                # h-part vs M-part strip
        hk = hv // (self.HV // self.H)        # GQA: key head for this v head
        NT = cute.ceil_div(T, self.BT)

        # Per-head 2D gmem views: (T, K) / (T, V)
        mW_h = mW[0, None, hv, None]
        mK_h = mK[0, None, hk, None]
        mV_h = mV[0, None, hv, None]

        # Shared memory (triple-buffered operand tiles)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sW = [storage.sW0.get_tensor(sWK_layout), storage.sW1.get_tensor(sWK_layout),
              storage.sW2.get_tensor(sWK_layout)]
        sK = [storage.sK0.get_tensor(sWK_layout), storage.sK1.get_tensor(sWK_layout),
              storage.sK2.get_tensor(sWK_layout)]
        sV0 = [storage.sVa0.get_tensor(sV_layout), storage.sVa1.get_tensor(sV_layout),
               storage.sVa2.get_tensor(sV_layout)]
        if cutlass.const_expr(self.paired):
            sV1 = [storage.sVb0.get_tensor(sV_layout),
                   storage.sVb1.get_tensor(sV_layout),
                   storage.sVb2.get_tensor(sV_layout)]
        else:
            sV1 = sV0
        # g slabs ([BT, g_slab_hv] flat, cp.async dst) + per-head read views
        g_layout = cute.make_layout(
            self.BT * self.g_slab_hv if self.use_g else 8
        )
        gk_layout = cute.make_layout(self.K if self.use_gk else 8)
        sGs = [storage.sG0.get_tensor(g_layout), storage.sG1.get_tensor(g_layout),
               storage.sG2.get_tensor(g_layout)]
        if cutlass.const_expr(self.use_g):
            g_view = cute.make_layout(self.BT, stride=self.g_slab_hv)
            sG = [cute.make_tensor(b.iterator + hv % self.g_slab_hv, g_view)
                  for b in sGs]
        else:
            sG = sGs
        sGkLast = [storage.sGk0.get_tensor(gk_layout), storage.sGk1.get_tensor(gk_layout),
                   storage.sGk2.get_tensor(gk_layout)]

        # Transposed views of sK: (K, BT) so k can be the (N=K_state, K=time)
        # B operand of gemm2 (same trick as FA2's sVt).
        kt_layout = cute.make_layout((self.K, self.BT), stride=(self.BT, 1))
        sKt = [cute.composition(sK[0], kt_layout), cute.composition(sK[1], kt_layout),
               cute.composition(sK[2], kt_layout)]

        # gmem copy partitions (smem side; the gmem side is partitioned per
        # chunk inside _issue_chunk from a dynamically offset view).
        # w/k copies use the whole CTA; v copies one 128-thread half.
        thr_copy_WK = tiled_copy_WK.get_slice(tidx)
        thr_copy_V = tiled_copy_V.get_slice(lidx)
        tWsW = [thr_copy_WK.partition_D(b) for b in sW]
        tKsK = [thr_copy_WK.partition_D(b) for b in sK]
        tV0sV = [thr_copy_V.partition_D(b) for b in sV0]
        tV1sV = [thr_copy_V.partition_D(b) for b in sV1]

        # MMA partitions (per 128-thread half)
        thr_mma = tiled_mma.get_slice(lidx)

        # Persistent transposed state: (M=64 strip, N=K) fp32 accumulator.
        hT = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((64, self.K)), cutlass.Float32
        )
        # Grouped view for slicing 16-wide k-dim blocks out of the state when
        # converting it to the A operand of gemm1: (V, M, (2, K/16)).
        hT_g2 = cute.make_tensor(
            hT.iterator, cute.logical_divide(hT.layout, (None, None, 2))
        )
        hT_mn = self._mn_view(hT)

        # Per-thread (m, n) coordinates of accumulator elements.
        tCcC_mn = self._mn_view(
            thr_mma.partition_C(cute.make_identity_tensor((64, 64)))
        )
        tCcS_mn = self._mn_view(
            thr_mma.partition_C(cute.make_identity_tensor((64, self.K)))
        )
        nrows = cute.size(tCcC_mn, mode=[0])
        ncols = cute.size(tCcC_mn, mode=[1])
        ncols_k = cute.size(tCcS_mn, mode=[1])

        # State init: 0 for the h-part; identity strip M_0 = I[:, c0:c0+64]
        # (transposed: hT[m, n] = 1 iff n == c0 + m) for the M-part.
        hT.fill(0.0)
        if not is_h:
            c0 = (strip - self.NV) * 64
            for r in cutlass.range_constexpr(nrows):
                for c in cutlass.range_constexpr(ncols_k):
                    crd = tCcS_mn[r, c]
                    if crd[1] == c0 + crd[0]:
                        hT_mn[r, c] = cutlass.Float32(1.0)

        # smem -> register B-operand copies (ldmatrix)
        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), dtype
        )
        smem_copy_atom_t = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), dtype
        )
        s_tiled_W = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
        s_tiled_Kt = cute.make_tiled_copy_B(smem_copy_atom_t, tiled_mma)
        s_thr_W = s_tiled_W.get_slice(lidx)
        s_thr_Kt = s_tiled_Kt.get_slice(lidx)
        tSsW = [s_thr_W.partition_S(b) for b in sW]
        tSsKt = [s_thr_Kt.partition_S(b) for b in sKt]
        # Single-k-block B fragments, reused across the k-block loops. The
        # k-block mode must be kept (fragment makers want 3-mode (V, N, K)),
        # so slice through a divide-by-1 of the k-block mode.
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

        # gemm1 accumulator: acc1[m, n] = (W_t @ h)[n, strip_base + m]
        acc1 = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((64, 64)), cutlass.Float32
        )
        acc1_mn = self._mn_view(acc1)

        lam = cute.make_rmem_tensor(ncols, cutlass.Float32)

        # ================= sequential scan over time chunks =================
        # Three-stage software pipeline (prefetch distance 2); the loop is
        # peeled by three chunks so the smem buffer index is compile-time.
        # _step's buffer block: compute buffer b, prefetch buffer (b + 2) % 3.
        self._issue_chunk(0, is_h, half, strip, T, mW_h, mK_h, mV_h,
                          tiled_copy_WK, tiled_copy_V, thr_copy_WK, thr_copy_V,
                          tWsW[0], tKsK[0], tV0sV[0], tV1sV[0])
        self._load_gates(0, T, tidx, hv, mG, mGK, sGs[0], sGkLast[0])
        cute.arch.cp_async_commit_group()
        if NT > 1:
            self._issue_chunk(1, is_h, half, strip, T, mW_h, mK_h, mV_h,
                              tiled_copy_WK, tiled_copy_V, thr_copy_WK, thr_copy_V,
                              tWsW[1], tKsK[1], tV0sV[1], tV1sV[1])
            self._load_gates(1, T, tidx, hv, mG, mGK, sGs[1], sGkLast[1])
        cute.arch.cp_async_commit_group()
        for i3 in cutlass.range(cute.ceil_div(NT, 3)):
            c = i3 * 3
            self._step(c, is_h, half, strip, T, NT, tidx, hv,
                       nrows, ncols, ncols_k,
                       mG, mGK, mW_h, mK_h, mV_h,
                       tiled_copy_WK, tiled_copy_V, thr_copy_WK, thr_copy_V,
                       tiled_mma, s_tiled_W, s_tiled_Kt,
                       tSsW[0], tSsKt[0], sV0[0], sV1[0], sG[0], sGkLast[0],
                       tWsW[2], tKsK[2], tV0sV[2], tV1sV[2], sGs[2], sGkLast[2],
                       tBrW, tBrW_cv, tBrKt, tBrKt_cv,
                       hT, hT_g2, hT_mn, acc1, acc1_mn, lam, tCcC_mn, tCcS_mn)
            if c + 1 < NT:
                self._step(c + 1, is_h, half, strip, T, NT, tidx, hv,
                           nrows, ncols, ncols_k,
                           mG, mGK, mW_h, mK_h, mV_h,
                           tiled_copy_WK, tiled_copy_V, thr_copy_WK, thr_copy_V,
                           tiled_mma, s_tiled_W, s_tiled_Kt,
                           tSsW[1], tSsKt[1], sV0[1], sV1[1], sG[1], sGkLast[1],
                           tWsW[0], tKsK[0], tV0sV[0], tV1sV[0], sGs[0], sGkLast[0],
                           tBrW, tBrW_cv, tBrKt, tBrKt_cv,
                           hT, hT_g2, hT_mn, acc1, acc1_mn, lam, tCcC_mn, tCcS_mn)
                if c + 2 < NT:
                    self._step(c + 2, is_h, half, strip, T, NT, tidx, hv,
                               nrows, ncols, ncols_k,
                               mG, mGK, mW_h, mK_h, mV_h,
                               tiled_copy_WK, tiled_copy_V, thr_copy_WK, thr_copy_V,
                               tiled_mma, s_tiled_W, s_tiled_Kt,
                               tSsW[2], tSsKt[2], sV0[2], sV1[2], sG[2], sGkLast[2],
                               tWsW[1], tKsK[1], tV0sV[1], tV1sV[1], sGs[1], sGkLast[1],
                               tBrW, tBrW_cv, tBrKt, tBrKt_cv,
                               hT, hT_g2, hT_mn, acc1, acc1_mn, lam, tCcC_mn, tCcS_mn)

        # ======================= epilogue: store hm ========================
        # hm[hv] is (K, V+K) row-major; our fragment is its transpose, so we
        # store through a transposed gmem view: hmT[col, row] = hm[row, col].
        # (When NSTRIPS is odd, both halves of the last paired CTA hold the
        # same strip and store identical values — benign.)
        mHM_h = mHM[hv, None, None]
        hmT = cute.composition(
            mHM_h,
            cute.make_layout((self.V + self.K, self.K), stride=(self.K, 1)),
        )
        gHM = cute.local_tile(hmT, (64, self.K), (strip, 0))
        tHMg = thr_mma.partition_C(gHM)
        for i in cutlass.range_constexpr(cute.size(hT)):
            tHMg[i] = hT[i]


# ---------------------------------------------------------------------- host
_COMPILE_CACHE: dict = {}


def _ptr(t: torch.Tensor, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


@torch.no_grad()
def pre_process_fwd_cutedsl(
    k: torch.Tensor,                 # [1, T, H, K] bf16
    v: torch.Tensor,                 # [1, T, HV, V] bf16
    w: torch.Tensor,                 # [1, T, HV, K] bf16
    g: torch.Tensor | None = None,   # [1, T, HV] fp32
    gk: torch.Tensor | None = None,  # [1, T, HV, K] fp32
    BT: int = 64,
) -> torch.Tensor:                   # -> hm [HV, K, V+K] fp32
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    assert B == 1 and T >= 1
    # dtypes must match exactly: the kernel reinterprets raw pointers
    assert k.dtype == v.dtype == w.dtype == torch.bfloat16
    assert k.is_contiguous() and v.is_contiguous() and w.is_contiguous()

    BT = 64
    if T < BT:
        # The kernel clamps the last chunk's tile window to [T-BT, T); pad
        # tiny inputs so at least one full window exists. T itself (and thus
        # the validity masking) is unchanged.
        pad = BT - T

        def _pad(t):
            return torch.nn.functional.pad(t, (0, 0) * (t.dim() - 2) + (0, pad, 0, 0))

        k, v, w = _pad(k), _pad(v), _pad(w)
        if g is not None:
            g = _pad(g.unsqueeze(-1)).squeeze(-1)
        if gk is not None:
            gk = _pad(gk)

    mode = "g" if g is not None else ("gk" if gk is not None else "none")
    if g is None:
        g = torch.zeros(1, 1, 1, dtype=torch.float32, device=k.device)
    if gk is None:
        gk = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=k.device)
    g = g.float().contiguous()
    gk = gk.float().contiguous()

    # the kernel overwrites every element of hm; empty is enough
    hm = torch.empty(HV, K, V + K, dtype=torch.float32, device=k.device)

    args = (
        _ptr(k, cutlass.BFloat16), _ptr(v, cutlass.BFloat16),
        _ptr(w, cutlass.BFloat16), _ptr(g, cutlass.Float32),
        _ptr(gk, cutlass.Float32), _ptr(hm, cutlass.Float32),
        cutlass.Int32(T),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    key = (H, HV, K, V, mode)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        op = PreProcessFwdMerged(H, HV, K, V, gate_mode=mode, BT=BT)
        compiled = cute.compile(op, *args)
        _COMPILE_CACHE[key] = compiled
    compiled(*args)
    return hm
