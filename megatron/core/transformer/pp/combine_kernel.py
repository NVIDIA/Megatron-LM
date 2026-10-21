# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.typing import Float32, Int32

from .api import TILE_KEYS
from .fold import fold_pair, fold_stats, normalize_scale

BATCH = 16


class PPAttentionCombine:
    """Fold the super-block partials left to right; warp w of block x owns row 4 * x + w."""

    WARPS = 4
    NTHREADS = 32 * WARPS
    COLS = 4

    def __init__(self, super_block_tiles: int, head_dim: int, smax: int):
        assert head_dim % 128 == 0, "combine needs head_dim as whole 128-column groups"
        assert smax >= BATCH and smax % 32 == 0, "smax must be a multiple of 32"
        self.G = super_block_tiles
        self.head_dim = int(head_dim)
        self.col_groups = self.head_dim // (32 * self.COLS)
        self.smax = int(smax)
        self.smem_bytes = self.WARPS * 2 * self.smax * 2 * 4

    @cute.jit
    def __call__(
        self,
        mOPartial: cute.Tensor,
        mMLPartial: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mOut: cute.Tensor,
        h_kv: Int32,
        softmax_scale: Float32,
        stream: cuda.CUstream = None,
    ):
        scale_log2 = softmax_scale * math.log2(math.exp(1.0))
        gO = cute.make_tensor(
            mOPartial.iterator,
            cute.make_layout(
                mOPartial.shape,
                stride=(
                    cute.assume(mOPartial.stride[0], divby=4),
                    cute.assume(mOPartial.stride[1], divby=4),
                    cute.assume(mOPartial.stride[2], divby=4),
                    mOPartial.stride[3],
                ),
            ),
        )
        gML = cute.make_tensor(
            mMLPartial.iterator,
            cute.make_layout(
                mMLPartial.shape,
                stride=(
                    cute.assume(mMLPartial.stride[0], divby=2),
                    cute.assume(mMLPartial.stride[1], divby=2),
                    cute.assume(mMLPartial.stride[2], divby=2),
                    mMLPartial.stride[3],
                ),
            ),
        )
        gOut = cute.make_tensor(
            mOut.iterator,
            cute.make_layout(
                mOut.shape,
                stride=(
                    cute.assume(mOut.stride[0], divby=4),
                    cute.assume(mOut.stride[1], divby=4),
                    cute.assume(mOut.stride[2], divby=4),
                    cute.assume(mOut.stride[3], divby=4),
                    mOut.stride[4],
                ),
            ),
        )
        self.kernel(gO, gML, mSeqUsedK, gOut, h_kv, scale_log2).launch(
            grid=(
                cute.ceil_div(mOut.shape[1] * mOut.shape[3], self.WARPS),
                mOut.shape[0] * mOut.shape[2],
                self.col_groups,
            ),
            block=(self.NTHREADS, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=False,
        )

    @cute.kernel
    def kernel(  # pylint: disable=missing-function-docstring
        self,
        mOPartial: cute.Tensor,
        mMLPartial: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mOut: cute.Tensor,
        h_kv: Int32,
        scale_log2: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        tid = tidx % 32
        row_block, flat, col_group = cute.arch.block_idx()
        rpt = mOut.shape[3]
        row = cutlass.min(row_block * self.WARPS + warp, mOut.shape[1] * rpt - 1)
        b = flat // h_kv
        b_o = flat // mOut.shape[2]
        kvh_o = flat - b_o * mOut.shape[2]
        j_o = row // rpt
        i_o = row - j_o * rpt
        n_tiles = cute.ceil_div(mSeqUsedK[b], TILE_KEYS)
        s_b = cute.ceil_div(n_tiles, self.G)
        s_b = cutlass.min(s_b, Int32(mOPartial.shape[1]))
        c = cute.assume(col_group * (32 * self.COLS) + self.COLS * tid, divby=4)
        vec = cute.make_layout(self.COLS)
        ml_vec = cute.make_layout(2)

        smem = utils.SmemAllocator()
        sML_all = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.WARPS, self.smax, 2), stride=(2 * self.smax, 2, 1)),
            byte_alignment=16,
        )
        sAB_all = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.WARPS, self.smax, 2), stride=(2 * self.smax, 2, 1)),
            byte_alignment=16,
        )
        sML = sML_all[warp, None, None]
        sAB = sAB_all[warp, None, None]

        obar = cute.make_rmem_tensor(vec, Float32)
        for q in cutlass.range_constexpr(self.COLS):
            obar[q] = Float32(0.0)
        if s_b > 0:
            L = Float32(0.0)
            for i in cutlass.range(tid, s_b, 32, unroll=1):
                cute.autovec_copy(
                    cute.make_tensor(mMLPartial[flat, i, row, None].iterator, ml_vec), sML[i, None]
                )
            cute.autovec_copy(
                cute.make_tensor(mOPartial[flat, 0, row, None].iterator + c, vec), obar
            )
            rN = cute.make_rmem_tensor(
                cute.make_layout((BATCH, self.COLS), stride=(self.COLS, 1)), Float32
            )
            for u in cutlass.range_constexpr(BATCH):
                s_clamped = cutlass.min(1 + u, s_b - 1)
                cute.autovec_copy(
                    cute.make_tensor(mOPartial[flat, s_clamped, row, None].iterator + c, vec),
                    rN[u, None],
                )
            cute.arch.sync_warp()
            M = sML[0, 0]
            L = sML[0, 1]
            for s in cutlass.range(1, s_b, 1, unroll=1):
                M, L, a, b_scale = fold_stats(M, L, sML[s, 0], sML[s, 1], scale_log2)
                if tid == 0:
                    sAB[s, 0] = a
                    sAB[s, 1] = b_scale
            cute.arch.sync_warp()
            for s0 in cutlass.range(1, s_b, BATCH, unroll=1):
                if s0 > 1:
                    for u in cutlass.range_constexpr(BATCH):
                        s_clamped = cutlass.min(s0 + u, s_b - 1)
                        cute.autovec_copy(
                            cute.make_tensor(
                                mOPartial[flat, s_clamped, row, None].iterator + c, vec
                            ),
                            rN[u, None],
                        )
                for u in cutlass.range_constexpr(BATCH):
                    if s0 + u < s_b:
                        a = sAB[s0 + u, 0]
                        b_scale = sAB[s0 + u, 1]
                        for q in cutlass.range_constexpr(0, self.COLS, 2):
                            o0, o1 = fold_pair(
                                (obar[q], obar[q + 1]), (rN[u, q], rN[u, q + 1]), a, b_scale
                            )
                            obar[q] = o0
                            obar[q + 1] = o1
            inv = normalize_scale(L)
            frg = cute.make_rmem_tensor(vec, Float32)
            for q in cutlass.range_constexpr(0, self.COLS, 2):
                r0, r1 = cute.arch.mul_packed_f32x2((obar[q], obar[q + 1]), (inv, inv))
                frg[q] = r0
                frg[q + 1] = r1
            out_frg = cute.make_rmem_tensor(vec, cutlass.BFloat16)
            out_frg.store(frg.load().to(cutlass.BFloat16))
            cute.autovec_copy(
                out_frg, cute.make_tensor(mOut[b_o, j_o, kvh_o, i_o, None].iterator + c, vec)
            )
        else:
            out_frg = cute.make_rmem_tensor(vec, cutlass.BFloat16)
            for q in cutlass.range_constexpr(self.COLS):
                out_frg[q] = cutlass.BFloat16(0.0)
            cute.autovec_copy(
                out_frg, cute.make_tensor(mOut[b_o, j_o, kvh_o, i_o, None].iterator + c, vec)
            )
