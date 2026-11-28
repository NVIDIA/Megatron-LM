# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# VR200 BDA backward adapted from Guihong Li, BSD-3-Clause (see LICENSE).
# https://github.com/guihong-nv/Megatron-LM/commit/e2a7d6ef817bec1c74d096daa70129e5f7124fc6

"""CuTe DSL kernels for the fast single-pass mHC path (``fused_mhc_single_pass``).

Imported lazily, and only when a device config there selects one of these kernels. Needs the
CUTLASS Python DSL (``nvidia-cutlass-dsl``, with its TVM-FFI launch path) and a GPU with packed
FP32 instructions (compute capability 10.x).

``try_bda_backward4``: the n = 4 BDA backward in one kernel, bit-identical to the Triton kernels.
One CTA per token row: its threads stream the row in 128-bit accesses, write g_res and g_y
(packed FP32 multiply-add chains in the Triton kernels' order), and reduce the 20 row sums
(g_h_post and g_h_res) in the Triton kernels' order too: per thread
((a0 + a2) + (a4 + a6)) + ((a1 + a3) + (a5 + a7)) over its 8 products, a butterfly over the
16 lanes of each 128-column tile (xor 8, 4, 2, 1), then the tiles added in column order. The
products of BF16 values are exact, so only that order matters. The butterfly runs as a swap
reduction (each step halves the sums a lane keeps), 20 shuffles per pass instead of 80. On VR200
at T 4096, C 5120 it takes 48.5 us against 69.8 us for the two Triton kernels.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, const_expr
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
except (ImportError, OSError) as exc:
    raise ImportError(
        "UnavailableError: mHC CuTe backward requires the CUTLASS DSL runtime"
    ) from exc

_N = 4
_VEC = 8  # BF16 elements per 128-bit access
_TILE = 16 * _VEC  # columns per row-sum tile of the Triton kernels (16 lanes x 8)
_SLOTS = _N + _N * _N  # row sums: g_h_post_i (0..3), g_h_res[j, i] (4 + 4 j + i)


class _BdaBackward4:
    """Grid (T,), block (tpr,): row ``block_idx`` in passes of 8 * tpr columns per stream."""

    def __init__(self, C: int, tpr: int):
        assert tpr % 32 == 0 and C % (_VEC * tpr) == 0, (C, tpr)
        self.tpr = tpr
        self.passes = C // (_VEC * tpr)
        self.tiles = C // _TILE

    @staticmethod
    def _slot(k, go, x, y):
        """Operands of row sum k: go_k . y for k < 4, else go_i . x_j (k = 4 + 4 j + i)."""
        if k < _N:
            return go[k], y
        return go[(k - _N) % _N], x[(k - _N) // _N]

    @cute.jit
    def __call__(
        self,
        mGo: cute.Tensor,
        mHr: cute.Tensor,
        mX: cute.Tensor,
        mHp: cute.Tensor,
        mY: cute.Tensor,
        mGres: cute.Tensor,
        mGy: cute.Tensor,
        mGhp: cute.Tensor,
        mGhr: cute.Tensor,
        stream: cuda.CUstream,
    ):
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mGo.element_type, num_bits_per_copy=128
        )
        tiled_copy = cute.make_tiled_copy_tv(
            atom, cute.make_ordered_layout((1, self.tpr), order=(1, 0)), cute.make_layout((1, _VEC))
        )
        # 512 threads per SM at most 128 registers each: left alone, the compiler took 136 for
        # tpr 128, which leaves room for 3 CTAs instead of 4 (59.8 us instead of 48.5 on VR200).
        self.kernel(mGo, mHr, mX, mHp, mY, mGres, mGy, mGhp, mGhr, tiled_copy).launch(
            grid=[mGo.shape[0], 1, 1],
            block=[self.tpr, 1, 1],
            stream=stream,
            min_blocks_per_mp=512 // self.tpr,
        )

    @cute.kernel
    def kernel(
        self,
        mGo: cute.Tensor,
        mHr: cute.Tensor,
        mX: cute.Tensor,
        mHp: cute.Tensor,
        mY: cute.Tensor,
        mGres: cute.Tensor,
        mGy: cute.Tensor,
        mGhp: cute.Tensor,
        mGhr: cute.Tensor,
        tiled_copy: cute.TiledCopy,
    ):
        """One token row: g_res, g_y and the 20 row sums (written to g_h_post and g_h_res)."""
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        P = const_expr(self.passes)
        tiler = (1, _VEC * self.tpr)
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mGo.element_type, num_bits_per_copy=128
        )
        thr = tiled_copy.get_slice(tid)
        smem = cutlass.utils.SmemAllocator()
        tsum = smem.allocate_tensor(
            Float32, cute.make_layout((_SLOTS, self.tiles)), byte_alignment=16
        )

        hr = [None] * (_N * _N)
        for k in cutlass.range_constexpr(_N * _N):
            hr[k] = mHr[row, k]
        hp = [None] * _N
        for k in cutlass.range_constexpr(_N):
            hp[k] = mHp[row, k]
        li = tid % 16  # lane within the 16-lane group of a 128-column tile
        bit = [(li & 8) != 0, (li & 4) != 0, (li & 2) != 0, (li & 1) != 0]

        for p in cutlass.range_constexpr(P):
            tGo = [None] * _N
            tX = [None] * _N
            tGr = [None] * _N
            rGo = [None] * _N
            rX = [None] * _N
            for i in cutlass.range_constexpr(_N):
                blk = (row, i * P + p)
                tGo[i] = thr.partition_S(cute.local_tile(mGo, tiler, blk))
                tX[i] = thr.partition_S(cute.local_tile(mX, tiler, blk))
                tGr[i] = thr.partition_D(cute.local_tile(mGres, tiler, blk))
                rGo[i] = cute.make_rmem_tensor_like(tGo[i])
                rX[i] = cute.make_rmem_tensor_like(tX[i])
            tY = thr.partition_S(cute.local_tile(mY, tiler, (row, p)))
            tGy = thr.partition_D(cute.local_tile(mGy, tiler, (row, p)))
            rY = cute.make_rmem_tensor_like(tY)
            # All nine loads first.
            for i in cutlass.range_constexpr(_N):
                cute.copy(atom, tGo[i], rGo[i])
            cute.copy(atom, tY, rY)
            for i in cutlass.range_constexpr(_N):
                cute.copy(atom, tX[i], rX[i])
            fGo = [None] * _N
            fX = [None] * _N
            oR = [None] * _N
            for i in cutlass.range_constexpr(_N):
                fGo[i] = cute.make_rmem_tensor_like(rGo[i], Float32)
                fGo[i].store(rGo[i].load().to(Float32))
                fX[i] = cute.make_rmem_tensor_like(rX[i], Float32)
                fX[i].store(rX[i].load().to(Float32))
                oR[i] = cute.make_rmem_tensor_like(rGo[i], Float32)
            fY = cute.make_rmem_tensor_like(rY, Float32)
            fY.store(rY.load().to(Float32))
            oY = cute.make_rmem_tensor_like(rY, Float32)

            # g_y = h_post_0 go_0, then fma(h_post_j, go_j, .); g_res_i likewise with h_res[i, j]:
            # the Triton kernels' chains, two columns per packed instruction.
            for v in cutlass.range_constexpr(0, _VEC, 2):
                g = [None] * _N
                for i in cutlass.range_constexpr(_N):
                    g[i] = (fGo[i][v], fGo[i][v + 1])
                a = cute.arch.mul_packed_f32x2(g[0], (hp[0], hp[0]))
                for j in cutlass.range_constexpr(1, _N):
                    a = cute.arch.fma_packed_f32x2(g[j], (hp[j], hp[j]), a)
                oY[v] = a[0]
                oY[v + 1] = a[1]
                for i in cutlass.range_constexpr(_N):
                    h0 = hr[_N * i]
                    r = cute.arch.mul_packed_f32x2(g[0], (h0, h0))
                    for j in cutlass.range_constexpr(1, _N):
                        hij = hr[_N * i + j]
                        r = cute.arch.fma_packed_f32x2(g[j], (hij, hij), r)
                    oR[i][v] = r[0]
                    oR[i][v + 1] = r[1]
            rGy = cute.make_rmem_tensor_like(tGy)
            rGy.store(oY.load().to(mGy.element_type))
            cute.copy(atom, rGy, tGy)
            for i in cutlass.range_constexpr(_N):
                rG = cute.make_rmem_tensor_like(tGr[i])
                rG.store(oR[i].load().to(mGres.element_type))
                cute.copy(atom, rG, tGr[i])

            # This thread's share of each row sum:
            # ((a0 + a2) + (a4 + a6)) + ((a1 + a3) + (a5 + a7)).
            ch = [None] * _SLOTS
            for k in cutlass.range_constexpr(_SLOTS):
                ga, wa = self._slot(k, fGo, fX, fY)
                q = cute.arch.mul_packed_f32x2((ga[0], ga[1]), (wa[0], wa[1]))
                q = cute.arch.fma_packed_f32x2((ga[2], ga[3]), (wa[2], wa[3]), q)
                r = cute.arch.mul_packed_f32x2((ga[4], ga[5]), (wa[4], wa[5]))
                r = cute.arch.fma_packed_f32x2((ga[6], ga[7]), (wa[6], wa[7]), r)
                s2 = cute.arch.add_packed_f32x2(q, r)
                ch[k] = s2[0] + s2[1]

            # Tile sums: the butterfly xor 8, 4, 2, 1 over the tile's 16 lanes as a swap
            # reduction. At each step a lane keeps half of its sums (the upper half if its bit is
            # set) and sends the other half to its partner; odd counts get a zero. 20 sums go
            # 20 -> 10 -> 5 -> (6) 3 -> (4) 2, so the lane with bits b3 b2 b1 b0 ends with sum
            # m + 2 b0 + 3 b1 + 5 b2 + 10 b3 in register m (m + 2 b0 < 3 and m + 2 b0 + 3 b1 < 5;
            # other registers hold padding).
            vals = list(ch)
            for s_ in cutlass.range_constexpr(4):
                if const_expr(len(vals) % 2 == 1):
                    vals = vals + [Float32(0.0)]
                h = len(vals) // 2
                new = [None] * h
                for m in cutlass.range_constexpr(h):
                    send = Float32(cutlass.select_(bit[s_], vals[m], vals[m + h]))
                    keep = Float32(cutlass.select_(bit[s_], vals[m + h], vals[m]))
                    new[m] = keep + cute.arch.shuffle_sync_bfly(send, offset=8 >> s_)
                vals = new
            tile = p * (self.tpr // 16) + tid // 16
            b0 = li & 1
            b1 = (li >> 1) & 1
            b2 = (li >> 2) & 1
            b3 = (li >> 3) & 1
            for m in cutlass.range_constexpr(2):
                idx3 = m + 2 * b0
                idx2 = idx3 + 3 * b1
                if idx3 < 3:
                    if idx2 < 5:
                        tsum[idx2 + 5 * b2 + 10 * b3, tile] = vals[m]

        # The tiles in column order, from zero (the Triton row-sum loop).
        cute.arch.barrier()
        if tid < _SLOTS:
            s = Float32(0.0)
            for t in cutlass.range_constexpr(self.tiles):
                s = s + tsum[tid, t]
            if tid < _N:
                mGhp[row, tid] = s
            else:
                mGhr[row, tid - _N] = s


_COMPILED: Dict[Tuple[int, int], object] = {}


def _compiled(C: int, tpr: int):
    key = (C, tpr)
    fn = _COMPILED.get(key)
    if fn is None:
        T = cute.sym_int()
        K = _N * C

        def fake(dtype, cols, align):
            return make_fake_tensor(dtype, (T, cols), stride=(cols, 1), assumed_align=align)

        bf16 = cutlass.BFloat16
        fn = cute.compile(
            _BdaBackward4(C, tpr),
            fake(bf16, K, 16),
            fake(Float32, _N * _N, 4),
            fake(bf16, K, 16),
            fake(Float32, _N, 4),
            fake(bf16, C, 16),
            fake(bf16, K, 16),
            fake(bf16, C, 16),
            fake(Float32, _N, 4),
            fake(Float32, _N * _N, 4),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    return fn


def try_bda_backward4(
    grad_out: Tensor,
    h_res: Tensor,
    x2d: Tensor,
    h_post: Tensor,
    y2d: Tensor,
    g_res: Tensor,
    g_y: Tensor,
    g_hp: Tensor,
    g_hr: Tensor,
) -> bool:
    """Fill all four gradients for supported 4-stream inputs. Other layouts use Triton."""
    rows, width = y2d.shape
    threads_per_row = 128
    if not rows or not width or width % (_VEC * threads_per_row):
        return False
    if not all(
        t.dtype == torch.bfloat16 and t.is_contiguous() and t.data_ptr() % 16 == 0
        for t in (grad_out, x2d, y2d)
    ):
        return False
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in (h_res, h_post)):
        return False
    _compiled(width, threads_per_row)(
        grad_out,
        h_res.view(rows, _N * _N),
        x2d,
        h_post,
        y2d,
        g_res,
        g_y,
        g_hp,
        g_hr.view(rows, _N * _N),
    )
    return True
