"""SM100 CuTe DSL kernel for KDA's backward state recurrence (the dS tape).

One CTA per (sequence, head) walks the sequence's 64-token chunks in reverse.
With dS the gradient of the chunk-end state (initially dht) and G_last the
chunk's last valid log2 gate row, each chunk:

* stores dS as this chunk's tape row (BF16 ``[K, V]``);
* dv2 = Aqk^T dO + kg dS, stored as dv_new (``kg = k 2^(G_last - G)``);
* dS <- 2^G_last dS + scale qg^T dO - w^T dv2.

These are the semantics of FLA's ``chunk_kda_bwd_dAv`` (dv) followed by
``chunk_gated_delta_rule_bwd_dhu`` with a per-key gate; qg, kg and w come
from ``prep``, and two otherwise idle warps mask Aqk causally in shared
memory.  Only this recurrence is serial, so the kernel keeps its chain short: dS lives in the registers of warps 0-3 (thread = key channel), every
product has the 128 key or value channels on TMEM lanes, and qg^T dO for the
next chunk is issued while the state warps finish the current update.

Per chunk, the chain is: state warps publish BF16 dS -> MMA dv2^T = dv^T +
dS^T kg^T, with dv^T = dO^T Aqk already in the accumulator -> value warps
(thread = value channel) publish -(dv2 / scale) ->
MMA acc += w^T (-(dv2 / scale)) -> state warps dS = 2^G_last dS + scale acc,
where acc already holds qg^T dO.  The tape row, the dv2 rows and the next
gate row are written or loaded off that chain, and the next chunk's qg^T dO
and dv^T are issued while the state warps update dS.

Ragged tails: rows of the first (partial) chunk past the sequence end are
zeroed in dv2 and in the dO tile, so they contribute nothing.

With ``signal``, the kernel lets the gradient kernel that follows it in the
stream launch at once (programmatic dependent launch) and, once a chunk's tape
row and dv2 rows are written, adds one per producer warpgroup to that chunk's
int32 ready flag (two in total), so the gradients of early chunks run while
the recurrence continues.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05, warpgroup

from .._cute_compat import _wrap_tma
from ..fused_gdr_bwd_cute import tcgen05_ws
from ..fused_gdr_bwd_cute.kernel import _load_shared_bf16x8, _store_shared_bf16x8
from .kernel import (
    _column_bases,
    _copy_bf16x8_s2g,
    _globaltimer,
    _ldg_f32,
    _swz_state,
    _swz_token,
    _tmem_ldn_async,
    _tmem_wait_ld,
    _view,
)
from .layouts import build_kda_layout_plan

_BT = 64
_DK = 128
_DV = 128
# Three full warpgroups (setmaxnreg is per warpgroup): state, value, and
# MMA + TMA + two warps that mask Aqk.
_THREADS = 384
_MMA_WARP = 8
_TMA_WARP = 9
_MASK_WARP = 10
# setmaxnreg split of the launch allocation (168 x 384): dS rows stay in the
# state warps' registers.
_STATE_REGISTERS = 240
_VALUE_REGISTERS = 168
_OTHER_REGISTERS = 40
_TMEM_COLUMNS = 512
# TMEM columns: dv2^T (128 value lanes x 64 tokens), then two dS update
# accumulators (128 key lanes x 128 values) used by alternate chunks.
_ACC_DV2 = 0
_ACC_DS = 64

# Two load groups per stage, each released by an MMA commit: A = (kg, qg,
# dO, Aqk) once gemm1 is done, B = w once gemm2 is done.
_B_LOAD_A, _B_LOAD_B, _B_EMPTY_A, _B_EMPTY_B, _B_S_READY, _B_D1, _B_DV2_READY, _B_D2 = (
    0,
    2,
    4,
    6,
    8,
    9,
    10,
    11,
)
# Per stage: Aqk masked in shared memory by the mask warps.
_B_AQK_READY = 13
_NUM_BARRIERS = 15

_BAR_STATE = 1
_BAR_VALUE = 2
_BAR_ALL = 3
_BAR_MASK = 4

_TILE = _BT * _DK
_SQUARE = _BT * _BT
_BYTES_A = 3 * _TILE * 2 + _SQUARE * 2
_BYTES_B = _TILE * 2


def _tile(dtype, elements):
    return cute.struct.Align[cute.struct.MemRange[dtype, elements], 1024]


@cute.struct
class DhuStorage:
    bars: cute.struct.MemRange[cutlass.Int64, 16]
    tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]
    # Two stages of the TMA inputs: four 64x128 token tiles and the 64x64
    # Aqk (masked in place).
    sKg: _tile(cutlass.BFloat16, 2 * _TILE)
    sW: _tile(cutlass.BFloat16, 2 * _TILE)
    sQg: _tile(cutlass.BFloat16, 2 * _TILE)
    sDo: _tile(cutlass.BFloat16, 2 * _TILE)
    sAqk: _tile(cutlass.BFloat16, 2 * _SQUARE)
    # BF16 dS (gemm1 A, MN-major) and -(dv2 / scale)^T (gemm2 B, two 64x64 K-major halves).
    sS: _tile(cutlass.BFloat16, _DK * _DV)
    sDv2: _tile(cutlass.BFloat16, _DV * _BT)
    # dv2 rows, copied out to dv_new.
    sOut: _tile(cutlass.BFloat16, _TILE)


@cute.jit
def _sts_bf16(address, value):
    """Round one FP32 to BF16 and store it to shared memory."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(address).ir_value(), cutlass.Float32(value).ir_value()],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.shared.b16 [$0], h; }",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _stg_b128(address, src):
    """Store eight BF16 (16 bytes) to global memory."""
    view = cute.make_tensor(cute.recast_ptr(src.iterator, dtype=cutlass.Int32), (4,))
    llvm.inline_asm(
        None,
        [cutlass.Int64(address).ir_value()] + [view[i].ir_value() for i in range(4)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _stg_f32(address, value):
    llvm.inline_asm(
        None,
        [cutlass.Int64(address).ir_value(), cutlass.Float32(value).ir_value()],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _signal(address):
    """Add one to a chunk's ready flag with gpu-scope release semantics, after
    this thread's completed bulk (async-proxy) and generic stores."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(address).ir_value()],
        "{ fence.proxy.async.global; red.release.gpu.global.add.s32 [$0], 1; }",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _fence_gpu():
    llvm.inline_asm(
        None,
        [],
        "fence.acq_rel.gpu;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class KdaDhuKernel:
    """dS tape for the KDA backward: one CTA per (sequence, head)."""

    def __init__(self, *, heads: int, signal: bool = False, trace: bool = False):
        self.heads = heads
        self.signal = signal
        self.trace = trace
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

    @staticmethod
    def _state_matrix(tensor):
        """[..., K, V] -> (K, V, (...)) for TMA tiling."""
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[-2], tensor.shape[-1], tuple(tensor.shape[:-2])),
                stride=(tensor.stride[-2], tensor.stride[-1], tuple(tensor.stride[:-2])),
            ),
        )

    @staticmethod
    def _store_layout(rows):
        """rows x 128 BF16 SW128 K-major tile: (r, c) at r * 64 + c % 64 + (c // 64) * rows * 64."""
        atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        return cute.tile_to_shape(atom, (rows, _DV), (0, 1))

    @staticmethod
    def _token_matrix(tensor):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[1], tensor.shape[3], (tensor.shape[0], tensor.shape[2])),
                stride=(tensor.stride[1], tensor.stride[3], (tensor.stride[0], tensor.stride[2])),
            ),
        )

    @cute.jit
    def __call__(
        self,
        kg: cute.Tensor,
        w: cute.Tensor,
        qg: cute.Tensor,
        do: cute.Tensor,
        aqk: cute.Tensor,
        gc: cute.Tensor,
        dht: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        dh: cute.Tensor,
        dv2: cute.Tensor,
        dh0: cute.Tensor,
        flags: cute.Tensor,
        trace_out: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        plan = build_kda_layout_plan(self.io_dtype, self.acc_dtype)
        load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        token = plan.variants["mm_64_64_128_k_k"]
        square = plan.variants["mm_64_64_64_k_k"]
        tma = [
            _wrap_tma(
                cute.nvgpu.make_tiled_tma_atom_A(
                    load_op, self._token_matrix(x), token.smem_a, token.spec.tile, token.tiled_mma
                )
            )
            for x in (kg, w, qg, do)
        ]
        tma.append(
            _wrap_tma(
                cute.nvgpu.make_tiled_tma_atom_A(
                    load_op,
                    self._token_matrix(aqk),
                    square.smem_a,
                    square.spec.tile,
                    square.tiled_mma,
                )
            )
        )
        store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_tape = _wrap_tma(
            cpasync.make_tiled_tma_atom(
                store_op, self._state_matrix(dh), self._store_layout(_DK), (_DK, _DV)
            )
        )
        tma_dv2 = _wrap_tma(
            cpasync.make_tiled_tma_atom(
                store_op, self._token_matrix(dv2), self._store_layout(_BT), (_BT, _DV)
            )
        )
        self.kernel(
            gc,
            dht,
            cu_seqlens,
            chunk_offsets,
            dh,
            dv2,
            dh0,
            flags,
            trace_out,
            scale,
            tma[0],
            tma[1],
            tma[2],
            tma[3],
            tma[4],
            tma_tape,
            tma_dv2,
            token.tiled_mma,
            square.tiled_mma,
            plan.variants["mm_128_64_128_mn_k"].tiled_mma,
            plan.variants["mm_128_64_64_mn_k"].tiled_mma,
            # One tiled MMA per qg^T dO and dv^T issue site: each sets its
            # accumulate flag inside its own dynamic branch.
            plan.variants["mm_128_64_64_mn_mn"].tiled_mma,
            plan.variants["mm_128_64_64_mn_mn"].tiled_mma,
        ).launch(
            grid=((cute.size(cu_seqlens, mode=[0]) - 1) * self.heads, 1, 1),
            block=(_THREADS, 1, 1),
            smem=DhuStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gc,
        dht,
        cu_seqlens,
        chunk_offsets,
        dh,
        dv2,
        dh0,
        flags,
        trace_out,
        scale,
        tma_kg,
        tma_w,
        tma_qg,
        tma_do,
        tma_aqk,
        tma_tape,
        tma_dv2,
        mma_token,
        mma_square,
        mma1,
        mma2,
        mma3_first,
        mma3_next,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.signal):
            # The gradient kernel may start now; it waits on the chunk flags.
            cute.arch.griddepcontrol_launch_dependents()
        self._plan = build_kda_layout_plan(self.io_dtype, self.acc_dtype)
        sequence = block // self.heads
        head = block % self.heads
        storage = cutlass.utils.SmemAllocator().allocate(DhuStorage)
        self._trace = trace_out
        bars = storage.bars.data_ptr()
        holding = storage.tmem_holding.data_ptr()
        if tidx == 0:
            for b in cutlass.range_constexpr(_NUM_BARRIERS):
                cute.arch.mbarrier_init(bars + b, 1)
        cute.arch.mbarrier_init_fence()
        if warp == 0:
            tcgen05_ws.tmem_allocate(holding, _TMEM_COLUMNS)
        cute.arch.sync_threads()
        tmem = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=holding
        )
        seq_start = cu_seqlens[sequence]
        seq_len = cu_seqlens[sequence + 1] - seq_start
        chunk_base = chunk_offsets[sequence]
        num_chunks = chunk_offsets[sequence + 1] - chunk_base
        s_kg = storage.sKg.data_ptr()
        s_w = storage.sW.data_ptr()
        s_qg = storage.sQg.data_ptr()
        s_do = storage.sDo.data_ptr()
        s_aqk = storage.sAqk.data_ptr()
        s_s = storage.sS.data_ptr()
        s_dv2 = storage.sDv2.data_ptr()
        s_out = storage.sOut.data_ptr()
        # This (sequence, head)'s ready flag of chunk c is at flag_base + c * flag_stride.
        flag_base = (
            cutlass.Int64(flags.iterator.toint())
            + cutlass.Int64(chunk_base * self.heads + head) * 4
        )
        flag_stride = self.heads * 4

        if warp < 4:
            cute.arch.setmaxregister_increase(_STATE_REGISTERS)
            self.run_state(
                tidx,
                warp,
                head,
                sequence,
                seq_start,
                seq_len,
                chunk_base,
                num_chunks,
                bars,
                tmem,
                s_s,
                s_do,
                gc,
                dht,
                dh0,
                scale,
                tma_tape,
                flag_base,
                flag_stride,
            )
        elif warp < 8:
            cute.arch.setmaxregister_decrease(_VALUE_REGISTERS)
            self.run_value(
                tidx - 128,
                warp,
                head,
                seq_start,
                seq_len,
                num_chunks,
                bars,
                tmem,
                s_out,
                s_dv2,
                dv2,
                scale,
                tma_dv2,
                flag_base,
                flag_stride,
            )
        else:
            cute.arch.setmaxregister_decrease(_OTHER_REGISTERS)
            if warp == _MMA_WARP:
                self.run_mma(
                    num_chunks,
                    bars,
                    tmem,
                    s_kg,
                    s_w,
                    s_qg,
                    s_do,
                    s_aqk,
                    s_s,
                    s_dv2,
                    mma1,
                    mma2,
                    mma3_first,
                    mma3_next,
                )
            elif warp == _TMA_WARP:
                self.run_tma(
                    head,
                    seq_start,
                    num_chunks,
                    bars,
                    (s_kg, s_w, s_qg, s_do, s_aqk),
                    (tma_kg, tma_w, tma_qg, tma_do, tma_aqk),
                    mma_token,
                    mma_square,
                )
            elif warp >= _MASK_WARP:
                self.run_mask(tidx - _MASK_WARP * 32, seq_len, num_chunks, bars, s_aqk)
        cute.arch.barrier(barrier_id=_BAR_ALL, number_of_threads=_THREADS)
        if warp == 0:
            cute.arch.relinquish_tmem_alloc_permit()
            tcgen05_ws.tmem_deallocate(holding, _TMEM_COLUMNS)

    # ------------------------------------------------------------ helpers

    @cute.jit
    def _stamp(self, slot: cutlass.Constexpr, j):
        """%globaltimer of CTA 0 for the first 16 chunks (development tracing)."""
        if cutlass.const_expr(self.trace):
            block, _, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            trace = self._trace
            if block == 0 and tidx % 32 == 0 and j < 16:
                trace[slot, j] = _globaltimer()

    @cute.jit
    def _commit(self, ptr):
        with cute.arch.elect_one():
            tcgen05.commit(ptr)

    @staticmethod
    def _acc(tmem, offset, tiled_mma, shape):
        fragment = tiled_mma.make_fragment_C(cute.append(tiled_mma.partition_shape_C(shape), 1))
        return cute.make_tensor(tmem + offset, fragment.layout)

    @cute.jit
    def _gemm(self, tiled_mma, acc, a, b, first_accumulate: cutlass.Constexpr):
        frag_a = tiled_mma.make_fragment_A(a)
        frag_b = tiled_mma.make_fragment_B(b)
        for kp in cutlass.range_constexpr(cute.size(frag_a, mode=[2])):
            tiled_mma.set(tcgen05.Field.ACCUMULATE, first_accumulate if kp == 0 else True)
            cute.gemm(
                tiled_mma,
                acc[None, None, None, 0],
                frag_a[None, None, kp, 0],
                frag_b[None, None, kp, 0],
                acc[None, None, None, 0],
            )

    def _operand(self, variant, ptr, which, offset=0):
        built = self._plan.variants[variant]
        return _view(ptr, built.smem_a_staged if which == "a" else built.smem_b_staged, offset)

    @cute.jit
    def _tma_store(self, info, smem_ptr, rows: cutlass.Constexpr, row, coord):
        """Bulk-store a rows x 128 tile (``_store_layout``) at (row, 0) of ``coord``."""
        layout = self._store_layout(rows)
        tile = _view(smem_ptr, layout)
        g = cute.domain_offset((row, cutlass.Int32(0)), info[1][None, None, coord])
        g = cute.flat_divide(g, (rows, _DV))
        s_part, g_part = cpasync.tma_partition(
            info[0], 0, cute.make_layout(1), cute.group_modes(tile, 0, 2), cute.group_modes(g, 0, 2)
        )
        cute.copy(info[0], s_part[None], g_part[(None, 0, 0)])
        cute.arch.cp_async_bulk_commit_group()

    # ---------------------------------------------------------------- TMA

    @cute.jit
    def _tma(self, info, smem, tiled_mma, tile: cutlass.Constexpr, row, coord, bar):
        g = cute.domain_offset((row, cutlass.Int32(0)), info[1][None, None, coord])
        g = tiled_mma.get_slice(0).partition_A(cute.flat_divide(g, tile))
        s_part, g_part = cpasync.tma_partition(
            info[0], 0, cute.make_layout(1), cute.group_modes(smem, 0, 3), cute.group_modes(g, 0, 3)
        )
        cute.copy(info[0], g_part[(None, 0, 0)], s_part[(None, 0)], tma_bar_ptr=bar)

    @cute.jit
    def run_tma(self, head, seq_start, num_chunks, bars, tiles, infos, mma_token, mma_square):
        layout = self._plan.canonical_staged.token_direct
        square = self._plan.canonical_staged.square_direct
        group = (cutlass.Int32(0), head)
        for j in cutlass.range(num_chunks):
            stage = j % 2
            row = seq_start + (num_chunks - 1 - j) * _BT
            if j >= 2:
                cute.arch.mbarrier_wait(bars + _B_EMPTY_A + stage, (j // 2 - 1) & 1)
            bar = bars + _B_LOAD_A + stage
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar, _BYTES_A)
            for i in cutlass.range_constexpr(4):
                if cutlass.const_expr(i != 1):
                    self._tma(
                        infos[i],
                        _view(tiles[i], layout, stage * _TILE),
                        mma_token,
                        (_BT, _DK),
                        row,
                        group,
                        bar,
                    )
            self._tma(
                infos[4],
                _view(tiles[4], square, stage * _SQUARE),
                mma_square,
                (_BT, _BT),
                row,
                group,
                bar,
            )
            if j >= 2:
                cute.arch.mbarrier_wait(bars + _B_EMPTY_B + stage, (j // 2 - 1) & 1)
            bar = bars + _B_LOAD_B + stage
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar, _BYTES_B)
            self._tma(
                infos[1],
                _view(tiles[1], layout, stage * _TILE),
                mma_token,
                (_BT, _DK),
                row,
                group,
                bar,
            )

    # ---------------------------------------------------------------- MMA

    @cute.jit
    def _gemm_qdo(self, tmem, mma3, s_qg, s_do, s_aqk, j):
        """acc[j % 2] = qg^T dO for chunk j (two 64-column value halves), then
        dv^T = dO^T Aqk into the dv2 accumulator."""
        stage = j % 2
        for hh in cutlass.range_constexpr(2):
            acc = self._acc(tmem, _ACC_DS + stage * 128 + hh * 64, mma3, (_DK, _BT))
            self._gemm(
                mma3,
                acc,
                self._operand("mm_128_64_64_mn_mn", s_qg, "a", stage * _TILE),
                self._operand("mm_128_64_64_mn_mn", s_do, "b", stage * _TILE + hh * _BT * _BT),
                False,
            )
        self._gemm(
            mma3,
            self._acc(tmem, _ACC_DV2, mma3, (_DV, _BT)),
            self._operand("mm_128_64_64_mn_mn", s_do, "a", stage * _TILE),
            self._operand("mm_128_64_64_mn_mn", s_aqk, "b", stage * _SQUARE),
            False,
        )

    @cute.jit
    def run_mma(
        self,
        num_chunks,
        bars,
        tmem,
        s_kg,
        s_w,
        s_qg,
        s_do,
        s_aqk,
        s_s,
        s_dv2,
        mma1,
        mma2,
        mma3_first,
        mma3_next,
    ):
        acc1 = self._acc(tmem, _ACC_DV2, mma1, (_DV, _BT))
        for j in cutlass.range(num_chunks):
            stage = j % 2
            cute.arch.mbarrier_wait(bars + _B_S_READY, j & 1)
            self._stamp(0, j)
            if j == 0:
                cute.arch.mbarrier_wait(bars + _B_LOAD_A, 0)
                cute.arch.mbarrier_wait(bars + _B_AQK_READY, 0)
                self._gemm_qdo(tmem, mma3_first, s_qg, s_do, s_aqk, j)
            # dv2^T = dv^T + dS^T kg^T (value lanes, token columns).
            self._gemm(
                mma1,
                acc1,
                self._operand("mm_128_64_128_mn_k", s_s, "a"),
                self._operand("mm_128_64_128_mn_k", s_kg, "b", stage * _TILE),
                True,
            )
            self._commit(bars + _B_D1)
            self._commit(bars + _B_EMPTY_A + stage)
            cute.arch.mbarrier_wait(bars + _B_LOAD_B + stage, (j // 2) & 1)
            cute.arch.mbarrier_wait(bars + _B_DV2_READY, j & 1)
            self._stamp(1, j)
            for hh in cutlass.range_constexpr(2):
                acc = self._acc(tmem, _ACC_DS + stage * 128 + hh * 64, mma2, (_DK, _BT))
                self._gemm(
                    mma2,
                    acc,
                    self._operand("mm_128_64_64_mn_k", s_w, "a", stage * _TILE),
                    self._operand("mm_128_64_64_mn_k", s_dv2, "b", hh * _BT * _BT),
                    True,
                )
            self._commit(bars + _B_D2 + stage)
            self._commit(bars + _B_EMPTY_B + stage)
            if j + 1 < num_chunks:
                # The next chunk's qg^T dO runs while the state warps update dS.
                cute.arch.mbarrier_wait(bars + _B_LOAD_A + (j + 1) % 2, ((j + 1) // 2) & 1)
                cute.arch.mbarrier_wait(bars + _B_AQK_READY + (j + 1) % 2, ((j + 1) // 2) & 1)
                self._stamp(7, j)
                self._gemm_qdo(tmem, mma3_next, s_qg, s_do, s_aqk, j + 1)

    # ------------------------------------------------------------- mask

    @cute.jit
    def run_mask(self, u, seq_len, num_chunks, bars, s_aqk):
        """Warps 10-11: keep Aqk on column <= row < valid, in shared memory.

        FLA leaves Aqk above the diagonal unwritten, and rows past a sequence's
        end belong to the next sequence, so both are zeroed before the MMA reads
        the tile (8 columns per access, 8 rows per thread).
        """
        vec = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        base = cutlass.Int32(s_aqk.toint())
        c0 = (u % 8) * 8
        for j in cutlass.range(num_chunks):
            stage = j % 2
            valid = seq_len - (num_chunks - 1 - j) * _BT
            cute.arch.mbarrier_wait(bars + _B_LOAD_A + stage, (j // 2) & 1)
            for i in cutlass.range_constexpr(_BT // 8):
                row = u // 8 + 8 * i
                address = _swz_token(base + stage * _SQUARE * 2, row, c0)
                _load_shared_bf16x8(address, vec)
                for e in cutlass.range_constexpr(8):
                    keep = (c0 + e <= row) and (row < valid)
                    vec[e] = vec[e] if keep else cutlass.BFloat16(0.0)
                _store_shared_bf16x8(address, vec)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=_BAR_MASK, number_of_threads=64)
            if u == 0:
                cute.arch.mbarrier_arrive(bars + _B_AQK_READY + stage)

    # -------------------------------------------------------------- state

    @cute.jit
    def run_state(
        self,
        tidx,
        warp,
        head,
        sequence,
        seq_start,
        seq_len,
        chunk_base,
        num_chunks,
        bars,
        tmem,
        s_s,
        s_do,
        gc,
        dht,
        dh0,
        scale,
        tma_tape,
        flag_base,
        flag_stride,
    ):
        """Warps 0-3: thread = key channel; dS row in registers."""
        heads = self.heads
        lane_base = cutlass.Int32(tmem.toint()) + ((warp % 4) * 32 << 16)
        ds = cute.make_rmem_tensor((_DV,), cutlass.Float32)
        buf = [cute.make_rmem_tensor((32,), cutlass.Float32) for _ in range(2)]
        out8 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        zero8 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        for e in cutlass.range_constexpr(8):
            zero8[e] = cutlass.BFloat16(0.0)
        row_offset = cutlass.Int64((sequence * heads + head) * _DK + tidx) * (_DV * 4)
        dht_row = cutlass.Int64(dht.iterator.toint()) + cutlass.Int64(row_offset)
        for c in cutlass.range_constexpr(_DV):
            ds[c] = _ldg_f32(dht_row + c * 4)
        gc_col = cutlass.Int64(gc.iterator.toint()) + (head * _DK + tidx) * 4
        gc_row = heads * _DK * 4
        base_s = cutlass.Int32(s_s.toint())
        base_do = cutlass.Int32(s_do.toint())
        valid0 = seq_len - (num_chunks - 1) * _BT
        # Gate row of the chunk being published, loaded one chunk ahead.
        g_next = _ldg_f32(gc_col + cutlass.Int64(cutlass.max(seq_start + seq_len - 1, 0)) * gc_row)
        gamma = cutlass.Float32(1.0)
        for j in cutlass.range(num_chunks):
            chunk = num_chunks - 1 - j
            if j == 0:
                self._publish(ds, out8, base_s, tidx, 0, _DV)
            else:
                # The previous tape row must be out of sS before it is overwritten.
                # With a signal, the row before it (long written by now) raises its
                # flag, so the chain never waits for a store to land.
                if tidx == 0:
                    if cutlass.const_expr(self.signal):
                        if j >= 2:
                            cute.arch.cp_async_bulk_wait_group(1)
                            _signal(flag_base + (chunk + 2) * flag_stride)
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=_BAR_STATE, number_of_threads=128)
                self._update(
                    warp, lane_base, bars, ds, buf, out8, base_s, tidx, gamma, scale, j - 1
                )
            if j == 0:
                if valid0 < _BT:
                    # Zero dO rows past the sequence end (stage 0 holds this chunk).
                    cute.arch.mbarrier_wait(bars + _B_LOAD_A, 0)
                    for r8 in cutlass.range_constexpr(_BT // 8):
                        row = r8 * 8 + tidx // 16
                        if row >= valid0:
                            _store_shared_bf16x8(_swz_token(base_do, row, (tidx % 16) * 8), zero8)
            cute.arch.fence_view_async_shared()
            if warp == 0:
                self._stamp(3, j)
            cute.arch.barrier(barrier_id=_BAR_STATE, number_of_threads=128)
            if tidx == 0:
                cute.arch.mbarrier_arrive(bars + _B_S_READY)
                self._tma_store(
                    tma_tape,
                    s_s,
                    _DK,
                    cutlass.Int32(0),
                    (cutlass.Int32(0), chunk_base + chunk, head),
                )
            # Off the chain: this chunk's decay and the next gate row.
            gamma = cute.math.exp2(g_next, fastmath=True)
            next_chunk = cutlass.max(chunk - 1, 0)
            g_next = _ldg_f32(
                gc_col + cutlass.Int64(seq_start + next_chunk * _BT + _BT - 1) * gc_row
            )
        if tidx == 0:
            cute.arch.cp_async_bulk_wait_group(0)
            if cutlass.const_expr(self.signal):
                if num_chunks >= 2:
                    _signal(flag_base + flag_stride)
                if num_chunks >= 1:
                    _signal(flag_base)
        if num_chunks > 0:
            self._update(
                warp,
                lane_base,
                bars,
                ds,
                buf,
                out8,
                base_s,
                tidx,
                gamma,
                scale,
                num_chunks - 1,
                publish=False,
            )
        out_row = cutlass.Int64(dh0.iterator.toint()) + cutlass.Int64(row_offset)
        for c in cutlass.range_constexpr(_DV):
            _stg_f32(out_row + c * 4, ds[c])

    @cute.jit
    def _publish(self, ds, out8, base_s, tidx, first: cutlass.Constexpr, count: cutlass.Constexpr):
        """BF16 dS columns [first, first + count) of this key row into sS."""
        for v8 in cutlass.range_constexpr(count // 8):
            for e in cutlass.range_constexpr(8):
                out8[e] = ds[first + v8 * 8 + e].to(self.io_dtype)
            _store_shared_bf16x8(_swz_state(base_s, tidx, first + v8 * 8), out8)

    @cute.jit
    def _update(
        self,
        warp,
        lane_base,
        bars,
        ds,
        buf,
        out8,
        base_s,
        tidx,
        gamma,
        scale,
        j,
        publish: cutlass.Constexpr = True,
    ):
        """dS = 2^G_last dS + scale acc, with acc = qg^T dO - w^T (dv2 / scale) of chunk j,
        published 32 columns at a time while the next 32 load from TMEM."""
        stage = j % 2
        cute.arch.mbarrier_wait(bars + _B_D2 + stage, (j // 2) & 1)
        if warp == 0:
            self._stamp(2, j)
        address = lane_base + _ACC_DS + stage * 128
        quarters = _DV // 32
        _tmem_ldn_async(address, buf[0], 32)
        _tmem_wait_ld()
        for q in cutlass.range_constexpr(quarters):
            if cutlass.const_expr(q + 1 < quarters):
                _tmem_ldn_async(address + (q + 1) * 32, buf[(q + 1) % 2], 32)
            for c in cutlass.range_constexpr(32):
                ds[q * 32 + c] = gamma * ds[q * 32 + c] + scale * buf[q % 2][c]
            if cutlass.const_expr(publish):
                self._publish(ds, out8, base_s, tidx, q * 32, 32)
            if cutlass.const_expr(q + 1 < quarters):
                _tmem_wait_ld()
        cute.arch.fence_view_async_tmem_load()

    # -------------------------------------------------------------- value

    @cute.jit
    def run_value(
        self,
        vtidx,
        warp,
        head,
        seq_start,
        seq_len,
        num_chunks,
        bars,
        tmem,
        s_out,
        s_dv2,
        dv2,
        scale,
        tma_dv2,
        flag_base,
        flag_stride,
    ):
        """Warps 4-7: thread = value channel; the dv2 rows of one chunk."""
        heads = self.heads
        lane_base = cutlass.Int32(tmem.toint()) + ((warp % 4) * 32 << 16)
        neg_inv_scale = -1.0 / scale
        out_base = cutlass.Int64(dv2.iterator.toint()) + head * _DV * 2
        token_stride = heads * _DV * 2
        buf = [cute.make_rmem_tensor((32,), cutlass.Float32) for _ in range(2)]
        out8 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        half_base = cutlass.Int32(s_dv2.toint()) + (vtidx // 64) * (_BT * _BT * 2)
        # Column vtidx of the dv2 output tile: element (row, vtidx) is at
        # bases[row % 8] + row * 128.
        bases = _column_bases(cutlass.Int32(s_out.toint()), vtidx)
        valid0 = seq_len - (num_chunks - 1) * _BT
        for j in cutlass.range(num_chunks):
            chunk = num_chunks - 1 - j
            tok0 = seq_start + chunk * _BT
            valid = seq_len - chunk * _BT
            if valid > _BT:
                valid = _BT
            cute.arch.mbarrier_wait(bars + _B_D1, j & 1)
            if warp == 4:
                self._stamp(4, j)
            for h32 in cutlass.range_constexpr(_BT // 32):
                _tmem_ldn_async(lane_base + _ACC_DV2 + h32 * 32, buf[h32], 32)
            _tmem_wait_ld()
            for h32 in cutlass.range_constexpr(_BT // 32):
                for c8 in cutlass.range_constexpr(4):
                    for e in cutlass.range_constexpr(8):
                        t = h32 * 32 + c8 * 8 + e
                        x = buf[h32][c8 * 8 + e]
                        if t >= valid:
                            x = cutlass.Float32(0.0)
                        _sts_bf16(bases[t % 8] + t * 128, x)
                        out8[e] = (x * neg_inv_scale).to(self.io_dtype)
                    _store_shared_bf16x8(_swz_token(half_base, vtidx % 64, h32 * 32 + c8 * 8), out8)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.fence_view_async_shared()
            if warp == 4:
                self._stamp(5, j)
            cute.arch.barrier(barrier_id=_BAR_VALUE, number_of_threads=128)
            if vtidx == 0:
                cute.arch.mbarrier_arrive(bars + _B_DV2_READY)
            if cutlass.const_expr(self.signal):
                # The previous chunk's bulk store has landed long ago: raise its
                # flag (a partial chunk, always the first, raised its own).
                if vtidx == 0 and j > 0:
                    if j >= 2 or valid0 == _BT:
                        cute.arch.cp_async_bulk_wait_group(0)
                        _signal(flag_base + (chunk + 1) * flag_stride)
            # dv2 rows out: one bulk store for a full chunk; a partial chunk
            # must not touch the next sequence's rows.  The tile is rewritten
            # only after the barrier below.
            if valid == _BT:
                if vtidx == 0:
                    self._tma_store(tma_dv2, s_out, _BT, tok0, (cutlass.Int32(0), head))
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                tile_base = cutlass.Int32(s_out.toint())
                for i in cutlass.range_constexpr(_BT * _DV // 8 // 128):
                    index = i * 128 + vtidx
                    row = index // 16
                    col = (index % 16) * 8
                    if row < valid:
                        _copy_bf16x8_s2g(
                            _swz_token(tile_base, row, col),
                            out_base + cutlass.Int64(tok0 + row) * token_stride + col * 2,
                        )
                if cutlass.const_expr(self.signal):
                    _fence_gpu()
            cute.arch.barrier(barrier_id=_BAR_VALUE, number_of_threads=128)
            if cutlass.const_expr(self.signal):
                if vtidx == 0 and valid < _BT:
                    _signal(flag_base + chunk * flag_stride)
        if cutlass.const_expr(self.signal):
            # The last (first in the sequence) chunk, if full, after its store landed.
            if vtidx == 0 and num_chunks > 0 and seq_len >= _BT:
                cute.arch.cp_async_bulk_wait_group(0)
                _signal(flag_base)
