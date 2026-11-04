"""SM100 CuTe DSL device kernel for the fused KDA backward.

``split`` CTAs per (sequence, head) walk the sequence's 64-token chunks in
reverse and keep the state gradient dH (128 key x 128 value, FP32) in tensor
memory.
A chunk reads FLA's forward residuals (q, k, v, dO, Aqk, Akk and the
chunk-start state h in BF16; the chunk-local log2 gate cumsum and beta in
FP32) and writes dq, dk, dv, dg (after the chunk-local reverse cumsum), dbeta
and, after the first chunk, dh0.

Per chunk, with G the gate tile, Gamma = 2^G, G_last its last valid row, S the
chunk-start state and dS the gradient of the chunk-end state (initially dht):

* X = (k*Gamma) S, Delta = v - X, v_new = (Akk diag(beta)) Delta;
* dv_new = (k*2^(G_last - G)) dS + Aqk^T dO;
* D = Akk^T dv_new, dv = beta*D, dbeta += rowsum(D*Delta);
* dAqk = scale dO v_new^T (causal), dAkk = (dv_new Delta^T) diag(beta)
  (strictly lower), dL = -Akk^T dAkk Akk^T;
* dq and dk collect state-path terms from S, dS and D and intra-chunk terms
  from dAqk and dL; dg = q*dq + k*dk-terms, then its chunk-local reverse
  cumsum;
* dS <- Gamma_last*dS + (scale q*Gamma)^T dO - (k*beta*Gamma)^T D.

Orientation: value-side products (X = k*Gamma*S, dv_new, v_new, D) are
token-major as in the GDR backward.  Every token x key-channel product (dq,
dk and the intra-chunk partials) accumulates channel-major, with the 128 key
channels on TMEM lanes, so per-channel gate factors, the diagonal fallback and
the dg reverse cumsum are per-thread work.

Numerics: the state path only needs 2^G and 2^(G_last - G), both <= 1.
Intra-chunk products work on 16-token blocks anchored at a block's last row
(lower products) or first row (upper products), so off-diagonal factors are
<= 1 for any gate.  A diagonal block uses the same tensor-core product for a
channel while that channel's log2 gate span is below ``DIAG_SPAN_LIMIT``;
otherwise the owning thread recomputes the channel's triangle on CUDA cores.
The anchored token operands are BF16 high/low pairs, which keeps dg at
FP32-operand accuracy; every other MMA operand is a single BF16 tile.

Split: only the dS recursion carries state across chunks, and it needs only
phases A-D of a chunk.  With ``split=2`` both CTAs of a (sequence, head) run
the recursion over every chunk, but each runs the intra-chunk products, the
dq/dk/dg epilogue and the dv, dbeta and dh0 stores for every second chunk
(phases C-E and the stores are skipped on the other CTA's chunks).  The
launcher picks two CTAs while they fit on the device in one wave.

Schedule: warps 0-3 (CH) do the channel-major element-wise work and stores,
warps 4-7 (TK) the token-major work, warp 8 issues tcgen05 MMAs and warp 9
issues TMA loads.  Every phase of a chunk is a pair of one-shot mbarriers
(consumers -> MMA "ready", MMA -> consumers "done").  The next chunk's v, dO,
Akk and h are loaded once MMA phase D has consumed them; q, k and Aqk are
loaded after the current chunk releases them (FREE).

The CTA topology, MMA layout contracts, TMA descriptors and TMEM helpers come
from the fused GDR backward (``fused_gdr_bwd_cute``), itself a CuTe DSL
derivative of QwenLM FlashQLA's Blackwell fused GDR backward (MIT License).
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05

from .._cute_compat import _wrap_tma
from ..fused_gdr_bwd_cute import tcgen05_ws
from ..fused_gdr_bwd_cute.kernel import _load_shared_bf16x8, _store_shared_bf16x8
from .layouts import build_kda_layout_plan
from .storage import TMEM_COLUMNS, SharedStorage, tmem_offset

_BT = 64
_BC = 16
_DK = 128
_DV = 128
# Three warpgroups (CH 0-3, TK 4-7, MMA/TMA 8-9 plus two idle warps).  Each
# SM sub-partition has a 16 K-register file and holds one warp of each
# warpgroup, so setmaxnreg splits its 504 registers per lane as 232 + 232 + 40.
# Fewer warps do not help: with ten warps two sub-partitions hold three warps
# and ptxas caps every thread at 168 registers.
_THREADS = 384
_CH_REGISTERS = 232
_TK_REGISTERS = 232
_OTHER_REGISTERS = 40
_CH_WARPS = (0, 1, 2, 3)
_TK_WARPS = (4, 5, 6, 7)
_MMA_WARP = 8
_TMA_WARP = 9
# A diagonal 16-token block uses the tensor-core product for a channel only
# while its log2 gate span is below this bound (see the module docstring).  At
# 112 the bounded-gate configuration (|g| <= 5 per token, span <= 108) never
# falls back.
DIAG_SPAN_LIMIT = 112.0
# The CUDA-core fallback splits the decay at the block's middle token while
# each half span is below this bound (FP32 factors stay within 2^+-100).
DIAG_HALF_SPAN_LIMIT = 100.0

(
    _E_LOAD,
    _E_READY_A,
    _E_DONE_A,
    _E_READY_B,
    _E_DONE_B,
    _E_READY_C,
    _E_DONE_C,
    _E_READY_D,
    _E_DONE_D,
    _E_READY_E,
    _E_DONE_E,
    _E_FREE,
    _E_LOAD_EARLY,
) = range(13)
# Arrivals per phase: consumer edges count one elected thread per warpgroup.
_EDGE_COUNTS = (1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1)

_BAR_ALL = 1
_BAR_CH = 2
_BAR_TK = 3
_BAR_CH_TK = 4

# Debug dump slots (FP32 [chunk, head, slot, 64, 128]).
DEBUG_SLOTS = (
    "x_state",
    "dv_new",
    "v_new",
    "d",
    "d_aqk",
    "d_akk",
    "d_lower",
    "dq_state",
    "dk_state",
    "dk_w",
)


@cute.jit
def _tmem_ldn_async(address, dst, n: cutlass.Constexpr):
    """Issue an n-column TMEM load of this thread's lane; pair with _tmem_wait_ld."""
    i32 = ir.IntegerType.get_signless(32)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([i32] * n),
        [cutlass.Int32(address).ir_value()],
        f"tcgen05.ld.sync.aligned.32x32b.x{n}.b32 {{"
        + ", ".join(f"${i}" for i in range(n))
        + f"}}, [${n}];",
        ",".join(["=r"] * n + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    view = cute.make_tensor(cute.recast_ptr(dst.iterator, dtype=cutlass.Int32), (n,))
    for i in cutlass.range_constexpr(n):
        view[i] = cutlass.Int32(llvm.extractvalue(i32, result, [i]))


@cute.jit
def _tmem_ld32_async(address, dst):
    """Issue a 32-column TMEM load of this thread's lane; pair with _tmem_wait_ld."""
    _tmem_ldn_async(address, dst, 32)


@cute.jit
def _tmem_ld1(address):
    """Load one FP32 column of this thread's TMEM lane (warp-collective)."""
    value = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [cutlass.Int32(address).ir_value()],
        "{ tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1]; tcgen05.wait::ld.sync.aligned; }",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(value)


@cute.jit
def _tmem_wait_ld():
    llvm.inline_asm(
        None,
        [],
        "tcgen05.wait::ld.sync.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _tmem_ld32(address, dst):
    """Load 32 consecutive FP32 columns of this thread's TMEM lane."""
    _tmem_ld32_async(address, dst)
    _tmem_wait_ld()


@cute.jit
def _tmem_st32(address, src):
    """Store 32 consecutive FP32 columns of this thread's TMEM lane."""
    _tmem_stn(address, src, 32)


@cute.jit
def _tmem_stn(address, src, n: cutlass.Constexpr):
    """Store n consecutive FP32 columns of this thread's TMEM lane."""
    view = cute.make_tensor(cute.recast_ptr(src.iterator, dtype=cutlass.Int32), (n,))
    llvm.inline_asm(
        None,
        [cutlass.Int32(address).ir_value()] + [view[i].ir_value() for i in range(n)],
        f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0], {{"
        + ", ".join(f"${i}" for i in range(1, n + 1))
        + "};",
        ",".join(["r"] * (n + 1)),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _globaltimer():
    """Nanosecond global timer (for phase tracing only)."""
    return cutlass.Int64(
        llvm.inline_asm(
            ir.IntegerType.get_signless(64),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _swz_token(base, row, col):
    """Shared address of 8 BF16 at (row, col..col+7) of a 64x128 SW128 K-major tile."""
    linear = base + (row * 64 + (col & 63) + (col >> 6) * (_BT * 64)) * 2
    return linear ^ ((linear >> 3) & 112)


@cute.jit
def _swz_state(base, row, col):
    """Shared address of 8 BF16 at (row, col..col+7) of a 128x128 SW128 K-major tile."""
    linear = base + (row * 64 + (col & 63) + (col >> 6) * (_DK * 64)) * 2
    return linear ^ ((linear >> 3) & 112)


@cute.jit
def _ldg_f32x8(address, dst):
    """Load 8 contiguous FP32 values (32-byte aligned) from global memory."""
    i32 = ir.IntegerType.get_signless(32)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([i32] * 8),
        [cutlass.Int64(address).ir_value()],
        "{ ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$8]; ld.global.nc.v4.b32 {$4, $5, $6, $7}, [$8+16]; }",
        ",".join(["=r"] * 8 + ["l"]),
        # The gate tensor is read-only in the kernel, so the loads may be scheduled freely.
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    view = cute.make_tensor(cute.recast_ptr(dst.iterator, dtype=cutlass.Int32), (8,))
    for i in cutlass.range_constexpr(8):
        view[i] = cutlass.Int32(llvm.extractvalue(i32, result, [i]))


@cute.jit
def _ld_f32(address):
    """One FP32 from global memory, coherent (written by another CTA in this kernel)."""
    bits = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [cutlass.Int64(address).ir_value()],
        "ld.global.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    view = cute.make_rmem_tensor((1,), cutlass.Int32)
    view[0] = cutlass.Int32(bits)
    return cute.make_tensor(cute.recast_ptr(view.iterator, dtype=cutlass.Float32), (1,))[0]


def _column_bases(base, col):
    """Per-row-residue shared addresses of column ``col`` in a 64x128 SW128
    token tile: element (row, col) is at ``bases[row % 8] + row * 128``.

    The tile is 1024-byte aligned, so the swizzle XOR (bits 4-6) depends only
    on ``row % 8``; with these eight bases every access is one LDS with an
    immediate offset."""
    linear = base + ((col & 63) + (col >> 6) * (_BT * 64)) * 2
    return [linear ^ (residue << 4) for residue in range(8)]


@cute.jit
def _lds_bf16(address):
    """One BF16 from shared memory, widened to FP32."""
    bits = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [cutlass.Int32(address).ir_value()],
        "{ .reg .b16 h, z; ld.shared.b16 h, [$1]; mov.b16 z, 0; mov.b32 $0, {z, h}; }",
        "=r,r",
        # Side effects keep it inside the chunk loop: the address is invariant, the tile is not.
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    view = cute.make_rmem_tensor((1,), cutlass.Int32)
    view[0] = cutlass.Int32(bits)
    return cute.make_tensor(cute.recast_ptr(view.iterator, dtype=cutlass.Float32), (1,))[0]


@cute.jit
def _ldg_f32(address):
    """One FP32 from global memory (read-only path)."""
    bits = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [cutlass.Int64(address).ir_value()],
        "ld.global.nc.b32 $0, [$1];",
        "=r,l",
        # The gate tensor is read-only in the kernel, so the loads may be scheduled freely.
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    view = cute.make_rmem_tensor((1,), cutlass.Int32)
    view[0] = cutlass.Int32(bits)
    return cute.make_tensor(cute.recast_ptr(view.iterator, dtype=cutlass.Float32), (1,))[0]


@cute.jit
def _copy_bf16x8_s2g(shared_address, global_address):
    """Copy one aligned 128-bit BF16 vector from shared to global memory."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(shared_address).ir_value(), cutlass.Int64(global_address).ir_value()],
        """{
.reg .b32 p0, p1, p2, p3;
ld.shared.v4.b32 {p0, p1, p2, p3}, [$0];
st.global.v4.b32 [$1], {p0, p1, p2, p3};
}""",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _stg_f32(address, value):
    """Store one FP32 to global memory."""
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
def _stg_bf16(address, value):
    """Round one FP32 to BF16 and store it to global memory."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(address).ir_value(), cutlass.Float32(value).ir_value()],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _mn_view(tensor):
    """Group ((atom_m, atom_k), rest_m, rest_k, ...) into (m, k, ...)."""
    layout = tensor.layout
    outer = layout.outer if isinstance(layout, cute.ComposedLayout) else layout
    shape, stride = outer.shape, outer.stride
    grouped = cute.make_layout(
        ((shape[0][0], shape[1]), (shape[0][1], shape[2]), *shape[3:]),
        stride=((stride[0][0], stride[1]), (stride[0][1], stride[2]), *stride[3:]),
    )
    if isinstance(layout, cute.ComposedLayout):
        grouped = cute.make_composed_layout(layout.inner, layout.offset, grouped)
    return cute.make_tensor(tensor.iterator, grouped)


def _view(ptr, layout, element_offset=0):
    """A swizzled BF16 view of a shared buffer, optionally at an element offset."""
    return cute.make_tensor(
        cute.recast_ptr(ptr + element_offset, layout.inner, dtype=cutlass.BFloat16), layout.outer
    )


@dataclass(frozen=True)
class _Smem:
    """Base pointers of the shared buffers (see ``storage.SharedStorage``)."""

    q: cute.Pointer
    k: cute.Pointer
    v: cute.Pointer
    do: cute.Pointer
    h: cute.Pointer
    akk: cute.Pointer
    aqk: cute.Pointer
    sq: cute.Pointer
    ds: cute.Pointer
    ka: cute.Pointer
    kb: cute.Pointer
    qa: cute.Pointer
    t21: cute.Pointer


class FusedKdaBwdKernel:
    """Warp-specialized SM100 fused KDA backward, ``split`` CTAs per (sequence, head)."""

    threads_per_cta = _THREADS

    def __init__(
        self,
        *,
        heads: int,
        split: int = 1,
        tape: bool = False,
        debug: bool = False,
        trace: bool = False,
    ):
        if heads <= 0:
            raise ValueError("heads must be positive")
        if split not in (1, 2):
            raise ValueError("split must be 1 or 2")
        self.heads = heads
        self.split = split
        # tape: dS per chunk and dv_new come from a separate recurrence, so the
        # chunks are independent and a runtime number of CTAs (``ways``) shares
        # each (sequence, head); ``split`` is then unused.
        self.tape = tape
        self.debug = debug
        self.trace = trace
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

    def _ctas(self, ways):
        """CTAs per (sequence, head): runtime with a dS tape, else the compiled ``split``."""
        if cutlass.const_expr(self.tape):
            return ways
        return self.split

    # ------------------------------------------------------------------ host

    @staticmethod
    def _token_matrix(tensor):
        """[1, N, H, D] -> (N, D, (1, H)) for TMA tiling."""
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[1], tensor.shape[3], (tensor.shape[0], tensor.shape[2])),
                stride=(tensor.stride[1], tensor.stride[3], (tensor.stride[0], tensor.stride[2])),
            ),
        )

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

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gc: cute.Tensor,
        beta: cute.Tensor,
        aqk: cute.Tensor,
        akk: cute.Tensor,
        h: cute.Tensor,
        dh_tape: cute.Tensor,
        dv_new: cute.Tensor,
        do: cute.Tensor,
        dht: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        dg: cute.Tensor,
        dbeta: cute.Tensor,
        dh0: cute.Tensor,
        xfer: cute.Tensor,
        debug_out: cute.Tensor,
        trace_out: cute.Tensor,
        scale: cutlass.Float32,
        ways: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        plan = build_kda_layout_plan(self.io_dtype, self.acc_dtype)
        load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        token = plan.variants["mm_64_64_128_k_k"]
        square = plan.variants["mm_64_64_64_k_k"]
        state = plan.variants["mm_128_64_128_k_k"]

        def token_tma(tensor):
            return _wrap_tma(
                cute.nvgpu.make_tiled_tma_atom_A(
                    load_op,
                    self._token_matrix(tensor),
                    token.smem_a,
                    token.spec.tile,
                    token.tiled_mma,
                )
            )

        tma_q, tma_k, tma_v, tma_do = (token_tma(x) for x in (q, k, v, do))
        tma_aqk, tma_akk = (
            _wrap_tma(
                cute.nvgpu.make_tiled_tma_atom_A(
                    load_op,
                    self._token_matrix(x),
                    square.smem_a,
                    square.spec.tile,
                    square.tiled_mma,
                )
            )
            for x in (aqk, akk)
        )
        tma_h, tma_dh = (
            _wrap_tma(
                cute.nvgpu.make_tiled_tma_atom_A(
                    load_op, self._state_matrix(x), state.smem_a, state.spec.tile, state.tiled_mma
                )
            )
            for x in (h, dh_tape)
        )
        tma_dvn = token_tma(dv_new)
        # Tiled MMAs carry runtime accumulate flags and are passed explicitly;
        # the kernel rebuilds the static layouts inside its own region.
        ops = plan.operations
        self.kernel(
            q,
            k,
            v,
            gc,
            beta,
            do,
            dht,
            cu_seqlens,
            chunk_offsets,
            dq,
            dk,
            dv,
            dg,
            dbeta,
            dh0,
            xfer,
            debug_out,
            trace_out,
            scale,
            ways,
            tma_q,
            tma_k,
            tma_v,
            tma_do,
            tma_aqk,
            tma_akk,
            tma_h,
            tma_dh,
            tma_dvn,
            token.tiled_mma,
            square.tiled_mma,
            state.tiled_mma,
            ops["x_state"].tiled_mma,
            ops["dvn_aqk"].tiled_mma,
            ops["vn"].tiled_mma,
            ops["d"].tiled_mma,
            ops["dakk"].tiled_mma,
            ops["inv_left"].tiled_mma,
            ops["inv_right"].tiled_mma,
            ops["dq_state"].tiled_mma,
            ops["dh_k"].tiled_mma,
            ops["intra_lower"].tiled_mma,
            ops["intra_upper"].tiled_mma,
        ).launch(
            # The sequence count is dynamic: ``split`` CTAs per (sequence, head).
            grid=((cute.size(cu_seqlens, mode=[0]) - 1) * self.heads * self._ctas(ways), 1, 1),
            block=(self.threads_per_cta, 1, 1),
            # The two CTAs of a (sequence, head) pass dS to each other.
            cluster=(1 if self.tape else self.split, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ---------------------------------------------------------------- device

    @cute.kernel
    def kernel(
        self,
        q,
        k,
        v,
        gc,
        beta,
        do,
        dht,
        cu_seqlens,
        chunk_offsets,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        dh0,
        xfer,
        debug_out,
        trace_out,
        scale,
        ways,
        tma_q,
        tma_k,
        tma_v,
        tma_do,
        tma_aqk,
        tma_akk,
        tma_h,
        tma_dh,
        tma_dvn,
        mma_token,
        mma_square,
        mma_state,
        mma_x,
        mma_aqk,
        mma_vn,
        mma_d,
        mma_sq_kk,
        mma_sq_mnmn,
        mma_sq_kk64,
        mma_ch128,
        mma_ch64,
        mma_lower,
        mma_upper,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block, _, _ = cute.arch.block_idx()
        # MLIR regions are isolated: rebuild the static layout plan in the
        # kernel so every device-side layout is defined inside it.
        self._plan = build_kda_layout_plan(self.io_dtype, self.acc_dtype)
        canonical = self._plan.canonical
        self._ways = self._ctas(ways)
        part = block % self._ways
        sequence = block // self._ways // self.heads
        head = block // self._ways % self.heads
        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)

        edges = storage.edges.data_ptr()
        ring_full = storage.ring_full.data_ptr()
        ring_empty = storage.ring_empty.data_ptr()
        holding = storage.tmem_holding.data_ptr()
        # Shared by the role methods during tracing (two-CTA dS hand-off).
        self._ds_ready = storage.ds_ready.data_ptr()
        self._tma_dh = tma_dh
        self._tma_dvn = tma_dvn
        self._xfer = xfer
        self._pair = block // self._ways
        self._cpart = part
        s_beta = storage.beta.get_tensor(cute.make_layout((_BT,)))
        s_dbeta = storage.dbeta.get_tensor(cute.make_layout((_BT,)))
        s_xchg = storage.xchg.get_tensor(cute.make_layout((2 * _DK,)))
        s_suffix = storage.suffix.get_tensor(cute.make_layout((_DK,)))
        smem = _Smem(
            storage.sQ.data_ptr(),
            storage.sK.data_ptr(),
            storage.sV.data_ptr(),
            storage.sDo.data_ptr(),
            storage.sH.data_ptr(),
            storage.sAkk.data_ptr(),
            storage.sAqk.data_ptr(),
            storage.sSq.data_ptr(),
            storage.sDS.data_ptr(),
            storage.sKA.data_ptr(),
            storage.sKB.data_ptr(),
            storage.sQA.data_ptr(),
            storage.sT21.data_ptr(),
        )

        if tidx == 0:
            for e in cutlass.range_constexpr(len(_EDGE_COUNTS)):
                cute.arch.mbarrier_init(edges + e, _EDGE_COUNTS[e])
            for s in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(ring_full + s, 1)
                cute.arch.mbarrier_init(ring_empty + s, 8)
            # One arrival per consumer warpgroup of the other CTA.
            cute.arch.mbarrier_init(self._ds_ready, 2)
        cute.arch.mbarrier_init_fence()
        if warp == 0:
            tcgen05_ws.tmem_allocate(holding, TMEM_COLUMNS)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.split == 2 and not self.tape):
            # The other CTA may arrive on DS_READY only after it is initialized.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        tmem = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=holding
        )

        seq_start = cu_seqlens[sequence]
        seq_len = cu_seqlens[sequence + 1] - seq_start
        chunk_base = chunk_offsets[sequence]
        num_chunks = chunk_offsets[sequence + 1] - chunk_base

        if warp < 4:
            cute.arch.setmaxregister_increase(_CH_REGISTERS)
            self.run_ch(
                tidx,
                warp,
                part,
                head,
                sequence,
                seq_start,
                seq_len,
                chunk_base,
                num_chunks,
                smem,
                edges,
                ring_full,
                ring_empty,
                tmem,
                s_beta,
                s_dbeta,
                s_xchg,
                s_suffix,
                q,
                k,
                gc,
                dht,
                dq,
                dk,
                dg,
                dh0,
                debug_out,
                trace_out,
                scale,
                canonical.token_direct,
                canonical.square_direct,
                canonical.state_direct,
                mma_sq_kk,
            )
        elif warp < 8:
            cute.arch.setmaxregister_increase(_TK_REGISTERS)
            self.run_tk(
                tidx - 128,
                warp,
                part,
                head,
                sequence,
                seq_start,
                seq_len,
                chunk_base,
                num_chunks,
                smem,
                edges,
                ring_full,
                ring_empty,
                tmem,
                s_beta,
                s_dbeta,
                s_xchg,
                s_suffix,
                q,
                k,
                gc,
                beta,
                dht,
                dq,
                dk,
                dv,
                dg,
                dbeta,
                debug_out,
                trace_out,
                scale,
                canonical.token_direct,
                canonical.square_direct,
                canonical.state_direct,
                mma_x,
                mma_sq_kk,
            )
        else:
            cute.arch.setmaxregister_decrease(_OTHER_REGISTERS)
            if warp == _MMA_WARP:
                self.run_mma(
                    part,
                    num_chunks,
                    smem,
                    edges,
                    ring_full,
                    ring_empty,
                    tmem,
                    trace_out,
                    mma_x,
                    mma_aqk,
                    mma_vn,
                    mma_d,
                    mma_sq_kk,
                    mma_sq_mnmn,
                    mma_sq_kk64,
                    mma_ch128,
                    mma_ch64,
                    mma_lower,
                    mma_upper,
                )
            elif warp == _TMA_WARP:
                self.run_tma(
                    part,
                    head,
                    seq_start,
                    chunk_base,
                    num_chunks,
                    smem,
                    edges,
                    trace_out,
                    tma_q,
                    tma_k,
                    tma_v,
                    tma_do,
                    tma_aqk,
                    tma_akk,
                    tma_h,
                    mma_token,
                    mma_square,
                    mma_state,
                )
        cute.arch.barrier(barrier_id=_BAR_ALL, number_of_threads=_THREADS)
        if cutlass.const_expr(self.split == 2 and not self.tape):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        if warp == 0:
            cute.arch.relinquish_tmem_alloc_permit()
            tcgen05_ws.tmem_deallocate(holding, TMEM_COLUMNS)

    # ------------------------------------------------------------ helpers

    @cute.jit
    def _wait(self, edges, edge: cutlass.Constexpr, iteration):
        cute.arch.mbarrier_wait(edges + edge, iteration & 1)

    @cute.jit
    def _arrive(self, edges, edge: cutlass.Constexpr, barrier_id: cutlass.Constexpr, local_tidx):
        """All warpgroup threads publish; one elected thread arrives."""
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
        if local_tidx == 0:
            cute.arch.mbarrier_arrive(edges + edge)

    @cute.jit
    def _stamp(self, trace, role: cutlass.Constexpr, slot: cutlass.Constexpr, iteration):
        """Record %globaltimer for CTA 0, first thread of ``role``, first 8 chunks."""
        if cutlass.const_expr(self.trace):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            if block == 0 and tidx == (0, 128, 256, 288)[role] and iteration < 8:
                trace[role, iteration, slot] = _globaltimer()

    @cute.jit
    def _commit(self, ptr):
        with cute.arch.elect_one():
            tcgen05.commit(ptr)

    @staticmethod
    def _acc(tmem, offset, tiled_mma, shape):
        fragment = tiled_mma.make_fragment_C(cute.append(tiled_mma.partition_shape_C(shape), 1))
        return cute.make_tensor(tmem + offset, fragment.layout)

    @cute.jit
    def _token_tile_copy(self, acc, ttidx, shape: cutlass.Constexpr):
        """Ld16x256b partition of an M=64 accumulator plus logical coordinates."""
        atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled = tcgen05.make_tmem_copy(atom, acc[(None, None), 0, 0, 0])
        thr = tiled.get_slice(ttidx)
        src = thr.partition_S(_mn_view(acc))
        crd = thr.partition_D(cute.make_identity_tensor(shape))
        return tiled, src, crd

    # ---------------------------------------------------------------- TMA

    @cute.jit
    def _tma(self, info, smem, tiled_mma, tile, row, coord, bar):
        g = cute.domain_offset((row, cutlass.Int32(0)), info[1][None, None, coord])
        g = tiled_mma.get_slice(0).partition_A(cute.flat_divide(g, tile))
        s_part, g_part = cpasync.tma_partition(
            info[0], 0, cute.make_layout(1), cute.group_modes(smem, 0, 3), cute.group_modes(g, 0, 3)
        )
        cute.copy(info[0], g_part[(None, 0, 0)], s_part[(None, 0)], tma_bar_ptr=bar)

    @cute.jit
    def run_tma(
        self,
        part,
        head,
        seq_start,
        chunk_base,
        num_chunks,
        smem,
        edges,
        trace,
        tma_q,
        tma_k,
        tma_v,
        tma_do,
        tma_aqk,
        tma_akk,
        tma_h,
        mma_token,
        mma_square,
        mma_state,
    ):
        # v, dO, Akk and h are last read by MMA phase D, so the next chunk's
        # copies are issued as soon as it completes; q, k and Aqk (still read
        # by the intra fold, the exact blocks and the epilogue) wait for FREE.
        early_bytes = 2 * _BT * _DK * 2 + _BT * _BT * 2 + _DK * _DV * 2
        staged = self._plan.canonical_staged
        l_token, l_square, l_state = staged.token_direct, staged.square_direct, staged.state_direct
        s_q, s_k, s_v, s_do = (_view(m, l_token) for m in (smem.q, smem.k, smem.v, smem.do))
        s_aqk, s_akk = _view(smem.aqk, l_square), _view(smem.akk, l_square)
        s_h = _view(smem.h, l_state)
        group = (cutlass.Int32(0), head)
        for j in cutlass.range((num_chunks - part + self._ways - 1) // self._ways):
            it = part + j * self._ways
            chunk = num_chunks - 1 - it
            row = seq_start + chunk * _BT
            if j > 0:
                self._wait(edges, _E_DONE_D, j - 1)
            bar = edges + _E_LOAD_EARLY
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar, early_bytes)
            for info, tile in ((tma_v, s_v), (tma_do, s_do)):
                self._tma(info, tile, mma_token, (_BT, _DK), row, group, bar)
            self._tma(tma_akk, s_akk, mma_square, (_BT, _BT), row, group, bar)
            self._tma(
                tma_h,
                s_h,
                mma_state,
                (_DK, _DV),
                cutlass.Int32(0),
                (cutlass.Int32(0), chunk_base + chunk, head),
                bar,
            )
            if j > 0:
                self._wait(edges, _E_FREE, j - 1)
            self._tma_late(
                smem,
                edges,
                tma_q,
                tma_k,
                tma_aqk,
                mma_token,
                mma_square,
                mma_state,
                row,
                chunk_base + chunk,
                group,
            )

    @cute.jit
    def _tma_late(
        self,
        smem,
        edges,
        tma_q,
        tma_k,
        tma_aqk,
        mma_token,
        mma_square,
        mma_state,
        row,
        gchunk,
        group,
    ):
        """q, k and Aqk of one chunk on the LOAD edge."""
        late_bytes = 2 * _BT * _DK * 2 + _BT * _BT * 2
        if cutlass.const_expr(self.tape):
            late_bytes += _DK * _DV * 2 + _BT * _DV * 2
        staged = self._plan.canonical_staged
        s_q, s_k = (_view(m, staged.token_direct) for m in (smem.q, smem.k))
        s_aqk = _view(smem.aqk, staged.square_direct)
        bar = edges + _E_LOAD
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, late_bytes)
        for info, tile in ((tma_q, s_q), (tma_k, s_k)):
            self._tma(info, tile, mma_token, (_BT, _DK), row, group, bar)
        self._tma(tma_aqk, s_aqk, mma_square, (_BT, _BT), row, group, bar)
        if cutlass.const_expr(self.tape):
            # dS of this chunk (the state-tape row) and dv_new: both buffers are
            # read by the previous chunk's intra MMAs, so they wait for FREE too.
            self._tma(
                self._tma_dvn,
                _view(smem.t21, staged.token_direct),
                mma_token,
                (_BT, _DV),
                row,
                group,
                bar,
            )
            self._tma(
                self._tma_dh,
                _view(smem.ds, staged.state_direct),
                mma_state,
                (_DK, _DV),
                cutlass.Int32(0),
                (cutlass.Int32(0), gchunk, group[1]),
                bar,
            )

    # ---------------------------------------------------------------- MMA

    @cute.jit
    def _gemm(
        self,
        tiled_mma,
        acc,
        a,
        b,
        first_accumulate: cutlass.Constexpr,
        k_phase: cutlass.Constexpr = -1,
    ):
        frag_a = tiled_mma.make_fragment_A(a)
        frag_b = tiled_mma.make_fragment_B(b)
        phases = cute.size(frag_a, mode=[2])
        for kp in cutlass.range_constexpr(phases):
            if cutlass.const_expr(k_phase < 0 or kp == k_phase):
                accumulate = first_accumulate if (k_phase >= 0 or kp == 0) else True
                tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate)
                cute.gemm(
                    tiled_mma,
                    acc[None, None, None, 0],
                    frag_a[None, None, kp, 0],
                    frag_b[None, None, kp, 0],
                    acc[None, None, None, 0],
                )

    @cute.jit
    def run_mma(
        self,
        part,
        num_chunks,
        smem,
        edges,
        ring_full,
        ring_empty,
        tmem,
        trace,
        mma_x,
        mma_aqk,
        mma_vn,
        mma_d,
        mma_sq_kk,
        mma_sq_mnmn,
        mma_sq_kk64,
        mma_ch128,
        mma_ch64,
        mma_lower,
        mma_upper,
    ):
        operand = self._operand
        half = _BT * (_DV // 2)
        token_tile = _BT * _DK
        slot_a = self._acc(tmem, tmem_offset("slot_a"), mma_x, (_BT, _DV))
        slot_b = self._acc(tmem, tmem_offset("slot_b"), mma_x, (_BT, _DV))
        sq_dakk = self._acc(tmem, tmem_offset("dk_minus"), mma_sq_kk, (_BT, _BT))
        sq_a = self._acc(tmem, tmem_offset("slot_a"), mma_sq_kk, (_BT, _BT))
        sq_b = self._acc(tmem, tmem_offset("slot_b"), mma_sq_kk, (_BT, _BT))
        dh_halves = (
            self._acc(tmem, tmem_offset("dh"), mma_ch64, (_DK, _BT)),
            self._acc(tmem, tmem_offset("dh") + 64, mma_ch64, (_DK, _BT)),
        )
        acc_dq = self._acc(tmem, tmem_offset("dq"), mma_ch128, (_DK, _BT))
        acc_dkp = self._acc(tmem, tmem_offset("dk_plus"), mma_ch128, (_DK, _BT))
        acc_dkm = self._acc(tmem, tmem_offset("dk_minus"), mma_ch128, (_DK, _BT))
        ring = (
            self._acc(tmem, tmem_offset("ring"), mma_lower, (_DK, _BT)),
            self._acc(tmem, tmem_offset("ring") + 64, mma_lower, (_DK, _BT)),
        )

        for j in cutlass.range((num_chunks - part + self._ways - 1) // self._ways):
            # This CTA's chunks; j indexes them for the one-shot edges.
            it = part + j * self._ways
            # Every CTA of a (sequence, head) runs the dS recursion; only the
            # chunk's owner runs the intra-chunk and dq/dk work (phases C-E).
            owned = True
            ring_base = j * 6
            # Phase A: X = kG S; dvn = kgL dS + Aqk^T dO; dH += (q G scale)^T dO.
            self._wait(edges, _E_READY_A, j)
            self._stamp(trace, 2, 0, j)
            self._wait(edges, _E_LOAD_EARLY, j)
            self._wait(edges, _E_LOAD, j)
            self._gemm(
                mma_x,
                slot_a,
                operand("x_state", smem.ka, "a"),
                operand("x_state", smem.h, "b"),
                False,
            )
            if cutlass.const_expr(not self.tape):
                # dv_new and the dS recurrence (a tape supplies both).
                self._gemm(
                    mma_x,
                    slot_b,
                    operand("dvn_state", smem.kb, "a"),
                    operand("dvn_state", smem.ds, "b"),
                    False,
                )
                self._gemm(
                    mma_aqk,
                    slot_b,
                    operand("dvn_aqk", smem.aqk, "a"),
                    operand("dvn_aqk", smem.do, "b"),
                    True,
                )
                for hh in cutlass.range_constexpr(2):
                    self._gemm(
                        mma_ch64,
                        dh_halves[hh],
                        operand("dh_q", smem.qa, "a"),
                        operand("dh_q", smem.do, "b", hh * half),
                        True,
                    )
            self._commit(edges + _E_DONE_A)
            self._stamp(trace, 2, 1, j)
            # Phase B: vn = (Akk beta) Delta; D = Akk^T dvn; dAkk = dvn Delta^T.
            self._wait(edges, _E_READY_B, j)
            self._stamp(trace, 2, 2, j)
            self._gemm(
                mma_d, slot_b, operand("d", smem.akk, "a"), operand("d", smem.t21, "b"), False
            )
            # dAkk stays unconditional: mma_sq_kk is reconfigured again in phase C.
            self._gemm(
                mma_sq_kk,
                sq_dakk,
                operand("dakk", smem.t21, "a"),
                operand("dakk", smem.v, "b"),
                False,
            )
            if owned:
                self._gemm(
                    mma_vn, slot_a, operand("vn", smem.sq, "a"), operand("vn", smem.v, "b"), False
                )
            self._commit(edges + _E_DONE_B)
            self._stamp(trace, 2, 3, j)
            # Phase C: dAqk, state-path token gradients, dh_k, inverse left.
            self._wait(edges, _E_READY_C, j)
            self._stamp(trace, 2, 4, j)
            if owned:
                self._gemm(
                    mma_sq_kk,
                    sq_b,
                    operand("daqk", smem.do, "a"),
                    operand("daqk", smem.t21, "b"),
                    False,
                )
                self._gemm(
                    mma_ch128,
                    acc_dkm,
                    operand("dk_state", smem.ds, "a"),
                    operand("dk_state", smem.t21, "b"),
                    False,
                )
                self._gemm(
                    mma_ch128,
                    acc_dq,
                    operand("dq_state", smem.h, "a"),
                    operand("dq_state", smem.do, "b"),
                    False,
                )
                self._gemm(
                    mma_ch128,
                    acc_dkp,
                    operand("dk_w", smem.h, "a"),
                    operand("dk_w", smem.kb, "b"),
                    False,
                )
            if cutlass.const_expr(not self.tape):
                for hh in cutlass.range_constexpr(2):
                    self._gemm(
                        mma_ch64,
                        dh_halves[hh],
                        operand("dh_k", smem.ka, "a"),
                        operand("dh_k", smem.kb, "b", hh * half),
                        True,
                    )
            if owned:
                self._gemm(
                    mma_sq_mnmn,
                    sq_a,
                    operand("inv_left", smem.akk, "a"),
                    operand("inv_left", smem.sq, "b"),
                    False,
                )
            self._commit(edges + _E_DONE_C)
            self._stamp(trace, 2, 5, j)
            # Phase D: T Akk^T (negated and masked by TK).
            self._wait(edges, _E_READY_D, j)
            self._stamp(trace, 2, 6, j)
            if owned:
                self._gemm(
                    mma_sq_kk64,
                    sq_a,
                    operand("inv_right", smem.sq, "a"),
                    operand("inv_right", smem.akk, "b"),
                    False,
                )
            self._commit(edges + _E_DONE_D)
            self._stamp(trace, 2, 7, j)
            # Phase E: intra ring (lower dq x4, lower k x4, upper dk x4).
            self._wait(edges, _E_READY_E, j)
            self._stamp(trace, 2, 8, j)
            if owned:
                for r in cutlass.range_constexpr(12):
                    slot = r % 2
                    use = ring_base + r // 2
                    if use > 0:
                        cute.arch.mbarrier_wait(ring_empty + slot, (use - 1) & 1)
                    block = r % 4
                    if cutlass.const_expr(r < 8):
                        square = smem.aqk if r < 4 else smem.sq
                        b = operand("intra_lower", square, "b")
                        self._gemm(
                            mma_lower,
                            ring[slot],
                            operand("intra_lower", smem.ds, "a"),
                            b,
                            False,
                            block,
                        )
                        self._gemm(
                            mma_lower,
                            ring[slot],
                            operand("intra_lower", smem.ds, "a", token_tile),
                            b,
                            True,
                            block,
                        )
                    else:
                        b_q = operand("intra_upper", smem.aqk, "b")
                        b_k = operand("intra_upper", smem.sq, "b")
                        self._gemm(
                            mma_upper,
                            ring[slot],
                            operand("intra_upper", smem.ka, "a"),
                            b_q,
                            False,
                            block,
                        )
                        self._gemm(
                            mma_upper,
                            ring[slot],
                            operand("intra_upper", smem.kb, "a"),
                            b_q,
                            True,
                            block,
                        )
                        self._gemm(
                            mma_upper,
                            ring[slot],
                            operand("intra_upper", smem.qa, "a"),
                            b_k,
                            True,
                            block,
                        )
                        self._gemm(
                            mma_upper,
                            ring[slot],
                            operand("intra_upper", smem.t21, "a"),
                            b_k,
                            True,
                            block,
                        )
                    self._commit(ring_full + slot)
            self._commit(edges + _E_DONE_E)
            self._stamp(trace, 2, 9, j)

    def _operand(self, name, ptr, which, offset=0):
        """Staged operand view of one shared buffer for MMA ``name``."""
        layouts = self._plan.operations[name].operands
        return _view(ptr, layouts.a_staged if which == "a" else layouts.b_staged, offset)

    # ----------------------------------------------------------------- TK

    @cute.jit
    def run_tk(
        self,
        ttidx,
        warp,
        part,
        head,
        sequence,
        seq_start,
        seq_len,
        chunk_base,
        num_chunks,
        smem,
        edges,
        ring_full,
        ring_empty,
        tmem,
        s_beta,
        s_dbeta,
        s_xchg,
        s_suffix,
        q,
        k,
        gc,
        beta,
        dht,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        debug_out,
        trace,
        scale,
        l_token,
        l_square,
        l_state,
        mma_x,
        mma_sq_kk,
    ):
        """Token-major warpgroup (value side, squares, dv/dbeta) plus the
        channel-major work for tokens 32..63 (``half=1``)."""
        s_do = _mn_view(_view(smem.do, l_token))
        s_v = _mn_view(_view(smem.v, l_token))
        s_t21 = _mn_view(_view(smem.t21, l_token))
        s_kb = _mn_view(_view(smem.kb, l_token))
        s_aqk = _mn_view(_view(smem.aqk, l_square))
        s_akk = _mn_view(_view(smem.akk, l_square))
        s_sq = _mn_view(_view(smem.sq, l_square))
        slot_a = self._acc(tmem, tmem_offset("slot_a"), mma_x, (_BT, _DV))
        slot_b = self._acc(tmem, tmem_offset("slot_b"), mma_x, (_BT, _DV))
        sq_dakk = self._acc(tmem, tmem_offset("dk_minus"), mma_sq_kk, (_BT, _BT))
        tok_copy, tok_a, tok_crd = self._token_tile_copy(slot_a, ttidx, (_BT, _DV))
        _, tok_b, _ = self._token_tile_copy(slot_b, ttidx, (_BT, _DV))
        sq_copy, sq_src_dakk, sq_crd = self._token_tile_copy(sq_dakk, ttidx, (_BT, _BT))
        regs = cute.make_rmem_tensor_like(tok_crd, self.acc_dtype)
        sregs = cute.make_rmem_tensor_like(sq_crd, self.acc_dtype)
        sq_out, t_sq_out, w_sq = self._square_stmatrix(sq_copy, sregs, ttidx, smem.sq, l_square)
        # Vectorized shared access in the TMEM-load fragment layout (as GDR):
        # stmatrix for register -> shared, 32-bit LDS for shared -> register.
        staged_token = self._plan.canonical_staged.token_direct
        r_out = cute.make_rmem_tensor_like(regs, self.io_dtype)
        tiled_r2s = cute.make_tiled_copy_D(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False), self.io_dtype
            ),
            tok_copy,
        )
        thr_r2s = tiled_r2s.get_slice(ttidx)
        t_r_out = tiled_r2s.retile(r_out)
        w_v = thr_r2s.partition_D(_mn_view(_view(smem.v, staged_token)))
        w_t21 = thr_r2s.partition_D(_mn_view(_view(smem.t21, staged_token)))
        w_kb = thr_r2s.partition_D(_mn_view(_view(smem.kb, staged_token)))
        r_in = cute.make_rmem_tensor_like(regs, self.io_dtype)
        tiled_s2r = cute.make_tiled_copy_D(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.io_dtype, num_bits_per_copy=32),
            tok_copy,
        )
        thr_s2r = tiled_s2r.get_slice(ttidx)
        t_r_in = tiled_s2r.retile(r_in)
        rd_v = thr_s2r.partition_S(_mn_view(_view(smem.v, staged_token)))
        # gate, two scratch columns, dbeta partials, upper/lower anchors
        cm = (
            cute.make_rmem_tensor((32,), cutlass.Float32),
            cute.make_rmem_tensor((32,), cutlass.Float32),
            cute.make_rmem_tensor((32,), cutlass.Float32),
            cute.make_rmem_tensor((32,), cutlass.Float32),
            cute.make_rmem_tensor((4,), cutlass.Float32),
            cute.make_rmem_tensor((4,), cutlass.Float32),
            cute.make_rmem_tensor((32,), cutlass.Float32),
        )
        lane_base = cutlass.Int32(tmem.toint()) + ((warp % 4) * 32 << 16)
        # Rows owned by this thread in the Ld16x256b layout of an M=64 tile.
        row0 = (ttidx // 32) * 16 + (ttidx % 32) // 4

        for j in cutlass.range((num_chunks - part + self._ways - 1) // self._ways):
            # This CTA's chunks; j indexes them for the one-shot edges.
            it = part + j * self._ways
            chunk = num_chunks - 1 - it
            tok0 = seq_start + chunk * _BT
            valid = cutlass.min(_BT, seq_len - chunk * _BT)
            gchunk = chunk_base + chunk
            owned = True
            # Scalars assigned only on owned chunks keep one type across the dynamic branches.
            t_o, col_o = cutlass.Int32(0), cutlass.Int32(0)
            x_o, delta_o, d_o, contrib_o = (cutlass.Float32(0.0),) * 4
            # Phase A: beta, tails and triangular masks; Akk * diag(beta).
            if ttidx < _BT:
                s_beta[ttidx] = (
                    beta[0, tok0 + ttidx, head] if ttidx < valid else cutlass.Float32(0.0)
                )
            # The gate column and the state seed do not read q, k or Aqk, so
            # they run before the late loads land (inside _cm_phase_a).
            self._load_gate_column(1, gc, tok0, valid, head, ttidx, cm)
            self._wait(edges, _E_LOAD_EARLY, j)
            self._stamp(trace, 1, 0, j)
            cute.arch.barrier(barrier_id=_BAR_TK, number_of_threads=128)
            if valid < _BT:
                for idx in cutlass.range(ttidx, _BT * _DV, 128):
                    t = idx // _DV
                    if t >= valid:
                        s_do[t, idx % _DV] = self.io_dtype(0.0)
            # Akk mask and Akk * diag(beta); the CH warpgroup masks Aqk.
            self._mask_square(1, ttidx, valid, smem, s_beta)
            dg_part = self._cm_phase_a(
                1,
                j,
                it,
                ttidx,
                sequence,
                head,
                tok0,
                valid,
                lane_base,
                smem,
                edges,
                q,
                k,
                gc,
                dht,
                scale,
                l_token,
                l_state,
                cm,
                s_beta,
                trace,
            )
            self._arrive(edges, _E_READY_A, _BAR_TK, ttidx)
            self._stamp(trace, 1, 1, j)

            # Phase B: Delta = v - X (in place), dvn -> sT21; -(k beta G) rows 32..63.
            self._wait(edges, _E_DONE_A, j)
            self._stamp(trace, 1, 2, j)
            # Delta feeds only vn, dAkk and dbeta, which the chunk's owner computes.
            for sub in cutlass.range_constexpr(cute.size(regs, mode=[2])):
                if owned:
                    cute.copy(tok_copy, tok_a[None, 0, sub, 0], regs[None, 0, sub])
                    cute.copy(tiled_s2r, rd_v[None, 0, sub, 0], t_r_in[None, 0, sub])
                    for e in cutlass.range_constexpr(cute.size(regs, mode=[0])):
                        t_o, col_o = tok_crd[e, 0, sub]
                        x_o = regs[e, 0, sub]
                        if cutlass.const_expr(self.debug):
                            debug_out[gchunk, head, 0, t_o, col_o] = x_o
                        delta_o = r_in[e, 0, sub].to(self.acc_dtype) - x_o
                        r_out[e, 0, sub] = (delta_o if t_o < valid else cutlass.Float32(0.0)).to(
                            self.io_dtype
                        )
                    cute.copy(tiled_r2s, t_r_out[None, 0, sub], w_v[None, 0, sub, 0])
                if cutlass.const_expr(not self.tape):
                    cute.copy(tok_copy, tok_b[None, 0, sub, 0], regs[None, 0, sub])
                    if cutlass.const_expr(self.debug):
                        for e in cutlass.range_constexpr(cute.size(regs, mode=[0])):
                            t_o, col_o = tok_crd[e, 0, sub]
                            debug_out[gchunk, head, 1, t_o, col_o] = regs[e, 0, sub]
                    r_out[None, 0, sub].store(regs[None, 0, sub].load().to(self.io_dtype))
                    cute.copy(tiled_r2s, t_r_out[None, 0, sub], w_t21[None, 0, sub, 0])
            cute.arch.fence_view_async_tmem_load()
            self._arrive(edges, _E_READY_B, _BAR_TK, ttidx)
            self._stamp(trace, 1, 3, j)
            # MMA B does not read sKA; -(k beta G) is published with READY_C.
            self._cm_phase_b(1, ttidx, head, tok0, valid, smem, gc, s_beta)
            if not owned:
                self._arrive(edges, _E_FREE, _BAR_TK, ttidx)

            # Phase C: vn -> sT21; D -> dv, dbeta(row), sKB; dAkk -> sSq.
            self._wait(edges, _E_DONE_B, j)
            self._stamp(trace, 1, 4, j)
            part0 = cutlass.Float32(0.0)
            part1 = cutlass.Float32(0.0)
            for sub in cutlass.range_constexpr(cute.size(regs, mode=[2])):
                if owned:
                    cute.copy(tok_copy, tok_a[None, 0, sub, 0], regs[None, 0, sub])
                    if cutlass.const_expr(self.debug):
                        for e in cutlass.range_constexpr(cute.size(regs, mode=[0])):
                            t_o, col_o = tok_crd[e, 0, sub]
                            debug_out[gchunk, head, 2, t_o, col_o] = regs[e, 0, sub]
                    r_out[None, 0, sub].store(regs[None, 0, sub].load().to(self.io_dtype))
                    cute.copy(tiled_r2s, t_r_out[None, 0, sub], w_t21[None, 0, sub, 0])
                cute.copy(tok_copy, tok_b[None, 0, sub, 0], regs[None, 0, sub])
                if owned:
                    cute.copy(tiled_s2r, rd_v[None, 0, sub, 0], t_r_in[None, 0, sub])
                    for e in cutlass.range_constexpr(cute.size(regs, mode=[0])):
                        t_o, col_o = tok_crd[e, 0, sub]
                        d_o = regs[e, 0, sub]
                        if cutlass.const_expr(self.debug):
                            debug_out[gchunk, head, 3, t_o, col_o] = d_o
                        contrib_o = (
                            d_o * r_in[e, 0, sub].to(self.acc_dtype)
                            if t_o < valid
                            else cutlass.Float32(0.0)
                        )
                        if t_o == row0:
                            part0 += contrib_o
                        else:
                            part1 += contrib_o
                        r_in[e, 0, sub] = (s_beta[t_o] * d_o).to(self.io_dtype)
                r_out[None, 0, sub].store(regs[None, 0, sub].load().to(self.io_dtype))
                cute.copy(tiled_r2s, t_r_out[None, 0, sub], w_kb[None, 0, sub, 0])
            if owned:
                for sub in cutlass.range_constexpr(cute.size(sregs, mode=[2])):
                    cute.copy(sq_copy, sq_src_dakk[None, 0, sub, 0], sregs[None, 0, sub])
                    for e in cutlass.range_constexpr(cute.size(sregs, mode=[0])):
                        row_i, col_j = sq_crd[e, 0, sub]
                        value = sregs[e, 0, sub] * s_beta[col_j]
                        keep = (col_j < row_i) and (row_i < valid)
                        if cutlass.const_expr(self.debug):
                            debug_out[gchunk, head, 5, row_i, col_j] = (
                                value if keep else cutlass.Float32(0.0)
                            )
                        sq_out[e, 0, sub] = (value if keep else cutlass.Float32(0.0)).to(
                            self.io_dtype
                        )
                    cute.copy(t_sq_out[0], t_sq_out[1][None, 0, sub], w_sq[None, 0, sub])
            cute.arch.fence_view_async_tmem_load()
            self._arrive(edges, _E_READY_C, _BAR_TK, ttidx)
            self._stamp(trace, 1, 5, j)
            # Off the MMA path: stage dv = beta * D in sV (Delta is consumed;
            # the next v load waits READY_D), store whole rows, and seed dbeta
            # with the row sums (each row spread over one aligned quad).
            if owned:
                for sub in cutlass.range_constexpr(cute.size(regs, mode=[2])):
                    r_out[None, 0, sub].store(r_in[None, 0, sub].load())
                    cute.copy(tiled_r2s, t_r_out[None, 0, sub], w_v[None, 0, sub, 0])
                part0 = cute.arch.warp_reduction_sum(part0, threads_in_group=4)
                part1 = cute.arch.warp_reduction_sum(part1, threads_in_group=4)
                if ttidx % 4 == 0:
                    s_dbeta[row0] = part0
                    s_dbeta[row0 + 8] = part1
                cute.arch.barrier(barrier_id=_BAR_TK, number_of_threads=128)
                self._store_rows(dv, smem.v, ttidx, head, tok0, valid)

            # Phase D: T = Akk^T dAkk -> sSq (the only operand MMA D waits for).
            self._wait(edges, _E_DONE_C, j)
            self._send_ds(1, ttidx, it, num_chunks, lane_base)
            self._stamp(trace, 1, 6, j)
            if owned:
                self._square_to_smem(
                    0,
                    ttidx,
                    valid,
                    tmem,
                    mma_sq_kk,
                    smem.sq,
                    l_square,
                    scale,
                    debug_out,
                    gchunk,
                    head,
                )
            self._arrive(edges, _E_READY_D, _BAR_TK, ttidx)
            self._stamp(trace, 1, 7, j)
            if owned:
                # Channel-major state-path scaling and anchored tiles, tokens 32..63.
                self._cm_phase_c(
                    1,
                    gchunk,
                    ttidx,
                    head,
                    tok0,
                    valid,
                    lane_base,
                    smem,
                    gc,
                    s_beta,
                    debug_out,
                    scale,
                    l_token,
                    cm,
                )
                self._stamp(trace, 1, 8, j)
                # dAqk -> sAqk (read only by the intra ring; the CH warpgroup writes dL).
                self._square_to_smem(
                    1,
                    ttidx,
                    valid,
                    tmem,
                    mma_sq_kk,
                    smem.aqk,
                    l_square,
                    scale,
                    debug_out,
                    gchunk,
                    head,
                )
            self._arrive(edges, _E_READY_E, _BAR_TK, ttidx)
            self._stamp(trace, 1, 9, j)

            if owned:
                dg_part += self._cm_fold(
                    1,
                    j,
                    ttidx,
                    head,
                    tok0,
                    valid,
                    lane_base,
                    smem,
                    ring_full,
                    ring_empty,
                    gc,
                    s_beta,
                    s_dbeta,
                    l_token,
                    l_square,
                    cm,
                    scale,
                    debug_out,
                    gchunk,
                )
                self._cm_epilogue(
                    1,
                    ttidx,
                    head,
                    tok0,
                    valid,
                    lane_base,
                    dg_part,
                    s_xchg,
                    s_suffix,
                    smem,
                    q,
                    k,
                    dq,
                    dk,
                    dg,
                    l_token,
                    cm,
                    trace,
                    j,
                )
                if ttidx < valid:
                    dbeta[0, tok0 + ttidx, head] = s_dbeta[ttidx]
            if owned:
                self._arrive(edges, _E_FREE, _BAR_TK, ttidx)
            self._stamp(trace, 1, 10, j)

    # ----------------------------------------------------------------- CH

    @cute.jit
    def run_ch(
        self,
        tidx,
        warp,
        part,
        head,
        sequence,
        seq_start,
        seq_len,
        chunk_base,
        num_chunks,
        smem,
        edges,
        ring_full,
        ring_empty,
        tmem,
        s_beta,
        s_dbeta,
        s_xchg,
        s_suffix,
        q,
        k,
        gc,
        dht,
        dq,
        dk,
        dg,
        dh0,
        debug_out,
        trace,
        scale,
        l_token,
        l_square,
        l_state,
        mma_sq_kk,
    ):
        """Channel-major warpgroup for tokens 0..31 (``half=0``); also stores
        dh0 and the strictly lower dL square for the intra ring."""
        # gate, two scratch columns, dbeta partials, upper/lower anchors
        cm = (
            cute.make_rmem_tensor((32,), cutlass.Float32),
            cute.make_rmem_tensor((32,), cutlass.Float32),
            cute.make_rmem_tensor((32,), cutlass.Float32),
            cute.make_rmem_tensor((32,), cutlass.Float32),
            cute.make_rmem_tensor((4,), cutlass.Float32),
            cute.make_rmem_tensor((4,), cutlass.Float32),
            cute.make_rmem_tensor((32,), cutlass.Float32),
        )
        lane_base = cutlass.Int32(tmem.toint()) + ((warp % 4) * 32 << 16)
        for j in cutlass.range((num_chunks - part + self._ways - 1) // self._ways):
            # This CTA's chunks; j indexes them for the one-shot edges.
            it = part + j * self._ways
            chunk = num_chunks - 1 - it
            tok0 = seq_start + chunk * _BT
            valid = cutlass.min(_BT, seq_len - chunk * _BT)
            gchunk = chunk_base + chunk
            owned = True
            self._load_gate_column(0, gc, tok0, valid, head, tidx, cm)
            self._wait(edges, _E_LOAD_EARLY, j)
            self._stamp(trace, 0, 0, j)
            dg_part = self._cm_phase_a(
                0,
                j,
                it,
                tidx,
                sequence,
                head,
                tok0,
                valid,
                lane_base,
                smem,
                edges,
                q,
                k,
                gc,
                dht,
                scale,
                l_token,
                l_state,
                cm,
                s_beta,
                trace,
            )
            self._arrive(edges, _E_READY_A, _BAR_CH, tidx)
            self._stamp(trace, 0, 2, j)
            self._wait(edges, _E_DONE_A, j)
            self._stamp(trace, 0, 3, j)
            self._cm_phase_b(0, tidx, head, tok0, valid, smem, gc, s_beta)
            self._arrive(edges, _E_READY_B, _BAR_CH, tidx)
            if not owned:
                # q, k and Aqk are not read again on a chunk this CTA does not own.
                self._arrive(edges, _E_FREE, _BAR_CH, tidx)
            self._stamp(trace, 0, 4, j)
            self._wait(edges, _E_DONE_C, j)
            self._send_ds(0, tidx, it, num_chunks, lane_base)
            self._stamp(trace, 0, 5, j)
            if cutlass.const_expr(self.tape):
                pass  # the recurrence that produced the tape also produced dh0
            elif it == num_chunks - 1:
                buf = cm[1]
                for part in cutlass.range_constexpr(_DV // 32):
                    _tmem_ld32(lane_base + tmem_offset("dh") + part * 32, buf)
                    for c in cutlass.range_constexpr(32):
                        dh0[sequence, head, tidx, part * 32 + c] = buf[c]
            if owned:
                self._cm_phase_c(
                    0,
                    gchunk,
                    tidx,
                    head,
                    tok0,
                    valid,
                    lane_base,
                    smem,
                    gc,
                    s_beta,
                    debug_out,
                    scale,
                    l_token,
                    cm,
                )
                # Phase E: dL = -(T Akk^T), strictly lower and valid -> sSq.
                self._wait(edges, _E_DONE_D, j)
                self._square_to_smem(
                    2,
                    tidx,
                    valid,
                    tmem,
                    mma_sq_kk,
                    smem.sq,
                    l_square,
                    scale,
                    debug_out,
                    gchunk,
                    head,
                )
            self._arrive(edges, _E_READY_E, _BAR_CH, tidx)
            self._stamp(trace, 0, 6, j)
            if owned:
                dg_part += self._cm_fold(
                    0,
                    j,
                    tidx,
                    head,
                    tok0,
                    valid,
                    lane_base,
                    smem,
                    ring_full,
                    ring_empty,
                    gc,
                    s_beta,
                    s_dbeta,
                    l_token,
                    l_square,
                    cm,
                    scale,
                    debug_out,
                    gchunk,
                )
                self._stamp(trace, 0, 7, j)

                self._cm_epilogue(
                    0,
                    tidx,
                    head,
                    tok0,
                    valid,
                    lane_base,
                    dg_part,
                    s_xchg,
                    s_suffix,
                    smem,
                    q,
                    k,
                    dq,
                    dk,
                    dg,
                    l_token,
                    cm,
                    trace,
                    j,
                )
            if owned:
                self._arrive(edges, _E_FREE, _BAR_CH, tidx)
            self._stamp(trace, 0, 8, j)

    # ------------------------------------------------- channel-major halves

    @cute.jit
    def _square_to_smem(
        self,
        kind: cutlass.Constexpr,
        ttidx,
        valid,
        tmem,
        mma_sq_kk,
        out_ptr,
        l_square,
        scale,
        debug_out,
        gchunk,
        head,
    ):
        """Copy one 64x64 FP32 TMEM square to a BF16 shared square.

        kind 0: T (slot_a) as is -> sSq.
        kind 1: dAqk (slot_b) * scale, kept on j <= i < valid -> sAqk.
        kind 2: dL = -(T Akk^T) (slot_a), kept on j < i < valid -> sSq.
        """
        offset = tmem_offset("slot_b") if kind == 1 else tmem_offset("slot_a")
        square = self._acc(tmem, offset, mma_sq_kk, (_BT, _BT))
        sq_copy, sq_src, sq_crd = self._token_tile_copy(square, ttidx, (_BT, _BT))
        sregs = cute.make_rmem_tensor_like(sq_crd, self.acc_dtype)
        r_out, t_r_out, w_out = self._square_stmatrix(sq_copy, sregs, ttidx, out_ptr, l_square)
        for sub in cutlass.range_constexpr(cute.size(sregs, mode=[2])):
            cute.copy(sq_copy, sq_src[None, 0, sub, 0], sregs[None, 0, sub])
            for e in cutlass.range_constexpr(cute.size(sregs, mode=[0])):
                i, j = sq_crd[e, 0, sub]
                value = sregs[e, 0, sub]
                if cutlass.const_expr(kind == 1):
                    keep = (j <= i) and (i < valid)
                    value = value * scale if keep else cutlass.Float32(0.0)
                    if cutlass.const_expr(self.debug):
                        debug_out[gchunk, head, 4, i, j] = value
                if cutlass.const_expr(kind == 2):
                    keep = (j < i) and (i < valid)
                    value = -value if keep else cutlass.Float32(0.0)
                    if cutlass.const_expr(self.debug):
                        debug_out[gchunk, head, 6, i, j] = value
                r_out[e, 0, sub] = value.to(self.io_dtype)
            cute.copy(t_r_out[0], t_r_out[1][None, 0, sub], w_out[None, 0, sub])
        cute.arch.fence_view_async_tmem_load()

    def _square_stmatrix(self, sq_copy, sregs, ttidx, out_ptr, l_square):
        """stmatrix staging of a 64x64 BF16 square in the TMEM-load fragment
        layout: (registers, (tiled copy, retiled registers), shared partition)."""
        r_out = cute.make_rmem_tensor_like(sregs, self.io_dtype)
        tiled = cute.make_tiled_copy_D(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False), self.io_dtype
            ),
            sq_copy,
        )
        w_out = tiled.get_slice(ttidx).partition_D(_mn_view(_view(out_ptr, l_square)))
        return r_out, (tiled, tiled.retile(r_out)), w_out

    @cute.jit
    def _mask_square(self, kind: cutlass.Constexpr, u, valid, smem, s_beta):
        """Vectorized triangular masks on 64x64 BF16 squares (8 columns per access).

        kind 0 (CH): Aqk kept on j <= i < valid.
        kind 1 (TK): Akk kept on j <= i < valid, and sSq = masked Akk * beta_j.
        """
        vec = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        out = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        base_in = cutlass.Int32((smem.aqk if kind == 0 else smem.akk).toint())
        base_sq = cutlass.Int32(smem.sq.toint())
        c0 = (u % 8) * 8
        for i in cutlass.range_constexpr(4):
            row = u // 8 + 16 * i
            address = _swz_token(base_in, row, c0)
            _load_shared_bf16x8(address, vec)
            for e in cutlass.range_constexpr(8):
                keep = (c0 + e <= row) and (row < valid)
                vec[e] = vec[e] if keep else cutlass.BFloat16(0.0)
                if cutlass.const_expr(kind == 1):
                    out[e] = (vec[e].to(self.acc_dtype) * s_beta[c0 + e]).to(self.io_dtype)
            _store_shared_bf16x8(address, vec)
            if cutlass.const_expr(kind == 1):
                _store_shared_bf16x8(_swz_token(base_sq, row, c0), out)

    @cute.jit
    def _store_rows(self, out, base_ptr, u, head, tok0, valid):
        """Copy a 64x128 BF16 token tile from shared memory to global rows of
        ``out`` [1, N, H, 128] with 128-bit accesses (rows past the tail skipped)."""
        base = cutlass.Int32(base_ptr.toint())
        g_base = cutlass.Int64(out.iterator.toint()) + head * _DV * 2
        row_stride = self.heads * _DV * 2
        c0 = (u % 16) * 8
        for i in cutlass.range_constexpr(8):
            row = u // 16 + 8 * i
            if row < valid:
                _copy_bf16x8_s2g(
                    _swz_token(base, row, c0), g_base + (tok0 + row) * row_stride + c0 * 2
                )

    @cute.jit
    def _row_tiles(
        self,
        kind: cutlass.Constexpr,
        half: cutlass.Constexpr,
        u,
        head,
        tok0,
        valid,
        smem,
        gc,
        s_beta,
        scale,
    ):
        """Gate-scaled operand tiles in a token-row mapping (8 channels per vector).

        kind 0: k*G -> sKA, k*2^(G_L-G) -> sKB, scale*q*G -> sQA (phase A).
        kind 1: -(k*beta*G) -> sKA (phase B, after x_state consumed k*G).
        kind 2: anchored BF16 hi/lo pairs for the intra MMAs: k_low -> sDS halves,
                q_up -> sKA/sKB, (beta k)_up -> sQA/sT21.
        Rows past the tail are written as zero.
        """
        kv = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        qv = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        g8 = cute.make_rmem_tensor((8,), cutlass.Float32)
        a8 = cute.make_rmem_tensor((8,), cutlass.Float32)
        b8 = cute.make_rmem_tensor((8,), cutlass.Float32)
        o0 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        o1 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        o2 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        o3 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        o4 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        o5 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        g_base = cutlass.Int64(gc.iterator.toint())
        row_stride = self.heads * _DK * 4
        head_offset = head * _DK * 4
        base_k = cutlass.Int32(smem.k.toint())
        base_q = cutlass.Int32(smem.q.toint())
        base_ka = cutlass.Int32(smem.ka.toint())
        base_kb = cutlass.Int32(smem.kb.toint())
        base_qa = cutlass.Int32(smem.qa.toint())
        base_t21 = cutlass.Int32(smem.t21.toint())
        base_ds = cutlass.Int32(smem.ds.toint())
        c0 = (u % 16) * 8
        for i in cutlass.range_constexpr(4):
            row = half * 32 + u // 16 + 8 * i
            live = row < valid
            _load_shared_bf16x8(_swz_token(base_k, row, c0), kv)
            if cutlass.const_expr(kind != 1):
                _load_shared_bf16x8(_swz_token(base_q, row, c0), qv)
            g_row = g_base + head_offset + c0 * 4
            _ldg_f32x8(g_row + (tok0 + cutlass.min(row, valid - 1)) * row_stride, g8)
            if cutlass.const_expr(kind == 0):
                _ldg_f32x8(g_row + (tok0 + valid - 1) * row_stride, a8)
            if cutlass.const_expr(kind == 2):
                first = (row // _BC) * _BC
                _ldg_f32x8(
                    g_row + (tok0 + cutlass.min(first + _BC - 1, valid - 1)) * row_stride, a8
                )
                _ldg_f32x8(g_row + (tok0 + cutlass.min(first, valid - 1)) * row_stride, b8)
            beta = s_beta[row]
            for e in cutlass.range_constexpr(8):
                k_value = kv[e].to(self.acc_dtype) if live else cutlass.Float32(0.0)
                if cutlass.const_expr(kind == 0):
                    q_value = qv[e].to(self.acc_dtype) if live else cutlass.Float32(0.0)
                    gamma = cute.math.exp2(g8[e], fastmath=True)
                    o0[e] = (k_value * gamma).to(self.io_dtype)
                    o1[e] = (k_value * cute.math.exp2(a8[e] - g8[e], fastmath=True)).to(
                        self.io_dtype
                    )
                    o2[e] = (scale * q_value * gamma).to(self.io_dtype)
                if cutlass.const_expr(kind == 1):
                    o0[e] = (-(k_value * beta) * cute.math.exp2(g8[e], fastmath=True)).to(
                        self.io_dtype
                    )
                if cutlass.const_expr(kind == 2):
                    q_value = qv[e].to(self.acc_dtype) if live else cutlass.Float32(0.0)
                    lower = k_value * cute.math.exp2(a8[e] - g8[e], fastmath=True)
                    upper_scale = cute.math.exp2(g8[e] - b8[e], fastmath=True)
                    upper_q = q_value * upper_scale
                    upper_k = beta * k_value * upper_scale
                    hi = lower.to(self.io_dtype)
                    o0[e] = hi
                    o1[e] = (lower - hi.to(self.acc_dtype)).to(self.io_dtype)
                    hi = upper_q.to(self.io_dtype)
                    o2[e] = hi
                    o3[e] = (upper_q - hi.to(self.acc_dtype)).to(self.io_dtype)
                    hi = upper_k.to(self.io_dtype)
                    o4[e] = hi
                    o5[e] = (upper_k - hi.to(self.acc_dtype)).to(self.io_dtype)
            if cutlass.const_expr(kind == 0):
                _store_shared_bf16x8(_swz_token(base_ka, row, c0), o0)
                _store_shared_bf16x8(_swz_token(base_kb, row, c0), o1)
                _store_shared_bf16x8(_swz_token(base_qa, row, c0), o2)
            if cutlass.const_expr(kind == 1):
                _store_shared_bf16x8(_swz_token(base_ka, row, c0), o0)
            if cutlass.const_expr(kind == 2):
                _store_shared_bf16x8(_swz_token(base_ds, row, c0), o0)
                _store_shared_bf16x8(_swz_token(base_ds + _BT * _DK * 2, row, c0), o1)
                _store_shared_bf16x8(_swz_token(base_ka, row, c0), o2)
                _store_shared_bf16x8(_swz_token(base_kb, row, c0), o3)
                _store_shared_bf16x8(_swz_token(base_qa, row, c0), o4)
                _store_shared_bf16x8(_swz_token(base_t21, row, c0), o5)

    @cute.jit
    def _cm_phase_a(
        self,
        half: cutlass.Constexpr,
        j,
        it,
        d,
        sequence,
        head,
        tok0,
        valid,
        lane_base,
        smem,
        edges,
        q,
        k,
        gc,
        dht,
        scale,
        l_token,
        l_state,
        cm,
        s_beta,
        trace,
    ):
        """Gate column (already loaded into ``cm``) and anchors, state seed
        (value columns of this half), and, once q and k land, the gate-scaled
        K/Q tiles (tokens of this half).  Returns this half's share of the dg
        boundary term."""
        gbuf, buf, _, _, up, lo, _ = cm
        _tmem_st32(lane_base + tmem_offset("gate") + half * 32, gbuf)
        cute.arch.fence_view_async_tmem_store()
        self._stamp(trace, half, 9, j)
        # The gate-scaled tiles and the Aqk mask do not depend on dS, so they
        # are built before waiting for it (the previous chunk no longer reads
        # these buffers).
        self._wait(edges, _E_LOAD, j)
        self._row_tiles(0, half, d, head, tok0, valid, smem, gc, s_beta, scale)
        if cutlass.const_expr(half == 0):
            self._mask_square(0, d, valid, smem, s_beta)
        self._stamp(trace, half, 11, j)
        g_last = lo[3]
        gamma_last = cute.math.exp2(g_last, fastmath=True)
        base_h = cutlass.Int32(smem.h.toint())
        base_ds = cutlass.Int32(smem.ds.toint())
        hv = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        dv8 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        ds_ready = self._ds_ready
        recv = j - 1 + self._cpart
        peer_slot = (
            cutlass.Int64(self._xfer.iterator.toint())
            + (cutlass.Int64(self._pair * 2 + 1 - self._cpart) * _DK * _DV + d) * 4
        )
        if cutlass.const_expr(self.tape):
            # dS arrived with the late loads; only the dg boundary term remains.
            dot = cutlass.Float32(0.0)
            for pp in cutlass.range_constexpr(2):
                quarter = half * 2 + pp
                for grp in cutlass.range_constexpr(4):
                    col = quarter * 32 + grp * 8
                    _load_shared_bf16x8(_swz_state(base_h, d, col), hv)
                    _load_shared_bf16x8(_swz_state(base_ds, d, col), dv8)
                    for e in cutlass.range_constexpr(8):
                        dot += hv[e].to(self.acc_dtype) * dv8[e].to(self.acc_dtype)
        else:
            if it > 0:
                if cutlass.const_expr(self.split == 1):
                    self._wait(edges, _E_DONE_C, j - 1)
                else:
                    # dS comes from the other CTA, which ran the previous chunk.
                    cute.arch.mbarrier_wait(ds_ready, recv & 1)
                    cute.arch.fence_acq_rel_cluster()
            dot = cutlass.Float32(0.0)
            for pp in cutlass.range_constexpr(2):
                quarter = half * 2 + pp
                address = lane_base + tmem_offset("dh") + quarter * 32
                if it == 0:
                    for c in cutlass.range_constexpr(32):
                        buf[c] = dht[sequence, head, d, quarter * 32 + c]
                else:
                    if cutlass.const_expr(self.split == 1):
                        _tmem_ld32(address, buf)
                    else:
                        for c in cutlass.range_constexpr(32):
                            buf[c] = _ld_f32(peer_slot + (quarter * 32 + c) * _DK * 4)
                for grp in cutlass.range_constexpr(4):
                    col = quarter * 32 + grp * 8
                    _load_shared_bf16x8(_swz_state(base_h, d, col), hv)
                    for e in cutlass.range_constexpr(8):
                        value = buf[grp * 8 + e]
                        dv8[e] = value.to(self.io_dtype)
                        dot += hv[e].to(self.acc_dtype) * value
                        buf[grp * 8 + e] = value * gamma_last
                    _store_shared_bf16x8(_swz_state(base_ds, d, col), dv8)
                _tmem_st32(address, buf)
        cute.arch.fence_view_async_tmem_store()
        self._stamp(trace, half, 10, j)
        return dot * gamma_last

    @cute.jit
    def _send_ds(self, half: cutlass.Constexpr, d, it, num_chunks, lane_base):
        """With two CTAs, hand the next chunk's dS (TMEM dh after phase C, FP32)
        to the other CTA through the global scratch rows of this CTA and its
        DS_READY barrier."""
        if cutlass.const_expr(self.split == 2 and not self.tape):
            # Locals only inside the dynamic branch (the DSL cannot carry self).
            ds_ready = self._ds_ready
            peer_rank = 1 - self._cpart
            # This CTA's FP32 dS slot, stored value-column-major so that a
            # warp's 32 consecutive channels form one coalesced 128-byte store.
            slot = (
                cutlass.Int64(self._xfer.iterator.toint())
                + (cutlass.Int64(self._pair * 2 + self._cpart) * _DK * _DV + d) * 4
            )
            buf = cute.make_rmem_tensor((32,), cutlass.Float32)
            if it + 1 < num_chunks:
                for pp in cutlass.range_constexpr(2):
                    quarter = half * 2 + pp
                    _tmem_ld32(lane_base + tmem_offset("dh") + quarter * 32, buf)
                    for c in cutlass.range_constexpr(32):
                        _stg_f32(slot + (quarter * 32 + c) * _DK * 4, buf[c])
                cute.arch.fence_acq_rel_cluster()
                cute.arch.barrier(
                    barrier_id=_BAR_CH if half == 0 else _BAR_TK, number_of_threads=128
                )
                if d == 0:
                    cute.arch.mbarrier_arrive(ds_ready, peer_cta_rank_in_cluster=peer_rank)

    @cute.jit
    def _load_gate_column(self, half: cutlass.Constexpr, gc, tok0, valid, head, d, cm):
        """Load this half's gate column of channel ``d`` (tokens clamped to the
        chunk tail) and the four blocks' first/last gates into ``cm``."""
        gbuf, _, _, _, up, lo, _ = cm
        g_row = self.heads * _DK * 4
        g_base = (
            cutlass.Int64(gc.iterator.toint())
            + (cutlass.Int64(tok0) * (self.heads * _DK) + head * _DK + d) * 4
        )
        for c in cutlass.range_constexpr(32):
            gbuf[c] = _ldg_f32(g_base + cutlass.min(half * 32 + c, valid - 1) * g_row)
        for b in cutlass.range_constexpr(4):
            up[b] = _ldg_f32(g_base + cutlass.min(b * _BC, valid - 1) * g_row)
            lo[b] = _ldg_f32(g_base + cutlass.min(b * _BC + _BC - 1, valid - 1) * g_row)

    @cute.jit
    def _cm_phase_b(self, half: cutlass.Constexpr, d, head, tok0, valid, smem, gc, s_beta):
        """kG -> -(k beta G) for dh_k once x_state has consumed kG."""
        self._row_tiles(1, half, d, head, tok0, valid, smem, gc, s_beta, cutlass.Float32(0.0))

    @cute.jit
    def _cm_phase_c(
        self,
        half: cutlass.Constexpr,
        gchunk,
        d,
        head,
        tok0,
        valid,
        lane_base,
        smem,
        gc,
        s_beta,
        debug_out,
        scale,
        l_token,
        cm,
    ):
        """Build the anchored hi/lo intra tiles for this half.  The state-path
        token gradients are scaled by the fold (``_fold_store``), which already
        holds the gate column."""
        self._row_tiles(2, half, d, head, tok0, valid, smem, gc, s_beta, scale)

    @cute.jit
    def _cm_fold(
        self,
        half: cutlass.Constexpr,
        j,
        d,
        head,
        tok0,
        valid,
        lane_base,
        smem,
        ring_full,
        ring_empty,
        gc,
        s_beta,
        s_dbeta,
        l_token,
        l_square,
        cm,
        scale,
        debug_out,
        gchunk,
    ):
        """Fold the 12 intra ring partials into dq, dk_plus (beta R) and
        dk_minus for the tokens of this half.

        A partial anchored at block b needs 2^(G_t - G_lo[b]) (lower) or
        2^(G_up[b] - G_t) (upper).  Both split at the token's own block tb into
        a per-slot scalar and a per-token factor that is at most 1:
        2^(G_t - G_up[tb]) 2^(G_up[tb] - G_lo[b]) and
        2^(G_lo[tb] - G_t) 2^(G_up[b] - G_lo[tb]).  The loop therefore keeps
        only the unscaled sum of each target live; the per-token factor, beta,
        the dbeta partials and the exact diagonal blocks are applied once per
        target in ``_cm_fold_target``, together with the gate scaling of the
        state-path gradient already in that target.  Returns this half's
        dk_state dg boundary term."""
        gbuf, buf, acc, pk, up, lo, _ = cm
        dg_term = cutlass.Float32(0.0)
        col = half * 32
        s_k = _mn_view(_view(smem.k, l_token))
        s_q = _mn_view(_view(smem.q, l_token))
        s_aqk = _mn_view(_view(smem.aqk, l_square))
        s_sq = _mn_view(_view(smem.sq, l_square))
        base_aqk = cutlass.Int32(smem.aqk.toint())
        base_sq = cutlass.Int32(smem.sq.toint())
        for r in cutlass.range_constexpr(12):
            slot = r % 2
            use = j * 6 + r // 2
            block = r % 4
            lower = r < 8
            touches = ((col + 31) // _BC >= block) if lower else (col // _BC <= block)
            if cutlass.const_expr(block == 0):
                for c in cutlass.range_constexpr(32):
                    acc[c] = cutlass.Float32(0.0)
            cute.arch.mbarrier_wait(ring_full + slot, use & 1)
            if cutlass.const_expr(touches):
                _tmem_ld32_async(lane_base + tmem_offset("ring") + slot * 64 + col, buf)
                # Per-slot scalars for the two token blocks of this half.
                f0 = cutlass.Float32(0.0)
                f1 = cutlass.Float32(0.0)
                for b2 in cutlass.range_constexpr(2):
                    tb = half * 2 + b2
                    if cutlass.const_expr((tb >= block) if lower else (tb <= block)):
                        if cutlass.const_expr(lower):
                            f = cute.math.exp2(up[tb] - lo[block], fastmath=True)
                        else:
                            f = cute.math.exp2(up[block] - lo[tb], fastmath=True)
                        if cutlass.const_expr(tb == block):
                            # Exact channels are recomputed in _cm_fold_target.
                            f = (
                                cutlass.Float32(0.0)
                                if up[block] - lo[block] >= DIAG_SPAN_LIMIT
                                else f
                            )
                        if cutlass.const_expr(b2 == 0):
                            f0 = f
                        else:
                            f1 = f
                _tmem_wait_ld()
                for c in cutlass.range_constexpr(32):
                    tb = (col + c) // _BC
                    if cutlass.const_expr((tb >= block) if lower else (tb <= block)):
                        acc[c] = acc[c] + (f0 if tb == half * 2 else f1) * buf[c]
                cute.arch.fence_view_async_tmem_load()
            # Each warp releases the slot once its own TMEM loads completed.
            cute.arch.sync_warp()
            if d % 32 == 0:
                cute.arch.mbarrier_arrive(ring_empty + slot)
            if cutlass.const_expr(block == 3):
                dg_term += self._cm_fold_target(
                    half,
                    r // 4,
                    d,
                    valid,
                    lane_base,
                    s_k,
                    s_q,
                    s_aqk,
                    s_sq,
                    s_beta,
                    s_dbeta,
                    gbuf,
                    buf,
                    acc,
                    pk,
                    up,
                    lo,
                    base_aqk,
                    base_sq,
                    scale,
                    debug_out,
                    gchunk,
                    head,
                )
        return dg_term

    @cute.jit
    def _cm_fold_target(
        self,
        half: cutlass.Constexpr,
        kind: cutlass.Constexpr,
        d,
        valid,
        lane_base,
        s_k,
        s_q,
        s_aqk,
        s_sq,
        s_beta,
        s_dbeta,
        gbuf,
        buf,
        acc,
        pk,
        up,
        lo,
        base_aqk,
        base_sq,
        scale,
        debug_out,
        gchunk,
        head,
    ):
        """Finish one fold target (kind 0 dq, 1 dk_plus, 2 dk_minus): apply the
        per-token factor to the intra sum, scale the state-path gradient held
        in TMEM, add them (with beta and the dbeta partials for dk_plus) and
        store.  The exact diagonal blocks of large-span channels follow as a
        separate TMEM update, so no accumulator stays live across that rare
        branch.  Returns the dk_state dg boundary term (kind 2)."""
        col = half * 32
        target = (
            tmem_offset("dq")
            if kind == 0
            else (tmem_offset("dk_plus") if kind == 1 else tmem_offset("dk_minus"))
        )
        _tmem_ld32(lane_base + tmem_offset("gate") + col, gbuf)
        for c in cutlass.range_constexpr(32):
            tb = (col + c) // _BC
            if cutlass.const_expr(kind < 2):
                factor = cute.math.exp2(gbuf[c] - up[tb], fastmath=True)
            else:
                factor = cute.math.exp2(lo[tb] - gbuf[c], fastmath=True)
            acc[c] = acc[c] * factor
        dg_term = self._fold_store(
            kind,
            col,
            True,
            d,
            valid,
            lane_base + target + col,
            s_k,
            s_beta,
            s_dbeta,
            pk,
            buf,
            acc,
            gbuf,
            lo[3],
            scale,
            debug_out,
            gchunk,
            head,
        )
        for b2 in cutlass.range_constexpr(2):
            block = half * 2 + b2
            exact = up[block] - lo[block] >= DIAG_SPAN_LIMIT
            # Rare path: exact diagonal block for channels with a large span.
            if cute.arch.vote_any_sync(exact):
                g = cute.make_rmem_tensor((32,), cutlass.Float32)
                pk = cute.make_rmem_tensor((32,), cutlass.Float32)
                total = cute.make_rmem_tensor((32,), cutlass.Float32)
                other = cute.make_rmem_tensor((32,), cutlass.Float32)
                for c in cutlass.range_constexpr(32):
                    total[c] = cutlass.Float32(0.0)
                _tmem_ld32(lane_base + tmem_offset("gate") + col, g)
                self._exact_block(
                    kind * 4 + block,
                    block,
                    exact,
                    d,
                    lane_base + tmem_offset("gate"),
                    valid,
                    s_k,
                    s_q,
                    s_aqk,
                    s_sq,
                    s_beta,
                    g,
                    total,
                    up[block],
                    lo[block],
                    base_aqk,
                    base_sq,
                )
                self._fold_store(
                    kind,
                    col,
                    False,
                    d,
                    valid,
                    lane_base + target + col,
                    s_k,
                    s_beta,
                    s_dbeta,
                    pk,
                    other,
                    total,
                    g,
                    lo[3],
                    scale,
                    debug_out,
                    gchunk,
                    head,
                )
        return dg_term

    @cute.jit
    def _fold_store(
        self,
        kind: cutlass.Constexpr,
        col: cutlass.Constexpr,
        state: cutlass.Constexpr,
        d,
        valid,
        address,
        s_k,
        s_beta,
        s_dbeta,
        pk,
        buf,
        acc,
        gbuf,
        g_last,
        scale,
        debug_out,
        gchunk,
        head,
    ):
        """TMEM[address] = S(TMEM[address]) + acc for this half's 32 token
        columns, with S the gate scaling of the state-path gradient when
        ``state`` (dq: scale 2^G; dk_plus: -2^G; dk_minus: 2^(G_last - G) on
        valid tokens) and the identity otherwise.  For dk_plus (kind 1) the sum
        is scaled by beta and the dbeta partials k * acc are reduced.  ``pk``
        and ``buf`` are scratch.  Returns the dk_minus dg boundary term."""
        dg_term = cutlass.Float32(0.0)
        k_col = _column_bases(cutlass.Int32(s_k.iterator.toint()), d)
        _tmem_ld32(address, buf)
        for c in cutlass.range_constexpr(32):
            t = col + c
            value = buf[c]
            if cutlass.const_expr(state):
                if cutlass.const_expr(kind == 0):
                    value = value * scale * cute.math.exp2(gbuf[c], fastmath=True)
                    if cutlass.const_expr(self.debug):
                        debug_out[gchunk, head, 7, t, d] = value
                if cutlass.const_expr(kind == 1):
                    value = -value * cute.math.exp2(gbuf[c], fastmath=True)
                    if cutlass.const_expr(self.debug):
                        debug_out[gchunk, head, 9, t, d] = value * s_beta[t]
                if cutlass.const_expr(kind == 2):
                    value = value * cute.math.exp2(g_last - gbuf[c], fastmath=True)
                    value = value if t < valid else cutlass.Float32(0.0)
                    if cutlass.const_expr(self.debug):
                        debug_out[gchunk, head, 8, t, d] = value
                    if t < valid:
                        dg_term += _lds_bf16(k_col[t % 8] + t * 128) * value
            if cutlass.const_expr(kind == 1):
                pk[c] = (
                    _lds_bf16(k_col[t % 8] + t * 128) * acc[c]
                    if t < valid
                    else cutlass.Float32(0.0)
                )
                if cutlass.const_expr(state):
                    value = (value + acc[c]) * s_beta[t]
                else:
                    value = value + acc[c] * s_beta[t]
            else:
                value = value + acc[c]
            buf[c] = value
        _tmem_st32(address, buf)
        cute.arch.fence_view_async_tmem_store()
        if cutlass.const_expr(kind == 1):
            self._reduce_dbeta(pk, s_dbeta, col, d)
        return dg_term

    @cute.jit
    def _cm_epilogue(
        self,
        half: cutlass.Constexpr,
        d,
        head,
        tok0,
        valid,
        lane_base,
        dg_part,
        s_xchg,
        s_suffix,
        smem,
        q,
        k,
        dq,
        dk,
        dg,
        l_token,
        cm,
        trace,
        it,
    ):
        """dq, dk and dg for this half, streamed 8 tokens at a time from TMEM.

        Both halves build their local reverse cumsum concurrently and park the
        local suffix sums in their consumed dq columns; after one exchange
        both write dg, the lower half adding the upper half's total."""
        col = half * 32
        k_col = _column_bases(cutlass.Int32(smem.k.toint()), d)
        q_col = _column_bases(cutlass.Int32(smem.q.toint()), d)
        # Per-thread global bases of (token tok0, head, channel d); rows are static offsets.
        row_bf16 = self.heads * _DK * 2
        row_f32 = self.heads * _DK * 4
        element = cutlass.Int64(tok0) * (self.heads * _DK) + head * _DK + d
        dq_base = cutlass.Int64(dq.iterator.toint()) + element * 2
        dk_base = cutlass.Int64(dk.iterator.toint()) + element * 2
        dg_base = cutlass.Int64(dg.iterator.toint()) + element * 4
        dkp = cute.make_rmem_tensor((8,), cutlass.Float32)
        dkm = cute.make_rmem_tensor((8,), cutlass.Float32)
        dqv = cute.make_rmem_tensor((8,), cutlass.Float32)
        s_xchg[half * _DK + d] = dg_part
        cute.arch.barrier(barrier_id=_BAR_CH_TK, number_of_threads=256)
        dg_last = s_xchg[d] + s_xchg[_DK + d]
        self._stamp(trace, half, 12, it)
        running = cutlass.Float32(0.0)
        for grp in cutlass.range_constexpr(4):
            g0 = col + (3 - grp) * 8
            _tmem_ldn_async(lane_base + tmem_offset("dk_plus") + g0, dkp, 8)
            _tmem_ldn_async(lane_base + tmem_offset("dk_minus") + g0, dkm, 8)
            _tmem_ldn_async(lane_base + tmem_offset("dq") + g0, dqv, 8)
            _tmem_wait_ld()
            for c in cutlass.range_constexpr(8):
                e = 7 - c
                t = g0 + e
                dg_value = cutlass.Float32(0.0)
                if t < valid:
                    k_value = _lds_bf16(k_col[t % 8] + t * 128)
                    q_value = _lds_bf16(q_col[t % 8] + t * 128)
                    dg_value = q_value * dqv[e] + k_value * (dkp[e] - dkm[e])
                    if t == valid - 1:
                        dg_value += dg_last
                    _stg_bf16(dq_base + t * row_bf16, dqv[e])
                    _stg_bf16(dk_base + t * row_bf16, dkp[e] + dkm[e])
                running += dg_value
                dqv[e] = running
            # dq of these tokens is stored; its columns now hold the suffix sums.
            _tmem_stn(lane_base + tmem_offset("dq") + g0, dqv, 8)
        if cutlass.const_expr(half == 1):
            s_suffix[d] = running
        cute.arch.fence_view_async_tmem_store()
        self._stamp(trace, half, 13, it)
        cute.arch.barrier(barrier_id=_BAR_CH_TK, number_of_threads=256)
        self._stamp(trace, half, 14, it)
        carry = s_suffix[d] if half == 0 else cutlass.Float32(0.0)
        for grp in cutlass.range_constexpr(4):
            g0 = col + grp * 8
            _tmem_ldn_async(lane_base + tmem_offset("dq") + g0, dqv, 8)
            _tmem_wait_ld()
            for c in cutlass.range_constexpr(8):
                t = g0 + c
                if t < valid:
                    _stg_f32(dg_base + t * row_f32, dqv[c] + carry)

    @cute.jit
    def _reduce_dbeta(self, pk, s_dbeta, first_token: cutlass.Constexpr, tidx):
        """Add sum over the 128 channels of pk[c] to dbeta[first_token + c].

        A butterfly transpose-reduce (31 shuffles): after it, lane l holds the
        warp sum for c = l, so each lane issues one shared atomic."""
        lane = tidx % 32
        for step in cutlass.range_constexpr(5):
            width = 16 >> step
            upper = (lane & width) != 0
            for i in cutlass.range_constexpr(width):
                send = pk[i] if upper else pk[i + width]
                keep = pk[i + width] if upper else pk[i]
                pk[i] = keep + cute.arch.shuffle_sync_bfly(send, offset=width)
        cute.arch.atomic_add(s_dbeta.iterator + (first_token + lane), pk[0])

    @cute.jit
    def _exact_block(
        self,
        r: cutlass.Constexpr,
        block: cutlass.Constexpr,
        exact,
        d,
        gate_address,
        valid,
        s_k,
        s_q,
        s_aqk,
        s_sq,
        s_beta,
        gbuf,
        acc,
        g_first,
        g_last,
        base_aqk,
        base_sq,
    ):
        """Exact 16x16 diagonal-block term (before beta) added to the register
        accumulator of this half (warp-uniform; lanes without a large span add
        zero).

        While both halves of the block span stay below DIAG_HALF_SPAN_LIMIT,
        the decay is split at the block's middle token in FP32: per-source
        factors come from the register gate column, square rows from 128-bit
        shared loads, and each unmasked term is one FMA.  Wider spans evaluate
        2^(G_late - G_early) per term in ``_exact_terms``."""
        offset = (block % 2) * _BC
        first = block * _BC
        g_mid = gbuf[offset + _BC // 2]
        split = (g_first - g_mid < DIAG_HALF_SPAN_LIMIT) and (g_mid - g_last < DIAG_HALF_SPAN_LIMIT)
        ex = cute.make_rmem_tensor((_BC,), cutlass.Float32)
        fa = cute.make_rmem_tensor((_BC,), cutlass.Float32)
        fb = cute.make_rmem_tensor((_BC,), cutlass.Float32)
        va = (
            cute.make_rmem_tensor((8,), cutlass.BFloat16),
            cute.make_rmem_tensor((8,), cutlass.BFloat16),
        )
        vb = (
            cute.make_rmem_tensor((8,), cutlass.BFloat16),
            cute.make_rmem_tensor((8,), cutlass.BFloat16),
        )
        for tt in cutlass.range_constexpr(_BC):
            ex[tt] = cutlass.Float32(0.0)
        # Per-source factors (zero past the tail so garbage rows cannot leak NaN).
        for jj in cutlass.range_constexpr(_BC):
            j = first + jj
            live = j < valid
            k_j = s_k[j, d].to(self.acc_dtype)
            if cutlass.const_expr(r < 8):
                kf = k_j * cute.math.exp2(g_mid - gbuf[offset + jj], fastmath=True)
                fa[jj] = kf if live else cutlass.Float32(0.0)
            else:
                u_j = cute.math.exp2(gbuf[offset + jj] - g_mid, fastmath=True)
                qf = s_q[j, d].to(self.acc_dtype) * u_j
                bf = s_beta[j] * k_j * u_j
                fa[jj] = qf if live else cutlass.Float32(0.0)
                fb[jj] = bf if live else cutlass.Float32(0.0)
        # Square rows by 128-bit loads; masked pairs are skipped at compile time.
        for row in cutlass.range_constexpr(_BC):
            if cutlass.const_expr(r < 8):
                base = base_aqk if r < 4 else base_sq
                for v8 in cutlass.range_constexpr(2):
                    _load_shared_bf16x8(_swz_token(base, first + row, first + v8 * 8), va[v8])
                for jj in cutlass.range_constexpr(_BC):
                    if cutlass.const_expr(jj < row or (jj == row and r < 4)):
                        ex[row] = ex[row] + va[jj // 8][jj % 8].to(self.acc_dtype) * fa[jj]
            else:
                for v8 in cutlass.range_constexpr(2):
                    _load_shared_bf16x8(_swz_token(base_aqk, first + row, first + v8 * 8), va[v8])
                    _load_shared_bf16x8(_swz_token(base_sq, first + row, first + v8 * 8), vb[v8])
                # Source row ``row`` feeds outputs tt <= row.
                for tt in cutlass.range_constexpr(row + 1):
                    ex[tt] = (
                        ex[tt]
                        + va[tt // 8][tt % 8].to(self.acc_dtype) * fa[row]
                        + vb[tt // 8][tt % 8].to(self.acc_dtype) * fb[row]
                    )
        for tt in cutlass.range_constexpr(_BC):
            g_t = gbuf[offset + tt]
            if cutlass.const_expr(r < 8):
                ex[tt] = ex[tt] * cute.math.exp2(g_t - g_mid, fastmath=True)
            else:
                ex[tt] = ex[tt] * cute.math.exp2(g_mid - g_t, fastmath=True)
        wide = exact and not split
        if cute.arch.vote_any_sync(wide):
            self._exact_terms(
                r,
                first,
                offset,
                wide,
                d,
                gate_address,
                valid,
                s_k,
                s_q,
                s_aqk,
                s_sq,
                s_beta,
                gbuf,
                ex,
            )
        for tt in cutlass.range_constexpr(_BC):
            t = first + tt
            acc[offset + tt] = acc[offset + tt] + (
                ex[tt] if exact and t < valid else cutlass.Float32(0.0)
            )

    @cute.jit
    def _exact_terms(
        self,
        r: cutlass.Constexpr,
        first,
        offset: cutlass.Constexpr,
        wide,
        d,
        gate_address,
        valid,
        s_k,
        s_q,
        s_aqk,
        s_sq,
        s_beta,
        gbuf,
        ex,
    ):
        """Per-term exact diagonal block for lanes whose half span is too wide
        to split (every factor is 2^(G_late - G_early) <= 1); replaces ``ex``
        on those lanes."""
        terms = cute.make_rmem_tensor((_BC,), cutlass.Float32)
        for tt in cutlass.range_constexpr(_BC):
            terms[tt] = cutlass.Float32(0.0)
        for j in cutlass.range(first, first + _BC):
            g_j = self._gate_at(gate_address, j)
            k_j = s_k[j, d].to(self.acc_dtype)
            if cutlass.const_expr(r >= 8):
                q_j = s_q[j, d].to(self.acc_dtype)
                bk_j = s_beta[j] * k_j
            for tt in cutlass.range_constexpr(_BC):
                t = first + tt
                g_t = gbuf[offset + tt]
                if cutlass.const_expr(r < 8):
                    square = s_aqk[t, j] if r < 4 else s_sq[t, j]
                    term = (
                        square.to(self.acc_dtype) * k_j * cute.math.exp2(g_t - g_j, fastmath=True)
                    )
                    pair = (j <= t) if r < 4 else (j < t)
                else:
                    term = (
                        s_aqk[j, t].to(self.acc_dtype) * q_j + s_sq[j, t].to(self.acc_dtype) * bk_j
                    ) * cute.math.exp2(g_j - g_t, fastmath=True)
                    pair = (j >= t) and (j < valid)
                terms[tt] = terms[tt] + (term if pair else cutlass.Float32(0.0))
        for tt in cutlass.range_constexpr(_BC):
            ex[tt] = terms[tt] if wide else ex[tt]

    @cute.jit
    def _gate_at(self, gate_address, token):
        """Gate value of this lane's channel at a chunk-local token (from TMEM)."""
        bits = _tmem_ld1(gate_address + token)
        view = cute.make_rmem_tensor((1,), cutlass.Int32)
        view[0] = bits
        return cute.make_tensor(cute.recast_ptr(view.iterator, dtype=cutlass.Float32), (1,))[0]


__all__ = ["DEBUG_SLOTS", "DIAG_SPAN_LIMIT", "FusedKdaBwdKernel"]
