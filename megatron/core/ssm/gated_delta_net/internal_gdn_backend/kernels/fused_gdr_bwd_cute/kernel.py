"""SM100 CuTe DSL device kernel for fused GDR backward.

This module is a CuTe DSL derivative of the fused GDR backward algorithm and
warp-specialized schedule from QwenLM/FlashQLA commit
050c6bbee9e03efbbfe41063fe4e33742c4a87cb, originally published as
``flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py``.

Copyright (c) 2026 The Qwen team, Alibaba Group. Licensed under the MIT License.
The package, module, test, and symbol names used here are independent GDR names
and do not imply endorsement by the upstream authors.

The inline-PTX helpers in ``tcgen05_ws.py`` are adapted from TileLang's CuTeDSL
backend. Copyright (c) Tile-AI. Licensed under the MIT License.
"""

from dataclasses import dataclass
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from . import tcgen05_ws
from .layouts import BackwardLayoutPlan, TmaDescriptorBundle, build_backward_layout_plan
from .storage import (
    TMEM_ALLOCATION_COLUMNS,
    TMEM_COLUMNS,
    TMEM_RANGES,
    LayoutBudget,
    SharedStorage,
    get_layout_budget,
    make_shared_views,
    make_tmem_views,
)

try:
    from cutlass.cute.experimental import iket as _cute_iket
except ImportError:
    _cute_iket = getattr(cute, "iket", None)
except NotImplementedError:
    _cute_iket = getattr(cute, "iket", None)


try:
    from cutlass.cute.nvgpu.cpasync import TmaInfo
except ImportError:
    from cutlass.base_dsl import extract_mlir_attributes as _ema
    from cutlass.base_dsl import extract_mlir_values as _emv
    from cutlass.base_dsl import get_mlir_types as _gmt
    from cutlass.base_dsl import new_from_mlir_values as _nfmv

    class TmaInfo:
        """MLIR-value shim for CuTeDSL releases returning an atom/tensor pair."""

        def __init__(self, atom, tma_tensor):
            self.atom = atom
            self.tma_tensor = tma_tensor

        def __extract_mlir_values__(self):
            return _emv(self.atom) + _emv(self.tma_tensor)

        def __extract_mlir_attributes__(self):
            return _ema(self.atom) + _ema(self.tma_tensor)

        def __new_from_mlir_values__(self, values):
            atom_values = len(_gmt(self.atom))
            return TmaInfo(
                _nfmv(self.atom, values[:atom_values]),
                _nfmv(self.tma_tensor, values[atom_values:]),
            )

        def __getitem__(self, index):
            return (self.atom, self.tma_tensor)[index]


def _wrap_tma(value):
    if isinstance(value, TmaInfo):
        return value
    return TmaInfo(value[0], value[1])


@dsl_user_op
def _phase_local_smem_ptr(ptr, *, loc=None, ip=None):
    """Re-materialize one shared pointer after a phase wait."""

    address = llvm.inline_asm(
        T.i32(),
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "mov.u32 $0, $1;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cute.make_ptr(
        ptr.dtype,
        address,
        ptr.memspace,
        assumed_align=ptr.alignment,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _store_shared_bf16x8(shared_address, src):
    src_view = cute.make_tensor(
        cute.recast_ptr(src.iterator, dtype=cutlass.Int32), (4,)
    )
    llvm.inline_asm(
        None,
        [cutlass.Int32(shared_address).ir_value()]
        + [src_view[index].ir_value() for index in range(4)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@cute.jit
def _store_shared_bf16x8_b32(shared_address, src):
    src_view = cute.make_tensor(
        cute.recast_ptr(src.iterator, dtype=cutlass.Int32), (4,)
    )
    llvm.inline_asm(
        None,
        [cutlass.Int32(shared_address).ir_value()]
        + [src_view[index].ir_value() for index in range(4)],
        '''{
        st.shared.b32 [$0], $1;
        st.shared.b32 [$0+4], $2;
        st.shared.b32 [$0+8], $3;
        st.shared.b32 [$0+12], $4;
        }''',
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _load_shared_bf16x8(shared_address, dst):
    i32_type = ir.IntegerType.get_signless(32)
    result_type = llvm.StructType.get_literal([i32_type] * 4)
    result = llvm.inline_asm(
        result_type,
        [cutlass.Int32(shared_address).ir_value()],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    dst_view = cute.make_tensor(
        cute.recast_ptr(dst.iterator, dtype=cutlass.Int32), (4,)
    )
    for register in cutlass.range_constexpr(4):
        dst_view[register] = cutlass.Int32(
            llvm.extractvalue(i32_type, result, [register])
        )


@cute.jit
def _direct_tmem_load32(tmem_ptr, dst):
    i32_type = ir.IntegerType.get_signless(32)
    result_type = llvm.StructType.get_literal([i32_type] * 32)
    address = cutlass.Int32(tmem_ptr.toint())
    dst_view = cute.make_tensor(
        cute.recast_ptr(dst.iterator, dtype=cutlass.Int32), (32,)
    )
    registers = ", ".join(f"${index}" for index in range(32))
    result = llvm.inline_asm(
        result_type,
        [address.ir_value()],
        f"tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        f"{{{registers}}}, [$32];",
        ",".join(["=r"] * 32 + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    for register in cutlass.range_constexpr(32):
        dst_view[register] = cutlass.Int32(
            llvm.extractvalue(i32_type, result, [register])
        )
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
def _direct_tmem_store32(tmem_ptr, src):
    address = tmem_ptr.toint()
    src_view = cute.make_tensor(
        cute.recast_ptr(src.iterator, dtype=cutlass.Int32), (32,)
    )
    registers = ", ".join(f"${index}" for index in range(1, 33))
    llvm.inline_asm(
        None,
        [address.ir_value()] + [
            src_view[index].ir_value() for index in range(32)
        ],
        f"tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {{{registers}}};",
        ",".join(["r"] * 33),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _load_store_dk_bf16x2(
    tmem_address,
    row0_address,
    row8_address,
):
    """Load four FP32 values and store two packed BF16 dK row pairs."""

    llvm.inline_asm(
        None,
        [
            cutlass.Int32(tmem_address).ir_value(),
            cutlass.Int64(row0_address).ir_value(),
            cutlass.Int64(row8_address).ir_value(),
        ],
        "{\n"
        ".reg .b32 f0, f1, f2, f3, b0, b1;\n"
        "tcgen05.ld.sync.aligned.16x256b.x1.b32 "
        "{f0, f1, f2, f3}, [$0];\n"
        "cvt.rn.bf16x2.f32 b0, f1, f0;\n"
        "cvt.rn.bf16x2.f32 b1, f3, f2;\n"
        "st.global.b32 [$1], b0;\n"
        "st.global.b32 [$2], b1;\n"
        "}",
        "r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _load_store_dk_bf16x2_tail(
    tmem_address,
    row0_address,
    row8_address,
    row0,
    row8,
    valid_tokens,
):
    """Load one dK fragment and predicate both packed row stores."""

    llvm.inline_asm(
        None,
        [
            cutlass.Int32(tmem_address).ir_value(),
            cutlass.Int64(row0_address).ir_value(),
            cutlass.Int64(row8_address).ir_value(),
            cutlass.Int32(row0).ir_value(),
            cutlass.Int32(row8).ir_value(),
            cutlass.Int32(valid_tokens).ir_value(),
        ],
        "{\n"
        ".reg .b32 f0, f1, f2, f3, b0, b1;\n"
        ".reg .pred p0, p1;\n"
        "tcgen05.ld.sync.aligned.16x256b.x1.b32 "
        "{f0, f1, f2, f3}, [$0];\n"
        "cvt.rn.bf16x2.f32 b0, f1, f0;\n"
        "cvt.rn.bf16x2.f32 b1, f3, f2;\n"
        "setp.lt.s32 p0, $3, $5;\n"
        "setp.lt.s32 p1, $4, $5;\n"
        "@p0 st.global.b32 [$1], b0;\n"
        "@p1 st.global.b32 [$2], b1;\n"
        "}",
        "r,l,l,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _direct_mma_ws(tmem_ptr, a_desc_tensor, b_desc_tensor, idesc, accumulate):
    a_desc = tcgen05.smem_descriptor_to_int(a_desc_tensor.iterator)
    b_desc = tcgen05.smem_descriptor_to_int(b_desc_tensor.iterator)
    llvm.inline_asm(
        None,
        [
            tmem_ptr.toint().ir_value(),
            a_desc.ir_value(),
            b_desc.ir_value(),
            cutlass.Int32(idesc).ir_value(),
            cutlass.Int32(accumulate).ir_value(),
        ],
        (
            "{\n"
            ".reg .pred p;\n"
            ".reg .pred q;\n"
            "elect.sync _|q, 0xFFFFFFFF;\n"
            "setp.ne.b32 p, $4, 0;\n"
            "@q tcgen05.mma.ws.cta_group::1.kind::f16 "
            "[$0], $1, $2, $3, p, 0;\n"
            "}"
        ),
        "r,l,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


_BT = 64
_TMA_VECTOR_HEADS = 4
_DK = 128
_DV = 128
_THREADS_PER_CTA = 384
_REUSE_V = 0
_REUSE_A = 1
_REUSE_DO = 2
_REUSE_BETA = 3

(
    _LOAD_QK,
    _LOAD_V,
    _LOAD_G,
    _LOAD_H,
    _LOAD_A,
    _LOAD_DO,
    _LOAD_BETA,
) = range(7)

(
    _MMA_P_DONE,
    _MMA_DVPRIME_STATE_DONE,
    _MMA_DVPRIME_PGAMMA_DONE,
    _MMA_U_STATE_DONE,
    _MMA_DV_DONE,
    _MMA_VPRIME_DONE,
    _MMA_DAG_DONE,
    _MMA_DPG_DONE,
    _MMA_DK_STATE_DONE,
    _MMA_DQ_STATE_DONE,
    _MMA_DK_STATE_G_DONE,
    _MMA_DQ_DP_DONE,
    _MMA_DH_K_LEFT_DONE,
    _MMA_DH_K_RIGHT_DONE,
    _MMA_DK_DP_DONE,
    _MMA_DA_LEFT_DONE,
    _MMA_DA_RIGHT_DONE,
    _MMA_AT_DONE,
    _MMA_DH_Q_LEFT_DONE,
    _MMA_DH_Q_RIGHT_DONE,
    _MMA_DK_DA_DONE,
) = range(21)

(
    _INPUT_P_READY,
    _INPUT_DVPRIME_STATE_READY,
    _INPUT_DVPRIME_PGAMMA_READY,
    _G_SMEM_REUSE_READY,
    _INPUT_DV_READY,
    _INPUT_VPRIME_DAG_READY,
    _INPUT_DPG_READY,
    _INPUT_DK_STATE_READY,
    _U_READERS_DONE_BEFORE_DQ_STATE_MMA,
    _INPUT_DK_STATE_G_READY,
    _INPUT_DQ_DP_READY,
    _INPUT_DH_K_READY,
    _INPUT_DA_LEFT_AT_READY,
    _INPUT_DA_RIGHT_READY,
    _INPUT_DH_Q_READY,
    _INPUT_DK_DA_READY,
) = range(16)


@dataclass(frozen=True)
class _CompletionOnlyUmmaPipeline(pipeline.PipelineUmmaAsync):
    """One-way MMA completion edge matching TileLang's tcbar protocol."""

    def producer_acquire(
        self, state, try_acquire_token=None, *, loc=None, ip=None
    ):
        _ = state, try_acquire_token, loc, ip

    def consumer_release(self, state, *, loc=None, ip=None):
        _ = state, loc, ip


@dataclass(frozen=True)
class _CompletionOnlyAsyncUmmaPipeline(pipeline.PipelineAsyncUmma):
    """One-way compute-to-MMA stage edge for overwrite-safety testing."""

    def producer_acquire(
        self, state, try_acquire_token=None, *, loc=None, ip=None
    ):
        _ = state, try_acquire_token, loc, ip

    def consumer_release(self, state, *, loc=None, ip=None):
        _ = state, loc, ip


@dataclass(frozen=True)
class _CompletionOnlyAsyncPipeline(pipeline.PipelineAsync):
    """One-way role edge for an exact physical-buffer reuse permission."""

    def producer_acquire(
        self, state, try_acquire_token=None, *, loc=None, ip=None
    ):
        _ = state, try_acquire_token, loc, ip

    def consumer_release(self, state, *, loc=None, ip=None):
        _ = state, loc, ip


@dataclass(frozen=True)
class _CompletionOnlyTmaUmmaPipeline(pipeline.PipelineTmaUmma):
    """TMA-to-MMA edge whose storage reuse is guarded explicitly."""

    def producer_acquire(
        self, state, try_acquire_token=None, *, loc=None, ip=None
    ):
        _ = try_acquire_token
        self.sync_object_full.arrive(
            state.index, self.producer_mask, loc=loc, ip=ip
        )

    def consumer_release(self, state, *, loc=None, ip=None):
        _ = state, loc, ip


@dataclass(frozen=True)
class _CompletionOnlyTmaAsyncPipeline(pipeline.PipelineTmaAsync):
    """TMA-to-thread edge whose storage reuse is guarded explicitly."""

    def producer_acquire(
        self, state, try_acquire_token=None, *, loc=None, ip=None
    ):
        _ = try_acquire_token
        self.sync_object_full.arrive(
            state.index, self.producer_mask, loc=loc, ip=ip
        )

    def consumer_release(self, state, *, loc=None, ip=None):
        _ = state, loc, ip


@dataclass(frozen=True)
class StaticTensorContract:
    """Trace-time rank plus static shape/inner-stride requirements."""

    name: str
    rank: int
    shape_modes: tuple[tuple[int, str], ...]
    stride_modes: tuple[tuple[int, str], ...]


STATIC_TENSOR_CONTRACTS = (
    StaticTensorContract(
        "q", 4, ((0, "one"), (2, "grouped_heads"), (3, "dk")),
        ((1, "grouped_token_stride"), (2, "dk"), (3, "one")),
    ),
    StaticTensorContract(
        "k", 4, ((0, "one"), (2, "grouped_heads"), (3, "dk")),
        ((1, "grouped_token_stride"), (2, "dk"), (3, "one")),
    ),
    StaticTensorContract(
        "v", 4, ((0, "one"), (2, "heads"), (3, "dv")),
        ((1, "value_token_stride"), (2, "dv"), (3, "one")),
    ),
    StaticTensorContract(
        "a", 4, ((0, "one"), (2, "heads"), (3, "bt")),
        ((1, "a_token_stride"), (2, "bt"), (3, "one")),
    ),
    StaticTensorContract(
        "g", 3, ((0, "one"), (2, "heads")),
        ((1, "heads"), (2, "one")),
    ),
    StaticTensorContract(
        "beta", 3, ((0, "one"), (2, "heads")),
        ((1, "heads"), (2, "one")),
    ),
    StaticTensorContract(
        "do", 4, ((0, "one"), (2, "heads"), (3, "dv")),
        ((1, "value_token_stride"), (2, "dv"), (3, "one")),
    ),
    StaticTensorContract(
        "dht",
        4,
        ((0, "num_sequences"), (1, "heads"), (2, "dk"), (3, "dv")),
        (
            (0, "state_batch_stride"),
            (1, "state_head_stride"),
            (2, "dv"),
            (3, "one"),
        ),
    ),
    StaticTensorContract(
        "h",
        5,
        ((0, "one"), (2, "heads"), (3, "dk"), (4, "dv")),
        ((2, "state_head_stride"), (3, "dv"), (4, "one")),
    ),
    StaticTensorContract(
        "cu_seqlens", 1, ((0, "metadata_size"),), ((0, "one"),)
    ),
    StaticTensorContract(
        "chunk_offsets", 1, ((0, "metadata_size"),), ((0, "one"),)
    ),
    StaticTensorContract(
        "dq", 4, ((0, "one"), (2, "grouped_heads"), (3, "dk")),
        ((1, "grouped_token_stride"), (2, "dk"), (3, "one")),
    ),
    StaticTensorContract(
        "dk", 4, ((0, "one"), (2, "grouped_heads"), (3, "dk")),
        ((1, "grouped_token_stride"), (2, "dk"), (3, "one")),
    ),
    StaticTensorContract(
        "dv", 4, ((0, "one"), (2, "heads"), (3, "dv")),
        ((1, "value_token_stride"), (2, "dv"), (3, "one")),
    ),
    StaticTensorContract(
        "dg", 3, ((0, "one"), (2, "heads")),
        ((1, "heads"), (2, "one")),
    ),
    StaticTensorContract(
        "db", 3, ((0, "one"), (2, "heads")),
        ((1, "heads"), (2, "one")),
    ),
    StaticTensorContract(
        "dh0",
        4,
        ((0, "num_sequences"), (1, "heads"), (2, "dk"), (3, "dv")),
        (
            (0, "state_batch_stride"),
            (1, "state_head_stride"),
            (2, "dv"),
            (3, "one"),
        ),
    ),
)


class FusedGdrBwdKernel:
    """SM100 role, memory, MMA, TMA, and launch structure."""

    threads_per_cta = _THREADS_PER_CTA

    sk_warp_ids = (0, 1, 2, 3)
    state_warp_ids = sk_warp_ids
    kv_warp_ids = sk_warp_ids
    aq_warp_ids = (4, 5, 6, 7)
    mma_warp_id = 8
    tma_warp_id = 9
    idle_producer_warp_id = 10
    dq_store_warp_id = 11

    consumer_SK_warpgroup_register_limit = 184
    consumer_A_warpgroup_register_limit = 160
    producer_warpgroup_register_limit = 160

    bt = _BT
    dk = _DK
    dv = _DV
    tmem_columns = TMEM_COLUMNS
    tmem_allocation_columns = TMEM_ALLOCATION_COLUMNS

    mm_64_64_128_tile = (64, 64, 128)
    mm_64_128_64_tile = (64, 128, 64)
    mm_128_64_64_tile = (128, 64, 64)
    mm_128_64_128_tile = (128, 64, 128)
    mm_64_64_64_tile = (64, 64, 64)

    p_tmem_offset = 0
    dk_tmem_offset = 0
    dv_tmem_offset = 64
    da_tmem_offset = 128
    u_tmem_offset = 192
    vprime_tmem_offset = 416
    dog_tmem_offset = 416
    dq_tmem_offset = 192
    a_tmem_offset = 160
    dp_tmem_offset = 160
    dh_left_tmem_offset = 256
    dh_right_tmem_offset = 320
    mask_tmem_offset = 384

    shared_storage = SharedStorage

    def __init__(
        self,
        io_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        *,
        heads: int = 64,
        grouped_heads: int = 64,
        num_sequences: int,
        use_dht: bool = True,
        state_v_first: bool = False,
        uniform_sequence_length: int = 0,
        enable_varlen_tail: bool = False,
        enable_iket: bool = False,
    ):
        if io_dtype is not cutlass.BFloat16:
            raise TypeError("FusedGdrBwdKernel requires BFloat16 inputs")
        if acc_dtype is not cutlass.Float32:
            raise TypeError("FusedGdrBwdKernel requires Float32 accumulation")
        if use_dht is not True:
            raise NotImplementedError("use_dht=False is not supported")
        if state_v_first is not False:
            raise NotImplementedError("state_v_first=True is not supported")
        if heads % _TMA_VECTOR_HEADS:
            raise ValueError(
                "heads must be divisible by four for g/beta TMA loads"
            )
        if grouped_heads != heads:
            raise ValueError(
                "grouped_heads must equal heads; GQA head mapping is unsupported"
            )
        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        if uniform_sequence_length < 0:
            raise ValueError("uniform_sequence_length must be non-negative")

        self.io_dtype = io_dtype
        self.acc_dtype = acc_dtype
        self.heads = heads
        self.grouped_heads = grouped_heads
        self.num_sequences = num_sequences
        self.use_dht = use_dht
        self.state_v_first = state_v_first
        self.uniform_sequence_length = uniform_sequence_length
        self.enable_varlen_tail = enable_varlen_tail
        self.enable_iket = enable_iket
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tmem_free_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.gate_materialized_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=256,
        )
        self.state_tmem_store_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=128,
        )
        self.kv_reduction_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=128,
        )
        self.aq_reduction_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=128,
        )
        self.tail_sk_barrier = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=128,
        )
        self.tail_aq_barrier = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=128,
        )
        self.tail_mma_barrier = pipeline.NamedBarrier(
            barrier_id=8,
            num_threads=32,
        )
        self.tail_dq_store_barrier = pipeline.NamedBarrier(
            barrier_id=9,
            num_threads=32,
        )

    def _build_mma_layouts(self) -> BackwardLayoutPlan:
        """Build named trace-time layout bindings and validate TMEM placement."""

        plan = build_backward_layout_plan(self.io_dtype, self.acc_dtype)
        self._validate_tmem_layout_plan(plan)
        return plan

    def _validate_tmem_layout_plan(self, plan: BackwardLayoutPlan) -> None:
        """Keep the physical TMEM reservations pinned to the baseline layout."""

        range_by_name = {
            name: (begin, end) for name, begin, end, _, _ in TMEM_RANGES
        }
        bindings = (
            ("p", self.p_tmem_offset, "mm_64_64_128_k_k"),
            ("dk", self.dk_tmem_offset, "mm_64_128_64_mn_mn"),
            ("dv", self.dv_tmem_offset, "mm_64_128_64_mn_mn"),
            ("da", self.da_tmem_offset, "mm_64_64_128_k_k"),
            ("u", self.u_tmem_offset, "mm_64_128_64_mn_mn"),
            ("vprime", self.vprime_tmem_offset, "mm_64_128_64_mn_mn"),
            ("dog", self.dog_tmem_offset, "mm_64_128_64_mn_mn"),
            ("dq", self.dq_tmem_offset, "mm_64_128_64_mn_mn"),
            ("a", self.a_tmem_offset, "mm_64_64_128_k_k"),
            ("dp", self.dp_tmem_offset, "mm_64_64_128_k_k"),
            ("at", self.a_tmem_offset, "mm_64_64_128_k_k"),
            ("dh_left", self.dh_left_tmem_offset, "mm_128_64_64_mn_mn"),
            ("dh_right", self.dh_right_tmem_offset, "mm_128_64_64_mn_mn"),
        )
        assert {binding[0] for binding in bindings} == set(range_by_name) - {
            "mask"
        }
        assert range_by_name["mask"] == (
            self.mask_tmem_offset,
            self.mask_tmem_offset + 32,
        )
        for name, offset, variant_name in bindings:
            begin, end = range_by_name[name]
            assert begin == offset
            logical_columns = plan.physical_columns_by_variant[variant_name]
            reservation_columns = (
                logical_columns // 2
                if logical_columns == 64
                and variant_name.startswith("mm_64_64_")
                else logical_columns
            )
            assert end - begin == reservation_columns, (
                f"{name}: TMEM range span {end - begin} does not match "
                f"{variant_name} physical reservation {reservation_columns}"
            )

    @cute.jit
    def _check_static_tensor_contracts(
        self,
        q,
        k,
        v,
        a,
        g,
        beta,
        do,
        dht,
        h,
        cu_seqlens,
        chunk_offsets,
        dq,
        dk,
        dv,
        dg,
        db,
        dh0,
    ):
        tensors = {
            "q": q,
            "k": k,
            "v": v,
            "a": a,
            "g": g,
            "beta": beta,
            "do": do,
            "dht": dht,
            "h": h,
            "cu_seqlens": cu_seqlens,
            "chunk_offsets": chunk_offsets,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "dg": dg,
            "db": db,
            "dh0": dh0,
        }
        expected = {
            "one": 1,
            "num_sequences": self.num_sequences,
            "metadata_size": self.num_sequences + 1,
            "heads": self.heads,
            "grouped_heads": self.grouped_heads,
            "bt": self.bt,
            "dk": self.dk,
            "dv": self.dv,
            "grouped_token_stride": self.grouped_heads * self.dk,
            "value_token_stride": self.heads * self.dv,
            "a_token_stride": self.heads * self.bt,
            "state_head_stride": self.dk * self.dv,
            "state_batch_stride": self.heads * self.dk * self.dv,
        }
        for spec in STATIC_TENSOR_CONTRACTS:
            tensor = tensors[spec.name]
            if cutlass.const_expr(len(tensor.shape) != spec.rank):
                raise ValueError(
                    f"{spec.name} must have rank {spec.rank}"
                )
            for mode, key in spec.shape_modes:
                if cutlass.const_expr(
                    not cute.is_static(tensor.shape[mode])
                ):
                    raise ValueError(
                        f"{spec.name}.shape[{mode}] must be static"
                    )
                if cutlass.const_expr(
                    tensor.shape[mode] != expected[key]
                ):
                    raise ValueError(
                        f"{spec.name}.shape[{mode}] has wrong extent"
                    )
            for mode, key in spec.stride_modes:
                if cutlass.const_expr(
                    not cute.is_static(tensor.stride[mode])
                ):
                    raise ValueError(
                        f"{spec.name}.stride[{mode}] must be static"
                    )
                if cutlass.const_expr(
                    tensor.stride[mode] != expected[key]
                ):
                    raise ValueError(
                        f"{spec.name}.stride[{mode}] has wrong value"
                    )

    @staticmethod
    def _token_matrix_view(tensor):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (
                    tensor.shape[1],
                    tensor.shape[-1],
                    (tensor.shape[0], tensor.shape[2]),
                ),
                stride=(
                    tensor.stride[1],
                    tensor.stride[-1],
                    (tensor.stride[0], tensor.stride[2]),
                ),
            ),
        )

    @staticmethod
    def _state_matrix_view(tensor):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (
                    tensor.shape[-2],
                    tensor.shape[-1],
                    tuple(tensor.shape[:-2]),
                ),
                stride=(
                    tensor.stride[-2],
                    tensor.stride[-1],
                    tuple(tensor.stride[:-2]),
                ),
            ),
        )

    @staticmethod
    def _vector_tma_view(tensor):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[2], tensor.shape[1], tensor.shape[0]),
                stride=(
                    tensor.stride[2], tensor.stride[1], tensor.stride[0],
                ),
            ),
        )

    def _build_tma_atoms(
        self,
        plan: BackwardLayoutPlan,
        q,
        k,
        v,
        a,
        g,
        beta,
        do,
        dht,
        h,
        dq,
    ):
        """Build descriptors from the same layouts consumed by each MMA."""

        load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        store_op = cpasync.CopyBulkTensorTileS2GOp()
        token = plan.variants.mm_64_64_128_k_k
        square = plan.variants.mm_64_64_64_k_k
        state = plan.variants.mm_128_64_128_k_k
        token_a = token.smem_a
        token_b = token.smem_b
        square_a = square.smem_a
        state_a = state.smem_a

        # Generic S2G TMA needs a flat top-level tile, unlike the nested
        # operand layouts accepted by the MMA-aware G2S helpers. These
        # repository-helper views preserve the same K-major physical swizzle.
        store_token_staged = sm100_utils.make_smem_layout(
            OperandMajorMode.K, (64, 128), self.io_dtype, 1
        )
        store_token = cute.select(store_token_staged, mode=[0, 1])
        assert store_token.inner == token_a.inner
        assert store_token.offset == token_a.offset

        q_view = self._token_matrix_view(q)
        k_view = self._token_matrix_view(k)
        v_view = self._token_matrix_view(v)
        a_view = self._token_matrix_view(a)
        g_view = self._vector_tma_view(g)
        beta_view = self._vector_tma_view(beta)
        do_view = self._token_matrix_view(do)
        h_view = self._state_matrix_view(h)
        dq_view = self._token_matrix_view(dq)
        vector_tma_layout = cute.make_layout(
            (_TMA_VECTOR_HEADS, _BT), stride=(1, _TMA_VECTOR_HEADS)
        )

        return TmaDescriptorBundle(
            q=_wrap_tma(cute.nvgpu.make_tiled_tma_atom_A(
                load_op,
                q_view,
                token_a,
                token.spec.tile,
                token.tiled_mma,
            )),
            k=_wrap_tma(cute.nvgpu.make_tiled_tma_atom_B(
                load_op,
                k_view,
                token_b,
                token.spec.tile,
                token.tiled_mma,
            )),
            v=_wrap_tma(cute.nvgpu.make_tiled_tma_atom_A(
                load_op,
                v_view,
                token_a,
                token.spec.tile,
                token.tiled_mma,
            )),
            g=_wrap_tma(cpasync.make_tiled_tma_atom(
                load_op,
                g_view,
                vector_tma_layout,
                (_TMA_VECTOR_HEADS, _BT),
            )),
            h=_wrap_tma(cute.nvgpu.make_tiled_tma_atom_A(
                load_op,
                h_view,
                state_a,
                state.spec.tile,
                state.tiled_mma,
            )),
            a=_wrap_tma(cute.nvgpu.make_tiled_tma_atom_A(
                load_op,
                a_view,
                square_a,
                square.spec.tile,
                square.tiled_mma,
            )),
            do=_wrap_tma(cute.nvgpu.make_tiled_tma_atom_A(
                load_op,
                do_view,
                token_a,
                token.spec.tile,
                token.tiled_mma,
            )),
            beta=_wrap_tma(cpasync.make_tiled_tma_atom(
                load_op,
                beta_view,
                vector_tma_layout,
                (_TMA_VECTOR_HEADS, _BT),
            )),
            dq_store=_wrap_tma(cpasync.make_tiled_tma_atom(
                store_op,
                dq_view,
                store_token,
                (64, 128),
            )),
        )

    @cute.jit
    def layout_probe(self, stream: cuda.CUstream):
        self._build_mma_layouts()
        self._layout_probe_kernel().launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _allocate_tmem(self, tmem_holding_ptr, warp_idx):
        if warp_idx == self.state_warp_ids[0]:
            for allocation_index in cutlass.range_constexpr(
                len(self.tmem_allocation_columns)
            ):
                tcgen05_ws.tmem_allocate(
                    tmem_holding_ptr + allocation_index,
                    self.tmem_allocation_columns[allocation_index],
                )
        cute.arch.sync_threads()
        return tuple(
            cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=(
                    tmem_holding_ptr + allocation_index
                ),
            )
            for allocation_index in range(len(self.tmem_allocation_columns))
        )

    @cute.jit
    def _release_tmem(self, tmem_holding_ptr, warp_idx):
        if warp_idx == self.state_warp_ids[0]:
            cute.arch.relinquish_tmem_alloc_permit()
        self.tmem_free_barrier.arrive_and_wait()
        if warp_idx == self.state_warp_ids[0]:
            for allocation_index in cutlass.range_constexpr(
                len(self.tmem_allocation_columns)
            ):
                tcgen05_ws.tmem_deallocate(
                    tmem_holding_ptr + allocation_index,
                    self.tmem_allocation_columns[allocation_index],
                )

    @cute.kernel
    def _layout_probe_kernel(self):
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tmem_holding_ptr = storage.tmem_holding_buf.data_ptr()
        self._allocate_tmem(tmem_holding_ptr, warp_idx)
        self._release_tmem(tmem_holding_ptr, warp_idx)

    @cute.jit
    def _make_role_pipelines(self, storage):
        """Create every one-stage edge before any warp specializes.

        The seven load edges own input transaction barriers, MMA-done edges carry
        tcgen05 completion to consumers, and input-ready or ownership edges carry
        exactly one semantic producer condition back to the issuing MMA warp.
        """

        def group(num_threads):
            return pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_threads
            )

        cg_tma = group(1)
        cg_mma = group(1)
        cg_sk = group(128)
        cg_aq = group(128)
        cg_sk_aq = group(256)
        # PipelineTmaAsync empty-barrier release is emitted only by lane 0 of
        # each participating warp, so its consumer group counts signalling
        # warps, not all data-consuming threads.
        cg_aq_tma = group(4)
        cg_sk_aq_tma = group(8)

        def tma_umma(ptr, tx_count):
            edge = pipeline.PipelineTmaUmma.create(
                num_stages=1,
                producer_group=cg_tma,
                consumer_group=cg_mma,
                tx_count=tx_count,
                barrier_storage=ptr.data_ptr(),
                defer_sync=True,
            )
            return _CompletionOnlyTmaUmmaPipeline(
                sync_object_full=edge.sync_object_full,
                sync_object_empty=edge.sync_object_empty,
                num_stages=edge.num_stages,
                producer_mask=edge.producer_mask,
                consumer_mask=edge.consumer_mask,
                is_leader_cta=edge.is_leader_cta,
                cta_group=edge.cta_group,
            )

        def tma_async(ptr, consumers, tx_count):
            edge = pipeline.PipelineTmaAsync.create(
                num_stages=1,
                producer_group=cg_tma,
                consumer_group=consumers,
                tx_count=tx_count,
                barrier_storage=ptr.data_ptr(),
                defer_sync=True,
            )
            common = dict(
                sync_object_full=edge.sync_object_full,
                sync_object_empty=edge.sync_object_empty,
                num_stages=edge.num_stages,
                producer_mask=edge.producer_mask,
                consumer_mask=edge.consumer_mask,
            )
            if cutlass.const_expr(hasattr(edge, "is_signalling_thread")):
                return _CompletionOnlyTmaAsyncPipeline(
                    **common,
                    is_signalling_thread=edge.is_signalling_thread,
                )
            return _CompletionOnlyTmaAsyncPipeline(
                **common,
                is_signaling_thread=edge.is_signaling_thread,
            )

        dq_smem_tile_ready_pipeline = pipeline.PipelineAsync.create(
            num_stages=1,
            # AQ's named barrier first joins all four writer warps; one
            # elected AQ thread then publishes the completed shared tile.
            # Avoid 128 independent mbarrier arrivals on this proxy handoff.
            producer_group=cg_tma,
            consumer_group=cg_tma,
            barrier_storage=storage.dq_smem_tile_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        aq_w_inputs_read_done_pipeline = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=cg_aq,
            consumer_group=cg_sk,
            barrier_storage=storage.aq_w_inputs_read_done_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        dog_smem_ready_pipeline = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=cg_sk,
            consumer_group=cg_aq,
            barrier_storage=storage.dog_smem_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        dg_rmw_done_pipeline = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=cg_sk,
            consumer_group=cg_aq,
            barrier_storage=storage.dg_rmw_done_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        load_tensor_pipelines = (
            tma_umma(storage.load_qk_mbar_ptr, 2 * _BT * _DK * 2),
            tma_async(storage.load_v_mbar_ptr, cg_aq_tma, _BT * _DV * 2),
            tma_async(
                storage.load_g_mbar_ptr,
                cg_sk_aq_tma,
                _TMA_VECTOR_HEADS * _BT * 4,
            ),
            tma_async(storage.load_h_mbar_ptr, cg_mma, _DK * _DV * 2),
            tma_async(storage.load_a_mbar_ptr, cg_aq_tma, _BT * _BT * 2),
            tma_async(storage.load_do_mbar_ptr, cg_mma, _BT * _DV * 2),
            tma_async(
                storage.load_beta_mbar_ptr,
                cg_aq_tma,
                _TMA_VECTOR_HEADS * _BT * 4,
            ),
        )

        def async_role(ptr, producers, consumers):
            edge = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=producers,
                consumer_group=consumers,
                barrier_storage=ptr.data_ptr(),
                defer_sync=True,
            )
            return _CompletionOnlyAsyncPipeline(
                sync_object_full=edge.sync_object_full,
                sync_object_empty=edge.sync_object_empty,
                num_stages=edge.num_stages,
                producer_mask=edge.producer_mask,
                consumer_mask=edge.consumer_mask,
            )

        # One-way last-reader -> TMA overwrite permissions. Tuple order is
        # V, A, dO, beta. Unlike MMA-input-ready edges, these edges
        # never carry compute-input readiness.
        smem_reuse_pipelines = (
            async_role(storage.reuse_v_tma_mbar_ptr, cg_sk, cg_tma),
            async_role(storage.reuse_a_tma_mbar_ptr, cg_aq, cg_tma),
            async_role(storage.reuse_do_tma_mbar_ptr, cg_aq, cg_tma),
            async_role(
                storage.reuse_beta_tma_mbar_ptr,
                cg_aq,
                cg_tma,
            ),
        )
        tmp21_reuse_pipeline = async_role(
            storage.reuse_tmp21_aq_mbar_ptr, cg_sk, cg_aq
        )

        def umma_async(ptr, consumers):
            edge = pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=cg_mma,
                consumer_group=consumers,
                barrier_storage=ptr.data_ptr(),
                defer_sync=True,
            )
            return _CompletionOnlyUmmaPipeline(
                sync_object_full=edge.sync_object_full,
                sync_object_empty=edge.sync_object_empty,
                num_stages=edge.num_stages,
                producer_mask=edge.producer_mask,
                consumer_mask=edge.consumer_mask,
                cta_group=edge.cta_group,
            )

        # Completion edges follow the semantic _MMA_* index contract below.
        # Operation metadata describes layout compatibility, not issue order.
        mma_done_pipelines = (
            umma_async(storage.mma_p_done_mbar_ptr, cg_aq),
            umma_async(storage.mma_dvprime_state_done_mbar_ptr, cg_sk),
            umma_async(storage.mma_dvprime_pgamma_done_mbar_ptr, cg_aq),
            umma_async(storage.mma_u_state_done_mbar_ptr, cg_sk_aq),
            umma_async(storage.mma_dv_done_mbar_ptr, cg_sk),
            umma_async(storage.mma_vprime_done_mbar_ptr, cg_aq),
            umma_async(storage.mma_dag_done_mbar_ptr, cg_aq),
            umma_async(storage.mma_dpg_done_mbar_ptr, cg_aq),
            umma_async(storage.mma_dk_state_done_mbar_ptr, cg_sk),
            # Stage 08: dQ = dO S_i^T has a dedicated TMEM range, so only AQ
            # consumes its result. U readers use a separate ordering edge.
            umma_async(storage.mma_dq_state_done_mbar_ptr, cg_aq),
            # Stage 09 completion is consumed directly by W8 for the next
            # dK accumulation; TMA also observes it before overwriting H.
            umma_async(storage.mma_dk_state_g_done_mbar_ptr, cg_mma),
            umma_async(storage.mma_dq_dp_done_mbar_ptr, cg_aq),
            umma_async(storage.mma_dh_k_left_done_mbar_ptr, cg_sk),
            umma_async(storage.mma_dh_k_right_done_mbar_ptr, cg_sk),
            umma_async(storage.mma_dk_dp_done_mbar_ptr, cg_sk),
            umma_async(storage.mma_da_left_done_mbar_ptr, cg_aq),
            umma_async(storage.mma_da_right_done_mbar_ptr, cg_aq),
            umma_async(storage.mma_at_done_mbar_ptr, cg_aq),
            umma_async(storage.mma_dh_q_left_done_mbar_ptr, cg_sk),
            umma_async(storage.mma_dh_q_right_done_mbar_ptr, cg_sk),
            umma_async(storage.mma_dk_da_done_mbar_ptr, cg_sk),
        )

        def async_umma(ptr, producers):
            edge = pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=producers,
                consumer_group=cg_mma,
                barrier_storage=ptr.data_ptr(),
                defer_sync=True,
            )
            return _CompletionOnlyAsyncUmmaPipeline(
                sync_object_full=edge.sync_object_full,
                sync_object_empty=edge.sync_object_empty,
                num_stages=edge.num_stages,
                producer_mask=edge.producer_mask,
                consumer_mask=edge.consumer_mask,
                cta_group=edge.cta_group,
            )

        mma_input_ready_pipelines = (
            # Stage 00: Q and K are ready for P = Q K^T.
            async_umma(storage.p_inputs_ready_mbar_ptr, cg_sk_aq),
            # Stage 01: raw dS_next is ready for dVprime = K dS_next.
            async_umma(storage.dvprime_state_inputs_ready_mbar_ptr, cg_sk),
            # Stage 02: P-gamma and dO are ready to update dVprime.
            async_umma(storage.dvprime_pgamma_inputs_ready_mbar_ptr, cg_sk_aq),
            # G shared storage is no longer read; this is not Stage 03 input.
            async_umma(storage.g_smem_reuse_mbar_ptr, cg_sk_aq),
            # Stage 04: A-gamma and materialized dVprime are ready for dV.
            async_umma(storage.dv_inputs_ready_mbar_ptr, cg_aq),
            # Stage 05: A-gamma, W, and dVprime are ready for Vprime and dA-gamma.
            async_umma(storage.vprime_dag_inputs_ready_mbar_ptr, cg_aq),
            # Stage 06: dO and Vprime are ready for dP-gamma.
            async_umma(storage.dpg_inputs_ready_mbar_ptr, cg_aq),
            # Stage 07: Vprime and raw dS_next are ready for dK.
            async_umma(storage.dk_state_inputs_ready_mbar_ptr, cg_sk_aq),
            # Stage 08 ordering: every U reader is done before dQ overwrites TMEM.
            async_umma(storage.u_readers_done_before_dq_state_mma_mbar_ptr, cg_sk_aq),
            # Stage 09: dV-gamma and S_i are ready to update dK.
            async_umma(storage.dk_state_g_inputs_ready_mbar_ptr, cg_sk),
            # Stage 10: dP and K are ready to update dQ.
            async_umma(storage.dq_dp_inputs_ready_mbar_ptr, cg_aq),
            # Stage 11: K and dV-gamma are ready to update dH.
            async_umma(storage.dh_k_inputs_ready_mbar_ptr, cg_sk),
            # Stage 13: dA-left and A^T inputs are ready.
            async_umma(storage.da_left_at_inputs_ready_mbar_ptr, cg_aq),
            # Stage 13a: dA-right inputs are ready.
            async_umma(storage.da_right_inputs_ready_mbar_ptr, cg_aq),
            # Stage 14: Q and dO-gamma are ready to update dH.
            async_umma(storage.dh_q_inputs_ready_mbar_ptr, cg_sk_aq),
            # Stage 15: dA_s and K are ready for the final dK update.
            async_umma(storage.dk_da_inputs_ready_mbar_ptr, cg_aq),
        )
        all_tmem_readers_done_pipeline = async_umma(
            storage.all_tmem_readers_done_mbar_ptr, cg_sk_aq
        )
        # SK publishes the completed sDg contribution to AQ. AQ owns all
        # subsequent sDg read-modify-write operations through the output store.
        sdg_ownership_order_pipeline = async_role(
            storage.sdg_ownership_order_mbar_ptr, cg_sk, cg_aq
        )
        return (
            load_tensor_pipelines,
            mma_done_pipelines,
            mma_input_ready_pipelines,
            sdg_ownership_order_pipeline,
            smem_reuse_pipelines,
            tmp21_reuse_pipeline,
            all_tmem_readers_done_pipeline,
            dq_smem_tile_ready_pipeline,
            aq_w_inputs_read_done_pipeline,
            dog_smem_ready_pipeline,
            dg_rmw_done_pipeline,
        )

    @cute.jit
    def _producer_at(self, edge, iteration):
        phase = cutlass.Int32(1) ^ (iteration & 1)
        state = pipeline.PipelineState(
            1,
            cutlass.Int32(iteration),
            cutlass.Int32(0),
            phase,
        )
        return pipeline.PipelineProducer(
            edge, state, edge.sync_object_full.cg
        )

    @cute.jit
    def _consumer_at(self, edge, iteration):
        state = pipeline.PipelineState(
            1,
            cutlass.Int32(iteration),
            cutlass.Int32(0),
            cutlass.Int32(iteration & 1),
        )
        return pipeline.PipelineConsumer(
            edge, state, edge.sync_object_empty.cg
        )

    @cute.jit
    def _wait_full_at(self, edge, iteration):
        """Wait for one full phase without releasing its empty barrier."""

        self._consumer_at(edge, iteration).wait()

    @cute.jit
    def _iket_push_stage(
        self,
        block_idx,
        chunk_iteration,
        event_name: cutlass.Constexpr,
    ):
        """Start one detailed iKET range for CTA0 iterations I0/I1."""

        if cutlass.const_expr(self.enable_iket):
            if block_idx == 0:
                if chunk_iteration < 2:
                    _cute_iket.range_push(event_name)

    @cute.jit
    def _iket_pop_stage(self, block_idx, chunk_iteration):
        """End a detailed iKET range started by :meth:`_iket_push_stage`."""

        if cutlass.const_expr(self.enable_iket):
            if block_idx == 0:
                if chunk_iteration < 2:
                    _cute_iket.range_pop()

    @staticmethod
    def _transform_tmem_layout(tensor):
        """Group MMA atom and tiled M/N modes for tcgen05 copy atoms."""

        layout = tensor.layout
        stored_layout = layout
        if isinstance(stored_layout, cute.ComposedLayout):
            layout = layout.outer
        shape = layout.shape
        stride = layout.stride
        transformed = cute.make_layout(
            shape=(
                (shape[0][0], shape[1]),
                (shape[0][1], shape[2]),
                *shape[3:],
            ),
            stride=(
                (stride[0][0], stride[1]),
                (stride[0][1], stride[2]),
                *stride[3:],
            ),
        )
        if isinstance(stored_layout, cute.ComposedLayout):
            transformed = cute.make_composed_layout(
                stored_layout.inner, stored_layout.offset, transformed
            )
        return cute.make_tensor(tensor.iterator, transformed)

    @cute.jit
    def _issue_matrix_tma(
        self,
        tma_info,
        smem_tensor,
        tiled_mma,
        producer,
        tile_shape,
        token_offset,
        grouped_coord,
        is_b_operand: bool,
    ):
        """Issue one exact, unmasked matrix TMA for the current chunk."""

        handle = producer.acquire()
        self._copy_matrix_tma(
            tma_info,
            smem_tensor,
            tiled_mma,
            handle,
            tile_shape,
            token_offset,
            grouped_coord,
            is_b_operand,
        )
        return handle

    @cute.jit
    def _copy_matrix_tma(
        self,
        tma_info,
        smem_tensor,
        tiled_mma,
        handle,
        tile_shape,
        token_offset,
        grouped_coord,
        is_b_operand: bool,
    ):
        """Attach a matrix copy to an already acquired TMA transaction."""

        global_matrix = cute.domain_offset(
            (token_offset, cutlass.Int32(0)),
            tma_info[1][None, None, grouped_coord],
        )
        global_tile = cute.flat_divide(global_matrix, tile_shape)
        thread_mma = tiled_mma.get_slice(0)
        if cutlass.const_expr(is_b_operand):
            partitioned_global = thread_mma.partition_B(global_tile)
        else:
            partitioned_global = thread_mma.partition_A(global_tile)
        partitioned_smem, partitioned_global = cpasync.tma_partition(
            tma_info[0],
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tensor, 0, 3),
            cute.group_modes(partitioned_global, 0, 3),
        )
        cute.copy(
            tma_info[0],
            partitioned_global[(None, 0, 0)],
            partitioned_smem[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )

    @cute.jit
    def _issue_vector_tma(
        self,
        tma_info,
        smem_tensor,
        producer,
        token_offset,
        head,
    ):
        """Issue one aligned four-head by 64-token FP32 TMA transaction."""

        head_base = head - head % _TMA_VECTOR_HEADS
        global_matrix = cute.domain_offset(
            (head_base, token_offset),
            tma_info[1][None, None, cutlass.Int32(0)],
        )
        global_tile = cute.flat_divide(
            global_matrix, (_TMA_VECTOR_HEADS, _BT)
        )
        partitioned_smem, partitioned_global = cpasync.tma_partition(
            tma_info[0],
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tensor, 0, 2),
            cute.group_modes(global_tile, 0, 2),
        )
        handle = producer.acquire()
        cute.copy(
            tma_info[0],
            partitioned_global[(None, 0, 0)],
            partitioned_smem[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )

    @cute.jit
    def _store_token_tile_tma(
        self,
        tma_info,
        smem_tensor,
        token_offset,
        grouped_coord,
    ):
        """Store one exact 64x128 BF16 shared tile with bulk TMA."""

        global_matrix = cute.domain_offset(
            (token_offset, cutlass.Int32(0)),
            tma_info[1][None, None, grouped_coord],
        )
        global_tile = cute.flat_divide(global_matrix, (_BT, _DV))
        partitioned_smem, partitioned_global = cpasync.tma_partition(
            tma_info[0],
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tensor, 0, 3),
            cute.group_modes(global_tile, 0, 2),
        )
        cute.copy(
            tma_info[0],
            partitioned_smem[(None, 0)],
            partitioned_global[(None, 0, 0)],
        )
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def _zero_token_matrix_tail(
        self,
        smem_tensor,
        stage,
        valid_tokens,
        group_tidx,
        group_threads: cutlass.Constexpr,
    ):
        """Neutralize token rows that belong to the next packed sequence."""

        matrix = self._transform_tmem_layout(smem_tensor)
        invalid_values = (_BT - valid_tokens) * _DV
        for linear in cutlass.range(
            group_tidx, invalid_values, group_threads
        ):
            row = valid_tokens + linear // _DV
            column = linear % _DV
            matrix[row, column, stage] = self.io_dtype(0.0)

    @cute.jit
    def _zero_square_tail(
        self,
        smem_tensor,
        stage,
        valid_tokens,
        group_tidx,
        group_threads: cutlass.Constexpr,
    ):
        """Clear invalid rows and columns in one 64x64 token-local tile."""

        matrix = self._transform_tmem_layout(smem_tensor)
        for linear in cutlass.range(
            group_tidx, _BT * _BT, group_threads
        ):
            row = linear // _BT
            column = linear % _BT
            if row >= valid_tokens:
                matrix[row, column, stage] = self.io_dtype(0.0)
            elif column >= valid_tokens:
                matrix[row, column, stage] = self.io_dtype(0.0)

    @cute.jit
    def _extend_gate_tail(
        self,
        s_g,
        stage,
        vector_head,
        valid_tokens,
        group_tidx,
    ):
        """Extend the last valid cumulative gate through invalid rows."""

        if group_tidx < _BT:
            if group_tidx >= valid_tokens:
                s_g[vector_head, group_tidx, stage] = s_g[
                    vector_head, valid_tokens - 1, stage
                ]

    @cute.jit
    def _seed_state_half(
        self,
        accumulator,
        dht_matrix,
        s_h,
        s_tmp41,
        sequence,
        head,
        state_tidx,
        g_last,
        seed_from_dht,
        loaded_h_stage,
        half_idx: cutlass.Constexpr,
        snapshot_only: cutlass.Constexpr,
    ):
        """Snapshot raw state or scale one state half for this chunk."""

        c_dh = cute.make_identity_tensor((_DK, _DV // 2))
        dh_mn_view = self._transform_tmem_layout(accumulator)
        dh_for_copy = accumulator[(None, None), 0, 0, 0]
        atom_dh_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
            self.acc_dtype,
        )
        tiled_dh_r2t = tcgen05.make_tmem_copy(atom_dh_r2t, dh_for_copy)
        thr_dh_r2t = tiled_dh_r2t.get_slice(state_tidx)
        t_rt_dh = thr_dh_r2t.partition_D(dh_mn_view)
        t_rt_cdh = thr_dh_r2t.partition_S(c_dh)
        gdht = cute.local_tile(
            dht_matrix,
            (_DK, _DV // 2),
            (None, None, None),
        )
        gdht_half = gdht[
            None, None, 0, half_idx, (sequence, head)
        ]
        t_rt_gdht = thr_dh_r2t.partition_S(gdht_half)
        r_dh = cute.make_rmem_tensor_like(t_rt_cdh, self.acc_dtype)
        if seed_from_dht:
            for sub in cutlass.range(r_dh.shape[2], unroll_full=True):
                cute.autovec_copy(
                    t_rt_gdht[None, 0, sub],
                    r_dh[None, 0, sub],
                )
                cute.copy(
                    tiled_dh_r2t,
                    r_dh[None, 0, sub],
                    t_rt_dh[None, 0, sub, 0],
                )
            cute.arch.fence_view_async_tmem_store()
            self.state_tmem_store_barrier.arrive_and_wait()

        atom_dh_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.acc_dtype,
        )
        tiled_dh_t2r = tcgen05.make_tmem_copy(atom_dh_t2r, dh_for_copy)
        thr_dh_t2r = tiled_dh_t2r.get_slice(state_tidx)
        t_tr_dh = thr_dh_t2r.partition_S(dh_mn_view)
        t_tr_cdh = thr_dh_t2r.partition_D(c_dh)
        s_h_mn_view = self._transform_tmem_layout(s_h)
        s_tmp41_mn_view = self._transform_tmem_layout(s_tmp41)
        r_dh_out = cute.make_rmem_tensor_like(r_dh, self.io_dtype)
        atom_dh_r2s = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
            num_bits_per_copy=128,
        )
        tiled_dh_r2s = cute.make_tiled_copy_D(
            atom_dh_r2s, tiled_dh_t2r
        )
        thr_dh_r2s = tiled_dh_r2s.get_slice(state_tidx)
        t_cs_dh = thr_dh_r2s.partition_D(s_tmp41_mn_view)
        t_cs_h = thr_dh_r2s.partition_S(s_h_mn_view)
        t_cr_dh = tiled_dh_r2s.retile(r_dh_out)
        state_dot0 = cutlass.Float32(0.0)
        if cutlass.const_expr(snapshot_only):
            for sub in cutlass.range(r_dh.shape[2], unroll_full=True):
                cute.copy(
                    tiled_dh_t2r,
                    t_tr_dh[None, 0, sub, 0],
                    r_dh[None, 0, sub],
                )
                r_dh_out[None, 0, sub].store(
                    r_dh[None, 0, sub].load().to(self.io_dtype)
                )
                cute.copy(
                    tiled_dh_r2s,
                    t_cr_dh[None, 0, sub],
                    t_cs_dh[
                        None, 0,
                        sub + half_idx * r_dh.shape[2], 0,
                    ],
                )
            cute.arch.fence_view_async_shared()
            return cutlass.Float32(0.0)

        state_dot1 = cutlass.Float32(0.0)
        for sub in cutlass.range(r_dh.shape[2], unroll_full=True):
            cute.copy(
                tiled_dh_t2r,
                t_tr_dh[None, 0, sub, 0],
                r_dh[None, 0, sub],
            )
            r_dh_out[None, 0, sub].store(
                r_dh[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_dh_r2s,
                t_cs_h[
                    None,
                    0,
                    sub + half_idx * r_dh.shape[2],
                    loaded_h_stage,
                ],
                t_cr_dh[None, 0, sub],
            )
            for element in cutlass.range(
                0,
                cute.size(r_dh[None, 0, sub]),
                2,
                unroll_full=True,
            ):
                (
                    r_dh[element, 0, sub],
                    r_dh[element + 1, 0, sub],
                ) = cute.arch.mul_packed_f32x2(
                    (
                        r_dh[element, 0, sub],
                        r_dh[element + 1, 0, sub],
                    ),
                    (g_last, g_last),
                    rnd="rn",
                    ftz=False,
                )
                state_dot0, state_dot1 = cute.arch.fma_packed_f32x2(
                    (
                        r_dh[element, 0, sub],
                        r_dh[element + 1, 0, sub],
                    ),
                    (
                        r_dh_out[element, 0, sub].to(self.acc_dtype),
                        r_dh_out[element + 1, 0, sub].to(self.acc_dtype),
                    ),
                    (state_dot0, state_dot1),
                    rnd="rn",
                    ftz=False,
                )
            cute.copy(
                tiled_dh_r2t,
                r_dh[None, 0, sub],
                t_rt_dh[None, 0, sub, 0],
            )
        cute.arch.fence_view_async_tmem_store()
        self.state_tmem_store_barrier.arrive_and_wait()
        return state_dot0 + state_dot1

    @cute.jit
    def _store_state_half(
        self,
        accumulator,
        dh0_matrix,
        sequence,
        head,
        state_tidx,
        half_idx: cutlass.Constexpr,
    ):
        """Store one independently addressed 128x64 TMEM state half."""

        c_dh = cute.make_identity_tensor((_DK, _DV // 2))
        dh_mn_view = self._transform_tmem_layout(accumulator)
        dh_for_copy = accumulator[(None, None), 0, 0, 0]
        atom_dh_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.acc_dtype,
        )
        tiled_dh_t2r = tcgen05.make_tmem_copy(atom_dh_t2r, dh_for_copy)
        thr_dh_t2r = tiled_dh_t2r.get_slice(state_tidx)
        t_tr_dh = thr_dh_t2r.partition_S(dh_mn_view)
        t_tr_cdh = thr_dh_t2r.partition_D(c_dh)
        gdh0 = cute.local_tile(
            dh0_matrix,
            (_DK, _DV // 2),
            (None, None, None),
        )
        gdh0_half = gdh0[
            None, None, 0, half_idx, (sequence, head)
        ]
        t_tr_gdh0 = thr_dh_t2r.partition_D(gdh0_half)
        r_dh = cute.make_rmem_tensor_like(t_tr_cdh, self.acc_dtype)
        for sub in cutlass.range(r_dh.shape[2], unroll_full=True):
            cute.copy(
                tiled_dh_t2r,
                t_tr_dh[None, 0, sub, 0],
                r_dh[None, 0, sub],
            )
            cute.autovec_copy(
                r_dh[None, 0, sub],
                t_tr_gdh0[None, 0, sub],
            )

    @cute.jit
    def _commit_row_pair(
        self,
        value0,
        value1,
        output,
        row0,
        row1,
        group_tidx,
        sync_barrier,
        accumulate: cutlass.Constexpr,
    ):
        """Reduce two logical rows across each proven four-lane owner quad."""

        value0 = cute.arch.warp_reduction_sum(
            value0, threads_in_group=4
        )
        value1 = cute.arch.warp_reduction_sum(
            value1, threads_in_group=4
        )
        if group_tidx % 4 == 0:
            if cutlass.const_expr(accumulate):
                output[row0] += value0
                output[row1] += value1
            else:
                output[row0] = value0
                output[row1] = value1
        sync_barrier.arrive_and_wait()

    @cute.jit
    def _store_token_tile_vector128(
        self,
        output,
        smem_tensor,
        s_dv_offset,
        group_tidx,
        token_offset,
        valid_tokens,
    ):
        """Store one 64x128 BF16 tile with coalesced 128-bit vectors."""

        smem_mn_view = self._transform_tmem_layout(smem_tensor)
        smem_base = smem_mn_view.iterator.toint()
        row_base = group_tidx // 16
        column = (group_tidx % 16) * 8
        linear_address = smem_base + (
            row_base * 64 + (column & 63) + (column // 64) * 4096
        ) * 2
        shared_base = linear_address ^ ((linear_address >> 3) & 112)
        global_base = (
            output.iterator.toint()
            + cutlass.Int64(token_offset * output.stride[1] * 2)
            + cutlass.Int64(s_dv_offset[group_tidx])
        )
        for iteration in cutlass.range_constexpr(8):
            row = row_base + iteration * 8
            if cutlass.const_expr(self.enable_varlen_tail):
                if row < valid_tokens:
                    tcgen05_ws.copy_bf16x8_s2g_offset(
                        shared_base,
                        global_base,
                        iteration * 1024,
                        iteration * output.stride[1] * 8 * 2,
                    )
            else:
                tcgen05_ws.copy_bf16x8_s2g_offset(
                    shared_base,
                    global_base,
                    iteration * 1024,
                    iteration * output.stride[1] * 8 * 2,
                )

    @cute.jit
    def _store_token_tile_tail(
        self,
        output,
        smem_tensor,
        group_tidx,
        group_threads: cutlass.Constexpr,
        token_offset,
        head,
        valid_tokens,
    ):
        """Store valid 16-byte vectors from a partial 64x128 tile."""

        smem_mn_view = self._transform_tmem_layout(smem_tensor)
        smem_base = smem_mn_view.iterator.toint()
        vectors_per_row = _DV // 8
        for linear in cutlass.range(
            group_tidx, valid_tokens * vectors_per_row, group_threads
        ):
            row = linear // vectors_per_row
            column = (linear % vectors_per_row) * 8
            linear_address = smem_base + (
                row * 64 + (column & 63) + (column // 64) * 4096
            ) * 2
            shared_address = linear_address ^ (
                (linear_address >> 3) & 112
            )
            global_address = (
                output.iterator.toint()
                + cutlass.Int64((token_offset + row) * output.stride[1] * 2)
                + cutlass.Int64(head * output.stride[2] * 2)
                + cutlass.Int64(column * output.stride[3] * 2)
            )
            tcgen05_ws.copy_bf16x8_s2g_offset(
                shared_address, global_address, 0, 0
            )

    @cute.jit
    def run_tma_warp(
        self,
        block_idx,
        chunk_base,
        head,
        load_tensor_pipelines,
        mma_done_pipelines,
        mma_state_direct,
        mma_square_direct,
        mma_token_direct,
        smem_reuse_pipelines,
        s_a,
        s_beta,
        s_do,
        s_g,
        s_h,
        s_k,
        s_q,
        s_v,
        sequence_chunks,
        sequence_start,
        mma_input_ready_pipelines,
        tidx,
        tma_a,
        tma_beta,
        tma_do,
        tma_g,
        tma_h,
        tma_k,
        tma_q,
        tma_v,
    ):
        """Run every input TMA operation from warp 9."""

        for chunk_iteration in cutlass.range(sequence_chunks):
            previous_iteration = chunk_iteration - 1
            chunk_index = sequence_chunks - 1 - chunk_iteration
            token_offset = sequence_start + chunk_index * _BT
            token_group_coord = (cutlass.Int32(0), head)
            if chunk_iteration > 0:
                self._iket_push_stage(block_idx, chunk_iteration, "TMA_WAIT_REUSE_G")
                self._wait_full_at(
                    mma_input_ready_pipelines[_G_SMEM_REUSE_READY], previous_iteration
                )
                self._iket_pop_stage(block_idx, chunk_iteration)
            load_g_producer = self._producer_at(
                load_tensor_pipelines[_LOAD_G], chunk_iteration
            )
            self._iket_push_stage(block_idx, chunk_iteration, "TMA_ISSUE_LOAD_G")
            self._issue_vector_tma(
                tma_g, s_g, load_g_producer, token_offset, head
            )
            self._iket_pop_stage(block_idx, chunk_iteration)

            if chunk_iteration > 0:
                self._iket_push_stage(block_idx, chunk_iteration, "TMA_WAIT_REUSE_H")
                self._wait_full_at(
                    mma_done_pipelines[_MMA_DK_STATE_G_DONE],
                    previous_iteration,
                )
                self._iket_pop_stage(block_idx, chunk_iteration)
            h_checkpoint = chunk_base + chunk_index
            load_h_producer = self._producer_at(
                load_tensor_pipelines[_LOAD_H], chunk_iteration
            )
            self._iket_push_stage(block_idx, chunk_iteration, "TMA_ISSUE_LOAD_H")
            self._issue_matrix_tma(
                tma_h, s_h, mma_state_direct, load_h_producer,
                (_DK, _DV), cutlass.Int32(0),
                (cutlass.Int32(0), h_checkpoint, head), False,
            )
            self._iket_pop_stage(block_idx, chunk_iteration)

            if chunk_iteration > 0:
                # Restrict the debug range to lane 0 so one source execution
                # produces exactly one warp-scoped range.
                if tidx % 32 == 0:
                    self._iket_push_stage(
                        block_idx, chunk_iteration, "TMA_WAIT_REUSE_V"
                    )
                self._wait_full_at(
                    smem_reuse_pipelines[_REUSE_V], previous_iteration
                )
                if tidx % 32 == 0:
                    self._iket_pop_stage(block_idx, chunk_iteration)
            load_v_producer = self._producer_at(
                load_tensor_pipelines[_LOAD_V], chunk_iteration
            )
            load_qk_producer = self._producer_at(
                load_tensor_pipelines[_LOAD_QK], chunk_iteration
            )
            # Arm V before Q/K can release consumers, but defer its data move
            # until both Q and K TMA copies have been issued.
            v_tma_handle = load_v_producer.acquire()
            qk_tma_handle = load_qk_producer.acquire()
            if chunk_iteration > 0:
                self._iket_push_stage(block_idx, chunk_iteration, "TMA_WAIT_REUSE_Q")
                # Stage 14 now runs before Stage 12. Both consume sQ, so the next
                # iteration may overwrite Q only after all three MMA
                # completion signals have arrived.
                self._wait_full_at(
                    mma_done_pipelines[_MMA_DH_Q_LEFT_DONE], previous_iteration
                )
                self._wait_full_at(
                    mma_done_pipelines[_MMA_DH_Q_RIGHT_DONE], previous_iteration
                )
                self._wait_full_at(
                    mma_done_pipelines[_MMA_DK_DP_DONE], previous_iteration
                )
                self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "TMA_ISSUE_LOAD_Q")
            self._copy_matrix_tma(
                tma_q, s_q, mma_token_direct, qk_tma_handle,
                (_BT, _DK), token_offset, token_group_coord, False,
            )
            self._iket_pop_stage(block_idx, chunk_iteration)

            # The previous Stage 15 MMA is the sole remaining sK reuse edge.
            # Its dK epilogue now stores TMEM fragments directly to global.
            if chunk_iteration > 0:
                self._iket_push_stage(block_idx, chunk_iteration, "TMA_WAIT_REUSE_K")
                self._wait_full_at(mma_done_pipelines[_MMA_DK_DA_DONE], previous_iteration)
                self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "TMA_ISSUE_LOAD_K")
            self._copy_matrix_tma(
                tma_k, s_k, mma_token_direct, qk_tma_handle,
                (_BT, _DK), token_offset, token_group_coord, True,
            )
            self._iket_pop_stage(block_idx, chunk_iteration)

            # P can start as soon as the shared Q/K transaction completes.
            # Issue A, beta, and dO only after Q/K so their independent TMA
            # traffic overlaps P instead of delaying its input barrier.
            if chunk_iteration > 0:
                self._iket_push_stage(block_idx, chunk_iteration, "TMA_WAIT_REUSE_A")
                self._wait_full_at(
                    smem_reuse_pipelines[_REUSE_A], previous_iteration
                )
                self._iket_pop_stage(block_idx, chunk_iteration)
                self._iket_push_stage(
                    block_idx, chunk_iteration, "TMA_WAIT_REUSE_BETA"
                )
                self._wait_full_at(
                    smem_reuse_pipelines[_REUSE_BETA], previous_iteration
                )
                self._iket_pop_stage(block_idx, chunk_iteration)
                self._iket_push_stage(
                    block_idx, chunk_iteration, "TMA_WAIT_REUSE_DO"
                )
                self._wait_full_at(
                    smem_reuse_pipelines[_REUSE_DO], previous_iteration
                )
                self._iket_pop_stage(block_idx, chunk_iteration)
            load_a_producer = self._producer_at(
                load_tensor_pipelines[_LOAD_A], chunk_iteration
            )
            self._iket_push_stage(block_idx, chunk_iteration, "TMA_ISSUE_LOAD_A_BETA_DO_V")
            self._issue_matrix_tma(
                tma_a, s_a, mma_square_direct, load_a_producer,
                (_BT, _BT), token_offset, token_group_coord, False,
            )
            load_beta_producer = self._producer_at(
                load_tensor_pipelines[_LOAD_BETA], chunk_iteration
            )
            self._issue_vector_tma(
                tma_beta, s_beta, load_beta_producer,
                token_offset, head,
            )
            load_do_producer = self._producer_at(
                load_tensor_pipelines[_LOAD_DO], chunk_iteration
            )
            self._issue_matrix_tma(
                tma_do, s_do, mma_token_direct, load_do_producer,
                (_BT, _DV), token_offset, token_group_coord, False,
            )
            self._copy_matrix_tma(
                tma_v, s_v, mma_token_direct, v_tma_handle,
                (_BT, _DV), token_offset, token_group_coord, False,
            )
            self._iket_pop_stage(block_idx, chunk_iteration)

    @cute.jit
    def run_dq_store_warp(
        self,
        block_idx,
        dq,
        head,
        s_dqkv,
        sequence_chunks,
        sequence_start,
        tail_tokens,
        dq_smem_tile_ready_pipeline,
        tidx,
        tma_dq_store,
    ):
        """Drain AQ's dQ staging tile from dedicated producer warp 11."""

        lane = tidx % 32
        if cutlass.const_expr(self.enable_varlen_tail):
            if lane == 0:
                cpasync.prefetch_descriptor(tma_dq_store[0])
            for chunk_iteration in cutlass.range(sequence_chunks):
                chunk_index = sequence_chunks - 1 - chunk_iteration
                token_offset = sequence_start + chunk_index * _BT
                valid_tokens = (
                    tail_tokens
                    if chunk_iteration == 0
                    else cutlass.Int32(_BT)
                )
                if chunk_iteration == 0:
                    dq_store_handle = self._consumer_at(
                        dq_smem_tile_ready_pipeline, chunk_iteration
                    ).wait()
                    if valid_tokens < _BT:
                        self._store_token_tile_tail(
                            dq,
                            s_dqkv,
                            lane,
                            32,
                            token_offset,
                            head,
                            valid_tokens,
                        )
                    elif lane == 0:
                        self._store_token_tile_tma(
                            tma_dq_store,
                            s_dqkv,
                            token_offset,
                            (cutlass.Int32(0), head),
                        )
                    self.tail_dq_store_barrier.arrive_and_wait()
                    if lane == 0:
                        dq_store_handle.release()
                elif lane == 0:
                    dq_store_handle = self._consumer_at(
                        dq_smem_tile_ready_pipeline, chunk_iteration
                    ).wait()
                    self._store_token_tile_tma(
                        tma_dq_store,
                        s_dqkv,
                        token_offset,
                        (cutlass.Int32(0), head),
                    )
                    dq_store_handle.release()
            if lane == 0:
                if sequence_chunks > 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=False)
        elif lane == 0:
            cpasync.prefetch_descriptor(tma_dq_store[0])
            for chunk_iteration in cutlass.range(sequence_chunks):
                chunk_index = sequence_chunks - 1 - chunk_iteration
                token_offset = sequence_start + chunk_index * _BT
                self._iket_push_stage(
                    block_idx, chunk_iteration, "STORE_WAIT_DQ"
                )
                dq_store_handle = self._consumer_at(
                    dq_smem_tile_ready_pipeline, chunk_iteration
                ).wait()
                self._iket_pop_stage(block_idx, chunk_iteration)
                self._iket_push_stage(
                    block_idx, chunk_iteration, "STORE_ISSUE_DQ"
                )
                self._store_token_tile_tma(
                    tma_dq_store,
                    s_dqkv,
                    token_offset,
                    (cutlass.Int32(0), head),
                )
                self._iket_pop_stage(block_idx, chunk_iteration)
                dq_store_handle.release()
            if sequence_chunks > 0:
                # Shared-source reuse is already protected by wait_group.read.
                # Drain the final global write before CTA exit.
                cute.arch.cp_async_bulk_wait_group(0, read=False)


    @cute.jit
    def _stage_do_gamma_to_tmem(
        self,
        dog_acc,
        s_do,
        s_scaled_exp_g,
        state_tidx,
        loaded_do_stage,
    ):
        """Load raw dO, apply gamma, and retain the FP32 tile in TMEM."""

        dog_mn_view = self._transform_tmem_layout(dog_acc)
        s_do_mn_view = self._transform_tmem_layout(s_do)
        c_dog = cute.make_identity_tensor((_BT, _DV))
        dog_for_copy = dog_acc[(None, None), 0, 0, 0]
        atom_dog_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )
        atom_dog_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )
        tiled_dog_t2r = tcgen05.make_tmem_copy(
            atom_dog_t2r, dog_for_copy
        )
        tiled_dog_r2t = tcgen05.make_tmem_copy(
            atom_dog_r2t, dog_for_copy
        )
        thr_dog_t2r = tiled_dog_t2r.get_slice(state_tidx)
        thr_dog_r2t = tiled_dog_r2t.get_slice(state_tidx)
        t_tr_cdog = thr_dog_t2r.partition_D(c_dog)
        t_rt_dog = thr_dog_r2t.partition_D(dog_mn_view)
        r_dog = cute.make_rmem_tensor_like(t_tr_cdog, self.acc_dtype)
        r_do = cute.make_rmem_tensor_like(t_tr_cdog, self.io_dtype)

        atom_dog_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                num_matrices=4, transpose=False
            ),
            self.io_dtype,
        )
        tiled_dog_s2r = cute.make_tiled_copy_D(
            atom_dog_s2r, tiled_dog_t2r
        )
        thr_dog_s2r = tiled_dog_s2r.get_slice(state_tidx)
        t_cs_do = thr_dog_s2r.partition_S(s_do_mn_view)
        t_cr_do = tiled_dog_s2r.retile(r_do)
        for sub in cutlass.range(r_dog.shape[2], unroll_full=True):
            cute.copy(
                tiled_dog_s2r,
                t_cs_do[None, 0, sub, loaded_do_stage],
                t_cr_do[None, 0, sub],
            )
            for element in cutlass.range(0, 32, 2, unroll_full=True):
                coord0 = t_tr_cdog[element, 0, sub]
                coord1 = t_tr_cdog[element + 1, 0, sub]
                (
                    r_dog[element, 0, sub],
                    r_dog[element + 1, 0, sub],
                ) = cute.arch.mul_packed_f32x2(
                    (
                        r_do[element, 0, sub].to(self.acc_dtype),
                        r_do[element + 1, 0, sub].to(self.acc_dtype),
                    ),
                    (
                        s_scaled_exp_g[coord0[0]],
                        s_scaled_exp_g[coord1[0]],
                    ),
                    rnd="rn",
                    ftz=False,
                )
            cute.copy(
                tiled_dog_r2t,
                r_dog[None, 0, sub],
                t_rt_dog[None, 0, sub, 0],
            )
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _materialize_do_gamma_from_tmem(
        self,
        dog_acc,
        s_tmp21,
        state_tidx,
    ):
        """Convert the staged FP32 dO-gamma tile to BF16 shared memory."""

        dog_mn_view = self._transform_tmem_layout(dog_acc)
        s_dog_mn_view = self._transform_tmem_layout(s_tmp21)
        c_dog = cute.make_identity_tensor((_BT, _DV))
        dog_for_copy = dog_acc[(None, None), 0, 0, 0]
        atom_dog_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )
        tiled_dog_t2r = tcgen05.make_tmem_copy(
            atom_dog_t2r, dog_for_copy
        )
        thr_dog_t2r = tiled_dog_t2r.get_slice(state_tidx)
        t_tr_dog = thr_dog_t2r.partition_S(dog_mn_view)
        t_tr_cdog = thr_dog_t2r.partition_D(c_dog)
        r_dog = cute.make_rmem_tensor_like(t_tr_cdog, self.acc_dtype)
        r_dog_out = cute.make_rmem_tensor_like(t_tr_cdog, self.io_dtype)

        atom_dog_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                num_matrices=4, transpose=False
            ),
            self.io_dtype,
        )
        tiled_dog_r2s = cute.make_tiled_copy_D(
            atom_dog_r2s, tiled_dog_t2r
        )
        thr_dog_r2s = tiled_dog_r2s.get_slice(state_tidx)
        t_cs_dog = thr_dog_r2s.partition_D(s_dog_mn_view)
        t_cr_dog = tiled_dog_r2s.retile(r_dog_out)
        for sub in cutlass.range(r_dog.shape[2], unroll_full=True):
            cute.copy(
                tiled_dog_t2r,
                t_tr_dog[None, 0, sub, 0],
                r_dog[None, 0, sub],
            )
            r_dog_out[None, 0, sub].store(
                r_dog[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_dog_r2s,
                t_cr_dog[None, 0, sub],
                t_cs_dog[None, 0, sub, 0],
            )
        cute.arch.fence_view_async_tmem_load()
        cute.arch.fence_view_async_shared()

    @cute.jit
    def run_sk_consumer(
        self,
        block_idx,
        dh0_matrix,
        dh_left_acc,
        dh_right_acc,
        dht_matrix,
        dk_acc,
        dog_acc,
        dk_matrix,
        dv,
        dv_acc,
        head,
        load_tensor_pipelines,
        mma_done_pipelines,
        smem_reuse_pipelines,
        tmp21_reuse_pipeline,
        dg_rmw_done_pipeline,
        all_tmem_readers_done_pipeline,
        dog_smem_ready_pipeline,
        s_db,
        aq_w_inputs_read_done_pipeline,
        s_dg,
        s_dv_offset,
        s_do,
        s_exp_g,
        s_g,
        s_h,
        s_k,
        s_rev_exp_g,
        s_scaled_exp_g,
        s_tmp21,
        s_tmp23,
        s_tmp41,
        s_v,
        scale,
        sequence,
        sequence_chunks,
        sequence_start,
        tail_tokens,
        mma_input_ready_pipelines,
        sdg_ownership_order_pipeline,
        tidx,
        u_acc,
        warp_idx,
    ):
        """Run the merged state and K/V consumer warpgroup."""

        cute.arch.setmaxregister_increase(
            self.consumer_SK_warpgroup_register_limit
        )
        dv_store_tidx = tidx - self.kv_warp_ids[0] * 32
        dv_store_row = dv_store_tidx // 16
        dv_store_column = (dv_store_tidx % 16) * 8
        s_dv_offset[dv_store_tidx] = cutlass.Int32(
            (
                dv_store_row * dv.stride[1]
                + head * dv.stride[2]
                + dv_store_column * dv.stride[3]
            )
            * 2
        )
        vector_head = head % _TMA_VECTOR_HEADS
        for chunk_iteration in cutlass.range(sequence_chunks):
            sk_tmem_reuse_producer = self._producer_at(
                all_tmem_readers_done_pipeline, chunk_iteration
            )
            state_dg_rmw_done_producer = self._producer_at(
                dg_rmw_done_pipeline, chunk_iteration
            )
            chunk_index = sequence_chunks - 1 - chunk_iteration
            token_offset = sequence_start + chunk_index * _BT
            valid_tokens = (
                tail_tokens if chunk_iteration == 0 else cutlass.Int32(_BT)
            )
            p_inputs_sk_ready_handle = self._producer_at(
                mma_input_ready_pipelines[_INPUT_P_READY], chunk_iteration
            ).acquire()
            p_inputs_sk_ready_handle.commit()
            self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_G")
            g_handle = self._consumer_at(
                load_tensor_pipelines[_LOAD_G], chunk_iteration
            ).wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            if cutlass.const_expr(self.enable_varlen_tail):
                if chunk_iteration == 0:
                    self._extend_gate_tail(
                        s_g, g_handle.index, vector_head, valid_tokens, tidx
                    )
                    cute.arch.fence_view_async_shared()
                    self.tail_sk_barrier.arrive_and_wait()
            self._iket_push_stage(block_idx, chunk_iteration, "SK_WORK_GATE_VECTORS")
            if tidx < _BT:
                g_value = s_g[vector_head, tidx, g_handle.index]
                g_last = s_g[
                    vector_head, _BT - 1, g_handle.index
                ]
                s_db[tidx] = 0.0
                exp_g_value = cute.math.exp2(
                    g_value * 1.4426950408889634,
                    fastmath=True,
                )
                s_dg[tidx] = 0.0
                s_exp_g[tidx] = exp_g_value
                s_scaled_exp_g[tidx] = exp_g_value * scale
                s_rev_exp_g[tidx] = cute.math.exp2(
                    (g_last - g_value) * 1.4426950408889634,
                    fastmath=True,
                )
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_GATE_VECTORS")
            self.state_tmem_store_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_GATE_PUBLISH")
            self.gate_materialized_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            g_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_H")
            loaded_h_stage = self._consumer_at(
                load_tensor_pipelines[_LOAD_H], chunk_iteration
            ).wait().index
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "SK_WORK_STATE_SEED_SNAPSHOT"
            )
            dvprime_state_inputs_ready_handle = self._producer_at(
                mma_input_ready_pipelines[_INPUT_DVPRIME_STATE_READY], chunk_iteration
            ).acquire()
            state_tidx = tidx
            g_last = s_exp_g[_BT - 1]
            self._seed_state_half(
                dh_left_acc,
                dht_matrix,
                s_h,
                s_tmp41,
                sequence,
                head,
                state_tidx,
                g_last,
                chunk_iteration == 0,
                loaded_h_stage,
                0,
                True,
            )
            self._seed_state_half(
                dh_right_acc,
                dht_matrix,
                s_h,
                s_tmp41,
                sequence,
                head,
                state_tidx,
                g_last,
                chunk_iteration == 0,
                loaded_h_stage,
                1,
                True,
            )
            dvprime_state_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self.run_kv_dvprime_scale(
                block_idx, dv_acc, s_rev_exp_g, mma_done_pipelines,
                mma_input_ready_pipelines, tidx, chunk_iteration,
            )

            self._iket_push_stage(
                block_idx, chunk_iteration, "SK_WORK_STATE_H_DOT"
            )
            state_dot = self._seed_state_half(
                dh_left_acc,
                dht_matrix,
                s_h,
                s_tmp41,
                sequence,
                head,
                state_tidx,
                g_last,
                False,
                loaded_h_stage,
                0,
                False,
            )
            state_dot += self._seed_state_half(
                dh_right_acc,
                dht_matrix,
                s_h,
                s_tmp41,
                sequence,
                head,
                state_tidx,
                g_last,
                False,
                loaded_h_stage,
                1,
                False,
            )
            lane = cute.arch.lane_idx()
            state_warp_sum = cute.arch.warp_reduction_sum(state_dot)
            if lane == 0:
                s_db[warp_idx] = state_warp_sum
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_STATE_REDUCE")
            self.state_tmem_store_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "SK_WORK_STATE_BOUNDARY_DG")
            if warp_idx == self.state_warp_ids[0]:
                boundary = s_db[lane] if lane < 4 else 0.0
                boundary = cute.arch.warp_reduction_sum(boundary)
                if lane == 0:
                    if cutlass.const_expr(self.enable_varlen_tail):
                        s_dg[valid_tokens - 1] += boundary
                    else:
                        s_dg[_BT - 1] += boundary
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_STATE_PUBLISH")
            self.state_tmem_store_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self.run_kv_consumer_early(
                block_idx,
                dk_acc,
                dog_acc,
                dv,
                dv_acc,
                s_exp_g,
                head,
                load_tensor_pipelines,
                mma_done_pipelines,
                smem_reuse_pipelines,
                dog_smem_ready_pipeline,
                s_db,
                aq_w_inputs_read_done_pipeline,
                s_dg,
                s_do,
                s_k,
                s_dv_offset,
                s_rev_exp_g,
                s_scaled_exp_g,
                s_tmp21,
                s_tmp23,
                s_v,
                mma_input_ready_pipelines,
                sdg_ownership_order_pipeline,
                tidx,
                token_offset,
                valid_tokens,
                u_acc,
                warp_idx,
                chunk_iteration,
            )
            self._iket_push_stage(
                block_idx, chunk_iteration, "SK_DG_RMW_DONE_PUBLISH"
            )
            cute.arch.fence_view_async_shared()
            dg_rmw_done_handle = state_dg_rmw_done_producer.acquire()
            dg_rmw_done_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "SK_WAIT_MMA_KT_DVG"
            )
            dh_k_left_result_handle = self._consumer_at(
                mma_done_pipelines[_MMA_DH_K_LEFT_DONE], chunk_iteration
            ).wait()
            dh_k_left_result_handle.release()
            dh_k_right_result_handle = self._consumer_at(
                mma_done_pipelines[_MMA_DH_K_RIGHT_DONE], chunk_iteration
            ).wait()
            dh_k_right_result_handle.release()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self.run_kv_consumer_late(
                block_idx,
                dk_acc, dk_matrix, head, mma_done_pipelines, mma_input_ready_pipelines,
                tidx, token_offset, valid_tokens, chunk_iteration,
            )
            self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_MMA_QT_DOG")
            dh_q_left_result_handle = self._consumer_at(
                mma_done_pipelines[_MMA_DH_Q_LEFT_DONE], chunk_iteration
            ).wait()
            dh_q_left_result_handle.release()
            dh_q_right_result_handle = self._consumer_at(
                mma_done_pipelines[_MMA_DH_Q_RIGHT_DONE], chunk_iteration
            ).wait()
            dh_q_right_result_handle.release()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "SK_TMP21_REUSE_PUBLISH"
            )
            tmp21_reuse_handle = self._producer_at(
                tmp21_reuse_pipeline, chunk_iteration
            ).acquire()
            tmp21_reuse_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            cute.arch.fence_view_async_tmem_load()
            self._iket_push_stage(
                block_idx, chunk_iteration, "SK_TMEM_REUSE_PUBLISH"
            )
            sk_tmem_reuse_handle = sk_tmem_reuse_producer.acquire()
            sk_tmem_reuse_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
        self._store_state_half(
            dh_left_acc,
            dh0_matrix,
            sequence,
            head,
            tidx,
            0,
        )
        self._store_state_half(
            dh_right_acc,
            dh0_matrix,
            sequence,
            head,
            tidx,
            1,
        )

    @cute.jit
    def run_kv_dvprime_scale(
        self, block_idx, dv_acc, s_rev_exp_g, mma_done_pipelines,
        mma_input_ready_pipelines, tidx, chunk_iteration,
    ):
        """Scale raw dVprime as soon as Stage 01 completes."""

        kv_tidx = tidx - self.kv_warp_ids[0] * 32
        dv_mn_view = self._transform_tmem_layout(dv_acc)
        c_dv = cute.make_identity_tensor((_BT, _DV))
        dv_for_copy = dv_acc[(None, None), 0, 0, 0]
        atom_dv_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )
        atom_dv_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )
        tiled_dv_t2r = tcgen05.make_tmem_copy(atom_dv_t2r, dv_for_copy)
        tiled_dv_r2t = tcgen05.make_tmem_copy(atom_dv_r2t, dv_for_copy)
        thr_dv_t2r = tiled_dv_t2r.get_slice(kv_tidx)
        thr_dv_r2t = tiled_dv_r2t.get_slice(kv_tidx)
        t_tr_dv = thr_dv_t2r.partition_S(dv_mn_view)
        t_tr_cdv = thr_dv_t2r.partition_D(c_dv)
        t_rt_dv = thr_dv_r2t.partition_D(dv_mn_view)
        r_dv = cute.make_rmem_tensor_like(t_tr_cdv, self.acc_dtype)
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_G_REUSE_PUBLISH"
        )
        g_smem_reuse_handle = self._producer_at(
            mma_input_ready_pipelines[_G_SMEM_REUSE_READY], chunk_iteration
        ).acquire()
        g_smem_reuse_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)
        dvprime_pgamma_inputs_ready_handle = self._producer_at(
            mma_input_ready_pipelines[_INPUT_DVPRIME_PGAMMA_READY], chunk_iteration
        ).acquire()
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_WAIT_MMA_DVPRIME_RAW"
        )
        dvprime_handle = self._consumer_at(
            mma_done_pipelines[_MMA_DVPRIME_STATE_DONE], chunk_iteration
        ).wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_WORK_DVPRIME_SCALE"
        )
        for sub in cutlass.range(r_dv.shape[2], unroll_full=True):
            cute.copy(
                tiled_dv_t2r,
                t_tr_dv[None, 0, sub, dvprime_handle.index],
                r_dv[None, 0, sub],
            )
            for element in cutlass.range(0, 32, 2):
                coord = t_tr_cdv[element, 0, sub]
                gate_scale = s_rev_exp_g[coord[0]]
                (
                    r_dv[element, 0, sub],
                    r_dv[element + 1, 0, sub],
                ) = cute.arch.mul_packed_f32x2(
                    (
                        r_dv[element, 0, sub],
                        r_dv[element + 1, 0, sub],
                    ),
                    (gate_scale, gate_scale),
                    rnd="rn", ftz=False,
                )
            cute.copy(
                tiled_dv_r2t,
                r_dv[None, 0, sub],
                t_rt_dv[None, 0, sub, dvprime_handle.index],
            )
        cute.arch.fence_view_async_tmem_store()
        dvprime_handle.release()
        dvprime_pgamma_inputs_ready_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)

    @cute.jit
    def run_kv_consumer_early(
        self,
        block_idx,
        dk_acc,
        dog_acc,
        dv,
        dv_acc,
        s_exp_g,
        head,
        load_tensor_pipelines,
        mma_done_pipelines,
        smem_reuse_pipelines,
        dog_smem_ready_pipeline,
        s_db,
        aq_w_inputs_read_done_pipeline,
        s_dg,
        s_do,
        s_k,
        s_dv_offset,
        s_rev_exp_g,
        s_scaled_exp_g,
        s_tmp21,
        s_tmp23,
        s_v,
        mma_input_ready_pipelines,
        sdg_ownership_order_pipeline,
        tidx,
        token_offset,
        valid_tokens,
        u_acc,
        warp_idx,
        chunk_iteration,
    ):
        """Advance the merged warpgroup through the early K/V workflow."""

        kv_load_qk_consumer = self._consumer_at(load_tensor_pipelines[_LOAD_QK], chunk_iteration)
        dvprime_state_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_DVPRIME_STATE_DONE], chunk_iteration)
        u_state_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_U_STATE_DONE], chunk_iteration)
        dv_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_DV_DONE], chunk_iteration)
        dk_state_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_DK_STATE_DONE], chunk_iteration)
        v_smem_reuse_sk_producer = self._producer_at(
            smem_reuse_pipelines[_REUSE_V], chunk_iteration
        )
        dvprime_pgamma_inputs_sk_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DVPRIME_PGAMMA_READY], chunk_iteration)
        g_smem_reuse_sk_producer = self._producer_at(mma_input_ready_pipelines[_G_SMEM_REUSE_READY], chunk_iteration)
        sdg_ownership_order_sk_producer = self._producer_at(
            sdg_ownership_order_pipeline, chunk_iteration
        )
        dk_state_inputs_sk_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DK_STATE_READY], chunk_iteration)
        u_readers_done_sk_producer = self._producer_at(mma_input_ready_pipelines[_U_READERS_DONE_BEFORE_DQ_STATE_MMA], chunk_iteration)
        dk_state_g_inputs_sk_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DK_STATE_G_READY], chunk_iteration)
        dh_k_inputs_sk_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DH_K_READY], chunk_iteration)
        kv_tidx = tidx - self.kv_warp_ids[0] * 32
        kv_row0 = (kv_tidx // 32) * 16 + (kv_tidx % 32) // 4
        kv_row1 = kv_row0 + 8
        dv_mn_view = self._transform_tmem_layout(dv_acc)
        c_dv = cute.make_identity_tensor((_BT, _DV))
        dv_for_copy = dv_acc[(None, None), 0, 0, 0]
        atom_dv_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )
        atom_dv_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )
        tiled_dv_t2r = tcgen05.make_tmem_copy(
            atom_dv_t2r, dv_for_copy
        )
        tiled_dv_r2t = tcgen05.make_tmem_copy(
            atom_dv_r2t, dv_for_copy
        )
        thr_dv_t2r = tiled_dv_t2r.get_slice(kv_tidx)
        thr_dv_r2t = tiled_dv_r2t.get_slice(kv_tidx)
        t_tr_dv = thr_dv_t2r.partition_S(dv_mn_view)
        t_tr_cdv = thr_dv_t2r.partition_D(c_dv)
        t_rt_dv = thr_dv_r2t.partition_D(dv_mn_view)
        r_dv = cute.make_rmem_tensor_like(t_tr_cdv, self.acc_dtype)
        exp_g0 = s_exp_g[kv_row0]
        exp_g1 = s_exp_g[kv_row1]
        r_dv_out = cute.make_rmem_tensor_like(r_dv, self.io_dtype)
        atom_dv_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                num_matrices=4, transpose=False
            ),
            self.io_dtype,
        )
        tiled_dv_r2s = cute.make_tiled_copy_D(
            atom_dv_r2s, tiled_dv_t2r
        )
        thr_dv_r2s = tiled_dv_r2s.get_slice(kv_tidx)
        t_cr_dv = tiled_dv_r2s.retile(r_dv_out)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_MMA_U")
        u_handle = u_state_mma_consumer.wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        sdg_ownership_order_handle = sdg_ownership_order_sk_producer.acquire()
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_MMA_DV")
        dv_handle = dv_mma_consumer.wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_WAIT_AQ_W_INPUT_READS"
        )
        w_input_reads_done_handle = self._consumer_at(
            aq_w_inputs_read_done_pipeline, chunk_iteration
        ).wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WORK_DV_DG_STORE")
        s_dvg_mn_view = self._transform_tmem_layout(s_tmp23)
        t_cs_dvg = thr_dv_r2s.partition_D(s_dvg_mn_view)
        s_dv_store_mn_view = self._transform_tmem_layout(s_v)
        t_cs_dv_store = thr_dv_r2s.partition_D(
            s_dv_store_mn_view
        )
        u_kv_mn_view = self._transform_tmem_layout(u_acc)
        t_tr_u_kv = thr_dv_t2r.partition_S(u_kv_mn_view)
        r_u_kv = cute.make_rmem_tensor_like(
            t_tr_cdv, self.acc_dtype
        )
        dvg_u_dot00 = cutlass.Float32(0.0)
        dvg_u_dot01 = cutlass.Float32(0.0)
        dvg_u_dot10 = cutlass.Float32(0.0)
        dvg_u_dot11 = cutlass.Float32(0.0)
        for sub in cutlass.range(r_dv.shape[2], unroll_full=True):
            cute.copy(
                tiled_dv_t2r,
                t_tr_dv[None, 0, sub, dv_handle.index],
                r_dv[None, 0, sub],
            )
            r_dv_out[None, 0, sub].store(
                r_dv[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_dv_r2s,
                t_cr_dv[None, 0, sub],
                t_cs_dv_store[None, 0, sub, 0],
            )
            for element in cutlass.range(0, 32, 2, unroll_full=True):
                coord = t_tr_cdv[element, 0, sub]
                gate_scale = -(
                    exp_g0 if coord[0] == kv_row0 else exp_g1
                )
                (
                    r_dv[element, 0, sub],
                    r_dv[element + 1, 0, sub],
                ) = cute.arch.mul_packed_f32x2(
                    (
                        r_dv[element, 0, sub],
                        r_dv[element + 1, 0, sub],
                    ),
                    (gate_scale, gate_scale),
                    rnd="rn",
                    ftz=False,
                )
            cute.copy(
                tiled_dv_t2r,
                t_tr_u_kv[None, 0, sub, u_handle.index],
                r_u_kv[None, 0, sub],
            )
            for group in cutlass.range(8, unroll_full=True):
                element = group * 4
                dvg_u_dot00, dvg_u_dot01 = (
                    cute.arch.fma_packed_f32x2(
                        (
                            r_dv[element, 0, sub],
                            r_dv[element + 1, 0, sub],
                        ),
                        (
                            r_u_kv[element, 0, sub],
                            r_u_kv[element + 1, 0, sub],
                        ),
                        (dvg_u_dot00, dvg_u_dot01),
                        rnd="rn",
                        ftz=False,
                    )
                )
                dvg_u_dot10, dvg_u_dot11 = (
                    cute.arch.fma_packed_f32x2(
                        (
                            r_dv[element + 2, 0, sub],
                            r_dv[element + 3, 0, sub],
                        ),
                        (
                            r_u_kv[element + 2, 0, sub],
                            r_u_kv[element + 3, 0, sub],
                        ),
                        (dvg_u_dot10, dvg_u_dot11),
                        rnd="rn",
                        ftz=False,
                    )
                )
            r_dv_out[None, 0, sub].store(
                r_dv[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_dv_r2s,
                t_cr_dv[None, 0, sub],
                t_cs_dvg[None, 0, sub, 0],
            )
        cute.arch.fence_view_async_tmem_load()
        cute.arch.fence_view_async_shared()
        # Stage 11 only consumes K and the completed dV-gamma tile in sTmp23.
        # Publish it before the independent U reduction, dV store, and sDg
        # ownership work so the MMA warp can start K^T dV-gamma immediately.
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_DVGAMMA_READY_PUBLISH"
        )
        dh_k_inputs_ready_handle = dh_k_inputs_sk_producer.acquire()
        dh_k_inputs_ready_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)
        u_handle.release()
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_WAIT_U_READERS"
        )
        self.kv_reduction_barrier.arrive_and_wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_U_READERS_DONE_PUBLISH"
        )
        u_readers_done_handle = u_readers_done_sk_producer.acquire()
        u_readers_done_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)
        dvg_u_dot0 = dvg_u_dot00 + dvg_u_dot01
        dvg_u_dot1 = dvg_u_dot10 + dvg_u_dot11
        self._commit_row_pair(
            dvg_u_dot0,
            dvg_u_dot1,
            s_dg,
            kv_row0,
            kv_row1,
            kv_tidx,
            self.kv_reduction_barrier,
            True,
        )
        self._store_token_tile_vector128(
            dv, s_v, s_dv_offset, kv_tidx, token_offset, valid_tokens
        )
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_V_REUSE_PUBLISH"
        )
        v_smem_reuse_handle = v_smem_reuse_sk_producer.acquire()
        v_smem_reuse_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)
        dv_handle.release()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_SDG_WRITE_OWNERSHIP_PUBLISH"
        )
        sdg_ownership_order_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_DV_DG_WORK_DONE_PUBLISH"
        )
        dk_state_inputs_ready_handle = dk_state_inputs_sk_producer.acquire()
        dk_state_inputs_ready_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_TMA_DO")
        loaded_do_stage = self._consumer_at(
            load_tensor_pipelines[_LOAD_DO], chunk_iteration
        ).wait().index
        self._iket_pop_stage(block_idx, chunk_iteration)
        if cutlass.const_expr(self.enable_varlen_tail):
            if chunk_iteration == 0:
                self._zero_token_matrix_tail(
                    s_do, loaded_do_stage, valid_tokens, kv_tidx, 128
                )
                cute.arch.fence_view_async_shared()
                self.tail_sk_barrier.arrive_and_wait()
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_WAIT_DQ_STATE_DONE"
        )
        self._wait_full_at(
            mma_done_pipelines[_MMA_DQ_STATE_DONE], chunk_iteration
        )
        self._iket_pop_stage(block_idx, chunk_iteration)
        w_input_reads_done_handle.release()
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_WORK_DOG_STAGE_TMEM"
        )
        self._stage_do_gamma_to_tmem(
            dog_acc,
            s_do,
            s_scaled_exp_g,
            kv_tidx,
            loaded_do_stage,
        )
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_TMA_QK")
        kv_qk_handle = kv_load_qk_consumer.wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        if cutlass.const_expr(self.enable_varlen_tail):
            if chunk_iteration == 0:
                self._zero_token_matrix_tail(
                    s_k, kv_qk_handle.index, valid_tokens, kv_tidx, 128
                )
                cute.arch.fence_view_async_shared()
                self.tail_sk_barrier.arrive_and_wait()
        s_k_mn_view = self._transform_tmem_layout(s_k)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_MMA_DK_STATE")
        dk_state_handle = dk_state_mma_consumer.wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_WORK_DOG_MATERIALIZE"
        )
        self._materialize_do_gamma_from_tmem(
            dog_acc,
            s_tmp21,
            kv_tidx,
        )
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_DO_REUSE_PUBLISH"
        )
        reuse_do_handle = self._producer_at(
            dog_smem_ready_pipeline, chunk_iteration
        ).acquire()
        reuse_do_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_DOGAMMA_READY_PUBLISH"
        )
        dh_q_inputs_ready_handle = self._producer_at(
            mma_input_ready_pipelines[_INPUT_DH_Q_READY], chunk_iteration
        ).acquire()
        dh_q_inputs_ready_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WORK_DK_STATE_DG")
        dk_mn_view = self._transform_tmem_layout(dk_acc)
        c_dk = cute.make_identity_tensor((_BT, _DK))
        dk_for_copy = dk_acc[(None, None), 0, 0, 0]
        atom_dk_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )
        atom_dk_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )
        tiled_dk_t2r = tcgen05.make_tmem_copy(
            atom_dk_t2r, dk_for_copy
        )
        tiled_dk_r2t = tcgen05.make_tmem_copy(
            atom_dk_r2t, dk_for_copy
        )
        thr_dk_t2r = tiled_dk_t2r.get_slice(kv_tidx)
        thr_dk_r2t = tiled_dk_r2t.get_slice(kv_tidx)
        t_tr_dk = thr_dk_t2r.partition_S(dk_mn_view)
        t_tr_cdk = thr_dk_t2r.partition_D(c_dk)
        t_rt_dk = thr_dk_r2t.partition_D(dk_mn_view)
        r_dk = cute.make_rmem_tensor_like(t_tr_cdk, self.acc_dtype)
        r_k_values = cute.make_rmem_tensor_like(
            t_tr_cdk[None, 0, 0], self.io_dtype
        )
        r_k = cute.make_tensor(
            r_k_values.iterator,
            cute.make_layout(
                (r_k_values.layout.shape, 1, 1),
                stride=(r_k_values.layout.stride, 0, 0),
            ),
        )
        atom_k_s2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
            num_bits_per_copy=32,
        )
        tiled_k_s2r = cute.make_tiled_copy_D(
            atom_k_s2r, tiled_dk_t2r
        )
        thr_k_s2r = tiled_k_s2r.get_slice(kv_tidx)
        t_cs_k = thr_k_s2r.partition_S(s_k_mn_view)
        t_cr_k = tiled_k_s2r.retile(r_k)
        k_dk_dot00 = cutlass.Float32(0.0)
        k_dk_dot01 = cutlass.Float32(0.0)
        k_dk_dot10 = cutlass.Float32(0.0)
        k_dk_dot11 = cutlass.Float32(0.0)
        gate_scale0 = s_rev_exp_g[kv_row0]
        gate_scale1 = s_rev_exp_g[kv_row1]
        for sub in cutlass.range(r_dk.shape[2], unroll_full=True):
            cute.copy(
                tiled_dk_t2r,
                t_tr_dk[None, 0, sub, dk_state_handle.index],
                r_dk[None, 0, sub],
            )
            cute.copy(
                tiled_k_s2r,
                t_cs_k[
                    None, 0, sub, kv_qk_handle.index
                ],
                t_cr_k[None, 0, 0],
            )
            for group in cutlass.range(8):
                element = group * 4
                (
                    r_dk[element, 0, sub],
                    r_dk[element + 1, 0, sub],
                ) = cute.arch.mul_packed_f32x2(
                    (
                        r_dk[element, 0, sub],
                        r_dk[element + 1, 0, sub],
                    ),
                    (gate_scale0, gate_scale0),
                    rnd="rn",
                    ftz=False,
                )
                (
                    r_dk[element + 2, 0, sub],
                    r_dk[element + 3, 0, sub],
                ) = cute.arch.mul_packed_f32x2(
                    (
                        r_dk[element + 2, 0, sub],
                        r_dk[element + 3, 0, sub],
                    ),
                    (gate_scale1, gate_scale1),
                    rnd="rn",
                    ftz=False,
                )
                k_dk_dot00, k_dk_dot01 = (
                    cute.arch.fma_packed_f32x2(
                        (
                            -r_k[element, 0, 0].to(self.acc_dtype),
                            -r_k[element + 1, 0, 0].to(
                                self.acc_dtype
                            ),
                        ),
                        (
                            r_dk[element, 0, sub],
                            r_dk[element + 1, 0, sub],
                        ),
                        (k_dk_dot00, k_dk_dot01),
                        rnd="rn",
                        ftz=False,
                    )
                )
                k_dk_dot10, k_dk_dot11 = (
                    cute.arch.fma_packed_f32x2(
                        (
                            -r_k[element + 2, 0, 0].to(
                                self.acc_dtype
                            ),
                            -r_k[element + 3, 0, 0].to(
                                self.acc_dtype
                            ),
                        ),
                        (
                            r_dk[element + 2, 0, sub],
                            r_dk[element + 3, 0, sub],
                        ),
                        (k_dk_dot10, k_dk_dot11),
                        rnd="rn",
                        ftz=False,
                    )
                )
            cute.copy(
                tiled_dk_r2t,
                r_dk[None, 0, sub],
                t_rt_dk[None, 0, sub, dk_state_handle.index],
            )
        cute.arch.fence_view_async_tmem_store()
        self._iket_push_stage(
            block_idx, chunk_iteration, "SK_DK_STATE_RESULT_READY_PUBLISH"
        )
        dk_state_g_inputs_ready_handle = dk_state_g_inputs_sk_producer.acquire()
        dk_state_g_inputs_ready_handle.commit()
        self._iket_pop_stage(block_idx, chunk_iteration)
        k_dk_dot0 = k_dk_dot00 + k_dk_dot01
        k_dk_dot1 = k_dk_dot10 + k_dk_dot11
        k_dk_raw_total = k_dk_dot0 + k_dk_dot1
        self._commit_row_pair(
            k_dk_dot0,
            k_dk_dot1,
            s_dg,
            kv_row0,
            kv_row1,
            kv_tidx,
            self.kv_reduction_barrier,
            True,
        )
        lane = cute.arch.lane_idx()
        warp_sum = cute.arch.warp_reduction_sum(k_dk_raw_total)
        if lane == 0:
            s_db[warp_idx - self.kv_warp_ids[0]] = warp_sum
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_DK_REDUCE_PARTIAL")
        self.kv_reduction_barrier.arrive_and_wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WORK_DK_BOUNDARY_DG")
        if warp_idx == self.kv_warp_ids[0]:
            total = s_db[lane] if lane < 4 else 0.0
            total = cute.arch.warp_reduction_sum(total)
            if lane == 0:
                if cutlass.const_expr(self.enable_varlen_tail):
                    s_dg[valid_tokens - 1] -= total
                else:
                    s_dg[_BT - 1] -= total
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_DK_REDUCE_PUBLISH")
        self.kv_reduction_barrier.arrive_and_wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        dk_state_handle.release()
        return u_handle

    @cute.jit
    def run_kv_consumer_late(
        self,
        block_idx,
        dk_acc,
        dk_matrix,
        head,
        mma_done_pipelines,
        mma_input_ready_pipelines,
        tidx,
        token_offset,
        valid_tokens,
        chunk_iteration,
    ):
        """Finish the K path after the Stage 12 and Stage 15 inputs are ready."""

        dk_dp_mma_consumer = self._consumer_at(
            mma_done_pipelines[_MMA_DK_DP_DONE], chunk_iteration
        )
        dk_da_mma_consumer = self._consumer_at(
            mma_done_pipelines[_MMA_DK_DA_DONE], chunk_iteration
        )
        dk_mn_view = self._transform_tmem_layout(dk_acc)
        kv_tidx = tidx - self.sk_warp_ids[0] * 32
        dk_for_copy = dk_acc[(None, None), 0, 0, 0]
        atom_dk_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(1)),
            self.acc_dtype,
        )
        tiled_dk_t2r = tcgen05.make_tmem_copy(
            atom_dk_t2r, dk_for_copy
        )
        thr_dk_t2r = tiled_dk_t2r.get_slice(kv_tidx)
        t_tr_dk = thr_dk_t2r.partition_S(dk_mn_view)
        if cutlass.const_expr(self.enable_varlen_tail):
            gdk_tile = cute.domain_offset(
                (token_offset, cutlass.Int32(0)),
                dk_matrix[None, None, (cutlass.Int32(0), head)],
            )
        else:
            gdk = cute.local_tile(
                dk_matrix, (_BT, _DK), (None, None, None)
            )
            gdk_tile = gdk[
                None,
                None,
                token_offset // _BT,
                0,
                (cutlass.Int32(0), head),
            ]
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_MMA_DK_DP")
        dk_dp_handle = dk_dp_mma_consumer.wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        dk_dp_handle.release()
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WAIT_MMA_DK_DA")
        dk_da_handle = dk_da_mma_consumer.wait()
        self._iket_pop_stage(block_idx, chunk_iteration)
        self._iket_push_stage(block_idx, chunk_iteration, "SK_WORK_DK_STORE")
        store_warp = kv_tidx // 32
        store_lane = kv_tidx % 32
        store_row = store_warp * 16 + store_lane // 4
        store_column = (store_lane % 4) * 2
        row0_address = cute.domain_offset(
            (store_row, store_column), gdk_tile
        ).iterator.toint()
        row8_address = cute.domain_offset(
            (store_row + 8, store_column), gdk_tile
        ).iterator.toint()
        for half in cutlass.range_constexpr(2):
            half_tmem_address = t_tr_dk[
                None, 0, half * 8, dk_da_handle.index
            ].iterator.toint()
            for group_in_half in cutlass.range(8, unroll=2):
                group = half * 8 + group_in_half
                if cutlass.const_expr(self.enable_varlen_tail):
                    _load_store_dk_bf16x2_tail(
                        half_tmem_address + group_in_half * 8,
                        row0_address + group * 16,
                        row8_address + group * 16,
                        store_row,
                        store_row + 8,
                        valid_tokens,
                    )
                else:
                    _load_store_dk_bf16x2(
                        half_tmem_address + group_in_half * 8,
                        row0_address + group * 16,
                        row8_address + group * 16,
                    )

        cute.arch.fence_view_async_tmem_load()
        dk_da_handle.release()
        self._iket_pop_stage(block_idx, chunk_iteration)

    @cute.jit
    def run_aq_consumer(
        self,
        block_idx,
        a_copy_acc,
        da_copy_acc,
        db,
        dg,
        dp_copy_acc,
        dq_acc,
        dv_acc,
        head,
        load_tensor_pipelines,
        mask_acc,
        mask_tmem_ptr,
        mma_done_pipelines,
        smem_reuse_pipelines,
        tmp21_reuse_pipeline,
        p_copy_acc,
        primary_tmem_ptr,
        da_tmem_ptr,
        dg_rmw_done_pipeline,
        all_tmem_readers_done_pipeline,
        dog_smem_ready_pipeline,
        s_a,
        aq_w_inputs_read_done_pipeline,
        s_beta,
        s_db,
        s_dg,
        s_do_padded,
        s_dqkv,
        s_exp_g,
        s_g,
        s_q,
        s_reduce,
        s_reduce2,
        s_scaled_exp_g,
        s_tmp11_base_ptr,
        s_tmp12,
        s_tmp21,
        s_tmp22,
        s_tmp22_padded,
        s_tmp41_padded,
        s_v,
        scale,
        sequence_chunks,
        sequence_start,
        tail_tokens,
        square_layout,
        square_tmem_ptr,
        mma_input_ready_pipelines,
        dq_smem_tile_ready_pipeline,
        sdg_ownership_order_pipeline,
        tidx,
        u_acc,
        vprime_acc,
    ):
        """Run the four attention/Q-consumer warps."""

        vector_head = head % _TMA_VECTOR_HEADS
        for chunk_iteration in cutlass.range(sequence_chunks):
            aq_tidx = tidx - self.aq_warp_ids[0] * 32
            valid_tokens = (
                tail_tokens if chunk_iteration == 0 else cutlass.Int32(_BT)
            )
            aq_store_dq_producer = self._producer_at(
                dq_smem_tile_ready_pipeline, chunk_iteration
            )
            aq_tmem_reuse_producer = self._producer_at(
                all_tmem_readers_done_pipeline, chunk_iteration
            )
            aq_w_inputs_read_done_producer = self._producer_at(
                aq_w_inputs_read_done_pipeline, chunk_iteration
            )
            aq_reuse_do_consumer = self._consumer_at(
                dog_smem_ready_pipeline, chunk_iteration
            )
            aq_dg_rmw_done_consumer = self._consumer_at(
                dg_rmw_done_pipeline, chunk_iteration
            )
            aq_load_a_consumer = self._consumer_at(load_tensor_pipelines[_LOAD_A], chunk_iteration)
            aq_load_beta_consumer = self._consumer_at(load_tensor_pipelines[_LOAD_BETA], chunk_iteration)
            aq_load_g_aq_consumer = self._consumer_at(load_tensor_pipelines[_LOAD_G], chunk_iteration)
            aq_load_qk_consumer = self._consumer_at(load_tensor_pipelines[_LOAD_QK], chunk_iteration)
            aq_load_v_consumer = self._consumer_at(load_tensor_pipelines[_LOAD_V], chunk_iteration)
            p_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_P_DONE], chunk_iteration)
            dvprime_pgamma_mma_consumer = self._consumer_at(
                mma_done_pipelines[_MMA_DVPRIME_PGAMMA_DONE], chunk_iteration
            )
            dq_dp_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_DQ_DP_DONE], chunk_iteration)
            da_left_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_DA_LEFT_DONE], chunk_iteration)
            da_right_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_DA_RIGHT_DONE], chunk_iteration)
            at_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_AT_DONE], chunk_iteration)
            u_state_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_U_STATE_DONE], chunk_iteration)
            vprime_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_VPRIME_DONE], chunk_iteration)
            dag_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_DAG_DONE], chunk_iteration)
            dpg_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_DPG_DONE], chunk_iteration)
            dq_state_mma_consumer = self._consumer_at(mma_done_pipelines[_MMA_DQ_STATE_DONE], chunk_iteration)
            p_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_P_READY], chunk_iteration)
            dq_dp_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DQ_DP_READY], chunk_iteration)
            da_left_at_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DA_LEFT_AT_READY], chunk_iteration)
            da_right_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DA_RIGHT_READY], chunk_iteration)
            dh_q_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DH_Q_READY], chunk_iteration)
            dk_da_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DK_DA_READY], chunk_iteration)
            dvprime_pgamma_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DVPRIME_PGAMMA_READY], chunk_iteration)
            g_smem_reuse_aq_producer = self._producer_at(mma_input_ready_pipelines[_G_SMEM_REUSE_READY], chunk_iteration)
            dv_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DV_READY], chunk_iteration)
            vprime_dag_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_VPRIME_DAG_READY], chunk_iteration)
            dpg_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DPG_READY], chunk_iteration)
            dk_state_inputs_aq_producer = self._producer_at(mma_input_ready_pipelines[_INPUT_DK_STATE_READY], chunk_iteration)
            u_readers_done_aq_producer = self._producer_at(mma_input_ready_pipelines[_U_READERS_DONE_BEFORE_DQ_STATE_MMA], chunk_iteration)
            sdg_ownership_order_aq_consumer = self._consumer_at(
                sdg_ownership_order_pipeline, chunk_iteration
            )
            chunk_index = sequence_chunks - 1 - chunk_iteration
            token_offset = sequence_start + chunk_index * _BT
            p_inputs_ready_handle = p_inputs_aq_producer.acquire()
            p_inputs_ready_handle.commit()
            dvprime_pgamma_inputs_ready_handle = dvprime_pgamma_inputs_aq_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_TMA_G")
            g_handle = aq_load_g_aq_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            if cutlass.const_expr(self.enable_varlen_tail):
                if chunk_iteration == 0:
                    self._extend_gate_tail(
                        s_g, g_handle.index, vector_head, valid_tokens, aq_tidx
                    )
                    cute.arch.fence_view_async_shared()
                    self.tail_aq_barrier.arrive_and_wait()
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_GAMMA")
            loaded_g_stage = g_handle.index
            aq_row0 = (aq_tidx // 32) * 16 + (aq_tidx % 32) // 4
            aq_row1 = aq_row0 + 8
            c_p_base = cute.make_identity_tensor((_BT, _BT))
            c_p = cute.make_tensor(
                c_p_base.iterator,
                cute.make_layout(
                    ((16, 4), (32, 2)),
                    stride=(
                        (cute.E(0), 16 * cute.E(0)),
                        (cute.E(1), 32 * cute.E(1)),
                    ),
                ),
            )
            atom_p_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
            tiled_p_t2r = tcgen05.make_tmem_copy(
                atom_p_t2r, p_copy_acc
            )
            thr_p_t2r = tiled_p_t2r.get_slice(aq_tidx)
            t_tr_p = thr_p_t2r.partition_S(p_copy_acc)
            t_tr_cp = thr_p_t2r.partition_D(c_p)
            t_tr_cp_flat = cute.group_modes(t_tr_cp, 0, cute.rank(t_tr_cp))

            c_mask = c_p
            atom_mask_r2t = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
            atom_mask_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
            tiled_mask_r2t = tcgen05.make_tmem_copy(
                atom_mask_r2t, mask_acc
            )
            tiled_mask_t2r = tcgen05.make_tmem_copy(
                atom_mask_t2r, mask_acc
            )
            thr_mask_r2t = tiled_mask_r2t.get_slice(aq_tidx)
            thr_mask_t2r = tiled_mask_t2r.get_slice(aq_tidx)
            t_rt_mask = thr_mask_r2t.partition_D(mask_acc)
            t_rt_cmask = thr_mask_r2t.partition_S(c_mask)
            t_rt_cmask_flat = cute.group_modes(
                t_rt_cmask, 0, cute.rank(t_rt_cmask)
            )
            t_tr_mask = thr_mask_t2r.partition_S(mask_acc)
            t_tr_cmask = thr_mask_t2r.partition_D(c_mask)
            r_mask = cute.make_rmem_tensor((32,), self.acc_dtype)
            mask_row = aq_tidx % _BT
            mask_column_base = (aq_tidx // _BT) * (_BT // 2)
            g_mask_row = s_g[vector_head, mask_row, loaded_g_stage]
            log2e = cutlass.Float32(1.4426950408889634)
            g_mask_row_log2e = g_mask_row * log2e
            for element in cutlass.range(0, 32, 2, unroll_full=True):
                mask_column0 = mask_column_base + element
                mask_column1 = mask_column0 + 1
                exp_arg0, exp_arg1 = cute.arch.fma_packed_f32x2(
                    (
                        -s_g[vector_head, mask_column0, loaded_g_stage],
                        -s_g[vector_head, mask_column1, loaded_g_stage],
                    ),
                    (log2e, log2e),
                    (g_mask_row_log2e, g_mask_row_log2e),
                    rnd="rn",
                    ftz=False,
                )
                r_mask[element] = cute.math.exp2(
                    exp_arg0, fastmath=True
                ) if mask_row >= mask_column0 else 0.0
                r_mask[element + 1] = cute.math.exp2(
                    exp_arg1, fastmath=True
                ) if mask_row >= mask_column1 else 0.0
            _direct_tmem_store32(mask_tmem_ptr, r_mask)
            cute.arch.fence_view_async_tmem_store()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_GAMMA_REDUCE")
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            # Gamma uses raw g only, so finish it before joining WG0's ExpG
            # publication and before waiting for P.
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_GATE_PUBLISH")
            self.gate_materialized_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            g_handle.release()
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_G_REUSE_PUBLISH")
            g_smem_reuse_handle = g_smem_reuse_aq_producer.acquire()
            g_smem_reuse_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_MMA_P")
            p_handle = p_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_P_GAMMA")
            s_tmp11_pg = cute.make_tensor(
                cute.recast_ptr(
                    _phase_local_smem_ptr(s_tmp11_base_ptr),
                    square_layout.inner,
                    dtype=self.io_dtype,
                ),
                square_layout.outer,
            )
            r_pg = cute.make_rmem_tensor((32,), self.acc_dtype)

            s_pg_rows = self._transform_tmem_layout(s_tmp11_pg)
            _direct_tmem_load32(
                primary_tmem_ptr, r_pg
            )
            r_square_out8 = cute.make_rmem_tensor(
                (8,), self.io_dtype
            )
            s_pg_base = s_pg_rows.iterator.toint()
            scale_pair = (scale, scale)
            for segment in cutlass.range_constexpr(4):
                offset = segment * 8
                for element in cutlass.range_constexpr(0, 8, 2):
                    index = offset + element
                    scaled_mask0, scaled_mask1 = cute.arch.mul_packed_f32x2(
                        (r_mask[index], r_mask[index + 1]),
                        scale_pair,
                        rnd="rn",
                        ftz=False,
                    )
                    pg0, pg1 = cute.arch.mul_packed_f32x2(
                        (r_pg[index], r_pg[index + 1]),
                        (scaled_mask0, scaled_mask1),
                        rnd="rn",
                        ftz=False,
                    )
                    r_square_out8[element] = pg0.to(self.io_dtype)
                    r_square_out8[element + 1] = pg1.to(self.io_dtype)
                linear_address = s_pg_base + (
                    mask_row * _BT + mask_column_base + offset
                ) * 2
                shared_address = linear_address ^ (
                    (linear_address >> 3) & 112
                )
                _store_shared_bf16x8(shared_address, r_square_out8)
            cute.arch.fence_view_async_shared()
            p_handle.release()
            dvprime_pgamma_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_TMA_A_BETA")
            a_handle = aq_load_a_consumer.wait()
            beta_handle = aq_load_beta_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            if cutlass.const_expr(self.enable_varlen_tail):
                if chunk_iteration == 0:
                    self._zero_square_tail(
                        s_a, a_handle.index, valid_tokens, aq_tidx, 128
                    )
                    if aq_tidx < _BT:
                        if aq_tidx >= valid_tokens:
                            s_beta[
                                vector_head, aq_tidx, beta_handle.index
                            ] = self.io_dtype(0.0)
                    cute.arch.fence_view_async_shared()
                    self.tail_aq_barrier.arrive_and_wait()
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_A_BETA_GAMMA")
            r_mask_load = cute.make_rmem_tensor((32,), self.acc_dtype)
            _direct_tmem_load32(mask_tmem_ptr, r_mask_load)
            s_a_rows = self._transform_tmem_layout(s_a)
            s_ag_rows = self._transform_tmem_layout(s_tmp12)
            s_a_base = s_a_rows.iterator.toint()
            s_ag_base = s_ag_rows.iterator.toint()
            r_a8 = cute.make_rmem_tensor((8,), self.io_dtype)
            for segment in cutlass.range_constexpr(4):
                offset = segment * 8
                linear_address = s_a_base + (
                    mask_row * _BT + mask_column_base + offset
                ) * 2
                shared_address = linear_address ^ (
                    (linear_address >> 3) & 112
                )
                _load_shared_bf16x8(shared_address, r_a8)
                for element in cutlass.range_constexpr(0, 8, 2):
                    index = offset + element
                    column0 = mask_column_base + index
                    column1 = column0 + 1
                    ag0, ag1 = cute.arch.mul_packed_f32x2(
                        (
                            r_a8[element].to(self.acc_dtype),
                            r_a8[element + 1].to(self.acc_dtype),
                        ),
                        (r_mask_load[index], r_mask_load[index + 1]),
                        rnd="rn",
                        ftz=False,
                    )
                    ag0, ag1 = cute.arch.mul_packed_f32x2(
                        (ag0, ag1),
                        (
                            s_beta[
                                vector_head, column0, beta_handle.index
                            ].to(self.acc_dtype),
                            s_beta[
                                vector_head, column1, beta_handle.index
                            ].to(self.acc_dtype),
                        ),
                        rnd="rn",
                        ftz=False,
                    )
                    r_square_out8[element] = ag0.to(self.io_dtype)
                    r_square_out8[element + 1] = ag1.to(self.io_dtype)
                linear_address = s_ag_base + (
                    mask_row * _BT + mask_column_base + offset
                ) * 2
                shared_address = linear_address ^ (
                    (linear_address >> 3) & 112
                )
                _store_shared_bf16x8(shared_address, r_square_out8)
            cute.arch.fence_view_async_shared()
            beta_handle.release()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_TMA_V")
            v_handle = aq_load_v_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            if cutlass.const_expr(self.enable_varlen_tail):
                if chunk_iteration == 0:
                    self._zero_token_matrix_tail(
                        s_v, v_handle.index, valid_tokens, aq_tidx, 128
                    )
                    cute.arch.fence_view_async_shared()
                    self.tail_aq_barrier.arrive_and_wait()
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_MMA_U")
            u_handle = u_state_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_W")

            aq_u_mn_view = self._transform_tmem_layout(u_acc)
            aq_vprime_mn_view = self._transform_tmem_layout(vprime_acc)
            c_u = cute.make_identity_tensor((_BT, _DV))
            u_for_t2r = u_acc[(None, None), 0, 0, 0]
            atom_u_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
            tiled_u_t2r = tcgen05.make_tmem_copy(atom_u_t2r, u_for_t2r)
            thr_u_t2r = tiled_u_t2r.get_slice(aq_tidx)
            t_tr_u = thr_u_t2r.partition_S(aq_u_mn_view)
            t_tr_vprime = thr_u_t2r.partition_S(aq_vprime_mn_view)
            t_tr_cu = thr_u_t2r.partition_D(c_u)
            r_w = cute.make_rmem_tensor_like(t_tr_cu, self.acc_dtype)
            r_w_out = cute.make_rmem_tensor_like(r_w, self.io_dtype)

            atom_v_s2r = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                self.io_dtype,
            )
            tiled_v_s2r = cute.make_tiled_copy_D(
                atom_v_s2r, tiled_u_t2r
            )
            thr_v_s2r = tiled_v_s2r.get_slice(aq_tidx)
            s_v_mn_view = self._transform_tmem_layout(s_v)
            t_cs_v = thr_v_s2r.partition_S(s_v_mn_view)
            r_v = cute.make_rmem_tensor_like(r_w_out, self.io_dtype)
            t_cr_v = tiled_v_s2r.retile(r_v)

            atom_w_r2s = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                self.io_dtype,
            )
            tiled_w_r2s = cute.make_tiled_copy_D(
                atom_w_r2s, tiled_u_t2r
            )
            thr_w_r2s = tiled_w_r2s.get_slice(aq_tidx)
            s_w_mn_view = self._transform_tmem_layout(s_tmp22)
            t_cs_w = thr_w_r2s.partition_D(s_w_mn_view)
            t_cr_w = tiled_w_r2s.retile(r_w_out)
            exp_g0 = s_exp_g[aq_row0]
            exp_g1 = s_exp_g[aq_row1]
            for sub in cutlass.range(r_w.shape[2], unroll_full=True):
                cute.copy(
                    tiled_u_t2r,
                    t_tr_u[None, 0, sub, u_handle.index],
                    r_w[None, 0, sub],
                )
                cute.copy(
                    tiled_v_s2r,
                    t_cs_v[None, 0, sub, v_handle.index],
                    t_cr_v[None, 0, sub],
                )
                for element in cutlass.range(32):
                    coord = t_tr_cu[element, 0, sub]
                    exp_g_value = (
                        exp_g0 if coord[0] == aq_row0 else exp_g1
                    )
                    r_w[element, 0, sub] = (
                        r_v[element, 0, sub].to(self.acc_dtype)
                        - exp_g_value * r_w[element, 0, sub]
                    )
                r_w_out[None, 0, sub].store(
                    r_w[None, 0, sub].load().to(self.io_dtype)
                )
                cute.copy(
                    tiled_w_r2s,
                    t_cr_w[None, 0, sub],
                    t_cs_w[None, 0, sub, 0],
                )
            cute.arch.fence_view_async_tmem_load()
            cute.arch.fence_view_async_shared()
            v_handle.release()
            u_handle.release()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_W_INPUT_READS_DONE_PUBLISH"
            )
            w_input_reads_done_handle = aq_w_inputs_read_done_producer.acquire()
            w_input_reads_done_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_U_READERS_DONE_PUBLISH"
            )
            u_readers_done_handle = u_readers_done_aq_producer.acquire()
            u_readers_done_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            if chunk_iteration > 0:
                self._iket_push_stage(
                    block_idx, chunk_iteration, "AQ_WAIT_TMP21_REUSE"
                )
                self._wait_full_at(
                    tmp21_reuse_pipeline, chunk_iteration - 1
                )
                self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_WAIT_MMA_DVPRIME_P"
            )
            dvprime_pg_handle = dvprime_pgamma_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_WORK_DVPRIME_MATERIALIZE"
            )
            aq_dvprime_mn_view = self._transform_tmem_layout(dv_acc)
            t_tr_dvprime = thr_u_t2r.partition_S(aq_dvprime_mn_view)
            atom_dvprime_r2s = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                self.io_dtype,
            )
            tiled_dvprime_r2s = cute.make_tiled_copy_D(
                atom_dvprime_r2s, tiled_u_t2r
            )
            thr_dvprime_r2s = tiled_dvprime_r2s.get_slice(aq_tidx)
            s_dvprime_mn_view = self._transform_tmem_layout(s_tmp21)
            t_cs_dvprime = thr_dvprime_r2s.partition_D(s_dvprime_mn_view)
            t_cr_dvprime = tiled_dvprime_r2s.retile(r_w_out)
            for sub in cutlass.range(r_w.shape[2], unroll_full=True):
                cute.copy(
                    tiled_u_t2r,
                    t_tr_dvprime[None, 0, sub, dvprime_pg_handle.index],
                    r_w[None, 0, sub],
                )
                r_w_out[None, 0, sub].store(
                    r_w[None, 0, sub].load().to(self.io_dtype)
                )
                cute.copy(
                    tiled_dvprime_r2s,
                    t_cr_dvprime[None, 0, sub],
                    t_cs_dvprime[None, 0, sub, 0],
                )
            cute.arch.fence_view_async_tmem_load()
            cute.arch.fence_view_async_shared()
            dvprime_pg_handle.release()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DV_INPUTS_READY_PUBLISH"
            )
            dv_inputs_ready_handle = dv_inputs_aq_producer.acquire()
            dv_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_VPRIME_DAG_READY_PUBLISH"
            )
            vprime_dag_inputs_ready_handle = vprime_dag_inputs_aq_producer.acquire()
            vprime_dag_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_MMA_VPRIME_DAG")
            vprime_handle = vprime_mma_consumer.wait()
            dag_handle = dag_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_VPRIME_DAG")
            dpg_inputs_ready_handle = dpg_inputs_aq_producer.acquire()
            atom_vprime_r2s = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                self.io_dtype,
            )
            tiled_vprime_r2s = cute.make_tiled_copy_D(
                atom_vprime_r2s, tiled_u_t2r
            )
            thr_vprime_r2s = tiled_vprime_r2s.get_slice(aq_tidx)
            s_vprime_mn_view = self._transform_tmem_layout(s_tmp21)
            t_cs_vprime = thr_vprime_r2s.partition_D(
                s_vprime_mn_view
            )
            t_cr_vprime = tiled_vprime_r2s.retile(r_w_out)
            for sub in cutlass.range(r_w.shape[2], unroll_full=True):
                cute.copy(
                    tiled_u_t2r,
                    t_tr_vprime[None, 0, sub, vprime_handle.index],
                    r_w[None, 0, sub],
                )
                r_w_out[None, 0, sub].store(
                    r_w[None, 0, sub].load().to(self.io_dtype)
                )
                cute.copy(
                    tiled_vprime_r2s,
                    t_cr_vprime[None, 0, sub],
                    t_cs_vprime[None, 0, sub, 0],
                )
            cute.arch.fence_view_async_shared()
            vprime_handle.release()

            t_tr_a_acc = thr_p_t2r.partition_S(a_copy_acc)
            t_tr_da_acc = thr_p_t2r.partition_S(da_copy_acc)
            atom_da_acc_r2t = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
            tiled_da_acc_r2t = tcgen05.make_tmem_copy(
                atom_da_acc_r2t, da_copy_acc
            )
            thr_da_acc_r2t = tiled_da_acc_r2t.get_slice(aq_tidx)
            t_rt_da_acc = thr_da_acc_r2t.partition_D(da_copy_acc)
            r_mask_phase6 = cute.make_rmem_tensor((32,), self.acc_dtype)
            _direct_tmem_load32(mask_tmem_ptr, r_mask_phase6)
            _direct_tmem_load32(
                square_tmem_ptr, r_pg
            )
            for element in cutlass.range(32, unroll_full=True):
                r_pg[element] *= r_mask_phase6[element]
            _direct_tmem_store32(
                da_tmem_ptr, r_pg
            )
            cute.arch.fence_view_async_tmem_store()
            dag_handle.release()
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DPG_INPUTS_READY_PUBLISH"
            )
            dpg_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_MMA_DP_GAMMA")
            dpg_handle = dpg_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_DP_GAMMA")
            s_tmp11_dp = cute.make_tensor(
                cute.recast_ptr(
                    _phase_local_smem_ptr(s_tmp11_base_ptr),
                    square_layout.inner,
                    dtype=self.io_dtype,
                ),
                square_layout.outer,
            )
            t_tr_dp = thr_p_t2r.partition_S(dp_copy_acc)
            r_p = cute.make_rmem_tensor_like(r_pg, self.acc_dtype)
            r_mask_phase7 = cute.make_rmem_tensor((32,), self.acc_dtype)
            _direct_tmem_load32(mask_tmem_ptr, r_mask_phase7)
            _direct_tmem_load32(
                square_tmem_ptr, r_pg
            )
            _direct_tmem_load32(
                primary_tmem_ptr, r_p
            )
            s_dp_rows = self._transform_tmem_layout(s_tmp11_dp)
            p_dp_antisym = cutlass.Float32(0.0)
            r_dp_out8 = cute.make_rmem_tensor((8,), self.io_dtype)
            r_dp_scratch8 = cute.make_rmem_tensor((8,), self.io_dtype)
            s_dp_base = s_dp_rows.iterator.toint()
            s_tmp22_base = s_tmp22_padded.iterator.toint()
            scale_pair = (scale, scale)
            for segment in cutlass.range_constexpr(4):
                offset = segment * 8
                for element in cutlass.range_constexpr(0, 8, 2):
                    index = offset + element
                    scaled_mask0, scaled_mask1 = cute.arch.mul_packed_f32x2(
                        (
                            r_mask_phase7[index],
                            r_mask_phase7[index + 1],
                        ),
                        scale_pair,
                        rnd="rn",
                        ftz=False,
                    )
                    r_pg[index], r_pg[index + 1] = (
                        cute.arch.mul_packed_f32x2(
                            (r_pg[index], r_pg[index + 1]),
                            (scaled_mask0, scaled_mask1),
                            rnd="rn",
                            ftz=False,
                        )
                    )
                    f_value0, f_value1 = cute.arch.mul_packed_f32x2(
                        (r_p[index], r_p[index + 1]),
                        (r_pg[index], r_pg[index + 1]),
                        rnd="rn",
                        ftz=False,
                    )
                    p_dp_antisym += f_value0
                    p_dp_antisym += f_value1
                    r_dp_scratch8[element] = f_value0.to(self.io_dtype)
                    r_dp_scratch8[element + 1] = f_value1.to(self.io_dtype)
                    r_dp_out8[element] = r_pg[index].to(self.io_dtype)
                    r_dp_out8[element + 1] = r_pg[index + 1].to(self.io_dtype)
                scratch_address = s_tmp22_base + (
                    mask_row * (_BT + 2) + mask_column_base + offset
                ) * 2
                _store_shared_bf16x8_b32(
                    scratch_address, r_dp_scratch8
                )
                linear_address = s_dp_base + (
                    mask_row * _BT + mask_column_base + offset
                ) * 2
                shared_address = linear_address ^ (
                    (linear_address >> 3) & 112
                )
                _store_shared_bf16x8(shared_address, r_dp_out8)
            cute.arch.fence_view_async_shared()
            dpg_handle.release()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_DP_REDUCE_TRANSPOSE")
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DP_REDUCE_TRANSPOSE"
            )
            for element in cutlass.range(32, unroll_full=True):
                column = mask_column_base + element
                p_dp_antisym -= s_tmp22_padded[
                    column, mask_row
                ].to(self.acc_dtype)
            s_reduce[mask_row, aq_tidx // _BT] = p_dp_antisym
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_DP_REDUCE_ACCUM")
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_WAIT_SDG_OWNERSHIP"
            )
            sdg_ownership_order_handle = (
                sdg_ownership_order_aq_consumer.wait()
            )
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DP_REDUCE_ACCUM"
            )
            if aq_tidx < _BT:
                s_dg[mask_row] += (
                    s_reduce[mask_row, 0] + s_reduce[mask_row, 1]
                )
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_DP_PUBLISH")
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_P_DG_WORK_DONE_PUBLISH"
            )
            dk_state_inputs_ready_handle = dk_state_inputs_aq_producer.acquire()
            dk_state_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_MMA_DQ_STATE")
            dq_state_handle = dq_state_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_DQ_STATE_SETUP")
            dq_mn_view = self._transform_tmem_layout(dq_acc)
            c_dq = cute.make_identity_tensor((_BT, _DK))
            dq_for_copy = dq_acc[(None, None), 0, 0, 0]
            atom_dq_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
            atom_dq_r2t = cute.make_copy_atom(
                tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
            tiled_dq_t2r = tcgen05.make_tmem_copy(
                atom_dq_t2r, dq_for_copy
            )
            tiled_dq_r2t = tcgen05.make_tmem_copy(
                atom_dq_r2t, dq_for_copy
            )
            thr_dq_t2r = tiled_dq_t2r.get_slice(aq_tidx)
            thr_dq_r2t = tiled_dq_r2t.get_slice(aq_tidx)
            t_tr_dq = thr_dq_t2r.partition_S(dq_mn_view)
            t_tr_cdq = thr_dq_t2r.partition_D(c_dq)
            t_rt_dq = thr_dq_r2t.partition_D(dq_mn_view)
            r_dq = cute.make_rmem_tensor_like(t_tr_cdq, self.acc_dtype)
            r_dq_out = cute.make_rmem_tensor_like(r_dq, self.io_dtype)
            atom_dq_r2s = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                self.io_dtype,
            )
            tiled_dq_r2s = cute.make_tiled_copy_D(
                atom_dq_r2s, tiled_dq_t2r
            )
            thr_dq_r2s = tiled_dq_r2s.get_slice(aq_tidx)
            s_dq_store_mn_view = self._transform_tmem_layout(s_dqkv)
            t_cs_dq_store = thr_dq_r2s.partition_D(
                s_dq_store_mn_view
            )
            t_cr_dq = tiled_dq_r2s.retile(r_dq_out)
            s_q_mn_view = self._transform_tmem_layout(s_q)
            r_q = cute.make_rmem_tensor_like(t_tr_cdq, self.io_dtype)
            atom_q_s2r = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                self.io_dtype,
            )
            tiled_q_s2r = cute.make_tiled_copy_D(
                atom_q_s2r, tiled_dq_t2r
            )
            thr_q_s2r = tiled_q_s2r.get_slice(aq_tidx)
            t_cs_q = thr_q_s2r.partition_S(s_q_mn_view)
            t_cr_q = tiled_q_s2r.retile(r_q)
            q_dq_dot00 = cutlass.Float32(0.0)
            q_dq_dot01 = cutlass.Float32(0.0)
            q_dq_dot10 = cutlass.Float32(0.0)
            q_dq_dot11 = cutlass.Float32(0.0)
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_TMA_QK")
            aq_qk_handle = aq_load_qk_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            if cutlass.const_expr(self.enable_varlen_tail):
                if chunk_iteration == 0:
                    self._zero_token_matrix_tail(
                        s_q, aq_qk_handle.index, valid_tokens, aq_tidx, 128
                    )
                    cute.arch.fence_view_async_shared()
                    self.tail_aq_barrier.arrive_and_wait()
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_DQ_STATE")
            for sub in cutlass.range(r_dq.shape[2], unroll_full=True):
                cute.copy(
                    tiled_dq_t2r,
                    t_tr_dq[None, 0, sub, dq_state_handle.index],
                    r_dq[None, 0, sub],
                )
                cute.copy(
                    tiled_q_s2r,
                    t_cs_q[None, 0, sub, aq_qk_handle.index],
                    t_cr_q[None, 0, sub],
                )
                for element in cutlass.range(0, 32, 2, unroll_full=True):
                    coord0 = t_tr_cdq[element, 0, sub]
                    coord1 = t_tr_cdq[element + 1, 0, sub]
                    (
                        r_dq[element, 0, sub],
                        r_dq[element + 1, 0, sub],
                    ) = cute.arch.mul_packed_f32x2(
                        (
                            r_dq[element, 0, sub],
                            r_dq[element + 1, 0, sub],
                        ),
                        (
                            s_scaled_exp_g[coord0[0]],
                            s_scaled_exp_g[coord1[0]],
                        ),
                        rnd="rn",
                        ftz=False,
                    )
                    if coord0[0] == aq_row0:
                        q_dq_dot00, q_dq_dot01 = (
                            cute.arch.fma_packed_f32x2(
                                (
                                    r_q[element, 0, sub].to(self.acc_dtype),
                                    r_q[element + 1, 0, sub].to(
                                        self.acc_dtype
                                    ),
                                ),
                                (
                                    r_dq[element, 0, sub],
                                    r_dq[element + 1, 0, sub],
                                ),
                                (q_dq_dot00, q_dq_dot01),
                                rnd="rn",
                                ftz=False,
                            )
                        )
                    else:
                        q_dq_dot10, q_dq_dot11 = (
                            cute.arch.fma_packed_f32x2(
                                (
                                    r_q[element, 0, sub].to(self.acc_dtype),
                                    r_q[element + 1, 0, sub].to(
                                        self.acc_dtype
                                    ),
                                ),
                                (
                                    r_dq[element, 0, sub],
                                    r_dq[element + 1, 0, sub],
                                ),
                                (q_dq_dot10, q_dq_dot11),
                                rnd="rn",
                                ftz=False,
                            )
                        )
                cute.copy(
                    tiled_dq_r2t,
                    r_dq[None, 0, sub],
                    t_rt_dq[None, 0, sub, dq_state_handle.index],
                )
            cute.arch.fence_view_async_tmem_store()
            dq_state_handle.release()
            q_dq_dot0 = q_dq_dot00 + q_dq_dot01
            q_dq_dot1 = q_dq_dot10 + q_dq_dot11
            # Keep q*dQ out of sDg until the dedicated SK->AQ edge proves
            # that every -K*dK/boundary contribution is published. This
            # separates sDg ownership from the later Stage 10 result lifetime.
            q_dq_dot0 = cute.arch.warp_reduction_sum(
                q_dq_dot0, threads_in_group=4
            )
            q_dq_dot1 = cute.arch.warp_reduction_sum(
                q_dq_dot1, threads_in_group=4
            )
            if aq_tidx % 4 == 0:
                s_reduce[aq_row0, 0] = q_dq_dot0
                s_reduce[aq_row1, 0] = q_dq_dot1
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_WAIT_DQ_PARTIAL_PUBLISH"
            )
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dq_dp_inputs_ready_handle = dq_dp_inputs_aq_producer.acquire()
            dq_dp_inputs_ready_handle.commit()
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DQ_PARTIAL_ORDER_PUBLISH"
            )
            dh_q_inputs_ready_handle = dh_q_inputs_aq_producer.acquire()
            dh_q_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_WAIT_DQ_STORE_REUSE"
            )
            dq_store_handle = aq_store_dq_producer.acquire()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_WAIT_DG_RMW_DONE"
            )
            dg_rmw_done_handle = aq_dg_rmw_done_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_WORK_DQ_DG_FINALIZE"
            )
            if aq_tidx < _BT:
                s_dg[mask_row] += s_reduce[mask_row, 0]
            self.aq_reduction_barrier.arrive_and_wait()
            dg_rmw_done_handle.release()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_DO_REUSE")
            reuse_do_handle = aq_reuse_do_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_DAB_DB")


            dab_ab_antisym0 = cutlass.Float32(0.0)
            dab_ab_antisym1 = cutlass.Float32(0.0)
            db_column0 = cutlass.Float32(0.0)
            db_column1 = cutlass.Float32(0.0)
            _direct_tmem_load32(
                da_tmem_ptr, r_pg
            )
            r_dab_out8 = cute.make_rmem_tensor((8,), self.io_dtype)
            r_ar8 = cute.make_rmem_tensor((8,), self.io_dtype)
            r_x_scratch8 = cute.make_rmem_tensor((8,), self.io_dtype)
            r_direct_scratch8 = cute.make_rmem_tensor((8,), self.io_dtype)
            s_a_base = s_a_rows.iterator.toint()
            s_dab_base = s_ag_rows.iterator.toint()
            s_dab_x_base = s_do_padded.iterator.toint()
            s_tmp22_base = s_tmp22_padded.iterator.toint()
            for segment in cutlass.range_constexpr(4):
                offset = segment * 8
                linear_address = s_a_base + (
                    mask_row * _BT + mask_column_base + offset
                ) * 2
                shared_address = linear_address ^ (
                    (linear_address >> 3) & 112
                )
                _load_shared_bf16x8(shared_address, r_ar8)
                for element in cutlass.range_constexpr(0, 8, 2):
                    index = offset + element
                    column0 = mask_column_base + index
                    column1 = column0 + 1
                    d_ab_pair = (r_pg[index], r_pg[index + 1])
                    beta_pair = (
                        s_beta[vector_head, column0, 0].to(self.acc_dtype),
                        s_beta[vector_head, column1, 0].to(self.acc_dtype),
                    )
                    x0, x1 = cute.arch.mul_packed_f32x2(
                        d_ab_pair,
                        (
                            r_ar8[element].to(self.acc_dtype),
                            r_ar8[element + 1].to(self.acc_dtype),
                        ),
                        rnd="rn",
                        ftz=False,
                    )
                    direct0, direct1 = cute.arch.fma_packed_f32x2(
                        (x0, x1),
                        beta_pair,
                        (cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        rnd="rn",
                        ftz=False,
                    )
                    dab_ab_antisym0, dab_ab_antisym1 = (
                        cute.arch.add_packed_f32x2(
                            (dab_ab_antisym0, dab_ab_antisym1),
                            (direct0, direct1),
                            rnd="rn",
                            ftz=False,
                        )
                    )
                    d_ab_beta0, d_ab_beta1 = cute.arch.mul_packed_f32x2(
                        d_ab_pair,
                        beta_pair,
                        rnd="rn",
                        ftz=False,
                    )
                    r_x_scratch8[element] = x0.to(self.io_dtype)
                    r_x_scratch8[element + 1] = x1.to(self.io_dtype)
                    r_direct_scratch8[element] = direct0.to(self.io_dtype)
                    r_direct_scratch8[element + 1] = direct1.to(self.io_dtype)
                    r_dab_out8[element] = d_ab_beta0.to(self.io_dtype)
                    r_dab_out8[element + 1] = d_ab_beta1.to(self.io_dtype)
                scratch_offset = (
                    mask_row * (_BT + 2) + mask_column_base + offset
                ) * 2
                _store_shared_bf16x8_b32(
                    s_dab_x_base + scratch_offset, r_x_scratch8
                )
                _store_shared_bf16x8_b32(
                    s_tmp22_base + scratch_offset, r_direct_scratch8
                )
                linear_address = s_dab_base + (
                    mask_row * _BT + mask_column_base + offset
                ) * 2
                shared_address = linear_address ^ (
                    (linear_address >> 3) & 112
                )
                _store_shared_bf16x8(shared_address, r_dab_out8)
            a_handle.release()
            cute.arch.fence_view_async_shared()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_DAB_REDUCE_TRANSPOSE")
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DAB_TRANSPOSE_REDUCE"
            )
            for element in cutlass.range(0, 32, 2, unroll_full=True):
                column0 = mask_column_base + element
                column1 = column0 + 1
                dab_ab_antisym0, dab_ab_antisym1 = (
                    cute.arch.add_packed_f32x2(
                        (dab_ab_antisym0, dab_ab_antisym1),
                        (
                            -s_tmp22_padded[column0, mask_row].to(
                                self.acc_dtype
                            ),
                            -s_tmp22_padded[column1, mask_row].to(
                                self.acc_dtype
                            ),
                        ),
                        rnd="rn",
                        ftz=False,
                    )
                )
                db_column0, db_column1 = cute.arch.add_packed_f32x2(
                    (db_column0, db_column1),
                    (
                        s_do_padded[column0, mask_row].to(self.acc_dtype),
                        s_do_padded[column1, mask_row].to(self.acc_dtype),
                    ),
                    rnd="rn",
                    ftz=False,
                )
            dab_ab_antisym = dab_ab_antisym0 + dab_ab_antisym1
            db_column = db_column0 + db_column1
            reuse_do_handle.release()
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DO_REUSE_PUBLISH"
            )
            do_reuse_handle = self._producer_at(
                smem_reuse_pipelines[_REUSE_DO], chunk_iteration
            ).acquire()
            do_reuse_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            reduce_half = aq_tidx // _BT
            s_reduce[mask_row, reduce_half] = dab_ab_antisym
            s_reduce2[mask_row, reduce_half] = db_column
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_DAB_REDUCE_FINALIZE")
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DAB_ROW_FINALIZE"
            )
            if aq_tidx < _BT:
                s_dg[mask_row] += (
                    s_reduce[mask_row, 0] + s_reduce[mask_row, 1]
                )
                s_db[mask_row] = (
                    s_reduce2[mask_row, 0] + s_reduce2[mask_row, 1]
                )
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_DAB_PUBLISH")
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DA_LEFT_AT_READY_PUBLISH"
            )
            da_left_at_inputs_ready_handle = da_left_at_inputs_aq_producer.acquire()
            da_left_at_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_MMA_DQ_DP")
            dq_dp_handle = dq_dp_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_DQ_DP_STORE")
            for sub in cutlass.range(r_dq.shape[2], unroll_full=True):
                cute.copy(
                    tiled_dq_t2r,
                    t_tr_dq[None, 0, sub, dq_dp_handle.index],
                    r_dq[None, 0, sub],
                )
                r_dq_out[None, 0, sub].store(
                    r_dq[None, 0, sub].load().to(self.io_dtype)
                )
                cute.copy(
                    tiled_dq_r2s,
                    t_cr_dq[None, 0, sub],
                    t_cs_dq_store[None, 0, sub, 0],
                )
            cute.arch.fence_view_async_shared()
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_WAIT_DQ_STORE_PUBLISH"
            )
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dq_dp_handle.release()
            if aq_tidx == 0:
                dq_store_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)


            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_MMA_DA_LEFT")
            da_left_handle = da_left_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_DA_LEFT")


            s_da_rows = self._transform_tmem_layout(s_tmp12)
            _direct_tmem_load32(
                da_tmem_ptr, r_pg
            )
            r_da_out8 = cute.make_rmem_tensor((8,), self.io_dtype)
            s_da_base = s_da_rows.iterator.toint()
            for segment in cutlass.range_constexpr(4):
                offset = segment * 8
                for element in cutlass.range_constexpr(8):
                    r_da_out8[element] = r_pg[offset + element].to(
                        self.io_dtype
                    )
                linear_address = s_da_base + (
                    mask_row * _BT + mask_column_base + offset
                ) * 2
                shared_address = linear_address ^ (
                    (linear_address >> 3) & 112
                )
                _store_shared_bf16x8(shared_address, r_da_out8)
            cute.arch.fence_view_async_shared()
            da_left_handle.release()
            da_right_inputs_ready_handle = da_right_inputs_aq_producer.acquire()
            da_right_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_MMA_DA_RIGHT")
            da_right_handle = da_right_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_DA_RIGHT")
            _direct_tmem_load32(
                da_tmem_ptr, r_pg
            )
            for element in cutlass.range(32, unroll_full=True):
                column = mask_column_base + element
                r_pg[element] = (
                    -r_pg[element] if mask_row > column else 0.0
                )
            da_right_handle.release()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_A_REUSE_PUBLISH"
            )
            a_reuse_handle = self._producer_at(
                smem_reuse_pipelines[_REUSE_A], chunk_iteration
            ).acquire()
            a_reuse_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_MMA_AT")
            at_handle = at_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WORK_AT_DBETA")
            r_at = cute.make_rmem_tensor_like(r_pg, self.acc_dtype)
            _direct_tmem_load32(
                square_tmem_ptr, r_at
            )
            db_at0 = cutlass.Float32(0.0)
            db_at1 = cutlass.Float32(0.0)
            beta_row = s_beta[vector_head, mask_row, 0].to(self.acc_dtype)
            r_at_scratch8 = cute.make_rmem_tensor((8,), self.io_dtype)
            s_tmp22_base = s_tmp22_padded.iterator.toint()
            for segment in cutlass.range_constexpr(4):
                offset = segment * 8
                for element in cutlass.range_constexpr(0, 8, 2):
                    index = offset + element
                    db_at0, db_at1 = cute.arch.fma_packed_f32x2(
                        (r_at[index], r_at[index + 1]),
                        (r_pg[index], r_pg[index + 1]),
                        (db_at0, db_at1),
                        rnd="rn",
                        ftz=False,
                    )
                    r_pg[index], r_pg[index + 1] = (
                        cute.arch.mul_packed_f32x2(
                            (r_pg[index], r_pg[index + 1]),
                            (beta_row, beta_row),
                            rnd="rn",
                            ftz=False,
                        )
                    )
                    r_at_scratch8[element] = r_pg[index].to(self.io_dtype)
                    r_at_scratch8[element + 1] = r_pg[index + 1].to(
                        self.io_dtype
                    )
                scratch_address = s_tmp22_base + (
                    mask_row * (_BT + 2) + mask_column_base + offset
                ) * 2
                _store_shared_bf16x8_b32(
                    scratch_address, r_at_scratch8
                )
            cute.arch.fence_view_async_shared()
            db_at = db_at0 + db_at1
            at_handle.release()
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_BETA_REUSE_PUBLISH"
            )
            beta_reuse_handle = self._producer_at(
                smem_reuse_pipelines[_REUSE_BETA], chunk_iteration
            ).acquire()
            beta_reuse_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            s_reduce[mask_row, aq_tidx // _BT] = db_at
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_WAIT_DBETA_REDUCE")
            self.aq_reduction_barrier.arrive_and_wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "AQ_DAS_FINAL")
            if aq_tidx < _BT:
                s_db[mask_row] += (
                    s_reduce[mask_row, 0] + s_reduce[mask_row, 1]
                )
            for segment in cutlass.range_constexpr(4):
                offset = segment * 8
                for element in cutlass.range_constexpr(8):
                    index = offset + element
                    column = mask_column_base + index
                    r_pg[index] += s_tmp22_padded[
                        column, mask_row
                    ].to(self.acc_dtype)
                    r_da_out8[element] = r_pg[index].to(self.io_dtype)
                linear_address = s_da_base + (
                    mask_row * _BT + mask_column_base + offset
                ) * 2
                shared_address = linear_address ^ (
                    (linear_address >> 3) & 112
                )
                _store_shared_bf16x8(shared_address, r_da_out8)
            cute.arch.fence_view_async_shared()
            self._iket_pop_stage(block_idx, chunk_iteration)

            # Stage 15 only consumes the finalized dA_s tile in sTmp12. Publish
            # it as soon as those shared-memory writes are visible so its MMA
            # can overlap the independent dg/db global stores below.
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_DK_DA_INPUTS_READY_PUBLISH"
            )
            dk_da_inputs_ready_handle = dk_da_inputs_aq_producer.acquire()
            dk_da_inputs_ready_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_OUTPUT_STORE"
            )
            if cutlass.const_expr(self.enable_varlen_tail):
                if aq_tidx < valid_tokens:
                    dg[0, token_offset + aq_tidx, head] = s_dg[aq_tidx]
                    db[0, token_offset + aq_tidx, head] = s_db[aq_tidx]
            elif aq_tidx < _BT:
                dg[0, token_offset + aq_tidx, head] = s_dg[aq_tidx]
                db[0, token_offset + aq_tidx, head] = s_db[aq_tidx]
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_SDG_OWNERSHIP_RELEASE"
            )
            sdg_ownership_order_handle.release()
            self._iket_pop_stage(block_idx, chunk_iteration)
            cute.arch.fence_view_async_tmem_load()
            self._iket_push_stage(
                block_idx, chunk_iteration, "AQ_TMEM_REUSE_PUBLISH"
            )
            tmem_reuse_handle = aq_tmem_reuse_producer.acquire()
            tmem_reuse_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

    @cute.jit
    def run_mma_warp(
        self,
        block_idx,
        dh_left_acc,
        dh_right_acc,
        dk_acc,
        dq_acc,
        dv_acc,
        load_tensor_pipelines,
        mma_done_pipelines,
        mma_p,
        mma_dvprime_state,
        mma_dvprime_pg,
        mma_u_state,
        mma_dv,
        mma_vprime,
        mma_dag,
        mma_dpg,
        mma_dk_state,
        mma_dq_state,
        mma_dk_state_g,
        mma_dq_dp,
        mma_dh_k,
        mma_dk_dp,
        mma_da_left,
        mma_at,
        mma_da_right,
        mma_dh_q,
        mma_dk_da,
        primary_tmem_ptr,
        da_tmem_ptr,
        s_do,
        s_k,
        s_q,
        s_a_p6,
        s_a_p7,
        s_do_p0,
        s_do_p1,
        s_do_p9,
        s_h_p8,
        s_h_p9,
        s_k_left_p7_a,
        s_k_left_p7_b,
        s_k_p0,
        s_k_p2,
        s_k_p8,
        s_k_right_p7_a,
        s_k_right_p7_b,
        s_k_transposed,
        s_q_p0,
        s_q_p1,
        s_q_transposed,
        s_tmp11_p1,
        s_tmp11_p2,
        s_tmp12_p1,
        s_tmp12_p2,
        s_tmp12_p6,
        s_tmp12_p7,
        s_tmp21_left_square_transposed,
        s_tmp21_p0a,
        s_tmp21_p0b,
        s_tmp21_p1,
        s_tmp21_p9,
        s_tmp21_right_square_transposed,
        s_tmp22_p0b,
        s_tmp22_p2,
        s_tmp23_left_square_transposed,
        s_tmp23_p9,
        s_tmp23_right_square_transposed,
        s_tmp41_p8,
        s_tmp41_p9,
        sequence_chunks,
        tail_tokens,
        square_tmem_ptr,
        mma_input_ready_pipelines,
        all_tmem_readers_done_pipeline,
        u_acc,
        vprime_acc,
    ):
        """Run warp 8 tcgen05 MMA production."""

        for chunk_iteration in cutlass.range(sequence_chunks):
            valid_tokens = (
                tail_tokens if chunk_iteration == 0 else cutlass.Int32(_BT)
            )
            if chunk_iteration > 0:
                self._iket_push_stage(
                    block_idx, chunk_iteration, "MMA_WAIT_TMEM_REUSE"
                )
                self._wait_full_at(
                    all_tmem_readers_done_pipeline, chunk_iteration - 1
                )
                self._iket_pop_stage(block_idx, chunk_iteration)
            mma_load_do_consumer = self._consumer_at(load_tensor_pipelines[_LOAD_DO], chunk_iteration)
            mma_load_h_consumer = self._consumer_at(load_tensor_pipelines[_LOAD_H], chunk_iteration)
            mma_load_qk_consumer = self._consumer_at(load_tensor_pipelines[_LOAD_QK], chunk_iteration)
            p_mma_producer = self._producer_at(mma_done_pipelines[_MMA_P_DONE], chunk_iteration)
            dk_state_g_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DK_STATE_G_DONE], chunk_iteration)
            dq_dp_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DQ_DP_DONE], chunk_iteration)
            dh_k_left_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DH_K_LEFT_DONE], chunk_iteration)
            dh_k_right_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DH_K_RIGHT_DONE], chunk_iteration)
            dk_dp_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DK_DP_DONE], chunk_iteration)
            da_left_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DA_LEFT_DONE], chunk_iteration)
            da_right_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DA_RIGHT_DONE], chunk_iteration)
            at_mma_producer = self._producer_at(mma_done_pipelines[_MMA_AT_DONE], chunk_iteration)
            dh_q_left_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DH_Q_LEFT_DONE], chunk_iteration)
            dh_q_right_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DH_Q_RIGHT_DONE], chunk_iteration)
            dvprime_state_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DVPRIME_STATE_DONE], chunk_iteration)
            dk_da_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DK_DA_DONE], chunk_iteration)
            dvprime_pgamma_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DVPRIME_PGAMMA_DONE], chunk_iteration)
            u_state_mma_producer = self._producer_at(mma_done_pipelines[_MMA_U_STATE_DONE], chunk_iteration)
            dv_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DV_DONE], chunk_iteration)
            vprime_mma_producer = self._producer_at(mma_done_pipelines[_MMA_VPRIME_DONE], chunk_iteration)
            dag_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DAG_DONE], chunk_iteration)
            dpg_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DPG_DONE], chunk_iteration)
            dk_state_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DK_STATE_DONE], chunk_iteration)
            dq_state_mma_producer = self._producer_at(mma_done_pipelines[_MMA_DQ_STATE_DONE], chunk_iteration)
            p_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_P_READY], chunk_iteration)
            dq_dp_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DQ_DP_READY], chunk_iteration)
            dh_k_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DH_K_READY], chunk_iteration)
            da_left_at_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DA_LEFT_AT_READY], chunk_iteration)
            da_right_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DA_RIGHT_READY], chunk_iteration)
            dh_q_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DH_Q_READY], chunk_iteration)
            dk_da_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DK_DA_READY], chunk_iteration)
            dvprime_state_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DVPRIME_STATE_READY], chunk_iteration)
            dvprime_pgamma_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DVPRIME_PGAMMA_READY], chunk_iteration)
            dv_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DV_READY], chunk_iteration)
            vprime_dag_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_VPRIME_DAG_READY], chunk_iteration)
            dpg_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DPG_READY], chunk_iteration)
            dk_state_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DK_STATE_READY], chunk_iteration)
            u_readers_done_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_U_READERS_DONE_BEFORE_DQ_STATE_MMA], chunk_iteration)
            dk_state_g_inputs_mma_consumer = self._consumer_at(mma_input_ready_pipelines[_INPUT_DK_STATE_G_READY], chunk_iteration)
            dk_da_handle = dk_da_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            p_inputs_ready_handle = p_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_TMA")
            qk_handle = mma_load_qk_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            if cutlass.const_expr(self.enable_varlen_tail):
                if chunk_iteration == 0:
                    mma_tidx = cute.arch.lane_idx()
                    self._zero_token_matrix_tail(
                        s_q, qk_handle.index, valid_tokens, mma_tidx, 32
                    )
                    self._zero_token_matrix_tail(
                        s_k, qk_handle.index, valid_tokens, mma_tidx, 32
                    )
                    cute.arch.fence_view_async_shared()
                    self.tail_mma_barrier.arrive_and_wait()
            p_handle = p_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_p
            q_operand = tiled_mma.make_fragment_A(s_q_p0)
            k_operand = tiled_mma.make_fragment_B(s_k_p0)
            num_kphases = cute.size(q_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                _direct_mma_ws(
                    primary_tmem_ptr,
                    q_operand[
                        None, None, kphase_idx, qk_handle.index
                    ],
                    k_operand[
                        None, None, kphase_idx, qk_handle.index
                    ],
                    68158608,
                    kphase_idx != 0,
                )
            qk_handle.release()
            p_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)

            p_inputs_ready_handle.release()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dvprime_state_inputs_ready_handle = dvprime_state_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dvprime_handle = dvprime_state_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dvprime_state
            k_operand = tiled_mma.make_fragment_A(s_k_p8)
            dh_operand = tiled_mma.make_fragment_B(s_tmp41_p8)
            num_kphases = cute.size(k_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                cute.gemm(
                    tiled_mma,
                    dv_acc[None, None, None, dvprime_handle.index],
                    k_operand[
                        None, None, kphase_idx, dvprime_state_inputs_ready_handle.index
                    ],
                    dh_operand[
                        None, None, kphase_idx, dvprime_state_inputs_ready_handle.index
                    ],
                    dv_acc[None, None, None, dvprime_handle.index],
                )
            dvprime_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dvprime_state_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_TMA")
            h_handle = mma_load_h_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            u_handle = u_state_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_u_state
            k_operand = tiled_mma.make_fragment_A(s_k_p8)
            h_operand = tiled_mma.make_fragment_B(s_h_p8)
            num_kphases = cute.size(k_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                cute.gemm(
                    tiled_mma,
                    u_acc[None, None, None, u_handle.index],
                    k_operand[
                        None, None, kphase_idx, cutlass.Int32(0)
                    ],
                    h_operand[
                        None, None, kphase_idx, h_handle.index
                    ],
                    u_acc[None, None, None, u_handle.index],
                )
            u_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            h_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dvprime_pgamma_inputs_ready_handle = dvprime_pgamma_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_TMA")
            do_handle = mma_load_do_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            if cutlass.const_expr(self.enable_varlen_tail):
                if chunk_iteration == 0:
                    self._zero_token_matrix_tail(
                        s_do,
                        do_handle.index,
                        valid_tokens,
                        cute.arch.lane_idx(),
                        32,
                    )
                    cute.arch.fence_view_async_shared()
                    self.tail_mma_barrier.arrive_and_wait()
            dvprime_pg_handle = dvprime_pgamma_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dvprime_pg
            pg_operand = tiled_mma.make_fragment_A(s_tmp11_p1)
            do_operand = tiled_mma.make_fragment_B(s_do_p1)
            num_kphases = cute.size(pg_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    tiled_mma,
                    dv_acc[None, None, None, dvprime_pg_handle.index],
                    pg_operand[
                        None, None, kphase_idx, dvprime_pgamma_inputs_ready_handle.index
                    ],
                    do_operand[
                        None, None, kphase_idx, do_handle.index
                    ],
                    dv_acc[None, None, None, dvprime_pg_handle.index],
                )
            dvprime_pg_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            do_handle.release()
            dvprime_pgamma_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dv_inputs_ready_handle = dv_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dv_handle = dv_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dv
            ag_operand = tiled_mma.make_fragment_A(s_tmp12_p1)
            dvprime_operand = tiled_mma.make_fragment_B(
                s_tmp21_p1
            )
            num_kphases = cute.size(ag_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                cute.gemm(
                    tiled_mma,
                    dv_acc[None, None, None, dv_handle.index],
                    ag_operand[
                        None, None, kphase_idx, dv_inputs_ready_handle.index
                    ],
                    dvprime_operand[
                        None, None, kphase_idx, dv_inputs_ready_handle.index
                    ],
                    dv_acc[None, None, None, dv_handle.index],
                )
            dv_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dv_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            vprime_dag_inputs_ready_handle = vprime_dag_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            vprime_handle = vprime_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_vprime
            ag_operand = tiled_mma.make_fragment_A(s_tmp12_p2)
            w_operand = tiled_mma.make_fragment_B(s_tmp22_p2)
            num_kphases = cute.size(ag_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                cute.gemm(
                    tiled_mma,
                    vprime_acc[None, None, None, vprime_handle.index],
                    ag_operand[
                        None, None, kphase_idx, vprime_dag_inputs_ready_handle.index
                    ],
                    w_operand[
                        None, None, kphase_idx, vprime_dag_inputs_ready_handle.index
                    ],
                    vprime_acc[None, None, None, vprime_handle.index],
                )
            vprime_handle.commit()

            dag_handle = dag_mma_producer.acquire()
            tiled_mma = mma_dag
            dvprime_operand = tiled_mma.make_fragment_A(s_tmp21_p0a)
            w_operand = tiled_mma.make_fragment_B(s_tmp22_p0b)
            num_kphases = cute.size(dvprime_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                _direct_mma_ws(
                    square_tmem_ptr,
                    dvprime_operand[
                        None, None, kphase_idx, vprime_dag_inputs_ready_handle.index
                    ],
                    w_operand[
                        None, None, kphase_idx, vprime_dag_inputs_ready_handle.index
                    ],
                    68158608,
                    kphase_idx != 0,
                )
            dag_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            vprime_dag_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dpg_inputs_ready_handle = dpg_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dpg_handle = dpg_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dpg
            do_operand = tiled_mma.make_fragment_A(s_do_p0)
            vprime_operand = tiled_mma.make_fragment_B(s_tmp21_p0b)
            num_kphases = cute.size(do_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                _direct_mma_ws(
                    square_tmem_ptr,
                    do_operand[
                        None, None, kphase_idx, dpg_inputs_ready_handle.index
                    ],
                    vprime_operand[
                        None, None, kphase_idx, dpg_inputs_ready_handle.index
                    ],
                    68158608,
                    kphase_idx != 0,
                )
            dpg_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dpg_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dh_k_inputs_ready_handle = dh_k_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dh_k_left_handle = dh_k_left_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dh_k
            k_operand = tiled_mma.make_fragment_A(s_k_transposed)
            dvg_left_operand = tiled_mma.make_fragment_B(
                s_tmp23_left_square_transposed
            )
            num_kphases = cute.size(k_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    tiled_mma,
                    dh_left_acc[
                        None, None, None, dh_k_left_handle.index
                    ],
                    k_operand[
                        None, None, kphase_idx, dh_k_inputs_ready_handle.index
                    ],
                    dvg_left_operand[
                        None, None, kphase_idx, dh_k_inputs_ready_handle.index
                    ],
                    dh_left_acc[
                        None, None, None, dh_k_left_handle.index
                    ],
                )
            dh_k_left_handle.commit()

            dh_k_right_handle = dh_k_right_mma_producer.acquire()
            dvg_right_operand = tiled_mma.make_fragment_B(
                s_tmp23_right_square_transposed
            )
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    tiled_mma,
                    dh_right_acc[
                        None, None, None, dh_k_right_handle.index
                    ],
                    k_operand[
                        None, None, kphase_idx, dh_k_inputs_ready_handle.index
                    ],
                    dvg_right_operand[
                        None, None, kphase_idx, dh_k_inputs_ready_handle.index
                    ],
                    dh_right_acc[
                        None, None, None, dh_k_right_handle.index
                    ],
                )
            dh_k_right_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dh_k_inputs_ready_handle.release()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            u_readers_done_handle = u_readers_done_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dq_state_handle = dq_state_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dq_state
            do_operand = tiled_mma.make_fragment_A(s_do_p9)
            h_operand = tiled_mma.make_fragment_B(s_h_p9)
            num_kphases = cute.size(do_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                cute.gemm(
                    tiled_mma,
                    dq_acc[None, None, None, dq_state_handle.index],
                    do_operand[
                        None, None, kphase_idx, u_readers_done_handle.index
                    ],
                    h_operand[
                        None, None, kphase_idx, u_readers_done_handle.index
                    ],
                    dq_acc[None, None, None, dq_state_handle.index],
                )
            dq_state_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            u_readers_done_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dk_state_inputs_ready_handle = dk_state_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dk_state_handle = dk_state_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dk_state
            vprime_operand = tiled_mma.make_fragment_A(s_tmp21_p9)
            dh_operand = tiled_mma.make_fragment_B(s_tmp41_p9)
            num_kphases = cute.size(vprime_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                cute.gemm(
                    tiled_mma,
                    dk_acc[None, None, None, dk_state_handle.index],
                    vprime_operand[
                        None, None, kphase_idx, dk_state_inputs_ready_handle.index
                    ],
                    dh_operand[
                        None, None, kphase_idx, dk_state_inputs_ready_handle.index
                    ],
                    dk_acc[None, None, None, dk_state_handle.index],
                )
            dk_state_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dk_state_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dq_dp_inputs_ready_handle = dq_dp_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dq_dp_handle = dq_dp_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dq_dp
            dp_operand = tiled_mma.make_fragment_A(s_tmp11_p2)
            k_operand = tiled_mma.make_fragment_B(s_k_p2)
            num_kphases = cute.size(dp_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    tiled_mma,
                    dq_acc[None, None, None, dq_dp_handle.index],
                    dp_operand[
                        None, None, kphase_idx, dq_dp_inputs_ready_handle.index
                    ],
                    k_operand[
                        None, None, kphase_idx, dq_dp_inputs_ready_handle.index
                    ],
                    dq_acc[None, None, None, dq_dp_handle.index],
                )
            dq_dp_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dq_dp_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dh_q_inputs_ready_handle = dh_q_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dh_q_left_handle = dh_q_left_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dh_q
            q_operand = tiled_mma.make_fragment_A(s_q_transposed)
            dog_left_operand = tiled_mma.make_fragment_B(
                s_tmp21_left_square_transposed
            )
            num_kphases = cute.size(q_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    tiled_mma,
                    dh_left_acc[
                        None, None, None, dh_q_left_handle.index
                    ],
                    q_operand[
                        None, None, kphase_idx, dh_q_inputs_ready_handle.index
                    ],
                    dog_left_operand[
                        None, None, kphase_idx, dh_q_inputs_ready_handle.index
                    ],
                    dh_left_acc[
                        None, None, None, dh_q_left_handle.index
                    ],
                )
            dh_q_left_handle.commit()

            dh_q_right_handle = dh_q_right_mma_producer.acquire()
            dog_right_operand = tiled_mma.make_fragment_B(
                s_tmp21_right_square_transposed
            )
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    tiled_mma,
                    dh_right_acc[
                        None, None, None, dh_q_right_handle.index
                    ],
                    q_operand[
                        None, None, kphase_idx, dh_q_inputs_ready_handle.index
                    ],
                    dog_right_operand[
                        None, None, kphase_idx, dh_q_inputs_ready_handle.index
                    ],
                    dh_right_acc[
                        None, None, None, dh_q_right_handle.index
                    ],
                )
            dh_q_right_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dh_q_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dk_state_g_inputs_ready_handle = dk_state_g_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dk_state_g_handle = dk_state_g_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dk_state_g
            dvg_operand = tiled_mma.make_fragment_A(s_tmp23_p9)
            h_operand = tiled_mma.make_fragment_B(s_h_p9)
            num_kphases = cute.size(dvg_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    tiled_mma,
                    dk_acc[None, None, None, dk_state_g_handle.index],
                    dvg_operand[
                        None, None, kphase_idx, dk_state_g_inputs_ready_handle.index
                    ],
                    h_operand[
                        None, None, kphase_idx, dk_state_g_inputs_ready_handle.index
                    ],
                    dk_acc[None, None, None, dk_state_g_handle.index],
                )
            dk_state_g_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dk_state_g_inputs_ready_handle.release()

            self._iket_push_stage(
                block_idx, chunk_iteration, "MMA_WAIT_DK_ACCUMULATOR_ORDER"
            )
            dk_state_g_done_handle = self._consumer_at(
                mma_done_pipelines[_MMA_DK_STATE_G_DONE], chunk_iteration
            ).wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dk_dp_handle = dk_dp_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dk_dp
            dp_operand = tiled_mma.make_fragment_A(s_tmp11_p1)
            q_operand = tiled_mma.make_fragment_B(s_q_p1)
            num_kphases = cute.size(dp_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    tiled_mma,
                    dk_acc[None, None, None, dk_dp_handle.index],
                    dp_operand[
                        None, None, kphase_idx, dk_state_g_done_handle.index
                    ],
                    q_operand[
                        None, None, kphase_idx, dk_state_g_done_handle.index
                    ],
                    dk_acc[None, None, None, dk_dp_handle.index],
                )
            dk_dp_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dk_state_g_done_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            da_left_at_inputs_ready_handle = da_left_at_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            da_left_handle = da_left_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_da_left
            a_operand = tiled_mma.make_fragment_A(s_a_p6)
            dar_operand = tiled_mma.make_fragment_B(s_tmp12_p6)
            num_kphases = cute.size(a_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                _direct_mma_ws(
                    da_tmem_ptr,
                    a_operand[
                        None, None, kphase_idx, da_left_at_inputs_ready_handle.index
                    ],
                    dar_operand[
                        None, None, kphase_idx, da_left_at_inputs_ready_handle.index
                    ],
                    68256912,
                    kphase_idx != 0,
                )
            da_left_handle.commit()

            at_handle = at_mma_producer.acquire()
            tiled_mma = mma_at
            k_left_a = tiled_mma.make_fragment_A(s_k_left_p7_a)
            k_left_b = tiled_mma.make_fragment_B(s_k_left_p7_b)
            num_kphases = cute.size(k_left_a, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                _direct_mma_ws(
                    square_tmem_ptr,
                    k_left_a[
                        None, None, kphase_idx, da_left_at_inputs_ready_handle.index
                    ],
                    k_left_b[
                        None, None, kphase_idx, da_left_at_inputs_ready_handle.index
                    ],
                    68158608,
                    kphase_idx != 0,
                )
            k_right_a = tiled_mma.make_fragment_A(s_k_right_p7_a)
            k_right_b = tiled_mma.make_fragment_B(s_k_right_p7_b)
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                _direct_mma_ws(
                    square_tmem_ptr,
                    k_right_a[
                        None, None, kphase_idx, da_left_at_inputs_ready_handle.index
                    ],
                    k_right_b[
                        None, None, kphase_idx, da_left_at_inputs_ready_handle.index
                    ],
                    68158608,
                    True,
                )
            at_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            da_left_at_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            da_right_inputs_ready_handle = da_right_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            da_right_handle = da_right_mma_producer.acquire()
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_da_right
            da_operand = tiled_mma.make_fragment_A(s_tmp12_p7)
            a_operand = tiled_mma.make_fragment_B(s_a_p7)
            num_kphases = cute.size(da_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, kphase_idx != 0
                )
                _direct_mma_ws(
                    da_tmem_ptr,
                    da_operand[
                        None, None, kphase_idx, da_right_inputs_ready_handle.index
                    ],
                    a_operand[
                        None, None, kphase_idx, da_right_inputs_ready_handle.index
                    ],
                    68158608,
                    kphase_idx != 0,
                )
            da_right_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            da_right_inputs_ready_handle.release()

            self._iket_push_stage(block_idx, chunk_iteration, "MMA_WAIT_STAGE")
            dk_da_inputs_ready_handle = dk_da_inputs_mma_consumer.wait()
            self._iket_pop_stage(block_idx, chunk_iteration)
            self._iket_push_stage(block_idx, chunk_iteration, "MMA_ISSUE")
            tiled_mma = mma_dk_da
            das_operand = tiled_mma.make_fragment_A(s_tmp12_p2)
            k_operand = tiled_mma.make_fragment_B(s_k_p2)
            num_kphases = cute.size(das_operand, mode=[2])
            for kphase_idx in cutlass.range(
                num_kphases, unroll_full=True
            ):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    tiled_mma,
                    dk_acc[None, None, None, dk_da_handle.index],
                    das_operand[
                        None, None, kphase_idx, dk_da_inputs_ready_handle.index
                    ],
                    k_operand[
                        None, None, kphase_idx, dk_da_inputs_ready_handle.index
                    ],
                    dk_acc[None, None, None, dk_da_handle.index],
                )
            dk_da_handle.commit()
            self._iket_pop_stage(block_idx, chunk_iteration)
            dk_da_inputs_ready_handle.release()

    @cute.kernel
    def _fused_bwd_kernel(
        self,
        q,
        k,
        v,
        a,
        g,
        beta,
        do,
        dht,
        dht_matrix,
        dh0_matrix,
        h,
        cu_seqlens,
        chunk_offsets,
        dq,
        dk,
        dv,
        dk_matrix,
        dg,
        db,
        dh0,
        tma_q,
        tma_k,
        tma_v,
        tma_g,
        tma_h,
        tma_a,
        tma_do,
        tma_beta,
        tma_dq_store,
        scale,
        mma_tma_state,
        mma_p,
        mma_dvprime_state,
        mma_dvprime_pg,
        mma_u_state,
        mma_dv,
        mma_vprime,
        mma_dag,
        mma_dpg,
        mma_dk_state,
        mma_dq_state,
        mma_dk_state_g,
        mma_dq_dp,
        mma_dh_k,
        mma_dk_dp,
        mma_da_left,
        mma_at,
        mma_da_right,
        mma_dh_q,
        mma_dk_da,
        token_layout,
        token_transposed_layout,
        square_layout,
        square_transposed_layout,
        state_layout,
        packed_dvprime_pg_a,
        packed_dvprime_pg_b,
        packed_vprime_a,
        packed_vprime_b,
        packed_dvprime_state_a,
        packed_dvprime_state_b,
        packed_dq_state_a,
        packed_dq_state_b,
        packed_p_a,
        packed_p_b,
        packed_da_left_a,
        packed_da_left_b,
        packed_at_a,
        packed_at_b,
    ):
        """Initialize the complete reverse packed-chunk execution topology."""

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        shared_views = make_shared_views(
            storage,
            token_layout=token_layout,
            token_transposed_layout=token_transposed_layout,
            square_layout=square_layout,
            square_transposed_layout=square_transposed_layout,
            state_layout=state_layout,
            packed_dvprime_pg_a=packed_dvprime_pg_a,
            packed_dvprime_pg_b=packed_dvprime_pg_b,
            packed_vprime_a=packed_vprime_a,
            packed_vprime_b=packed_vprime_b,
            packed_dvprime_state_a=packed_dvprime_state_a,
            packed_dvprime_state_b=packed_dvprime_state_b,
            packed_dq_state_a=packed_dq_state_a,
            packed_dq_state_b=packed_dq_state_b,
            packed_p_a=packed_p_a,
            packed_p_b=packed_p_b,
            packed_da_left_a=packed_da_left_a,
            packed_da_left_b=packed_da_left_b,
            packed_at_a=packed_at_a,
            packed_at_b=packed_at_b,
            io_dtype=self.io_dtype,
        )
        canonical_views = shared_views.canonical
        s_q = canonical_views.q
        s_q_transposed = canonical_views.q_transposed
        s_k = canonical_views.k
        s_k_transposed = canonical_views.k_transposed
        s_v = canonical_views.v
        s_do = canonical_views.do
        s_do_padded = canonical_views.do_padded
        s_a = canonical_views.a
        s_h = canonical_views.h
        s_tmp11_base_ptr = canonical_views.tmp11_base_ptr
        s_tmp12 = canonical_views.tmp12
        s_dqkv = canonical_views.dqkv
        s_tmp21 = canonical_views.tmp21
        s_tmp21_left_square_transposed = (
            canonical_views.tmp21_left_square_transposed
        )
        s_tmp21_right_square_transposed = (
            canonical_views.tmp21_right_square_transposed
        )
        s_tmp22 = canonical_views.tmp22
        s_tmp22_padded = canonical_views.tmp22_padded
        s_tmp23 = canonical_views.tmp23
        s_tmp23_left_square_transposed = (
            canonical_views.tmp23_left_square_transposed
        )
        s_tmp23_right_square_transposed = (
            canonical_views.tmp23_right_square_transposed
        )
        s_tmp41 = canonical_views.tmp41
        s_tmp41_padded = canonical_views.tmp41_padded

        packed_views = shared_views.packed
        s_tmp11_p1 = packed_views.tmp11_p1
        s_tmp12_p1 = packed_views.tmp12_p1
        s_do_p1 = packed_views.do_p1
        s_tmp21_p1 = packed_views.tmp21_p1
        s_q_p1 = packed_views.q_p1
        s_tmp11_p2 = packed_views.tmp11_p2
        s_tmp12_p2 = packed_views.tmp12_p2
        s_tmp22_p2 = packed_views.tmp22_p2
        s_k_p2 = packed_views.k_p2
        s_k_p8 = packed_views.k_p8
        s_tmp41_p8 = packed_views.tmp41_p8
        s_h_p8 = packed_views.h_p8
        s_tmp21_p9 = packed_views.tmp21_p9
        s_do_p9 = packed_views.do_p9
        s_tmp23_p9 = packed_views.tmp23_p9
        s_tmp41_p9 = packed_views.tmp41_p9
        s_h_p9 = packed_views.h_p9
        s_q_p0 = packed_views.q_p0
        s_k_p0 = packed_views.k_p0
        s_tmp21_p0a = packed_views.tmp21_p0a
        s_tmp22_p0b = packed_views.tmp22_p0b
        s_do_p0 = packed_views.do_p0
        s_tmp21_p0b = packed_views.tmp21_p0b
        s_a_p6 = packed_views.a_p6
        s_tmp12_p6 = packed_views.tmp12_p6
        s_tmp12_p7 = packed_views.tmp12_p7
        s_a_p7 = packed_views.a_p7
        s_k_left_p7_a = packed_views.k_left_p7_a
        s_k_left_p7_b = packed_views.k_left_p7_b
        s_k_right_p7_a = packed_views.k_right_p7_a
        s_k_right_p7_b = packed_views.k_right_p7_b

        vector_views = shared_views.vectors
        s_g = vector_views.g
        s_exp_g = vector_views.exp_g
        s_scaled_exp_g = vector_views.scaled_exp_g
        s_rev_exp_g = vector_views.rev_exp_g
        s_beta = vector_views.beta
        s_dg = vector_views.dg
        s_db = vector_views.db
        s_dv_offset = vector_views.dv_offset
        s_reduce = vector_views.reduce
        s_reduce2 = vector_views.reduce2

        if warp_idx == self.mma_warp_id:
            cpasync.prefetch_descriptor(tma_q[0])
            cpasync.prefetch_descriptor(tma_k[0])
            cpasync.prefetch_descriptor(tma_v[0])
            cpasync.prefetch_descriptor(tma_g[0])
            cpasync.prefetch_descriptor(tma_h[0])
            cpasync.prefetch_descriptor(tma_a[0])
            cpasync.prefetch_descriptor(tma_do[0])
            cpasync.prefetch_descriptor(tma_beta[0])
        (
            load_tensor_pipelines,
            mma_done_pipelines,
            mma_input_ready_pipelines,
            sdg_ownership_order_pipeline,
            smem_reuse_pipelines,
            tmp21_reuse_pipeline,
            all_tmem_readers_done_pipeline,
            dq_smem_tile_ready_pipeline,
            aq_w_inputs_read_done_pipeline,
            dog_smem_ready_pipeline,
            dg_rmw_done_pipeline,
        ) = self._make_role_pipelines(storage)
        # Donor WGs lower their limits before the existing pipeline-init
        # barrier; WG0 raises its limit only after this barrier in its branch.
        if warp_idx > self.sk_warp_ids[-1]:
            if warp_idx <= self.aq_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(
                    self.consumer_A_warpgroup_register_limit
                )
            else:
                cute.arch.setmaxregister_decrease(
                    self.producer_warpgroup_register_limit
                )
        pipeline.pipeline_init_arrive(is_relaxed=True)
        pipeline.pipeline_init_wait()

        # Preserve the verified 256+128+32+64 allocation. U and dQ reuse
        # primary [192, 256) at disjoint phases; Vprime and dO-gamma reuse the
        # independent [416, 480) block at disjoint phases.
        tmem_holding_ptr = storage.tmem_holding_buf.data_ptr()
        tmem_views = make_tmem_views(
            self._allocate_tmem(tmem_holding_ptr, warp_idx),
            mma_dvprime_pg,
            mma_dh_k,
        )
        primary_tmem_ptr = tmem_views.primary_ptr
        secondary_tmem_ptr = tmem_views.secondary_ptr
        mask_tmem_ptr = tmem_views.mask_ptr
        vprime_tmem_ptr = tmem_views.vprime_ptr
        dog_tmem_ptr = tmem_views.dog_ptr
        dq_tmem_ptr = tmem_views.dq_ptr
        da_tmem_ptr = tmem_views.da_ptr
        square_tmem_ptr = tmem_views.square_ptr
        u_acc = tmem_views.u_acc
        vprime_acc = tmem_views.vprime_acc
        dog_acc = tmem_views.dog_acc
        dq_acc = tmem_views.dq_acc
        dk_acc = tmem_views.dk_acc
        dv_acc = tmem_views.dv_acc
        dh_left_acc = tmem_views.dh_left_acc
        dh_right_acc = tmem_views.dh_right_acc
        p_copy_acc = tmem_views.p_copy_acc
        dp_copy_acc = tmem_views.dp_copy_acc
        a_copy_acc = tmem_views.a_copy_acc
        da_copy_acc = tmem_views.da_copy_acc
        mask_acc = tmem_views.mask_acc
        block_idx, _, _ = cute.arch.block_idx()
        sequence = block_idx // self.heads
        head = block_idx % self.heads
        if cutlass.const_expr(self.uniform_sequence_length != 0):
            sequence_length = cutlass.Int32(self.uniform_sequence_length)
            sequence_chunks = cute.ceil_div(sequence_length, self.bt)
            sequence_start = sequence * self.uniform_sequence_length
            chunk_base = sequence * sequence_chunks
        else:
            sequence_start = cu_seqlens[sequence]
            sequence_length = cu_seqlens[sequence + 1] - sequence_start
            chunk_base = chunk_offsets[sequence]
            sequence_chunks = chunk_offsets[sequence + 1] - chunk_base
        tail_tokens = sequence_length - (sequence_chunks - 1) * self.bt

        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_push("WG2_TMA")
            self.run_tma_warp(
                block_idx,
                chunk_base,
                head,
                load_tensor_pipelines,
                mma_done_pipelines,
                mma_tma_state,
                mma_at,
                mma_p,
                smem_reuse_pipelines,
                s_a,
                s_beta,
                s_do,
                s_g,
                s_h,
                s_k,
                s_q,
                s_v,
                sequence_chunks,
                sequence_start,
                mma_input_ready_pipelines,
                tidx,
                tma_a,
                tma_beta,
                tma_do,
                tma_g,
                tma_h,
                tma_k,
                tma_q,
                tma_v,
            )
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_pop()
        elif warp_idx == self.idle_producer_warp_id:
            pass
        elif warp_idx == self.dq_store_warp_id:
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_push("WG2_STORE")
            self.run_dq_store_warp(
                block_idx,
                dq,
                head,
                s_dqkv,
                sequence_chunks,
                sequence_start,
                tail_tokens,
                dq_smem_tile_ready_pipeline,
                tidx,
                tma_dq_store,
            )
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_pop()
        elif warp_idx <= self.sk_warp_ids[-1]:
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_push("WG0_SK")
            self.run_sk_consumer(
                block_idx,
                dh0_matrix,
                dh_left_acc,
                dh_right_acc,
                dht_matrix,
                dk_acc,
                dog_acc,
                dk_matrix,
                dv,
                dv_acc,
                head,
                load_tensor_pipelines,
                mma_done_pipelines,
                smem_reuse_pipelines,
                tmp21_reuse_pipeline,
                dg_rmw_done_pipeline,
                all_tmem_readers_done_pipeline,
                dog_smem_ready_pipeline,
                s_db,
                aq_w_inputs_read_done_pipeline,
                s_dg,
                s_dv_offset,
                s_do,
                s_exp_g,
                s_g,
                s_h,
                s_k,
                s_rev_exp_g,
                s_scaled_exp_g,
                s_tmp21,
                s_tmp23,
                s_tmp41,
                s_v,
                scale,
                sequence,
                sequence_chunks,
                sequence_start,
                tail_tokens,
                mma_input_ready_pipelines,
                sdg_ownership_order_pipeline,
                tidx,
                u_acc,
                warp_idx,
            )
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_pop()
        elif warp_idx <= self.aq_warp_ids[-1]:
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_push("WG1_AQ")
            self.run_aq_consumer(
                block_idx,
                a_copy_acc,
                da_copy_acc,
                db,
                dg,
                dp_copy_acc,
                dq_acc,
                dv_acc,
                head,
                load_tensor_pipelines,
                mask_acc,
                mask_tmem_ptr,
                mma_done_pipelines,
                smem_reuse_pipelines,
                tmp21_reuse_pipeline,
                p_copy_acc,
                primary_tmem_ptr,
                da_tmem_ptr,
                dg_rmw_done_pipeline,
                all_tmem_readers_done_pipeline,
                dog_smem_ready_pipeline,
                s_a,
                aq_w_inputs_read_done_pipeline,
                s_beta,
                s_db,
                s_dg,
                s_do_padded,
                s_dqkv,
                s_exp_g,
                s_g,
                s_q,
                s_reduce,
                s_reduce2,
                s_scaled_exp_g,
                s_tmp11_base_ptr,
                s_tmp12,
                s_tmp21,
                s_tmp22,
                s_tmp22_padded,
                s_tmp41_padded,
                s_v,
                scale,
                sequence_chunks,
                sequence_start,
                tail_tokens,
                square_layout,
                square_tmem_ptr,
                mma_input_ready_pipelines,
                dq_smem_tile_ready_pipeline,
                sdg_ownership_order_pipeline,
                tidx,
                u_acc,
                vprime_acc,
            )
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_pop()
        elif warp_idx == self.mma_warp_id:
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_push("WG2_MMA")
            self.run_mma_warp(
                block_idx,
                dh_left_acc,
                dh_right_acc,
                dk_acc,
                dq_acc,
                dv_acc,
                load_tensor_pipelines,
                mma_done_pipelines,
                mma_p,
                mma_dvprime_state,
                mma_dvprime_pg,
                mma_u_state,
                mma_dv,
                mma_vprime,
                mma_dag,
                mma_dpg,
                mma_dk_state,
                mma_dq_state,
                mma_dk_state_g,
                mma_dq_dp,
                mma_dh_k,
                mma_dk_dp,
                mma_da_left,
                mma_at,
                mma_da_right,
                mma_dh_q,
                mma_dk_da,
                primary_tmem_ptr,
                da_tmem_ptr,
                s_do,
                s_k,
                s_q,
                s_a_p6,
                s_a_p7,
                s_do_p0,
                s_do_p1,
                s_do_p9,
                s_h_p8,
                s_h_p9,
                s_k_left_p7_a,
                s_k_left_p7_b,
                s_k_p0,
                s_k_p2,
                s_k_p8,
                s_k_right_p7_a,
                s_k_right_p7_b,
                s_k_transposed,
                s_q_p0,
                s_q_p1,
                s_q_transposed,
                s_tmp11_p1,
                s_tmp11_p2,
                s_tmp12_p1,
                s_tmp12_p2,
                s_tmp12_p6,
                s_tmp12_p7,
                s_tmp21_left_square_transposed,
                s_tmp21_p0a,
                s_tmp21_p0b,
                s_tmp21_p1,
                s_tmp21_p9,
                s_tmp21_right_square_transposed,
                s_tmp22_p0b,
                s_tmp22_p2,
                s_tmp23_left_square_transposed,
                s_tmp23_p9,
                s_tmp23_right_square_transposed,
                s_tmp41_p8,
                s_tmp41_p9,
                sequence_chunks,
                tail_tokens,
                square_tmem_ptr,
                mma_input_ready_pipelines,
                all_tmem_readers_done_pipeline,
                u_acc,
                vprime_acc,
            )
            if cutlass.const_expr(self.enable_iket):
                if block_idx == 0:
                    _cute_iket.range_pop()
        # The final role-specific drain/store must complete before TMEM free.
        cute.arch.sync_threads()
        self._release_tmem(tmem_holding_ptr, warp_idx)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        g: cute.Tensor,
        beta: cute.Tensor,
        do: cute.Tensor,
        dht: cute.Tensor,
        h: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        dg: cute.Tensor,
        db: cute.Tensor,
        dh0: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self._check_static_tensor_contracts(
            q,
            k,
            v,
            a,
            g,
            beta,
            do,
            dht,
            h,
            cu_seqlens,
            chunk_offsets,
            dq,
            dk,
            dv,
            dg,
            db,
            dh0,
        )
        dht_matrix = self._state_matrix_view(dht)
        dh0_matrix = self._state_matrix_view(dh0)
        dk_matrix = self._token_matrix_view(dk)
        plan = self._build_mma_layouts()
        tma = self._build_tma_atoms(
            plan,
            q,
            k,
            v,
            a,
            g,
            beta,
            do,
            dht,
            h,
            dq,
        )
        self._fused_bwd_kernel(
            q,
            k,
            v,
            a,
            g,
            beta,
            do,
            dht,
            dht_matrix,
            dh0_matrix,
            h,
            cu_seqlens,
            chunk_offsets,
            dq,
            dk,
            dv,
            dk_matrix,
            dg,
            db,
            dh0,
            tma.q,
            tma.k,
            tma.v,
            tma.g,
            tma.h,
            tma.a,
            tma.do,
            tma.beta,
            tma.dq_store,
            scale,
            plan.variants.mm_128_64_128_k_k.tiled_mma,
            plan.operations.p.tiled_mma,
            plan.operations.dvprime_state.tiled_mma,
            plan.operations.dvprime_pg.tiled_mma,
            plan.operations.u_state.tiled_mma,
            plan.operations.dv.tiled_mma,
            plan.operations.vprime.tiled_mma,
            plan.operations.dag.tiled_mma,
            plan.operations.dpg.tiled_mma,
            plan.operations.dk_state.tiled_mma,
            plan.operations.dq_state.tiled_mma,
            plan.operations.dk_state_g.tiled_mma,
            plan.operations.dq_dp.tiled_mma,
            plan.operations.dh_k.tiled_mma,
            plan.operations.dk_dp.tiled_mma,
            plan.operations.da_left.tiled_mma,
            plan.operations.at.tiled_mma,
            plan.operations.da_right.tiled_mma,
            plan.operations.dh_q.tiled_mma,
            plan.operations.dk_da.tiled_mma,
            plan.canonical_staged.token_direct,
            plan.canonical_staged.token_transposed,
            plan.canonical_staged.square_direct,
            plan.canonical_staged.square_transposed,
            plan.canonical_staged.state_direct,
            plan.operations.dvprime_pg.operands.a_staged,
            plan.operations.dvprime_pg.operands.b_staged,
            plan.operations.vprime.operands.a_staged,
            plan.operations.vprime.operands.b_staged,
            plan.operations.dvprime_state.operands.a_staged,
            plan.operations.dvprime_state.operands.b_staged,
            plan.operations.dq_state.operands.a_staged,
            plan.operations.dq_state.operands.b_staged,
            plan.operations.p.operands.a_staged,
            plan.operations.p.operands.b_staged,
            plan.operations.da_left.operands.a_staged,
            plan.operations.da_left.operands.b_staged,
            plan.operations.at.operands.a_staged,
            plan.operations.at.operands.b_staged,
        ).launch(
            grid=(self.num_sequences * self.heads, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )


__all__ = [
    "FusedGdrBwdKernel",
    "LayoutBudget",
    "SharedStorage",
    "get_layout_budget",
]
