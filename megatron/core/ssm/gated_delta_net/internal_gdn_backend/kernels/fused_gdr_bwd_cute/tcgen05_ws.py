"""Low-level SM100 warp-specialized data-movement helpers.

The inline-PTX patterns are adapted from TileLang's CuTeDSL backend. TileLang
is distributed under the MIT license; its complete notice is retained in
``LICENSE.tile-ai`` next to this module.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


def _ir(value, loc=None, ip=None):
    return value.ir_value(loc=loc, ip=ip)


@cute.jit
def tmem_allocate(tmem_buffer_ptr: cute.Pointer, num_columns: int):
    """Allocate one physical TMEM block and store its address in shared memory."""

    @dsl_user_op
    def _do_allocate(dst, columns, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [_ir(dst, loc, ip), _ir(columns, loc, ip)],
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], $1;",
            "r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do_allocate(
        cutlass.Int32(tmem_buffer_ptr.toint()), cutlass.Int32(num_columns)
    )


@cute.jit
def tmem_deallocate(tmem_buffer_ptr: cute.Pointer, num_columns: int):
    """Deallocate the TMEM block whose address is stored in shared memory."""

    tmem_address = cute.make_tensor(tmem_buffer_ptr, (1,))[0]

    @dsl_user_op
    def _do_deallocate(address, columns, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [_ir(address, loc, ip), _ir(columns, loc, ip)],
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;",
            "r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do_deallocate(cutlass.Int32(tmem_address), cutlass.Int32(num_columns))


@cute.jit
def load_f32x2(shared_address, dst_ptr):
    """Load two contiguous FP32 values with one 64-bit shared instruction."""

    i32_type = ir.IntegerType.get_signless(32)
    result_type = llvm.StructType.get_literal([i32_type, i32_type])
    result = llvm.inline_asm(
        result_type,
        [cutlass.Int32(shared_address).ir_value()],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    packed = cute.make_rmem_tensor((2,), cutlass.Int32)
    for index in cutlass.range_constexpr(2):
        packed[index] = cutlass.Int32(
            llvm.extractvalue(i32_type, result, [index])
        )
    values = cute.make_tensor(
        cute.recast_ptr(packed.iterator, dtype=cutlass.Float32), (2,)
    )
    dst = cute.make_tensor(dst_ptr, (2,))
    for index in cutlass.range_constexpr(2):
        dst[index] = values[index]


@cute.jit
def copy_bf16x8_s2g(shared_address, global_address):
    """Copy one aligned 128-bit BF16 vector from shared to global memory."""

    llvm.inline_asm(
        None,
        [
            cutlass.Int32(shared_address).ir_value(),
            cutlass.Int64(global_address).ir_value(),
        ],
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
def copy_bf16x8_s2g_offset(
    shared_base,
    global_base,
    shared_offset: int,
    global_offset: int,
):
    """Copy one BF16x8 vector using compile-time address displacements."""

    llvm.inline_asm(
        None,
        [
            cutlass.Int32(shared_base).ir_value(),
            cutlass.Int64(global_base).ir_value(),
        ],
        f"""{{
.reg .b32 p0, p1, p2, p3;
ld.shared.v4.b32 {{p0, p1, p2, p3}}, [$0+{shared_offset}];
st.global.v4.b32 [$1+{global_offset}], {{p0, p1, p2, p3}};
}}""",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
