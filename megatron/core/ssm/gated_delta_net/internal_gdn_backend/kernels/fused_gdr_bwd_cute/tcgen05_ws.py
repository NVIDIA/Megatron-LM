# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Low-level SM100 warp-specialized data-movement helpers.

The inline-PTX patterns are adapted from TileLang's CuTeDSL backend. TileLang
is distributed under the MIT License; attribution is recorded in ``kernel.py``.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


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

    _do_allocate(cutlass.Int32(tmem_buffer_ptr.toint()), cutlass.Int32(num_columns))


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
def copy_bf16x8_s2g_offset(shared_base, global_base, shared_offset: int, global_offset: int):
    """Copy one BF16x8 vector using compile-time address displacements."""

    llvm.inline_asm(
        None,
        [cutlass.Int32(shared_base).ir_value(), cutlass.Int64(global_base).ir_value()],
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
