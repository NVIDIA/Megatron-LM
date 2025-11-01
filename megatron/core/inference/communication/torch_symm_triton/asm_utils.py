# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import triton
import triton.language as tl 

@triton.jit
def multimem_ld_reduce_128(multicast_ptrs, mask):
    """
    Multicast load and reduce 128 bits (4 x bf16) from all peers over nvlink
    Outputs are returned as 4 tl.uint32 registers, each containing 2 bf16 values 
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[multicast_ptrs, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def multimem_st_128(multicast_ptrs, x, y, z, w, mask):
    """
    Multicast store 128 bits (4 x bf16) to all peers over nvlink
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[multicast_ptrs, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )

@triton.jit
def ld_128(ptr, mask):
    """
    Load 128 bits (4 x bf16) from ptr
    Outputs are returned as 4 tl.uint32 registers, each containing 2 bf16 values
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            ld.global.relaxed.sys.v4.u32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def st_128(ptr, x, y, z, w, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            st.global.relaxed.sys.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[ptr, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )

@triton.jit
def add_v8_bf16_from_u32(
    a0, a1, a2, a3,  # First vector of 8 bf16s, packed in 4 uint32s
    b0, b1, b2, b3,  # Second vector of 8 bf16s, packed in 4 uint32s
):
    """
    Adds two vectors of 8 bfloat16 numbers.
    Each vector is passed as four tl.uint32 tensors.
    Returns the result as a tuple of four tl.uint32 tensors.
    """
    return tl.inline_asm_elementwise(
        """
        {
            add.bf16x2 $0, $4, $8;
            add.bf16x2 $1, $5, $9;
            add.bf16x2 $2, $6, $10;
            add.bf16x2 $3, $7, $11;
        }
        """,
        # 8 outputs (=r), 8 inputs (r)
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r",
        args=[a0, a1, a2, a3, b0, b1, b2, b3],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )

@triton.jit 
def asm_rsqrt(x, eps):
    """
    Computes the reciprocal square root of a float32 number using inline assembly.
    """
    return tl.inline_asm_elementwise(
        """
        {
            add.f32 $1, $1, $2;
            rsqrt.approx.f32 $0, $1;
        }
        """,
        "=f, f, f",
        args=[x, eps],
        dtype=(tl.float32),
        is_pure=True,
        pack=1,
    )

