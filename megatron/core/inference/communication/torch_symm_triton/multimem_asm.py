# Copyright (c) Meta Platforms, Inc. and affiliates.
# pylint: disable=line-too-long

# Adapted from https://github.com/yifuwang/symm-mem-recipes.git


from unittest.mock import MagicMock

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = MagicMock()
    tl = MagicMock()
    triton.jit = null_decorator


@triton.jit
def ld_128(ptr, mask, multicast_op: tl.constexpr, reduce_f32: tl.constexpr = False):
    """
    Loads 128 bits from memory into registers.

    This function abstracts two distinct hardware behaviors based on `multicast_op`:

    1.  **Standard Load (`multicast_op=False`)**:
        -   **Semantics:** Local Global Memory Load.
        -   **Action:** Reads 128 bits from `ptr` in global memory into the local register file.

    2.  **Multicast Reduce-Load (`multicast_op=True`)**:
        -   **Semantics:** "Pull" Reduction over NVLink.
        -   **Action:** Simultaneously reads 128 bits from the *same* address across all peer GPUs
            in the multicast group, sums them, and loads the result into the local register file.
        -   **Hardware:** Uses `multimem.ld_reduce` (Hopper+).
        -   When `reduce_f32=False` (default): bf16x2 addition with f32 accumulation
            (128 bits = 8 x bf16, 2 per register).
        -   When `reduce_f32=True`: native f32 addition
            (128 bits = 4 x fp32, 1 per register).

    Args:
        ptr: Memory pointer to the source buffer.
        mask: Boolean predicate. If False, the operation is skipped (no-op).
        multicast_op (tl.constexpr): Toggles between standard load (False)
            and multicast-reduce (True).
        reduce_f32 (tl.constexpr): When True and multicast_op=True, uses f32 reduction
            instead of bf16x2 reduction. Default False.

    Returns:
        Four 32-bit registers (tl.uint32), representing 128 bits of loaded data.
    """
    if multicast_op:
        if reduce_f32:
            # fp32 reduction: multimem.ld_reduce.add.v4.f32
            # Each 128-bit load reduces 4 x fp32 values across peers.
            return tl.inline_asm_elementwise(
                """
                {
                    .reg .pred %p0;
                    setp.ne.s32 %p0, $5, 1;
                    @%p0 bra end;
                    multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {$0, $1, $2, $3}, [$4];
                    end:
                }
                """,
                "=r,=r,=r,=r,l,r",
                args=[ptr, mask.to(tl.int32)],
                dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
                is_pure=True,
                pack=1,
            )
        else:
            # bf16x2 reduction with f32 accumulation: multimem.ld_reduce.add.acc::f32.v4.bf16x2
            # Each 128-bit load reduces 8 x bf16 values (packed as 4 x bf16x2) across peers.
            return tl.inline_asm_elementwise(
                """
                {
                    .reg .pred %p0;
                    setp.ne.s32 %p0, $5, 1;
                    @%p0 bra end;
                    multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4]; 
                    end:
                }
                """,
                "=r,=r,=r,=r,l,r",
                args=[ptr, mask.to(tl.int32)],
                dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
                is_pure=True,
                pack=1,
            )
    else:
        return tl.inline_asm_elementwise(
            """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $5, 1;
            @%p0 bra end;
            ld.global.v4.u32 {$0, $1, $2, $3}, [$4];
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
def st_128(ptr, x, y, z, w, mask, multicast_op):
    """
    Stores 128 bits (8 x bf16) from registers to memory.

    This function abstracts two distinct hardware behaviors based on `multicast_op`:

    1.  **Standard Store (`multicast_op=False`)**:
        -   **Semantics:** Local Global Memory Store.
        -   **Action:** Writes 128 bits from local registers to `ptr` in global memory.

    2.  **Multicast Store (`multicast_op=True`)**:
        -   **Semantics:** "Push" Broadcast over NVLink.
        -   **Action:** Writes 128 bits from local registers to the `ptr` address in
            the global memory of **all** peer GPUs in the multicast group simultaneously.
        -   **Hardware:** Uses `multimem.st` (Hopper+).
        -   **Use Case:** The "Broadcast" or "All-Gather" step in collective operations.

    Args:
        ptr: Memory pointer to the destination buffer.
        x, y, z, w: Four 32-bit registers containing the data to store.
        mask: Boolean predicate. If False, the store is skipped.
        multicast_op (tl.constexpr): Toggles between standard store (False)
        and multicast broadcast (True).
    """
    # PTX Assembly Logic:
    # 1. @$6: Predication. Only execute if argument 6 (mask) is True.
    # 2. Opcode Selection:
    #    - 'multimem.st...v4.f32': Broadcasts data to all peers.
    #      (Note: .f32 type used for bit-movement, equivalent to .u32 for storage).
    #    - 'st.global...v4.u32': Standard 128-bit memory write.
    # 3. Operands:
    #    - [$1]: Destination memory address.
    #    - {$2, $3, $4, $5}: Source registers containing data.
    if multicast_op:
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.ne.s32 %p0, $6, 1;
                @%p0 bra end;
                multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
                end:
            }
            """,
            "=r,l,r,r,r,r,r",
            args=[ptr, x, y, z, w, mask.to(tl.int32)],
            dtype=(tl.uint32),
            is_pure=False,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $6, 1;
            @%p0 bra end;
            st.global.v4.f32 [$1], {$2, $3, $4, $5};
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
    a0,
    a1,
    a2,
    a3,  # First vector of 8 bf16s, packed in 4 uint32s
    b0,
    b1,
    b2,
    b3,  # Second vector of 8 bf16s, packed in 4 uint32s
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
