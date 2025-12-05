# Copyright (c) Meta Platforms, Inc. and affiliates.

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
def ld_128(ptr, mask, multicast_op: tl.constexpr):
    """
    Loads 128 bits (8 x bf16) from memory into registers.

    This function abstracts two distinct hardware behaviors based on `multicast_op`:

    1.  **Standard Load (`multicast_op=False`)**:
        -   **Semantics:** Local Global Memory Load.
        -   **Action:** Reads 128 bits from `ptr` in global memory into the local register file.
        -   **Use Case:** Standard tensor processing.

    2.  **Multicast Reduce-Load (`multicast_op=True`)**:
        -   **Semantics:** "Pull" Reduction over NVLink.
        -   **Action:** Simultaneously reads 128 bits from the *same* address across all peer GPUs
            in the multicast group, sums them (add reduction), and loads the result into the
            local register file.
        -   **Hardware:** Uses `multimem.ld_reduce` (Hopper+).
        -   **Use Case:** The "Reduce" step in collective operations.

    Args:
        ptr: Memory pointer to the source buffer.
        mask: Boolean predicate. If False, the operation is skipped (no-op).
        multicast_op (tl.constexpr): Toggles between standard load (False)
        and multicast-reduce (True).

    Returns:
        Four 32-bit registers (tl.uint32), representing 128 bits of loaded data.
        Note: When interpreting as bf16, this equates to 8 values (2 per register).
    """
    # PTX Assembly Logic:
    # 1. @$5: Predication. Only execute if argument 5 (mask) is True (1).
    # 2. Opcode Selection:
    #    - 'multimem.ld_reduce...add.v4.bf16x2': Hardware-accelerated reduction across peers.
    #    - 'ld.global...v4.u32': Standard 128-bit memory read.
    # 3. Operands:
    #    - {$0, $1, $2, $3}: Destination registers (Output).
    #    - [$4]: Source memory address (Input).
    if multicast_op:
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
