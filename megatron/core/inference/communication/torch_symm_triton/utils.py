# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from: https://github.com/meta-pytorch/kraken.git

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
def get_tid():
    """
    Returns the thread IDs in x, y, z dimensions.
    """
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_ntid():
    """
    Returns the number of threads in x, y, z dimensions.
    """
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_flat_tid():
    """
    Calculates a unique, one-dimensional ID for each thread within its thread block.
    """
    tid_x, tid_y, tid_z = get_tid()
    ntid_x, ntid_y, _ = get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def get_flat_bid():
    """
    Calculates a unique, one-dimensional ID for each block within the grid."""
    return (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )


@triton.jit
def sync_threads():
    """
    Synchronize all threads within a block.
    """
    tl.inline_asm_elementwise("bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
