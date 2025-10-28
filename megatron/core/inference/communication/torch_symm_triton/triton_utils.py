# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def get_tid():
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
    return (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )


@triton.jit
def sync_threads():
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )

triton_kernels = {}

def log_triton_kernel(kernel):
    import atexit
    import tempfile
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    def on_exit():
        print("PTX files:")
        for kernel in triton_kernels:
            f = tempfile.NamedTemporaryFile(dir="/tmp", delete=False)
            f.write(kernel.asm["ptx"].encode("utf-8"))
            print(f"+- {kernel.name}: {f.name}")

    if len(triton_kernels) == 0:
        atexit.register(on_exit)

    if kernel not in triton_kernels:
        triton_kernels[kernel] = None