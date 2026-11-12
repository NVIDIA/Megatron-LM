# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Programmatic dependent launch (PDL) for Megatron's inference Triton kernels.

On NVIDIA GPUs of compute capability 9.0 or newer, with a Triton that has the
grid-dependency intrinsics, the kernels that opt in are launched with programmatic
stream serialization. Each such kernel executes griddepcontrol.wait before it touches
memory, which blocks until the previous kernel on the stream has completed and its
writes are visible. The kernels do not trigger their dependents early: a following PDL
kernel is launched as this kernel's blocks exit, so the two never share SMs. Inside a
CUDA graph this lets the launch of the next kernel overlap the end of the previous one.

Set MCORE_INFERENCE_TRITON_PDL=0 to launch the kernels without PDL (for A/B comparison);
USE_PDL=False removes the wait instruction at compile time.
"""

import os

import torch

try:
    import triton

    try:
        from triton.language.extra.cuda import gdc_wait

        HAVE_GDC = True
    except ImportError:  # Triton without the grid-dependency intrinsics
        HAVE_GDC = False

        @triton.jit
        def gdc_wait():  # pragma: no cover
            """No-op stand-in; never compiled because USE_PDL stays False."""
            pass

except ImportError:  # no Triton at all: the kernels importing this are not built either
    HAVE_GDC = False
    gdc_wait = None

__all__ = ["HAVE_GDC", "gdc_wait", "pdl_enabled", "pdl_launch_kwargs"]

_enabled = None


def pdl_enabled() -> bool:
    """True when inference Triton kernels should launch with PDL."""
    global _enabled
    if _enabled is None:
        _enabled = (
            HAVE_GDC
            and os.environ.get("MCORE_INFERENCE_TRITON_PDL", "1") != "0"
            and torch.cuda.is_available()
            and torch.version.hip is None
            and torch.cuda.get_device_capability()[0] >= 9
        )
    return _enabled


def pdl_launch_kwargs() -> dict:
    """Launch options for a kernel compiled with USE_PDL=pdl_enabled()."""
    return {"launch_pdl": True} if pdl_enabled() else {}
