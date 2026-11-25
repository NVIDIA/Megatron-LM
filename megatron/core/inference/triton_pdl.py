# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Programmatic dependent launch (PDL) for Megatron's inference Triton kernels.

A kernel that opts in takes a ``USE_PDL`` constexpr, calls ``gdc_wait()`` (griddepcontrol.wait)
as its first statement and is launched with ``**pdl_kwargs()``. gdc_wait must stay before any
memory access: it blocks until the previous kernel on the stream has completed and its writes
are visible. Inside a CUDA graph the launch of the next kernel then overlaps the end of the
previous one.

PDL applies only under torch.inference_mode(), so training is unaffected, on compute capability
9.0 or newer with a Triton that has gdc_wait and a launcher that accepts launch_pdl. Set
MCORE_INFERENCE_TRITON_PDL=0 to turn it off.
"""

import os

import torch

try:
    import triton

    try:
        from triton.backends.nvidia.compiler import CUDAOptions
        from triton.language.extra.cuda import gdc_wait

        HAVE_PDL = "launch_pdl" in CUDAOptions.__dataclass_fields__
    except ImportError:
        HAVE_PDL = False

    if not HAVE_PDL:

        @triton.jit
        def gdc_wait():  # pragma: no cover
            """No-op stand-in; never compiled because USE_PDL stays False."""
            pass

except ImportError:  # no Triton at all: the kernels importing this are not built either
    HAVE_PDL = False
    gdc_wait = None

__all__ = ["HAVE_PDL", "gdc_wait", "pdl_kwargs"]

# Process-wide (Triton, environment) conditions, resolved on first use.
_enabled = None


def pdl_kwargs() -> dict:
    """Launch arguments for a kernel with a ``USE_PDL`` constexpr: PDL under
    torch.inference_mode() when supported, otherwise none (``USE_PDL`` defaults to False)."""
    global _enabled
    if not torch.is_inference_mode_enabled():
        return {}
    if _enabled is None:
        _enabled = (
            HAVE_PDL
            and os.environ.get("MCORE_INFERENCE_TRITON_PDL", "1") != "0"
            and torch.cuda.is_available()
            and torch.version.hip is None
        )
    # The capability is checked per call: a process may drive GPUs of different generations.
    if _enabled and torch.cuda.get_device_capability()[0] >= 9:
        return {"USE_PDL": True, "launch_pdl": True}
    return {}
