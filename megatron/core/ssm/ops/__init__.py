# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated import path; use ``megatron.core.ops.ssm``.

The historical package-level names resolve from the modules that define them.
The optional kernel entry points keep their historical value of ``None`` when their
kernels cannot be imported.
"""

from megatron.core.ops._compat import deprecated_module

_forward, __dir__ = deprecated_module(
    __name__,
    "megatron.core.ops.ssm",
    exports={
        "mamba_chunk_scan_combined_varlen": "megatron.core.ops.ssm.mamba2.ssd_combined",
        "causal_conv1d_varlen_fn": "megatron.core.ops.ssm.common.causal_conv1d_varlen",
    },
)

_OPTIONAL = ("mamba_chunk_scan_combined_varlen", "causal_conv1d_varlen_fn")


def __getattr__(name: str) -> object:
    """Resolve ``name`` like any forwarded name, keeping the optional ``None`` fallback."""
    try:
        return _forward(name)
    except ImportError:
        if name in _OPTIONAL:
            return None
        raise
