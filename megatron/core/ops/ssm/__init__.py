# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Local SSM kernels, separate from mixers, recurrent caches and CP communication.

``common`` holds causal convolution, state gather/scatter and Triton determinism
helpers; ``mamba2`` holds SSD kernels; ``gdp`` holds the forward-only GDP fork;
``gated_delta`` holds GDN/GDN2 references and optional FLA targets. Phase-specific
contracts live with these subpackages. No kernel creates a process group.

The two historical exports are loaded on demand, so importing this namespace
does not import Triton, FLA, causal-conv1d or mamba-ssm.
"""

from importlib import import_module

__all__ = ["mamba_chunk_scan_combined_varlen", "causal_conv1d_varlen_fn"]


def __getattr__(name: str):
    """Load a historical entry point without eagerly importing every kernel family."""
    if name == "mamba_chunk_scan_combined_varlen":
        module_name = ".mamba2.ssd_combined"
    elif name == "causal_conv1d_varlen_fn":
        module_name = ".common.causal_conv1d_varlen"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(import_module(module_name, __name__), name)
    except ImportError:
        return None
