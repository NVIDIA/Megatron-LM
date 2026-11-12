# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""An interpretable baseline family, not an accuracy-approved frozen model."""

from .geometry import GEOMETRIES

def tensor_bytes_per_token(geometry, phase):
    if tuple(geometry) not in GEOMETRIES or phase not in ("forward", "backward"):
        raise ValueError("unsupported geometry or phase")
    hq, hkv, dim = geometry
    # BF16 Q,K,V,O in F; Q,K,V,O,dO,dQ,dK,dV in B. FP32 LSE.
    return 2 * dim * ((2 * hq + 2 * hkv)
        if phase == "forward" else (4 * hq + 4 * hkv)) + 4 * hq

def features(pack, geometry, phase):
    """Work, compulsory tensor traffic, padding extent, and boundary metadata.

    Tensor traffic is a lower-bound work descriptor, not a measurement of all
    HBM traffic. A padding coefficient may be zero if the kernel skips it.
    No sequence is enlarged when counting attention interactions.
    """
    if tuple(geometry) not in GEOMETRIES or phase not in ("forward", "backward"):
        raise ValueError("unsupported geometry or phase")
    hq, hkv, dim = geometry
    tokens = sum(pack.lengths)
    pairs = sum(length * (length + 1) // 2 for length in pack.lengths)
    # QK/PV forward; backward recomputes QK then forms dV,dP,dQ,dK.
    flops = (4 if phase == "forward" else 10) * hq * dim * pairs
    bytes_per_token = tensor_bytes_per_token(geometry, phase)
    # Q/K/V cumulative logical and physical bounds are shared in self-attn.
    metadata_bytes = 2 * 4 * len(pack.cu_seqlens)
    return (1.0, flops / 1e9, tokens * bytes_per_token / 1e6,
            (pack.storage_tokens - tokens) * bytes_per_token / 1e6,
            metadata_bytes / 1e3)
