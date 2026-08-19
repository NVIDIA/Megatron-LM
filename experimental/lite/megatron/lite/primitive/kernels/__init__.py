# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional kernel shims used by MLite primitives."""

from __future__ import annotations

from .vllm_ds4 import (
    DS4KVInsertAdapter,
    FlashMLAAdapter,
    FusedQKVRMSNormAdapter,
    KVCacheLayout,
    MHCKernel,
    MHCTileLangAdapter,
    OProjectionAdapter,
)

__all__ = [
    "DS4KVInsertAdapter",
    "FlashMLAAdapter",
    "FusedQKVRMSNormAdapter",
    "KVCacheLayout",
    "MHCKernel",
    "MHCTileLangAdapter",
    "OProjectionAdapter",
]
