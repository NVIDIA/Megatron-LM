# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer moe_ep mega-kernel integration for inference-optimized MoE."""

from ._deps import require_flashinfer_moe_ep
from .adapter import MegatronMegaMoEAdapter
from .registry import MegaPrecisionSpec, build_megakernel_config, get_mega_precision_spec

__all__ = [
    "MegatronMegaMoEAdapter",
    "MegaPrecisionSpec",
    "build_megakernel_config",
    "get_mega_precision_spec",
    "require_flashinfer_moe_ep",
]
