"""Strict public wrapper for the SM100 fused GDR backward CuTeDSL kernel.

Copyright (c) 2026 The Qwen team, Alibaba Group.
Licensed under the MIT License.
"""

from __future__ import annotations

from .fused_bwd import fused_gdr_bwd

__all__ = ["fused_gdr_bwd"]
