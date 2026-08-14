"""Strict public wrapper for the SM100 fused GDR backward CuTeDSL kernel.

Copyright (c) 2026 The Qwen team, Alibaba Group.
Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import annotations

from .fused_bwd import fused_gdr_bwd

__all__ = ["fused_gdr_bwd"]
