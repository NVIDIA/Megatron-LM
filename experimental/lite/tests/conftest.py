# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pytest configuration for Megatron Lite tests."""

from __future__ import annotations

import sys
from pathlib import Path


LITE_ROOT = Path(__file__).resolve().parents[1]
if str(LITE_ROOT) not in sys.path:
    sys.path.insert(0, str(LITE_ROOT))
