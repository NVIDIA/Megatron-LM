# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Core attention only; model scheduling and non-attention costs live elsewhere."""
from .api import AttentionEstimate, AttentionPredictor, GEOMETRIES, Pack

__all__ = ["AttentionEstimate", "AttentionPredictor", "GEOMETRIES", "Pack"]
__version__ = "0.1.0.dev20261109"
