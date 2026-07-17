# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint-format quantization primitives."""

from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

__all__ = ["quantize_block_fp8", "quantize_mxfp4"]
