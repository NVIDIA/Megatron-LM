# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Layer layouts only. Hardware and measured non-attention inputs are external."""
from .api import ModelConfig


def qwen3_30b(*, world_size, **parallelism):
    return ModelConfig(("gqa",) * 48, world_size=world_size, **parallelism)


def qwen35_35b(*, world_size, **parallelism):
    """The full 40-decoder-layer Qwen3.5-35B model and its FA MTP layer."""
    return ModelConfig(("gdn", "gdn", "gdn", "gqa") * 10 + ("gqa",),
                       world_size=world_size, **parallelism)
