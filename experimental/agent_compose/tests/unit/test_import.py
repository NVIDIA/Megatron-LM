# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Import checks for the Agent Compose package skeleton."""

from __future__ import annotations

import importlib


def test_three_layer_skeleton_imports() -> None:
    modules = [
        "megatron.lite",
        "megatron.lite.primitive",
        "megatron.lite.model",
        "megatron.lite.runtime",
    ]

    for module in modules:
        imported = importlib.import_module(module)
        assert imported.__name__ == module
