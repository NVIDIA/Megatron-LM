# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Execution-contract coverage for Qwen lite smoke tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_qwen35_tp_replication_smoke_requests_four_gpus():
    path = (
        Path(__file__).parents[3] / "tests/smoke/model/test_qwen_lite_forward_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("qwen_lite_forward_smoke_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    markers = module.test_qwen35_tp2_tp4_mixed_attention_parity_and_backward.pytestmark
    gpus = next(marker for marker in markers if marker.name == "gpus")
    assert gpus.args == (4,)
