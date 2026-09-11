# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import subprocess
from types import SimpleNamespace


def test_extract_series_accepts_precision_metadata(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="/workspace/mlm\n")
    )
    compare_golden_values = importlib.import_module(
        "tests.test_utils.python_scripts.compare_golden_values_kl"
    )
    metric = {"value_precision": "full", "values": {"1": 1.23456789}}

    assert compare_golden_values._extract_series(metric) == {1: 1.23456789}
