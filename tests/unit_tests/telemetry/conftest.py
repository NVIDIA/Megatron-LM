# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixtures shared by focused telemetry tests."""

import importlib.util
import sys
from pathlib import Path

import pytest
from nemo.lens import SpanRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_PATH = REPO_ROOT / "megatron/core/telemetry/telemetry.py"
TELEMETRY_MODULE = "_megatron_focused_test_telemetry"


@pytest.fixture(scope="session")
def telemetry_module():
    """Load the production shim once and isolate its Lens registration."""
    if "megatron" in SpanRegistry.namespaces():
        SpanRegistry.unregister("megatron")

    spec = importlib.util.spec_from_file_location(TELEMETRY_MODULE, TELEMETRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[TELEMETRY_MODULE] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(TELEMETRY_MODULE, None)
        if "megatron" in SpanRegistry.namespaces():
            SpanRegistry.unregister("megatron")
