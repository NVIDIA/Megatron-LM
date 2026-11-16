# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixtures shared by focused telemetry tests."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_PATH = REPO_ROOT / "megatron/core/telemetry/telemetry.py"
TELEMETRY_MODULE = "_megatron_focused_test_telemetry"


def _lens_installed():
    try:
        return importlib.util.find_spec("nemo.lens") is not None
    except ModuleNotFoundError:
        return False


def pytest_configure(config):
    """Register the marker for tests that exercise an installed nemo-lens."""
    config.addinivalue_line("markers", "requires_lens: test needs the optional nemo-lens package")


def pytest_runtest_setup(item):
    """Skip Lens-backed tests in environments without the optional dependency."""
    if item.get_closest_marker("requires_lens") is not None and not _lens_installed():
        pytest.skip("requires nemo-lens")


@pytest.fixture(scope="session")
def telemetry_module():
    """Load the production shim once and isolate its Lens registration."""
    SpanRegistry = pytest.importorskip("nemo.lens").SpanRegistry
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
