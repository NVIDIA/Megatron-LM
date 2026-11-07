# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Markers and hardware gating for the inference kernel oracle tests (see README.md)."""

from pathlib import Path

import pytest

from tests.unit_tests.find_test_cases import file_has_marker
from tests.unit_tests.inference.oracles import oracle_utils


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_sm(*majors): run only on these CUDA compute-capability majors; skip elsewhere,"
        " fail when the run targets them (oracle_utils.requires_sm)",
    )
    config.addinivalue_line(
        "markers",
        "sanitizer: small oracle case for the compute-sanitizer pass (-m sanitizer under"
        " compute-sanitizer --tool memcheck)",
    )
    config.addinivalue_line(
        "markers",
        "large: needs tens of GB (e.g. int32-overflow sizes); skipped unless the -m expression"
        " names it",
    )
    oracle_utils.configure_session(getattr(config.option, "markexpr", ""))


def pytest_runtest_setup(item):
    # One GPU per rank when CI runs the bucket on every GPU, before anything here touches CUDA.
    # Not in pytest_configure: pytest also loads this conftest in the buckets that ignore this
    # directory's files, and their tests must keep their own device.
    oracle_utils.use_local_rank_device()
    if item.get_closest_marker("large") and not oracle_utils.large_selected():
        pytest.skip("large test: opt in with -m large")
    path = Path(str(item.fspath))
    for mark in item.iter_markers("requires_sm"):
        launch_markers = oracle_utils.LAUNCH_MARKERS.values()
        action, reason = oracle_utils.hardware_decision(
            mark.args,
            device_major=oracle_utils.device_sm_major(),
            target_sm=oracle_utils.session_target_sm(),
            launch_markers=[m for m in launch_markers if item.get_closest_marker(m)],
            file_has_markers=[m for m in launch_markers if file_has_marker(path, m)],
        )
        if action == "skip":
            pytest.skip(reason)
        if action == "fail":
            pytest.fail(reason, pytrace=False)
