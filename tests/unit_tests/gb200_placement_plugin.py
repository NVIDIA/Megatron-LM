# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Apply the portable CI placement before pytest evaluates marker expressions."""

import os

import pytest

from tests.unit_tests.gb200_placement import is_gb200_candidate


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gb200_only: portable case assigned exclusively to the enabled GB200 CI lane"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    if os.environ.get("UNIT_TEST_GB200_ENABLED") != "true":
        return
    for item in items:
        path, _, name = item.nodeid.partition("::")
        params = item.callspec.params if hasattr(item, "callspec") else {}
        if is_gb200_candidate(path, name, params):
            item.add_marker(pytest.mark.launch_on_gb200)
            item.add_marker(pytest.mark.gb200_only)
