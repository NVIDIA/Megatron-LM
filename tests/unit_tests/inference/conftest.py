# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in ``large`` marker for inference tests that need tens of GiB of device memory."""

import re

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "large: needs tens of GiB of device memory (e.g. int32-limit sizes); skipped unless the"
        " -m expression names it",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("large") and not re.search(
        r"\blarge\b", item.config.getoption("markexpr") or ""
    ):
        pytest.skip("large test: opt in with -m large")
