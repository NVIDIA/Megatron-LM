# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pytest hook used to detect an unexpectedly empty selective test run."""

import os
from pathlib import Path

import pytest


def pytest_collection_finish(session: pytest.Session) -> None:
    """Record that at least one selected test survived marker filtering."""

    sentinel = os.environ.get("MCORE_SELECTED_TEST_SENTINEL")
    if sentinel and session.items:
        Path(sentinel).touch()
