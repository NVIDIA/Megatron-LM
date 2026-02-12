# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Local conftest for simulation tests.
Overrides the ensure_test_data fixture to skip data download.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Override parent fixture - simulation tests use mock data, no download needed."""
    # Do nothing - simulation tests use mock data
    pass
