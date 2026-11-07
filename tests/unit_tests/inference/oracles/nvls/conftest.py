# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fixtures for the NVLS collective oracles: subgroups with symmetric buffers, created once."""

import pytest

from tests.unit_tests.inference.oracles.nvls import nvls_oracle_utils as nu


@pytest.fixture(scope="session")
def nvls_groups():
    """``NvlsGroups``: ``nvls_groups.get(ws)`` is this rank's group of ``ws`` ranks."""
    return nu.NvlsGroups()


@pytest.fixture(autouse=True)
def _finish_before_next_test():
    """Every rank finishes a test's kernels before any rank starts the next test's.

    The reduce-scatter kernels end without a barrier, so the next test's first local write to a
    symmetric buffer could otherwise land while a peer still reads it.
    """
    yield
    nu.sync_world()
