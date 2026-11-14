# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Self-test of the opt-in ``large`` marker hook in ``tests/unit_tests/conftest.py``."""

import re

import pytest


@pytest.mark.large
def test_large_test_runs_only_when_the_marker_expression_names_it(request):
    """The default launch skips this test, and ``-m large`` runs it.

    If the hook stopped skipping ``large`` tests, a default launch would run this test with a
    marker expression that does not name ``large``, and the assertion fails.
    """
    assert re.search(r"\blarge\b", request.config.getoption("markexpr") or "")
