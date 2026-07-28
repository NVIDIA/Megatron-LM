# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""pytest configuration for the ``examples/multimodal_dev`` test suite.

This suite runs as its own CI bucket (see
``tests/test_utils/recipes/h100/unit-tests.yaml``), so
``tests/unit_tests/conftest.py`` is not an ancestor of the collected files
and none of its hooks apply here.  The pieces the CI runner
(``tests/unit_tests/run_ci_test.sh``) depends on are reproduced below:

* ``--experimental`` — the runner invokes pytest a second time with this
  flag, which would otherwise be rejected as an unknown option;
* ``pytest_sessionfinish`` — that second invocation collects nothing in
  this bucket (there are no experimental tests), and pytest's "no tests
  collected" exit code 5 must not fail the job.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.core import config  # noqa: E402


def pytest_addoption(parser):
    """Mirror the ``--experimental`` option from tests/unit_tests/conftest.py."""
    parser.addoption(
        '--experimental',
        action='store_true',
        help="pass that argument to enable experimental flag during testing (DEFAULT: False)",
    )


@pytest.fixture(autouse=True)
def experimental(request):
    """Simple fixture setting the experimental flag [CPU | GPU]"""
    config.ENABLE_EXPERIMENTAL = request.config.getoption("--experimental") is True


def pytest_sessionfinish(session, exitstatus):
    """Treat "no tests collected" as success, as the unit-test suite does."""
    if exitstatus == 5:
        session.exitstatus = 0
