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


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Let torchrun continue to the other marker phase after an empty phase.

    ``torch.distributed.run`` maps a worker's pytest exit 5 to launcher exit 1,
    so this translation must happen inside each pytest worker. The collection
    sentinel remains absent, allowing the shell runner to reject the job if
    both selective marker phases are empty.
    """

    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
