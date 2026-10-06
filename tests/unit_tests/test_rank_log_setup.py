# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from megatron.rank_log_setup import suppress_duplicate_logs_off_rank0

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_module_does_not_import_torch():
    """The helper must stay importable before torch.

    Callers run it above their torch import so it can reach the deprecations
    torch raises while being imported. An import here that reaches torch would
    defeat that silently, with the only symptom a larger log.
    """
    code = textwrap.dedent("""
        import sys

        import megatron.rank_log_setup  # noqa: F401

        leaked = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
        assert not leaked, f"importing megatron.rank_log_setup pulled in {leaked}"
        """)
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


@pytest.mark.parametrize(
    "rank, expect_warning",
    [("0", True), ("1", False), ("37", False)],
    ids=["rank0", "rank1", "rank37"],
)
def test_warnings_suppressed_off_rank_zero(rank, expect_warning, monkeypatch):
    monkeypatch.setenv("RANK", rank)
    pytree_logger = logging.getLogger("torch.utils._pytree")
    original_level = pytree_logger.level
    try:
        # catch_warnings restores both the filter list and showwarning on exit.
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            suppress_duplicate_logs_off_rank0()
            warnings.warn("a job-wide deprecation", category=UserWarning)
        assert bool(raised) is expect_warning
        expected_level = logging.NOTSET if expect_warning else logging.ERROR
        assert pytree_logger.level == expected_level
    finally:
        pytree_logger.setLevel(original_level)


def test_unset_rank_is_treated_as_rank_zero(monkeypatch):
    """A bare import outside a launcher must not silently swallow warnings."""
    monkeypatch.delenv("RANK", raising=False)
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        suppress_duplicate_logs_off_rank0()
        warnings.warn("a job-wide deprecation", category=UserWarning)
    assert raised
