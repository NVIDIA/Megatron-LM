# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from megatron.core.process_groups_config import ProcessGroupFallbackWarning
from megatron.rank_log_setup import suppress_duplicate_logs_off_rank0

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def restore_pytree_logger_level():
    """The suppression sets the level of torch's pytree logger; later tests expect the default."""
    pytree_logger = logging.getLogger("torch.utils._pytree")
    original_level = pytree_logger.level
    yield
    pytree_logger.setLevel(original_level)


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


@pytest.mark.parametrize("rank", ["0", "1"], ids=["rank0", "rank1"])
def test_process_group_fallback_warning_is_reported_on_every_rank(rank, monkeypatch):
    """Ranks fall back to the global process groups differently, for example per pipeline stage."""
    monkeypatch.setenv("RANK", rank)
    with warnings.catch_warnings(record=True) as raised:
        # Start from no filters, so that the filters the suppression adds decide.
        warnings.resetwarnings()
        suppress_duplicate_logs_off_rank0()
        warnings.warn("a fallback", ProcessGroupFallbackWarning)
        warnings.warn("a job-wide deprecation", FutureWarning)
        # Libraries such as CUTLASS DSL show every warning inside their own catch_warnings().
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("a fallback inside a library", ProcessGroupFallbackWarning)
            warnings.warn("a deprecation inside a library", FutureWarning)
    if rank == "0":
        expected = [
            "a fallback",
            "a job-wide deprecation",
            "a fallback inside a library",
            "a deprecation inside a library",
        ]
    else:
        expected = ["a fallback", "a fallback inside a library"]
    assert [str(w.message) for w in raised] == expected


@pytest.mark.parametrize("installed", ["before", "after"])
@pytest.mark.parametrize(
    "category", [ProcessGroupFallbackWarning, FutureWarning, UserWarning, DeprecationWarning]
)
def test_error_filters_still_raise_off_rank_zero(category, installed, monkeypatch):
    """Rank 0 raises under an error filter, so every other rank must raise too.

    A rank that ignored the warning instead would wait for rank 0 at its next collective.
    """
    monkeypatch.setenv("RANK", "1")
    with warnings.catch_warnings():
        if installed == "before":
            warnings.simplefilter("error", category)
        suppress_duplicate_logs_off_rank0()
        if installed == "after":
            warnings.simplefilter("error", category)
        with pytest.raises(category, match="under an error filter"):
            warnings.warn("a warning under an error filter", category)


@pytest.mark.filterwarnings(
    "error::megatron.core.process_groups_config.ProcessGroupFallbackWarning"
)
def test_pytest_error_filter_still_raises_off_rank_zero(monkeypatch):
    """pytest installs the marker's filter, like one from -W, before the test calls the helper."""
    monkeypatch.setenv("RANK", "1")
    with warnings.catch_warnings():
        suppress_duplicate_logs_off_rank0()
        with pytest.raises(ProcessGroupFallbackWarning, match="a fallback"):
            warnings.warn("a fallback", ProcessGroupFallbackWarning)
