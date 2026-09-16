"""Exercise exact original/refactored run comparisons with real TensorBoard files."""

import pytest
from torch.utils.tensorboard import SummaryWriter

from tests.test_utils.python_scripts.compare_training_runs import (
    DEFAULT_TAGS,
    compare_training_runs,
    read_training_scalars,
)


def _write(log_dir, values, *, tags=DEFAULT_TAGS, suffix=""):
    with SummaryWriter(str(log_dir), filename_suffix=suffix) as writer:
        for tag in tags:
            for step, value in values:
                writer.add_scalar(tag, value, step)


def test_exact_comparison_includes_both_resume_phases(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for directory in (baseline, candidate):
        _write(directory, [(1, 1.25), (2, 1.5)], suffix=".fresh")
        _write(directory, [(3, 1.0), (4, 0.75)], suffix=".resume")
    result = compare_training_runs(baseline, candidate, start_step=1, end_step=4)
    assert result["status"] == "passed"
    assert result["baseline"] == result["candidate"]
    assert result["baseline"]["lm loss"] == [(1, 1.25), (2, 1.5), (3, 1.0), (4, 0.75)]


@pytest.mark.parametrize(
    "values",
    [
        [(1, 1.0)],  # Missing required terminal step.
        [(2, 1.0)],  # Missing required initial step.
        [(1, 1.0), (1, 1.0), (2, 1.0)],  # Duplicate, even if values match.
        [(2, 1.0), (1, 1.0)],  # Out of order.
    ],
)
def test_incomplete_or_duplicate_coverage_is_not_a_pass(tmp_path, values):
    _write(tmp_path, values)
    with pytest.raises(ValueError, match="exactly one ordered value"):
        read_training_scalars(tmp_path, start_step=1, end_step=2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(tmp_path, value):
    _write(tmp_path, [(1, value)])
    with pytest.raises(ValueError, match="non-finite"):
        read_training_scalars(tmp_path, start_step=1, end_step=1)


def test_missing_metric_is_not_silently_omitted(tmp_path):
    _write(tmp_path, [(1, 1.0)], tags=("lm loss",))
    with pytest.raises(ValueError, match="missing metric"):
        read_training_scalars(tmp_path, start_step=1, end_step=1)


def test_differences_below_five_decimal_places_still_fail(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _write(baseline, [(1, 1.5)])
    _write(candidate, [(1, 1.500000238418579)])
    with pytest.raises(AssertionError, match="at step 1"):
        compare_training_runs(baseline, candidate, start_step=1, end_step=1)


def test_requested_window_does_not_require_unrequested_steps(tmp_path):
    _write(tmp_path, [(0, 2.0), (1, 1.25), (2, 1.0), (3, 0.75)])
    result = read_training_scalars(tmp_path, start_step=1, end_step=2)
    assert result["lm loss"] == [(1, 1.25), (2, 1.0)]


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 1)])
def test_invalid_step_windows_are_rejected(tmp_path, start, end):
    with pytest.raises(ValueError, match="start_step"):
        read_training_scalars(tmp_path, start_step=start, end_step=end)


@pytest.mark.parametrize("tags", [(), ("lm loss", "lm loss")])
def test_required_tags_must_be_nonempty_and_unique(tmp_path, tags):
    with pytest.raises(ValueError, match="nonempty and unique"):
        read_training_scalars(tmp_path, start_step=1, end_step=1, tags=tags)
