# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from tests.functional_tests.python_test_utils.common import (
    ApproximateTest,
    DeterministicTest,
    GoldenValueMetric,
    MissingTensorboardLogsError,
    NotApproximateError,
    NotDeterminsticError,
    TypeOfTestResult,
    _filter_checks,
    pipeline,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def make_metric(values: dict, step_interval: int = 1) -> GoldenValueMetric:
    steps = sorted(values)
    return GoldenValueMetric(
        start_step=steps[0], end_step=steps[-1], step_interval=step_interval, values=values
    )


def run(golden, actual, checks, compare_approximate=False):
    pipeline(
        compare_approximate_results=compare_approximate,
        golden_values=golden,
        actual_values=actual,
        checks=checks,
    )


# ── ApproximateTest ───────────────────────────────────────────────────────────


class TestApproximateTest:
    def test_defaults(self):
        t = ApproximateTest()
        assert t.atol == 0
        assert t.rtol == 1e-5

    def test_type(self):
        assert ApproximateTest().type_of_test_result == TypeOfTestResult.APPROXIMATE

    def test_error_message_type(self):
        assert isinstance(ApproximateTest().error_message("m"), NotApproximateError)


# ── DeterministicTest ─────────────────────────────────────────────────────────


class TestDeterministicTest:
    def test_tolerances_are_zero(self):
        t = DeterministicTest()
        assert t.atol == 0
        assert t.rtol == 0.0

    def test_type(self):
        assert DeterministicTest().type_of_test_result == TypeOfTestResult.DETERMINISTIC

    def test_error_message_type(self):
        assert isinstance(DeterministicTest().error_message("m"), NotDeterminsticError)


# ── _filter_checks ────────────────────────────────────────────────────────────


class TestFilterChecks:
    def test_returns_only_approximate(self):
        checks = [ApproximateTest(), DeterministicTest(), ApproximateTest()]
        result = _filter_checks(checks, TypeOfTestResult.APPROXIMATE)
        assert len(result) == 2
        assert all(c.type_of_test_result == TypeOfTestResult.APPROXIMATE for c in result)

    def test_returns_only_deterministic(self):
        checks = [ApproximateTest(), DeterministicTest()]
        result = _filter_checks(checks, TypeOfTestResult.DETERMINISTIC)
        assert len(result) == 1
        assert result[0].type_of_test_result == TypeOfTestResult.DETERMINISTIC

    def test_empty_when_no_match(self):
        assert _filter_checks([ApproximateTest()], TypeOfTestResult.DETERMINISTIC) == []


# ── pipeline — deterministic ──────────────────────────────────────────────────


class TestPipelineDeterministic:
    def test_exact_match_passes(self):
        gv = make_metric({1: 1.0, 2: 2.0, 3: 3.0})
        run({"loss": gv}, {"loss": gv}, {"loss": [DeterministicTest()]})

    def test_single_mismatch_fails(self):
        golden = make_metric({1: 1.0, 2: 2.0, 3: 3.0})
        actual = make_metric({1: 1.0, 2: 2.0, 3: 3.1})
        with pytest.raises(AssertionError, match="loss"):
            run({"loss": golden}, {"loss": actual}, {"loss": [DeterministicTest()]})

    def test_skipped_in_compare_approximate_mode(self):
        # Deterministic checks must be silently skipped when
        # compare_approximate_results=True, even if values differ wildly.
        golden = make_metric({1: 1.0})
        actual = make_metric({1: 999.0})
        run(
            {"loss": golden},
            {"loss": actual},
            {"loss": [DeterministicTest()]},
            compare_approximate=True,
        )


# ── pipeline — approximate ────────────────────────────────────────────────────


class TestPipelineApproximate:
    def test_within_rtol_passes(self):
        golden = make_metric({1: 1.0, 2: 2.0})
        actual = make_metric({1: 1.04, 2: 2.04})  # 4 % < 5 % rtol
        run({"loss": golden}, {"loss": actual}, {"loss": [ApproximateTest(rtol=0.05)]})

    def test_outside_rtol_fails(self):
        golden = make_metric({1: 1.0, 2: 2.0})
        actual = make_metric({1: 1.2, 2: 2.4})  # 20 % > 5 % rtol
        with pytest.raises(AssertionError, match="loss"):
            run({"loss": golden}, {"loss": actual}, {"loss": [ApproximateTest(rtol=0.05)]})

    def test_within_atol_passes(self):
        golden = make_metric({1: 0.0, 2: 0.0})
        actual = make_metric({1: 0.5, 2: 0.5})  # within atol=1
        run({"loss": golden}, {"loss": actual}, {"loss": [ApproximateTest(atol=1, rtol=0)]})

    def test_outside_atol_fails(self):
        golden = make_metric({1: 0.0})
        actual = make_metric({1: 2.0})  # outside atol=1
        with pytest.raises(AssertionError):
            run({"loss": golden}, {"loss": actual}, {"loss": [ApproximateTest(atol=1, rtol=0)]})

    def test_single_bad_step_in_large_run_passes(self):
        # With 1000 steps: total_steps_evaluated=1001, num_failing_allowed=10.
        # 1 bad step → mean(is_close) = 999/1000 = 0.999, well above threshold.
        n = 1000
        golden = {i: 1.0 for i in range(1, n + 1)}
        actual = {**golden, n: 999.0}
        run(
            {"loss": make_metric(golden)},
            {"loss": make_metric(actual)},
            {"loss": [ApproximateTest(rtol=0.05)]},
        )

    def test_majority_bad_steps_fails(self):
        n = 1000
        golden = {i: 1.0 for i in range(1, n + 1)}
        actual = {i: 999.0 for i in range(1, n + 1)}  # all wrong
        with pytest.raises(AssertionError):
            run(
                {"loss": make_metric(golden)},
                {"loss": make_metric(actual)},
                {"loss": [ApproximateTest(rtol=0.05)]},
            )


# ── pipeline — missing metric ─────────────────────────────────────────────────


class TestPipelineMissingMetric:
    def test_raises_when_metric_absent_from_actual(self):
        gv = make_metric({1: 1.0})
        with pytest.raises(MissingTensorboardLogsError):
            run({"loss": gv}, {}, {"loss": [DeterministicTest()]})


# ── pipeline — iteration-time special case ────────────────────────────────────


class TestPipelineIterationTime:
    def test_uses_median_not_per_step_values(self):
        # Per-step values diverge but medians match — should pass.
        golden = make_metric({1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25})
        # median([0.10, 0.25, 0.26, 100.0]) = (0.25+0.26)/2 = 0.255 ≈ 0.25 ✓
        actual = make_metric({1: 0.10, 2: 0.25, 3: 0.26, 4: 100.0})
        run(
            {"iteration-time": golden},
            {"iteration-time": actual},
            {"iteration-time": [ApproximateTest(rtol=0.05)]},
        )

    def test_diverging_medians_fails(self):
        golden = make_metric({1: 0.25, 2: 0.25, 3: 0.25})
        actual = make_metric({1: 0.50, 2: 0.50, 3: 0.50})  # median 0.50, 100 % off
        with pytest.raises(AssertionError, match="iteration-time"):
            run(
                {"iteration-time": golden},
                {"iteration-time": actual},
                {"iteration-time": [ApproximateTest(rtol=0.05)]},
            )

    def test_nan_warmup_step_does_not_break_median(self):
        # Step 1 is "nan" (warm-up). Median of [inf, 0.25, 0.25, 0.25] = 0.25.
        golden = make_metric({1: "nan", 2: 0.25, 3: 0.25, 4: 0.25})
        actual = make_metric({1: "nan", 2: 0.25, 3: 0.25, 4: 0.25})
        run(
            {"iteration-time": golden},
            {"iteration-time": actual},
            {"iteration-time": [ApproximateTest(rtol=0.05)]},
        )


# ── pipeline — "nan" string handling ─────────────────────────────────────────


class TestPipelineNanHandling:
    def test_matching_nan_strings_pass(self):
        # "nan" → np.inf; np.isclose(inf, inf) is True (numpy treats same-sign
        # infinities as close), so matching "nan" steps are not penalised.
        golden = make_metric({1: "nan", 2: 1.0, 3: 1.0})
        actual = make_metric({1: "nan", 2: 1.0, 3: 1.0})
        run({"loss": golden}, {"loss": actual}, {"loss": [ApproximateTest(rtol=0.05)]})

    def test_nan_in_golden_but_not_actual_fails_deterministic(self):
        golden = make_metric({1: 1.0, 2: "nan"})
        actual = make_metric({1: 1.0, 2: 1.0})
        with pytest.raises(AssertionError):
            run({"loss": golden}, {"loss": actual}, {"loss": [DeterministicTest()]})


# ── pipeline — multiple metrics ───────────────────────────────────────────────


class TestPipelineMultipleMetrics:
    def test_all_pass(self):
        gv = make_metric({1: 1.0, 2: 2.0})
        run(
            {"loss": gv, "num-zeros": gv},
            {"loss": gv, "num-zeros": gv},
            {"loss": [DeterministicTest()], "num-zeros": [DeterministicTest()]},
        )

    def test_one_failing_metric_fails_overall(self):
        good = make_metric({1: 1.0})
        bad_golden = make_metric({1: 1.0})
        bad_actual = make_metric({1: 999.0})
        with pytest.raises(AssertionError, match="num-zeros"):
            run(
                {"loss": good, "num-zeros": bad_golden},
                {"loss": good, "num-zeros": bad_actual},
                {"loss": [DeterministicTest()], "num-zeros": [DeterministicTest()]},
            )

    def test_failure_message_lists_all_failed_metrics(self):
        bad_golden = make_metric({1: 1.0})
        bad_actual = make_metric({1: 999.0})
        with pytest.raises(AssertionError, match="loss") as exc_info:
            run(
                {"loss": bad_golden, "num-zeros": bad_golden},
                {"loss": bad_actual, "num-zeros": bad_actual},
                {"loss": [DeterministicTest()], "num-zeros": [DeterministicTest()]},
            )
        assert "num-zeros" in str(exc_info.value)
