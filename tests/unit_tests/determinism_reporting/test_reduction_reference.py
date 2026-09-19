# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU numerical regressions for cancellation, rounding and faulty reductions."""

import copy
from dataclasses import replace

import pytest
import torch

from tools.determinism import reference as reference_module
from tools.determinism.activation_reference import activation_reference
from tools.determinism.checks import collect_checks, has_comparisons
from tools.determinism.reduction_reference import ReductionReference
from tools.determinism.reference import assert_reference_close

SIGNATURE = {"phase": "forward_backward"}


def check_reduction(actual, reduction, **kwargs):
    expected = reduction.total.float()
    return assert_reference_close(
        ({"out": torch.ones(1)}, {"w": actual}),
        ({"out": torch.ones(1)}, {"w": expected}),
        signature=SIGNATURE,
        reference_id="independent_sum:test",
        rtol=1e-6,
        atol=1e-6,
        reductions={"gradient:w": reduction},
        **kwargs,
    )


def test_cancellation_uses_term_magnitudes_but_still_rejects_local_corruption():
    terms = torch.ones(64, 2, dtype=torch.float64)
    terms[::2, 0] = -1
    terms[:, 1] = 1e6 / 64
    reduction = ReductionReference.from_terms(terms, dim=0, keepdim=False, rounding="none")
    actual = reduction.total.float()
    actual[0] = 2e-6
    assert not torch.allclose(actual, reduction.total.float(), rtol=1e-6, atol=1e-6)
    result = check_reduction(actual, reduction, numerical_controls=True)
    assert result["status"] == "passed"
    assert result["metrics"]["gradient:w"]["reduction"]["zero_sum_components"] == 1
    assert result["numerical_controls"]["gradient:w"]["flat_index"] == 0
    budget, _ = reduction.error_budget(reduction.total.float(), rtol=1e-6, atol=1e-6)
    actual[0] = 2 * budget[0]
    checks = []
    with collect_checks(checks.append), pytest.raises(AssertionError, match="differs"):
        check_reduction(actual, reduction)
    metric = checks[0]["metrics"]["gradient:w"]
    assert metric["violations"] == 1
    assert not metric["l2_failed"]  # A norm-only check would miss this error.


def test_norm_guard_rejects_systematic_drift_inside_conservative_component_bounds():
    terms = torch.tensor([[1e6, 1e6], [-1e6 + 1, -1e6 + 1]], dtype=torch.float64)
    reduction = ReductionReference.from_terms(terms, dim=0, keepdim=False, rounding="none")
    checks = []
    with collect_checks(checks.append), pytest.raises(AssertionError, match="differs"):
        check_reduction(reduction.total.float() + 1e-4, reduction)
    metric = checks[0]["metrics"]["gradient:w"]
    assert metric["violations"] == 0  # A worst-case component bound alone is too loose.
    assert metric["l2_failed"]


@pytest.mark.parametrize("error", ["nan", "inf", "missing_row", "wrong_sign"])
def test_wrong_reductions_fail(error):
    terms = torch.tensor([[2.0, -3.0], [4.0, -5.0]], dtype=torch.float64)
    reduction = ReductionReference.from_terms(terms, dim=0, keepdim=False, rounding="none")
    actual = reduction.total.float()
    if error in ("nan", "inf"):
        actual[0] = float(error)
    elif error == "missing_row":
        actual -= terms[0].float()
    else:
        actual *= -1
    with pytest.raises(AssertionError, match="differs"):
        check_reduction(actual, reduction)


def test_reduction_budget_accounts_for_length_and_rejects_malformed_contracts():
    short = ReductionReference.from_terms(
        torch.ones(2, 1, dtype=torch.float64), dim=0, keepdim=False, rounding="none"
    )
    long = ReductionReference.from_terms(
        torch.full((64, 1), 2 / 64, dtype=torch.float64), dim=0, keepdim=False, rounding="none"
    )
    short_bound, _ = short.error_budget(short.total.float(), rtol=0, atol=0)
    long_bound, _ = long.error_budget(long.total.float(), rtol=0, atol=0)
    assert long_bound > short_bound
    for invalid in (
        replace(short, terms=0),
        replace(short, terms=True),
        replace(short, terms=2**25),
        replace(short, total=short.total.float()),
        replace(short, sum_absolute_terms=torch.zeros_like(short.total)),
        replace(short, sum_absolute_terms=torch.full_like(short.total, float("inf"))),
    ):
        with pytest.raises(ValueError):
            invalid.error_budget(short.total.float(), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="mismatched"):
        short.error_budget(short.total.float() + 1, rtol=1e-6, atol=1e-6)


def test_unknown_reduction_and_nonfinite_terms_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        ReductionReference.from_terms(
            torch.tensor([float("nan")]), dim=0, keepdim=False, rounding="none"
        )
    pair = ({"out": torch.ones(1)}, {"w": torch.ones(1)})
    with pytest.raises(ValueError, match="unknown"):
        assert_reference_close(
            pair,
            pair,
            signature=SIGNATURE,
            reference_id="test",
            rtol=0,
            atol=0,
            reductions={"gradient:typo": None},
        )


def test_numerical_control_detects_disabled_comparator(monkeypatch):
    monkeypatch.setattr(reference_module, "_reference_metrics", lambda *args: {"violations": 0})
    pair = ({"out": torch.ones(1)}, {"w": torch.ones(1)})
    with pytest.raises(AssertionError, match="missed"):
        assert_reference_close(
            pair,
            pair,
            signature=SIGNATURE,
            reference_id="test",
            rtol=1e-6,
            atol=1e-6,
            numerical_controls=True,
        )


def test_missing_required_numerical_controls_cannot_receive_passing_credit():
    terms = torch.ones(2, 1, dtype=torch.float64)
    reduction = ReductionReference.from_terms(terms, dim=0, keepdim=False, rounding="none")
    check = check_reduction(reduction.total.float(), reduction, numerical_controls=True)
    assert has_comparisons(check)
    for corrupt in ("missing", "undetected", "wrong_key"):
        changed = copy.deepcopy(check)
        if corrupt == "missing":
            changed["numerical_controls"].pop("gradient:w")
        elif corrupt == "undetected":
            changed["numerical_controls"]["gradient:w"]["detected"] = False
        else:
            changed["numerical_controls"]["wrong"] = changed["numerical_controls"].pop("gradient:w")
        assert not has_comparisons(changed)


def test_bf16_bias_reference_preserves_round_before_sum_and_mathematical_diagnostic():
    generator = torch.Generator().manual_seed(17)
    inputs = (
        torch.randn(64, 16, generator=generator, dtype=torch.bfloat16, requires_grad=True),
        torch.randn(16, generator=generator, dtype=torch.bfloat16, requires_grad=True),
    )
    expected, reductions, mathematical = activation_reference("bias_swiglu", inputs)
    staged = mathematical[1]["in[0]"].bfloat16().double().sum(0).bfloat16()
    assert torch.equal(expected[1]["in[1]"], staged)
    assert not torch.equal(staged, mathematical[1]["in[1]"].bfloat16())
    assert all(value.grad is None for value in inputs)
    result = assert_reference_close(
        expected,
        expected,
        signature=SIGNATURE,
        reference_id="test",
        rtol=0.02,
        atol=0.001,
        reductions=reductions,
        mathematical_reference=mathematical,
        numerical_controls=True,
    )
    assert result["status"] == "passed"
    assert result["mathematical_reference_metrics"]["gradient:in[1]"]["max_absolute_error"] > 0


@pytest.mark.parametrize("case", ["bias_swiglu", "weighted_swiglu", "weighted_squared_relu"])
def test_fp64_reference_matches_hand_computed_values_and_all_input_gradients(case):
    if case == "weighted_squared_relu":
        inputs = (
            torch.tensor([[-2.0, 0.0, 2.0]], requires_grad=True),
            torch.tensor([[0.5]], requires_grad=True),
        )
        expected_out, expected_grad, reduction_value = [[0.0, 0.0, 2.0]], [[0.0, 0.0, 2.0]], [[4.0]]
    else:
        x = torch.tensor([[0.0, 0.0, 2.0, -4.0]], requires_grad=True)
        if case == "bias_swiglu":
            inputs = (x, torch.zeros(4, requires_grad=True))
            expected_out, expected_grad, reduction_value = (
                [[0.0, 0.0]],
                [[1.0, -2.0, 0.0, 0.0]],
                [1.0, -2.0, 0.0, 0.0],
            )
        else:
            inputs = (x, None, torch.tensor([[0.5]], requires_grad=True))
            expected_out, expected_grad, reduction_value = (
                [[0.0, 0.0]],
                [[0.5, -1.0, 0.0, 0.0]],
                [[0.0]],
            )
    expected, reductions, mathematical = activation_reference(case, inputs)
    assert torch.equal(expected[0]["out"], torch.tensor(expected_out))
    assert torch.equal(expected[1]["in[0]"], torch.tensor(expected_grad))
    assert torch.equal(
        next(iter(reductions.values())).total, torch.tensor(reduction_value).double()
    )
    assert len(expected[1]) == sum(value is not None for value in inputs)
    assert all(value.dtype == torch.float64 for group in mathematical for value in group.values())
