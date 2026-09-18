# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for collective references and replay failure/mutation behavior."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tools.determinism.collective_reference import (
    collective_reference,
    collective_rtol,
    collective_shapes,
)
from tools.determinism.coverage import ReplayMismatch, collect_observations
from tools.determinism.reduction_reference import ReductionReference
from tools.determinism.reference import assert_reference_close


@pytest.fixture(scope="module")
def harness():
    """Load the real CPU-capable harness without the kernel package's GPU bootstrap."""
    path = Path(__file__).resolve().parents[3] / "tests/unit_tests/determinism/kernels/harness.py"
    spec = importlib.util.spec_from_file_location("_collective_replay_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "case,outputs,gradients",
    [
        ("copy", [[[1, 2], [3, 4]], [[10, 20], [30, 40]]], [[[3, 6], [9, 12]]] * 2),
        ("reduce", [[[11, 22], [33, 44]]] * 2, [[[1, 2], [3, 4]], [[2, 4], [6, 8]]]),
        (
            "gather_first",
            [[[1, 2], [3, 4], [10, 20], [30, 40]]] * 2,
            [[[3, 6], [9, 12]], [[15, 18], [21, 24]]],
        ),
        (
            "scatter_first",
            [[[11, 22], [33, 44]], [[55, 66], [77, 88]]],
            [[[1, 2], [3, 4], [2, 4], [6, 8]]] * 2,
        ),
        (
            "gather_last",
            [[[1, 2, 10, 20], [3, 4, 30, 40]]] * 2,
            [[[3, 6], [15, 18]], [[9, 12], [21, 24]]],
        ),
        (
            "scatter_last",
            [[[11, 22], [55, 66]], [[33, 44], [77, 88]]],
            [[[1, 2, 2, 4], [3, 4, 6, 8]]] * 2,
        ),
    ],
)
def test_all_mapping_outputs_and_gradients_match_hand_calculated_values(
    dtype, case, outputs, gradients
):
    input_shape, output_shape = collective_shapes(case, 2, (2, 2))
    x = torch.arange(1, 1 + input_shape[0] * input_shape[1], dtype=dtype).reshape(input_shape)
    dy = torch.arange(1, 1 + output_shape[0] * output_shape[1], dtype=dtype).reshape(output_shape)
    for rank in range(2):
        expected, reductions, mathematical = collective_reference(
            case, [x, 10 * x], [dy, 2 * dy], rank
        )
        assert torch.equal(expected[0]["out"], torch.tensor(outputs[rank], dtype=dtype))
        assert torch.equal(expected[1]["in[0]"], torch.tensor(gradients[rank], dtype=dtype))
        assert len(reductions) == 1
        assert all(t.dtype == torch.float64 for values in mathematical for t in values.values())
        reduction = next(iter(reductions.values()))
        assert reduction.accumulation_dtype == dtype and reduction.exact_terms
        assert_reference_close(
            expected,
            expected,
            signature={"phase": "forward_backward"},
            reference_id="hand_calculated_two_rank",
            rtol=collective_rtol(dtype, 2),
            atol=0,
            reductions=reductions,
            numerical_controls=True,
        )


@pytest.mark.parametrize("layout", ["contiguous_offset", "strided_offset", "broadcast_offset"])
def test_replay_clone_preserves_captured_storage_offset(harness, layout):
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)[1:]
    if layout == "strided_offset":
        source = source[:, ::2]
    elif layout == "broadcast_offset":
        source = source[:1].expand(3, 6)
    source.requires_grad_()
    clone = harness.clone_inputs((source,))[0]
    assert clone.storage_offset() == source.storage_offset()
    assert clone.stride() == source.stride()
    assert clone.requires_grad
    assert torch.equal(clone, source)
    assert clone.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fault", ["missing_rank", "wrong_rank_slice", "gradient_sign"])
def test_reference_rejects_rank_routing_and_reduction_corruption(dtype, fault):
    x = torch.arange(1, 9, dtype=dtype).reshape(4, 2)
    dy = torch.arange(1, 5, dtype=dtype).reshape(2, 2)
    expected, reductions, _ = collective_reference("scatter_first", [x, x * 10], [dy, dy * 2], 0)
    actual = tuple({key: value.clone() for key, value in values.items()} for values in expected)
    if fault == "missing_rank":
        actual[0]["out"] -= x[:2] * 10
    elif fault == "wrong_rank_slice":
        actual[0]["out"] = x[2:] * 11
    else:
        actual[1]["in[0]"] *= -1
    with pytest.raises(AssertionError, match="differs"):
        assert_reference_close(
            actual,
            expected,
            signature={"phase": "forward_backward"},
            reference_id="hand_calculated_two_rank",
            rtol=collective_rtol(dtype, 2),
            atol=0,
            reductions=reductions,
        )


@pytest.mark.parametrize(
    "case", ["copy", "reduce", "gather_first", "scatter_first", "gather_last", "scatter_last"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
def test_reference_training_shapes_match_the_two_dimensional_contract(case, dtype, shape):
    value = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)
    dim = 0 if case.endswith("first") else value.ndim - 1
    output_shape = list(shape)
    if case.startswith("gather"):
        output_shape[dim] *= 2
    elif case.startswith("scatter"):
        output_shape[dim] //= 2
    gradient = torch.ones(output_shape, dtype=dtype)

    def flatten(tensor):
        return (
            tensor.reshape(tensor.shape[0], -1)
            if case.endswith("first")
            else tensor.reshape(-1, tensor.shape[-1])
        )

    for rank in range(2):
        reference, reductions, mathematical = collective_reference(
            case, [value, 10 * value], [gradient, 2 * gradient], rank
        )
        flat_reference, flat_reductions, flat_math = collective_reference(
            case,
            [flatten(value), flatten(10 * value)],
            [flatten(gradient), flatten(2 * gradient)],
            rank,
        )
        for key, category in (("out", 0), ("in[0]", 1)):
            assert torch.equal(flatten(reference[category][key]), flat_reference[category][key])
            assert torch.equal(flatten(mathematical[category][key]), flat_math[category][key])
        for key, reduction in reductions.items():
            assert torch.equal(flatten(reduction.total), flat_reductions[key].total)
            assert torch.equal(
                flatten(reduction.sum_absolute_terms), flat_reductions[key].sum_absolute_terms
            )


def test_bf16_reduction_contract_does_not_assume_fp32_accumulation():
    terms = torch.tensor([[256.0, 1], [1, -1], [-256, 1], [1, -1]], dtype=torch.bfloat16)
    actual = terms[0]
    for term in terms[1:]:
        actual = (actual + term).bfloat16()
    reduction = ReductionReference.from_terms(
        terms,
        dim=0,
        keepdim=False,
        rounding="materialized BF16 terms",
        accumulation_dtype=torch.bfloat16,
        exact_terms=True,
    )
    expected = reduction.total.bfloat16()
    assert not torch.equal(actual, expected)
    bf16_bound, protocol = reduction.error_budget(expected, rtol=0, atol=0)
    fp32_bound, _ = replace(reduction, accumulation_dtype=torch.float32).error_budget(
        expected, rtol=0, atol=0
    )
    error = (actual.double() - expected.double()).abs()
    assert bool((error <= bf16_bound).all()) and bool((error > fp32_bound).any())
    assert protocol["accumulation_dtype"] == "torch.bfloat16" and protocol["exact_terms"]
    loose_bound, _ = reduction.error_budget(expected, rtol=1, atol=10)
    assert torch.equal(loose_bound, bf16_bound)  # Exact terms never acquire pointwise slack.
    for invalid in (
        replace(reduction, accumulation_dtype=torch.int32),
        replace(reduction, exact_terms=1),
    ):
        with pytest.raises(ValueError):
            invalid.error_budget(expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case", ["copy", "reduce", "gather_first", "scatter_first", "gather_last", "scatter_last"]
)
def test_missing_ranks_and_invalid_reference_shapes_fail_closed(case):
    x = torch.ones(2, 2)
    with pytest.raises(ValueError, match="every group rank"):
        collective_reference(case, [x, x], [x], 0)
    with pytest.raises(ValueError):
        collective_reference(case, [x, x[:, :1]], [x, x], 0)
    with pytest.raises(ValueError):
        collective_shapes(case, 1)


@pytest.mark.parametrize("deferred,expected_calls", [(False, 2), (True, 3)])
def test_detected_mismatch_finishes_collective_replays_before_raising(
    monkeypatch, harness, deferred, expected_calls
):
    calls = []

    def run(*args, **kwargs):
        calls.append(len(calls))
        return {"out": torch.tensor([float(len(calls))])}, {"in[0]": torch.ones(1)}

    monkeypatch.setattr(harness, "run_once", run)
    observations = []
    with collect_observations(observations.append), pytest.raises(ReplayMismatch):
        harness.assert_replays_bit_exact(
            lambda x: x, (torch.ones(1, requires_grad=True),), replays=3, defer_comparison=deferred
        )
    assert len(calls) == expected_calls
    assert observations[0]["status"] == "verified_nondeterministic"
    if deferred:
        assert observations[0]["protocol"]["comparison_timing"] == "after_all_replays"


def test_replay_preserves_explicit_upstream_gradient_when_backward_mutates_it(monkeypatch, harness):
    class MutatingBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, gradient):
            return gradient.mul_(2)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    upstream = torch.arange(1.0, 7.0).reshape(2, 3).t()
    unchanged = upstream.clone()
    actual = harness.assert_replays_bit_exact(
        MutatingBackward.apply,
        (torch.ones(3, 2, requires_grad=True),),
        replays=3,
        grad_outputs={"out": upstream},
        defer_comparison=True,
    )
    assert torch.equal(upstream, unchanged) and not upstream.is_contiguous()
    assert torch.equal(actual[1]["in[0]"], unchanged * 2)
