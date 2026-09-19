# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical reference diagnostics and byte-comparator sensitivity controls."""

from __future__ import annotations

import math
from typing import Callable

import torch

from tools.determinism.checks import observe_check
from tools.determinism.coverage import ReplayMismatch
from tools.determinism.reduction_reference import ReductionReference

TensorPair = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


def _number(value: torch.Tensor) -> float | str:
    number = float(value)
    return number if math.isfinite(number) else str(number)


def _error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    rtol: float,
    atol: float,
    allowed_error: torch.Tensor | None = None,
) -> dict:
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise AssertionError("Reference and actual tensors have different shapes or dtypes")
    if not actual.numel() or not actual.is_floating_point():
        raise ValueError("Reference checks require nonempty real floating-point tensors")
    values, expected = actual.detach().reshape(-1).double(), reference.detach().reshape(-1).double()
    finite = values.isfinite() & expected.isfinite()
    absolute = (values - expected).abs()
    allowed = (
        atol + rtol * expected.abs()
        if allowed_error is None
        else allowed_error.reshape(-1).double()
    )
    bad = ~finite | (absolute > allowed)
    nonzero = finite & (expected != 0)
    severity = torch.where(finite, absolute, torch.full_like(absolute, float("inf")))
    if bool(bad.any()):
        severity = severity.masked_fill(~bad, -1)
    location = int(severity.argmax())

    return {
        "elements": actual.numel(),
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "violations": int(bad.sum()),
        "nonfinite_elements": int((~finite).sum()),
        "zero_reference_elements": int((finite & (expected == 0)).sum()),
        "max_absolute_error": (_number(absolute[finite].max()) if bool(finite.any()) else None),
        "max_relative_error": (
            _number((absolute[nonzero] / expected[nonzero].abs()).max())
            if bool(nonzero.any())
            else None
        ),
        "sample": {
            "flat_index": location,
            "actual": _number(values[location]),
            "reference": _number(expected[location]),
            "absolute_error": _number(absolute[location]),
            "allowed_absolute_error": _number(allowed[location]),
        },
    }


def _reference_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
    reduction: ReductionReference | None,
) -> dict:
    if reduction is None:
        return _error_metrics(actual, expected, rtol, atol)
    budget, protocol = reduction.error_budget(expected, rtol=rtol, atol=atol)
    metric = _error_metrics(actual, expected, rtol, atol, budget)
    error_norm = (actual.detach().double() - expected.double()).norm()
    reference_norm = expected.double().norm()
    allowed_norm = atol * math.sqrt(expected.numel()) + rtol * reference_norm
    metric["reduction"] = protocol
    metric["l2_error"] = _number(error_norm)
    metric["allowed_l2_error"] = _number(allowed_norm)
    metric["relative_l2_error"] = (
        _number(error_norm / reference_norm) if float(reference_norm) else None
    )
    metric["l2_failed"] = not bool(
        error_norm.isfinite() & allowed_norm.isfinite() & (error_norm <= allowed_norm)
    )
    return metric


def _failed(metric: dict) -> bool:
    return bool(metric["violations"] or metric.get("l2_failed", False))


def _numerical_control(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
    reduction: ReductionReference | None,
) -> dict:
    # Exercise the same numerical evaluator, without emitting synthetic reference
    # failures. The smallest reference component includes cancellation-sensitive
    # locations. Choose a representable perturbation beyond its stated budget.
    index = int(expected.abs().reshape(-1).argmin())
    baseline = expected.reshape(-1)[index].double()
    if reduction is None:
        allowed = atol + rtol * baseline.abs()
    else:
        budget, _ = reduction.error_budget(expected, rtol=rtol, atol=atol)
        allowed = budget.reshape(-1)[index]
    spacing = torch.finfo(actual.dtype).eps * baseline.abs().clamp_min(1)
    changed = actual.detach().contiguous().clone()
    changed.reshape(-1)[index] = baseline + 4 * torch.maximum(allowed, spacing)
    if not bool(changed.reshape(-1)[index].isfinite()):
        raise ValueError("Cannot represent a finite numerical negative control")
    metric = _reference_metrics(changed, expected, rtol, atol, reduction)
    if not _failed(metric):
        raise AssertionError("Numerical comparator missed an out-of-budget tensor perturbation")
    return {
        "flat_index": index,
        "injected_value": float(changed.reshape(-1)[index]),
        "detected": True,
    }


def assert_reference_close(
    actual: TensorPair,
    reference: TensorPair,
    *,
    signature: dict,
    reference_id: str,
    rtol: float,
    atol: float,
    reductions: dict[str, ReductionReference] | None = None,
    mathematical_reference: TensorPair | None = None,
    numerical_controls: bool = False,
) -> dict:
    """Check every output/gradient against an independently computed reference.

    Both tuples contain output and input-gradient dictionaries, as returned by
    the replay harness. Key, shape and dtype sets must agree. Nonfinite values
    fail even if both implementations return the same infinity or NaN. Relative
    error excludes zero references; their absolute error still participates in
    ``abs(actual-reference) <= atol + rtol*abs(reference)``. Explicit reduction
    entries use a componentwise accumulation budget AND the original tolerances
    as an L2 guard. Mathematical-reference diagnostics do not replace these gates.
    """
    if not reference_id or any(not math.isfinite(value) or value < 0 for value in (rtol, atol)):
        raise ValueError("A reference ID and explicit finite nonnegative tolerances are required")
    with observe_check(
        "reference",
        signature,
        {
            "reference_id": reference_id,
            "rtol": rtol,
            "atol": atol,
            "numerical_controls_required": numerical_controls,
        },
    ) as check:
        metrics: dict[str, dict] = {}
        check["metrics"] = metrics
        reductions = reductions or {}
        keys = {
            f"{category}:{name}"
            for category, values in zip(("output", "gradient"), reference)
            for name in values
        }
        if reductions.keys() - keys:
            raise ValueError("Reduction contracts contain unknown tensor keys")
        if not actual[0] or (signature.get("phase") == "forward_backward" and not actual[1]):
            raise AssertionError(
                "Reference validation requires outputs and every backward input gradient"
            )
        for category, observed, expected in zip(("output", "gradient"), actual, reference):
            if observed.keys() != expected.keys():
                raise AssertionError(
                    f"Different {category} tensor keys: {sorted(observed)} vs {sorted(expected)}"
                )
            for name, tensor in observed.items():
                key = f"{category}:{name}"
                metrics[key] = _reference_metrics(
                    tensor, expected[name], rtol, atol, reductions.get(key)
                )
        check.update(compared_outputs=len(actual[0]), compared_gradients=len(actual[1]))
        if mathematical_reference is not None:
            diagnostics: dict[str, dict] = {}
            check["mathematical_reference_metrics"] = diagnostics
            for category, observed, expected in zip(
                ("output", "gradient"), actual, mathematical_reference
            ):
                if observed.keys() != expected.keys():
                    raise ValueError("Mathematical reference tensor keys differ")
                for name, tensor in observed.items():
                    diagnostics[f"{category}:{name}"] = _error_metrics(
                        tensor.double(), expected[name].double(), rtol, atol
                    )
        failures = {name: row for name, row in metrics.items() if _failed(row)}
        if failures:
            raise AssertionError(f"Independent reference {reference_id} differs: {failures}")
        if numerical_controls:
            controls: dict[str, dict] = {}
            check["numerical_controls"] = controls
            for category, observed, expected in zip(("output", "gradient"), actual, reference):
                for name, tensor in observed.items():
                    key = f"{category}:{name}"
                    controls[key] = _numerical_control(
                        tensor, expected[name], rtol, atol, reductions.get(key)
                    )
    return check


def assert_replay_sensitivity(
    actual: TensorPair, compare: Callable[[TensorPair], None], *, signature: dict
) -> dict:
    """Require the real replay comparator to detect a byte change in every tensor.

    First check the unchanged baseline, then flip one bit separately in each
    output and gradient. This checks comparator wiring; it is not evidence that
    a real scheduling race was exercised. No synthetic mismatch is recorded as
    kernel nondeterminism.
    """
    with observe_check(
        "sensitivity", signature, {"control": "one_byte_bit_flip_per_tensor"}
    ) as check:
        if not actual[0] or (signature.get("phase") == "forward_backward" and not actual[1]):
            raise AssertionError("Sensitivity validation requires outputs and backward gradients")
        compare(actual)
        detected = 0
        for category, tensors in enumerate(actual):
            for name, tensor in tensors.items():
                if not tensor.numel():
                    raise AssertionError("Cannot perturb an empty tensor")
                changed = tensor.detach().contiguous().clone()
                raw = changed.reshape(-1).view(torch.uint8)
                raw[0] ^= 1
                perturbed = (dict(actual[0]), dict(actual[1]))
                perturbed[category][name] = changed
                try:
                    compare(perturbed)
                except ReplayMismatch:
                    detected += 1
                else:
                    raise AssertionError(
                        f"Replay comparator missed the perturbation in {category}:{name}"
                    )
        check.update(
            compared_outputs=len(actual[0]),
            compared_gradients=len(actual[1]),
            detected_perturbations=detected,
        )
    return check
