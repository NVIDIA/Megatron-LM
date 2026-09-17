# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical reference diagnostics and byte-comparator sensitivity controls."""

from __future__ import annotations

import math
from typing import Callable

import torch

from tools.determinism.checks import observe_check
from tools.determinism.coverage import ReplayMismatch

TensorPair = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


def _error_metrics(actual: torch.Tensor, reference: torch.Tensor, rtol: float, atol: float) -> dict:
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise AssertionError("Reference and actual tensors have different shapes or dtypes")
    if not actual.numel() or not actual.is_floating_point():
        raise ValueError("Reference checks require nonempty real floating-point tensors")
    values, expected = actual.detach().reshape(-1).double(), reference.detach().reshape(-1).double()
    finite = values.isfinite() & expected.isfinite()
    absolute = (values - expected).abs()
    bad = ~finite | (absolute > atol + rtol * expected.abs())
    nonzero = finite & (expected != 0)
    severity = torch.where(finite, absolute, torch.full_like(absolute, float("inf")))
    if bool(bad.any()):
        severity = severity.masked_fill(~bad, -1)
    location = int(severity.argmax())

    def number_or_string(value: torch.Tensor) -> float | str:
        number = float(value)
        return number if math.isfinite(number) else str(number)

    return {
        "elements": actual.numel(),
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "violations": int(bad.sum()),
        "nonfinite_elements": int((~finite).sum()),
        "zero_reference_elements": int((finite & (expected == 0)).sum()),
        "max_absolute_error": (
            number_or_string(absolute[finite].max()) if bool(finite.any()) else None
        ),
        "max_relative_error": (
            number_or_string((absolute[nonzero] / expected[nonzero].abs()).max())
            if bool(nonzero.any())
            else None
        ),
        "sample": {
            "flat_index": location,
            "actual": number_or_string(values[location]),
            "reference": number_or_string(expected[location]),
        },
    }


def assert_reference_close(
    actual: TensorPair,
    reference: TensorPair,
    *,
    signature: dict,
    reference_id: str,
    rtol: float,
    atol: float,
) -> dict:
    """Check every output/gradient against an independently computed reference.

    Both tuples contain output and input-gradient dictionaries, as returned by
    the replay harness. Key, shape and dtype sets must agree. Nonfinite values
    fail even if both implementations return the same infinity or NaN. Relative
    error excludes zero references; their absolute error still participates in
    ``abs(actual-reference) <= atol + rtol*abs(reference)``.
    """
    if not reference_id or any(not math.isfinite(value) or value < 0 for value in (rtol, atol)):
        raise ValueError("A reference ID and explicit finite nonnegative tolerances are required")
    with observe_check(
        "reference", signature, {"reference_id": reference_id, "rtol": rtol, "atol": atol}
    ) as check:
        metrics: dict[str, dict] = {}
        check["metrics"] = metrics
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
                metrics[f"{category}:{name}"] = _error_metrics(tensor, expected[name], rtol, atol)
        check.update(compared_outputs=len(actual[0]), compared_gradients=len(actual[1]))
        failures = {name: row for name, row in metrics.items() if row["violations"]}
        if failures:
            raise AssertionError(f"Independent reference {reference_id} differs: {failures}")
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
