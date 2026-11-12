# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

#!/usr/bin/env python3
"""Predict isolated TE packed-attention backward time from frozen measurements.

The timed operation is core-attention backward only: dQ, dK, and dV. Forward
graph construction, QKV/output projections, RoPE, MLP/MoE, optimizer work, and
pipeline/data-parallel synchronization are outside this model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent

from ..cp_round_geometry import build_round_geometries, critical_rank_work


DEFAULT_ARTIFACT = ROOT / "frozen_backward_attention_model.json"
# Ring geometry is defined for every positive integer CP through the measured
# topology limit. Frozen artifacts still expose only their measured branches;
# a missing branch is rejected separately in ``predict_with_details``.
SUPPORTED_CP_SIZES = tuple(range(1, 65))
SUPPORTED_STORAGE_TOKENS = (4096, 8192, 16384)
DEFAULT_SEQUENCE_ALIGNMENT = 128
GB200_D256_CP1_MINIMUM_SEQUENCE_TOKENS = 256


def _positive_lengths(values: Sequence[int]) -> tuple[int, ...]:
    lengths = tuple(int(value) for value in values)
    if not lengths or min(lengths) <= 0:
        raise ValueError("sequence_lengths must contain positive integers")
    return lengths


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def resolve_sequence_alignment(
    cp_size: int,
    calibration_alignment: int,
    *,
    max_cp_size: int | None = None,
    sequence_alignment: int | None = None,
) -> tuple[int, int]:
    """Resolve PR5541 per-sequence alignment and scheduler maximum CP size."""
    cp_size = int(cp_size)
    if cp_size not in SUPPORTED_CP_SIZES:
        raise ValueError(f"unsupported CP size {cp_size}")
    if max_cp_size is not None and sequence_alignment is not None:
        raise ValueError("provide max_cp_size or sequence_alignment, not both")
    if sequence_alignment is None:
        if max_cp_size is None:
            alignment = int(calibration_alignment)
            if alignment % 2:
                raise ValueError("calibration sequence alignment must be even")
            resolved_max_cp = alignment // 2
        else:
            resolved_max_cp = int(max_cp_size)
            alignment = 2 * resolved_max_cp
    else:
        alignment = int(sequence_alignment)
        if alignment % 2:
            raise ValueError("sequence_alignment must be even")
        resolved_max_cp = alignment // 2
    if resolved_max_cp not in SUPPORTED_CP_SIZES or resolved_max_cp < cp_size:
        raise ValueError(
            f"max_cp_size must be in {SUPPORTED_CP_SIZES} and >= cp_size"
        )
    if alignment % (2 * cp_size):
        raise ValueError("sequence_alignment must be divisible by 2 * cp_size")
    return resolved_max_cp, alignment


def branch_key(
    hardware: str,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_tokens_per_rank: int,
    cp_size: int,
) -> str:
    return ":".join(
        str(value)
        for value in (
            hardware,
            int(query_heads),
            int(kv_heads),
            int(head_dim),
            int(physical_tokens_per_rank),
            int(cp_size),
        )
    )


def fixed_capacity_layout(
    sequence_lengths: Sequence[int],
    cp_size: int,
    physical_tokens_per_rank: int,
    sequence_alignment: int = DEFAULT_SEQUENCE_ALIGNMENT,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return logical and PR5541 padded lengths for one fixed-capacity pack."""
    lengths = _positive_lengths(sequence_lengths)
    cp_size = int(cp_size)
    storage = int(physical_tokens_per_rank)
    alignment = int(sequence_alignment)
    if cp_size not in SUPPORTED_CP_SIZES:
        raise ValueError(f"unsupported CP size {cp_size}")
    if storage not in SUPPORTED_STORAGE_TOKENS:
        raise ValueError(f"unsupported physical_tokens_per_rank {storage}")
    if alignment <= 0 or alignment % (2 * cp_size):
        raise ValueError("sequence_alignment must be divisible by 2 * cp_size")

    capacity = cp_size * storage
    padded = [_round_up(length, alignment) for length in lengths]
    if sum(padded) > capacity:
        raise ValueError(
            f"aligned pack requires {sum(padded)} tokens, capacity is {capacity}"
        )
    # Tail padding extends the final real sequence's physical boundary. It is
    # exchanged by CP but cu_seqlens keeps it outside attention work.
    padded[-1] += capacity - sum(padded)
    return lengths, tuple(padded)


def _coefficient_of_variation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def extract_backward_features(
    sequence_lengths: Sequence[int],
    cp_size: int,
    physical_tokens_per_rank: int,
    sequence_alignment: int = DEFAULT_SEQUENCE_ALIGNMENT,
) -> dict[str, float]:
    """Build phase-specific features from valid work and exact CP round geometry."""
    lengths, padded = fixed_capacity_layout(
        sequence_lengths,
        cp_size,
        physical_tokens_per_rank,
        sequence_alignment,
    )
    capacity = int(cp_size) * int(physical_tokens_per_rank)
    aligned_before_tail = tuple(
        _round_up(length, int(sequence_alignment)) for length in lengths
    )
    logical_s2 = float(sum(length * length for length in lengths))
    alignment_work = float(
        sum(aligned * length for aligned, length in zip(aligned_before_tail, lengths))
    )
    valid_tokens = float(sum(lengths))
    aligned_tokens = float(sum(aligned_before_tail))
    has_physical_padding = any(
        logical != physical for logical, physical in zip(lengths, padded)
    )
    count = len(lengths)
    mean_length = valid_tokens / count
    max_length = max(lengths)
    max_length_power_of_two = 1 << (max_length - 1).bit_length()
    sequence_length_powers_of_two = tuple(
        1 << (length - 1).bit_length() for length in lengths
    )
    unfused_dense_s2 = float(count * max_length**2)
    unfused_dense_tokens = float(count * max_length)
    length_cv = _coefficient_of_variation([float(value) for value in lengths])

    features = {
        "logical_s2_m": logical_s2 / 1.0e6,
        "unfused_dense_s2_m": unfused_dense_s2 / 1.0e6,
        "unfused_dense_tokens_k": unfused_dense_tokens / 1024.0,
        "unfused_logical_work_fraction": logical_s2 / unfused_dense_s2,
        "unfused_valid_token_fraction": valid_tokens / unfused_dense_tokens,
        "alignment_work_m": alignment_work / 1.0e6,
        "alignment_squared_m": (
            sum(aligned * aligned for aligned in aligned_before_tail) / 1.0e6
        ),
        "alignment_padding_work_m": (alignment_work - logical_s2) / 1.0e6,
        "power_of_two_work_m": (
            sum(
                length * power
                for length, power in zip(lengths, sequence_length_powers_of_two)
            )
            / 1.0e6
        ),
        "power_of_two_squared_m": (
            sum(power * power for power in sequence_length_powers_of_two) / 1.0e6
        ),
        "max_length_valid_work_m": max_length * valid_tokens / 1.0e6,
        "max_length_aligned_work_m": max_length * aligned_tokens / 1.0e6,
        "valid_tokens_k": valid_tokens / 1024.0,
        "valid_tokens_per_rank_k": valid_tokens / cp_size / 1024.0,
        "aligned_tokens_k": aligned_tokens / 1024.0,
        "tail_padding_k": (capacity - aligned_tokens) / 1024.0,
        "capacity_utilization": valid_tokens / capacity,
        "aligned_capacity_utilization": aligned_tokens / capacity,
        "has_physical_padding": float(has_physical_padding),
        "exact_unpadded_capacity": float(not has_physical_padding),
        "sequence_count": float(count),
        "sqrt_sequence_count": math.sqrt(count),
        "log_sequence_count": math.log1p(count),
        "mean_length_k": mean_length / 1024.0,
        "max_length_k": max_length / 1024.0,
        "min_length_k": min(lengths) / 1024.0,
        "max_length_fraction": max_length / capacity,
        # TE's fused masked-softmax predicate uses exact 8-token divisibility
        # and a next-power-of-two batch-per-block calculation.
        "softmax_max_length_pow2_k": max_length_power_of_two / 1024.0,
        "softmax_max_to_pow2": max_length / max_length_power_of_two,
        "softmax_max_mod8_fraction": (max_length % 8) / 8.0,
        "length_cv": length_cv,
    }

    if cp_size == 1:
        features.update(
            {
                "critical_rank_work_m": logical_s2 / 1.0e6,
                "mean_rank_work_m": logical_s2 / 1.0e6,
                "rank_work_cv": 0.0,
                "round_critical_work_m": logical_s2 / 1.0e6,
                "round_critical_sqrt_work": math.sqrt(logical_s2) / 1000.0,
                "critical_rank_sqrt_work": math.sqrt(logical_s2) / 1000.0,
                "max_round_work_m": logical_s2 / 1.0e6,
                "mean_active_sequences": float(count),
                "max_active_sequences": float(count),
                "active_round_density": 1.0,
            }
        )
        return features

    rounds = build_round_geometries(lengths, padded, cp_size)
    rank_work_map = critical_rank_work(rounds)
    rank_work = [float(rank_work_map[rank]) for rank in range(cp_size)]
    work_by_step: dict[int, list[float]] = {step: [] for step in range(cp_size)}
    active_by_step: dict[int, list[float]] = {step: [] for step in range(cp_size)}
    sqrt_work_by_rank: dict[int, float] = {rank: 0.0 for rank in range(cp_size)}
    active_rounds = 0
    for item in rounds:
        work = float(item.cp1_equivalent_sum_squared_tokens)
        work_by_step[item.step].append(work)
        active_by_step[item.step].append(float(item.active_sequences))
        sqrt_work_by_rank[item.rank] += math.sqrt(work)
        active_rounds += int(work > 0.0)

    critical_by_step = [max(work_by_step[step]) for step in range(cp_size)]
    active_by_round = [max(active_by_step[step]) for step in range(cp_size)]
    features.update(
        {
            "critical_rank_work_m": max(rank_work) / 1.0e6,
            "mean_rank_work_m": (sum(rank_work) / cp_size) / 1.0e6,
            "rank_work_cv": _coefficient_of_variation(rank_work),
            "round_critical_work_m": sum(critical_by_step) / 1.0e6,
            "round_critical_sqrt_work": (
                sum(math.sqrt(value) for value in critical_by_step) / 1000.0
            ),
            "critical_rank_sqrt_work": max(sqrt_work_by_rank.values()) / 1000.0,
            "max_round_work_m": max(critical_by_step) / 1.0e6,
            "mean_active_sequences": sum(active_by_round) / cp_size,
            "max_active_sequences": max(active_by_round),
            "active_round_density": active_rounds / (cp_size * cp_size),
        }
    )
    return features


def unfused_softmax_uses_fused_kernel(
    sequence_lengths: Sequence[int], query_heads: int
) -> bool:
    """Mirror TE 2.14.1 masked-softmax kernel availability for padded THD."""
    lengths = _positive_lengths(sequence_lengths)
    max_length = max(lengths)
    attention_batches = len(lengths) * int(query_heads)
    if not 16 < max_length < 16384:
        return False
    if max_length % 8 or max_length % 4 or attention_batches % 4:
        return False
    power_of_two = 1 << (max_length - 1).bit_length()
    warp_size = min(power_of_two, 32)
    batches_per_warp = 2 if power_of_two <= 128 else 1
    batches_per_block = (128 // warp_size) * batches_per_warp
    return max_length % batches_per_block == 0


def backward_backend_state(
    features: dict[str, float],
    *,
    sequence_lengths: Sequence[int],
    hardware: str,
    cp_size: int,
    query_heads: int,
    head_dim: int,
) -> str:
    """Return the measured TE layout and inner-softmax execution state."""
    if features["exact_unpadded_capacity"] == 1.0:
        return "exact_unpadded_capacity"
    if hardware == "gb200" and int(cp_size) == 1 and int(head_dim) == 256:
        suffix = (
            "fused_softmax"
            if unfused_softmax_uses_fused_kernel(sequence_lengths, query_heads)
            else "torch_softmax"
        )
        return f"padded_unfused_{suffix}"
    return "padded"


def _evaluate_branch(model: dict[str, Any], features: dict[str, float]) -> float:
    if model.get("model_kind") == "gradient_boosted_regression_trees":
        names = model["tree_feature_names"]
        values = [float(features[name]) for name in names]
        prediction = float(model["tree_initial_prediction_ms"])
        learning_rate = float(model["tree_learning_rate"])
        for tree in model["trees"]:
            left = tree["children_left"]
            right = tree["children_right"]
            split_features = tree["feature"]
            thresholds = tree["threshold"]
            outputs = tree["value"]
            size = len(left)
            if not (
                len(right)
                == len(split_features)
                == len(thresholds)
                == len(outputs)
                == size
            ):
                raise ValueError("gradient-boosting tree arrays have mixed sizes")
            node = 0
            while int(left[node]) != int(right[node]):
                feature_index = int(split_features[node])
                node = (
                    int(left[node])
                    if values[feature_index] <= float(thresholds[node])
                    else int(right[node])
                )
                if node < 0 or node >= size:
                    raise ValueError("gradient-boosting tree has an invalid child")
            prediction += learning_rate * float(outputs[node])
        return max(float(model["prediction_floor_ms"]), prediction)

    base_names = model["base_feature_names"]
    base_values = np.asarray([features[name] for name in base_names], dtype=float)
    base_center = np.asarray(model["base_feature_center"], dtype=float)
    base_scale = np.asarray(model["base_feature_scale"], dtype=float)
    base_coefficients = np.asarray(model["base_coefficients"], dtype=float)
    standardized = (base_values - base_center) / base_scale
    prediction = float(base_coefficients[0] + standardized @ base_coefficients[1:])

    if model.get("kernel_centers"):
        kernel_names = model["kernel_feature_names"]
        kernel_values = np.asarray(
            [features[name] for name in kernel_names], dtype=float
        )
        center = np.asarray(model["kernel_feature_center"], dtype=float)
        scale = np.asarray(model["kernel_feature_scale"], dtype=float)
        z = (kernel_values - center) / scale
        anchors = np.asarray(model["kernel_centers"], dtype=float)
        distance_squared = np.mean((anchors - z) ** 2, axis=1)
        weights = np.exp(-float(model["kernel_gamma"]) * distance_squared)
        prediction += float(
            weights @ np.asarray(model["kernel_coefficients"], dtype=float)
        )
    return max(float(model["prediction_floor_ms"]), prediction)


def _interpolate_points(
    points: Sequence[Sequence[float]], value: float
) -> tuple[float, str]:
    ordered = sorted((float(x), float(y)) for x, y in points)
    if not ordered:
        raise ValueError("sequence-count correction curve is empty")
    if len({x for x, _ in ordered}) != len(ordered):
        raise ValueError("sequence-count correction curve has duplicate coordinates")
    for x, y in ordered:
        if value == x:
            return y, "exact"
    if value < ordered[0][0]:
        return ordered[0][1], "clamped_low"
    if value > ordered[-1][0]:
        return ordered[-1][1], "clamped_high"
    for (left_x, left_y), (right_x, right_y) in zip(ordered, ordered[1:]):
        if left_x < value < right_x:
            fraction = (value - left_x) / (right_x - left_x)
            return left_y + fraction * (right_y - left_y), "interpolated"
    raise AssertionError("correction interpolation interval was not found")


def _evaluate_sequence_count_correction(
    model: dict[str, Any],
    features: dict[str, float],
    *,
    base_time_ms: float | None = None,
) -> tuple[float, dict[str, Any] | None]:
    correction = model.get("sequence_count_correction")
    if not correction:
        return 0.0, None
    if correction.get("kind") != (
        "backend_pow2_max_length_sequence_count_linear_interpolation"
    ):
        raise ValueError("unsupported sequence-count correction kind")
    power = int(round(float(features["softmax_max_length_pow2_k"]) * 1024.0))
    curves = correction.get("power_of_two_curves", {}).get(str(power))
    if not curves:
        return 0.0, {
            "applied": False,
            "reason": "unmeasured_power_of_two_specialization",
            "maximum_sequence_power_of_two": power,
        }

    sequence_count = float(features["sequence_count"])
    maximum_sequence_tokens = float(features["max_length_k"]) * 1024.0
    values_by_maximum = []
    count_states = []
    for curve in curves:
        ratio, state = _interpolate_points(
            curve["sequence_count_points"], sequence_count
        )
        values_by_maximum.append(
            (float(curve["maximum_sequence_tokens"]), ratio)
        )
        count_states.append(state)
    raw_ratio, maximum_state = _interpolate_points(
        values_by_maximum, maximum_sequence_tokens
    )
    target = correction.get("target")
    if target == "multiplicative_residual_ratio_on_base_time_ms":
        lower, upper = (float(value) for value in correction["ratio_clip"])
        applied_value = min(upper, max(lower, raw_ratio))
        applied_ratio = applied_value
        raw_name = "raw_ratio"
    elif target == "additive_residual_ms_on_base_time_ms":
        if base_time_ms is None or base_time_ms <= 0.0:
            raise ValueError("additive sequence-count correction needs base_time_ms")
        lower, upper = (
            float(value) for value in correction["residual_clip_ms"]
        )
        applied_value = min(upper, max(lower, raw_ratio))
        applied_ratio = applied_value / base_time_ms
        raw_name = "raw_residual_ms"
    else:
        raise ValueError("unsupported sequence-count correction target")
    return applied_ratio, {
        "applied": True,
        "kind": correction["kind"],
        "target": target,
        "maximum_sequence_power_of_two": power,
        "sequence_count_interpolation": sorted(set(count_states)),
        "maximum_length_interpolation": maximum_state,
        raw_name: raw_ratio,
        "applied_value": applied_value,
        "applied_ratio": applied_ratio,
        "value_was_clipped": applied_value != raw_ratio,
        "ratio_was_clipped": applied_value != raw_ratio,
    }


def _evaluate_singleton_time_curve(
    model: dict[str, Any], features: dict[str, float]
) -> tuple[float | None, dict[str, Any] | None]:
    curve = model.get("singleton_time_curve")
    if not curve:
        return None, None
    if curve.get("kind") != "piecewise_linear_measured_time_ms_by_length":
        raise ValueError("unsupported singleton time curve kind")
    if int(round(float(features["sequence_count"]))) != 1:
        return None, {"applied": False, "reason": "pack_is_not_singleton"}
    points = curve.get("points", ())
    if not points:
        raise ValueError("singleton time curve is empty")
    length = float(features["max_length_k"]) * 1024.0
    lower = min(float(point[0]) for point in points)
    upper = max(float(point[0]) for point in points)
    if not lower <= length <= upper:
        return None, {
            "applied": False,
            "reason": "length_outside_measured_curve",
            "measured_length_range": [lower, upper],
        }
    time_ms, state = _interpolate_points(points, length)
    return time_ms, {
        "applied": True,
        "kind": curve["kind"],
        "length_tokens": length,
        "interpolation": state,
        "measured_length_range": [lower, upper],
    }


def _evaluate_sequence_count_time_curve(
    model: dict[str, Any], features: dict[str, float]
) -> tuple[float | None, dict[str, Any] | None]:
    curves = model.get("sequence_count_time_curves")
    if not curves:
        return None, None
    if curves.get("kind") != (
        "piecewise_linear_measured_time_ms_by_max_length_and_sequence_count"
    ):
        raise ValueError("unsupported sequence-count time curve kind")
    sequence_count = int(round(float(features["sequence_count"])))
    points = curves.get("curves", {}).get(str(sequence_count))
    if not points:
        return None, {
            "applied": False,
            "reason": "sequence_count_has_no_measured_curve",
            "sequence_count": sequence_count,
        }
    maximum_length = float(features["max_length_k"]) * 1024.0
    lower = min(float(point[0]) for point in points)
    upper = max(float(point[0]) for point in points)
    if not lower <= maximum_length <= upper:
        return None, {
            "applied": False,
            "reason": "maximum_length_outside_measured_curve",
            "sequence_count": sequence_count,
            "measured_maximum_length_range": [lower, upper],
        }
    time_ms, state = _interpolate_points(points, maximum_length)
    return time_ms, {
        "applied": True,
        "kind": curves["kind"],
        "sequence_count": sequence_count,
        "maximum_length_tokens": maximum_length,
        "interpolation": state,
        "measured_maximum_length_range": [lower, upper],
    }


class FrozenBackwardAttentionPredictor:
    """Evaluate one immutable backward-attention calibration artifact."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported backward model schema")
        contract = payload.get("contract", {})
        if contract.get("phase") != "backward":
            raise ValueError("artifact is not a backward-only model")
        self.payload = payload
        self.sequence_alignment = int(contract["sequence_alignment"])
        self.gb200_d256_cp1_minimum_sequence_tokens = int(
            contract.get("gb200_d256_cp1_minimum_sequence_tokens", 1)
        )
        self.branches = payload["branches"]

    @classmethod
    def from_path(cls, path: Path | str = DEFAULT_ARTIFACT) -> "FrozenBackwardAttentionPredictor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict_with_details(
        self,
        sequence_lengths: Sequence[int],
        *,
        hardware: str,
        cp_size: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        physical_tokens_per_rank: int,
        max_cp_size: int | None = None,
        sequence_alignment: int | None = None,
    ) -> dict[str, Any]:
        key = branch_key(
            hardware,
            query_heads,
            kv_heads,
            head_dim,
            physical_tokens_per_rank,
            cp_size,
        )
        if key not in self.branches:
            supported = sorted(self.branches)
            raise ValueError(f"no measured backward branch {key}; supported: {supported}")
        resolved_max_cp, resolved_alignment = resolve_sequence_alignment(
            cp_size,
            self.sequence_alignment,
            max_cp_size=max_cp_size,
            sequence_alignment=sequence_alignment,
        )
        features = extract_backward_features(
            sequence_lengths,
            cp_size,
            physical_tokens_per_rank,
            resolved_alignment,
        )
        model = self.branches[key]
        backend_state = backward_backend_state(
            features,
            sequence_lengths=sequence_lengths,
            hardware=hardware,
            cp_size=cp_size,
            query_heads=query_heads,
            head_dim=head_dim,
        )
        minimum_sequence_tokens = 1
        if hardware == "gb200" and int(cp_size) == 1 and int(head_dim) == 256:
            minimum_sequence_tokens = (
                self.gb200_d256_cp1_minimum_sequence_tokens
            )
        if min(int(length) for length in sequence_lengths) < minimum_sequence_tokens:
            raise ValueError(
                "GB200 head-dim-256 CP1 backward requires every sequence "
                f"to contain at least {minimum_sequence_tokens} tokens"
            )
        supported_states = tuple(
            model.get(
                "supported_backend_states",
                ("exact_unpadded_capacity", "padded"),
            )
        )
        if backend_state not in supported_states:
            raise ValueError(
                f"backward branch {key} does not support backend state "
                f"{backend_state}; measured states: {supported_states}"
            )
        selected_model = model.get("backend_state_models", {}).get(
            backend_state, model
        )
        empirical_domain = selected_model["feature_domain"]
        supported_domain = selected_model.get(
            "supported_feature_domain", empirical_domain
        )
        outside_empirical = [
            name
            for name, bounds in empirical_domain.items()
            if features[name] < float(bounds[0]) or features[name] > float(bounds[1])
        ]
        outside_supported = [
            name
            for name, bounds in supported_domain.items()
            if features[name] < float(bounds[0]) or features[name] > float(bounds[1])
        ]
        if resolved_alignment != self.sequence_alignment:
            outside_empirical.append("sequence_alignment")
            outside_supported.append("sequence_alignment")
        calibration_box = "inside"
        if outside_supported:
            calibration_box = "outside_supported_domain"
        elif outside_empirical:
            calibration_box = "supported_boundary"
        base_time_ms = _evaluate_branch(selected_model, features)
        correction_ratio, correction_details = _evaluate_sequence_count_correction(
            selected_model, features, base_time_ms=base_time_ms
        )
        time_ms = max(
            float(selected_model["prediction_floor_ms"]),
            base_time_ms * (1.0 + correction_ratio),
        )
        sequence_count_time_ms, sequence_count_time_details = (
            _evaluate_sequence_count_time_curve(selected_model, features)
        )
        if sequence_count_time_ms is not None:
            time_ms = max(
                float(selected_model["prediction_floor_ms"]),
                sequence_count_time_ms,
            )
        singleton_time_ms, singleton_details = _evaluate_singleton_time_curve(
            selected_model, features
        )
        if singleton_time_ms is not None:
            time_ms = max(
                float(selected_model["prediction_floor_ms"]),
                singleton_time_ms,
            )
        return {
            "time_ms": time_ms,
            "base_time_ms": base_time_ms,
            "sequence_count_correction_ratio": correction_ratio,
            "sequence_count_correction": correction_details,
            "sequence_count_time_curve": sequence_count_time_details,
            "singleton_time_curve": singleton_details,
            "phase": "backward",
            "timed_operation": "core_attention_dq_dk_dv",
            "hardware": hardware,
            "cp_size": int(cp_size),
            "geometry": {
                "query_heads": int(query_heads),
                "kv_heads": int(kv_heads),
                "head_dim": int(head_dim),
            },
            "physical_tokens_per_rank": int(physical_tokens_per_rank),
            "max_cp_size": resolved_max_cp,
            "sequence_alignment": resolved_alignment,
            "calibration_sequence_alignment": self.sequence_alignment,
            "sequence_count": len(sequence_lengths),
            "minimum_sequence_tokens": minimum_sequence_tokens,
            "backend_state": backend_state,
            "supported_backend_states": list(supported_states),
            "features": features,
            "calibration_box": calibration_box,
            "outside_calibration_features": outside_empirical,
            "outside_supported_domain_features": outside_supported,
            "model_key": key,
            "backend_state_model_kind": selected_model["model_kind"],
        }

    def predict_time_ms(self, sequence_lengths: Sequence[int], **kwargs: Any) -> float:
        return float(self.predict_with_details(sequence_lengths, **kwargs)["time_ms"])


@lru_cache(maxsize=1)
def load_default_predictor() -> FrozenBackwardAttentionPredictor:
    return FrozenBackwardAttentionPredictor.from_path(DEFAULT_ARTIFACT)


def predict_backward_attention_ms(
    sequence_lengths: Sequence[int],
    *,
    hardware: str,
    cp_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_tokens_per_rank: int,
    max_cp_size: int | None = None,
    sequence_alignment: int | None = None,
) -> float:
    return load_default_predictor().predict_time_ms(
        sequence_lengths,
        hardware=hardware,
        cp_size=cp_size,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        physical_tokens_per_rank=physical_tokens_per_rank,
        max_cp_size=max_cp_size,
        sequence_alignment=sequence_alignment,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--hardware", choices=("h100", "gb200"), required=True)
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--physical-tokens-per-rank", type=int, required=True)
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument("--max-cp-size", type=int)
    alignment.add_argument("--sequence-alignment", type=int)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FrozenBackwardAttentionPredictor.from_path(args.artifact)
    details = predictor.predict_with_details(
        [int(value) for value in args.lengths.split(",") if value],
        hardware=args.hardware,
        cp_size=args.cp_size,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        physical_tokens_per_rank=args.physical_tokens_per_rank,
        max_cp_size=args.max_cp_size,
        sequence_alignment=args.sequence_alignment,
    )
    print(json.dumps(details, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
