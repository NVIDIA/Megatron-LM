# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

#!/usr/bin/env python3
"""Predict provisional H100 Qwen3-30B backward attention at integer CP sizes.

Power-of-two branches are evaluated from the frozen measured backward model.
For a non-power CP, logical lengths are scaled to the adjacent power anchors
while preserving each sequence's physical tokens per rank. The two anchor
times are linearly interpolated in CP size and multiplied by one frozen scale.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BACKWARD_ROOT = HERE.parent / "backward_model"

from ..backward_model.backward_attention_predictor import (  # noqa: E402
    FrozenBackwardAttentionPredictor,
    _evaluate_branch,
    _evaluate_sequence_count_correction,
    _evaluate_sequence_count_time_curve,
    _evaluate_singleton_time_curve,
    backward_backend_state,
    branch_key,
    extract_backward_features,
)


SUPPORTED_CP_SIZES = tuple(range(1, 65))
POWER_CP_SIZES = (1, 2, 4, 8, 16, 32, 64)


def _round_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _anchor_pair(cp_size: int) -> tuple[int, int]:
    lower = 1 << (int(cp_size).bit_length() - 1)
    return lower, 2 * lower


def scale_pack_to_anchor(
    lengths: Sequence[int],
    *,
    cp_size: int,
    anchor_cp: int,
    sequence_alignment: int,
) -> tuple[int, ...]:
    """Preserve per-rank physical sequence boundaries at one power anchor."""
    result = []
    target_unit = 2 * int(cp_size)
    anchor_unit = 2 * int(anchor_cp)
    for length in lengths:
        physical = _round_up(int(length), int(sequence_alignment))
        if physical % target_unit:
            raise ValueError("target sequence boundary is not divisible by 2 * CP")
        anchor_boundary = physical // target_unit * anchor_unit
        scaled = max(1, round(int(length) * int(anchor_cp) / int(cp_size)))
        # Keep the scaled valid length inside the exact mapped physical boundary.
        scaled = min(anchor_boundary, max(anchor_boundary - anchor_unit + 1, scaled))
        if _round_up(scaled, anchor_unit) != anchor_boundary:
            raise AssertionError("anchor sequence boundary was not preserved")
        result.append(scaled)
    return tuple(result)


def _evaluate_power_branch(
    base: FrozenBackwardAttentionPredictor,
    lengths: Sequence[int],
    *,
    cp_size: int,
    sequence_alignment: int,
) -> float:
    features = extract_backward_features(
        lengths, cp_size, 4096, sequence_alignment
    )
    key = branch_key("h100", 32, 4, 128, 4096, cp_size)
    model = base.branches[key]
    state = backward_backend_state(
        features,
        sequence_lengths=lengths,
        hardware="h100",
        cp_size=cp_size,
        query_heads=32,
        head_dim=128,
    )
    selected = model.get("backend_state_models", {}).get(state, model)
    base_time = _evaluate_branch(selected, features)
    correction, _ = _evaluate_sequence_count_correction(
        selected, features, base_time_ms=base_time
    )
    time_ms = max(
        float(selected["prediction_floor_ms"]),
        base_time * (1.0 + correction),
    )
    count_time, _ = _evaluate_sequence_count_time_curve(selected, features)
    if count_time is not None:
        time_ms = max(float(selected["prediction_floor_ms"]), count_time)
    singleton_time, _ = _evaluate_singleton_time_curve(selected, features)
    if singleton_time is not None:
        time_ms = max(float(selected["prediction_floor_ms"]), singleton_time)
    return float(time_ms)


class ArbitraryCPBackwardPredictor:
    """Evaluate one frozen non-power backward candidate artifact."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported arbitrary-CP backward schema")
        contract = payload.get("contract", {})
        if contract.get("phase") != "backward":
            raise ValueError("artifact is not backward-only")
        if contract.get("hardware") != "h100":
            raise ValueError("candidate currently supports H100 only")
        geometry = contract.get("attention_geometry", {})
        geometry_key = (
            geometry.get("query_heads"),
            geometry.get("kv_heads"),
            geometry.get("head_dim"),
        )
        if geometry_key != (32, 4, 128):
            raise ValueError("candidate currently supports Qwen3-30B geometry only")
        artifact_cp_sizes = tuple(int(value) for value in contract.get("cp_sizes", ()))
        if not artifact_cp_sizes or any(
            value not in SUPPORTED_CP_SIZES for value in artifact_cp_sizes
        ):
            raise ValueError("artifact has an invalid CP-size contract")
        base_path = Path(payload["base_artifact"])
        if not base_path.is_absolute():
            base_path = PROJECT_ROOT / base_path
        self.payload = payload
        self.base = FrozenBackwardAttentionPredictor.from_path(base_path)
        self.non_power_scale = float(payload["model"]["non_power_time_scale"])
        self.supported_cp_sizes = frozenset(artifact_cp_sizes)

    @classmethod
    def from_path(cls, path: Path | str) -> "ArbitraryCPBackwardPredictor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict_with_details(
        self,
        sequence_lengths: Sequence[int],
        cp_size: int,
        *,
        sequence_alignment: int | None = None,
    ) -> dict[str, Any]:
        lengths = tuple(int(value) for value in sequence_lengths)
        cp = int(cp_size)
        if cp not in self.supported_cp_sizes:
            raise ValueError("cp_size is outside this artifact's contract")
        if not lengths or min(lengths) <= 0:
            raise ValueError("sequence_lengths must contain positive integers")
        alignment = 2 * cp if sequence_alignment is None else int(sequence_alignment)
        if alignment <= 0 or alignment % (2 * cp):
            raise ValueError("sequence_alignment must be divisible by 2 * cp_size")
        physical = sum(_round_up(value, alignment) for value in lengths)
        if physical > cp * 4096:
            raise ValueError("aligned pack exceeds CP capacity")

        if _is_power_of_two(cp):
            raw_time = _evaluate_power_branch(
                self.base,
                lengths,
                cp_size=cp,
                sequence_alignment=alignment,
            )
            return {
                "time_ms": raw_time,
                "raw_time_ms": raw_time,
                "cp_size": cp,
                "sequence_alignment": alignment,
                "method": "measured_power_branch",
                "anchor_cp_sizes": [cp],
                "non_power_time_scale": 1.0,
                "provisional": alignment != int(self.base.sequence_alignment),
            }

        lower, upper = _anchor_pair(cp)
        lower_lengths = scale_pack_to_anchor(
            lengths,
            cp_size=cp,
            anchor_cp=lower,
            sequence_alignment=alignment,
        )
        upper_lengths = scale_pack_to_anchor(
            lengths,
            cp_size=cp,
            anchor_cp=upper,
            sequence_alignment=alignment,
        )
        lower_time = _evaluate_power_branch(
            self.base,
            lower_lengths,
            cp_size=lower,
            sequence_alignment=2 * lower,
        )
        upper_time = _evaluate_power_branch(
            self.base,
            upper_lengths,
            cp_size=upper,
            sequence_alignment=2 * upper,
        )
        weight = (cp - lower) / (upper - lower)
        raw_time = lower_time + weight * (upper_time - lower_time)
        return {
            "time_ms": raw_time * self.non_power_scale,
            "raw_time_ms": raw_time,
            "cp_size": cp,
            "sequence_alignment": alignment,
            "method": "per_rank_geometry_scaled_adjacent_power_linear_interpolation",
            "anchor_cp_sizes": [lower, upper],
            "anchor_lengths": [list(lower_lengths), list(upper_lengths)],
            "anchor_times_ms": [lower_time, upper_time],
            "interpolation_weight": weight,
            "non_power_time_scale": self.non_power_scale,
            "provisional": True,
        }

    def predict_ms(
        self,
        sequence_lengths: Sequence[int],
        cp_size: int,
        **kwargs: Any,
    ) -> float:
        details = self.predict_with_details(sequence_lengths, cp_size, **kwargs)
        return float(details["time_ms"])
