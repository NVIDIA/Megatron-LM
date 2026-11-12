# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

#!/usr/bin/env python3
"""GB200 CP1 predictor with explicit tail-padding backend-state dispatch."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from .cp1_gqa_attention_predictor import (
    FrozenCP1GQAAttentionPredictor,
    _length_features,
)


class GB200CP1AttentionPredictor:
    """Combine standard head-dim-128 models with split head-dim-256 models."""

    def __init__(
        self,
        standard_payload: dict[str, Any],
        backend_state_payload: dict[str, Any],
    ) -> None:
        self.standard = FrozenCP1GQAAttentionPredictor(standard_payload)
        contract = backend_state_payload["contract"]
        if contract["gpu"] != self.standard.gpu or contract["phase"] != "forward":
            raise ValueError("GB200 CP1 artifacts target different contracts")
        self.backend_payload = backend_state_payload
        self.gpu = self.standard.gpu
        self.minimum_total = self.standard.minimum_total
        self.maximum_total = self.standard.maximum_total
        self.minimum_sequence_tokens = self.standard.minimum_sequence_tokens
        self.minimum_sequence_count = self.standard.minimum_sequence_count
        self.maximum_sequence_count = self.standard.maximum_sequence_count
        self.sequence_alignment = self.standard.sequence_alignment
        self.calibration_max_cp_size = self.standard.calibration_max_cp_size
        self.pack_alignment = self.standard.pack_alignment
        self.models_by_geometry = dict(self.standard.models_by_geometry)
        profiles = {
            profile["name"]: profile for profile in backend_state_payload["profiles"]
        }
        self.backend_models_by_geometry = {}
        for name, model in backend_state_payload["models"].items():
            profile = profiles[name]
            geometry = (
                int(profile["q_heads"]),
                int(profile["kv_heads"]),
                int(profile["head_dim"]),
            )
            self.backend_models_by_geometry[geometry] = model
            self.models_by_geometry[geometry] = model

    @classmethod
    def from_paths(
        cls, standard_path: Path | str, backend_state_path: Path | str
    ) -> "GB200CP1AttentionPredictor":
        return cls(
            json.loads(Path(standard_path).read_text(encoding="utf-8")),
            json.loads(Path(backend_state_path).read_text(encoding="utf-8")),
        )

    def has_geometry(self, geometry: tuple[int, int, int]) -> bool:
        return geometry in self.models_by_geometry

    def sequence_alignment_for(self, max_cp_size: int | None = None) -> int:
        return self.standard.sequence_alignment_for(max_cp_size)

    def _predict_linear(
        self,
        model: dict[str, Any],
        lengths: Sequence[int],
        max_cp_size: int | None,
    ) -> float:
        include_legacy_tiles = any(
            str(name).startswith(("tile128", "tile256"))
            for name in model["features"]
        )
        features = _length_features(
            lengths,
            self.sequence_alignment_for(max_cp_size),
            include_legacy_tile_features=include_legacy_tiles,
        )
        value = float(model["coefficients"][0])
        for index, name in enumerate(model["features"]):
            standardized = (
                features[name] - float(model["feature_centers"][index])
            ) / float(model["feature_scales"][index])
            value += float(model["coefficients"][index + 1]) * standardized
        if model["kind"] == "linear_log":
            value = math.exp(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("GB200 backend-state model produced invalid time")
        return value

    def predict_time_ms(
        self,
        sequence_lengths: Sequence[int],
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        *,
        max_cp_size: int | None = None,
        allow_extrapolation: bool = False,
        storage_tokens: int | None = None,
    ) -> float:
        geometry = (int(query_heads), int(kv_heads), int(head_dim))
        backend = self.backend_models_by_geometry.get(geometry)
        if backend is None:
            return self.standard.predict_time_ms(
                sequence_lengths,
                query_heads,
                kv_heads,
                head_dim,
                max_cp_size=max_cp_size,
                allow_extrapolation=allow_extrapolation,
                storage_tokens=storage_tokens,
            )
        lengths = [int(length) for length in sequence_lengths]
        if not lengths or min(lengths) <= 0:
            raise ValueError("sequence_lengths must contain positive integers")
        total = sum(lengths)
        storage = (
            (4096 if total <= 4096 else 8192)
            if storage_tokens is None
            else int(storage_tokens)
        )
        if storage <= 0:
            raise ValueError("storage_tokens must be a positive integer")
        if total > storage:
            raise ValueError(
                f"logical total {total} exceeds the requested {storage}-token "
                "CP1 storage branch"
            )
        if not allow_extrapolation:
            if not self.minimum_total <= total <= self.maximum_total:
                raise ValueError("logical total is outside the measured envelope")
            if (
                not self.minimum_sequence_count
                <= len(lengths)
                <= self.maximum_sequence_count
            ):
                raise ValueError("sequence count is outside the measured envelope")
            if min(lengths) < self.minimum_sequence_tokens:
                raise ValueError("sequence length is outside the measured envelope")
        branch = backend.get(str(storage))
        if branch is None:
            available = tuple(sorted(int(value) for value in backend))
            raise ValueError(
                f"GB200 backend-state artifact has no {storage}-token storage "
                f"branch; available: {available}"
            )
        if total == storage:
            exact_model = branch["exact_storage"]
            if exact_model["kind"] == "constant_median":
                return float(exact_model["time_ms"])
            return self._predict_linear(exact_model, lengths, max_cp_size)
        return self._predict_linear(branch["tail_padded"], lengths, max_cp_size)
