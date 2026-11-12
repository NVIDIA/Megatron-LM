# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

#!/usr/bin/env python3
"""Exact measured singleton-SOL lookup for packed GQA attention."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SingletonSOLPoint:
    hardware: str
    profile_name: str
    query_heads: int
    kv_heads: int
    head_dim: int
    cp_size: int
    sequence_length: int
    time_ms: float
    measurement_run: str
    measurement_provenance: str


class FrozenSingletonSOLPredictor:
    """Look up exact one-sequence measurements without extrapolation."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported singleton SOL artifact schema")
        contract = payload["contract"]
        if (
            contract["phase"] != "forward"
            or contract["scope"] != "te_dot_product_attention"
            or contract["dtype"] != "bfloat16"
            or contract["mask"] != "causal"
            or int(contract["physical_tokens_per_cp_rank"]) != 4096
            or contract["lookup_policy"] != "exact_only"
        ):
            raise ValueError("invalid singleton SOL artifact contract")

        self.payload = payload
        self.points: dict[tuple[str, int, int, int, int, int], SingletonSOLPoint] = {}
        for raw in payload["points"]:
            point = SingletonSOLPoint(
                hardware=str(raw["hardware"]),
                profile_name=str(raw["profile_name"]),
                query_heads=int(raw["query_heads"]),
                kv_heads=int(raw["kv_heads"]),
                head_dim=int(raw["head_dim"]),
                cp_size=int(raw["cp_size"]),
                sequence_length=int(raw["sequence_length"]),
                time_ms=float(raw["time_ms"]),
                measurement_run=str(raw["measurement_run"]),
                measurement_provenance=str(raw["measurement_provenance"]),
            )
            if min(
                point.query_heads,
                point.kv_heads,
                point.head_dim,
                point.cp_size,
                point.sequence_length,
            ) <= 0:
                raise ValueError("singleton SOL point dimensions must be positive")
            if point.query_heads % point.kv_heads:
                raise ValueError("singleton SOL point is not valid GQA")
            if point.sequence_length != 4096 * point.cp_size:
                raise ValueError("singleton SOL point must use 4096 tokens per CP rank")
            if point.time_ms <= 0:
                raise ValueError("singleton SOL timing must be positive")
            key = self._key(
                point.hardware,
                point.cp_size,
                point.query_heads,
                point.kv_heads,
                point.head_dim,
                point.sequence_length,
            )
            if key in self.points:
                raise ValueError(f"duplicate singleton SOL point {key}")
            self.points[key] = point

        expected_count = int(contract["point_count"])
        if len(self.points) != expected_count:
            raise ValueError(
                f"singleton SOL artifact expected {expected_count} points, "
                f"found {len(self.points)}"
            )

    @staticmethod
    def _key(
        hardware: str,
        cp_size: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        sequence_length: int,
    ) -> tuple[str, int, int, int, int, int]:
        return (
            str(hardware).lower(),
            int(cp_size),
            int(query_heads),
            int(kv_heads),
            int(head_dim),
            int(sequence_length),
        )

    @classmethod
    def from_path(cls, path: Path | str) -> "FrozenSingletonSOLPredictor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def lookup(
        self,
        sequence_lengths: Sequence[int],
        hardware: str,
        cp_size: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> SingletonSOLPoint | None:
        lengths = tuple(int(length) for length in sequence_lengths)
        if len(lengths) != 1 or lengths[0] <= 0:
            return None
        return self.points.get(
            self._key(
                hardware,
                cp_size,
                query_heads,
                kv_heads,
                head_dim,
                lengths[0],
            )
        )

    def supported_geometries(
        self, hardware: str, cp_size: int
    ) -> tuple[tuple[int, int, int], ...]:
        normalized = str(hardware).lower()
        return tuple(
            sorted(
                {
                    (point.query_heads, point.kv_heads, point.head_dim)
                    for point in self.points.values()
                    if point.hardware == normalized and point.cp_size == int(cp_size)
                }
            )
        )
