# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

#!/usr/bin/env python3
"""Predict BF16 causal GQA forward attention time through CP64.

CP1 delegates to the measured standalone predictor. CP>1 reconstructs TE's
DualChunkSwap round geometry and evaluates a frozen component model produced by
the CP2 overlap calibration. Six measured attention geometries have independent
frozen branches; this module contains no placeholder timing coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from .cp1_gqa_attention_predictor import (
    FrozenCP1GQAAttentionPredictor,
    predict_cp1_gqa_attention_time_ms,
    predict_cp1_gqa_attention_with_variable_heads_details,
)
from .cp_pipeline import PipelineRound, simulate_two_stream_pipeline
from .cp_round_geometry import RoundGeometry, build_round_geometries

# The TE P2P ring and geometry model are defined for every positive integer CP
# size. Artifacts may still cover only a subset because communication timing is
# hardware- and topology-specific.
SUPPORTED_CP_SIZES = tuple(range(1, 65))
SUPPORTED_CP_IMPLEMENTATIONS = (
    "te_p2p",
    "compute_only",
    "ulysses_compute_only",
)
SUPPORTED_ULYSSES_KV_STRATEGIES = ("strict", "replicate")
SUPPORTED_PHYSICAL_TOKENS_PER_RANK = (4096, 8192, 16384)
DEFAULT_PHYSICAL_TOKENS_PER_RANK = 4096
CP1_MAX_LOGICAL_TOKENS = 8192
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DefaultModelArtifact:
    profile_name: str
    relative_path: str
    sha256: str


DEFAULT_MODEL_ARTIFACTS_BY_GEOMETRY = {
    (32, 4, 128): DefaultModelArtifact(
        profile_name="qwen3_30b",
        relative_path=(
            "cp_overlap/models/frozen_pre_fresh_holdout_v2/"
            "frozen_cp_gqa_attention_model_v2.json"
        ),
        sha256="720f9d12c535f0da8c7b0f053a1f280ed51a59895f80d32ff4b09a6d154188f0",
    ),
    (64, 4, 128): DefaultModelArtifact(
        profile_name="qwen3_235b",
        relative_path=(
            "cp_overlap/models/multi_profile_20260720_v3/qwen3_235b/frozen_v3/"
            "frozen_cp_gqa_attention_model_v3.json"
        ),
        sha256="8227d18c5a7598087d25e4ad27789a213f9c5634b5e2aea1a2231bcc506b578a",
    ),
    (16, 2, 256): DefaultModelArtifact(
        profile_name="qwen3_next_80b",
        relative_path=(
            "cp_overlap/models/multi_profile_20260720_v3/qwen3_next_80b/"
            "frozen_v3/frozen_cp_gqa_attention_model_v3.json"
        ),
        sha256="49b38ee027af69a184f7e0ff143150c0cd2940678bed4408a270fcc8ccff3f6d",
    ),
    (32, 2, 256): DefaultModelArtifact(
        profile_name="qwen3_5_397b",
        relative_path=(
            "cp_overlap/models/multi_profile_20260720_v3/qwen3_5_397b/frozen_v3/"
            "frozen_cp_gqa_attention_model_v3.json"
        ),
        sha256="28d3bcbee193c2ccb245943a8afa6044f2d2b366b71c5e479417a68b5f456ff2",
    ),
    (32, 8, 128): DefaultModelArtifact(
        profile_name="mixtral_8x7b",
        relative_path=(
            "cp_overlap/models/multi_profile_20260720_v3/mixtral_8x7b/frozen_v3/"
            "frozen_cp_gqa_attention_model_v3.json"
        ),
        sha256="c73c3703204d289edc52d7f2e86de59d687e9384c478a98b633d59ff962f4356",
    ),
    (64, 8, 128): DefaultModelArtifact(
        profile_name="llama3_70b",
        relative_path=(
            "cp_overlap/models/multi_profile_20260720_v3/llama3_70b/frozen_v3/"
            "frozen_cp_gqa_attention_model_v3.json"
        ),
        sha256="ea0ac673d6f25a00cc3df6f2bc7d1794d0ed9ef3eb3667c28b4e446a86ec5648",
    ),
}


@dataclass(frozen=True)
class FrozenLinearModel:
    features: tuple[str, ...]
    coefficients: tuple[float, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrozenLinearModel":
        model = cls(
            features=tuple(payload["features"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            centers=tuple(float(value) for value in payload.get("centers", [])),
            scales=tuple(float(value) for value in payload.get("scales", [])),
            minimum=(
                None if payload.get("minimum") is None else float(payload["minimum"])
            ),
            maximum=(
                None if payload.get("maximum") is None else float(payload["maximum"])
            ),
        )
        if len(model.coefficients) != len(model.features) + 1:
            raise ValueError(
                "linear model needs one intercept plus one coefficient per feature"
            )
        if len(model.centers) != len(model.features) or len(model.scales) != len(
            model.features
        ):
            raise ValueError("linear model centers/scales do not match features")
        if any(scale <= 0 for scale in model.scales):
            raise ValueError("linear model scales must be positive")
        return model

    def predict(self, values: dict[str, float]) -> float:
        result = self.coefficients[0]
        for index, name in enumerate(self.features):
            if name not in values:
                raise KeyError(f"missing feature {name!r}")
            standardized = (float(values[name]) - self.centers[index]) / self.scales[
                index
            ]
            result += self.coefficients[index + 1] * standardized
        if self.minimum is not None:
            result = max(self.minimum, result)
        if self.maximum is not None:
            result = min(self.maximum, result)
        return result


@dataclass(frozen=True)
class FixedCapacityLayout:
    valid_lengths: tuple[int, ...]
    padded_lengths: tuple[int, ...]
    capacity_tokens: int
    max_cp_size: int
    sequence_alignment: int
    pad_between_seqs: bool


@dataclass(frozen=True)
class RankPrediction:
    rank: int
    time_ms: float
    timeline: dict[str, Any]
    rounds: tuple[dict[str, Any], ...]


def _positive_lengths(values: Sequence[int]) -> tuple[int, ...]:
    lengths = tuple(int(value) for value in values)
    if not lengths or min(lengths) <= 0:
        raise ValueError("sequence_lengths must contain positive integers")
    return lengths


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def normalize_max_cp_size(cp_size: int, max_cp_size: int | None = None) -> int:
    """Validate the scheduler CP limit used by PR5541 sequence padding."""
    cp_size = _positive_int(cp_size, "cp_size")
    resolved = cp_size if max_cp_size is None else _positive_int(
        max_cp_size, "max_cp_size"
    )
    if cp_size not in SUPPORTED_CP_SIZES or resolved not in SUPPORTED_CP_SIZES:
        raise ValueError(f"cp_size and max_cp_size must be in {SUPPORTED_CP_SIZES}")
    if resolved < cp_size:
        raise ValueError("max_cp_size must be greater than or equal to cp_size")
    return resolved


def normalize_physical_tokens_per_rank(
    value: int | None,
    *,
    default: int = DEFAULT_PHYSICAL_TOKENS_PER_RANK,
) -> int:
    """Validate and normalize the modeled per-rank physical buffer size."""
    resolved = default if value is None else value
    _positive_int(resolved, "physical_tokens_per_rank")
    if resolved not in SUPPORTED_PHYSICAL_TOKENS_PER_RANK:
        raise ValueError(
            "physical_tokens_per_rank must be one of "
            f"{SUPPORTED_PHYSICAL_TOKENS_PER_RANK}, got {resolved}"
        )
    return resolved


def sequence_alignment_for_cp_sizes(cp_sizes: Sequence[int]) -> int:
    """Return the minimum DualChunkSwap alignment shared by all CP choices."""
    sizes = tuple(_positive_int(int(value), "cp_size") for value in cp_sizes)
    if not sizes:
        raise ValueError("cp_sizes must contain at least one value")
    if any(value not in SUPPORTED_CP_SIZES for value in sizes):
        raise ValueError(f"cp_sizes must be in {SUPPORTED_CP_SIZES}")
    alignment = 1
    for cp_size in sizes:
        required = 2 * cp_size
        alignment = alignment * required // math.gcd(alignment, required)
    return alignment


def _scale_lengths_to_cp1_limit(
    sequence_lengths: Sequence[int],
    maximum_logical_tokens: int = CP1_MAX_LOGICAL_TOKENS,
) -> tuple[tuple[int, ...], bool]:
    """Preserve pack composition while fitting a compute proxy into a CP1 limit."""
    lengths = _positive_lengths(sequence_lengths)
    maximum_logical_tokens = _positive_int(
        maximum_logical_tokens, "maximum_logical_tokens"
    )
    total = sum(lengths)
    if total <= maximum_logical_tokens:
        return lengths, False
    if len(lengths) > maximum_logical_tokens:
        raise ValueError("too many sequences to build a positive-length CP1 proxy")

    remaining = maximum_logical_tokens - len(lengths)
    exact_allocations = [remaining * length / total for length in lengths]
    allocations = [math.floor(value) for value in exact_allocations]
    residual = remaining - sum(allocations)
    residual_order = sorted(
        range(len(lengths)),
        key=lambda index: exact_allocations[index] - allocations[index],
        reverse=True,
    )
    for index in residual_order[:residual]:
        allocations[index] += 1
    return tuple(1 + allocation for allocation in allocations), True


def _predict_cp1_compute_scaled_with_details(
    sequence_lengths: Sequence[int],
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    max_cp_size: int = 1,
    cp1_model_artifact: FrozenCP1GQAAttentionPredictor | None = None,
    storage_tokens: int | None = None,
) -> dict[str, Any]:
    """Evaluate a storage-aware CP1 proxy and scale work outside its envelope."""
    lengths = _positive_lengths(sequence_lengths)
    model_storage = storage_tokens if cp1_model_artifact is not None else None
    if cp1_model_artifact is None:
        maximum_logical_tokens = CP1_MAX_LOGICAL_TOKENS
        sequence_alignment = 2 * max_cp_size
        physical_capacity = CP1_MAX_LOGICAL_TOKENS
    else:
        physical_capacity = int(
            cp1_model_artifact.maximum_total if model_storage is None else model_storage
        )
        maximum_logical_tokens = min(
            int(cp1_model_artifact.maximum_total), physical_capacity
        )
        sequence_alignment = cp1_model_artifact.sequence_alignment_for(max_cp_size)

    physical_input = sum(_round_up(length, sequence_alignment) for length in lengths)
    fits_directly = (
        sum(lengths) <= maximum_logical_tokens and physical_input <= physical_capacity
    )
    if fits_directly:
        proxy_lengths, normalized = lengths, False
    else:
        # Reserve the worst-case per-sequence alignment overhead so the scaled
        # proxy is guaranteed to fit the explicitly selected storage branch.
        safe_logical_limit = min(
            maximum_logical_tokens,
            physical_capacity - len(lengths) * (sequence_alignment - 1),
        )
        proxy_lengths, normalized = _scale_lengths_to_cp1_limit(
            lengths, safe_logical_limit
        )
    head_model = predict_cp1_gqa_attention_with_variable_heads_details(
        proxy_lengths,
        query_heads,
        kv_heads,
        head_dim,
        max_cp_size=max_cp_size,
        allow_extrapolation=True,
        model_artifact=cp1_model_artifact,
        storage_tokens=model_storage,
    )
    model_storage_resolver = (
        None
        if cp1_model_artifact is None
        else getattr(cp1_model_artifact, "model_storage_tokens", None)
    )
    prediction_model_storage = (
        model_storage
        if model_storage_resolver is None
        else model_storage_resolver(
            proxy_lengths,
            query_heads,
            kv_heads,
            head_dim,
            max_cp_size=max_cp_size,
        )
    )
    useful_work = float(sum(length * length for length in lengths))
    proxy_work = float(sum(length * length for length in proxy_lengths))
    work_scale = useful_work / proxy_work
    return {
        "time_ms": float(head_model["time_ms"]) * work_scale,
        "input_lengths": lengths,
        "proxy_lengths": proxy_lengths,
        "length_normalized_to_8k": normalized,
        "length_normalized_to_cp1_limit": normalized,
        "cp1_proxy_logical_limit_tokens": maximum_logical_tokens,
        "cp1_physical_storage_tokens": model_storage,
        "cp1_model_storage_tokens": prediction_model_storage,
        "useful_work_sum_squares": useful_work,
        "proxy_work_sum_squares": proxy_work,
        "work_scale": work_scale,
        "head_model": head_model,
    }


def predict_compute_only_attention_with_details(
    sequence_lengths: Sequence[int],
    cp_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    max_cp_size: int | None = None,
    physical_tokens_per_rank: int | None = None,
    cp1_model_artifact: FrozenCP1GQAAttentionPredictor | None = None,
) -> dict[str, Any]:
    """Estimate multi-CP time from CP1 useful compute, excluding communication.

    The local CP1 proxy uses each sequence length divided by CP. Its causal
    attention work is then scaled from ``sum((L/CP)^2)`` to the target per-rank
    work ``sum(L^2)/CP``. No TE round layout, communication, overlap, correction,
    or host-dispatch model is used.
    """
    lengths = _positive_lengths(sequence_lengths)
    cp_size = _positive_int(cp_size, "cp_size")
    max_cp_size = normalize_max_cp_size(cp_size, max_cp_size)
    storage = (
        None
        if physical_tokens_per_rank is None
        else normalize_physical_tokens_per_rank(physical_tokens_per_rank)
    )
    local_lengths = tuple(
        max(1, (length + cp_size // 2) // cp_size) for length in lengths
    )
    local_proxy = _predict_cp1_compute_scaled_with_details(
        local_lengths,
        query_heads,
        kv_heads,
        head_dim,
        max_cp_size=max_cp_size,
        cp1_model_artifact=cp1_model_artifact,
        storage_tokens=storage,
    )
    local_reference_work = float(sum(length * length for length in local_lengths))
    target_per_rank_work = float(sum(length * length for length in lengths)) / cp_size
    cp_compute_scale = target_per_rank_work / local_reference_work
    return {
        "time_ms": float(local_proxy["time_ms"]) * cp_compute_scale,
        "cp_size": cp_size,
        "max_cp_size": max_cp_size,
        "cp_implementation": "compute_only",
        "communication_included": False,
        "validated_against_multi_cp": False,
        "physical_tokens_per_rank": storage,
        "global_lengths": lengths,
        "local_cp1_proxy_lengths": local_lengths,
        "target_per_rank_work_sum_squares": target_per_rank_work,
        "local_reference_work_sum_squares": local_reference_work,
        "cp_compute_scale": cp_compute_scale,
        "local_cp1_proxy": local_proxy,
    }


def _ulysses_local_heads(
    query_heads: int,
    kv_heads: int,
    cp_size: int,
    kv_strategy: str,
) -> tuple[int, int, str]:
    query_heads = _positive_int(query_heads, "query_heads")
    kv_heads = _positive_int(kv_heads, "kv_heads")
    cp_size = _positive_int(cp_size, "cp_size")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads for GQA")
    if query_heads % cp_size:
        raise ValueError("Ulysses requires query_heads to be divisible by cp_size")
    if kv_strategy not in SUPPORTED_ULYSSES_KV_STRATEGIES:
        raise ValueError(
            f"ulysses_kv_strategy must be one of {SUPPORTED_ULYSSES_KV_STRATEGIES}"
        )

    local_query_heads = query_heads // cp_size
    if kv_heads % cp_size == 0:
        return local_query_heads, kv_heads // cp_size, "sharded"
    if kv_strategy == "strict":
        raise ValueError(
            "strict Ulysses requires kv_heads to be divisible by cp_size; "
            "use ulysses_kv_strategy='replicate' only for an implementation "
            "that replicates KV groups"
        )
    if cp_size % kv_heads:
        raise ValueError(
            "replicated-KV Ulysses requires cp_size and kv_heads to divide one another"
        )
    local_kv_heads = 1
    if local_query_heads % local_kv_heads:
        raise ValueError(
            "local Ulysses query heads must be divisible by local KV heads"
        )
    return local_query_heads, local_kv_heads, "replicated"


def predict_ulysses_compute_only_attention_with_details(
    sequence_lengths: Sequence[int],
    cp_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    max_cp_size: int | None = None,
    ulysses_kv_strategy: str = "replicate",
    physical_tokens_per_rank: int | None = None,
    cp1_model_artifact: FrozenCP1GQAAttentionPredictor | None = None,
) -> dict[str, Any]:
    """Estimate Ulysses local core-attention time with all communication omitted."""
    lengths = _positive_lengths(sequence_lengths)
    cp_size = _positive_int(cp_size, "cp_size")
    max_cp_size = normalize_max_cp_size(cp_size, max_cp_size)
    storage = (
        None
        if physical_tokens_per_rank is None
        else normalize_physical_tokens_per_rank(physical_tokens_per_rank)
    )
    local_query_heads, local_kv_heads, kv_partition = _ulysses_local_heads(
        query_heads,
        kv_heads,
        cp_size,
        ulysses_kv_strategy,
    )
    local_proxy = _predict_cp1_compute_scaled_with_details(
        lengths,
        local_query_heads,
        local_kv_heads,
        head_dim,
        max_cp_size=max_cp_size,
        cp1_model_artifact=cp1_model_artifact,
        storage_tokens=storage,
    )
    return {
        "time_ms": float(local_proxy["time_ms"]),
        "cp_size": cp_size,
        "max_cp_size": max_cp_size,
        "cp_implementation": "ulysses_compute_only",
        "communication_included": False,
        "validated_against_multi_cp": False,
        "physical_tokens_per_rank": storage,
        "global_lengths": lengths,
        "global_geometry": {
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        "local_attention_geometry": {
            "query_heads": local_query_heads,
            "kv_heads": local_kv_heads,
            "head_dim": head_dim,
        },
        "ulysses_kv_strategy": ulysses_kv_strategy,
        "kv_partition": kv_partition,
        "local_cp1_proxy": local_proxy,
    }


def make_fixed_capacity_layout(
    sequence_lengths: Sequence[int],
    cp_size: int,
    *,
    physical_tokens_per_rank: int = DEFAULT_PHYSICAL_TOKENS_PER_RANK,
    max_cp_size: int | None = None,
    sequence_alignment: int | None = None,
) -> FixedCapacityLayout:
    lengths = _positive_lengths(sequence_lengths)
    if cp_size not in SUPPORTED_CP_SIZES:
        raise ValueError(f"cp_size must be one of {SUPPORTED_CP_SIZES}")
    physical_tokens_per_rank = normalize_physical_tokens_per_rank(
        physical_tokens_per_rank
    )
    if sequence_alignment is None:
        max_cp_size = normalize_max_cp_size(cp_size, max_cp_size)
        sequence_alignment = 2 * max_cp_size
    else:
        sequence_alignment = _positive_int(sequence_alignment, "sequence_alignment")
        if sequence_alignment % (2 * cp_size):
            raise ValueError("sequence_alignment must be divisible by 2 * cp_size")
        max_cp_size = normalize_max_cp_size(cp_size, max_cp_size)
    capacity = physical_tokens_per_rank * cp_size
    padded = [_round_up(length, sequence_alignment) for length in lengths]
    if sum(padded) > capacity:
        raise ValueError(
            f"aligned sequence storage {sum(padded)} exceeds CP{cp_size} "
            f"capacity {capacity}"
        )
    padded[-1] += capacity - sum(padded)
    has_physical_padding = any(
        valid != physical for valid, physical in zip(lengths, padded)
    )
    return FixedCapacityLayout(
        valid_lengths=lengths,
        padded_lengths=tuple(padded),
        capacity_tokens=capacity,
        max_cp_size=max_cp_size,
        sequence_alignment=sequence_alignment,
        pad_between_seqs=has_physical_padding,
    )


def cp1_equivalent_lengths(geometry: RoundGeometry) -> tuple[int, ...]:
    """Map one CP tile to causal CP1 lengths with the same useful QK+PV work."""
    result = []
    for tile in geometry.sequence_tiles:
        if tile.q_tokens <= 0 or tile.kv_tokens <= 0:
            continue
        if geometry.section == "diagonal":
            equivalent = tile.q_tokens
        else:
            equivalent = max(1, round(math.sqrt(2 * tile.q_tokens * tile.kv_tokens)))
        result.append(equivalent)
    return tuple(result)


class CPGQAAttentionPredictor:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        if int(artifact.get("schema_version", 0)) != 1:
            raise ValueError("unsupported CP attention model schema")
        contract = artifact["contract"]
        self.query_heads = int(contract["query_heads"])
        self.kv_heads = int(contract["kv_heads"])
        self.head_dim = int(contract["head_dim"])
        self.physical_tokens_per_rank = normalize_physical_tokens_per_rank(
            int(contract["physical_tokens_per_rank"])
        )
        self.sequence_alignment = int(contract["sequence_alignment"])
        if self.sequence_alignment <= 0 or self.sequence_alignment % 2:
            raise ValueError("CP artifact sequence alignment must be positive and even")
        self.calibration_max_cp_size = self.sequence_alignment // 2
        cp1_payload = artifact.get("cp1_model_artifact")
        self.cp1_model_artifact = (
            FrozenCP1GQAAttentionPredictor(cp1_payload)
            if cp1_payload is not None
            else None
        )
        if (
            self.cp1_model_artifact is not None
            and not self.cp1_model_artifact.has_geometry(
                (self.query_heads, self.kv_heads, self.head_dim)
            )
        ):
            raise ValueError("embedded CP1 artifact does not cover model geometry")
        self.communication_ms = {
            int(key): float(value)
            for key, value in artifact["communication_wall_p10_ms"].items()
        }
        self.communication_rank_ms = {
            int(key): tuple(float(item) for item in values)
            for key, values in artifact.get("communication_rank_p10_ms", {}).items()
        }
        self.sustained_ring_communication_ms = {
            int(key): float(value)
            for key, value in artifact.get(
                "sustained_ring_effective_communication_wall_p10_ms", {}
            ).items()
        }
        self.sustained_ring_communication_rank_ms = {
            int(key): tuple(float(item) for item in values)
            for key, values in artifact.get(
                "sustained_ring_effective_communication_rank_p10_ms", {}
            ).items()
        }
        self.communication_host_ms = {
            int(key): float(value)
            for key, value in artifact.get(
                "communication_host_launch_wall_p10_ms", {}
            ).items()
        }
        models = artifact["models"]
        self.attention_pure = self._model_map(models["attention_pure"])
        self.attention_active_ratio = FrozenLinearModel.from_dict(
            models["attention_active_ratio"]
        )
        self.attention_active_ratio_scaling = models.get(
            "attention_active_ratio_scaling",
            {"model": "legacy_linear_by_communication_ratio"},
        )
        if self.attention_active_ratio_scaling["model"] not in {
            "legacy_linear_by_communication_ratio",
            "direct_fitted_component_anchors",
        }:
            raise ValueError("unsupported attention active-ratio scaling model")
        if (
            self.attention_active_ratio_scaling["model"]
            == "direct_fitted_component_anchors"
        ):
            communication_range = self.attention_active_ratio_scaling.get(
                "communication_ms_range", ()
            )
            if (
                len(communication_range) != 2
                or float(communication_range[0]) <= 0
                or float(communication_range[1]) < float(communication_range[0])
            ):
                raise ValueError("invalid attention interference communication range")
        self.attention_drain_ratio = self._model_map(models["attention_drain_ratio"])
        self.prepare = self._model_map(models["prepare"])
        self.lse = self._model_map(models["lse"])
        self.output = self._model_map(models["output"])
        self.initialization_ms = float(models["initialization_ms"])
        self.timeline_scale = float(models.get("timeline_scale", 1.0))
        self.fixed_overhead_ms = float(models.get("fixed_overhead_ms", 0.0))
        host_models = models.get("host_critical_rank")
        self.host_diagonal_round = (
            FrozenLinearModel.from_dict(host_models["diagonal_round"])
            if host_models is not None
            else None
        )
        self.host_lower_round = (
            FrozenLinearModel.from_dict(host_models["lower_round"])
            if host_models is not None
            else None
        )
        self.host_cp_scaling = models.get("host_cp_scaling")
        self.host_communication_slowdown = None
        legacy_singleton_floor = models.get("cp2_single_sequence_wall_floor")
        singleton_floors = models.get("single_sequence_wall_floor_by_cp")
        if legacy_singleton_floor is not None and singleton_floors is not None:
            raise ValueError("model cannot contain both singleton floor schemas")
        if singleton_floors is not None:
            self.single_sequence_wall_floor_by_cp = {
                int(cp_size): floor
                for cp_size, floor in singleton_floors["by_cp"].items()
            }
        elif legacy_singleton_floor is not None:
            self.single_sequence_wall_floor_by_cp = {
                int(legacy_singleton_floor["cp_size"]): legacy_singleton_floor
            }
        else:
            self.single_sequence_wall_floor_by_cp = {}
        padded_path_floors = models.get("padded_path_wall_floor_by_cp")
        self.padded_path_wall_floor_by_cp = (
            {
                int(cp_size): floor
                for cp_size, floor in padded_path_floors["by_cp"].items()
            }
            if padded_path_floors is not None
            else {}
        )
        self.wall_time_calibration = models.get("wall_time_calibration")
        if self.wall_time_calibration is not None:
            if self.wall_time_calibration.get("model") != (
                "multiplicative_by_cp_for_exact_pack_contract"
            ):
                raise ValueError("unsupported final wall-time calibration model")
            scope = self.wall_time_calibration["scope"]
            if not isinstance(scope.get("pad_between_seqs"), bool):
                raise ValueError("wall-time calibration padding scope must be boolean")
            _positive_int(int(scope["sequence_count"]), "calibration sequence_count")
            normalize_max_cp_size(1, int(scope["max_cp_size"]))
            normalize_physical_tokens_per_rank(
                int(scope["physical_tokens_per_rank"])
            )
            branches = self.wall_time_calibration.get("by_cp", {})
            if not branches:
                raise ValueError("wall-time calibration must contain CP branches")
            for cp_text, branch in branches.items():
                branch_cp = int(cp_text)
                if branch_cp not in self.communication_ms:
                    raise ValueError(
                        f"wall-time calibration targets uncalibrated CP{branch_cp}"
                    )
                if float(branch["time_scale"]) <= 0:
                    raise ValueError("wall-time calibration scale must be positive")
        if self.initialization_ms < 0 or self.timeline_scale <= 0:
            raise ValueError("invalid frozen timeline coefficients")
        for cp_size, values in self.communication_rank_ms.items():
            if cp_size not in self.communication_ms:
                raise ValueError(
                    f"rank communication calibration targets unknown CP{cp_size}"
                )
            if len(values) != cp_size or any(value <= 0 for value in values):
                raise ValueError(
                    f"rank communication calibration must contain {cp_size} "
                    "positive values"
                )
        if set(self.sustained_ring_communication_ms) != set(
            self.sustained_ring_communication_rank_ms
        ):
            raise ValueError(
                "sustained-ring wall and rank communication maps must cover "
                "the same CP sizes"
            )
        for cp_size, values in self.sustained_ring_communication_rank_ms.items():
            if cp_size not in self.communication_ms:
                raise ValueError(
                    f"sustained-ring calibration targets unknown CP{cp_size}"
                )
            if len(values) != cp_size or any(value <= 0 for value in values):
                raise ValueError(
                    f"sustained-ring calibration must contain {cp_size} "
                    "positive rank values"
                )
            if self.sustained_ring_communication_ms[cp_size] <= 0:
                raise ValueError("sustained-ring wall timing must be positive")
        if (self.host_diagonal_round is None) != (self.host_lower_round is None):
            raise ValueError("host critical-rank models must be provided together")
        if self.host_diagonal_round is not None and set(
            self.communication_host_ms
        ) != set(self.communication_ms):
            raise ValueError(
                "host communication calibration must cover every modeled CP size"
            )
        if self.host_cp_scaling is not None:
            minimum_cp_size = int(self.host_cp_scaling["minimum_cp_size"])
            if minimum_cp_size < 2:
                raise ValueError("host CP scaling minimum must be at least two")
            model_kind = self.host_cp_scaling.get(
                "model", "fixed_per_call_plus_cp_rounds"
            )
            if model_kind == "fixed_per_call_plus_cp_rounds":
                for group in ("aligned", "inter_sequence_padded"):
                    parameters = self.host_cp_scaling["groups"][group]
                    if float(parameters["per_round_ms"]) < 0:
                        raise ValueError(
                            "host CP scaling per-round cost must be nonnegative"
                        )
            elif model_kind == "linear_features":
                self.host_cp_scaling_models = self._model_map(
                    self.host_cp_scaling["groups"]
                )
                topology = self.host_cp_scaling.get(
                    "outside_single_nvlink_domain_by_cp", {}
                )
                if not set(self.communication_ms).issubset(
                    {int(cp_size) for cp_size in topology}
                ):
                    raise ValueError(
                        "host topology map must cover every modeled CP size"
                    )
            else:
                raise ValueError(f"unsupported host CP scaling model {model_kind!r}")
            for cp_text, scale in self.host_cp_scaling.get(
                "calibration_time_scale_by_cp", {}
            ).items():
                scale_cp = int(cp_text)
                if scale_cp not in self.communication_ms:
                    raise ValueError(
                        "host calibration scale targets uncalibrated "
                        f"CP{scale_cp}"
                    )
                if float(scale) <= 0:
                    raise ValueError("host calibration scales must be positive")
            non_power_scale = self.host_cp_scaling.get(
                "non_power_of_two_time_scale"
            )
            if non_power_scale is not None and float(non_power_scale) <= 0:
                raise ValueError(
                    "non-power-of-two host calibration scale must be positive"
                )
            correction = self.host_cp_scaling.get(
                "communication_host_slowdown_correction"
            )
            if correction is not None:
                if correction.get("model") != (
                    "hinge_ratio_to_interpolated_reference"
                ):
                    raise ValueError("unsupported host communication correction")
                slope = float(correction["slope"])
                minimum_scale = float(correction.get("minimum_scale", 1.0))
                reference_cp_sizes = tuple(
                    int(value) for value in correction["reference_cp_sizes"]
                )
                if slope < 0 or minimum_scale <= 0:
                    raise ValueError("invalid host communication correction")
                if (
                    len(reference_cp_sizes) < 2
                    or tuple(sorted(set(reference_cp_sizes))) != reference_cp_sizes
                    or any(
                        cp not in self.communication_host_ms
                        for cp in reference_cp_sizes
                    )
                ):
                    raise ValueError(
                        "host communication correction references invalid CP sizes"
                    )
                self.host_communication_slowdown = {
                    **correction,
                    "slope": slope,
                    "minimum_scale": minimum_scale,
                    "reference_cp_sizes": reference_cp_sizes,
                }
        for cp_size, floor in self.single_sequence_wall_floor_by_cp.items():
            if cp_size not in self.communication_ms:
                raise ValueError(
                    f"single-sequence wall floor targets uncalibrated CP{cp_size}"
                )
            if "cp_size" in floor and int(floor["cp_size"]) != cp_size:
                raise ValueError("single-sequence wall floor CP key differs")
            points = floor["points"]
            utilizations = [float(point["utilization"]) for point in points]
            if len(points) < 2 or utilizations != sorted(set(utilizations)):
                raise ValueError(
                    "single-sequence wall floor needs unique sorted utilization points"
                )
            if utilizations[0] <= 0 or utilizations[-1] > 1:
                raise ValueError("single-sequence wall floor utilization is invalid")
            if any(float(point["wall_p10_ms"]) <= 0 for point in points):
                raise ValueError("single-sequence wall floor times must be positive")
        for cp_size, floor in self.padded_path_wall_floor_by_cp.items():
            if cp_size not in self.communication_ms:
                raise ValueError(
                    f"padded-path wall floor targets uncalibrated CP{cp_size}"
                )
            if "cp_size" in floor and int(floor["cp_size"]) != cp_size:
                raise ValueError("padded-path wall floor CP key differs")
            if float(floor["wall_p10_ms"]) <= 0:
                raise ValueError("padded-path wall floor time must be positive")
            floor_alignment = floor.get("sequence_alignment")
            if floor_alignment is not None and (
                int(floor_alignment) <= 0
                or int(floor_alignment) % (2 * cp_size)
            ):
                raise ValueError(
                    "padded-path floor sequence alignment must be divisible "
                    "by 2 * CP"
                )
            floor_storage = floor.get("physical_tokens_per_rank")
            if floor_storage is not None:
                normalize_physical_tokens_per_rank(int(floor_storage))

    @staticmethod
    def _model_map(payload: dict[str, Any]) -> dict[str, FrozenLinearModel]:
        return {
            str(name): FrozenLinearModel.from_dict(model)
            for name, model in payload.items()
        }

    @classmethod
    def from_path(cls, path: Path | str) -> "CPGQAAttentionPredictor":
        model_path = Path(path)
        return cls(json.loads(model_path.read_text(encoding="utf-8")))

    def _round_features(
        self,
        geometry: RoundGeometry,
        *,
        pad_between_seqs: bool,
        communication_ms: float,
        global_lengths: Sequence[int],
        cp_size: int,
        max_cp_size: int,
    ) -> dict[str, float]:
        equivalent = cp1_equivalent_lengths(geometry)
        cp1_proxy = (
            _predict_cp1_compute_scaled_with_details(
                equivalent,
                self.query_heads,
                self.kv_heads,
                self.head_dim,
                max_cp_size=max_cp_size,
                cp1_model_artifact=self.cp1_model_artifact,
                storage_tokens=self.physical_tokens_per_rank,
            )
            if equivalent
            else {"time_ms": 0.0}
        )
        cp1_ms = float(cp1_proxy["time_ms"])
        cp2_comm = self.communication_ms[2]
        base_local_lengths = [length / cp_size for length in global_lengths]
        base_local_max = max(base_local_lengths, default=0.0)
        sequence_count = len(global_lengths)
        return {
            "cp1_ms": cp1_ms,
            "s2_m": geometry.cp1_equivalent_sum_squared_tokens / 1.0e6,
            "tile128_work_m": sum(
                ((length + 127) // 128) * 128 * length for length in equivalent
            )
            / 1.0e6,
            "q_tokens_k": geometry.q_valid_tokens / 1024.0,
            "kv_tokens_k": geometry.kv_valid_tokens / 1024.0,
            "max_q_k": max(
                (tile.q_tokens for tile in geometry.sequence_tiles), default=0
            )
            / 1024.0,
            "max_kv_k": max(
                (tile.kv_tokens for tile in geometry.sequence_tiles), default=0
            )
            / 1024.0,
            "active_sequences": float(geometry.active_sequences),
            "pad_between_seqs": float(pad_between_seqs),
            "q_physical_tokens_k": geometry.q_physical_tokens / 1024.0,
            "kv_physical_tokens_k": geometry.kv_physical_tokens / 1024.0,
            "base_local_total_k": sum(base_local_lengths) / 1024.0,
            "base_local_max_k": base_local_max / 1024.0,
            "base_local_max_sq_m": base_local_max * base_local_max / 1.0e6,
            "base_local_s2_m": sum(length * length for length in base_local_lengths)
            / 1.0e6,
            "capacity_utilization": sum(global_lengths)
            / (self.physical_tokens_per_rank * cp_size),
            "sequence_count": float(sequence_count),
            "single_sequence": float(sequence_count == 1),
            "cp_size": float(cp_size),
            "log2_cp_size": math.log2(cp_size),
            "communication_ms": communication_ms,
            "communication_ratio_to_cp2": (
                communication_ms / cp2_comm if cp2_comm > 0 else 1.0
            ),
        }

    def _reference_host_communication_ms(self, cp_size: int) -> float:
        if self.host_communication_slowdown is None:
            raise RuntimeError("host communication correction is not configured")
        reference_cp_sizes = self.host_communication_slowdown["reference_cp_sizes"]
        if cp_size <= reference_cp_sizes[0]:
            return self.communication_host_ms[reference_cp_sizes[0]]
        if cp_size >= reference_cp_sizes[-1]:
            return self.communication_host_ms[reference_cp_sizes[-1]]
        upper_index = next(
            index
            for index, reference_cp in enumerate(reference_cp_sizes)
            if reference_cp >= cp_size
        )
        upper_cp = reference_cp_sizes[upper_index]
        lower_cp = reference_cp_sizes[upper_index - 1]
        lower_ms = self.communication_host_ms[lower_cp]
        upper_ms = self.communication_host_ms[upper_cp]
        weight = (cp_size - lower_cp) / (upper_cp - lower_cp)
        return lower_ms + weight * (upper_ms - lower_ms)

    def predict_with_details(
        self,
        sequence_lengths: Sequence[int],
        cp_size: int,
        *,
        max_cp_size: int | None = None,
        sequence_alignment: int | None = None,
        physical_tokens_per_rank: int | None = None,
    ) -> dict[str, Any]:
        default_max_cp_size = (
            cp_size
            if sequence_alignment is not None
            else self.calibration_max_cp_size
        )
        max_cp_size = normalize_max_cp_size(
            cp_size,
            default_max_cp_size if max_cp_size is None else max_cp_size,
        )
        requested_storage = normalize_physical_tokens_per_rank(
            physical_tokens_per_rank,
            default=self.physical_tokens_per_rank,
        )
        if requested_storage != self.physical_tokens_per_rank:
            raise ValueError(
                "CP attention artifact is calibrated for "
                f"{self.physical_tokens_per_rank} physical tokens per rank, not "
                f"{requested_storage}; load the matching storage branch"
            )
        if cp_size == 1:
            if sequence_alignment is not None and sequence_alignment != 2 * max_cp_size:
                raise ValueError(
                    "explicit CP1 sequence_alignment must equal 2 * max_cp_size"
                )
            time_ms = predict_cp1_gqa_attention_time_ms(
                sequence_lengths,
                self.query_heads,
                self.kv_heads,
                self.head_dim,
                max_cp_size=max_cp_size,
                model_artifact=self.cp1_model_artifact,
                storage_tokens=requested_storage,
            )
            return {
                "time_ms": time_ms,
                "cp_size": 1,
                "max_cp_size": max_cp_size,
                "sequence_alignment": 2 * max_cp_size,
                "physical_tokens_per_rank": requested_storage,
                "ranks": [],
            }
        if cp_size not in self.communication_ms:
            raise ValueError(f"model has no CP{cp_size} communication calibration")

        layout = make_fixed_capacity_layout(
            sequence_lengths,
            cp_size,
            physical_tokens_per_rank=requested_storage,
            max_cp_size=max_cp_size,
            sequence_alignment=sequence_alignment,
        )
        geometries = build_round_geometries(
            layout.valid_lengths, layout.padded_lengths, cp_size
        )
        comm_ms = self.communication_ms[cp_size]
        rank_communication = self.communication_rank_ms.get(cp_size)
        sustained_comm_ms = self.sustained_ring_communication_ms.get(cp_size, comm_ms)
        sustained_rank_communication = self.sustained_ring_communication_rank_ms.get(
            cp_size
        )
        rank_predictions = []
        for rank in range(cp_size):
            rank_rounds = sorted(
                (item for item in geometries if item.rank == rank),
                key=lambda item: item.step,
            )
            pipeline_rounds = []
            output_durations = []
            round_details = []
            for geometry in rank_rounds:
                calibrated_rank_comm = (
                    rank_communication[rank]
                    if rank_communication is not None
                    else comm_ms
                )
                pipeline_rank_comm = (
                    sustained_rank_communication[rank]
                    if sustained_rank_communication is not None
                    else (
                        rank_communication[rank]
                        if rank_communication is not None
                        else sustained_comm_ms
                    )
                )
                interference_comm = (
                    calibrated_rank_comm if geometry.step < cp_size - 1 else 0.0
                )
                pipeline_outgoing_comm = (
                    pipeline_rank_comm if geometry.step < cp_size - 1 else 0.0
                )
                features = self._round_features(
                    geometry,
                    pad_between_seqs=layout.pad_between_seqs,
                    communication_ms=interference_comm,
                    global_lengths=layout.valid_lengths,
                    cp_size=cp_size,
                    max_cp_size=max_cp_size,
                )
                pure_ms = self.attention_pure[geometry.section].predict(features)
                features["pure_attention_ms"] = pure_ms
                if interference_comm:
                    if (
                        self.attention_active_ratio_scaling["model"]
                        == "direct_fitted_component_anchors"
                    ):
                        lower, upper = self.attention_active_ratio_scaling[
                            "communication_ms_range"
                        ]
                        ratio_features = dict(features)
                        ratio_features["communication_ms"] = min(
                            float(upper),
                            max(float(lower), features["communication_ms"]),
                        )
                        features["attention_ratio_communication_ms"] = ratio_features[
                            "communication_ms"
                        ]
                        fitted_ratio = self.attention_active_ratio.predict(
                            ratio_features
                        )
                        ratio = fitted_ratio
                    else:
                        fitted_ratio = self.attention_active_ratio.predict(features)
                        ratio = (
                            1.0
                            + (fitted_ratio - 1.0)
                            * features["communication_ratio_to_cp2"]
                        )
                else:
                    ratio = self.attention_drain_ratio[geometry.section].predict(
                        features
                    )
                ratio = max(0.0, ratio)
                attention_ms = pure_ms * ratio
                features["attention_ms"] = attention_ms
                prepare_ms = self.prepare[geometry.section].predict(features)
                correction_kind = (
                    "full" if geometry.step <= geometry.rank else "second_half"
                )
                lse_ms = (
                    self.initialization_ms
                    if geometry.step == 0
                    else self.lse[correction_kind].predict(features)
                )
                output_ms = self.output[correction_kind].predict(features)
                pipeline_rounds.append(
                    PipelineRound(
                        step=geometry.step,
                        prepare_ms=prepare_ms,
                        attention_ms=attention_ms,
                        lse_or_init_ms=lse_ms,
                        outgoing_communication_ms=pipeline_outgoing_comm,
                    )
                )
                output_durations.append(output_ms)
                round_details.append(
                    {
                        "geometry": geometry.to_dict(),
                        "features": features,
                        "pure_attention_ms": pure_ms,
                        "attention_ratio": ratio,
                        "attention_ms": attention_ms,
                        "prepare_ms": prepare_ms,
                        "lse_or_init_ms": lse_ms,
                        "output_correction_ms": output_ms,
                        "interference_communication_ms": interference_comm,
                        "outgoing_communication_ms": pipeline_outgoing_comm,
                        "communication_timing_source": (
                            "sustained_ring_effective_per_exchange"
                            if sustained_rank_communication is not None
                            else "isolated_single_exchange"
                        ),
                    }
                )
            timeline = simulate_two_stream_pipeline(pipeline_rounds, output_durations)
            calibrated_time = (
                self.timeline_scale * timeline.total_ms + self.fixed_overhead_ms
            )
            rank_predictions.append(
                RankPrediction(
                    rank=rank,
                    time_ms=calibrated_time,
                    timeline=timeline.to_dict(),
                    rounds=tuple(round_details),
                )
            )

        wall_ms = max(item.time_ms for item in rank_predictions)
        host_details = None
        host_ms = 0.0
        use_cp_scaling = self.host_cp_scaling is not None and cp_size >= int(
            self.host_cp_scaling["minimum_cp_size"]
        )
        if use_cp_scaling:
            group = "inter_sequence_padded" if layout.pad_between_seqs else "aligned"
            model_kind = self.host_cp_scaling.get(
                "model", "fixed_per_call_plus_cp_rounds"
            )
            if model_kind == "linear_features":
                topology = self.host_cp_scaling["outside_single_nvlink_domain_by_cp"]
                outside_domain = float(topology[str(cp_size)])
                sequence_count_gt4 = float(len(layout.valid_lengths) > 4)
                host_features = {
                    "cp_size": float(cp_size),
                    "sequence_count": float(len(layout.valid_lengths)),
                    "sequence_count_gt4": sequence_count_gt4,
                    "outside_single_nvlink_domain": outside_domain,
                    "cp_times_sequence_count_gt4": (cp_size * sequence_count_gt4),
                    "cp_times_outside_single_nvlink_domain": (cp_size * outside_domain),
                }
                host_ms = self.host_cp_scaling_models[group].predict(host_features)
                host_details = {
                    "model": "linear_features",
                    "critical_rank": cp_size - 1,
                    "group": group,
                    "features": host_features,
                    "selected_features": self.host_cp_scaling_models[group].features,
                    "time_ms": host_ms,
                }
            else:
                parameters = self.host_cp_scaling["groups"][group]
                fixed_ms = float(parameters["fixed_ms"])
                per_round_ms = float(parameters["per_round_ms"])
                host_ms = max(0.0, fixed_ms + cp_size * per_round_ms)
                host_details = {
                    "model": "fixed_per_call_plus_cp_rounds",
                    "critical_rank": cp_size - 1,
                    "group": group,
                    "fixed_ms": fixed_ms,
                    "per_round_ms": per_round_ms,
                    "round_count": cp_size,
                    "time_ms": host_ms,
                }
        elif self.host_diagonal_round is not None:
            host_features = {
                "pad_between_seqs": float(layout.pad_between_seqs),
                "sequence_count": float(len(layout.valid_lengths)),
                "sequence_count_gt4": float(len(layout.valid_lengths) > 4),
            }
            diagonal_ms = self.host_diagonal_round.predict(host_features)
            lower_cp2_ms = self.host_lower_round.predict(host_features)
            communication_delta_ms = (
                self.communication_host_ms[cp_size] - self.communication_host_ms[2]
            )
            lower_ms = max(0.0, lower_cp2_ms + communication_delta_ms)
            host_ms = diagonal_ms + (cp_size - 1) * lower_ms
            host_details = {
                "model": "cp2_diagonal_plus_lower_rounds",
                "critical_rank": cp_size - 1,
                "features": host_features,
                "diagonal_round_ms": diagonal_ms,
                "lower_round_ms": lower_ms,
                "lower_round_count": cp_size - 1,
                "communication_host_delta_ms": communication_delta_ms,
                "time_ms": host_ms,
            }
        host_communication_correction = None
        if host_details is not None and self.host_communication_slowdown is not None:
            measured_host_communication_ms = self.communication_host_ms[cp_size]
            reference_host_communication_ms = (
                self._reference_host_communication_ms(cp_size)
            )
            slowdown_ratio = (
                measured_host_communication_ms / reference_host_communication_ms
            )
            slope = self.host_communication_slowdown["slope"]
            correction_scale = max(
                self.host_communication_slowdown["minimum_scale"],
                1.0 + slope * (slowdown_ratio - 1.0),
            )
            uncorrected_host_ms = host_ms
            host_ms *= correction_scale
            host_details["communication_slowdown_uncorrected_time_ms"] = (
                uncorrected_host_ms
            )
            host_details["communication_slowdown_scale"] = correction_scale
            host_details["time_ms"] = host_ms
            host_communication_correction = {
                "model": self.host_communication_slowdown["model"],
                "measured_host_communication_ms": measured_host_communication_ms,
                "reference_host_communication_ms": reference_host_communication_ms,
                "slowdown_ratio": slowdown_ratio,
                "slope": slope,
                "scale": correction_scale,
                "uncorrected_time_ms": uncorrected_host_ms,
                "time_ms": host_ms,
            }
        host_calibration_scale = None
        host_calibration_scale_source = None
        if host_details is not None and self.host_cp_scaling is not None:
            scale = self.host_cp_scaling.get(
                "calibration_time_scale_by_cp", {}
            ).get(str(cp_size))
            if scale is not None:
                host_calibration_scale_source = "cp_specific"
            elif cp_size & (cp_size - 1):
                scale = self.host_cp_scaling.get(
                    "non_power_of_two_time_scale"
                )
                if scale is not None:
                    host_calibration_scale_source = "non_power_of_two"
            if scale is not None:
                uncalibrated_host_ms = host_ms
                host_calibration_scale = float(scale)
                host_ms *= host_calibration_scale
                host_details["uncalibrated_time_ms"] = uncalibrated_host_ms
                host_details["calibration_time_scale"] = host_calibration_scale
                host_details["calibration_time_scale_source"] = (
                    host_calibration_scale_source
                )
                host_details["time_ms"] = host_ms
        predicted_ms = max(wall_ms, host_ms)
        padded_path_floor = None
        if (
            layout.pad_between_seqs
            and cp_size in self.padded_path_wall_floor_by_cp
        ):
            floor_model = self.padded_path_wall_floor_by_cp[cp_size]
            floor_alignment = floor_model.get("sequence_alignment")
            floor_storage = floor_model.get("physical_tokens_per_rank")
            contract_matches = (
                (
                    floor_alignment is None
                    or int(floor_alignment) == layout.sequence_alignment
                )
                and (
                    floor_storage is None
                    or int(floor_storage) == requested_storage
                )
            )
            if contract_matches:
                floor_ms = float(floor_model["wall_p10_ms"])
                predicted_ms = max(predicted_ms, floor_ms)
                padded_path_floor = {
                    "model": "measured_constant_wall_p10",
                    "cp_size": cp_size,
                    "calibration_source": floor_model.get("calibration_source"),
                    "sequence_alignment": floor_alignment,
                    "physical_tokens_per_rank": floor_storage,
                    "time_ms": floor_ms,
                    "active": floor_ms >= max(wall_ms, host_ms),
                }
        single_sequence_floor = None
        if (
            cp_size in self.single_sequence_wall_floor_by_cp
            and len(layout.valid_lengths) == 1
        ):
            utilization = sum(layout.valid_lengths) / layout.capacity_tokens
            floor_model = self.single_sequence_wall_floor_by_cp[cp_size]
            points = floor_model["points"]
            upper_index = next(
                (
                    index
                    for index, point in enumerate(points)
                    if float(point["utilization"]) >= utilization
                ),
                len(points) - 1,
            )
            lower_index = max(0, upper_index - 1)
            if upper_index == lower_index and upper_index + 1 < len(points):
                upper_index += 1
            lower = points[lower_index]
            upper = points[upper_index]
            lower_u = float(lower["utilization"])
            upper_u = float(upper["utilization"])
            weight = (
                0.0
                if upper_u == lower_u
                else (utilization - lower_u) / (upper_u - lower_u)
            )
            floor_ms = float(lower["wall_p10_ms"]) + weight * (
                float(upper["wall_p10_ms"]) - float(lower["wall_p10_ms"])
            )
            floor_ms = max(0.0, floor_ms)
            predicted_ms = max(predicted_ms, floor_ms)
            single_sequence_floor = {
                "model": "piecewise_linear_by_capacity_utilization",
                "cp_size": cp_size,
                "calibration_source": floor_model.get("calibration_source"),
                "utilization": utilization,
                "lower_point": lower,
                "upper_point": upper,
                "interpolation_weight": weight,
                "time_ms": floor_ms,
                "active": floor_ms >= max(wall_ms, host_ms),
            }
        wall_time_calibration = None
        if self.wall_time_calibration is not None:
            scope = self.wall_time_calibration["scope"]
            branch = self.wall_time_calibration["by_cp"].get(str(cp_size))
            contract_matches = (
                branch is not None
                and bool(scope["pad_between_seqs"]) == layout.pad_between_seqs
                and int(scope["sequence_count"]) == len(layout.valid_lengths)
                and int(scope["max_cp_size"]) == max_cp_size
                and int(scope["physical_tokens_per_rank"]) == requested_storage
            )
            if contract_matches:
                uncalibrated_ms = predicted_ms
                time_scale = float(branch["time_scale"])
                predicted_ms *= time_scale
                wall_time_calibration = {
                    "model": self.wall_time_calibration["model"],
                    "calibration_source": branch.get("calibration_source"),
                    "fit_case_count": branch.get("fit_case_count"),
                    "time_scale": time_scale,
                    "uncalibrated_time_ms": uncalibrated_ms,
                    "time_ms": predicted_ms,
                }
        return {
            "time_ms": predicted_ms,
            "cp_size": cp_size,
            "max_cp_size": max_cp_size,
            "sequence_alignment": layout.sequence_alignment,
            "physical_tokens_per_rank": requested_storage,
            "layout": asdict(layout),
            "gpu_time_ms": wall_ms,
            "host_time_ms": host_ms,
            "limiting_timeline": "host" if host_ms > wall_ms else "gpu",
            "host": host_details,
            "host_communication_slowdown_correction": (
                host_communication_correction
            ),
            "host_calibration_time_scale": host_calibration_scale,
            "host_calibration_time_scale_source": (
                host_calibration_scale_source
            ),
            "padded_path_wall_floor": padded_path_floor,
            "single_sequence_wall_floor": single_sequence_floor,
            "wall_time_calibration": wall_time_calibration,
            "ranks": [asdict(item) for item in rank_predictions],
        }

    def predict_ms(
        self,
        sequence_lengths: Sequence[int],
        cp_size: int,
        *,
        max_cp_size: int | None = None,
        sequence_alignment: int | None = None,
        physical_tokens_per_rank: int | None = None,
    ) -> float:
        return float(
            self.predict_with_details(
                sequence_lengths,
                cp_size,
                max_cp_size=max_cp_size,
                sequence_alignment=sequence_alignment,
                physical_tokens_per_rank=physical_tokens_per_rank,
            )["time_ms"]
        )


def supported_attention_geometries() -> tuple[tuple[int, int, int], ...]:
    """Return measured ``(query_heads, kv_heads, head_dim)`` branches."""
    return tuple(DEFAULT_MODEL_ARTIFACTS_BY_GEOMETRY)


def _default_artifact(
    query_heads: int, kv_heads: int, head_dim: int
) -> DefaultModelArtifact:
    geometry = (int(query_heads), int(kv_heads), int(head_dim))
    artifact = DEFAULT_MODEL_ARTIFACTS_BY_GEOMETRY.get(geometry)
    if artifact is None:
        supported = ", ".join(
            f"{hq}/{hkv}/{dimension}"
            for hq, hkv, dimension in supported_attention_geometries()
        )
        raise ValueError(
            "no measured CP1/2/4/8 branch for geometry "
            f"{geometry[0]}/{geometry[1]}/{geometry[2]}; supported: {supported}"
        )
    return artifact


@lru_cache(maxsize=None)
def load_default_predictor(
    query_heads: int, kv_heads: int, head_dim: int
) -> CPGQAAttentionPredictor:
    """Load and verify the frozen model for one measured geometry."""
    artifact = _default_artifact(query_heads, kv_heads, head_dim)
    model_path = ROOT / artifact.relative_path
    actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_sha256 != artifact.sha256:
        raise ValueError(
            f"{artifact.profile_name} model hash mismatch: expected "
            f"{artifact.sha256}, got {actual_sha256}"
        )
    predictor = CPGQAAttentionPredictor.from_path(model_path)
    expected_geometry = (int(query_heads), int(kv_heads), int(head_dim))
    model_geometry = (
        predictor.query_heads,
        predictor.kv_heads,
        predictor.head_dim,
    )
    if model_geometry != expected_geometry:
        raise ValueError(
            f"{artifact.profile_name} model geometry is {model_geometry}, "
            f"expected {expected_geometry}"
        )
    return predictor


def predict_gqa_attention_with_details(
    sequence_lengths: Sequence[int],
    cp_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    max_cp_size: int | None = None,
    sequence_alignment: int | None = None,
    cp_implementation: str = "te_p2p",
    ulysses_kv_strategy: str = "replicate",
    physical_tokens_per_rank: int | None = None,
) -> dict[str, Any]:
    """Predict H100 forward-attention time using an explicit CP implementation.

    ``te_p2p`` is the measured default. The two ``*_compute_only`` modes are
    analytical fallbacks that intentionally omit all communication and are not
    substitutes for a measured implementation-specific model.
    """
    if cp_implementation not in SUPPORTED_CP_IMPLEMENTATIONS:
        raise ValueError(
            f"cp_implementation must be one of {SUPPORTED_CP_IMPLEMENTATIONS}"
        )
    if cp_implementation != "te_p2p":
        if sequence_alignment is not None:
            raise ValueError(
                "sequence_alignment is only supported by the te_p2p predictor"
            )
        if cp_implementation == "compute_only":
            result = predict_compute_only_attention_with_details(
                sequence_lengths,
                cp_size,
                query_heads,
                kv_heads,
                head_dim,
                max_cp_size=max_cp_size,
                physical_tokens_per_rank=physical_tokens_per_rank,
            )
        else:
            result = predict_ulysses_compute_only_attention_with_details(
                sequence_lengths,
                cp_size,
                query_heads,
                kv_heads,
                head_dim,
                max_cp_size=max_cp_size,
                ulysses_kv_strategy=ulysses_kv_strategy,
                physical_tokens_per_rank=physical_tokens_per_rank,
            )
        measured_artifact = DEFAULT_MODEL_ARTIFACTS_BY_GEOMETRY.get(
            (int(query_heads), int(kv_heads), int(head_dim))
        )
        return {
            "profile_name": (
                measured_artifact.profile_name
                if measured_artifact is not None
                else "variable_head_fallback"
            ),
            "geometry": {
                "query_heads": int(query_heads),
                "kv_heads": int(kv_heads),
                "head_dim": int(head_dim),
            },
            "model_sha256": None,
            **result,
        }

    artifact = _default_artifact(query_heads, kv_heads, head_dim)
    predictor = load_default_predictor(query_heads, kv_heads, head_dim)
    result = predictor.predict_with_details(
        sequence_lengths,
        cp_size,
        max_cp_size=max_cp_size,
        sequence_alignment=sequence_alignment,
        physical_tokens_per_rank=physical_tokens_per_rank,
    )
    return {
        "profile_name": artifact.profile_name,
        "geometry": {
            "query_heads": int(query_heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
        },
        "model_sha256": artifact.sha256,
        **result,
    }


def predict_gqa_attention_time_ms(
    sequence_lengths: Sequence[int],
    cp_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    max_cp_size: int | None = None,
    sequence_alignment: int | None = None,
    cp_implementation: str = "te_p2p",
    ulysses_kv_strategy: str = "replicate",
    physical_tokens_per_rank: int | None = None,
) -> float:
    """Return predicted H100 BF16 causal GQA forward time in milliseconds."""
    return float(
        predict_gqa_attention_with_details(
            sequence_lengths,
            cp_size,
            query_heads,
            kv_heads,
            head_dim,
            max_cp_size=max_cp_size,
            sequence_alignment=sequence_alignment,
            cp_implementation=cp_implementation,
            ulysses_kv_strategy=ulysses_kv_strategy,
            physical_tokens_per_rank=physical_tokens_per_rank,
        )["time_ms"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        help="Explicit frozen model; omit to select a measured geometry branch.",
    )
    parser.add_argument(
        "--sequence-alignment",
        type=int,
        help=(
            "Explicit physical per-sequence alignment. It must be divisible "
            "by 2 * cp_size; use this when allowed CP sizes are not a divisor "
            "chain."
        ),
    )
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument(
        "--max-cp-size",
        type=int,
        choices=SUPPORTED_CP_SIZES,
        help=(
            "Scheduler maximum CP size. Without --sequence-alignment, physical "
            "per-sequence alignment is twice this value."
        ),
    )
    parser.add_argument("--query-heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument(
        "--physical-tokens-per-rank",
        type=int,
        choices=SUPPORTED_PHYSICAL_TOKENS_PER_RANK,
    )
    parser.add_argument(
        "--cp-implementation",
        choices=SUPPORTED_CP_IMPLEMENTATIONS,
        default="te_p2p",
    )
    parser.add_argument(
        "--ulysses-kv-strategy",
        choices=SUPPORTED_ULYSSES_KV_STRATEGIES,
        default="replicate",
        help="Used only by --cp-implementation ulysses_compute_only.",
    )
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",") if value]
    geometry_values = (args.query_heads, args.kv_heads, args.head_dim)
    if args.model is not None:
        if args.cp_implementation != "te_p2p":
            parser.error("--model only supports --cp-implementation te_p2p")
        if any(value is not None for value in geometry_values):
            parser.error("--model cannot be combined with geometry arguments")
        predictor = CPGQAAttentionPredictor.from_path(args.model)
        result = predictor.predict_with_details(
            lengths,
            args.cp_size,
            max_cp_size=args.max_cp_size,
            sequence_alignment=args.sequence_alignment,
            physical_tokens_per_rank=args.physical_tokens_per_rank,
        )
    else:
        if any(value is None for value in geometry_values):
            parser.error(
                "omit --model only when --query-heads, --kv-heads, and "
                "--head-dim are all provided"
            )
        result = predict_gqa_attention_with_details(
            lengths,
            args.cp_size,
            args.query_heads,
            args.kv_heads,
            args.head_dim,
            max_cp_size=args.max_cp_size,
            sequence_alignment=args.sequence_alignment,
            cp_implementation=args.cp_implementation,
            ulysses_kv_strategy=args.ulysses_kv_strategy,
            physical_tokens_per_rank=args.physical_tokens_per_rank,
        )
    if args.details:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['time_ms']:.9f}")


if __name__ == "__main__":
    main()
