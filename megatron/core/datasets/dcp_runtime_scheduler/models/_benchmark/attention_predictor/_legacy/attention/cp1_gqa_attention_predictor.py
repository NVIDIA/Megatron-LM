# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

#!/usr/bin/env python3
"""Standalone H100 CP1 packed-THD GQA forward-attention time predictor.

The returned value is forward-only time in milliseconds for one isolated
Transformer Engine dot-product-attention call. Backward, QKV projection, RoPE,
output projection, MLP/MoE, optimizer, and other end-to-end work are excluded.

The six supported GQA geometries use independently measured models. Under the
PR5541 layout contract, every real sequence is aligned to
``2 * max_cp_size`` before the final pack boundary is filled. Storage validation
uses those per-sequence-aligned physical lengths, not the logical-token sum.
Tail padding only fills the final real sequence's padded boundary: it is not a
sequence, is not attention work, and cannot change the predicted time for a
fixed pack. The built-in legacy H100 artifact covers 3K-8K; external artifacts
may add a 16K length region.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MIN_TOTAL = 3072
_MAX_TOTAL = 8192
_DEFAULT_SEQUENCE_ALIGNMENT = 1
_DEFAULT_MAX_CP_SIZE = 1
_DEFAULT_PACK_ALIGNMENT = 4096
_ALIGNMENT_HINGE_THRESHOLDS_M = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
)


@dataclass(frozen=True)
class _LinearModel:
    kind: str
    features: tuple[str, ...]
    coefficients: tuple[float, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]


@dataclass(frozen=True)
class _GeometryModels:
    name: str
    by_storage: dict[int, _LinearModel]
    floor_by_storage: dict[int, _LinearModel] = field(default_factory=dict)
    floor_work_threshold_m: float | None = None
    standard_sequence_count_limit: int | None = None


@dataclass(frozen=True)
class CP1PhysicalLayout:
    logical_lengths: tuple[int, ...]
    per_sequence_padded_lengths: tuple[int, ...]
    final_padded_lengths: tuple[int, ...]
    max_cp_size: int
    sequence_alignment: int
    physical_input_tokens: int
    storage_tokens: int
    per_sequence_padding_tokens: int
    tail_padding_tokens: int


class FrozenCP1GQAAttentionPredictor:
    """Evaluate a frozen CP1 attention-timing artifact."""

    def __init__(self, payload: dict[str, Any]) -> None:
        contract = payload["contract"]
        if int(contract["cp"]) != 1 or contract["phase"] != "forward":
            raise ValueError("CP1 artifact must contain forward-only CP1 timing")
        self.payload = payload
        self.gpu = str(contract["gpu"])
        self.minimum_total = int(contract["valid_total_tokens"][0])
        self.maximum_total = int(contract["valid_total_tokens"][1])
        self.minimum_sequence_tokens = int(contract["minimum_sequence_tokens"])
        self.minimum_sequence_count = int(contract["measured_sequence_count"][0])
        self.maximum_sequence_count = int(contract["measured_sequence_count"][1])
        alignment_contract = contract.get("per_sequence_alignment")
        if alignment_contract is not None:
            if alignment_contract.get("formula") != "2 * max_cp_size":
                raise ValueError("unsupported CP1 per-sequence alignment formula")
            self.calibration_max_cp_size = _positive_int(
                int(alignment_contract["calibration_max_cp_size"]),
                "calibration_max_cp_size",
            )
            self.sequence_alignment = 2 * self.calibration_max_cp_size
            calibration_tokens = int(
                alignment_contract.get("calibration_tokens", self.sequence_alignment)
            )
            if calibration_tokens != self.sequence_alignment:
                raise ValueError(
                    "CP1 calibration alignment must equal 2 * calibration_max_cp_size"
                )
            self.measured_max_cp_sizes = tuple(
                _positive_int(int(value), "measured_max_cp_size")
                for value in alignment_contract.get(
                    "measured_max_cp_sizes", [self.calibration_max_cp_size]
                )
            )
        else:
            # Legacy artifacts stored only the calibration alignment. Preserve
            # their default behavior; explicit max_cp_size still uses PR5541.
            self.sequence_alignment = int(
                contract.get(
                    "per_sequence_alignment_tokens", _DEFAULT_SEQUENCE_ALIGNMENT
                )
            )
            self.calibration_max_cp_size = max(1, self.sequence_alignment // 2)
            self.measured_max_cp_sizes = (self.calibration_max_cp_size,)
        self.pack_alignment = int(
            contract.get("pack_alignment_tokens", _DEFAULT_PACK_ALIGNMENT)
        )
        if "arbitrary_branch_dispatch" in contract or "arbitrary_models" in payload:
            raise ValueError(
                "P-dependent CP1 arbitrary-branch artifacts are unsupported; "
                "use a padding-invariant artifact"
            )
        work_dispatch = contract.get("attention_work_dispatch")
        if (
            work_dispatch is not None
            and work_dispatch.get("work_feature") != "tile256_work_m"
        ):
            raise ValueError("CP1 attention-work dispatch must use tile256_work_m")
        work_dispatch_by_profile = (
            {} if work_dispatch is None else work_dispatch.get("profiles", {})
        )
        if self.sequence_alignment <= 0 or self.pack_alignment <= 0:
            raise ValueError("CP1 artifact alignments must be positive")
        profiles = {profile["name"]: profile for profile in payload["profiles"]}
        self.models_by_geometry: dict[tuple[int, int, int], _GeometryModels] = {}
        for name, by_storage in payload["models"].items():
            if name not in profiles:
                raise ValueError(f"CP1 artifact model {name!r} has no profile metadata")
            profile = profiles[name]
            geometry = (
                int(profile["q_heads"]),
                int(profile["kv_heads"]),
                int(profile["head_dim"]),
            )
            if geometry in self.models_by_geometry:
                raise ValueError(f"duplicate CP1 geometry {geometry}")
            branches = {}
            for storage_text, raw in by_storage.items():
                storage = _positive_int(int(storage_text), "CP1 storage branch")
                if storage % self.pack_alignment:
                    raise ValueError(
                        f"CP1 storage branch {storage} is not divisible by "
                        f"pack alignment {self.pack_alignment}"
                    )
                branches[storage] = _LinearModel(
                    kind=str(raw["kind"]),
                    features=tuple(str(value) for value in raw["features"]),
                    coefficients=tuple(float(value) for value in raw["coefficients"]),
                    centers=tuple(float(value) for value in raw["feature_centers"]),
                    scales=tuple(float(value) for value in raw["feature_scales"]),
                )
            floor_branches = {}
            for storage_text, raw in (
                payload.get("floor_models", {}).get(name, {}).items()
            ):
                storage = _positive_int(int(storage_text), "CP1 floor branch")
                if storage not in branches:
                    raise ValueError(
                        f"CP1 floor branch {storage} has no standard branch"
                    )
                floor_branches[storage] = _LinearModel(
                    kind=str(raw["kind"]),
                    features=tuple(str(value) for value in raw["features"]),
                    coefficients=tuple(float(value) for value in raw["coefficients"]),
                    centers=tuple(float(value) for value in raw["feature_centers"]),
                    scales=tuple(float(value) for value in raw["feature_scales"]),
                )
            dispatch = work_dispatch_by_profile.get(name)
            if floor_branches and dispatch is None:
                raise ValueError("CP1 floor models require an attention-work dispatch")
            if dispatch is not None and not floor_branches:
                raise ValueError(
                    f"CP1 profile {name!r} has attention-work dispatch but no floor models"
                )
            self.models_by_geometry[geometry] = _GeometryModels(
                name=name,
                by_storage=branches,
                floor_by_storage=floor_branches,
                floor_work_threshold_m=(
                    None
                    if dispatch is None
                    else float(dispatch["floor_below_work_m"])
                ),
                standard_sequence_count_limit=(
                    None
                    if dispatch is None
                    else _positive_int(
                        int(dispatch["standard_sequence_count_limit"]),
                        "standard_sequence_count_limit",
                    )
                ),
            )

    @classmethod
    def from_path(cls, path: Path | str) -> "FrozenCP1GQAAttentionPredictor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def has_geometry(self, geometry: tuple[int, int, int]) -> bool:
        return geometry in self.models_by_geometry

    def sequence_alignment_for(self, max_cp_size: int | None = None) -> int:
        """Resolve PR5541 alignment, preserving legacy artifact defaults."""
        if max_cp_size is None:
            return self.sequence_alignment
        return sequence_alignment_for_max_cp_size(max_cp_size)

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
        return _predict_from_geometry_models(
            sequence_lengths,
            query_heads,
            kv_heads,
            head_dim,
            self.models_by_geometry,
            minimum_total=self.minimum_total,
            maximum_total=self.maximum_total,
            minimum_sequence_tokens=self.minimum_sequence_tokens,
            minimum_sequence_count=self.minimum_sequence_count,
            maximum_sequence_count=self.maximum_sequence_count,
            sequence_alignment=self.sequence_alignment_for(max_cp_size),
            pack_alignment=self.pack_alignment,
            allow_extrapolation=allow_extrapolation,
            storage_tokens=storage_tokens,
        )

    def model_storage_tokens(
        self,
        sequence_lengths: Sequence[int],
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        *,
        max_cp_size: int | None = None,
    ) -> int:
        """Return the calibrated length region selected by the real pack."""
        geometry = (int(query_heads), int(kv_heads), int(head_dim))
        geometry_models = self.models_by_geometry.get(geometry)
        if geometry_models is None:
            storage_sets = {
                tuple(sorted(models.by_storage))
                for models in self.models_by_geometry.values()
            }
            if len(storage_sets) != 1:
                raise ValueError(
                    f"no measured CP1 branch for geometry {geometry} and calibrated "
                    "length regions differ by geometry"
                )
            available_storages = tuple(
                sorted(next(iter(self.models_by_geometry.values())).by_storage)
            )
        else:
            available_storages = tuple(sorted(geometry_models.by_storage))
        layout = resolve_cp1_physical_layout(
            sequence_lengths,
            sequence_alignment=self.sequence_alignment_for(max_cp_size),
            pack_alignment=self.pack_alignment,
            maximum_storage=max(available_storages),
            available_storages=available_storages,
        )
        return layout.storage_tokens

    def physical_layout(
        self,
        sequence_lengths: Sequence[int],
        *,
        max_cp_size: int | None = None,
        storage_tokens: int | None = None,
    ) -> CP1PhysicalLayout:
        available_storages = tuple(
            sorted(
                {
                    storage
                    for geometry in self.models_by_geometry.values()
                    for storage in geometry.by_storage
                }
            )
        )
        return resolve_cp1_physical_layout(
            sequence_lengths,
            sequence_alignment=self.sequence_alignment_for(max_cp_size),
            pack_alignment=self.pack_alignment,
            maximum_storage=max(available_storages),
            target_storage=storage_tokens,
            available_storages=available_storages,
        )


_PER_GEOMETRY_MODELS = {
    (32, 4, 128): _GeometryModels(
        "qwen3_30b",
        {
            4096: _LinearModel(
                "linear_time",
                ("s2_m",),
                (0.18976877777777773, -0.00040167674295392647),
                (2.5135786666666666,),
                (0.6859587395327312,),
            ),
            8192: _LinearModel(
                "linear_time",
                ("tile128_hinge_7p0m",),
                (0.24030716666666663, 0.08867580759346894),
                (3.5747022222222222,),
                (6.290566830895533,),
            ),
        },
    ),
    (64, 4, 128): _GeometryModels(
        "qwen3_235b",
        {
            4096: _LinearModel(
                "linear_log",
                ("total_k", "s2_m", "sequence_count"),
                (
                    -1.6577132195555977,
                    -0.026380390580378388,
                    0.03672332887198114,
                    0.029445835588573433,
                ),
                (3.5, 2.5135786666666666, 5.611111111111111),
                (0.3227486121839514, 0.6859587395327311, 1.1124991330278213),
            ),
            8192: _LinearModel(
                "linear_time",
                ("total_k", "s2_m", "sequence_count", "tile128_work_m"),
                (
                    0.39851316666666664,
                    0.014092927709634285,
                    0.06822654959089637,
                    -0.0007295278201384321,
                    0.1336841419507821,
                ),
                (
                    6.166666666666667,
                    9.077191111111112,
                    7.111111111111111,
                    9.24512711111111,
                ),
                (
                    1.1960583412006103,
                    7.131898340797701,
                    2.6010444246043614,
                    7.1679215759801576,
                ),
            ),
        },
    ),
    (16, 2, 256): _GeometryModels(
        "qwen3_next_80b",
        {
            4096: _LinearModel(
                "linear_log",
                ("s2_m",),
                (-1.663231440895659, -0.0020526129897760403),
                (2.5135786666666666,),
                (0.6859587395327312,),
            ),
            8192: _LinearModel(
                "linear_time",
                (
                    "tile128_hinge_8p0m",
                    "tile128_hinge_8p0m_times_n",
                    "tile128_hinge_8p0m_times_max_k",
                ),
                (
                    0.23440894444444446,
                    0.031009615573479088,
                    0.04850615261628625,
                    0.0023490771462184156,
                ),
                (3.0995342222222226, 13.177201777777775, 12.610243555555556),
                (6.006838676402122, 24.002509514580545, 27.86066392827255),
            ),
        },
    ),
    (32, 2, 256): _GeometryModels(
        "qwen3_5_397b",
        {
            4096: _LinearModel(
                "linear_log",
                ("s2_m",),
                (-1.662263895419861, -0.0015809145893165221),
                (2.5135786666666666,),
                (0.6859587395327312,),
            ),
            8192: _LinearModel(
                "linear_time",
                ("total_k", "s2_m", "sequence_count", "tile256_work_m"),
                (
                    0.36736105555555554,
                    0.012229155977067323,
                    0.09091898625873313,
                    -0.0017353237180074738,
                    0.0892735267366835,
                ),
                (
                    6.166666666666667,
                    9.077191111111112,
                    7.111111111111111,
                    9.661098666666668,
                ),
                (
                    1.1960583412006103,
                    7.131898340797701,
                    2.6010444246043614,
                    7.335109460478297,
                ),
            ),
        },
    ),
    (32, 8, 128): _GeometryModels(
        "mixtral_8x7b",
        {
            4096: _LinearModel(
                "linear_time",
                ("s2_m",),
                (0.18968022222222222, -0.00046896290728953366),
                (2.5135786666666666,),
                (0.6859587395327312,),
            ),
            8192: _LinearModel(
                "linear_time",
                ("s2_hinge_6p5m",),
                (0.2412728333333333, 0.08997329365768073),
                (3.7298311111111113,),
                (6.357405493349354,),
            ),
        },
    ),
    (64, 8, 128): _GeometryModels(
        "llama3_70b",
        {
            4096: _LinearModel(
                "linear_log",
                ("total_k", "s2_m", "sequence_count", "tile128_work_m"),
                (
                    -1.6553566586599322,
                    -0.02813081277353232,
                    -0.0007319058547441162,
                    0.035520253914708394,
                    0.043703479615790526,
                ),
                (3.5, 2.5135786666666666, 5.611111111111111, 2.6005048888888886),
                (
                    0.3227486121839514,
                    0.6859587395327311,
                    1.1124991330278213,
                    0.6943050771828452,
                ),
            ),
            8192: _LinearModel(
                "linear_time",
                ("total_k", "s2_m", "sequence_count", "tile128_work_m"),
                (
                    0.40002094444444447,
                    0.016450639745386374,
                    0.09568207714156456,
                    -0.0031316080301419433,
                    0.10449008072830454,
                ),
                (
                    6.166666666666667,
                    9.077191111111112,
                    7.111111111111111,
                    9.24512711111111,
                ),
                (
                    1.1960583412006103,
                    7.131898340797701,
                    2.6010444246043614,
                    7.1679215759801576,
                ),
            ),
        },
    ),
}


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def sequence_alignment_for_max_cp_size(max_cp_size: int) -> int:
    """Return the PR5541 per-sequence alignment for a scheduler CP limit."""
    return 2 * _positive_int(max_cp_size, "max_cp_size")


def resolve_cp1_physical_layout(
    sequence_lengths: Sequence[int],
    *,
    max_cp_size: int | None = None,
    sequence_alignment: int | None = None,
    pack_alignment: int,
    maximum_storage: int = _MAX_TOTAL,
    target_storage: int | None = None,
    available_storages: Sequence[int] | None = None,
) -> CP1PhysicalLayout:
    """Rebuild the two-stage physical padding used by packed THD CP1."""
    lengths = tuple(
        _positive_int(length, f"sequence_lengths[{index}]")
        for index, length in enumerate(sequence_lengths)
    )
    if not lengths:
        raise ValueError("sequence_lengths must not be empty")
    if max_cp_size is not None and sequence_alignment is not None:
        raise ValueError("provide max_cp_size or sequence_alignment, not both")
    if max_cp_size is None and sequence_alignment is None:
        max_cp_size = _DEFAULT_MAX_CP_SIZE
    if max_cp_size is not None:
        max_cp_size = _positive_int(max_cp_size, "max_cp_size")
        sequence_alignment = sequence_alignment_for_max_cp_size(max_cp_size)
    else:
        sequence_alignment = _positive_int(sequence_alignment, "sequence_alignment")
        max_cp_size = max(1, sequence_alignment // 2)
    pack_alignment = _positive_int(pack_alignment, "pack_alignment")
    maximum_storage = _positive_int(maximum_storage, "maximum_storage")

    per_sequence_padded = tuple(
        _round_up(length, sequence_alignment) for length in lengths
    )
    physical_input = sum(per_sequence_padded)
    calibrated_storages = ()
    if available_storages is not None:
        calibrated_storages = tuple(
            sorted(
                {
                    _positive_int(int(value), "available_storage")
                    for value in available_storages
                }
            )
        )
        if not calibrated_storages:
            raise ValueError("available_storages must not be empty")
        if calibrated_storages[-1] > maximum_storage:
            raise ValueError(
                "available_storages cannot exceed maximum_storage "
                f"{maximum_storage}"
            )
        if any(value % pack_alignment for value in calibrated_storages):
            raise ValueError(
                "every available storage must be divisible by pack_alignment "
                f"{pack_alignment}"
            )

    if target_storage is not None:
        storage = _positive_int(target_storage, "target_storage")
        if calibrated_storages and storage not in calibrated_storages:
            raise ValueError(
                f"target_storage {storage} is not a calibrated CP1 branch; "
                f"available: {calibrated_storages}"
            )
    elif calibrated_storages:
        storage = next(
            (value for value in calibrated_storages if value >= physical_input),
            _round_up(physical_input, pack_alignment),
        )
    else:
        storage = _round_up(physical_input, pack_alignment)
    if target_storage is not None and storage % pack_alignment:
        raise ValueError(
            f"target_storage {storage} must be divisible by pack_alignment "
            f"{pack_alignment}"
        )
    if physical_input > storage:
        raise ValueError(
            f"per-sequence-aligned physical total {physical_input} exceeds "
            f"the requested CP1 storage {storage}"
        )
    if storage > maximum_storage:
        raise ValueError(
            f"per-sequence-aligned physical total {physical_input} requires "
            f"{storage} storage tokens, exceeding the CP1 {maximum_storage}-token limit"
        )
    final_padded = list(per_sequence_padded)
    final_padded[-1] += storage - physical_input
    return CP1PhysicalLayout(
        logical_lengths=lengths,
        per_sequence_padded_lengths=per_sequence_padded,
        final_padded_lengths=tuple(final_padded),
        max_cp_size=max_cp_size,
        sequence_alignment=sequence_alignment,
        physical_input_tokens=physical_input,
        storage_tokens=storage,
        per_sequence_padding_tokens=physical_input - sum(lengths),
        tail_padding_tokens=storage - physical_input,
    )


def _length_features(
    lengths: Sequence[int],
    sequence_alignment: int = _DEFAULT_SEQUENCE_ALIGNMENT,
    storage_tokens: int | None = None,
    *,
    include_legacy_tile_features: bool = False,
) -> dict[str, float]:
    logical_tokens = sum(lengths)
    sum_squared_tokens = sum(length * length for length in lengths)
    physical_lengths = [_round_up(length, sequence_alignment) for length in lengths]
    alignment_padding = [
        physical - logical for physical, logical in zip(physical_lengths, lengths)
    ]
    physical_input_tokens = sum(physical_lengths)
    alignment_padding_tokens = sum(alignment_padding)
    alignment_padding_weighted = sum(
        padding * length for padding, length in zip(alignment_padding, lengths)
    )
    s2_m = sum_squared_tokens / 1.0e6
    max_k = max(lengths) / 1024.0
    sequence_count = float(len(lengths))
    sequence_capacity = (
        storage_tokens // sequence_alignment
        if storage_tokens is not None
        else len(lengths)
    )
    features = {
        "total_k": logical_tokens / 1024.0,
        "s2_m": s2_m,
        "sqrt_s2_k": math.sqrt(sum_squared_tokens) / 1024.0,
        "sequence_count": sequence_count,
        "log2_sequence_count": math.log2(len(lengths)),
        "sequence_count_sq_k": len(lengths) * len(lengths) / 1024.0,
        "n_hinge_quarter_capacity": float(
            max(0, len(lengths) - sequence_capacity // 4)
        ),
        "n_hinge_half_capacity": float(
            max(0, len(lengths) - sequence_capacity // 2)
        ),
        "max_k": max_k,
        "min_k": min(lengths) / 1024.0,
        "mean_k": logical_tokens / len(lengths) / 1024.0,
        "physical_input_k": physical_input_tokens / 1024.0,
        "per_sequence_padding_k": (physical_input_tokens - logical_tokens) / 1024.0,
        "alignment_padding_k": alignment_padding_tokens / 1024.0,
        "alignment_padding_weighted_m": alignment_padding_weighted / 1.0e6,
        "alignment_padding_fraction": alignment_padding_tokens / logical_tokens,
        "aligned_work_proxy_m": (
            sum_squared_tokens + alignment_padding_weighted
        )
        / 1.0e6,
        "physical_s2_m": sum(length * length for length in physical_lengths) / 1.0e6,
        "max_physical_k": max(physical_lengths) / 1024.0,
        "unaligned_count": float(
            sum(
                logical != physical
                for logical, physical in zip(lengths, physical_lengths)
            )
        ),
        "pad_between_seqs": float(
            any(
                logical != physical
                for logical, physical in zip(lengths, physical_lengths)
            )
        ),
    }
    for source_label, source_feature in (("s2", "s2_m"),):
        for half_step in range(8, 25):
            threshold = half_step / 2.0
            threshold_label = f"{threshold:.1f}".replace(".", "p")
            name = f"{source_label}_hinge_{threshold_label}m"
            hinge = max(0.0, features[source_feature] - threshold)
            features[name] = hinge
            features[f"{name}_times_n"] = hinge * sequence_count
            features[f"{name}_times_max_k"] = hinge * max_k
    if include_legacy_tile_features:
        # Old frozen artifacts used empirical 128/256-token proxies. Keep them
        # loadable without making those unexplained constants part of the
        # current max-CP-alignment formula.
        features["tile128_work_m"] = (
            sum(_round_up(length, 128) * length for length in lengths) / 1.0e6
        )
        features["tile256_work_m"] = (
            sum(_round_up(length, 256) * length for length in lengths) / 1.0e6
        )
        for source_label, source_feature in (
            ("tile128", "tile128_work_m"),
            ("tile256", "tile256_work_m"),
        ):
            for half_step in range(8, 25):
                threshold = half_step / 2.0
                threshold_label = f"{threshold:.1f}".replace(".", "p")
                name = f"{source_label}_hinge_{threshold_label}m"
                hinge = max(0.0, features[source_feature] - threshold)
                features[name] = hinge
                features[f"{name}_times_n"] = hinge * sequence_count
                features[f"{name}_times_max_k"] = hinge * max_k
    for source_label, source_feature in (
        ("s2", "s2_m"),
        ("aligned", "aligned_work_proxy_m"),
    ):
        for threshold in _ALIGNMENT_HINGE_THRESHOLDS_M:
            precision = 1 if float(threshold * 2).is_integer() else 2
            threshold_label = f"{threshold:.{precision}f}".replace(".", "p")
            name = f"{source_label}_hinge_{threshold_label}m"
            hinge = max(0.0, features[source_feature] - threshold)
            features[name] = hinge
            features[f"{name}_times_n"] = hinge * sequence_count
            features[f"{name}_times_max_k"] = hinge * max_k
    return features


def _predict_from_geometry_models(
    sequence_lengths: Sequence[int],
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    models_by_geometry: dict[tuple[int, int, int], _GeometryModels],
    *,
    minimum_total: int,
    maximum_total: int,
    minimum_sequence_tokens: int,
    minimum_sequence_count: int,
    maximum_sequence_count: int,
    sequence_alignment: int,
    pack_alignment: int,
    allow_extrapolation: bool,
    storage_tokens: int | None = None,
) -> float:
    lengths = [
        _positive_int(length, f"sequence_lengths[{index}]")
        for index, length in enumerate(sequence_lengths)
    ]
    if not lengths:
        raise ValueError("sequence_lengths must not be empty")
    query_heads = _positive_int(query_heads, "query_heads")
    kv_heads = _positive_int(kv_heads, "kv_heads")
    head_dim = _positive_int(head_dim, "head_dim")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads for GQA")

    geometry = (query_heads, kv_heads, head_dim)
    geometry_models = models_by_geometry.get(geometry)
    if geometry_models is None:
        supported = ", ".join(
            f"{hq}/{hkv}/{dimension}" for hq, hkv, dimension in models_by_geometry
        )
        raise ValueError(
            "no measured CP1 branch for geometry "
            f"{query_heads}/{kv_heads}/{head_dim}; supported: {supported}"
        )

    logical_tokens = sum(lengths)
    if logical_tokens > maximum_total:
        raise ValueError(
            f"logical total {logical_tokens} exceeds the CP1 "
            f"{maximum_total}-token physical limit"
        )
    if not allow_extrapolation:
        outside = []
        if not minimum_total <= logical_tokens <= maximum_total:
            outside.append(
                f"logical_tokens={logical_tokens} "
                f"(measured: {minimum_total}-{maximum_total})"
            )
        if not minimum_sequence_count <= len(lengths) <= maximum_sequence_count:
            outside.append(
                f"sequence_count={len(lengths)} "
                f"(measured: {minimum_sequence_count}-{maximum_sequence_count})"
            )
        if min(lengths) < minimum_sequence_tokens:
            outside.append(
                f"minimum_length={min(lengths)} "
                f"(measured minimum: {minimum_sequence_tokens})"
            )
        if outside:
            raise ValueError(
                "input is outside the measured envelope: " + "; ".join(outside)
            )

    layout = resolve_cp1_physical_layout(
        lengths,
        sequence_alignment=sequence_alignment,
        pack_alignment=pack_alignment,
        maximum_storage=max(geometry_models.by_storage),
        target_storage=storage_tokens,
        available_storages=tuple(geometry_models.by_storage),
    )
    # The explicit layout storage is only a capacity and tail-padding choice.
    # Select the timing region from the smallest calibrated branch that holds
    # the real, per-sequence-aligned pack, so extra tail pad cannot affect time.
    model_storage = next(
        (
            value
            for value in sorted(geometry_models.by_storage)
            if value >= layout.physical_input_tokens
        ),
        None,
    )
    if model_storage is None:
        raise ValueError(
            f"CP1 artifact has no length region for "
            f"{layout.physical_input_tokens} physical input tokens"
        )
    candidate_models = [geometry_models.by_storage[model_storage]]
    if geometry_models.floor_by_storage:
        candidate_models.append(geometry_models.floor_by_storage[model_storage])
    include_legacy_tiles = bool(geometry_models.floor_by_storage) or any(
        name.startswith(("tile128", "tile256"))
        for candidate in candidate_models
        for name in candidate.features
    )
    features = _length_features(
        lengths,
        sequence_alignment,
        model_storage,
        include_legacy_tile_features=include_legacy_tiles,
    )
    use_floor = False
    if geometry_models.floor_by_storage:
        threshold = geometry_models.floor_work_threshold_m
        sequence_count_limit = geometry_models.standard_sequence_count_limit
        if threshold is None or sequence_count_limit is None:
            raise ValueError("CP1 artifact lacks attention-work dispatch thresholds")
        use_floor = (
            features["tile256_work_m"] < threshold
            or len(lengths) > sequence_count_limit
        )
    model = (
        geometry_models.floor_by_storage[model_storage]
        if use_floor
        else geometry_models.by_storage[model_storage]
    )
    prediction = model.coefficients[0]
    for index, name in enumerate(model.features):
        standardized = (features[name] - model.centers[index]) / model.scales[index]
        prediction += model.coefficients[index + 1] * standardized
    if model.kind == "linear_log":
        prediction = math.exp(prediction)
    if prediction <= 0:
        raise ValueError(
            f"model produced a non-positive time ({prediction:.6f} ms); "
            "the input is too far outside the fitted region"
        )
    return prediction


def predict_cp1_gqa_attention_time_ms(
    sequence_lengths: Sequence[int],
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    max_cp_size: int | None = None,
    allow_extrapolation: bool = False,
    model_artifact: FrozenCP1GQAAttentionPredictor | None = None,
    storage_tokens: int | None = None,
) -> float:
    """Predict BF16 causal GQA core-attention forward time.

    The built-in artifact is H100. Pass ``model_artifact`` for another measured
    hardware branch. Tail padding is storage-only: it is not an extra sequence
    and does not change the prediction for fixed logical sequence lengths.
    """
    if model_artifact is not None:
        return model_artifact.predict_time_ms(
            sequence_lengths,
            query_heads,
            kv_heads,
            head_dim,
            max_cp_size=max_cp_size,
            allow_extrapolation=allow_extrapolation,
            storage_tokens=storage_tokens,
        )
    return _predict_from_geometry_models(
        sequence_lengths,
        query_heads,
        kv_heads,
        head_dim,
        _PER_GEOMETRY_MODELS,
        minimum_total=_MIN_TOTAL,
        maximum_total=_MAX_TOTAL,
        minimum_sequence_tokens=512,
        minimum_sequence_count=4,
        maximum_sequence_count=12,
        sequence_alignment=sequence_alignment_for_max_cp_size(
            _DEFAULT_MAX_CP_SIZE if max_cp_size is None else max_cp_size
        ),
        pack_alignment=_DEFAULT_PACK_ALIGNMENT,
        allow_extrapolation=allow_extrapolation,
        storage_tokens=storage_tokens,
    )


def predict_cp1_gqa_attention_with_variable_heads_details(
    sequence_lengths: Sequence[int],
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    max_cp_size: int | None = None,
    allow_extrapolation: bool = False,
    model_artifact: FrozenCP1GQAAttentionPredictor | None = None,
    storage_tokens: int | None = None,
) -> dict[str, Any]:
    """Predict CP1 time while allowing unmeasured head counts.

    Every measured geometry dispatches directly to
    :func:`predict_cp1_gqa_attention_time_ms`. Unmeasured head counts use a
    shape-specific log-time surface through the measured branches, so enabling
    this fallback cannot change predictions for the six calibrated geometries.
    Only head dimensions 128 and 256 are supported.
    """
    lengths = [
        _positive_int(length, f"sequence_lengths[{index}]")
        for index, length in enumerate(sequence_lengths)
    ]
    if not lengths:
        raise ValueError("sequence_lengths must not be empty")

    query_heads = _positive_int(query_heads, "query_heads")
    kv_heads = _positive_int(kv_heads, "kv_heads")
    head_dim = _positive_int(head_dim, "head_dim")
    if query_heads % kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads for GQA")

    geometry = (query_heads, kv_heads, head_dim)
    measured_models = (
        model_artifact.models_by_geometry
        if model_artifact is not None
        else _PER_GEOMETRY_MODELS
    )
    if geometry in measured_models:
        time_ms = predict_cp1_gqa_attention_time_ms(
            lengths,
            query_heads,
            kv_heads,
            head_dim,
            max_cp_size=max_cp_size,
            allow_extrapolation=allow_extrapolation,
            model_artifact=model_artifact,
            storage_tokens=storage_tokens,
        )
        return {
            "time_ms": time_ms,
            "geometry": geometry,
            "method": "exact_measured_geometry",
            "measured_geometry": True,
            "head_count_extrapolation": False,
            "anchors_ms": {},
        }

    minimum_total = (
        model_artifact.minimum_total if model_artifact is not None else _MIN_TOTAL
    )
    maximum_total = (
        model_artifact.maximum_total if model_artifact is not None else _MAX_TOTAL
    )
    minimum_sequence_tokens = (
        model_artifact.minimum_sequence_tokens if model_artifact is not None else 512
    )
    minimum_sequence_count = (
        model_artifact.minimum_sequence_count if model_artifact is not None else 4
    )
    maximum_sequence_count = (
        model_artifact.maximum_sequence_count if model_artifact is not None else 12
    )
    logical_tokens = sum(lengths)
    if logical_tokens > maximum_total:
        raise ValueError(
            f"logical total {logical_tokens} exceeds the CP1 "
            f"{maximum_total}-token physical limit"
        )
    if not allow_extrapolation:
        outside = []
        if not minimum_total <= logical_tokens <= maximum_total:
            outside.append(
                f"logical_tokens={logical_tokens} "
                f"(measured: {minimum_total}-{maximum_total})"
            )
        if not minimum_sequence_count <= len(lengths) <= maximum_sequence_count:
            outside.append(
                f"sequence_count={len(lengths)} "
                f"(measured: {minimum_sequence_count}-{maximum_sequence_count})"
            )
        if min(lengths) < minimum_sequence_tokens:
            outside.append(
                f"minimum_length={min(lengths)} "
                f"(measured minimum: {minimum_sequence_tokens})"
            )
        if outside:
            raise ValueError(
                "input is outside the measured envelope: " + "; ".join(outside)
            )
    if head_dim not in (128, 256):
        raise ValueError(
            "variable-head fallback only supports measured head dimensions 128 and 256"
        )

    def anchor(anchor_query_heads: int, anchor_kv_heads: int, dimension: int) -> float:
        return predict_cp1_gqa_attention_time_ms(
            lengths,
            anchor_query_heads,
            anchor_kv_heads,
            dimension,
            max_cp_size=max_cp_size,
            allow_extrapolation=True,
            model_artifact=model_artifact,
            storage_tokens=storage_tokens,
        )

    if head_dim == 128:
        anchors = {
            "q32_kv4_d128": anchor(32, 4, 128),
            "q64_kv4_d128": anchor(64, 4, 128),
            "q32_kv8_d128": anchor(32, 8, 128),
            "q64_kv8_d128": anchor(64, 8, 128),
        }
        query_coordinate = math.log2(query_heads / 32.0)
        kv_coordinate = math.log2(kv_heads / 4.0)
        f00 = math.log(anchors["q32_kv4_d128"])
        f10 = math.log(anchors["q64_kv4_d128"])
        f01 = math.log(anchors["q32_kv8_d128"])
        f11 = math.log(anchors["q64_kv8_d128"])
        interaction = f11 - f10 - f01 + f00
        log_time_ms = (
            f00
            + query_coordinate * (f10 - f00)
            + kv_coordinate * (f01 - f00)
            + query_coordinate * kv_coordinate * interaction
        )
        method = "log_bilinear_q32_64_kv4_8_d128"
        inside_head_box = 32 <= query_heads <= 64 and 4 <= kv_heads <= 8
    else:
        anchors = {
            "q16_kv2_d256": anchor(16, 2, 256),
            "q32_kv2_d256": anchor(32, 2, 256),
            "q32_kv4_d128": anchor(32, 4, 128),
            "q32_kv8_d128": anchor(32, 8, 128),
            "q64_kv4_d128": anchor(64, 4, 128),
            "q64_kv8_d128": anchor(64, 8, 128),
        }
        query_coordinate = math.log2(query_heads / 16.0)
        log_q16 = math.log(anchors["q16_kv2_d256"])
        log_q32 = math.log(anchors["q32_kv2_d256"])
        log_time_ms = log_q16 + query_coordinate * (log_q32 - log_q16)

        # D=256 has no measured KV-head sweep. Borrow only that weak slope
        # from D=128 at the equivalent query-channel endpoints.
        kv_slope_q16 = math.log2(anchors["q32_kv8_d128"] / anchors["q32_kv4_d128"])
        kv_slope_q32 = math.log2(anchors["q64_kv8_d128"] / anchors["q64_kv4_d128"])
        kv_slope = kv_slope_q16 + query_coordinate * (kv_slope_q32 - kv_slope_q16)
        log_time_ms += math.log2(kv_heads / 2.0) * kv_slope * math.log(2.0)
        method = "log_power_q16_32_d256_with_d128_kv_slope"
        inside_head_box = 16 <= query_heads <= 32 and kv_heads == 2

    time_ms = math.exp(log_time_ms)
    if not math.isfinite(time_ms) or time_ms <= 0:
        raise ValueError(
            "variable-head model produced a non-positive or non-finite time"
        )
    return {
        "time_ms": time_ms,
        "geometry": geometry,
        "method": method,
        "measured_geometry": False,
        "head_count_extrapolation": not inside_head_box,
        "anchors_ms": anchors,
    }


def predict_cp1_gqa_attention_time_ms_variable_heads(
    sequence_lengths: Sequence[int],
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    max_cp_size: int | None = None,
    allow_extrapolation: bool = False,
    model_artifact: FrozenCP1GQAAttentionPredictor | None = None,
    storage_tokens: int | None = None,
) -> float:
    """Return CP1 time, using head-count fallback only when needed."""
    return float(
        predict_cp1_gqa_attention_with_variable_heads_details(
            sequence_lengths,
            query_heads,
            kv_heads,
            head_dim,
            max_cp_size=max_cp_size,
            allow_extrapolation=allow_extrapolation,
            model_artifact=model_artifact,
            storage_tokens=storage_tokens,
        )["time_ms"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        required=True,
        help="Comma-separated logical sequence lengths, for example 512,1024,2048,3584",
    )
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument(
        "--max-cp-size",
        type=int,
        help=(
            "Scheduler maximum CP size; per-sequence alignment is twice this "
            "value. Defaults to the artifact calibration value."
        ),
    )
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional frozen CP1 artifact; omit to use the built-in H100 model.",
    )
    parser.add_argument(
        "--storage-tokens",
        type=int,
        choices=(4096, 8192, 16384),
        help=(
            "Explicit physical CP1 layout capacity. Extra tail padding does not "
            "change attention time."
        ),
    )
    parser.add_argument(
        "--variable-heads",
        action="store_true",
        help="Allow approximate unmeasured query/KV head counts.",
    )
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",") if value]
    predictor = (
        predict_cp1_gqa_attention_time_ms_variable_heads
        if args.variable_heads
        else predict_cp1_gqa_attention_time_ms
    )
    prediction_ms = predictor(
        lengths,
        args.query_heads,
        args.kv_heads,
        args.head_dim,
        max_cp_size=args.max_cp_size,
        allow_extrapolation=args.allow_extrapolation,
        model_artifact=(
            FrozenCP1GQAAttentionPredictor.from_path(args.model)
            if args.model is not None
            else None
        ),
        storage_tokens=args.storage_tokens,
    )
    print(f"{prediction_ms:.9f}")


if __name__ == "__main__":
    main()
