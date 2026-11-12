# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Stable, CPU-only API around explicitly selected frozen model snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ._native.geometry import GEOMETRIES, Pack, positive_int
from ._native.whole_resource_cp1 import predict_ms


@lru_cache(maxsize=2)
def _model(hardware):
    root = Path(__file__).resolve().parent / "data"
    name = f"experimental_cp1_{hardware}.json"
    blob = (root / name).read_bytes()
    expected = json.loads((root / "manifest.json").read_text())[name]
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("CP1 parameter artifact hash mismatch")
    return json.loads(blob), expected


@dataclass(frozen=True)
class AttentionEstimate:
    time_ms: float
    phase: str
    cp_size: int
    hardware: str
    geometry: tuple[int, int, int]
    profile: str
    pack: Pack
    ranks: tuple[int, ...] | None
    components_ms: Mapping[str, float]
    details: Mapping
    scope: str = "core_attention"
    provisional: bool = True


@dataclass(frozen=True)
class AttentionPredictor:
    """Predict one pack's group wall time in ms, not a model/layer time.

    The default profile is the unapproved native CP1 diagnostic snapshot.
    Multi-CP is only available by explicitly selecting the older legacy stack.
    No automatic fallback or backward/forward multiplier is used.
    """

    hardware: str
    query_heads: int
    kv_heads: int
    head_dim: int
    sequence_alignment: int
    pack_alignment: int
    profile: str = "experimental_cp1_bounded"
    allow_provisional: bool = False
    legacy_model_family: str = "registry"
    legacy_transport_backend: str = "nccl_p2p"
    legacy_domain_size: int | None = None

    @property
    def attention_kind(self):
        return "gqa"

    def __post_init__(self):
        if self.hardware not in ("h100", "gb200"):
            raise ValueError("hardware must be h100 or gb200; GB300 is not this calibration")
        for name in ("query_heads", "kv_heads", "head_dim", "sequence_alignment", "pack_alignment"):
            positive_int(getattr(self, name), name)
        if (self.query_heads, self.kv_heads, self.head_dim) not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}")
        if self.profile not in ("experimental_cp1_bounded", "legacy_frozen_20260922"):
            raise ValueError("unknown predictor profile")
        if type(self.allow_provisional) is not bool:
            raise ValueError("allow_provisional must be a bool")

    def predict(self, lengths: Sequence[int], cp_size: int = 1, *,
                phase: str = "forward", ranks: Sequence[int] | None = None) -> AttentionEstimate:
        if not self.allow_provisional:
            raise ValueError("these snapshots require explicit allow_provisional=True")
        positive_int(cp_size, "cp_size")
        if phase not in ("forward", "backward", "training"):
            raise ValueError("phase must be forward, backward, or training")
        physical_ranks = None if ranks is None else tuple(ranks)
        if physical_ranks is not None and (
            len(physical_ranks) != cp_size or len(set(physical_ranks)) != cp_size
            or any(type(r) is not int or r < 0 for r in physical_ranks)
        ):
            raise ValueError("ranks must contain cp_size distinct nonnegative physical rank IDs")
        geometry = self.query_heads, self.kv_heads, self.head_dim
        phases = ("forward", "backward") if phase == "training" else (phase,)
        if self.profile == "experimental_cp1_bounded":
            if cp_size != 1:
                raise ValueError("native multi-CP model is unfinished; no implicit legacy fallback")
            pack = Pack.build(lengths, self.sequence_alignment, self.pack_alignment, static_slots=64)
            model, sha = _model(self.hardware)
            parts = {p: predict_ms(model, pack.lengths, sequence_alignment=self.sequence_alignment,
                pack_alignment=self.pack_alignment, hardware=self.hardware, query_heads=self.query_heads,
                kv_heads=self.kv_heads, head_dim=self.head_dim, phase=p, allow_provisional=True) for p in phases}
            heads = [h for h in model["heads"] if tuple(h["geometry"]) == geometry and h["phase"] in phases]
            details = {"parameter_sha256": sha, "source_report_sha256": model["source_report_sha256"],
                "development_errors_only": {h["phase"]: h["development_error"] for h in heads},
                "independent_accuracy_passed": False, "software_path": "native, BF16, eager, static65",
                "warning": "diagnostic coefficients, not adopted runtime coefficients or a 5% guarantee"}
        else:
            if cp_size > 64:
                raise ValueError("legacy CP coverage stops at 64; native H100 CP128 remains unfinished")
            if cp_size > 1 and physical_ranks is None:
                raise ValueError("multi-CP requires actual physical ranks; topology must not be guessed")
            if self.sequence_alignment % (2 * cp_size):
                raise ValueError("legacy sequence alignment must be divisible by 2*CP")
            pack = Pack.build(lengths, self.sequence_alignment, self.pack_alignment, cp_size=cp_size)
            from ._legacy.attention.estimator import AttentionEstimator
            estimator = AttentionEstimator(hardware=self.hardware, query_heads=self.query_heads,
                kv_heads=self.kv_heads, head_dim=self.head_dim,
                physical_tokens_per_rank=pack.storage_tokens // cp_size, max_cp_size=cp_size,
                alignments=((cp_size, self.sequence_alignment),), allow_provisional=True, require_ranks=True,
                model_family=self.legacy_model_family, transport_backend=self.legacy_transport_backend,
                domain_size=self.legacy_domain_size)
            result = estimator.predict(pack.lengths, cp_size, phase=phase, ranks=physical_ranks)
            parts = {p: float(result["components"][p]["time_ms"]) for p in phases}
            details = {"legacy_result": result, "warning": "explicit legacy software/calibration; not current native validation"}
        return AttentionEstimate(sum(parts.values()), phase, cp_size, self.hardware, geometry,
                                 self.profile, pack, physical_ranks, parts, details)
