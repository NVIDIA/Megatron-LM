# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Portable adapters for the original frozen forward/backward timing models."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from ..validation import positive_int
from .topology import CALIBRATION_DOMAIN_SIZES, RingPlacement, predict_forward_at_placement

ROOT = Path(__file__).resolve().parent
INTEGER_FORWARD = "non_power_cp_20260829/models/h100_qwen3_30b_cp2_32_integer_candidate.json"
INTEGER_BACKWARD = "non_power_cp_20260829/models/h100_qwen3_30b_backward_cp1_32_integer_holdout_backed.json"


@lru_cache(maxsize=1)
def verify_calibration():
    manifest = json.loads((ROOT / "calibration_manifest.json").read_text())
    for name, entry in manifest["artifacts"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != entry["export_sha256"]:
            raise ValueError(f"calibration hash mismatch: {name}")
    return hashlib.sha256((ROOT / "calibration_manifest.json").read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _backward():
    from .backward_model.backward_attention_predictor import FrozenBackwardAttentionPredictor
    verify_calibration()
    return FrozenBackwardAttentionPredictor.from_path()


@lru_cache(maxsize=1)
def _integer_models():
    from .cp_gqa_attention_predictor import CPGQAAttentionPredictor
    from .non_power_cp_20260829.arbitrary_cp_backward_predictor import ArbitraryCPBackwardPredictor
    verify_calibration()
    return (CPGQAAttentionPredictor.from_path(ROOT / INTEGER_FORWARD),
            ArbitraryCPBackwardPredictor.from_path(ROOT / INTEGER_BACKWARD))


@dataclass(frozen=True)
class AttentionEstimator:
    hardware: str = "h100"
    query_heads: int = 32
    kv_heads: int = 4
    head_dim: int = 128
    physical_tokens_per_rank: int = 4096
    max_cp_size: int = 64
    model_family: str = "registry"
    cp1_branch: str = "current"
    cp_implementation: str = "te_p2p"
    ulysses_kv_strategy: str = "replicate"
    allow_provisional: bool = False
    domain_size: int | None = None
    transport_backend: str = "nccl_p2p"
    require_ranks: bool = False
    placement_calibration: str | None = None
    # Scheduler bindings override max_cp_size with the exact storage alignment.
    alignments: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if self.hardware not in ("h100", "gb200"):
            raise ValueError("hardware must be h100 or gb200")
        for key in ("query_heads", "kv_heads", "head_dim", "physical_tokens_per_rank", "max_cp_size"):
            positive_int(getattr(self, key), key)
        if self.query_heads % self.kv_heads or self.max_cp_size > 64:
            raise ValueError("query_heads must divide into KV groups; max_cp_size must be <=64")
        if self.model_family not in ("registry", "h100_qwen3_integer"):
            raise ValueError("unknown model_family")
        if self.cp1_branch not in ("current", "embedded_legacy"):
            raise ValueError("cp1_branch must be current or embedded_legacy")
        if self.model_family == "h100_qwen3_integer" and (
            self.hardware, self.query_heads, self.kv_heads, self.head_dim,
            self.physical_tokens_per_rank, self.cp_implementation
        ) != ("h100", 32, 4, 128, 4096, "te_p2p"):
            raise ValueError("integer model requires H100, 32/4/128, P4096, TE P2P")
        if type(self.allow_provisional) is not bool:
            raise ValueError("allow_provisional must be boolean")
        if type(self.require_ranks) is not bool:
            raise ValueError("require_ranks must be boolean")
        if self.domain_size is not None:
            positive_int(self.domain_size, "domain_size")
        if self.transport_backend not in ("nccl_p2p", "native"):
            raise ValueError("transport_backend must be nccl_p2p or native")
        if self.transport_backend == "native" and self.cp_implementation != "te_p2p":
            raise ValueError("native transport proxy requires the TE P2P compute path")
        object.__setattr__(self, "alignments", tuple(tuple(pair) for pair in self.alignments))
        if self.placement_calibration is not None:
            from .placement_calibration import validate_scope
            object.__setattr__(self, "placement_calibration", str(Path(self.placement_calibration).resolve()))
            validate_scope(self)

    def for_scheduler(self, config):
        if config.tokens_per_rank != self.physical_tokens_per_rank:
            raise ValueError("estimator and scheduler storage capacities disagree")
        if config.dataset_alignment != 1 or config.tail_padding != "extend_last":
            raise ValueError("estimator padding requires dataset_alignment=1 and tail_padding='extend_last'; "
                             "other conventions need a matching predictor")
        return replace(self, alignments=tuple(config.alignments().items()),
                       max_cp_size=max(config.allowed_cp_sizes))

    def scheduler_metadata(self, lengths, rounds, config, phase):
        from ..rank_timeline import rank_intervals
        seen, reasons, placements, backends = set(), set(), set(), set()
        provisional = False
        for groups in rounds:
            for group, local in zip(groups, rank_intervals(groups, config.world_size)):
                values = tuple(lengths[i] for i in group["sample_ids"])
                for rank_map in config.rank_maps():
                    ranks = tuple(rank_map[r] for r in local)
                    key = (values, group["cp"], ranks)
                    if key in seen:
                        continue
                    seen.add(key)
                    detail = self.predict(values, group["cp"], phase=phase, ranks=ranks)
                    provisional |= detail["provisional"]
                    reasons.update(detail["provisional_reasons"])
                    backends.add(detail["estimated_transport_backend"] or "none")
                    placements.add(ranks)
        return dict(hardware=self.hardware, domain_size=self.domain_size or CALIBRATION_DOMAIN_SIZES[self.hardware],
                    transport_backend=self.transport_backend, cp_implementation=self.cp_implementation,
                    estimated_transport_backends=sorted(backends),
                    provisional=provisional, provisional_reasons=sorted(reasons),
                    unique_pack_placements=len(seen), unique_rank_groups=len(placements))

    def _placement(self, ranks, cp_size):
        if ranks is None and cp_size > 1 and self.require_ranks:
            raise ValueError("actual ranks in ring order are required; the legacy two-argument "
                             "scheduler callback does not supply placement")
        return RingPlacement.resolve(ranks, cp_size, self.hardware, self.domain_size)

    def predict(self, lengths, cp_size=1, *, phase="training", ranks=None):
        """Predict one CP group; ranks are physical global IDs in logical ring order."""
        lengths = tuple(positive_int(n, "sequence length") for n in lengths)
        if not lengths:
            raise ValueError("lengths must not be empty")
        positive_int(cp_size, "cp_size")
        if cp_size > self.max_cp_size:
            raise ValueError("cp_size exceeds max_cp_size")
        if phase not in ("forward", "backward", "training"):
            raise ValueError("phase must be forward, backward or training")
        placement = self._placement(ranks, cp_size)
        alignment = dict(self.alignments).get(cp_size, 2 * self.max_cp_size)
        if self.alignments and cp_size not in dict(self.alignments):
            raise ValueError("CP not allowed by this scheduler binding")
        if alignment <= 0 or alignment % (2 * cp_size):
            raise ValueError("sequence alignment must be divisible by 2*CP")
        if sum((n + alignment - 1) // alignment * alignment for n in lengths) > cp_size * self.physical_tokens_per_rank:
            raise ValueError("aligned pack exceeds physical capacity")
        if placement.changed and cp_size > 1 and phase != "forward" and self.placement_calibration is None:
            raise ValueError("backward has no placement-specific communication decomposition; "
                             "cannot reuse the old whole-call fit for a changed domain partition. "
                             "Use phase='forward' for the available communication transfer")
        manifest_hash = verify_calibration()
        parts = {}
        if self.model_family == "h100_qwen3_integer":
            if cp_size > 32:
                raise ValueError("integer extension is calibrated/scoped only through CP32")
            fwd, bwd = _integer_models()
            if phase != "backward":
                if cp_size == 1 and self.cp1_branch == "current":
                    from .hardware_gqa_attention_predictor import predict_attention_with_details
                    parts["forward"] = predict_attention_with_details(lengths, 1,
                        self.query_heads, self.kv_heads, self.head_dim, hardware=self.hardware,
                        max_cp_size=alignment//2, physical_tokens_per_rank=self.physical_tokens_per_rank)
                else:
                    parts["forward"] = fwd.predict_with_details(lengths, cp_size,
                        max_cp_size=cp_size if cp_size > 1 else alignment//2,
                        sequence_alignment=alignment)
            if phase != "forward":
                parts["backward"] = bwd.predict_with_details(lengths, cp_size,
                                                             sequence_alignment=alignment)
        else:
            if phase != "backward":
                from . import hardware_gqa_attention_predictor as registry
                if self.hardware == "gb200" and cp_size == 1 and self.cp_implementation == "te_p2p":
                    exact = registry._singleton_sol_details(lengths, self.hardware, cp_size,
                        self.query_heads, self.kv_heads, self.head_dim, self.physical_tokens_per_rank)
                    if exact is not None:
                        parts["forward"] = exact
                    else:
                        model = registry.load_gb200_cp1_predictor()
                        # The legacy registry wrapper omitted both arguments. Keep
                        # physical boundaries identical to the scheduler binding.
                        time_ms = model.predict_time_ms(lengths, self.query_heads, self.kv_heads,
                            self.head_dim, max_cp_size=alignment//2,
                            storage_tokens=self.physical_tokens_per_rank,
                            allow_extrapolation=self.allow_provisional)
                        inside = (model.standard.minimum_total <= sum(lengths) <= model.standard.maximum_total
                                  and model.standard.minimum_sequence_count <= len(lengths) <= model.standard.maximum_sequence_count
                                  and min(lengths) >= model.standard.minimum_sequence_tokens)
                        parts["forward"] = dict(time_ms=time_ms, cp_size=1,
                            sequence_alignment=alignment, storage_tokens=self.physical_tokens_per_rank,
                            provisional=(not inside or alignment != model.standard.sequence_alignment),
                            model_kind="frozen_gb200_cp1_with_explicit_layout")
                else:
                    parts["forward"] = registry.predict_attention_with_details(lengths, cp_size,
                        self.query_heads, self.kv_heads, self.head_dim, hardware=self.hardware,
                        max_cp_size=alignment//2 if cp_size == 1 or self.cp_implementation != "te_p2p" else cp_size,
                        sequence_alignment=alignment if cp_size > 1 and self.cp_implementation == "te_p2p" else None,
                        physical_tokens_per_rank=self.physical_tokens_per_rank,
                        cp_implementation=self.cp_implementation, ulysses_kv_strategy=self.ulysses_kv_strategy)
            if phase != "forward":
                if self.cp_implementation != "te_p2p":
                    raise ValueError("communication-free and Ulysses fallbacks are forward-only")
                parts["backward"] = _backward().predict_with_details(lengths,
                    hardware=self.hardware, cp_size=cp_size, query_heads=self.query_heads,
                    kv_heads=self.kv_heads, head_dim=self.head_dim,
                    physical_tokens_per_rank=self.physical_tokens_per_rank,
                    sequence_alignment=alignment)
        if self.placement_calibration is not None and cp_size > 1:
            from .placement_calibration import apply_calibration
            parts = {p: apply_calibration(self, lengths, cp_size, placement, alignment, p, value)
                     for p, value in parts.items()}
        elif placement.changed and cp_size > 1 and self.cp_implementation == "te_p2p":
            if not self.allow_provisional:
                raise ValueError("new domain placement requires allow_provisional=True")
            if self.model_family == "h100_qwen3_integer":
                model, _ = _integer_models()
            else:
                from .hardware_gqa_attention_predictor import load_te_p2p_predictor
                _, _, model = load_te_p2p_predictor(self.hardware, self.query_heads,
                    self.kv_heads, self.head_dim, self.physical_tokens_per_rank)
            parts["forward"] = predict_forward_at_placement(model, lengths, cp_size,
                hardware=self.hardware, placement=placement, sequence_alignment=alignment)
        reasons = []
        for condition, reason in (
            (bool(cp_size & (cp_size-1)), "non_power_cp"),
            (self.cp_implementation != "te_p2p", "communication_free_fallback"),
            (alignment != 128, "sequence_alignment"),
            (cp_size > 1 and placement.assumed, "actual_ranks_not_supplied"),
            (cp_size > 1 and placement.changed and self.cp_implementation == "te_p2p"
             and self.placement_calibration is None, "domain_placement_transfer"),
            (cp_size > 1 and self.placement_calibration is not None, "explicit_placement_calibration"),
            (placement.domain_size != CALIBRATION_DOMAIN_SIZES[self.hardware], "non_calibration_domain_size"),
            (cp_size > 1 and self.transport_backend == "native", "native_using_nccl_calibration_proxy"),
        ):
            if condition:
                reasons.append(reason)
        for value in parts.values():
            measured = value.get("placement_calibration")
            if measured and not measured["validation_passed"]:
                reasons.append("placement_holdout_failed")
            if measured and measured["outside_features"]:
                reasons.append("placement_features_extrapolated")
        for value in parts.values():
            if value.get("provisional") or value.get("outside_supported_domain_features") or value.get("outside_calibration_features"):
                reasons.append("component_model_extrapolation")
                break
        provisional = bool(reasons)
        if provisional and not self.allow_provisional:
            raise ValueError("prediction is outside the frozen measured scope (" + ", ".join(reasons)
                             + "); set allow_provisional=True explicitly")
        result = {f"{p}_ms": float(parts[p]["time_ms"]) if p in parts else 0.0
                  for p in ("forward", "backward")}
        total = result["forward_ms"] + result["backward_ms"]
        if not math.isfinite(total) or total <= 0:
            raise ValueError("invalid frozen-model prediction")
        return dict(time_ms=total, **result, phase=phase, hardware=self.hardware,
                    geometry=[self.query_heads, self.kv_heads, self.head_dim], cp_size=cp_size,
                    sequence_alignment=alignment, physical_tokens_per_rank=self.physical_tokens_per_rank,
                    model_family=self.model_family, cp1_branch=self.cp1_branch,
                    calibration_manifest_sha256=manifest_hash,
                    topology=placement.to_dict(), transport_backend=self.transport_backend,
                    estimated_transport_backend="nccl_p2p" if cp_size > 1 and self.cp_implementation == "te_p2p" else "none",
                    provisional=provisional, provisional_reasons=reasons, components=parts,
                    scope="core_attention_only_no_projections; frozen_model_not_hardware_measurement")

    @lru_cache(maxsize=32768)
    def _time(self, lengths, cp, phase, ranks):
        return self.predict(lengths, cp, phase=phase, ranks=ranks)["time_ms"]

    def forward(self, lengths, cp, *, ranks=None):
        placement = self._placement(ranks, cp)
        values = tuple(positive_int(n, "sequence length") for n in lengths)
        return self._time(values, cp, "forward", None if ranks is None else placement.ranks)

    def backward(self, lengths, cp, *, ranks=None):
        placement = self._placement(ranks, cp)
        values = tuple(positive_int(n, "sequence length") for n in lengths)
        return self._time(values, cp, "backward", None if ranks is None else placement.ranks)


def predict_attention(lengths, *, cp_size=1, phase="training", ranks=None, **configuration):
    return AttentionEstimator(**configuration).predict(lengths, cp_size, phase=phase, ranks=ranks)
