# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Dependency-aware model timing with fixed, explicitly supplied measurements."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Protocol, Sequence

from ._timeline.training_timeline import evaluate_training_schedule, resolve_pipeline_layers


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _duration(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite nonnegative number of milliseconds")
    return float(value)


class CorePredictor(Protocol):
    def predict(self, lengths, cp_size=1, *, phase, ranks): ...


@dataclass(frozen=True)
class PackGroup:
    lengths: tuple[int, ...]
    ranks: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "lengths", tuple(self.lengths))
        object.__setattr__(self, "ranks", tuple(self.ranks))
        if not self.lengths or not self.ranks:
            raise ValueError("a group needs real sequences and ranks")
        for length in self.lengths:
            _positive(length, "sequence length")
        if any(type(r) is not int or r < 0 for r in self.ranks):
            raise ValueError("ranks must be nonnegative integers")
        if self.ranks != tuple(range(self.ranks[0], self.ranks[0] + len(self.ranks))):
            raise ValueError("this timeline supports contiguous ascending physical rank groups only")


@dataclass(frozen=True)
class NonAttentionTimes:
    """Required fixed measurements: per layer, per microbatch, per rank, ms.

    A value can be a scalar or a mapping from layer type to scalar; either can
    contain one fixed value per physical PP stage. No token-dependent fit is
    performed. Include projections, EP communication and other excluded work
    in these measurements; exclude the core attention measured by predictors.
    """

    forward_ms_per_layer: float | Sequence[float] | Mapping
    backward_ms_per_layer: float | Sequence[float] | Mapping
    source: str
    iteration_tail_ms: float = 0.0

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must identify the user-supplied measurements")
        _duration(self.iteration_tail_ms, "iteration_tail_ms")

    def at(self, phase, kind, stage, pp_size):
        value = self.forward_ms_per_layer if phase == "forward" else self.backward_ms_per_layer
        if isinstance(value, Mapping):
            if kind not in value:
                raise ValueError("missing non-attention measurement for " + kind)
            value = value[kind]
        if isinstance(value, (tuple, list)):
            if len(value) != pp_size:
                raise ValueError("stage timings need exactly pp_size entries")
            value = value[stage]
        return _duration(value, phase + " non-attention")


@dataclass(frozen=True)
class ModelConfig:
    layer_types: tuple[str, ...]
    world_size: int
    ep_size: int = 1
    pp_size: int = 1
    vpp_size: int = 1
    layers_per_stage: tuple[int, ...] | None = None
    layers_per_virtual_stage: tuple[int, ...] | None = None
    microbatch_group_size_per_vp_stage: int | None = None
    first_forward_dp_cp_sync: bool = True
    recompute_forward: bool = False
    forward_pipeline_latency_ms: float = 0.0
    backward_pipeline_latency_ms: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "layer_types", tuple(self.layer_types))
        if not self.layer_types or any(not isinstance(k, str) or not k for k in self.layer_types):
            raise ValueError("layer_types needs one nonempty kind per modeled layer")
        for key in ("world_size", "ep_size", "pp_size", "vpp_size"):
            _positive(getattr(self, key), key)
        if self.world_size % self.ep_size:
            raise ValueError("ep_size must divide world_size per PP stage")
        for key in ("first_forward_dp_cp_sync", "recompute_forward"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(key + " must be a bool")
        for key in ("layers_per_stage", "layers_per_virtual_stage"):
            values = getattr(self, key)
            if values is not None:
                for v in values:
                    _positive(v, key)
                object.__setattr__(self, key, tuple(values))
        if self.microbatch_group_size_per_vp_stage is not None:
            _positive(self.microbatch_group_size_per_vp_stage, "microbatch_group_size_per_vp_stage")
        _duration(self.forward_pipeline_latency_ms, "forward_pipeline_latency_ms")
        _duration(self.backward_pipeline_latency_ms, "backward_pipeline_latency_ms")


@dataclass(frozen=True)
class E2EEstimate:
    time_ms: float
    timeline: object
    non_attention_source: str
    attention_predictions: tuple
    execution_program: tuple
    provisional: bool = True
    assumptions: tuple[str, ...] = (
        "fixed measured non-attention inputs; no regression or E2E residual",
        "HybridEP valid-token-dependent work is not captured by fixed inputs",
        "CP group wall time applies to all member ranks; no internal rank model",
        "first-forward barrier is an optional parameter-gather approximation",
        "PP/VPP dependencies are modeled; pipeline contention/overlap is not calibrated",
        "modeled time excludes CPU planning unless explicitly supplied as iteration tail",
    )

    def as_dict(self):
        return asdict(self)


class ModelEvaluator:
    """Score an existing schedule; never creates or optimizes one.

    Predictors are independent duck-typed objects keyed by layer kind. Each
    estimate must carry scope='core_attention' and time_ms. GDN therefore
    needs its own core predictor, not the ordinary GQA module by another name.
    """

    def __init__(self, config: ModelConfig, non_attention: NonAttentionTimes,
                 predictors: Mapping[str, CorePredictor]):
        if not isinstance(config, ModelConfig) or not isinstance(non_attention, NonAttentionTimes):
            raise TypeError("ModelConfig and explicit NonAttentionTimes are required")
        self.config, self.non_attention, self.predictors = config, non_attention, dict(predictors)
        missing = set(config.layer_types) - set(predictors)
        if missing:
            raise ValueError("missing core attention predictor(s): " + ", ".join(sorted(missing)))
        for kind in set(config.layer_types):
            declared_kind = getattr(self.predictors[kind], "attention_kind", None)
            if declared_kind is not None and declared_kind != kind:
                raise ValueError("attention kind mismatch: " + kind + " vs " + declared_kind)

    def _program(self):
        c = self.config
        physical, virtual = resolve_pipeline_layers(layer_count=len(c.layer_types), pp_size=c.pp_size,
            vpp_size=c.vpp_size, layers_per_stage=c.layers_per_stage,
            layers_per_virtual_stage=c.layers_per_virtual_stage)
        program, start = [], 0
        for slot, count in enumerate(virtual):
            kinds = c.layer_types[start:start + count]
            start += count
            stage = slot % c.pp_size
            forward, backward = [], []
            for kind in kinds:
                forward += [{"sync": "cp", "cost": kind + ":forward"}, {"sync": "ep", "duration_ms":
                    self.non_attention.at("forward", kind, stage, c.pp_size)}]
            for kind in reversed(kinds):
                if c.recompute_forward:
                    backward += [{"sync": "cp", "cost": kind + ":forward"}, {"sync": "ep", "duration_ms":
                        self.non_attention.at("forward", kind, stage, c.pp_size)}]
                backward += [{"sync": "ep", "duration_ms": self.non_attention.at("backward", kind, stage, c.pp_size)},
                             {"sync": "cp", "cost": kind + ":backward"}]
            program.append({"layer_count": count, "forward": forward, "backward": backward})
        return physical, virtual, program

    def predict(self, rounds: Sequence[Sequence[PackGroup]], *, microbatch_order=None) -> E2EEstimate:
        c = self.config
        rounds = tuple(tuple(row) for row in rounds)
        if not rounds:
            raise ValueError("at least one microbatch is required")
        physical, virtual, program = self._program()
        costed, details, cache = [], [], {}
        for row in rounds:
            if not row or any(not isinstance(g, PackGroup) for g in row):
                raise ValueError("every microbatch needs PackGroup entries")
            groups = sorted(row, key=lambda g: g.ranks[0])
            if tuple(r for g in groups for r in g.ranks) != tuple(range(c.world_size)):
                raise ValueError("each microbatch must occupy exactly world_size ranks, once each")
            result_row = []
            for group in groups:
                stages = []
                for stage in range(c.pp_size):
                    ranks = tuple(stage * c.world_size + r for r in group.ranks)
                    costs = {}
                    kinds = set(k for slot in range(stage, len(virtual), c.pp_size)
                                for k in c.layer_types[sum(virtual[:slot]):sum(virtual[:slot + 1])])
                    for kind in sorted(kinds):
                        for phase in ("forward", "backward"):
                            key = kind, group.lengths, ranks, phase
                            if key not in cache:
                                estimate = self.predictors[kind].predict(group.lengths, len(ranks), phase=phase, ranks=ranks)
                                scope = estimate.get("scope") if isinstance(estimate, Mapping) else getattr(estimate, "scope", None)
                                value = estimate.get("time_ms") if isinstance(estimate, Mapping) else getattr(estimate, "time_ms", None)
                                if scope != "core_attention":
                                    raise ValueError("predictor scope must be core_attention (no projection double count)")
                                cache[key] = _duration(value, "attention time")
                                details.append({"kind": kind, "lengths": group.lengths, "ranks": ranks,
                                                "phase": phase, "estimate": estimate})
                            costs[kind + ":" + phase] = cache[key]
                    stages.append(costs)
                result_row.append({"cp": len(group.ranks), "rank_layout": "arbitrary_contiguous",
                                   "program_costs_ms_by_stage": stages})
            costed.append(result_row)
        timeline = evaluate_training_schedule(costed, world_size=c.world_size, ep_size=c.ep_size,
            pp_size=c.pp_size, vpp_size=c.vpp_size, layer_count=len(c.layer_types),
            layers_per_stage=physical, layers_per_virtual_stage=virtual,
            microbatch_group_size_per_vp_stage=c.microbatch_group_size_per_vp_stage,
            first_forward_dp_cp_sync=c.first_forward_dp_cp_sync,
            forward_pipeline_latency_ms=c.forward_pipeline_latency_ms,
            backward_pipeline_latency_ms=c.backward_pipeline_latency_ms,
            iteration_tail_ms=self.non_attention.iteration_tail_ms, microbatch_order=microbatch_order,
            execution_program=program)
        return E2EEstimate(timeline.makespan_ms, timeline, self.non_attention.source, tuple(details), tuple(program))
