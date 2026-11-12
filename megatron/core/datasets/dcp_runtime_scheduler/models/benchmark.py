# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adapt the benchmark attention and model-E2E predictors to the CPU scheduler."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from ..runtime.planner import MAX_PRICED_GROUP_SEQUENCES
from ..runtime.timeline import TimelineModel, _freeze_op
from ._benchmark.attention_predictor import GEOMETRIES, AttentionPredictor
from ._benchmark.model_e2e import E2EEstimate, ModelConfig, ModelEvaluator, NonAttentionTimes
from ._benchmark.model_e2e._timeline.training_timeline import evaluate_training_schedule
from .base import NVLINK_DOMAIN_SIZES, PackLimits, largest_priced_sequences


@dataclass(frozen=True)
class BenchmarkTimeline(TimelineModel):
    """Use the exported E2E engine for final scoring and its PP1 program for search."""

    execution_program: tuple = ()

    def evaluate(self, rounds):
        """Score cached core costs with the original dependency-aware E2E engine."""
        costed = []
        for row in rounds:
            groups, start = [], 0
            for group in row:
                if group[0] != start:
                    raise ValueError("E2E scoring requires contiguous physical rank intervals")
                start += group[1]
                groups.append(
                    dict(
                        cp=group[1],
                        rank_layout="arbitrary_contiguous",
                        program_costs_ms_by_stage=[group[4]],
                    )
                )
            if start != self.world_size:
                raise ValueError("Every microbatch must occupy exactly world_size ranks")
            costed.append(groups)
        result = evaluate_training_schedule(
            costed,
            world_size=self.world_size,
            ep_size=self.ep_size,
            pp_size=1,
            vpp_size=1,
            layer_count=self.layer_count,
            first_forward_dp_cp_sync=self.first_forward_dp_cp_sync,
            iteration_tail_ms=self.iteration_tail_ms,
            execution_program=self.execution_program,
        )
        return result.makespan_ms


class _BoundAttention:
    """Keep physical per-rank capacity fixed while changing a pack's CP width."""

    attention_kind = "gqa"

    def __init__(self, geometry, options):
        self.geometry, self.options = geometry, dict(options)

    @lru_cache(maxsize=64)
    def _predictor(self, cp):
        return AttentionPredictor(
            sequence_alignment=self.geometry.dataset_alignment,
            pack_alignment=cp * self.geometry.tokens_per_rank,
            **self.options,
        )

    def predict(self, lengths, cp_size=1, *, phase, ranks):
        """Predict core attention with the selected CP group and physical ranks."""
        return self._predictor(cp_size).predict(lengths, cp_size, phase=phase, ranks=ranks)


class BenchmarkCostModel:
    """Core-attention predictions plus explicit fixed non-attention measurements.

    Attention parameters are selected internally. No coefficients are refitted,
    no residual is added, and no missing non-attention time is inferred. Custom
    layer families require their own core predictors, supplied by the caller.
    The current training adapter supports PP1/VPP1; the exported E2E package
    retains its independent PP/VPP API.

    NVLink topology follows from ``hardware``: the predictor reads rank ``r`` as a GPU of
    domain ``r // nvlink_domain_size`` and has parameters for one way of splitting a CP group
    over domains, the split of a group that starts on a domain boundary. Such a group has one
    price wherever it starts; any other split is rejected. No argument sets the domain size.
    """

    name = "dcp_benchmark"
    placement_invariant = False
    # Layer types a per-type ``non_attention`` mapping must name for this model.
    non_attention_layer_types = ("gqa",)
    # (num_attention_heads, num_query_groups, kv_channels) with packaged attention parameters.
    attention_geometries = GEOMETRIES
    # Largest DPxCP the attention parameters cover: the predictor takes the dataset's
    # sequence alignment (2 * DPxCP) as its own largest CP size, and that stops at 64.
    max_dp_cp_size = 64
    # Largest CP group this model prices; the scheduler forms no larger one.
    largest_cp_size = 64

    def __init__(
        self,
        geometry,
        *,
        hardware,
        profile="legacy_frozen_20260922",
        query_heads,
        kv_heads,
        head_dim,
        non_attention,
        layer_count=1,
        layer_types=None,
        allow_provisional=True,
        legacy_model_family="registry",
        legacy_transport_backend="nccl_p2p",
        first_forward_dp_cp_sync=True,
        recompute_forward=False,
        predictors=None,
    ):
        if allow_provisional is not True:
            raise ValueError("Benchmark snapshots require explicit allow_provisional=True")
        if type(layer_count) is not int or layer_count < 1:
            raise ValueError("layer_count must be a positive integer")
        if not isinstance(non_attention, dict):
            raise TypeError("Explicit non_attention measurements are required")
        non_attention = dict(non_attention)
        non_attention.setdefault("source", "user-provided non-attention times")
        self.geometry = geometry
        self.hardware, self.profile = hardware, profile
        self.attention_geometry = (query_heads, kv_heads, head_dim)
        self.layer_types = tuple(layer_types) if layer_types is not None else ("gqa",) * layer_count
        self.layer_count = len(self.layer_types)
        self.non_attention = NonAttentionTimes(**non_attention)
        self.config = ModelConfig(
            layer_types=self.layer_types,
            world_size=geometry.world_size,
            ep_size=geometry.ep_size,
            first_forward_dp_cp_sync=first_forward_dp_cp_sync,
            recompute_forward=recompute_forward,
        )
        self.predictor_options = dict(
            hardware=hardware,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            profile=profile,
            allow_provisional=allow_provisional,
            legacy_model_family=legacy_model_family,
            legacy_transport_backend=legacy_transport_backend,
        )
        # Validate configuration at construction even before the first prediction.
        AttentionPredictor(
            sequence_alignment=geometry.dataset_alignment,
            pack_alignment=geometry.tokens_per_rank,
            **self.predictor_options,
        )
        # The predictor accepted the hardware; it keeps the same domain size for it.
        self.nvlink_domain_size = NVLINK_DOMAIN_SIZES[hardware]
        self.predictors = (
            {"gqa": _BoundAttention(geometry, self.predictor_options)}
            if predictors is None
            else dict(predictors)
        )
        self.evaluator = ModelEvaluator(self.config, self.non_attention, self.predictors)
        _, _, program = self.evaluator._program()
        self.execution_program = tuple(program)
        self._timing = dict(
            execution_program=program,
            iteration_tail_ms=self.non_attention.iteration_tail_ms,
            first_forward_dp_cp_sync=first_forward_dp_cp_sync,
        )
        root = Path(__file__).with_name("_benchmark")
        manifest_sha256 = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        origin = json.loads((Path(__file__).parents[1] / "ORIGIN.json").read_text())
        if (
            manifest_sha256
            != origin["runtime_parameters_sha256"]["models/_benchmark/manifest.json"]
        ):
            raise ValueError("Benchmark predictor snapshot hash mismatch")
        self.source_hashes = {
            "benchmark_manifest": manifest_sha256,
            "non_attention": hashlib.sha256(
                json.dumps(non_attention, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        self._pack_limits = {}
        self.check_domain(geometry, None)

    @lru_cache(maxsize=65536)
    def _costs(self, lengths, cp, ranks):
        if cp not in self.geometry.allowed_cp_sizes:
            raise ValueError("CP size is outside the configured prediction domain")
        if not lengths or len(lengths) > min(
            self.geometry.max_group_seqs or MAX_PRICED_GROUP_SEQUENCES, MAX_PRICED_GROUP_SEQUENCES
        ):
            raise ValueError("A pack needs one to the configured maximum real sequences")
        if any(type(n) is not int or n < 1 for n in lengths):
            raise ValueError("Logical sequence lengths must be positive integers")
        if sum(self.geometry.stored_length(n, cp) for n in lengths) > self.geometry.capacity(cp):
            raise ValueError("Pack exceeds physical per-rank capacity")
        if not ranks or ranks != tuple(range(ranks[0], ranks[0] + cp)):
            raise ValueError("Actual contiguous physical rank IDs are required")
        if len(ranks) != cp or ranks[0] < 0 or ranks[-1] >= self.geometry.world_size:
            raise ValueError("Physical rank IDs must be inside the parent domain")
        costs = {}
        for kind in sorted(set(self.layer_types)):
            for phase in ("forward", "backward"):
                estimate = self.predictors[kind].predict(lengths, cp, phase=phase, ranks=ranks)
                scope = estimate.get("scope") if isinstance(estimate, Mapping) else estimate.scope
                value = (
                    estimate.get("time_ms") if isinstance(estimate, Mapping) else estimate.time_ms
                )
                if scope != "core_attention":
                    raise ValueError(
                        "Predictor scope must be core_attention, excluding projections"
                    )
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Core attention prediction must be finite nonnegative ms")
                costs[kind + ":" + phase] = float(value)
        return costs

    def pack_limits(self, cp):
        """Declare how many sequences this predictor prices in one pack of ``cp`` ranks.

        The attention predictor has no table of it: with a large sequence alignment it
        stops pricing packs of many sequences (33 or more at CP 64 with DPxCP=64 and 4096
        tokens per rank). The count is found once per CP size by pricing full packs.
        """
        limits = self._pack_limits.get(cp)
        if limits is None:
            cap = min(
                self.geometry.max_group_seqs or MAX_PRICED_GROUP_SEQUENCES,
                MAX_PRICED_GROUP_SEQUENCES,
            )
            limits = self._pack_limits[cp] = PackLimits(
                max_sequences=largest_priced_sequences(self, self.geometry, cp, cap)
            )
        return limits

    def program_costs(self, lengths, cp, *, ranks=None):
        """Return one core-attention F/B prediction for each distinct layer family."""
        return dict(self._costs(tuple(lengths), cp, None if ranks is None else tuple(ranks)))

    def forward(self, lengths, cp, *, ranks=None):
        """Return mean per-layer core forward time for the search heuristic."""
        costs = self.program_costs(lengths, cp, ranks=ranks)
        return sum(costs[kind + ":forward"] for kind in self.layer_types) / self.layer_count

    def backward(self, lengths, cp, *, ranks=None):
        """Return mean per-layer core backward time for the search heuristic."""
        costs = self.program_costs(lengths, cp, ranks=ranks)
        return sum(costs[kind + ":backward"] for kind in self.layer_types) / self.layer_count

    def timing(self):
        """Expose the benchmark E2E layer program, including measured remainders."""
        return self._timing

    def build_timeline(self, geometry):
        """Retain the source E2E scorer and the compatible incremental PP1 timeline."""
        self.check_domain(geometry, None)
        program = self.execution_program[0]
        return BenchmarkTimeline(
            world_size=geometry.world_size,
            ep_size=geometry.ep_size,
            layer_count=self.layer_count,
            iteration_tail_ms=self.non_attention.iteration_tail_ms,
            first_forward_dp_cp_sync=self.config.first_forward_dp_cp_sync,
            program=tuple(
                tuple(_freeze_op(op) for op in program[phase]) for phase in ("forward", "backward")
            ),
            execution_program=self.execution_program,
        )

    def clear(self):
        """Reset the per-global-batch core prediction cache."""
        self._costs.cache_clear()

    def check_domain(self, geometry, config=None):
        """Reject incompatible storage, model structure or execution configuration."""
        if geometry != self.geometry:
            raise ValueError("Cost model and planner geometry differ")
        if any(geometry.dataset_alignment % (2 * cp) for cp in geometry.allowed_cp_sizes):
            raise ValueError("Sequence alignment must be divisible by 2*CP for every allowed width")
        if self.profile == "experimental_cp1_bounded" and geometry.allowed_cp_sizes != (1,):
            raise ValueError("The experimental profile covers CP1 only; multi-CP is unfinished")
        if self.profile == "experimental_cp1_bounded" and geometry.tokens_per_rank not in (
            4096,
            8192,
            16384,
        ):
            raise ValueError("Experimental CP1 storage must be 4096, 8192 or 16384 tokens")
        if self.profile == "legacy_frozen_20260922" and max(geometry.allowed_cp_sizes) > 64:
            raise ValueError("Legacy multi-CP coverage stops at CP64")
        if self.predictor_options["legacy_model_family"] == "h100_qwen3_integer" and (
            self.hardware != "h100"
            or self.attention_geometry != (32, 4, 128)
            or geometry.tokens_per_rank != 4096
            or max(geometry.allowed_cp_sizes) > 32
        ):
            raise ValueError("Integer extension requires H100 32/4/128, P4096 and CP<=32")
        if config is None:
            return
        if getattr(config, "thd_tail_padding_policy", None) != "extend_last":
            raise ValueError("Benchmark predictors require extend_last tail padding")
        if getattr(config, "pad_packed_seq_alignment", None) not in (
            "max",
            geometry.tokens_per_rank,
        ):
            raise ValueError("Benchmark predictors require fixed per-rank capacity padding")
        if getattr(config, "pipeline_model_parallel_size", 1) != 1 or getattr(
            config, "virtual_pipeline_model_parallel_size", None
        ) not in (None, 1):
            raise ValueError("The current runtime scheduler adapter requires PP1/VPP1")
        if getattr(config, "tensor_model_parallel_size", 1) != 1:
            raise ValueError("The current benchmark rank/attention mapping requires TP1")
        actual = (
            getattr(config, "num_attention_heads", None),
            getattr(config, "num_query_groups", None),
            getattr(config, "kv_channels", None),
        )
        if (
            actual != self.attention_geometry
            or getattr(config, "num_layers", None) != self.layer_count
        ):
            raise ValueError(
                "Runtime model shape must match the supplied attention geometry and layers"
            )
        if (
            not getattr(config, "bf16", False)
            or getattr(config, "fp8", None)
            or getattr(config, "fp4", None)
        ):
            raise ValueError("Packaged benchmark profiles describe BF16 execution")
        if (
            getattr(config, "multi_latent_attention", False)
            or getattr(config, "experimental_attention_variant", None)
            or getattr(config, "linear_attention_type", None)
            or getattr(config, "is_hybrid_model", False)
        ):
            raise ValueError(
                "Ordinary GQA parameters cannot represent another attention architecture"
            )
        backend = getattr(config, "attention_backend", None)
        if getattr(backend, "name", backend) != "fused":
            raise ValueError("Packaged benchmark profiles require fused attention")
        communication = getattr(config, "cp_comm_type", None)
        if communication != "p2p" and not (
            isinstance(communication, list)
            and communication
            and all(value == "p2p" for value in communication)
        ):
            raise ValueError("Packaged multi-CP parameters describe TE P2P communication")
        if getattr(config, "cuda_graph_impl", None) not in (None, "none", ""):
            raise ValueError("This runtime recipe uses eager execution")
        recompute = getattr(config, "recompute_granularity", None)
        if recompute not in (None, "full"):
            raise ValueError(
                "The benchmark layer program covers recompute_granularity None or 'full', "
                f"got {recompute!r}"
            )
        if (recompute == "full") != self.config.recompute_forward:
            raise ValueError("Runtime recomputation must match the supplied E2E layer program")
        if recompute == "full" and getattr(config, "recompute_method", None) != "uniform":
            raise ValueError("The E2E recompute program requires all layers to be recomputed")

    def metadata(self):
        """Report selected predictor identity and user-supplied non-attention scope."""
        return dict(
            name=self.name,
            hardware=self.hardware,
            nvlink_domain_size=self.nvlink_domain_size,
            profile=self.profile,
            predictor_options=dict(self.predictor_options),
            attention_geometry=list(self.attention_geometry),
            layer_types=list(self.layer_types),
            source_hashes=dict(self.source_hashes),
            non_attention=asdict(self.non_attention),
            non_attention_source=self.non_attention.source,
            non_attention_scope=(
                "fixed ms per layer, microbatch and rank; includes projections and EP"
            ),
            scope="core attention + benchmark model-E2E timeline",
            provisional=True,
            independent_accuracy_passed=False,
            assumptions=list(E2EEstimate.assumptions),
        )
