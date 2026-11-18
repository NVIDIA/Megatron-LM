# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ordinary GQA: predicted core attention plus the user's fixed non-attention times."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from functools import lru_cache

from ..runtime.planner import MAX_PRICED_GROUP_SEQUENCES
from ..runtime.timeline import layer_program
from . import attention
from .base import NVLINK_DOMAIN_SIZES, uses_p2p


def _duration(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite nonnegative number of milliseconds")
    return float(value)


def _layer_duration(non_attention, phase):
    """The user's fixed time of one layer: a number, or a mapping that names ``gqa``."""
    value = non_attention[phase + "_ms_per_layer"]
    if isinstance(value, Mapping):
        if "gqa" not in value:
            raise ValueError("missing non-attention measurement for gqa")
        value = value["gqa"]
    return _duration(value, phase + " non-attention")


class BenchmarkCostModel:
    """Core-attention predictions plus explicit fixed non-attention measurements.

    Every layer is a GQA layer. ``models.attention`` predicts the core attention of a CP
    group (the attention kernel with its context-parallel communication, without the
    projections) for the forward and the backward pass, at every CP size including 1; the
    rest of a layer is the user's measurement. No coefficient is refitted, no residual is
    added, and no missing non-attention time is inferred.
    """

    name = "dcp_benchmark"
    placement_invariant = False
    # Layer types a per-type ``non_attention`` mapping must name for this model.
    non_attention_layer_types = ("gqa",)
    # (num_attention_heads, num_query_groups, kv_channels) with packaged attention parameters.
    attention_geometries = attention.GEOMETRIES
    # Largest DPxCP and largest CP group of this model. The predictor has one form for every
    # ring size; 64 is the largest size at which it was compared with its source model.
    max_dp_cp_size = 64
    largest_cp_size = 64

    def __init__(
        self,
        geometry,
        *,
        hardware,
        query_heads,
        kv_heads,
        head_dim,
        non_attention,
        layer_count=1,
        first_forward_dp_cp_sync=True,
        recompute_forward=False,
    ):
        if not isinstance(hardware, str) or hardware not in NVLINK_DOMAIN_SIZES:
            raise ValueError("hardware must be h100 or gb200; GB300 is not this calibration")
        for flag in (first_forward_dp_cp_sync, recompute_forward):
            if type(flag) is not bool:
                raise ValueError("first_forward_dp_cp_sync and recompute_forward must be booleans")
        self.geometry, self.hardware, self.layer_count = geometry, hardware, layer_count
        self.nvlink_domain_size = NVLINK_DOMAIN_SIZES[hardware]
        self.attention_geometry = (query_heads, kv_heads, head_dim)
        self.first_forward_dp_cp_sync = first_forward_dp_cp_sync
        self.recompute_forward = recompute_forward
        self.non_attention = dict(non_attention)
        self.non_attention.setdefault("source", "user-provided non-attention times")
        source = self.non_attention["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must identify the user-supplied measurements")
        required = {"forward_ms_per_layer", "backward_ms_per_layer"}
        if not required <= set(self.non_attention) <= required | {"source", "iteration_tail_ms"}:
            raise TypeError(
                "non_attention takes forward_ms_per_layer and backward_ms_per_layer, and "
                "optionally source and iteration_tail_ms"
            )
        self.iteration_tail_ms = _duration(
            self.non_attention.get("iteration_tail_ms", 0.0), "iteration_tail_ms"
        )
        layer = (
            "gqa:forward",
            _layer_duration(self.non_attention, "forward"),
            "gqa:backward",
            _layer_duration(self.non_attention, "backward"),
        )
        self._timing = dict(
            execution_program=[
                layer_program([layer] * layer_count, recompute_forward=recompute_forward)
            ],
            iteration_tail_ms=self.iteration_tail_ms,
            first_forward_dp_cp_sync=first_forward_dp_cp_sync,
        )
        # Price one token with the packaged predictor: an attention geometry without
        # coefficients, a per-rank buffer above what it prices or a changed coefficient file
        # fails here, before the first plan.
        self._packaged_predictor((1,), 1, (0,))
        # ``predictor(lengths, cp, ranks) -> (forward_ms, backward_ms)``; tests replace it.
        self.predictor = self._packaged_predictor
        self.source_hashes = {
            "attention_parameters": attention.PARAMETERS_SHA256[hardware],
            "non_attention": hashlib.sha256(
                json.dumps(self.non_attention, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    def _packaged_predictor(self, lengths, cp, ranks):
        # The per-rank capacity is fixed, so the pack buffer grows with the CP size.
        return attention.predict_pair(
            self.hardware,
            *self.attention_geometry,
            lengths,
            cp,
            sequence_alignment=self.geometry.dataset_alignment,
            pack_alignment=cp * self.geometry.tokens_per_rank,
            ranks=ranks,
        )

    @lru_cache(maxsize=65536)
    def _costs(self, lengths, cp, ranks):
        if cp not in self.geometry.allowed_cp_sizes:
            raise ValueError("CP size is outside the configured prediction domain")
        if not lengths or len(lengths) > min(
            self.geometry.max_group_seqs or MAX_PRICED_GROUP_SEQUENCES, MAX_PRICED_GROUP_SEQUENCES
        ):
            raise ValueError("A pack needs one to the configured maximum real sequences")
        forward, backward = self.predictor(lengths, cp, ranks)
        return {"gqa:forward": float(forward), "gqa:backward": float(backward)}

    def program_costs(self, lengths, cp, *, ranks=None):
        """Return the core-attention forward and backward prediction of one pack."""
        return dict(self._costs(tuple(lengths), cp, None if ranks is None else tuple(ranks)))

    def forward(self, lengths, cp, *, ranks=None):
        """Return the per-layer core forward time for the search heuristic."""
        return self.program_costs(lengths, cp, ranks=ranks)["gqa:forward"]

    def backward(self, lengths, cp, *, ranks=None):
        """Return the per-layer core backward time for the search heuristic."""
        return self.program_costs(lengths, cp, ranks=ranks)["gqa:backward"]

    def timing(self):
        """Expose the layer program: attention per CP group, the rest up to the EP group."""
        return self._timing

    def clear(self):
        """Reset the per-global-batch core prediction cache."""
        self._costs.cache_clear()

    def check_domain(self, geometry, config=None):
        """Reject incompatible storage, model structure or execution configuration."""
        if config is None:
            return
        if getattr(config, "thd_tail_padding_policy", None) != "extend_last":
            raise ValueError("The attention predictor requires extend_last tail padding")
        if getattr(config, "pad_packed_seq_alignment", None) not in (
            "max",
            geometry.tokens_per_rank,
        ):
            raise ValueError("The attention predictor requires fixed per-rank capacity padding")
        if getattr(config, "pipeline_model_parallel_size", 1) != 1 or getattr(
            config, "virtual_pipeline_model_parallel_size", None
        ) not in (None, 1):
            raise ValueError("The current runtime scheduler adapter requires PP1/VPP1")
        if getattr(config, "tensor_model_parallel_size", 1) != 1:
            raise ValueError("The current benchmark rank/attention mapping requires TP1")
        if (
            not getattr(config, "bf16", False)
            or getattr(config, "fp8", None)
            or getattr(config, "fp4", None)
        ):
            raise ValueError("The attention parameters describe BF16 execution")
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
            raise ValueError("The attention parameters require fused attention")
        if not uses_p2p(config):
            raise ValueError("The attention parameters describe TE P2P communication")
        if getattr(config, "cuda_graph_impl", None) not in (None, "none", ""):
            raise ValueError("This runtime recipe uses eager execution")
        recompute = getattr(config, "recompute_granularity", None)
        if recompute not in (None, "full"):
            raise ValueError(
                "The benchmark layer program covers recompute_granularity None or 'full', "
                f"got {recompute!r}"
            )
        if recompute == "full" and getattr(config, "recompute_method", None) != "uniform":
            raise ValueError("The recompute layer program requires all layers to be recomputed")

    def metadata(self):
        """Report the predictor identity and the scope of the user's non-attention times."""
        return dict(
            name=self.name,
            hardware=self.hardware,
            nvlink_domain_size=self.nvlink_domain_size,
            attention_geometry=list(self.attention_geometry),
            layer_count=self.layer_count,
            source_hashes=dict(self.source_hashes),
            non_attention=dict(self.non_attention, iteration_tail_ms=self.iteration_tail_ms),
            non_attention_source=self.non_attention["source"],
            non_attention_scope=(
                "fixed ms per layer, microbatch and rank; includes projections and EP"
            ),
            scope="core attention + fixed non-attention layer program",
            provisional=True,
            independent_accuracy_passed=False,
        )
