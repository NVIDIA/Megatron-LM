# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Measured Qwen3-30B-A3B costs for the declared H100 P2P recipe.

Attention predictions include the complete self-attention module, including its
projections and exposed CP communication. Non-attention contributions depend on
valid tokens and are averaged over each forced-balanced EP8 block. This is a
recipe-specific approximation whose mixed-layout accuracy requires validation.
This module performs CPU arithmetic only.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

_MODEL_NAME = "qwen3_h100_p2p"
_CP_SIZES = (1, 2, 4, 8)
_MODEL_CONFIG = {
    "num_layers": 48,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_query_groups": 4,
    "kv_channels": 128,
    "num_moe_experts": 128,
    "moe_router_topk": 8,
    "moe_ffn_hidden_size": 768,
    "moe_router_force_load_balancing": True,
    "cp_partition_mode": "zigzag",
    "experimental_attention_variant": None,
    "multi_latent_attention": False,
    "qk_layernorm": True,
    "normalization": "RMSNorm",
    "attention_backend": "fused",
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "expert_model_parallel_size": 8,
    "expert_tensor_parallel_size": 1,
    "bf16": True,
    "fp16": False,
    "fp8": None,
    "fp4": None,
}
_FEATURES = {
    "bias",
    "logical_s1_per_rank",
    "logical_s2_per_rank",
    "padded_s1_per_rank",
    "padded_s2_per_rank",
    "sequence_count",
    "zigzag_max_causal_work",
}


def _nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _zigzag_max_causal_work(logical_lengths, physical_lengths, cp, token_unit):
    """Return twice the largest rank's valid causal query work, in token-unit squared.

    A physical sequence of P tokens has 2*CP equal blocks. Rank r owns blocks
    r and 2*CP-1-r. A valid query at sequence offset q contributes q+1 causal
    keys. Only q < logical_length contributes, even when the physical tail is
    much larger. Each interval is summed exactly with integer arithmetic.
    """
    cp = _positive_int(cp, "causal-work CP size")
    token_unit = _positive_int(token_unit, "causal-work token unit")
    logical_lengths = tuple(logical_lengths)
    physical_lengths = tuple(physical_lengths)
    if not logical_lengths or len(logical_lengths) != len(physical_lengths):
        raise ValueError("Causal-work logical and physical sequence arrays must match")
    work = [0] * cp
    for logical, physical in zip(logical_lengths, physical_lengths):
        if (
            type(logical) is not int
            or logical < 0
            or type(physical) is not int
            or physical < 1
            or logical > physical
            or physical % (2 * cp)
        ):
            raise ValueError("Causal-work storage must be positive and divisible by 2*CP")
        block_size = physical // (2 * cp)
        for rank in range(cp):
            for block in (rank, 2 * cp - 1 - rank):
                begin = block * block_size
                end = min(begin + block_size, logical)
                if end > begin:
                    count = end - begin
                    work[rank] += count * (begin + end + 1) // 2
    return 2 * max(work) / (token_unit * token_unit)


class Qwen3H100P2PCostModel:
    """Predict one measured recipe, with independent coefficients for each CP tier.

    S1 features are ``sum(length / token_unit) / cp`` and S2 features are
    ``sum((length / token_unit) ** 2) / cp``. Padded features use the exact
    dataset/CP alignment and configured tail allocation. ``sequence_count``
    counts real sequences, excluding any appended padding sequence.
    Optional ``zigzag_max_causal_work`` uses physical blocks and valid queries
    to represent the slowest rank, rather than an average across CP ranks.
    """

    name = _MODEL_NAME
    placement_invariant = True
    supported_cp_sizes = _CP_SIZES

    def __init__(
        self, geometry, *, transport="classic_p2p", profile_path=None, profile_sha256=None
    ) -> None:
        self.geometry = geometry
        self.transport = transport
        path = (
            Path(profile_path) if profile_path is not None else Path(__file__).with_suffix(".json")
        )
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if profile_path is None:
            origin = json.loads((Path(__file__).parents[1] / "ORIGIN.json").read_text())
            expected = origin["runtime_parameters_sha256"].get("models/qwen3_h100_p2p.json")
            if expected is None:
                raise ValueError("The built-in H100 profile has no recorded parameter hash")
            if digest != expected:
                raise ValueError("H100 P2P calibration hash mismatch")
        if profile_sha256 is not None and profile_sha256 != digest:
            raise ValueError("H100 P2P calibration hash mismatch")
        self.profile = json.loads(raw)
        self.source_hashes = {"qwen3_h100_p2p.json": digest}
        try:
            self._validate_profile()
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError("Malformed Qwen3 H100 P2P calibration profile") from error
        self.check_domain(geometry, None)

    def _validate_profile(self):
        data = self.profile
        if (
            type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or data.get("name") != self.name
            or data.get("unit") != "ms_per_layer"
        ):
            raise ValueError("Unsupported Qwen3 H100 P2P calibration schema")
        profiles = data["profiles"]
        if self.transport not in ("classic_p2p", "native_p2p"):
            raise ValueError("Unknown H100 P2P calibration transport")
        if self.transport not in profiles:
            raise ValueError(f"No measured H100 P2P profile for transport {self.transport!r}")
        self.table = profiles[self.transport]
        for name in ("source", "validation"):
            if not isinstance(self.table.get(name), dict) or not self.table[name]:
                raise ValueError(f"H100 P2P calibration requires {name} metadata")
        self.domain = data["domain"]
        expected = {
            "hardware": "h100",
            "nodes": 4,
            "ranks_per_node": 8,
            "world_size": 32,
            "ep_size": 8,
            "tokens_per_rank": 4096,
            "dataset_alignment": 64,
            "cp_sizes": list(self.supported_cp_sizes),
            "partition": "zigzag",
        }
        if any(self.domain.get(key) != value for key, value in expected.items()):
            raise ValueError("The profile is outside the 32-H100 P2P calibration recipe")
        self.config_constraints = self.domain["config"]
        if not _MODEL_CONFIG.keys() <= self.config_constraints.keys() or any(
            self.config_constraints[key] != value for key, value in _MODEL_CONFIG.items()
        ):
            raise ValueError("The profile does not describe BF16 Qwen3-30B-A3B TP1/PP1/EP8")
        self.layer_count = self.config_constraints["num_layers"]
        self.minimum_length = _positive_int(self.domain["min_logical_length"], "minimum length")
        self.maximum_length = _positive_int(self.domain["max_logical_length"], "maximum length")
        if self.minimum_length > self.maximum_length:
            raise ValueError("The calibrated logical-length interval is empty")
        self.maximum_sequences = _positive_int(self.domain["max_group_seqs"], "sequence limit")
        if self.maximum_sequences > 64:
            raise ValueError("The H100 profile supports at most 64 real sequences per pack")
        if self.domain["tail_padding"] not in ("extend_last", "append_dummy_seq"):
            raise ValueError("Unknown calibrated tail-padding policy")
        features = data["features"]
        self.feature_names = tuple(features["names"])
        if (
            not self.feature_names
            or len(set(self.feature_names)) != len(self.feature_names)
            or not set(self.feature_names) <= _FEATURES
        ):
            raise ValueError("Invalid H100 P2P feature names")
        self.token_unit = _positive_int(features["token_unit"], "feature token unit")
        self.coefficients = {}
        if set(self.table["attention"]) != {str(cp) for cp in self.supported_cp_sizes}:
            raise ValueError("Attention calibration must cover exactly CP1/2/4/8")
        for cp in self.supported_cp_sizes:
            phases = self.table["attention"][str(cp)]
            self.coefficients[cp] = {}
            for phase in ("forward", "backward"):
                values = phases[phase]
                if len(values) != len(self.feature_names):
                    raise ValueError("Attention coefficients and features have different lengths")
                values = tuple(_nonnegative(value, "attention coefficient") for value in values)
                if not any(values):
                    raise ValueError("Attention coefficients cannot all be zero")
                self.coefficients[cp][phase] = values
        self.remainder = {}
        for phase in ("forward", "backward"):
            coefficients = self.table["remainder_ms"][phase]
            if len(coefficients) != 2:
                raise ValueError("EP remainder requires bias and valid-token coefficients")
            self.remainder[phase] = tuple(
                _nonnegative(value, "non-attention remainder coefficient") for value in coefficients
            )
        self.outer = {
            phase: _nonnegative(self.table["microbatch_overhead_ms"][phase], "microbatch overhead")
            for phase in ("forward", "backward")
        }
        self.iteration_tail_ms = _nonnegative(self.table["iteration_tail_ms"], "iteration tail")

    def check_domain(self, geometry, config=None):
        """Reject unmeasured geometry, transport, precision and model configurations."""
        if (
            geometry != self.geometry
            or geometry.world_size != self.domain["world_size"]
            or geometry.ep_size != self.domain["ep_size"]
            or geometry.tokens_per_rank != self.domain["tokens_per_rank"]
            or geometry.dataset_alignment != self.domain["dataset_alignment"]
            or not geometry.allowed_cp_sizes
            or not set(geometry.allowed_cp_sizes) <= set(self.supported_cp_sizes)
            or not geometry.align_group_starts
            or geometry.max_group_seqs is None
            or geometry.max_group_seqs > self.maximum_sequences
        ):
            raise ValueError("Qwen3 H100 P2P geometry is outside the measured calibration domain")
        if config is None:
            return
        actual_transport = (
            "native_p2p" if getattr(config, "use_native_cp_transport", False) else "classic_p2p"
        )
        if actual_transport != self.transport:
            raise ValueError("H100 P2P calibration transport differs from the execution transport")
        missing = object()
        for key, expected in self.config_constraints.items():
            if key == "cp_comm_type":
                continue
            actual = getattr(config, key, missing)
            if key == "attention_backend":
                actual = getattr(actual, "name", actual)
            if actual != expected:
                raise ValueError(f"Qwen3 H100 P2P calibration requires {key}={expected!r}")
        if getattr(config, "virtual_pipeline_model_parallel_size", None) is not None:
            raise ValueError("Qwen3 H100 P2P calibration requires PP1/VPP1")
        communication = getattr(config, "cp_comm_type", None)
        if communication != "p2p" and not (
            isinstance(communication, list)
            and communication
            and all(c == "p2p" for c in communication)
        ):
            raise ValueError("Qwen3 H100 P2P calibration requires cp_comm_type='p2p'")
        padding = getattr(config, "pad_packed_seq_alignment", None)
        if padding not in ("max", self.domain["tokens_per_rank"]):
            raise ValueError("Qwen3 H100 P2P calibration requires fixed local-capacity padding")
        tail_policy = getattr(config, "thd_tail_padding_policy", None) or "append_dummy_seq"
        if tail_policy != self.domain["tail_padding"]:
            raise ValueError("Qwen3 H100 P2P tail padding differs from the measured recipe")

    def _features(self, lengths, cp):
        if type(cp) is not int or cp not in self.geometry.allowed_cp_sizes:
            raise ValueError("CP size is outside the measured H100 P2P tiers")
        maximum_sequences = min(self.maximum_sequences, self.geometry.max_group_seqs)
        if (
            not lengths
            or len(lengths) > maximum_sequences
            or any(
                type(length) is not int or not self.minimum_length <= length <= self.maximum_length
                for length in lengths
            )
        ):
            raise ValueError("Pack lengths are outside the measured H100 P2P domain")
        padded = [self.geometry.stored_length(length, cp) for length in lengths]
        remaining = self.geometry.capacity(cp) - sum(padded)
        if remaining < 0:
            raise ValueError("Pack exceeds the measured physical token capacity")
        if remaining:
            if self.domain["tail_padding"] == "extend_last":
                padded[-1] += remaining
            else:
                padded.append(remaining)
        unit = self.token_unit
        features = {
            "bias": 1.0,
            "logical_s1_per_rank": sum(lengths) / unit / cp,
            "logical_s2_per_rank": sum((length / unit) ** 2 for length in lengths) / cp,
            "padded_s1_per_rank": sum(padded) / unit / cp,
            "padded_s2_per_rank": sum((length / unit) ** 2 for length in padded) / cp,
            "sequence_count": float(len(lengths)),
        }
        if "zigzag_max_causal_work" in self.feature_names:
            query_lengths = list(lengths)
            if remaining and self.domain["tail_padding"] == "append_dummy_seq":
                # The appended dummy has ordinary cu_seqlens, so all its queries
                # participate in attention despite being excluded from MoE.
                query_lengths.append(remaining)
            features["zigzag_max_causal_work"] = _zigzag_max_causal_work(
                query_lengths, padded, cp, unit
            )
        return tuple(features[name] for name in self.feature_names)

    def _check_ranks(self, cp, ranks):
        if ranks is None:
            return
        ranks = tuple(ranks)
        if (
            not ranks
            or len(ranks) != cp
            or any(type(rank) is not int for rank in ranks)
            or ranks != tuple(range(ranks[0], ranks[0] + cp))
            or not self.geometry.supports_group(ranks[0], cp)
            or ranks[0] // 8 != ranks[-1] // 8
        ):
            raise ValueError("H100 P2P calibration requires an aligned group inside one H100 node")

    @lru_cache(maxsize=65536)
    def _costs(self, lengths, cp):
        features = self._features(lengths, cp)
        values = tuple(
            sum(value * weight for value, weight in zip(features, self.coefficients[cp][phase]))
            for phase in ("forward", "backward")
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Invalid measured H100 P2P prediction")
        return values

    def forward(self, lengths, cp, *, ranks=None):
        """Predict complete self-attention forward milliseconds per layer and pack."""
        self._check_ranks(cp, ranks)
        return self._costs(tuple(lengths), cp)[0]

    def backward(self, lengths, cp, *, ranks=None):
        """Predict complete self-attention backward milliseconds per layer and pack."""
        self._check_ranks(cp, ranks)
        return self._costs(tuple(lengths), cp)[1]

    def program_costs(self, lengths, cp, *, ranks=None):
        """Return attention times and per-rank forced-balanced EP cost contributions."""
        self._check_ranks(cp, ranks)
        lengths = tuple(lengths)
        forward, backward = self._costs(lengths, cp)
        load = sum(lengths) / self.geometry.tokens_per_rank / cp
        return {
            "att_f": forward,
            "att_b": backward,
            "rest_f": self.remainder["forward"][0] + self.remainder["forward"][1] * load,
            "rest_b": self.remainder["backward"][0] + self.remainder["backward"][1] * load,
        }

    guide_program_costs = program_costs

    def timing(self):
        """Build the measured layer order, per-microbatch overhead and one iteration tail."""
        return {
            "iteration_tail_ms": self.iteration_tail_ms,
            "execution_program": [
                {
                    "layer_count": self.layer_count,
                    "forward": [dict(sync="local", duration_ms=self.outer["forward"])]
                    + [
                        operation
                        for _ in range(self.layer_count)
                        for operation in (
                            dict(sync="cp", cost="att_f"),
                            dict(sync="ep", cost="rest_f", aggregation="mean"),
                        )
                    ],
                    "backward": [
                        operation
                        for _ in range(self.layer_count)
                        for operation in (
                            dict(sync="ep", cost="rest_b", aggregation="mean"),
                            dict(sync="cp", cost="att_b"),
                        )
                    ]
                    + [dict(sync="local", duration_ms=self.outer["backward"])],
                }
            ],
        }

    def clear(self):
        """Drop pack prediction caches before planning another global batch."""
        self._costs.cache_clear()

    def metadata(self):
        """Expose measured provenance and validation without extending their scope."""
        return {
            "name": self.name,
            "source_hashes": dict(self.source_hashes),
            "transport": self.transport,
            "source": self.table["source"],
            "domain": self.domain,
            "validation": self.table["validation"],
            "scope": (
                "Qwen3-30B-A3B BF16 self-attention for the selected P2P transport plus "
                "valid-token EP mean contributions for forced-balanced routing and microbatch "
                "overhead; mixed-layout validation is reported separately"
            ),
        }
