# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configure coordinator-native disaggregated inference shards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch.distributed as dist

from megatron.core.inference.shards_spec import (
    InferenceShardSpec,
    normalize_shard_specs,
    resolve_inference_shard,
)
from megatron.core.utils import get_pg_rank, get_pg_size

PREFILL = "prefill"
DECODE = "decode"


@dataclass(frozen=True)
class NativeDisaggregationConfig:
    """One engine's role and connection to the native coordinator."""

    role: Literal["prefill", "decode"]
    identity: str
    spawn_coordinator: bool
    coordinator_group: dist.ProcessGroup


def validate_disaggregation_shards(
    shards: str | Sequence[InferenceShardSpec] | Sequence[dict], world_size: int
) -> list[InferenceShardSpec]:
    """Normalize and validate a prefill/decode shard layout.

    Args:
        shards: Any shard layout accepted by ``normalize_shard_specs``.
        world_size: Total number of ranks that the shards must partition.

    Returns:
        The normalized shard specifications.

    Raises:
        ValueError: If any shard lacks a role or the layout does not contain
            at least one prefill and one decode shard.
    """
    specs = normalize_shard_specs(shards, world_size)
    untagged = [spec for spec in specs if spec.role not in (PREFILL, DECODE)]
    if untagged:
        raise ValueError(
            "every disaggregated shard must declare role=prefill or role=decode; "
            f"{len(untagged)} shard(s) did not: {untagged}"
        )
    roles = {spec.role for spec in specs}
    if not {PREFILL, DECODE}.issubset(roles):
        raise ValueError("disaggregation needs at least one prefill and one decode shard")
    return specs


def configure_prebuilt_disagg_engine(engine: Any) -> None:
    """Configure an engine for coordinator-native disaggregation."""
    shards = engine.context.config.disaggregation_shards
    if shards is None:
        raise ValueError("disaggregation_shards must be configured")
    specs = validate_disaggregation_shards(shards, dist.get_world_size())
    ctx = engine.context
    # Decode admits imported KV through the prefix cache.
    if not ctx.enable_prefix_caching:
        raise ValueError("disaggregation requires prefix caching")
    rank = dist.get_rank()
    assignment = resolve_inference_shard(specs, dist.get_world_size(), rank)
    spec = assignment.spec
    pgc = engine.pg_collection
    model_config = engine.controller.inference_wrapped_model.model.config
    built = tuple(
        get_pg_size(group) for group in (pgc.tp, pgc.pp, pgc.dp, pgc.ep, pgc.cp, pgc.expt_tp)
    )
    configured = (
        model_config.tensor_model_parallel_size,
        model_config.pipeline_model_parallel_size,
        model_config.context_parallel_size,
        model_config.expert_model_parallel_size,
        model_config.expert_tensor_parallel_size,
    )
    expected_built = (spec.tp, spec.pp, spec.dp, spec.ep, spec.cp, spec.expt_tp)
    expected_configured = (spec.tp, spec.pp, spec.cp, spec.ep, spec.expt_tp)
    if built != expected_built or configured != expected_configured:
        raise ValueError(
            f"inference shard {assignment.index} parallelism mismatch: "
            f"process groups {built}, expected {expected_built}; "
            f"model config {configured}, expected {expected_configured}"
        )

    # Each shard replica needs a distinct coordinator identity.
    dp_rank = get_pg_rank(engine.pg_collection.dp)
    engine.set_disaggregation_config(
        role=assignment.spec.role,
        identity=f"{assignment.spec.role}_s{assignment.index}_dp{dp_rank}",
        spawn_coordinator=(rank == 0),
        coordinator_group=dist.group.WORLD,
    )
