# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configure coordinator-native disaggregated inference shards."""

from __future__ import annotations

from typing import Any, Sequence

import torch.distributed as dist

from megatron.core.inference.shards_spec import (
    InferenceShardSpec,
    normalize_shard_specs,
    resolve_inference_shard,
)
from megatron.core.utils import get_pg_rank

PREFILL = "prefill"
DECODE = "decode"


def _validate_disagg_specs(specs: list[InferenceShardSpec]) -> None:
    """Require at least one shard for each disaggregated role."""
    untagged = [spec for spec in specs if spec.role not in (PREFILL, DECODE)]
    if untagged:
        raise ValueError(
            "every disaggregated shard must declare role=prefill or role=decode; "
            f"{len(untagged)} shard(s) did not: {untagged}"
        )
    roles = {spec.role for spec in specs}
    if not {PREFILL, DECODE}.issubset(roles):
        raise ValueError("disaggregation needs at least one prefill and one decode shard")


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
    _validate_disagg_specs(specs)
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

    # Each shard replica needs a distinct coordinator identity.
    dp_rank = get_pg_rank(engine.pg_collection.dp)
    engine.set_disaggregation_config(
        role=assignment.spec.role,
        identity=f"{assignment.spec.role}_s{assignment.index}_dp{dp_rank}",
        spawn_coordinator=(rank == 0),
        coordinator_group=dist.group.WORLD,
    )
