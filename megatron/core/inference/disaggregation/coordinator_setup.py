# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configure coordinator-native disaggregated inference shards."""

from __future__ import annotations

from typing import Any

import torch.distributed as dist

from megatron.core.inference.shards_spec import InferenceShardSpec, normalize_shard_specs
from megatron.core.utils import get_pg_rank

PREFILL = "prefill"
DECODE = "decode"


def _validate_disagg_specs(specs: list[InferenceShardSpec]) -> None:
    """Require at least one shard for each disaggregated role."""
    untagged = [s for s in specs if s.role not in (PREFILL, DECODE)]
    if untagged:
        raise ValueError(
            "every disaggregated shard must declare role=prefill or role=decode; "
            f"{len(untagged)} shard(s) did not: {untagged}"
        )
    roles = {s.role for s in specs}
    if not {PREFILL, DECODE}.issubset(roles):
        raise ValueError("disaggregation needs at least one prefill and one decode shard")


def configure_prebuilt_disagg_engine(engine: Any) -> None:
    """Configure an engine for coordinator-native disaggregation."""
    shards = engine.context.config.disaggregation_shards
    if shards is None:
        raise ValueError("disaggregation_shards must be configured")
    specs = normalize_shard_specs(shards, dist.get_world_size())
    _validate_disagg_specs(specs)
    ctx = engine.context
    # Decode admits imported KV through the prefix cache.
    if not ctx.enable_prefix_caching:
        raise ValueError("disaggregation requires prefix caching")
    rank = dist.get_rank()

    # Shards occupy contiguous world-rank windows.
    offset = 0
    my_index = None
    my_spec = None
    for i, s in enumerate(specs):
        if offset <= rank < offset + s.world_size:
            my_index, my_spec = i, s
            break
        offset += s.world_size
    assert my_spec is not None, f"rank {rank} not in any disagg shard window"

    # Each shard replica needs a distinct coordinator identity.
    dp_rank = get_pg_rank(engine.pg_collection.dp)
    engine.set_disaggregation_config(
        role=my_spec.role,
        identity=f"{my_spec.role}_s{my_index}_dp{dp_rank}",
        spawn_coordinator=(rank == 0),
        coordinator_group=dist.group.WORLD,
    )
