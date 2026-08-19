# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configure coordinator-native disaggregated inference shards."""

from __future__ import annotations

from typing import Any, List

import torch.distributed as dist

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.shards_spec import InferenceShardSpec
from megatron.core.utils import get_pg_rank

PREFILL = "prefill"
DECODE = "decode"


def _validate_disagg_specs(specs: List[InferenceShardSpec]) -> None:
    """Require at least one shard for each disaggregated role."""
    prefill = [s for s in specs if s.role == PREFILL]
    decode = [s for s in specs if s.role == DECODE]
    untagged = [s for s in specs if s.role not in (PREFILL, DECODE)]
    assert not untagged, (
        f"every shard must declare role=prefill or role=decode for "
        f"disaggregation; {len(untagged)} shard(s) had none: {untagged}"
    )
    assert (
        prefill and decode
    ), "disaggregation needs at least one prefill shard and one decode shard."


def configure_prebuilt_disagg_engine(
    engine: Any,
    specs: List[InferenceShardSpec],
    disagg_router: str = "round_robin",
    kv_transport_backend: str = "nixl",
) -> None:
    """Configure an engine for coordinator-native disaggregation."""
    _validate_disagg_specs(specs)
    ctx = engine.context
    # Decode admits imported KV through the prefix cache.
    assert ctx.enable_prefix_caching, (
        "disaggregation requires prefix caching (enable_prefix_caching=True); "
        "the decode side admits handed-off KV via a prefix-cache hit."
    )
    # REF_ZERO would deregister imported blocks before decode admission.
    assert ctx.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU, (
        "disaggregation requires the LRU prefix-cache eviction policy "
        "(--inference-dynamic-batching-prefix-caching-eviction-policy lru); "
        f"got {ctx.prefix_caching_eviction_policy!r}."
    )
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
        disagg_router=disagg_router,
        kv_transport_backend=kv_transport_backend,
        coordinator_group=dist.group.WORLD,
    )
