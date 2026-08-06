# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configure a prefill/decode shard engine for the coordinator-native 2-hop
disaggregated mode; called with an --inference-shards spec. Also holds the
shard helpers shared with the refit path."""

from __future__ import annotations

import functools
from typing import Any, List, Tuple

import torch.distributed as dist

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.shards_spec import (
    InferenceShardSpec,
    parse_inference_shards_spec,
    spec_declares_disaggregation,
)
from megatron.core.utils import get_pg_rank

PREFILL = "prefill"
DECODE = "decode"


def _validate_disagg_specs(specs: List[InferenceShardSpec]) -> None:
    """Check the role layout.

    Any number of prefill and decode instances is allowed; each instance (a
    shard's dp replica) is an independent routing target.
    """
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


@functools.lru_cache(maxsize=None)
def disagg_refit_pools(inference_shards, world_size: int, rank: int = None) -> Tuple[int, int]:
    """Map an --inference-shards spec to (num_dst_pools, dst_pool_index) for
    swap_model_weights.

    Disaggregated serving refits the training model into each shard's
    inference model (disjoint rank windows, possibly at different
    parallelism), so the refit runs one collective pass per shard. Returns
    (1, 0) when the spec is absent or not disaggregated, so callers can pass
    the result unconditionally. Memoized: the result is a pure function of
    the process-constant spec, world size, and rank.
    """
    if rank is None:
        rank = dist.get_rank()
    if not (inference_shards and spec_declares_disaggregation(inference_shards)):
        return 1, 0
    specs = parse_inference_shards_spec(inference_shards, world_size)
    offset = 0
    for index, s in enumerate(specs):
        if offset <= rank < offset + s.world_size:
            return len(specs), index
        offset += s.world_size
    raise RuntimeError(f"rank {rank} not in any disagg shard window")


def configure_prebuilt_disagg_engine(
    engine: Any,
    specs: List[InferenceShardSpec],
    disagg_router: str = "round_robin",
    kv_transport_backend: str = "nixl",
) -> None:
    """Configure an already-built engine for the coordinator-native 2-hop mode.

    Resolves this rank's role from its shard window, checks the prefix-cache
    requirements the hand-off admission relies on, and sets the disagg config
    on the engine (which brings up the transfer agents). The per-request
    hand-off metadata is self-describing, so no layout exchange happens here.
    """
    _validate_disagg_specs(specs)
    ctx = engine.context
    # The decode admits handed-off KV via a prefix-cache hit (the import
    # registers the block hashes), so prefix caching is required.
    assert ctx.enable_prefix_caching, (
        "disaggregation requires prefix caching (enable_prefix_caching=True); "
        "the decode side admits handed-off KV via a prefix-cache hit."
    )
    # ref_zero eviction deregisters blocks the moment their ref count hits 0,
    # which would discard the imported KV before admission sees it.
    assert ctx.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU, (
        "disaggregation requires the LRU prefix-cache eviction policy "
        "(--inference-dynamic-batching-prefix-caching-eviction-policy lru); "
        f"got {ctx.prefix_caching_eviction_policy!r}."
    )
    rank = dist.get_rank()

    # Locate this rank's shard. Shard windows are contiguous (tp*pp*dp ranks
    # each) regardless of the intra-shard rank ordering.
    offset = 0
    my_index = None
    my_spec = None
    for i, s in enumerate(specs):
        if offset <= rank < offset + s.world_size:
            my_index, my_spec = i, s
            break
        offset += s.world_size
    assert my_spec is not None, f"rank {rank} not in any disagg shard window"

    # Unique per instance (shard index + dp replica), so each prefill/decode
    # replica gets a distinct ZMQ identity.
    dp_rank = get_pg_rank(engine.pg_collection.dp)
    engine.set_disaggregation_config(
        role=my_spec.role,
        identity=f"{my_spec.role}_s{my_index}_dp{dp_rank}",
        spawn_coordinator=(rank == 0),
        disagg_router=disagg_router,
        kv_transport_backend=kv_transport_backend,
    )
