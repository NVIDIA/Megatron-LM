# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared test fixtures and helpers for inference tests."""

import itertools
from collections import OrderedDict, deque

import numpy as np

from megatron.core.inference.config import (
    MediaCacheCoordinatorPolicy,
    PrefixCachingCoordinatorPolicy,
)
from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)


def make_coordinator_direct(
    data_parallel_size=2,
    block_size_tokens=4,
    enable_prefix_caching=True,
    deterministic_mode=True,
    prefix_caching_routing_alpha=0.5,
    prefix_cache_ttl_seconds=300.0,
    max_requests=10,
    policy=PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
    media_policy=MediaCacheCoordinatorPolicy.AFFINITY,
    vision_embedding_cache_enabled=True,
    tokenizer=None,
    rank_name_template="rank_{}",
):
    """Create a coordinator with mock ZMQ, for unit testing routing logic.

    Returns the coordinator instance with fake rank identities.

    Args:
        data_parallel_size: Number of DP ranks.
        block_size_tokens: Block size in tokens.
        enable_prefix_caching: Whether prefix caching is enabled.
        deterministic_mode: If True, sort identities for deterministic ordering.
        prefix_caching_routing_alpha: Alpha for prefix-aware scoring.
        prefix_cache_ttl_seconds: How long a routed block is assumed still held.
        max_requests: Max requests per rank (None disables vectorized scoring).
        policy: Prefix caching coordinator routing policy.
        media_policy: Media-cache coordinator routing policy.
        vision_embedding_cache_enabled: Whether projected media embeddings are cached.
        tokenizer: Optional tokenizer instance (set on the coordinator).
        rank_name_template: Format string for rank names, e.g. ``"rank_{}"``
            or ``"rank-{}"``.  The integer rank index is substituted.
    """
    coordinator = object.__new__(DataParallelInferenceCoordinator)
    coordinator.tokenizer = tokenizer
    coordinator.data_parallel_size = data_parallel_size
    coordinator.block_size_tokens = block_size_tokens
    coordinator.enable_prefix_caching = enable_prefix_caching
    coordinator.prefix_caching_coordinator_policy = policy
    coordinator.prefix_caching_routing_alpha = prefix_caching_routing_alpha
    coordinator.prefix_cache_ttl_seconds = prefix_cache_ttl_seconds
    coordinator._hash_expiry = deque()
    coordinator.media_cache_coordinator_policy = media_policy
    coordinator.media_cache_routing_weight = 1.0
    coordinator.vision_embedding_cache_enabled = vision_embedding_cache_enabled
    coordinator.max_requests = max_requests

    # Create fake rank identities.
    coordinator.identities_of_data_parallel_ranks = deque(
        [rank_name_template.format(i).encode() for i in range(data_parallel_size)]
    )
    coordinator.removed_engine_identities = set()
    if deterministic_mode:
        coordinator.identities_of_data_parallel_ranks = deque(
            sorted(coordinator.identities_of_data_parallel_ranks)
        )
    coordinator.data_parallel_rank_iterator = itertools.cycle(
        coordinator.identities_of_data_parallel_ranks
    )

    n_ranks = data_parallel_size
    coordinator._hash_table = {}
    coordinator._hash_assignment_counter = 0
    coordinator._media_cache_affinity = OrderedDict()
    coordinator._media_cache_affinity_max_entries = 65536

    sorted_identities = sorted(coordinator.identities_of_data_parallel_ranks)
    coordinator.identity_to_rank_index = {
        identity: idx for idx, identity in enumerate(sorted_identities)
    }

    coordinator._pending_counts = np.zeros(n_ranks, dtype=np.int32)
    coordinator._identities_list = list(sorted_identities)

    return coordinator
