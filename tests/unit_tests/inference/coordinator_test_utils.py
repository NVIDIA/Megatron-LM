# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared test fixtures and helpers for inference tests."""

import itertools
from collections import deque

import numpy as np

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.hash_rank_table import HashRankTable


def make_coordinator_direct(
    data_parallel_size=2,
    block_size_tokens=4,
    enable_prefix_caching=True,
    deterministic_mode=True,
    prefix_caching_routing_alpha=0.5,
    max_requests=None,
    policy=PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
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
        max_requests: Max requests per rank (None disables vectorized scoring).
        policy: Prefix caching coordinator routing policy.
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
    coordinator.max_requests = max_requests

    # Create fake rank identities.
    coordinator.identities_of_data_parallel_ranks = deque(
        [rank_name_template.format(i).encode() for i in range(data_parallel_size)]
    )
    if deterministic_mode:
        coordinator.identities_of_data_parallel_ranks = deque(
            sorted(coordinator.identities_of_data_parallel_ranks)
        )
    coordinator.data_parallel_rank_iterator = itertools.cycle(
        coordinator.identities_of_data_parallel_ranks
    )

    n_ranks = data_parallel_size
    coordinator.hash_table = HashRankTable(n_ranks)
    coordinator._round_robin_idx = 0

    sorted_identities = sorted(coordinator.identities_of_data_parallel_ranks)
    coordinator.identity_to_rank_index = {
        identity: idx for idx, identity in enumerate(sorted_identities)
    }

    coordinator._pending_counts = np.zeros(n_ranks, dtype=np.int32)
    coordinator._identities_list = list(sorted_identities)
    coordinator._active_mask = np.ones(n_ranks, dtype=bool)

    return coordinator
