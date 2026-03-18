# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for prefix-cache-aware coordinator routing.

Validates that the DataParallelInferenceCoordinator correctly computes block
hashes from prompts, routes requests to the DP rank with the longest consecutive
prefix match, and maintains per-rank shadow state (cached hashes and timestamps).
"""

import asyncio
import itertools
from collections import deque
from typing import Dict, Optional
from unittest.mock import MagicMock

import msgpack
import pytest
import torch

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, RequestEntry
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
    compute_block_hashes_batched,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_asyncio_loop
from tests.unit_tests.test_utilities import Utils

try:
    import zmq

    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False


# ============================================================================
# Shared fixtures and helpers
# ============================================================================

DEFAULT_PORT = 46582
ZMQ_FLAKY_SHUTDOWN = True

BLOCK_SIZE = 4


class DummyTokenizer:
    """Dummy tokenizer that splits on whitespace and converts to ints."""

    def __init__(self, vocab_size: int = 10, bos: int | None = None, eod: int = 0, pad: int = 0):
        self.vocab_size = vocab_size
        self.bos = bos
        self.eod = eod
        self.pad = pad

    def tokenize(self, prompt):
        if isinstance(prompt, str):
            return [int(tok) % self.vocab_size for tok in prompt.strip().split()]
        return list(prompt)

    def detokenize(self, tokens, skip_special_tokens: bool = False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if skip_special_tokens and self.eod in tokens:
            tokens = [tok for tok in tokens if tok != self.eod]
        return " ".join(str(tok) for tok in tokens)


class DummyContext:
    """Dummy inference context."""

    def __init__(self):
        self.active_cnt = 0

    def get_active_request_count(self) -> int:
        return self.active_cnt


class DummyController:
    """Dummy inference controller."""

    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def dummy_forward(self):
        pass


class DummyEngine(DynamicInferenceEngine):
    """Dummy inference engine that only implements coordinator-related methods."""

    def __init__(self):
        self.waiting_request_ids = deque()
        self.requests: Dict[int, RequestEntry] = {}
        self.suspend_signal = False
        self.is_suspended = False
        self._loop = get_asyncio_loop()
        self.context = DummyContext()
        self.controller = DummyController()
        self.running = asyncio.Event()
        self.paused = asyncio.Event()
        self.stopped = asyncio.Event()
        self.pending_microbatch = deque()
        self.received_pause: bool = False
        self.received_stop: bool = False
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.rank = torch.distributed.get_rank()

    def add_request(
        self, request_id: int, prompt: str, sampling_params: Optional[SamplingParams] = None
    ) -> asyncio.Future[DynamicInferenceRequestRecord]:
        self.requests[request_id] = RequestEntry(
            record=DynamicInferenceRequestRecord.from_request(
                DynamicInferenceRequest(
                    prompt=prompt,
                    request_id=request_id,
                    sampling_params=sampling_params,
                    status=Status.WAITING_IN_QUEUE,
                )
            ),
            future=self._loop.create_future(),
        )
        self.waiting_request_ids.append(request_id)
        return self.requests[request_id].future

    async def async_step(self, *, verbose: Optional[bool] = False) -> Dict:
        finished_request_records = []
        to_remove = []
        for request_id, entry in self.requests.items():
            request = entry.record[-1]
            if request.status == Status.ACTIVE_AND_GENERATING_TOKENS:
                request.sampling_params.num_tokens_to_generate -= 1
                if request.sampling_params.num_tokens_to_generate > 0:
                    continue
                request.status = Status.COMPLETED
                self.context.active_cnt -= 1
                finished_request_records.append(entry.record)
                entry.future.set_result(entry.record)
                to_remove.append(request_id)
                if self.is_mp_coordinator:
                    payload = msgpack.packb(
                        [Headers.ENGINE_REPLY.value, [entry.record.serialize()]], use_bin_type=True
                    )
                    self.socket_for_receiving_requests.send(payload)

        for request_id in to_remove:
            del self.requests[request_id]

        active_request_ids = []
        while self.waiting_request_ids:
            request_id = self.waiting_request_ids.popleft()
            record = self.requests[request_id].record
            record[-1].status = Status.ACTIVE_AND_GENERATING_TOKENS
            self.context.active_cnt += 1
            active_request_ids.append(request_id)

        return {
            "active_request_ids": active_request_ids,
            "finished_request_records": finished_request_records,
            "step_time": 0.01,
            "cuda_graph_request_count": 1,
        }


@pytest.fixture
def initialize_model_parallel(request, monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    tp, pp, ep = getattr(request, "param", (1, 1, 1))
    world_size = Utils.world_size
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
    )
    dp = world_size // (tp * pp * ep)
    yield world_size, dp, tp, pp, ep
    Utils.destroy_model_parallel()


def make_coordinator_direct(
    data_parallel_size=2,
    block_size_tokens=BLOCK_SIZE,
    enable_prefix_caching=True,
    deterministic_mode=True,
):
    """Create a coordinator with mock ZMQ, for unit testing routing logic.

    Returns the coordinator instance with fake rank identities.
    """
    coordinator = object.__new__(DataParallelInferenceCoordinator)
    coordinator.tokenizer = DummyTokenizer()
    coordinator.data_parallel_size = data_parallel_size
    coordinator.block_size_tokens = block_size_tokens
    coordinator.enable_prefix_caching = enable_prefix_caching
    coordinator.prefix_caching_coordinator_policy = PrefixCachingCoordinatorPolicy.LONGEST_PREFIX

    # Create fake rank identities.
    coordinator.identities_of_data_parallel_ranks = deque(
        [f"rank_{i}".encode() for i in range(data_parallel_size)]
    )
    if deterministic_mode:
        coordinator.identities_of_data_parallel_ranks = deque(
            sorted(coordinator.identities_of_data_parallel_ranks)
        )
    coordinator.data_parallel_rank_iterator = itertools.cycle(
        coordinator.identities_of_data_parallel_ranks
    )

    coordinator.hash_to_rank_info = {}
    coordinator._round_robin_idx = 0
    coordinator._assignment_counter = 0
    coordinator._rank_pending_count = {}

    return coordinator


# ============================================================================
# Test classes
# ============================================================================


class TestCoordinatorHashComputation:
    """Test that the coordinator computes correct block hashes from prompts."""

    def test_hash_from_token_list(self):
        """Hashes from a list of token IDs match compute_block_hashes_batched."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        expected = compute_block_hashes_batched(torch.tensor(tokens, dtype=torch.int64), BLOCK_SIZE)
        assert hashes == expected
        assert len(hashes) == 2  # 8 tokens / block_size 4 = 2 blocks

    def test_hash_from_string_prompt(self):
        """Hashes from a string prompt match hashes from tokenized form."""
        coordinator = make_coordinator_direct()
        prompt = "1 2 3 4 5 6 7 8"
        hashes_from_str = coordinator.compute_request_hashes(prompt)

        # DummyTokenizer tokenizes "1 2 3 4 5 6 7 8" -> [1, 2, 3, 4, 5, 6, 7, 8]
        hashes_from_list = coordinator.compute_request_hashes([1, 2, 3, 4, 5, 6, 7, 8])
        assert hashes_from_str == hashes_from_list

    def test_hash_empty_when_disabled(self):
        """Returns empty list when prefix caching is disabled."""
        coordinator = make_coordinator_direct(enable_prefix_caching=False)
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])
        assert hashes == []

    def test_hash_empty_when_no_block_size(self):
        """Returns empty list when block_size_tokens is None."""
        coordinator = make_coordinator_direct(block_size_tokens=None)
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])
        assert hashes == []

    def test_hash_partial_block_ignored(self):
        """Tokens that don't fill a complete block produce no hash."""
        coordinator = make_coordinator_direct()
        hashes = coordinator.compute_request_hashes([1, 2, 3])
        assert hashes == []

    def test_hash_deterministic(self):
        """Same tokens always produce the same hashes."""
        coordinator = make_coordinator_direct()
        tokens = [10, 20, 30, 40, 50, 60, 70, 80]
        h1 = coordinator.compute_request_hashes(tokens)
        h2 = coordinator.compute_request_hashes(tokens)
        assert h1 == h2

    def test_hash_parent_chaining(self):
        """Different prefixes produce different hashes even for same block tokens."""
        coordinator = make_coordinator_direct()
        # Two prompts share tokens [5,6,7,8] in block 2, but differ in block 1.
        h1 = coordinator.compute_request_hashes([1, 2, 3, 4, 5, 6, 7, 8])
        h2 = coordinator.compute_request_hashes([9, 8, 7, 6, 5, 6, 7, 8])

        # Block 1 hashes differ.
        assert h1[0] != h2[0]
        # Block 2 hashes also differ due to parent chaining.
        assert h1[1] != h2[1]


class TestCoordinatorPrefixRouting:
    """Test routing decisions based on prefix cache affinity."""

    def test_no_match_uses_round_robin(self):
        """When no rank has matching hashes, falls back to round-robin."""
        coordinator = make_coordinator_direct()
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])

        rank1 = coordinator.get_best_data_parallel_rank(hashes)
        rank2 = coordinator.get_best_data_parallel_rank(hashes)
        # Round-robin cycles through ranks.
        assert rank1 != rank2

    def test_routes_to_rank_with_longest_match(self):
        """Request is routed to the rank with the longest consecutive prefix match."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
        hashes = coordinator.compute_request_hashes(tokens)
        assert len(hashes) == 3  # 12 tokens / 4 = 3 blocks

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # rank_0 has first block only.
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_0] = 1

        # rank_1 has first two blocks.
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_1] = 1
        coordinator.hash_to_rank_info.setdefault(hashes[1], {})[rank_1] = 1

        selected = coordinator.get_best_data_parallel_rank(hashes)
        assert selected == rank_1

    def test_recency_tiebreaker(self):
        """When two ranks tie on match length, prefer the more recent one."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # Both ranks have both blocks, but rank_1 has higher timestamps.
        for h in hashes:
            coordinator.hash_to_rank_info.setdefault(h, {})[rank_0] = 1
            coordinator.hash_to_rank_info.setdefault(h, {})[rank_1] = 5

        selected = coordinator.get_best_data_parallel_rank(hashes)
        assert selected == rank_1

    def test_empty_hashes_uses_round_robin(self):
        """Empty hash list falls back to round-robin."""
        coordinator = make_coordinator_direct()
        rank1 = coordinator.get_best_data_parallel_rank([])
        rank2 = coordinator.get_best_data_parallel_rank([])
        assert rank1 != rank2

    def test_disabled_prefix_caching_uses_round_robin(self):
        """With prefix caching disabled, always uses round-robin."""
        coordinator = make_coordinator_direct(enable_prefix_caching=False)
        rank1 = coordinator.get_best_data_parallel_rank([1, 2, 3])
        rank2 = coordinator.get_best_data_parallel_rank([1, 2, 3])
        assert rank1 != rank2


class TestCoordinatorShadowState:
    """Test that shadow state (rank_cached_hashes, timestamps) is updated correctly."""

    def test_update_rank_hashes_adds_to_set(self):
        """_update_rank_hashes adds hashes to the rank's set."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]

        coordinator._update_rank_hashes(rank_0, [100, 200, 300])
        assert all(rank_0 in coordinator.hash_to_rank_info[h] for h in [100, 200, 300])

    def test_update_rank_hashes_increments_counter(self):
        """Each call to _update_rank_hashes increments the assignment counter."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]

        assert coordinator._assignment_counter == 0
        coordinator._update_rank_hashes(rank_0, [100])
        assert coordinator._assignment_counter == 1
        coordinator._update_rank_hashes(rank_0, [200])
        assert coordinator._assignment_counter == 2

    def test_timestamps_updated_on_reassignment(self):
        """Re-assigning a hash to the same rank updates its timestamp."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]

        coordinator._update_rank_hashes(rank_0, [100])
        ts1 = coordinator.hash_to_rank_info[100][rank_0]

        coordinator._update_rank_hashes(rank_0, [100])
        ts2 = coordinator.hash_to_rank_info[100][rank_0]

        assert ts2 > ts1

    def test_multiple_requests_accumulate_hashes(self):
        """Multiple requests to the same rank accumulate their hashes."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]

        coordinator._update_rank_hashes(rank_0, [10, 20])
        coordinator._update_rank_hashes(rank_0, [30, 40])
        assert all(rank_0 in coordinator.hash_to_rank_info[h] for h in [10, 20, 30, 40])

    def test_hash_can_appear_in_multiple_ranks(self):
        """The same hash can be owned by multiple ranks."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        coordinator._update_rank_hashes(rank_0, [100])
        coordinator._update_rank_hashes(rank_1, [100])

        assert rank_0 in coordinator.hash_to_rank_info[100]
        assert rank_1 in coordinator.hash_to_rank_info[100]

    def test_routing_then_state_update_flow(self):
        """Full flow: compute hashes, route, update state, then re-route to same rank."""
        coordinator = make_coordinator_direct()

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        # First request: no matches, round-robin.
        rank = coordinator.get_best_data_parallel_rank(hashes)
        coordinator._update_rank_hashes(rank, hashes)

        # Second request with same tokens: should go to same rank.
        rank2 = coordinator.get_best_data_parallel_rank(hashes)
        assert rank2 == rank


@pytest.mark.skipif(ZMQ_FLAKY_SHUTDOWN, reason="ZMQ shutdown is flaky")
class TestCoordinatorEndToEnd:
    """End-to-end test with real ZMQ sockets and DummyEngines."""

    async def run_coordinator_test(
        self, requests, block_size_tokens=BLOCK_SIZE, enable_prefix_caching=True
    ):
        """Submit requests through a real coordinator and return results."""
        engine = DummyEngine()

        dp_addr = await engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
        )

        try:
            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                client.start()

                futures = [
                    client.add_request(prompt=prompt, sampling_params=params)
                    for prompt, params in requests
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=10.0)

                for record in results:
                    assert record[-1].status == Status.COMPLETED
        finally:
            if torch.distributed.get_rank() == 0:
                await asyncio.wait_for(client.stop_engines(), timeout=10.0)
                client.stop()
            try:
                await asyncio.wait_for(engine.engine_loop_task, timeout=30.0)
            except asyncio.TimeoutError:
                engine.engine_loop_task.cancel()

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required")
    @pytest.mark.asyncio
    async def test_shared_prefix_requests(self, initialize_model_parallel):
        """Requests with shared prefixes complete successfully through the coordinator."""
        requests = [
            ("1 2 3 4 5 6 7 8", SamplingParams(num_tokens_to_generate=2)),
            ("1 2 3 4 9 8 7 6", SamplingParams(num_tokens_to_generate=2)),
            ("1 2 3 4 5 6 7 8", SamplingParams(num_tokens_to_generate=2)),
        ]
        await self.run_coordinator_test(requests)


def make_first_prefix_block_coordinator(**kwargs):
    """Create a coordinator configured with FIRST_PREFIX_BLOCK policy."""
    coordinator = make_coordinator_direct(**kwargs)
    coordinator.prefix_caching_coordinator_policy = (
        PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
    )
    return coordinator


class TestFirstPrefixBlockRouting:
    """Test routing decisions using the FIRST_PREFIX_BLOCK policy."""

    def test_first_block_match_routes_to_rank(self):
        """Request is routed to the rank that has the first block cached."""
        coordinator = make_first_prefix_block_coordinator()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # Only rank_1 has the first block.
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_1] = 1

        selected = coordinator.get_best_data_parallel_rank(hashes[:1])
        assert selected == rank_1

    def test_first_block_ignores_longer_match(self):
        """Rank with more blocks cached is not preferred; only first block matters."""
        coordinator = make_first_prefix_block_coordinator()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
        hashes = coordinator.compute_request_hashes(tokens)
        assert len(hashes) == 3

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # rank_0 has first block only, with higher timestamp.
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_0] = 10

        # rank_1 has all three blocks, but lower timestamp on first block.
        for h in hashes:
            coordinator.hash_to_rank_info.setdefault(h, {})[rank_1] = 1

        # rank_0 wins because it has higher recency on the first block.
        # Caller truncates to [:1] before calling get_best_data_parallel_rank.
        selected = coordinator.get_best_data_parallel_rank(hashes[:1])
        assert selected == rank_0

    def test_first_block_recency_tiebreaker(self):
        """When multiple ranks have the first block, higher timestamp wins."""
        coordinator = make_first_prefix_block_coordinator()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # Both ranks have the first block.
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_0] = 3
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_1] = 7

        selected = coordinator.get_best_data_parallel_rank(hashes[:1])
        assert selected == rank_1

    def test_no_first_block_match_uses_round_robin(self):
        """Falls back to round-robin when no rank has the first block."""
        coordinator = make_first_prefix_block_coordinator()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]

        # rank_0 has block 1 (second block), but not block 0.
        coordinator.hash_to_rank_info.setdefault(hashes[1], {})[rank_0] = 1

        # No rank has the first block, so round-robin.
        r1 = coordinator.get_best_data_parallel_rank(hashes[:1])
        r2 = coordinator.get_best_data_parallel_rank(hashes[:1])
        assert r1 != r2

    def test_first_block_policy_with_single_block_prompt(self):
        """Works correctly with a prompt that has only one block."""
        coordinator = make_first_prefix_block_coordinator()
        tokens = [1, 2, 3, 4]
        hashes = coordinator.compute_request_hashes(tokens)
        assert len(hashes) == 1

        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_1] = 1

        selected = coordinator.get_best_data_parallel_rank(hashes[:1])
        assert selected == rank_1


class TestLoadAwarePrefixRouting:
    """Test that prefix routing spreads load across ranks with the same prefix."""

    def test_spreads_across_ranks_with_same_prefix(self):
        """When three ranks all cache the same prefix, requests spread by load."""
        coordinator = make_coordinator_direct(data_parallel_size=3)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        rank_2 = coordinator.identities_of_data_parallel_ranks[2]

        # All three ranks have both blocks cached with the same timestamp.
        for h in hashes:
            coordinator.hash_to_rank_info.setdefault(h, {})[rank_0] = 1
            coordinator.hash_to_rank_info.setdefault(h, {})[rank_1] = 1
            coordinator.hash_to_rank_info.setdefault(h, {})[rank_2] = 1

        # Simulate sending 6 requests. With load-aware routing, they should
        # spread across ranks rather than all going to one.
        assigned_ranks = []
        for _ in range(6):
            rank = coordinator.get_best_data_parallel_rank(hashes)
            coordinator._rank_pending_count[rank] = (
                coordinator._rank_pending_count.get(rank, 0) + 1
            )
            assigned_ranks.append(rank)

        # Each rank should get exactly 2 of the 6 requests.
        from collections import Counter

        counts = Counter(assigned_ranks)
        assert counts[rank_0] == 2
        assert counts[rank_1] == 2
        assert counts[rank_2] == 2

    def test_load_overrides_recency(self):
        """A rank with a higher timestamp but more pending requests is not preferred."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # Both ranks have the prefix. rank_1 has a higher (more recent) timestamp.
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_0] = 1
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_1] = 10

        # But rank_1 already has 5 pending requests while rank_0 has 0.
        coordinator._rank_pending_count[rank_1] = 5

        selected = coordinator.get_best_data_parallel_rank(hashes)
        assert selected == rank_0

    def test_pending_count_decremented_on_completion(self):
        """Completing a request frees capacity on the assigned rank."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_0] = 1
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_1] = 1

        # Simulate assigning a request to rank_0.
        coordinator._rank_pending_count[rank_0] = 1
        coordinator.request_id_to_rank = {42: rank_0}

        # Simulate completion: decrement pending count.
        assigned_rank = coordinator.request_id_to_rank.pop(42, None)
        if assigned_rank is not None and assigned_rank in coordinator._rank_pending_count:
            coordinator._rank_pending_count[assigned_rank] = max(
                0, coordinator._rank_pending_count[assigned_rank] - 1
            )

        assert coordinator._rank_pending_count[rank_0] == 0

    def test_equal_load_falls_back_to_recency(self):
        """With equal pending counts, the most recent timestamp wins."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # Equal pending counts (both 0), but rank_0 has higher timestamp.
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_0] = 10
        coordinator.hash_to_rank_info.setdefault(hashes[0], {})[rank_1] = 1

        selected = coordinator.get_best_data_parallel_rank(hashes)
        assert selected == rank_0
