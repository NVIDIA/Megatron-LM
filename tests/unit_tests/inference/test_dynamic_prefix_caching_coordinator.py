# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for prefix-cache-aware coordinator routing.

Validates that the DataParallelInferenceCoordinator correctly computes block
hashes from prompts, routes requests to the DP rank with the longest consecutive
prefix match, and maintains per-rank shadow state (cached hashes and timestamps).
"""

import asyncio
import itertools
import time
from collections import deque
from typing import Dict, Optional
from unittest.mock import MagicMock

import msgpack
import numpy as np
import pytest
import torch

from megatron.core.inference.config import (
    MediaCacheCoordinatorPolicy,
    PrefixCachingCoordinatorPolicy,
)
from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.data_parallel_inference_coordinator.handlers import (
    handle_engine_reply,
    handle_submit_request,
)
from megatron.core.inference.engines.dynamic_engine import (
    DynamicInferenceEngine,
    RequestEntry,
    _engine_reply_frames,
)
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


def _set_hash_rank(coordinator, h, rank_identity, timestamp):
    """Test helper: set a hash→rank timestamp in the coordinator's dict."""
    rank_idx = coordinator.identity_to_rank_index[rank_identity]
    coordinator._hash_table.setdefault(h, {})[rank_idx] = timestamp


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
                    self.socket_for_receiving_requests.send_multipart(
                        _engine_reply_frames([entry.record.serialize()])
                    )

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
    prefix_caching_routing_alpha=0.5,
    prefix_cache_ttl_seconds=300.0,
    media_policy=MediaCacheCoordinatorPolicy.AFFINITY,
    vision_embedding_cache_enabled=True,
    max_requests=10,
):
    """Create a coordinator with mock ZMQ, for unit testing routing logic.

    Thin wrapper around the shared helper in coordinator_test_utils.py that
    supplies a DummyTokenizer and this module's BLOCK_SIZE default.
    """
    from tests.unit_tests.inference.coordinator_test_utils import (
        make_coordinator_direct as _make_coordinator,
    )

    return _make_coordinator(
        data_parallel_size=data_parallel_size,
        block_size_tokens=block_size_tokens,
        enable_prefix_caching=enable_prefix_caching,
        deterministic_mode=deterministic_mode,
        prefix_caching_routing_alpha=prefix_caching_routing_alpha,
        prefix_cache_ttl_seconds=prefix_cache_ttl_seconds,
        media_policy=media_policy,
        vision_embedding_cache_enabled=vision_embedding_cache_enabled,
        max_requests=max_requests,
        tokenizer=DummyTokenizer(),
    )


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


class TestSubmitDoesNotDecodePrompt:
    """The coordinator must not decode the prompt when no routing needs it.

    Decoding is O(prompt length) on the coordinator's single serial loop, so
    skipping it is the point of carrying the prompt in its own frame. Each test
    sends a prompt frame that is deliberately *not* valid msgpack: if the
    handler tried to decode it the call would raise, so completing cleanly is
    proof the bytes were forwarded untouched.
    """

    UNDECODABLE_PROMPT = b"\xc1not-valid-msgpack"
    UNDECODABLE_MEDIA = b"\xc1not-valid-msgpack-either"

    def _submit(self, coordinator, block_hashes=None, media_meta=None):
        """Drive handle_submit_request once and return the frames sent onward."""
        coordinator.known_clients = {b"client-A"}
        coordinator.next_request_id = 0
        coordinator.request_id_to_client_id = {}
        coordinator.request_id_to_client_request_id = {}
        coordinator.client_request_to_request_id = {}
        coordinator.request_id_to_rank = {}
        coordinator.schedule_records = None
        coordinator.router_socket = MagicMock()

        metadata = [Headers.SUBMIT_REQUEST.value, 7, {"temperature": 1.0}, media_meta]
        bodies = [
            self.UNDECODABLE_PROMPT,
            msgpack.packb(block_hashes, use_bin_type=True),
            self.UNDECODABLE_MEDIA,
        ]
        handle_submit_request(coordinator, b"client-A", metadata, bodies)
        return coordinator.router_socket.send_multipart.call_args.args[0]

    def test_load_balanced_forwards_prompt_verbatim(self):
        """LOAD_BALANCED ignores hashes, so the prompt is never decoded."""
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.prefix_caching_coordinator_policy = PrefixCachingCoordinatorPolicy.LOAD_BALANCED
        _identity, _metadata, prompt_frame, _media = self._submit(coordinator, block_hashes=[])
        assert prompt_frame is self.UNDECODABLE_PROMPT

    def test_disabled_prefix_caching_forwards_prompt_verbatim(self):
        """With prefix caching off there are no hashes to compute either."""
        coordinator = make_coordinator_direct(data_parallel_size=2, enable_prefix_caching=False)
        _identity, _metadata, prompt_frame, _media = self._submit(coordinator, block_hashes=[])
        assert prompt_frame is self.UNDECODABLE_PROMPT

    def test_prefix_routing_uses_frontend_hashes_without_decoding_prompt(self):
        """Prefix-affinity routing reads the frontend's hashes, not the prompt.

        This is the case the split exists for: the coordinator has to route on
        prefix affinity *and* still never look at the prompt. It only holds
        because the frontend hashed the tokens it already had.
        """
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.prefix_caching_coordinator_policy = (
            PrefixCachingCoordinatorPolicy.LONGEST_PREFIX
        )
        _identity, _metadata, prompt_frame, _media = self._submit(
            coordinator, block_hashes=[11, 22]
        )
        assert prompt_frame is self.UNDECODABLE_PROMPT

    def test_frontend_hashes_are_recorded_against_the_chosen_rank(self):
        """The supplied hashes drive affinity, so they must reach the rank table."""
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.prefix_caching_coordinator_policy = (
            PrefixCachingCoordinatorPolicy.LONGEST_PREFIX
        )
        identity, _metadata, _prompt, _media = self._submit(coordinator, block_hashes=[11, 22])
        # A second request with the same prefix must now land on the same rank.
        again, _m, _p, _md = self._submit(coordinator, block_hashes=[11, 22])
        assert again == identity

    def test_unhashed_prompt_falls_back_to_the_coordinator(self):
        """A client that could not hash sends None, and the coordinator hashes.

        That happens for a string prompt, which needs a tokenizer the client does
        not have. It is the only case that still decodes the prompt here, and it
        is distinct from an empty list, which means the client hashed and the
        prompt was shorter than one block.
        """
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.prefix_caching_coordinator_policy = (
            PrefixCachingCoordinatorPolicy.LONGEST_PREFIX
        )
        coordinator.known_clients = {b"client-A"}
        coordinator.next_request_id = 0
        coordinator.request_id_to_client_id = {}
        coordinator.request_id_to_client_request_id = {}
        coordinator.client_request_to_request_id = {}
        coordinator.request_id_to_rank = {}
        coordinator.schedule_records = None
        coordinator.router_socket = MagicMock()
        coordinator.compute_request_hashes = MagicMock(return_value=[5])

        metadata = [
            Headers.SUBMIT_REQUEST.value,
            7,
            {"temperature": 1.0},
            {"media_cache_key": "img-1"},
        ]
        prompt = msgpack.packb([1, 2, 3], use_bin_type=True)
        handle_submit_request(
            coordinator,
            b"client-A",
            metadata,
            [prompt, msgpack.packb(None, use_bin_type=True), self.UNDECODABLE_MEDIA],
        )

        # Decoded here, and salted with whatever media key the metadata carried.
        coordinator.compute_request_hashes.assert_called_once_with([1, 2, 3], cache_salt="img-1")

    def test_empty_hash_list_is_not_a_fallback(self):
        """An empty list means "hashed, no complete blocks" -- do not re-hash.

        Treating it as unhashed would decode the prompt on this loop for every
        short request, which is exactly the cost this design removes.
        """
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.prefix_caching_coordinator_policy = (
            PrefixCachingCoordinatorPolicy.LONGEST_PREFIX
        )
        coordinator.compute_request_hashes = MagicMock(return_value=[5])
        _identity, _metadata, prompt_frame, _media = self._submit(coordinator, block_hashes=[])
        assert prompt_frame is self.UNDECODABLE_PROMPT
        coordinator.compute_request_hashes.assert_not_called()

    def test_media_frame_is_forwarded_without_being_decoded(self):
        """The media bytes reach the engine untouched.

        Media is the largest thing on the wire -- raw video runs to hundreds of
        megabytes -- and this loop is shared by every rank, so decoding it here
        would cost far more than the prompt decode the split already removed.
        An undecodable frame proves nothing looked at it.
        """
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.prefix_caching_coordinator_policy = (
            PrefixCachingCoordinatorPolicy.LONGEST_PREFIX
        )
        _identity, _metadata, _prompt, media_frame = self._submit(
            coordinator, block_hashes=[11, 22], media_meta={"media_cache_key": "img-1"}
        )
        assert media_frame is self.UNDECODABLE_MEDIA

    def test_media_identity_routes_without_the_media_payload(self):
        """Affinity keys on the descriptor in metadata, never on the bytes."""
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.prefix_caching_coordinator_policy = PrefixCachingCoordinatorPolicy.LOAD_BALANCED
        identity, _m, _p, _md = self._submit(
            coordinator, block_hashes=[], media_meta={"media_cache_key": "img-1"}
        )
        assert coordinator._media_cache_affinity["img-1"] == identity

    def test_metadata_frame_is_rewritten_with_server_request_id(self):
        """The client's request id is swapped for the coordinator's own."""
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.prefix_caching_coordinator_policy = PrefixCachingCoordinatorPolicy.LOAD_BALANCED
        _identity, metadata_frame, _prompt, _media = self._submit(coordinator, block_hashes=[])
        header, request_id, sampling_params, media_meta = msgpack.unpackb(metadata_frame, raw=False)
        assert header == Headers.SUBMIT_REQUEST.value
        assert request_id == 0  # server-side id, not the client's 7
        assert sampling_params == {"temperature": 1.0}
        assert media_meta is None
        assert coordinator.request_id_to_client_request_id[0] == 7


class TestCoordinatorPrefixRouting:
    """Test routing decisions based on prefix cache affinity."""

    def test_no_match_prefers_least_loaded(self):
        """When no rank has matching hashes, the rank with most free capacity wins."""
        coordinator = make_coordinator_direct()
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # rank_1 has fewer pending requests, so more free capacity.
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 5
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 1

        selected = coordinator.get_best_data_parallel_rank(hashes)
        assert selected == rank_1

    def test_routes_to_rank_with_longest_match(self):
        """Request is routed to the rank with the longest consecutive prefix match."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
        hashes = coordinator.compute_request_hashes(tokens)
        assert len(hashes) == 3  # 12 tokens / 4 = 3 blocks

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # Ensure no rank is idle so prefix-matching logic is exercised.
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 1
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 1

        # rank_0 has first block only.
        _set_hash_rank(coordinator, hashes[0], rank_0, 1)

        # rank_1 has first two blocks.
        _set_hash_rank(coordinator, hashes[0], rank_1, 1)
        _set_hash_rank(coordinator, hashes[1], rank_1, 1)

        selected = coordinator.get_best_data_parallel_rank(hashes)
        assert selected == rank_1

    def test_equal_scores_tiebreak_by_rank_index(self):
        """When two ranks have equal scores, the lower rank index wins."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # Both ranks have same pending counts, same match, and same timestamp.
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 1
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 1

        for h in hashes:
            _set_hash_rank(coordinator, h, rank_0, 1)
            _set_hash_rank(coordinator, h, rank_1, 1)

        # Equal scores → lowest rank index (rank_0) wins.
        selected = coordinator.get_best_data_parallel_rank(hashes)
        assert selected == rank_0

    def test_empty_hashes_uses_load_balanced(self):
        """Empty hash list falls back to the least-loaded rank."""
        coordinator = make_coordinator_direct()
        identities = list(coordinator.identities_of_data_parallel_ranks)
        for identity in identities:
            coordinator._pending_counts[coordinator.identity_to_rank_index[identity]] = 2
        # Make the second rank the least loaded.
        coordinator._pending_counts[coordinator.identity_to_rank_index[identities[1]]] = 0
        assert coordinator.get_best_data_parallel_rank([]) == identities[1]

    def test_disabled_prefix_caching_uses_load_balanced(self):
        """With prefix caching disabled, always routes to the least-loaded rank."""
        coordinator = make_coordinator_direct(enable_prefix_caching=False)
        identities = list(coordinator.identities_of_data_parallel_ranks)
        for identity in identities:
            coordinator._pending_counts[coordinator.identity_to_rank_index[identity]] = 2
        coordinator._pending_counts[coordinator.identity_to_rank_index[identities[1]]] = 0
        assert coordinator.get_best_data_parallel_rank([1, 2, 3]) == identities[1]


class TestMultimodalAffinityRouting:
    """Media and prompt-prefix reuse participate in one routing score."""

    def test_media_salt_partitions_prompt_affinity(self):
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        image_a = coordinator.compute_request_hashes(tokens, cache_salt="image-a")
        image_b = coordinator.compute_request_hashes(tokens, cache_salt="image-b")
        assert image_a != image_b
        assert image_a == coordinator.compute_request_hashes(tokens, cache_salt="image-a")

    def test_media_affinity_works_without_prefix_caching(self):
        coordinator = make_coordinator_direct(
            enable_prefix_caching=False, prefix_caching_routing_alpha=1.0
        )
        media_rank = coordinator._identities_list[1]
        coordinator._update_media_affinity("image-a", media_rank)
        assert coordinator.get_best_data_parallel_rank([], media_cache_key="image-a") == media_rank

    def test_removing_engine_prunes_its_media_affinity(self):
        coordinator = make_coordinator_direct()
        removed_rank, retained_rank = coordinator._identities_list
        coordinator._update_media_affinity("removed-image", removed_rank)
        coordinator._update_media_affinity("retained-image", retained_rank)

        coordinator._remove_engine(removed_rank)

        assert "removed-image" not in coordinator._media_cache_affinity
        assert coordinator._media_cache_affinity["retained-image"] == retained_rank

    def test_cold_media_falls_back_to_least_loaded_rank(self):
        coordinator = make_coordinator_direct(
            enable_prefix_caching=False, prefix_caching_routing_alpha=1.0
        )
        busy_rank, free_rank = coordinator._identities_list
        coordinator._pending_counts[coordinator.identity_to_rank_index[busy_rank]] = 5
        coordinator._pending_counts[coordinator.identity_to_rank_index[free_rank]] = 1
        assert (
            coordinator.get_best_data_parallel_rank([], media_cache_key="unseen-image") == free_rank
        )

    def test_media_affinity_is_ignored_when_vision_cache_is_disabled(self):
        coordinator = make_coordinator_direct(
            enable_prefix_caching=False,
            prefix_caching_routing_alpha=1.0,
            vision_embedding_cache_enabled=False,
        )
        load_rank, media_rank = coordinator._identities_list
        coordinator._pending_counts[coordinator.identity_to_rank_index[media_rank]] = 1
        coordinator._update_media_affinity("image-a", media_rank)
        assert coordinator.get_best_data_parallel_rank([], media_cache_key="image-a") == load_rank

    def test_prefix_load_balanced_policy_still_uses_media_affinity(self):
        coordinator = make_coordinator_direct(enable_prefix_caching=False)
        coordinator.prefix_caching_coordinator_policy = PrefixCachingCoordinatorPolicy.LOAD_BALANCED
        _, media_rank = coordinator._identities_list
        coordinator._pending_counts[coordinator.identity_to_rank_index[media_rank]] = 1
        coordinator._update_media_affinity("image-a", media_rank)
        assert coordinator.get_best_data_parallel_rank([], media_cache_key="image-a") == media_rank

    def test_both_load_balanced_policies_ignore_media_affinity(self):
        coordinator = make_coordinator_direct(
            enable_prefix_caching=False, media_policy=MediaCacheCoordinatorPolicy.LOAD_BALANCED
        )
        coordinator.prefix_caching_coordinator_policy = PrefixCachingCoordinatorPolicy.LOAD_BALANCED
        load_rank, media_rank = coordinator._identities_list
        coordinator._pending_counts[coordinator.identity_to_rank_index[media_rank]] = 1
        coordinator._update_media_affinity("image-a", media_rank)
        assert coordinator.get_best_data_parallel_rank([], media_cache_key="image-a") == load_rank

    def test_long_prefix_can_outweigh_media_hit(self):
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=1.0)
        coordinator.media_cache_routing_weight = 1.0
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4, 5, 6, 7, 8], cache_salt="image-a")
        media_rank, prefix_rank = coordinator._identities_list
        coordinator._update_media_affinity("image-a", media_rank)
        for block_hash in hashes:
            _set_hash_rank(coordinator, block_hash, prefix_rank, 1)

        assert (
            coordinator.get_best_data_parallel_rank(hashes, media_cache_key="image-a")
            == prefix_rank
        )

    def test_expensive_media_can_outweigh_long_prefix(self):
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=1.0)
        coordinator.media_cache_routing_weight = 3.0
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4, 5, 6, 7, 8], cache_salt="image-a")
        media_rank, prefix_rank = coordinator._identities_list
        coordinator._update_media_affinity("image-a", media_rank)
        for block_hash in hashes:
            _set_hash_rank(coordinator, block_hash, prefix_rank, 1)

        assert (
            coordinator.get_best_data_parallel_rank(hashes, media_cache_key="image-a") == media_rank
        )


class TestCoordinatorShadowState:
    """Test that shadow state (rank_cached_hashes, timestamps) is updated correctly."""

    def test_update_rank_hashes_adds_to_set(self):
        """_update_rank_hashes adds hashes to the rank's set."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        idx_0 = coordinator.identity_to_rank_index[rank_0]

        coordinator._update_rank_hashes(rank_0, [100, 200, 300])
        assert all(coordinator._hash_table.get(h, {}).get(idx_0, 0) > 0 for h in [100, 200, 300])

    def test_update_rank_hashes_stamps_the_current_time(self, monkeypatch):
        """Entries carry the time they were routed, which is what expiry reads.

        A monotonic clock replaced the assignment counter: ordering alone cannot
        say whether an entry is older than the TTL.
        """
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_idx = coordinator.identity_to_rank_index[rank_0]

        monkeypatch.setattr(time, "monotonic", lambda: 500.0)
        coordinator._update_rank_hashes(rank_0, [100])
        assert coordinator._hash_table[100][rank_idx] == 500.0

        monkeypatch.setattr(time, "monotonic", lambda: 700.0)
        coordinator._update_rank_hashes(rank_0, [200])
        assert coordinator._hash_table[200][rank_idx] == 700.0
        # One queue entry per (touch, hash), so expiry can sweep in order.
        assert list(coordinator._hash_expiry) == [(500.0, 100), (700.0, 200)]

    def test_timestamps_updated_on_reassignment(self):
        """Re-assigning a hash to the same rank updates its timestamp."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        idx_0 = coordinator.identity_to_rank_index[rank_0]

        coordinator._update_rank_hashes(rank_0, [100])
        ts1 = coordinator._hash_table[100][idx_0]

        coordinator._update_rank_hashes(rank_0, [100])
        ts2 = coordinator._hash_table[100][idx_0]

        assert ts2 > ts1

    def test_multiple_requests_accumulate_hashes(self):
        """Multiple requests to the same rank accumulate their hashes."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        idx_0 = coordinator.identity_to_rank_index[rank_0]

        coordinator._update_rank_hashes(rank_0, [10, 20])
        coordinator._update_rank_hashes(rank_0, [30, 40])
        assert all(coordinator._hash_table.get(h, {}).get(idx_0, 0) > 0 for h in [10, 20, 30, 40])

    def test_hash_can_appear_in_multiple_ranks(self):
        """The same hash can be owned by multiple ranks."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        idx_0 = coordinator.identity_to_rank_index[rank_0]
        idx_1 = coordinator.identity_to_rank_index[rank_1]

        coordinator._update_rank_hashes(rank_0, [100])
        coordinator._update_rank_hashes(rank_1, [100])

        assert coordinator._hash_table[100].get(idx_0, 0) > 0
        assert coordinator._hash_table[100].get(idx_1, 0) > 0

    def test_routing_then_state_update_flow(self):
        """Full flow: compute hashes, route, update state, then re-route to same rank."""
        coordinator = make_coordinator_direct()

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        # First request: no matches, routed by load (least-loaded rank).
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

        # Ensure no rank is idle so prefix-matching logic is exercised.
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 1
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 1

        # Only rank_1 has the first block.
        _set_hash_rank(coordinator, hashes[0], rank_1, 1)

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

        # Ensure no rank is idle so prefix-matching logic is exercised.
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 1
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 1

        # rank_0 has first block only, with higher timestamp.
        _set_hash_rank(coordinator, hashes[0], rank_0, 10)

        # rank_1 has all three blocks, but lower timestamp on first block.
        for h in hashes:
            _set_hash_rank(coordinator, h, rank_1, 1)

        # rank_0 wins because it has higher recency on the first block.
        # Caller truncates to [:1] before calling get_best_data_parallel_rank.
        selected = coordinator.get_best_data_parallel_rank(hashes[:1])
        assert selected == rank_0

    def test_first_block_equal_match_tiebreaks_by_rank_index(self):
        """When multiple ranks have the first block with equal load, lowest index wins."""
        coordinator = make_first_prefix_block_coordinator()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 1
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 1

        # Both ranks have the first block with the same timestamp.
        _set_hash_rank(coordinator, hashes[0], rank_0, 3)
        _set_hash_rank(coordinator, hashes[0], rank_1, 3)

        # Equal scores → lowest rank index wins.
        selected = coordinator.get_best_data_parallel_rank(hashes[:1])
        assert selected == rank_0

    def test_no_first_block_match_prefers_least_loaded(self):
        """When no rank has the first block, the least loaded rank wins."""
        coordinator = make_first_prefix_block_coordinator()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # rank_0 has block 1 (second block), but not block 0.
        _set_hash_rank(coordinator, hashes[1], rank_0, 1)

        # rank_1 has fewer pending requests.
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 5
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 1

        # No rank has the first block → load determines winner.
        selected = coordinator.get_best_data_parallel_rank(hashes[:1])
        assert selected == rank_1

    def test_first_block_policy_with_single_block_prompt(self):
        """Works correctly with a prompt that has only one block."""
        coordinator = make_first_prefix_block_coordinator()
        tokens = [1, 2, 3, 4]
        hashes = coordinator.compute_request_hashes(tokens)
        assert len(hashes) == 1

        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # Ensure no rank is idle so prefix-matching logic is exercised.
        for identity in coordinator.identities_of_data_parallel_ranks:
            coordinator._pending_counts[coordinator.identity_to_rank_index[identity]] = 1

        _set_hash_rank(coordinator, hashes[0], rank_1, 1)

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
            _set_hash_rank(coordinator, h, rank_0, 1)
            _set_hash_rank(coordinator, h, rank_1, 1)
            _set_hash_rank(coordinator, h, rank_2, 1)

        # Simulate sending 6 requests. With load-aware routing, they should
        # spread across ranks rather than all going to one.
        assigned_ranks = []
        for _ in range(6):
            rank = coordinator.get_best_data_parallel_rank(hashes)
            coordinator._pending_counts[coordinator.identity_to_rank_index[rank]] += 1
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
        _set_hash_rank(coordinator, hashes[0], rank_0, 1)
        _set_hash_rank(coordinator, hashes[0], rank_1, 10)

        # But rank_1 already has 5 pending requests while rank_0 has only 1.
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 1
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 5

        selected = coordinator.get_best_data_parallel_rank(hashes)
        assert selected == rank_0

    def test_pending_count_decremented_on_completion(self):
        """Completing a request frees capacity on the assigned rank."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        _set_hash_rank(coordinator, hashes[0], rank_0, 1)
        _set_hash_rank(coordinator, hashes[0], rank_1, 1)

        # Simulate assigning a request to rank_0.
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 1
        coordinator.request_id_to_rank = {42: rank_0}

        # Simulate completion: decrement pending count.
        assigned_rank = coordinator.request_id_to_rank.pop(42, None)
        if assigned_rank is not None:
            idx = coordinator.identity_to_rank_index.get(assigned_rank)
            if idx is not None:
                coordinator._pending_counts[idx] = max(0, coordinator._pending_counts[idx] - 1)

        assert coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] == 0

    def test_equal_load_tiebreaks_by_rank_index(self):
        """With equal pending counts and match, lowest rank index wins."""
        coordinator = make_coordinator_direct()
        tokens = [1, 2, 3, 4]
        hashes = coordinator.compute_request_hashes(tokens)

        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]

        # Equal pending counts, both have the prefix cached.
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 1
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 1
        _set_hash_rank(coordinator, hashes[0], rank_0, 10)
        _set_hash_rank(coordinator, hashes[0], rank_1, 1)

        # Equal scores → lowest rank index (rank_0) wins.
        selected = coordinator.get_best_data_parallel_rank(hashes)
        assert selected == rank_0


class TestScoringFunctionRouting:
    """The scoring function: score = cache_score - alpha * relative_load."""

    def test_zero_alpha_is_pure_prefix_affinity(self):
        """alpha=0 drops the load term, so the prefix holder wins however loaded."""
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=0.0, max_requests=10)
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        _set_hash_rank(coordinator, hashes[0], rank_0, 1)
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 9
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 0
        assert coordinator.get_best_data_parallel_rank(hashes) == rank_0

    def test_mild_imbalance_does_not_overturn_a_cache_hit(self):
        """One extra in-flight request must not cost a whole prefill.

        At the default alpha a full hit stays decisive until the fleet is
        genuinely uneven; alpha=1 would put this exactly on a knife edge.
        """
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=0.5, max_requests=10)
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        _set_hash_rank(coordinator, hashes[0], rank_0, 1)
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 1
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 0
        # mean 0.5 (floored divisor 1.0) -> relative_load [+0.5, -0.5]
        # scores = [1 - 0.5*0.5, 0 + 0.5*0.5] = [0.75, 0.25]
        assert coordinator.get_best_data_parallel_rank(hashes) == rank_0

    def test_saturated_rank_loses_to_an_idle_one(self):
        """The drain at the end of a batch: spread rather than strand work."""
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=0.5, max_requests=10)
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        _set_hash_rank(coordinator, hashes[0], rank_0, 1)
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 10
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 0
        # mean 5 -> relative_load [+1, -1]; scores tie at 0.5, tiebreak to least loaded.
        assert coordinator.get_best_data_parallel_rank(hashes) == rank_1

    def test_balanced_fleet_ignores_load_entirely(self):
        """Equal load is a zero penalty for everyone, whatever alpha is.

        This is what measuring against the fleet mean buys: the term expresses
        *imbalance*, so with none it cannot outvote a cache hit.
        """
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=5.0, max_requests=10)
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        _set_hash_rank(coordinator, hashes[0], rank_0, 1)
        for rank in (rank_0, rank_1):
            coordinator._pending_counts[coordinator.identity_to_rank_index[rank]] = 8
        assert coordinator.get_best_data_parallel_rank(hashes) == rank_0

    def test_deeper_prefix_outscores_a_shallower_one(self):
        """Depth is graded, not awarded only to the deepest holder."""
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=0.0, max_requests=10)
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4, 5, 6, 7, 8])
        assert len(hashes) >= 2
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        for h in hashes:
            _set_hash_rank(coordinator, h, rank_0, 1)
        _set_hash_rank(coordinator, hashes[0], rank_1, 1)
        assert coordinator.get_best_data_parallel_rank(hashes) == rank_0

    def test_deep_block_without_its_prefix_is_not_credited(self):
        """Depth counts forward from block 0, so an evicted prefix earns nothing.

        The hashes chain, so a later block whose prefix is gone cannot be reused
        and must not be scored as if it could.
        """
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=0.0, max_requests=10)
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4, 5, 6, 7, 8])
        assert len(hashes) >= 2
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        _set_hash_rank(coordinator, hashes[-1], rank_1, 1)
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 0
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 3
        # No genuine hit anywhere, so this falls back to load balancing.
        assert coordinator.get_best_data_parallel_rank(hashes) == rank_0

    def test_ties_break_towards_the_least_loaded_rank(self):
        """Equal affinity, so only the tiebreak decides."""
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=0.0, max_requests=10)
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        for rank in (rank_0, rank_1):
            _set_hash_rank(coordinator, hashes[0], rank, 1)
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 5
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 1
        assert coordinator.get_best_data_parallel_rank(hashes) == rank_1


class TestPrefixCacheTTL:
    """Entries the coordinator has not routed for the TTL stop counting as hits.

    The coordinator only ever observes blocks being routed, never blocks being
    evicted, so without expiry its view of each engine's cache is monotonically
    optimistic and it keeps routing for prefixes that are long gone.
    """

    def _coord(self, ttl=300.0):
        coordinator = make_coordinator_direct(
            prefix_caching_routing_alpha=0.0, prefix_cache_ttl_seconds=ttl
        )
        coordinator.request_id_to_hashes = {}
        return coordinator

    def test_untouched_entries_are_dropped_after_the_ttl(self, monkeypatch):
        coordinator = self._coord(ttl=300.0)
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])

        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        coordinator._update_rank_hashes(rank_0, hashes)
        assert hashes[0] in coordinator._hash_table

        # Still inside the TTL: another request touching different blocks must not
        # evict this one.
        monkeypatch.setattr(time, "monotonic", lambda: 1200.0)
        coordinator._update_rank_hashes(rank_0, coordinator.compute_request_hashes([9, 9, 9, 9]))
        assert hashes[0] in coordinator._hash_table

        # Past it now.
        monkeypatch.setattr(time, "monotonic", lambda: 1400.0)
        coordinator._update_rank_hashes(rank_0, coordinator.compute_request_hashes([8, 8, 8, 8]))
        assert hashes[0] not in coordinator._hash_table

    def test_rerouting_a_block_refreshes_it(self, monkeypatch):
        """A block still in use must survive its original queue entry."""
        coordinator = self._coord(ttl=300.0)
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])

        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        coordinator._update_rank_hashes(rank_0, hashes)
        monkeypatch.setattr(time, "monotonic", lambda: 1250.0)
        coordinator._update_rank_hashes(rank_0, hashes)

        # The 1000.0 queue entry expires here, but the block was re-routed at
        # 1250.0 and carries the newer timestamp, so it is left alone.
        monkeypatch.setattr(time, "monotonic", lambda: 1400.0)
        coordinator._update_rank_hashes(rank_0, coordinator.compute_request_hashes([7, 7, 7, 7]))
        assert hashes[0] in coordinator._hash_table

    def test_expired_entries_stop_winning_routing(self, monkeypatch):
        """The point of expiring: a cold rank must stop attracting requests."""
        coordinator = self._coord(ttl=300.0)
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        hashes = coordinator.compute_request_hashes([1, 2, 3, 4])

        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        coordinator._update_rank_hashes(rank_0, hashes)
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_0]] = 2
        coordinator._pending_counts[coordinator.identity_to_rank_index[rank_1]] = 0
        assert coordinator.get_best_data_parallel_rank(hashes) == rank_0

        monkeypatch.setattr(time, "monotonic", lambda: 1400.0)
        coordinator._update_rank_hashes(rank_1, coordinator.compute_request_hashes([5, 5, 5, 5]))
        # rank_0's claim has aged out, so this falls back to load balancing.
        assert coordinator.get_best_data_parallel_rank(hashes) == rank_1

    def test_expiry_survives_an_engine_being_removed(self, monkeypatch):
        """Removing an engine renumbers ranks; queued expiry must not follow stale indices.

        The queue is keyed on the hash rather than the rank index for exactly this
        reason: after a renumber, an index queued earlier names a different rank.
        """
        coordinator = self._coord(ttl=300.0)
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        rank_1 = coordinator.identities_of_data_parallel_ranks[1]
        kept = coordinator.compute_request_hashes([1, 2, 3, 4])

        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        coordinator._update_rank_hashes(rank_0, kept)
        monkeypatch.setattr(time, "monotonic", lambda: 1100.0)
        coordinator._update_rank_hashes(rank_1, kept)

        coordinator._remove_engine(rank_0)

        # rank_1's entry, now at a shifted index, must still be here and must not
        # be evicted by the queue entry that was made for rank_0.
        monkeypatch.setattr(time, "monotonic", lambda: 1350.0)
        coordinator._update_rank_hashes(rank_1, coordinator.compute_request_hashes([6, 6, 6, 6]))
        assert kept[0] in coordinator._hash_table


class TestMalformedSubmissionsAreDropped:
    """A badly framed submission must cost its sender, not the coordinator.

    The event loop serves every rank and every client, so an exception raised out
    of a handler takes the whole coordinator down with it.
    """

    def _coordinator(self):
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.known_clients = {b"client-A"}
        coordinator.next_request_id = 0
        coordinator.request_id_to_client_id = {}
        coordinator.request_id_to_client_request_id = {}
        coordinator.client_request_to_request_id = {}
        coordinator.request_id_to_rank = {}
        coordinator.schedule_records = None
        coordinator.router_socket = MagicMock()
        return coordinator

    def test_submission_missing_the_hash_frame_is_dropped(self):
        """Two frames was the old wire format; it must not raise IndexError."""
        coordinator = self._coordinator()
        metadata = [Headers.SUBMIT_REQUEST.value, 7, {"temperature": 1.0}, None]
        handle_submit_request(coordinator, b"client-A", metadata, [b"\xc0"])
        coordinator.router_socket.send_multipart.assert_not_called()
        assert coordinator.next_request_id == 0

    def test_submission_missing_the_media_frame_is_dropped(self):
        """Three frames was the previous wire format; it must not raise either."""
        coordinator = self._coordinator()
        metadata = [Headers.SUBMIT_REQUEST.value, 7, {"temperature": 1.0}, None]
        handle_submit_request(coordinator, b"client-A", metadata, [b"\xc0", b"\xc0"])
        coordinator.router_socket.send_multipart.assert_not_called()
        assert coordinator.next_request_id == 0

    def test_submission_with_short_metadata_is_dropped(self):
        """Too few metadata fields must not raise a ValueError on unpack."""
        coordinator = self._coordinator()
        metadata = [Headers.SUBMIT_REQUEST.value, 7, {"temperature": 1.0}]
        handle_submit_request(coordinator, b"client-A", metadata, [b"\xc0", b"\xc0", b"\xc0"])
        coordinator.router_socket.send_multipart.assert_not_called()
        assert coordinator.next_request_id == 0


class TestEngineReplyDetokenization:
    """The coordinator detokenizes a reply only when its client asked it to."""

    def _coordinator(self):
        coordinator = make_coordinator_direct(data_parallel_size=2)
        coordinator.request_id_to_client_id = {5: b"client-A"}
        coordinator.request_id_to_client_request_id = {5: 55}
        coordinator.client_request_to_request_id = {(b"client-A", 55): 5}
        coordinator.request_id_to_rank = {}
        coordinator.identities_of_data_parallel_ranks = deque([b"rank_0"])
        coordinator._pending_counts = np.zeros(1, dtype=np.int32)
        coordinator.identity_to_rank_index = {b"rank_0": 0}
        coordinator.router_socket = MagicMock()
        coordinator.detokenize = MagicMock()
        return coordinator

    def test_detokenizes_when_the_client_asked(self):
        coordinator = self._coordinator()
        metadata = [Headers.ENGINE_REPLY.value, [[5, True]]]
        body = msgpack.packb({"request_id": 5}, use_bin_type=True)
        handle_engine_reply(coordinator, b"rank_0", metadata, [body])
        coordinator.detokenize.assert_called_once()

    def test_forwards_the_body_untouched_when_it_did_not(self):
        """The opt-out is the whole point: the body is never decoded."""
        coordinator = self._coordinator()
        metadata = [Headers.ENGINE_REPLY.value, [[5, False]]]
        body = msgpack.packb({"request_id": 5}, use_bin_type=True)
        handle_engine_reply(coordinator, b"rank_0", metadata, [body])
        coordinator.detokenize.assert_not_called()
        sent = coordinator.router_socket.send_multipart.call_args.args[0]
        assert body in sent, "an un-detokenized body must be forwarded verbatim"
