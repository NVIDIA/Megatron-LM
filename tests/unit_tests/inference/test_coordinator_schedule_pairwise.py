# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import copy

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model, until


async def _submit_paused(h, prompt, params):
    before = set(h.service.coordinator.request_id_to_rank)
    future = h.clients[0].add_request(prompt, copy.deepcopy(params))
    await until(lambda: len(h.service.coordinator.request_id_to_rank) == len(before) + 1)
    request_id = (set(h.service.coordinator.request_id_to_rank) - before).pop()
    return request_id, h.service.coordinator.request_id_to_rank[request_id], future


@pytest.mark.internal
@pytest.mark.asyncio
async def test_routed_chunk_partition_beside_decode(monkeypatch):
    prompt, params = list(range(4, 69)), greedy_params(num_tokens_to_generate=16)
    async with routed_model(monkeypatch, enable_chunked_prefill=True, context_max_tokens=32) as h:
        assert h.dp_size == 2
        direct = await h.direct(prompt, params)
        await h.start()
        parts = []
        target = [None, None]
        controller = h.engine.controller
        forward = controller._dynamic_step_forward_logits

        def observe(*args, **kwargs):
            c = h.engine.context
            rows = slice(c.paused_request_count, c.total_request_count)
            tensors = (c.request_ids, c.request_query_lengths, c.request_in_prefill_status_tensor)
            snapshot = [
                (count, c.num_decode_requests)
                for rid, count, prefill in zip(*(tensor[rows].tolist() for tensor in tensors))
                if prefill and rid == target[0] and rid in h.engine.requests
            ]
            result = forward(*args, **kwargs)
            parts.extend(snapshot)
            return result

        monkeypatch.setattr(controller, "_dynamic_step_forward_logits", observe)
        await h.pause()
        if h.rank == 0:
            _, decode_owner, a = await _submit_paused(h, [4, 5, 6, 7], params)
            _, _, b = await _submit_paused(h, [8, 9, 10, 11], params)
            target[0], target[1], future = await _submit_paused(h, prompt, params)
            assert decode_owner == target[1]
        torch.distributed.broadcast_object_list(target, src=0)
        await h.unpause()
        if h.rank == 0:
            _, _, final = await asyncio.wait_for(asyncio.gather(a, b, future), timeout=60)
            assert final["generated_tokens"] == direct["generated_tokens"]
        await h.barrier()
        owns_target = target[1] == f"mp-coord-{h.rank}".encode()
        assert bool(parts) == owns_target
        if parts:
            assert len(parts) >= 3 and sum(count for count, _ in parts) == len(prompt)
            assert all(0 < count <= 32 for count, _ in parts)
            assert any(decode > 0 for _, decode in parts)
        h.assert_retired()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["affinity", "load", "expired"])
async def test_routed_affinity_competes_with_load_and_expiry(monkeypatch, choice):
    prompt, params = list(range(4, 37)), greedy_params()
    options = dict(
        prefix_caching_routing_alpha=100 if choice == "load" else 0,
        prefix_cache_ttl_seconds=0 if choice == "expired" else 300,
    )
    async with routed_model(
        monkeypatch,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
        coordinator_options=options,
    ) as h:
        assert h.dp_size == 2
        direct = await h.direct(prompt, params)
        await h.start()
        await h.pause()
        if h.rank == 0:
            _, _, filler = await _submit_paused(h, [91, 92, 93, 94], params)
            _, warm_owner, warm = await _submit_paused(h, prompt, params)
        await h.unpause()
        if h.rank == 0:
            await asyncio.wait_for(asyncio.gather(filler, warm), timeout=60)
            if choice == "expired":
                await asyncio.wait_for(h.clients[0].add_request(list(range(60, 77)), params), 60)
        await h.barrier()
        await h.pause()
        if h.rank == 0:
            coordinator = h.service.coordinator
            pending = []
            if choice != "expired":
                _, _, blocker = await _submit_paused(h, prompt, params)
                pending.append(blocker)
                assert coordinator._pending_counts.sum() == 1
                assert coordinator.get_least_loaded_data_parallel_rank() != warm_owner
            target, owner, future = await _submit_paused(h, prompt, params)
            assert (owner == warm_owner) == (choice == "affinity")
        target = [target if h.rank == 0 else None, owner if h.rank == 0 else None]
        torch.distributed.broadcast_object_list(target, src=0)
        await h.unpause()
        if h.rank == 0:
            outputs = await asyncio.wait_for(asyncio.gather(*pending, future), timeout=60)
            assert outputs[-1]["generated_tokens"] == direct["generated_tokens"]
        await h.barrier()
        steps = [s for s in h.witnesses if target[0] in s["ids"]]
        owns_target = target[1] == f"mp-coord-{h.rank}".encode()
        assert bool(steps) == owns_target
        if steps:
            assert (max(s["cached"][target[0]] for s in steps) >= 16) == (choice == "affinity")
        h.assert_retired()
