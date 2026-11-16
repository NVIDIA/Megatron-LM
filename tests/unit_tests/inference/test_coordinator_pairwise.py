# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real DP routing crossed with inference behavior, not dummy-engine plumbing."""

import asyncio
import copy

import pytest
import torch

from megatron.core.inference.config import (
    AsyncScheduleMode,
    PrefixCachingCoordinatorPolicy,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.headers import Headers
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model, until


async def _collect(stream):
    return [item async for item in stream]


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(AsyncScheduleMode))
@pytest.mark.parametrize("streaming_interval", [0, 1, 3])
async def test_routed_output_matches_direct(monkeypatch, mode, streaming_interval):
    """Two clients both use ID zero; each DP owner runs the direct schedule."""
    prompt = list(range(4, 20))
    params = greedy_params(return_log_probs=True, return_prompt_tokens=True, top_n_logprobs=3)
    async with routed_model(
        monkeypatch, async_sched_mode=mode, materialize_only_last_token_logits=False
    ) as h:
        direct = await h.direct(prompt, params)
        await h.start()
        await h.pause()
        pending = []
        if h.rank == 0:
            for client in h.clients:
                p = copy.deepcopy(params)
                if streaming_interval:
                    p.streaming_interval = streaming_interval
                    stream = client.add_request_streaming(prompt, p)
                    assert stream.request_id == 0
                    pending.append(asyncio.create_task(_collect(stream)))
                else:
                    request_id, future = client.add_request_with_id(prompt, p)
                    assert request_id == 0
                    pending.append(future)
            await until(lambda: len(h.service.coordinator.request_id_to_rank) == h.dp_size)
            assert sorted(h.service.coordinator._pending_counts.tolist()) == [1] * h.dp_size
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            outputs = await asyncio.wait_for(asyncio.gather(*pending), timeout=60)
            engine_request_ids = []
            for output in outputs:
                if streaming_interval:
                    assert len([item for item in output if "final" in item]) == 1
                    assert "final" in output[-1]
                    final = output[-1]["final"]
                    partials = [item["partial"] for item in output if "partial" in item]
                    assert partials, "A final-only stream does not cover partial delivery"
                    tokens = [token for part in partials for token in part["new_tokens"]]
                    assert tokens == final["generated_tokens"][: len(tokens)]
                    assert all(len(p["new_tokens"]) >= streaming_interval for p in partials)
                    for part in partials:
                        assert len(part["new_log_probs"]) == len(part["new_tokens"])
                else:
                    final = output
                engine_request_ids.append(final["request_id"])
                for key in ("status", "generated_tokens", "prompt_tokens"):
                    assert final[key] == direct[key], key
                for key in ("prompt_log_probs", "generated_log_probs"):
                    torch.testing.assert_close(torch.tensor(final[key]), torch.tensor(direct[key]))
                assert final["generated_text"] == h.tokenizer.detokenize(final["generated_tokens"])
            assert sorted(engine_request_ids) == list(range(h.dp_size))
            replies = [e for e in h.service.events if e["header"] == Headers.ENGINE_REPLY]
            assert sum(len(e["metadata"][1]) for e in replies) == h.dp_size
        await h.barrier()
        assert h.witnesses, "Every DP replica must execute its assigned real request"
        assert {rid for step in h.witnesses for rid in step["ids"]} == {h.rank}
        assert any(step["prefill"] for step in h.witnesses)
        assert any(step["decode"] for step in h.witnesses)
        if mode == AsyncScheduleMode.ASYNC:
            async_steps = [step["async_step"] for step in h.witnesses]
            assert max(async_steps) > min(async_steps)
        h.assert_retired()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy",
    [
        PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
        PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK,
    ],
)
async def test_routed_prefix_reuse_executes_on_warm_owner(monkeypatch, policy):
    """Affinity must lead to real KV reuse, not just matching shadow hashes."""
    prompt = list(range(4, 37))
    params = greedy_params()
    async with routed_model(
        monkeypatch,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
        coordinator_options={"prefix_caching_coordinator_policy": policy},
    ) as h:
        direct = await h.direct(prompt, params)
        await h.start()
        if h.rank == 0:
            for _ in range(2):
                final = await asyncio.wait_for(
                    h.clients[0].add_request(prompt, copy.deepcopy(params)), timeout=60
                )
                assert final["generated_tokens"] == direct["generated_tokens"]
            assert final["num_cached_tokens"] >= 16
            submits = [e for e in h.service.events if e["header"] == Headers.SUBMIT_REQUEST]
            owners = [next(iter(e["after"].values())) for e in submits]
            assert len(owners) == 2 and owners[0] == owners[1]
        await h.barrier()
        reused = any(step["cached"].get(1, 0) >= 16 and 1 in step["ids"] for step in h.witnesses)
        assert await h.sync.all_reduce_max(int(reused)) == 1
        h.assert_retired()
