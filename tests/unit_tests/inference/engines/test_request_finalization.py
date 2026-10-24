# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from collections import deque
from functools import partial
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
)
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.text_generation_controllers.test_text_generation_controller import (
    _make_async_sched_context,
    _make_async_sched_controller,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["legacy", "no_overlap", "overlap"])
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize("trim_stop", [False, True])
async def test_finalization_precedes_reuse_but_publication_waits_for_bookkeeping(
    path, mtp, prefix_caching, trim_stop
):
    """A finishes while B reuses its blocks; A's complete result remains intact."""
    context = _make_async_sched_context(total_request_count=2)
    context.num_decode_requests = 2
    context.request_metadata["termination_id"] = torch.tensor([1, 99])
    context.active_request_metadata["termination_id"] = torch.tensor([1, 99])
    context.chunked_prefill_request_id = -1
    context.remove_vlm_request_data = mock.Mock()
    allocator = KVBlockAllocator(
        context, pool_size=8, paused_limit=0, enable_prefix_caching=prefix_caching
    )
    blocks = allocator.allocate_memory_blocks(4).clone()
    context.kv_block_allocator = allocator
    context.request_to_kv_block_ids = blocks.reshape(2, 2)
    context.get_index_of_chunked_prefill_request = mock.Mock(return_value=-1)
    controller = _make_async_sched_controller(context)
    controller.num_speculative_tokens = 2 if mtp else 0
    controller.tokenizer = SimpleNamespace(detokenize=lambda ids: str(ids[0]))
    controller._compact_async_sched_forward = mock.Mock()
    controller._synchronize_async_sched_event = mock.Mock()

    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.controller = controller
    engine.context = context
    engine.requests = {}
    for request_id, token in [(10, 8), (11, 9)]:
        request = DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=torch.tensor([3, 4]),
            generated_tokens=[token],
            generated_log_probs=[-0.8],
            tpot=[0.8],
            sampling_params=SamplingParams(
                num_tokens_to_generate=8,
                termination_id=1,
                return_log_probs=True,
                skip_prompt_log_probs=True,
                top_n_logprobs=1,
            ),
        )
        request.add_event_add_engine()
        if trim_stop and request_id == 10:
            request.stop_word_ids = [[1]]
        engine.requests[request_id] = SimpleNamespace(
            record=DynamicInferenceRequestRecord.from_request(request),
            future=asyncio.get_running_loop().create_future(),
        )
    future = engine.requests[10].future
    engine.finished_request_count = engine.evicted_request_count = 0
    engine.track_generated_token_events = False
    engine.num_speculative_tokens = controller.num_speculative_tokens
    engine._spec_steps = 0
    engine._spec_tokens_proposed_per_pos = torch.zeros(2, dtype=torch.long)
    engine._spec_tokens_accepted_per_pos = torch.zeros(2, dtype=torch.long)
    engine.stop_word_being_finished_ids = set()
    engine.stop_word_finished_request_ids = set()
    engine.waiting_request_ids = deque()
    engine.local_metadata_ledger_enabled = False

    expected_routing = np.arange(4, dtype=np.int32).reshape(4, 1, 1)
    for i, block in enumerate(blocks[:2].tolist()):
        allocator.store_block_routing(block, np.arange(2), expected_routing[2 * i : 2 * i + 2])
    samples = torch.tensor([1, 2])
    accepted = torch.tensor([[6, -1], [7, -1]]) if mtp else None
    score_count = 2 if mtp else 1
    log_probs = [[-0.1] * score_count, [-0.2] * score_count]
    top_n = {
        row: [(torch.tensor([-0.1]), torch.tensor([1])) for _ in range(score_count)]
        for row in range(2)
    }
    completed = {}
    finalize = partial(
        engine._finalize_finished_requests,
        completed,
        {"chunked_prefill_request_id": -1, "active_token_count": 2, "step_count": 3},
        True,
    )

    def reuse_finished_blocks(*args):
        assert set(completed) == {10}
        assert not future.done()
        assert engine.get_request(11).generated_tokens == [9]
        np.testing.assert_array_equal(
            completed[10][0].routing_indices, expected_routing[: 2 + score_count - int(trim_stop)]
        )
        allocator.release_memory_blocks(blocks[:2])
        reused = allocator.allocate_memory_blocks(2).clone()
        assert set(reused.tolist()) == set(blocks[:2].tolist())
        for block in reused.tolist():
            allocator.store_block_routing(block, np.arange(2), expected_routing[:2] + 100)
        if path == "overlap":
            return torch.tensor([10]), torch.tensor([1])
        return {}

    context.update_requests = mock.Mock(side_effect=reuse_finished_blocks)
    context.resolve_requests = mock.Mock(side_effect=reuse_finished_blocks)
    if path == "legacy":
        controller._sampled_tokens_cuda[:2] = samples
        controller._sampled_mtp_tokens_cuda = torch.zeros((2, 2), dtype=torch.long)
        controller._accepted_tokens_per_request = accepted
        controller._dynamic_step_context_bookkeeping(
            finalize_requests=finalize, log_probs=log_probs, top_n_logprobs=top_n
        )
    else:
        sample_result = SimpleNamespace(
            sampled_tokens_cpu_view=samples,
            sampled_mtp_tokens_cpu_view=None,
            accepted_tokens_cpu_view=accepted,
            accepted_counts_cpu_view=torch.ones(2, dtype=torch.long) if mtp else None,
            routing_record=None,
        )
        controller._materialize_async_sched_log_probs = mock.Mock(return_value=(log_probs, top_n))
        method = (
            controller._run_async_sched_resolve
            if path == "overlap"
            else controller._run_async_sched_update_requests
        )
        method(sample_result, context.get_active_sequence_lengths() + 1, finalize_requests=finalize)

    assert not future.done()
    active, finished = engine.post_process_requests(
        request_ids=torch.tensor([10, 11]),
        finished_request_ids=torch.tensor([10]),
        evict_request_ids=None,
        step_time=0.25,
        sample=samples,
        accepted_tokens=accepted,
        log_probs=log_probs,
        top_n_logprobs=top_n,
        consumed_chunked_prefill_request_id=-1,
        completed_requests=completed,
    )
    assert active == [11]
    assert future.result() is finished[0]
    expected_tokens = [8, 6, 1] if mtp else [8, 1]
    kept = score_count - int(trim_stop)
    assert finished[0].generated_tokens == (expected_tokens[:-1] if trim_stop else expected_tokens)
    assert len(finished[0].generated_log_probs) == 1 + kept
    assert len(finished[0].generated_top_n_logprobs) == kept
    assert finished[0].tpot == [0.8] + [0.25 / score_count] * kept
    assert engine.get_request(11).generated_tokens == ([9, 7, 2] if mtp else [9, 2])
    assert engine.finished_request_count == 1
    assert engine._spec_steps == int(mtp)
    np.testing.assert_array_equal(finished[0].routing_indices, expected_routing[: 2 + kept])
