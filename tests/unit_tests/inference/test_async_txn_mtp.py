# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.inference.async_txn import AsyncTxnSkipReason, StepTxn
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class FakeFuture:
    def __init__(self):
        self.result = None

    def set_result(self, result):
        self.result = result


class FakeTokenizer:
    def detokenize(self, tokens):
        return "".join(str(int(token)) for token in tokens)


class FakeKVAllocator:
    total_count = 8
    total_avail = 4
    enable_prefix_caching = False

    def reconstruct_routing_from_blocks(self, block_ids, total_routing_tokens):
        return None


class FakeContext:
    chunked_prefill_request_id = -1
    kv_block_allocator = FakeKVAllocator()


class FakeLaunchContext:
    total_request_count = 1
    paused_request_count = 0

    def using_cuda_graph_this_step(self):
        return False


def _make_request(request_id: int) -> DynamicInferenceRequest:
    request = DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=torch.tensor([1], dtype=torch.int64),
        sampling_params=SamplingParams(num_tokens_to_generate=5),
        status=Status.ACTIVE_AND_GENERATING_TOKENS,
    )
    request.add_event_add_engine()
    return request


def _make_engine(request_ids, *, num_speculative_tokens=2):
    engine = object.__new__(DynamicInferenceEngine)
    engine.context = FakeContext()
    engine.controller = SimpleNamespace(tokenizer=FakeTokenizer())
    engine.requests = {
        request_id: SimpleNamespace(
            record=DynamicInferenceRequestRecord.from_request(_make_request(request_id)),
            future=FakeFuture(),
        )
        for request_id in request_ids
    }
    engine.finished_request_count = 0
    engine.evicted_request_count = 0
    engine.track_generated_token_events = False
    engine.stop_word_finished_request_ids = set()
    engine.stop_word_being_finished_ids = set()
    engine.num_speculative_tokens = num_speculative_tokens
    engine._spec_steps = 0
    engine._spec_tokens_proposed_per_pos = torch.zeros(num_speculative_tokens, dtype=torch.int64)
    engine._spec_tokens_accepted_per_pos = torch.zeros(num_speculative_tokens, dtype=torch.int64)
    engine._check_stop_words_for_request_post_append = lambda request: (False, 0)
    return engine


def test_mtp_accepted_tokens_publish_by_request_id_after_middle_finish():
    engine = _make_engine([101, 102, 103])

    active_request_ids, finished_records = DynamicInferenceEngine.post_process_requests(
        engine,
        torch.tensor([101, 102, 103], dtype=torch.int64),
        torch.tensor([102], dtype=torch.int64),
        evict_request_ids=None,
        step_time=0.0,
        sample=torch.tensor([11, 22, 33], dtype=torch.int64),
        accepted_tokens=torch.tensor([[91, 92], [93, 94], [95, 96]], dtype=torch.int64),
        log_probs=None,
        accepted_tokens_by_request_id={
            103: [77, 78],
            101: [44, -1],
            102: [55, 56],
        },
    )

    assert active_request_ids == [101, 103]
    assert engine.get_request(101).generated_tokens == [44, 11]
    assert engine.get_request(103).generated_tokens == [77, 78, 33]
    assert finished_records[0][-1].request_id == 102
    assert finished_records[0][-1].generated_tokens == [55, 56, 22]


def test_mtp_rejected_speculative_suffix_is_not_published():
    engine = _make_engine([101])

    DynamicInferenceEngine.post_process_requests(
        engine,
        torch.tensor([101], dtype=torch.int64),
        torch.tensor([], dtype=torch.int64),
        evict_request_ids=None,
        step_time=0.0,
        sample=torch.tensor([11], dtype=torch.int64),
        accepted_tokens=torch.tensor([[91, 92]], dtype=torch.int64),
        log_probs=None,
        accepted_tokens_by_request_id={101: [-1, -1]},
    )

    assert engine.get_request(101).generated_tokens == [11]
    assert engine._spec_tokens_accepted_per_pos.tolist() == [0, 0]


def test_mtp_publication_helper_maps_accepted_tokens_by_request_id():
    controller = object.__new__(TextGenerationController)

    publication = controller._dynamic_step_publication_by_request_id(
        torch.tensor([102, 101], dtype=torch.int64),
        torch.tensor([22, 11], dtype=torch.int64),
        accepted_tokens=torch.tensor([[5, -1], [6, 7]], dtype=torch.int64),
        log_probs=None,
        top_n_logprobs=None,
    )

    assert publication["accepted_tokens_by_request_id"] == {102: [5, -1], 101: [6, 7]}


def test_mtp_child_launch_is_gated_instead_of_rejecting_after_launch():
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = 2
    controller._enable_cuda_graph = False
    controller.model_config = SimpleNamespace(expert_model_parallel_size=1, num_moe_experts=None)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=FakeLaunchContext())

    reason = controller._async_child_launch_skip_reason(
        StepTxn(step_id=4, request_ids=(101,)),
        return_log_probs=False,
        return_top_n_logprobs=False,
        skip_bookkeeping=False,
    )

    assert reason == AsyncTxnSkipReason.MTP_ACTIVE
