# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.async_txn import StepTxn
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

    def done(self):
        return self.result is not None


class FakeTokenizer:
    def detokenize(self, tokens):
        return "".join(f"T{int(token)}" for token in tokens)


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
    def __init__(self):
        self.total_request_count = 2
        self.paused_request_count = 0

    def using_cuda_graph_this_step(self):
        return False


def _make_request(request_id: int, *, top_n_logprobs: int = 2) -> DynamicInferenceRequest:
    request = DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=torch.tensor([1], dtype=torch.int64),
        sampling_params=SamplingParams(
            num_tokens_to_generate=2,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            top_n_logprobs=top_n_logprobs,
        ),
        status=Status.ACTIVE_AND_GENERATING_TOKENS,
    )
    request.add_event_add_engine()
    return request


def _make_engine(request_ids):
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
    engine.num_speculative_tokens = 0
    engine._check_stop_words_for_request_post_append = lambda request: (False, 0)
    return engine


def test_publication_maps_generated_artifacts_by_request_id_after_compaction():
    controller = object.__new__(TextGenerationController)
    top_n = {
        0: [(torch.tensor([-0.11, -1.1]), torch.tensor([11, 1]))],
        1: [(torch.tensor([-0.22, -2.2]), torch.tensor([22, 2]))],
        2: [(torch.tensor([-0.33, -3.3]), torch.tensor([33, 3]))],
    }

    publication = controller._dynamic_step_publication_by_request_id(
        torch.tensor([101, 102, 103], dtype=torch.int64),
        torch.tensor([11, 22, 33], dtype=torch.int64),
        accepted_tokens=None,
        log_probs=[[-0.11], [-0.22], [-0.33]],
        top_n_logprobs=top_n,
    )

    assert publication["sample_by_request_id"] == {101: 11, 102: 22, 103: 33}
    assert publication["log_probs_by_request_id"] == {
        101: [-0.11],
        102: [-0.22],
        103: [-0.33],
    }
    assert publication["top_n_logprobs_by_request_id"][103] is top_n[2]


def test_post_process_prefers_request_id_publication_for_logprobs_and_top_n():
    engine = _make_engine([101, 102, 103])

    active_request_ids, finished_records = DynamicInferenceEngine.post_process_requests(
        engine,
        torch.tensor([101, 102, 103], dtype=torch.int64),
        torch.tensor([102], dtype=torch.int64),
        evict_request_ids=None,
        step_time=0.0,
        sample=torch.tensor([11, 22, 33], dtype=torch.int64),
        accepted_tokens=None,
        log_probs=[[-9.01], [-9.02], [-9.03]],
        top_n_logprobs={
            0: [(torch.tensor([-9.01]), torch.tensor([91]))],
            1: [(torch.tensor([-9.02]), torch.tensor([92]))],
            2: [(torch.tensor([-9.03]), torch.tensor([93]))],
        },
        log_probs_by_request_id={103: [-0.33], 101: [-0.11], 102: [-0.22]},
        top_n_logprobs_by_request_id={
            103: [(torch.tensor([-0.33]), torch.tensor([33]))],
            101: [(torch.tensor([-0.11]), torch.tensor([11]))],
            102: [(torch.tensor([-0.22]), torch.tensor([22]))],
        },
    )

    assert active_request_ids == [101, 103]
    assert len(finished_records) == 1
    assert engine.get_request(101).generated_log_probs == [-0.11]
    assert engine.get_request(103).generated_log_probs == [-0.33]
    assert finished_records[0][-1].request_id == 102
    assert finished_records[0][-1].generated_log_probs == [-0.22]
    assert engine.get_request(101).generated_top_n_logprobs[0]["T11"] == pytest.approx(-0.11)
    assert engine.get_request(103).generated_top_n_logprobs[0]["T33"] == pytest.approx(-0.33)
    assert finished_records[0][-1].generated_top_n_logprobs[0]["T22"] == pytest.approx(-0.22)


def test_no_logprob_publication_keeps_logprob_maps_empty():
    controller = object.__new__(TextGenerationController)

    publication = controller._dynamic_step_publication_by_request_id(
        torch.tensor([101], dtype=torch.int64),
        torch.tensor([11], dtype=torch.int64),
        accepted_tokens=None,
        log_probs=None,
        top_n_logprobs=None,
    )

    assert publication["log_probs_by_request_id"] == {}
    assert publication["top_n_logprobs_by_request_id"] == {}


def test_logprobs_and_stop_words_do_not_force_serial_child_launch():
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = 0
    controller._enable_cuda_graph = False
    controller._get_stop_word_finished_ids_callback = lambda request_ids: set()
    controller.model_config = SimpleNamespace(expert_model_parallel_size=1, num_moe_experts=None)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=FakeLaunchContext())

    reason = controller._async_child_launch_skip_reason(
        StepTxn(step_id=4, request_ids=(101, 102)),
        return_log_probs=True,
        return_top_n_logprobs=True,
        skip_bookkeeping=False,
    )

    assert reason is None
