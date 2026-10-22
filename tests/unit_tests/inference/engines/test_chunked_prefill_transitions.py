# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicInferenceEngineTestBase,
    set_rounder,
)
from tests.unit_tests.test_utilities import Utils

_CHUNKED_REQUEST_ID = 17
_PEER_REQUEST_ID = 3
_OUTPUT_TOKEN = 5


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestChunkedPrefillTransitions(DynamicInferenceEngineTestBase):
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    def teardown_method(self):
        gc.collect()
        delete_cuda_graphs()
        torch.cuda.empty_cache()

    @classmethod
    def _build_chunked_env(
        cls,
        kv_mode: KVCacheManagementMode,
        schedule_mode: AsyncScheduleMode,
        *,
        with_peer: bool = False,
    ):
        max_tokens = 8 if with_peer else 4
        env = cls._build_test_env(
            DynamicEngineTestConfig(
                num_requests=0,
                max_sequence_length=16,
                context_buffer_size_gb=0.01,
                context_block_size_tokens=256,
                context_max_requests=4,
                context_max_tokens=max_tokens,
                num_tokens_to_generate=3,
                num_gap_steps=0,
                enable_chunked_prefill=True,
                kv_cache_management_mode=kv_mode.value,
                static_kv_memory_pointers=False,
                async_sched_mode=schedule_mode,
                sampling_backend="torch",
            )
        )
        engine = env.engine
        context = engine.context

        async def ignore_notification():
            return None

        engine._notify_cond_for_new_request = ignore_notification
        requests = []
        if with_peer:
            requests.append(
                DynamicInferenceRequest(
                    request_id=_PEER_REQUEST_ID,
                    prompt_tokens=torch.tensor([2, 3], dtype=torch.int64, device="cuda"),
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=3, termination_id=-1, top_k=1
                    ),
                )
            )
        chunked_request = DynamicInferenceRequest(
            request_id=_CHUNKED_REQUEST_ID,
            prompt_tokens=torch.arange(10, 20, dtype=torch.int64, device="cuda"),
            sampling_params=SamplingParams(num_tokens_to_generate=3, termination_id=-1, top_k=1),
        )
        requests.append(chunked_request)
        env.requests = requests
        for request in requests:
            engine._add_request(request)
            request.state = "pending"
        forward_states = []

        def deterministic_forward(input_ids, position_ids):
            del position_ids
            partial_id = context.chunked_prefill_request_id
            partial_progress = (
                engine.get_request(partial_id).finished_chunk_token_count
                if partial_id in engine.requests
                else None
            )
            forward_states.append({"partial_id": partial_id, "partial_progress": partial_progress})
            row_count = context.num_last_token_logits
            logits = torch.full(
                (1, row_count, env.config.vocab_size),
                -100.0,
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
            logits[..., _OUTPUT_TOKEN] = 100.0
            engine.controller._all_logits_cuda = logits

        engine.controller._dynamic_step_forward_logits = deterministic_forward
        return env, chunked_request, forward_states

    @staticmethod
    def _assert_partial_primer(env, request, forward_states):
        result = env.engine.step_modern()
        progress = request.finished_chunk_token_count
        assert result["finished_requests"] == []
        assert 0 < progress < len(request.prompt_tokens)
        assert torch.equal(request.remaining_prompt_tokens, request.prompt_tokens[progress:])
        assert env.engine.context.chunked_prefill_request_id == request.request_id
        assert list(env.engine.waiting_request_ids)[0] == request.request_id
        assert forward_states[-1]["partial_id"] == request.request_id
        assert forward_states[-1]["partial_progress"] == progress
        return result, progress

    @staticmethod
    def _finish_request(env, request_id):
        completed = None
        for _ in range(20):
            if not env.engine.has_unfinished_requests():
                break
            result = env.engine.step_modern()
            for request in result["finished_requests"]:
                if request.request_id == request_id:
                    completed = request
        assert not env.engine.has_unfinished_requests()
        assert completed is not None
        assert completed.status == Status.COMPLETED
        return completed

    @torch.inference_mode()
    def test_sync_partial_prefill_sample_is_suppressed(self):
        """The legacy controller emits a sample, but sync bookkeeping must discard it."""
        env, request, forward_states = self._build_chunked_env(
            KVCacheManagementMode.PERSIST, AsyncScheduleMode.LEGACY
        )
        result, progress = self._assert_partial_primer(env, request, forward_states)

        assert progress == env.engine.context.max_tokens
        assert result["active_request_ids"] == [request.request_id]
        sampled_token = env.engine.controller._sampled_tokens_cuda[0].item()
        assert sampled_token == request.prompt_tokens[progress].item()
        assert request.generated_tokens == []
        assert env.engine.context.get_index_of_chunked_prefill_request(safe=True) == -1
        assert (
            env.engine.context.get_index_of_chunked_prefill_request(safe=False)
            == env.engine.context.total_request_count
        )

    @torch.inference_mode()
    def test_async_bookkeeping_consumes_the_captured_partial_id(self):
        """A later live-ID mutation cannot reclassify output from the consumed forward."""
        env, request, forward_states = self._build_chunked_env(
            KVCacheManagementMode.PERSIST, AsyncScheduleMode.ASYNC
        )
        _, progress = self._assert_partial_primer(env, request, forward_states)
        consumed_boundary = progress
        step_result, consumed_state, step_time = env.engine._run_coroutine_sync(
            env.engine.async_forward()
        )
        assert consumed_state["chunked_prefill_request_id"] == request.request_id
        assert request.finished_chunk_token_count > consumed_boundary
        assert step_result["sample"].tolist() == [request.prompt_tokens[consumed_boundary].item()]
        env.engine.context.chunked_prefill_request_id = request.request_id + 1000
        result = env.engine._run_coroutine_sync(
            env.engine.async_bookkeep(step_result, consumed_state, step_time)
        )

        assert (
            env.engine.context.chunked_prefill_request_id
            != consumed_state["chunked_prefill_request_id"]
        )
        assert result["active_request_ids"] == [request.request_id]
        assert request.generated_tokens == []

    @torch.inference_mode()
    def test_hidden_partial_row_is_excluded_from_generated_logits(self):
        """update_requests hides the partial tail, and last_token_logits drops its row."""
        env, request, forward_states = self._build_chunked_env(
            KVCacheManagementMode.PERSIST, AsyncScheduleMode.ASYNC, with_peer=True
        )
        self._assert_partial_primer(env, request, forward_states)
        context = env.engine.context
        transition = {}
        update_requests = context.update_requests

        def observe_update(*args, **kwargs):
            before_logits = torch.arange(
                context.padded_active_token_count * 3, device="cuda"
            ).reshape(1, context.padded_active_token_count, 3)
            transition["before_ids"] = context.request_ids[: context.total_request_count].tolist()
            transition["before_logits"] = context.last_token_logits(before_logits).clone()
            result = update_requests(*args, **kwargs)
            transition["after_ids"] = context.request_ids[
                : context.total_request_count + 1
            ].tolist()
            transition["after_count"] = context.total_request_count
            transition["after_index"] = context.get_index_of_chunked_prefill_request(safe=False)
            after_logits = torch.arange(
                context.padded_active_token_count * 3, device="cuda"
            ).reshape(1, context.padded_active_token_count, 3)
            transition["after_logits"] = context.last_token_logits(after_logits).clone()
            return result

        context.update_requests = observe_update
        try:
            env.engine._run_coroutine_sync(env.engine.async_forward())
        finally:
            context.update_requests = update_requests

        assert transition["before_ids"] == [_PEER_REQUEST_ID, _CHUNKED_REQUEST_ID]
        assert transition["before_logits"].shape[0] == 2
        assert transition["after_ids"] == [_PEER_REQUEST_ID, _CHUNKED_REQUEST_ID]
        assert transition["after_count"] == 1
        assert transition["after_index"] == 1
        assert transition["after_logits"].shape[0] == 1
        assert torch.equal(transition["after_logits"], transition["before_logits"][:1])

    @pytest.mark.parametrize(
        "kv_mode", [KVCacheManagementMode.PERSIST, KVCacheManagementMode.OFFLOAD]
    )
    @torch.inference_mode()
    def test_mid_partial_residency_cycle_preserves_progress(self, kv_mode):
        """PERSIST and OFFLOAD retain the in-flight chunk and finish deterministically."""
        env, request, forward_states = self._build_chunked_env(kv_mode, AsyncScheduleMode.ASYNC)
        _, progress = self._assert_partial_primer(env, request, forward_states)
        context = env.engine.context
        storage_bytes = context.memory_buffer.untyped_storage().nbytes()
        memory_address = context.memory_buffer.data_ptr()
        assert env.engine.controller._async_sched_forward.is_valid
        env.engine.suspend()
        assert env.engine.state == EngineState.SUSPENDED
        assert request.finished_chunk_token_count == progress
        assert context.chunked_prefill_request_id == request.request_id
        assert env.engine.controller._async_sched_forward.is_valid
        if kv_mode == KVCacheManagementMode.PERSIST:
            assert context.memory_buffer.untyped_storage().nbytes() == storage_bytes
            assert context.memory_buffer.data_ptr() == memory_address
        else:
            assert context.memory_buffer.untyped_storage().nbytes() == 0
        env.engine.resume()
        assert env.engine.state == EngineState.RUNNING
        assert request.finished_chunk_token_count == progress
        assert context.chunked_prefill_request_id == request.request_id
        assert context.memory_buffer.untyped_storage().nbytes() == storage_bytes
        completed = self._finish_request(env, request.request_id)
        assert completed.generated_tokens == [_OUTPUT_TOKEN] * 3
        assert [
            state["partial_progress"]
            for state in forward_states
            if state["partial_id"] == request.request_id
        ] == [progress, 2 * progress]
