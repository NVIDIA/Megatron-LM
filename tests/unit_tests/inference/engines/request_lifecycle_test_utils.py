# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
from collections import Counter
from dataclasses import dataclass
from unittest import mock

import msgpack
import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.data_parallel_inference_coordinator.handlers import handle_engine_reply
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    Status,
    unwrap_serialized_tensors,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.unified_memory import prefetch_managed_tensor
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig as _DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase as _DynamicInferenceEngineTestBase,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder as _set_rounder
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import (
    _BASE_PAIR_CONFIG,
    _assert_request_parity,
    _check_scenario_prerequisite,
    _instrument_scenario_runtime,
    _make_scenario_requests,
    _snapshot_requests,
)
from tests.unit_tests.inference.test_dynamic_prefix_caching_coordinator import (
    make_coordinator_direct,
)
from tests.unit_tests.test_utilities import Utils


@dataclass
class _RunResult:
    requests: list[DynamicInferenceRequest]
    checkpoint_counts: dict[int, int]
    runtime: Counter
    witness: dict[str, object] | None


def _configure_test_tokenizer(engine):
    detokenize = lambda tokens, **_kwargs: "".join(f"<{token}>" for token in tokens)
    engine.controller.tokenizer.bos = None
    engine.controller.tokenizer.tokenize = lambda text: [int(token) for token in text.split()]
    engine.controller.tokenizer.detokenize = detokenize
    engine.controller.detokenize = lambda _tokenizer, tokens, **kwargs: detokenize(tokens, **kwargs)


def _make_manual_request(
    context,
    request_id,
    prompt_tokens,
    num_tokens_to_generate,
    *,
    full_logprobs=False,
    stop_words=None,
    keep_stop=False,
):
    return DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        sampling_params=SamplingParams(
            num_tokens_to_generate=num_tokens_to_generate,
            termination_id=-1,
            top_k=1,
            return_log_probs=True,
            skip_prompt_log_probs=not full_logprobs,
            top_n_logprobs=3,
            stop_words=stop_words,
            detokenize_stop_sequence=keep_stop,
        ),
        block_size_tokens=context.block_size_tokens,
        enable_prefix_caching=context.enable_prefix_caching,
    )


def _active_request_ids(context):
    return [
        request_id
        for request_id in context.request_ids[
            context.paused_request_count : context.total_request_count
        ].tolist()
        if request_id >= 0
    ]


def _install_incrementing_logits(env, runtime, modulo):
    engine = env.engine
    context = engine.context
    model = engine.controller.inference_wrapped_model.model
    original = model.forward

    def deterministic_forward(*args, **kwargs):
        request_ids = _active_request_ids(context)
        phase = engine.evicted_request_count
        logits = original(*args, **kwargs)
        tokens = kwargs.get("tokens", kwargs.get("input_ids", args[0] if args else None))
        assert tokens is not None
        if logits.shape[1] != tokens.shape[1]:
            tokens = tokens[:, context.active_logit_idxs[: logits.shape[1]]]
        next_tokens = torch.remainder(tokens + 1, modulo)
        logits.zero_()
        logits.scatter_(-1, next_tokens.unsqueeze(-1), 100.0)
        for request_id in request_ids:
            runtime[("model-forward", phase, request_id)] += 1
        return logits

    model.forward = deterministic_forward
    return model


def _install_real_mtp_logits(env, runtime, modulo):
    model = _install_incrementing_logits(env, runtime, modulo)
    engine = env.engine
    context = engine.context
    original = model.compute_mtp_single_step

    def deterministic_mtp(
        hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
    ):
        request_ids = _active_request_ids(context)
        phase = engine.evicted_request_count
        hidden_states, logits = original(
            hidden_states, next_token_ids, position_ids, depth, eager=eager, cache_key=cache_key
        )
        predicted = torch.remainder(next_token_ids + 1, modulo)
        logits.zero_()
        logits.scatter_(-1, predicted.transpose(0, 1).unsqueeze(-1), 100.0)
        for request_id in request_ids:
            runtime[("mtp-depth", phase, request_id, depth)] += 1
        return hidden_states, logits

    model.compute_mtp_single_step = deterministic_mtp
    return model


def _event_count(request, event_type):
    return sum(event.type == event_type for event in request.events)


def _track_checkpoint_calls(engine, checkpoint_counts, *request_ids):
    for request_id in request_ids:
        record = engine.requests[request_id].record
        record.checkpoint = mock.Mock(
            wraps=record.checkpoint,
            side_effect=lambda _request_id=request_id: checkpoint_counts.update([_request_id])
            or mock.DEFAULT,
        )


def _collect_finished(engine, result, completed):
    for request in result["finished_requests"]:
        assert request.generated_text is None
        request.finalize_text(engine.controller.tokenizer)
        completed[request.request_id] = request


def _assert_engine_drained(engine, futures, completed, request_ids):
    assert set(completed) == set(request_ids)
    assert all(future.done() for future in futures)
    assert not engine.requests
    assert not engine.waiting_request_ids
    assert not engine.failed_request_ids
    assert engine.context.total_request_count == 0
    assert engine.context.paused_request_count == 0


def _snapshot_run(result):
    snapshots = _snapshot_requests(result.requests)
    return {
        "requests": {
            request.request_id: snapshot for request, snapshot in zip(result.requests, snapshots)
        },
        "text": {request.request_id: request.generated_text for request in result.requests},
    }


def _assert_run_parity(actual, expected, atol=1.0e-3):
    _assert_request_parity(
        actual.requests,
        [expected["requests"][request.request_id] for request in actual.requests],
        atol,
        exact_numerics=True,
        exact_top_n=True,
    )
    assert {request.request_id: request.generated_text for request in actual.requests} == expected[
        "text"
    ]


def _run_treatment_pair(run, cleanup):
    baseline = run(treatment=False)
    expected = _snapshot_run(baseline)
    del baseline
    cleanup()
    treatment = run(treatment=True)
    _assert_run_parity(treatment, expected)
    return treatment


def _allocate_leaving(allocator, remaining):
    count = allocator.get_allocatable_count() - remaining
    assert count >= 0
    if count == 0:
        return None
    blocks = allocator.allocate_memory_blocks(count)
    assert blocks is not None and allocator.get_allocatable_count() == remaining
    return blocks.clone()


def _release_filler(allocator, blocks):
    if blocks is not None:
        allocator.release_memory_blocks(blocks)


def _install_mamba_slot_witnesses(env, runtime):
    context = env.engine.context
    metadata = context.mamba_metadata
    original_add = context.add_request

    def traced_add(request, *args, **kwargs):
        result = original_add(request, *args, **kwargs)
        row = _active_request_row(context, request.request_id)
        slot = int(metadata.request_to_mamba_state_idx[row].item())
        assert slot >= 0
        runtime[("mamba-slot-acquire", request.request_id)] += 1
        runtime[("mamba-slot-value", request.request_id, slot)] += 1
        return result

    context.add_request = traced_add
    original_free = metadata.free_slots

    def traced_free(request_indices):
        request_ids = context.request_ids[request_indices].tolist()
        for request_id in request_ids:
            if request_id >= 0:
                runtime[("mamba-slot-free", request_id)] += 1
        return original_free(request_indices)

    metadata.free_slots = traced_free
    installed = 0
    for module in env.engine.controller.inference_wrapped_model.model.modules():
        if type(module).__name__ != "MambaMixer":
            continue
        original_forward = module.forward

        def traced_mamba(*args, _original=original_forward, **kwargs):
            request_ids = _active_request_ids(context)
            phase = env.engine.evicted_request_count
            output = _original(*args, **kwargs)
            for request_id in request_ids:
                runtime[("mamba-forward", phase, request_id)] += 1
            return output

        module.forward = traced_mamba
        installed += 1
    assert installed > 0


def _install_nccl_request_witnesses(env, runtime):
    context = env.engine.context
    installed = 0
    for module in env.engine.controller.inference_wrapped_model.model.modules():
        dispatcher = getattr(module, "_inference_token_dispatcher", None)
        if dispatcher is None or type(dispatcher).__name__ != "NCCLAllGatherDispatcher":
            continue
        original = dispatcher.token_dispatch

        def traced_dispatch(*args, _original=original, **kwargs):
            request_ids = _active_request_ids(context)
            result = _original(*args, **kwargs)
            for request_id in request_ids:
                runtime[("nccl-dispatch", request_id)] += 1
            return result

        dispatcher.token_dispatch = traced_dispatch
        installed += 1
    assert installed > 0


def _feature_keys(scenario):
    return {
        "chunked-partial-recompute-api-coordinator": ("log-probs-calculations",),
        "persist-te-swa-stochastic": (
            "module-forward:transformer-engine",
            "swa-kernel-calls",
            "full-attention-kernel-calls",
            "sink-correction-calls",
            "temperature-filter",
            "top-k-filter",
        ),
        "offload-dynamic-fp8": (
            "module-forward:fp8",
            "fp8-context-forwards",
            "fp8-quantized-forwards",
            "fp8-recipe-forwards",
        ),
        "uvm-offload-static-managed-capacity": ("module-forward:gpt",),
    }[scenario.name]


def _request_feature_key(request_id, feature):
    return ("request-feature", request_id, feature)


def _instrument_request_correlated_runtime(env, scenario, runtime):
    _instrument_scenario_runtime(env, scenario, runtime)
    controller = env.engine.controller
    context = env.engine.context
    feature_keys = set(_feature_keys(scenario))

    def active_request_ids():
        return context.request_ids[
            context.paused_request_count : context.total_request_count
        ].tolist()

    def instrument_call(owner, method_name):
        original = getattr(owner, method_name)

        def correlated(*args, **kwargs):
            request_ids = active_request_ids()
            before = {key: runtime[key] for key in feature_keys}
            result = original(*args, **kwargs)
            for key in feature_keys:
                delta = runtime[key] - before[key]
                if delta > 0:
                    for request_id in request_ids:
                        runtime[_request_feature_key(request_id, key)] += delta
            return result

        setattr(owner, method_name, correlated)

    instrument_call(controller, "_dynamic_step_forward_logits")
    instrument_call(controller._sampling, "sample_kernel")
    instrument_call(context, "calculate_log_probs_tensors")


def _active_request_row(context, request_id):
    rows = torch.nonzero(
        context.request_ids[: context.total_request_count] == request_id, as_tuple=False
    ).flatten()
    assert rows.numel() == 1, (request_id, context.request_ids[: context.total_request_count])
    return int(rows.item())


def _request_kv_snapshot(context, request_id):
    row = _active_request_row(context, request_id)
    block_ids = context.request_to_kv_block_ids[row]
    block_ids = torch.unique(block_ids[block_ids >= 0]).to(
        device=context.memory_buffer.device, dtype=torch.long
    )
    assert block_ids.numel() > 0
    return row, block_ids, context.memory_buffer.index_select(2, block_ids).clone()


def _coordinator_projection(request):
    coordinator = make_coordinator_direct(data_parallel_size=1, enable_prefix_caching=False)
    coordinator.tokenizer.detokenize = mock.Mock(wraps=coordinator.tokenizer.detokenize)
    rank = coordinator.identities_of_data_parallel_ranks[0]
    request_id = request.request_id
    client = b"request-lifecycle-client"
    client_request_id = 1000 + request_id
    coordinator.router_socket = mock.Mock()
    coordinator.request_id_to_client_id = {request_id: client}
    coordinator.request_id_to_client_request_id = {request_id: client_request_id}
    coordinator.client_request_to_request_id = {(client, client_request_id): request_id}
    coordinator.request_id_to_rank = {request_id: rank}
    coordinator._pending_counts[coordinator.identity_to_rank_index[rank]] = 1

    engine_payload = msgpack.packb(
        [Headers.ENGINE_REPLY.value, [{**request.serialize(), "generated_text": None}]],
        use_bin_type=True,
    )
    handle_engine_reply(coordinator, rank, msgpack.unpackb(engine_payload, raw=False))
    coordinator.tokenizer.detokenize.assert_called_once()
    frames = coordinator.router_socket.send_multipart.call_args.args[0]
    assert frames[0] == client
    header, returned_id, returned = msgpack.unpackb(frames[1], raw=False)
    assert header == Headers.ENGINE_REPLY.value
    assert returned_id == client_request_id
    assert request_id not in coordinator.request_id_to_client_id
    return DynamicInferenceRequest.deserialize(returned), unwrap_serialized_tensors(returned)


def _assert_coordinator_parity(actual, expected):
    exact_fields = (
        "request_id",
        "status",
        "prompt",
        "prompt_tokens",
        "remaining_prompt_tokens",
        "prompt_length",
        "generated_tokens",
        "generated_text",
        "generated_length",
        "sampling_params",
    )
    for field_name in exact_fields:
        actual_value = getattr(actual, field_name)
        expected_value = getattr(expected, field_name)
        if isinstance(actual_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value)
        elif field_name == "sampling_params":
            assert actual_value.serialize() == expected_value.serialize()
        else:
            assert actual_value == expected_value
    for field_name in ("prompt_log_probs", "generated_log_probs"):
        assert getattr(actual, field_name) == pytest.approx(
            getattr(expected, field_name), rel=0, abs=1.0e-3
        )
    for field_name in ("prompt_top_n_logprobs", "generated_top_n_logprobs"):
        actual_values = getattr(actual, field_name)
        expected_values = getattr(expected, field_name)
        assert len(actual_values) == len(expected_values)
        for actual_row, expected_row in zip(actual_values, expected_values):
            assert actual_row.keys() == expected_row.keys()
            assert list(actual_row.values()) == pytest.approx(
                list(expected_row.values()), rel=0, abs=1.0e-3
            )


def _assert_coordinator_result_contract(request, endpoint_result):
    prompt_length = len(request.prompt_tokens)
    generated_length = len(request.generated_tokens)
    assert request.prompt_length == prompt_length
    assert torch.equal(request.remaining_prompt_tokens, request.prompt_tokens)
    assert endpoint_result["prompt_tokens"] == request.prompt_tokens.tolist()
    assert endpoint_result["remaining_prompt_tokens"] == request.prompt_tokens.tolist()
    assert len(request.prompt_log_probs) == max(0, prompt_length - 1)
    assert len(request.prompt_top_n_logprobs) == max(0, prompt_length - 1)
    assert len(request.generated_log_probs) == generated_length
    assert len(request.generated_top_n_logprobs) == generated_length
    assert request.generated_length == generated_length


class RequestLifecyclePairwiseBase(_DynamicInferenceEngineTestBase):

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
        _set_rounder(64)
        Utils.destroy_model_parallel()

    @staticmethod
    def _cleanup():
        gc.collect()
        delete_cuda_graphs()
        torch.cuda.empty_cache()

    @classmethod
    def _intervene(cls, scenario, engine, target_id, runtime):
        context = engine.context
        request = engine.get_request(target_id)
        feature_keys = _feature_keys(scenario)
        counts_before = {key: runtime[_request_feature_key(target_id, key)] for key in feature_keys}
        assert all(count > 0 for count in counts_before.values())

        if scenario.name == "chunked-partial-recompute-api-coordinator":
            assert context.chunked_prefill_request_id == target_id
            assert request.finished_chunk_token_count > 0
            assert 0 < len(request.remaining_prompt_tokens) < len(request.prompt_tokens)
            assert request.generated_tokens == []
            witness = {
                "request_id": target_id,
                "finished_chunk_token_count": request.finished_chunk_token_count,
                "remaining_prompt_length": len(request.remaining_prompt_tokens),
            }
            engine.suspend()
            assert engine.resume_request_ids == [3, 1, 0]
            engine.resume()
            resumed = engine.get_request(target_id)
            assert resumed.finished_chunk_token_count == 0
            assert torch.equal(resumed.remaining_prompt_tokens, resumed.prompt_tokens)
        else:
            row, block_ids, kv_before = _request_kv_snapshot(context, target_id)
            pointer_before = context.memory_buffer.data_ptr()
            storage_bytes_before = context.memory_buffer.untyped_storage().nbytes()
            witness = {
                "request_id": target_id,
                "row": row,
                "block_ids": block_ids.tolist(),
                "pointer_before": pointer_before,
                "storage_bytes_before": storage_bytes_before,
            }
            engine.suspend()
            assert not context.is_tensor_state_allocated
            if scenario.name == "persist-te-swa-stochastic":
                assert context.kv_cache_management_mode == KVCacheManagementMode.PERSIST
                assert context.static_kv_memory_pointers
                assert context.memory_buffer.data_ptr() == pointer_before
                assert torch.equal(context.memory_buffer.index_select(2, block_ids), kv_before)
            elif scenario.name == "uvm-offload-static-managed-capacity":
                allocator = context.kv_block_allocator
                cuda_only_usable_blocks = allocator.pool_size - allocator.paused_limit - 1
                assert context.unified_memory_level == 1
                assert context.unified_memory_mempool is not None
                assert context.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD
                assert context.static_kv_memory_pointers
                assert any(block_id >= cuda_only_usable_blocks for block_id in block_ids.tolist())
                prefetch_managed_tensor(context.memory_buffer, device=-1)
                torch.cuda.current_stream().synchronize()
                prefetch_managed_tensor(context.memory_buffer, device=torch.cuda.current_device())
                torch.cuda.current_stream().synchronize()
                assert context.memory_buffer.data_ptr() == pointer_before
                assert torch.equal(context.memory_buffer.index_select(2, block_ids), kv_before)
                witness["cuda_only_usable_blocks"] = cuda_only_usable_blocks
                witness["prefetch_succeeded"] = True
            else:
                assert scenario.name == "offload-dynamic-fp8"
                assert context.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD
                assert not context.static_kv_memory_pointers
                assert context.memory_buffer.untyped_storage().nbytes() == 0
                assert context._offloadable_cpu_backups
            engine.resume()
            assert context.is_tensor_state_allocated
            resumed_row = _active_request_row(context, target_id)
            assert resumed_row == row
            assert torch.equal(context.memory_buffer.index_select(2, block_ids), kv_before)
            if scenario.name in (
                "persist-te-swa-stochastic",
                "uvm-offload-static-managed-capacity",
            ):
                assert context.memory_buffer.data_ptr() == pointer_before
            else:
                assert context.memory_buffer.untyped_storage().nbytes() == storage_bytes_before

        witness["feature_counts_before"] = counts_before
        return witness

    @classmethod
    def _run_once(cls, scenario, *, treatment):
        is_chunked = scenario.name == "chunked-partial-recompute-api-coordinator"
        config_values = dict(_BASE_PAIR_CONFIG)
        config_values.update(scenario.config)
        config_values.update(num_requests=0, async_sched_mode=AsyncScheduleMode.LEGACY)
        test_config = _DynamicEngineTestConfig(**config_values)
        if test_config.unified_memory_level:
            test_config.inference_config_overrides["unified_memory_level"] = (
                test_config.unified_memory_level
            )
        test_config.use_flashinfer_fused_rope = False
        env = cls._build_test_env(test_config)
        engine = env.engine
        if scenario.name == "uvm-offload-static-managed-capacity":
            assert (
                engine.context.unified_memory_level == 1
            ), "the designated UVM owner must use managed allocation"
        all_requests = _make_scenario_requests(env, scenario)
        engine.controller.tokenizer.detokenize = lambda tokens, **_kwargs: "".join(
            f"<{token}>" for token in tokens
        )
        requests = [all_requests[i] for i in (3, 1, 0)] if is_chunked else all_requests[:3]
        target_id = requests[-1 if is_chunked else 0].request_id
        futures = [engine._add_request(request) for request in requests]
        records = {request_id: entry.record for request_id, entry in engine.requests.items()}
        for record in records.values():
            record.checkpoint = mock.Mock(wraps=record.checkpoint)
        runtime = Counter()
        _instrument_request_correlated_runtime(env, scenario, runtime)
        feature_keys = _feature_keys(scenario)
        intervention = None
        completed = {}

        for _ in range(128):
            result = engine.step_modern()
            runtime["steps"] += 1
            _collect_finished(engine, result, completed)

            if treatment and intervention is None and target_id in engine.requests:
                target = engine.get_request(target_id)
                if is_chunked:
                    ready = (
                        target.finished_chunk_token_count > 0
                        and len(target.remaining_prompt_tokens) < len(target.prompt_tokens)
                        and not target.generated_tokens
                    )
                else:
                    ready = len(target.generated_tokens) >= (
                        2 if scenario.name == "persist-te-swa-stochastic" else 1
                    )
                if ready:
                    intervention = cls._intervene(scenario, engine, target_id, runtime)

            if not engine.has_unfinished_requests():
                break
        else:
            pytest.fail(f"{scenario.name} did not drain within 128 steps")

        assert set(completed) == {request.request_id for request in requests}
        assert all(future.done() for future in futures)
        assert not engine.requests
        assert not engine.waiting_request_ids
        assert not engine.failed_request_ids
        assert engine.context.total_request_count == 0
        assert engine.context.paused_request_count == 0
        if treatment:
            assert intervention is not None
            for key in feature_keys:
                assert (
                    runtime[_request_feature_key(target_id, key)]
                    > intervention["feature_counts_before"][key]
                )

        finished = [completed[request_id] for request_id in sorted(completed)]
        assert all(request.status == Status.COMPLETED for request in finished)
        if is_chunked:
            assert runtime["log-probs-calculations"] > 0
            assert all(request.prompt_log_probs is not None for request in finished)
            assert all(request.generated_log_probs is not None for request in finished)
            assert all(request.prompt_top_n_logprobs for request in finished)
            assert all(request.generated_top_n_logprobs for request in finished)
        checkpoint_counts = {
            request_id: record.checkpoint.call_count for request_id, record in records.items()
        }
        return _RunResult(finished, checkpoint_counts, runtime, intervention)

    @classmethod
    def _assert_pair(cls, scenario, *, coordinator=False):
        _check_scenario_prerequisite(scenario)
        baseline = cls._run_once(scenario, treatment=False)
        baseline_by_id = {request.request_id: request for request in baseline.requests}
        expected = {
            request.request_id: snapshot
            for request, snapshot in zip(baseline.requests, _snapshot_requests(baseline.requests))
        }
        expected_text = {
            request.request_id: request.generated_text for request in baseline.requests
        }
        baseline_wire = (
            {
                request_id: _coordinator_projection(request)
                for request_id, request in baseline_by_id.items()
            }
            if coordinator
            else None
        )
        del baseline
        cls._cleanup()

        treatment = cls._run_once(scenario, treatment=True)
        _assert_request_parity(
            treatment.requests,
            [expected[request.request_id] for request in treatment.requests],
            scenario.atol,
            exact_numerics=True,
            exact_top_n=True,
        )
        assert {
            request.request_id: request.generated_text for request in treatment.requests
        } == expected_text
        if coordinator:
            treatment_wire = {
                request.request_id: _coordinator_projection(request)
                for request in treatment.requests
            }
            assert set(treatment_wire) == set(baseline_wire)
            for request_id, (actual, actual_endpoint) in treatment_wire.items():
                expected_request, _ = baseline_wire[request_id]
                _assert_coordinator_parity(actual, expected_request)
                _assert_coordinator_result_contract(actual, actual_endpoint)
        assert treatment.witness is not None
        return treatment

    @classmethod
    def _manual_env(cls, **config_values):
        vocab_size = config_values.pop("vocab_size", None)
        values = {
            "context_max_requests": 4,
            "context_buffer_size_gb": 0.01,
            "context_paused_buffer_size_gb": 0.0,
            "track_generated_token_events": True,
            "inference_config_overrides": {"track_paused_request_events": True},
        }
        values.update(config_values)
        config = _DynamicEngineTestConfig(num_requests=0, **values)
        if vocab_size is not None:
            config.vocab_size = vocab_size
        config.use_flashinfer_fused_rope = False
        env = cls._build_test_env(config)
        _configure_test_tokenizer(env.engine)
        return config, env
