# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import asyncio
import hashlib
from functools import wraps
from types import SimpleNamespace

import msgpack
import pytest
import torch
import zmq

from megatron.core.activations import squared_relu
from megatron.core.inference.config import PrefixCachingCoordinatorPolicy as Policy
from megatron.core.inference.engines.async_zmq_communicator import AsyncZMQCommunicator
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.moe import batch_invariant as moe_bi
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.moe import token_dispatcher_inference as dispatch
from tests.unit_tests.inference.engines import batch_invariant_test_utils as bi
from tests.unit_tests.inference.engines import test_batch_invariant_vlm as vlm
from tests.unit_tests.inference.test_data_parallel_inference_coordinator import cleanup_engine


class _FrameObserver:
    def __init__(self, socket):
        self.socket, self.frames = socket, []

    def recv_multipart(self, *args, **kwargs):
        frames = self.socket.recv_multipart(*args, **kwargs)
        self.frames.append(frames)
        return frames

    def __getattr__(self, name):
        return getattr(self.socket, name)


def _params(**overrides):
    options = dict(
        num_tokens_to_generate=6,
        top_k=1,
        termination_id=-1,
        return_log_probs=False,
        return_prompt_tokens=False,
        detokenize_generations=True,
        streaming_interval=2,
    )
    options.update(overrides)
    return SamplingParams(**options)


async def _sync(comm, value=1, timeout=60):
    return await asyncio.wait_for(comm.all_reduce_max(value), timeout)


async def _collect(stream):
    return [message async for message in stream]


def _stream_result(messages):
    partials = [message["partial"] for message in messages if "partial" in message]
    finals = [message["final"] for message in messages if "final" in message]
    assert len(finals) == 1 and [len(p["new_tokens"]) for p in partials] == [2, 2]
    result = finals[0]
    streamed = [token for partial in partials for token in partial["new_tokens"]]
    assert streamed == result.generated_tokens[:4] and len(result.generated_tokens) == 6
    assert result.prompt_tokens is None and result.prompt_length == 258
    assert result.generated_text == " ".join(map(str, result.generated_tokens))
    return result


def _headers(observer, request_id):
    metadata = [msgpack.unpackb(frames[0], raw=False) for frames in observer.frames]
    return [Headers(data[0]) for data in metadata if len(data) > 1 and data[1] == request_id]


async def _shutdown(engine, client, address, comm, context, rank):
    await cleanup_engine(engine, client, timeout=30)
    if rank == 0:
        shutdown_client = InferenceClient(address)
        shutdown_client.start(connect_timeout_seconds=10)
        shutdown_client.shutdown_coordinator()
        engine.inference_coordinator_process.join(timeout=10)
        shutdown_client.stop()
        assert not engine.inference_coordinator_process.is_alive()
    comm.close()
    context.term()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,progress",
    [
        pytest.param(Policy.LONGEST_PREFIX, None, id="longest-prefix"),
        pytest.param(Policy.FIRST_PREFIX_BLOCK, None, id="first-block"),
        pytest.param(Policy.LONGEST_PREFIX, (False, False, 1), id="ep-async"),
        pytest.param(Policy.LONGEST_PREFIX, (True, False, 3), id="ep-sync"),
        pytest.param(Policy.LONGEST_PREFIX, (False, True, 1), id="ep-disabled"),
    ],
)
async def test_live_coordinator_batch_invariance_and_output_contracts(policy, progress):
    synchronous, disabled, interval = progress or (False, False, 1)
    model = {}
    if progress is not None and not disabled:
        model = dict(
            expert_model_parallel_size=2,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_router_dtype="fp32",
            transformer_impl="inference_optimized",
            inference_grouped_gemm_backend="vllm",
            inference_moe_token_dispatcher_type="nvls",
            activation_func=squared_relu,
            add_bias_linear=False,
        )
    case = bi.Case(
        "live-coordinator",
        model=model,
        context=dict(
            enable_prefix_caching=True,
            prefix_caching_coordinator_policy=policy,
            prefix_caching_routing_alpha=0.01,
            prefix_cache_ttl_seconds=300.0,
        ),
    )
    if progress is not None:
        case.context.update(
            use_synchronous_zmq_collectives=synchronous,
            disable_ep_consensus=disabled,
            ep_consensus_interval=interval,
        )
    client = None
    with bi.invariant_runtime(case) as (backend, fa_version):
        rank = torch.distributed.get_rank()
        engine = bi.build_engine(case, backend, fa_version)
        allocator = engine.context.kv_block_allocator
        sync_context = zmq.Context()
        test_communicator = AsyncZMQCommunicator(sync_context, process_group=None)
        try:
            address = await engine.start_listening_to_data_parallel_coordinator()
            prompt = bi.target_prompt(258)
            await _sync(test_communicator)
            if rank == 0:
                client = InferenceClient(
                    address,
                    deserialize=True,
                    block_size_tokens=engine.context.block_size_tokens,
                    prefix_caching_coordinator_policy=policy,
                )
                frame_observer = _FrameObserver(client.socket)
                client.socket = frame_observer
                client.start(connect_timeout_seconds=10)
            if rank == 0:
                await asyncio.wait_for(
                    client.add_request(prompt, _params(num_tokens_to_generate=1)), 30
                )
            await _sync(test_communicator)
            cache_owner = await _sync(test_communicator, rank if allocator.get_total_used() else -1)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(bi, "TARGET", 1)
                baseline_witness = bi.ForwardWitness(engine, patch)
                if rank == 0:
                    baseline_result = _stream_result(
                        await _collect(client.add_request_streaming(prompt, _params()))
                    )
                await _sync(test_communicator)
            owner = await _sync(test_communicator, rank if baseline_witness.steps else -1)
            assert cache_owner == owner

            if rank == 0:
                client.pause_engines()
            await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), 30)
            steps_before = engine.context.step_count
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(bi, "TARGET", 2)
                joint_witness = bi.ForwardWitness(engine, patch)
                ep_calls, ep_sync = [], []
                if progress is not None:
                    ctx = engine.context
                    for module, name, tensor in (
                        (dispatch, "multimem_all_gatherv_3tensor", 3),
                        (moe_bi, "ordered_reduce_scatter_v", 0),
                    ):
                        original = getattr(module, name)

                        def collective(*args, fn=original, name=name, tensor=tensor, **kwargs):
                            result = fn(*args, **kwargs)
                            target = 2 in engine.requests and not ctx._bookkeeping_no_real_work
                            target = target and 2 in ctx.request_ids[: ctx.total_request_count]
                            counter = engine._ep_consensus_loop_counter
                            real = ctx.gpu_view.real_token_count.clone()
                            routing = args[4].clone() if tensor == 3 else None
                            ep_calls.append(
                                (name, counter, target, args[tensor].shape[0], real, routing)
                            )
                            return result

                        patch.setattr(module, name, collective)
                    communicator, name = (
                        (engine, "_ep_establish_consensus")
                        if disabled
                        else (engine.expert_parallel_zmq_communicator, "all_reduce_max")
                    )
                    consensus = getattr(communicator, name)

                    async def reduce(*args, **kwargs):
                        result = await consensus(*args, **kwargs)
                        ep_sync.append((engine._ep_consensus_loop_counter, kwargs))
                        return result

                    patch.setattr(communicator, name, reduce)
                admitted = []
                admitted_events = {request_id: asyncio.Event() for request_id in (2, 3, 67)}
                original_add = engine.add_request

                @wraps(original_add)
                def observe_add(request_id, *args, **kwargs):
                    result = original_add(request_id, *args, **kwargs)
                    admitted.append(request_id)
                    if request_id in admitted_events:
                        admitted_events[request_id].set()
                    return result

                patch.setattr(engine, "add_request", observe_add)
                if rank == 0:
                    target_stream = client.add_request_streaming(prompt, _params())
                if rank == owner:
                    await asyncio.wait_for(admitted_events[2].wait(), 30)
                await _sync(test_communicator)
                if rank == 0:
                    decoy_future = client.add_request([23], _params(num_tokens_to_generate=1))
                if rank != owner:
                    await asyncio.wait_for(admitted_events[3].wait(), 30)
                await _sync(test_communicator)
                if rank == 0:
                    neighbor_futures = [client.add_request(prompt, _params()) for _ in range(64)]
                if rank == owner:
                    await asyncio.wait_for(admitted_events[67].wait(), 30)
                await _sync(test_communicator)
                if rank == 0:
                    client.unpause_engines()
                await asyncio.wait_for(engine.wait_until(EngineState.RUNNING), 30)
                if rank == 0:
                    results = await asyncio.wait_for(
                        asyncio.gather(_collect(target_stream), decoy_future, *neighbor_futures), 60
                    )
                    joint_result = _stream_result(results[0])
                await _sync(test_communicator)

            if progress is not None:
                if rank == 0:
                    client.pause_engines()
                await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), 30)
                ep_calls = [
                    tuple(value.cpu() if torch.is_tensor(value) else value for value in call)
                    for call in ep_calls
                ]
                all_ep = [None, None]
                torch.distributed.all_gather_object(all_ep, (ep_calls, ep_sync))
                if disabled:
                    assert not ep_calls and not ep_sync
                    assert any(
                        s["physical"] == 64 and s["attention"] and s["gemms"]
                        for s in joint_witness.dummy_steps
                    ), "missing actual disabled-consensus dummy model"
                wide = [
                    (i, call)
                    for i, call in enumerate(all_ep[owner][0])
                    if call[2] and call[3] == 128 and call[4] == 65
                ]
                assert {call[0] for _, call in wide} == (
                    set()
                    if disabled
                    else {"multimem_all_gatherv_3tensor", "ordered_reduce_scatter_v"}
                )
                for index, call in wide:
                    peer = all_ep[owner ^ 1][0][index]
                    assert peer[0] == call[0] and not peer[2] and peer[3] == 64 and peer[4] == 0
                    if peer[5] is not None:
                        assert (peer[5] == -1).all(), "idle peer routed a real token"
                for _, sync in all_ep:
                    assert bool(sync) != disabled
                    assert all(kwargs["async_op"] != synchronous for _, kwargs in sync)
                if not disabled:
                    seen = {counter for counter, _ in all_ep[owner][1]}
                    assert {c[1] - 1 in seen for _, c in wide} == {True, interval == 1}

            assert engine.context.step_count > steps_before
            assert admitted == ([2, *range(4, 68)] if rank == owner else [3])
            assert engine.context.total_request_count == 0
            assert allocator.get_active_used() == 0
            if rank == 0:
                assert baseline_result.generated_tokens == joint_result.generated_tokens
                for request_id in (1, 2):
                    assert _headers(frame_observer, request_id) == [
                        Headers.ENGINE_REPLY_PARTIAL,
                        Headers.ENGINE_REPLY_PARTIAL,
                        Headers.ENGINE_REPLY,
                    ]
            if rank == owner:
                baseline_witness.assert_active(fa_version)
                joint_witness.assert_active(fa_version)
                assert any(
                    step["physical"] == 64 and step["requests"] == 1
                    for step in baseline_witness.steps
                )
                assert min(int(s["positions"].min()) for s in baseline_witness.steps) >= 256
                positions = [int(step["positions"].min()) for step in joint_witness.steps]
                assert min(positions) >= 256, positions
                assert any(
                    step["physical"] == 128 and step["requests"] == 65
                    for step in joint_witness.steps
                ), [(s["logical"], s["physical"], s["requests"]) for s in joint_witness.steps]
                pair = (None, baseline_witness), (None, joint_witness)
                bi.assert_same_target(*pair, require_trajectory=False)
                saved = joint_witness.steps[0]["logits"]
                joint_witness.steps[0]["logits"] = saved + 1
                with pytest.raises(AssertionError, match="target logits changed"):
                    bi.assert_same_target(*pair, require_trajectory=False)
                joint_witness.steps[0]["logits"] = saved
        finally:
            await _shutdown(engine, client, address, test_communicator, sync_context, rank)


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ttl", "load", "load-balanced", "media-fresh", "media-stale"])
async def test_live_routing_cold_history_and_media_epoch(mode):
    media = mode.startswith("media")
    policy = Policy.LOAD_BALANCED if mode == "load-balanced" else Policy.LONGEST_PREFIX
    case = bi.Case(
        mode,
        context=dict(
            enable_prefix_caching=not media,
            prefix_caching_coordinator_policy=policy,
            prefix_cache_ttl_seconds=0.05 if mode == "ttl" else 300.0,
            prefix_caching_routing_alpha=2.0 if mode == "load" else 0.01,
            allow_stale_multimodal_embeddings=mode == "media-stale",
            max_tokens=512,
        ),
    )
    client = None
    with bi.invariant_runtime(case) as (backend, version):
        rank = torch.distributed.get_rank()
        engine = (
            vlm._build_engine(case, backend, version, modality="image", engines=[], wire=True)
            if media
            else bi.build_engine(case, backend, version)
        )
        context = zmq.Context()
        comm = AsyncZMQCommunicator(context, process_group=None)
        try:
            async with asyncio.timeout(180):
                weights = hashlib.sha256()
                for parameter in engine.controller.inference_wrapped_model.model.parameters():
                    weights.update(parameter.detach().cpu().view(torch.uint8).numpy().tobytes())
                digests = [None, None]
                torch.distributed.all_gather_object(digests, weights.hexdigest())
                assert len(set(digests)) == 1, "cross-owner model weights differ"
                address = await engine.start_listening_to_data_parallel_coordinator()
                prompt = bi.target_prompt(17 if media else 258)
                if media:
                    prompt[3] = vlm.MEDIA_TOKEN
                payload = engine._bi_wire_media if media else None
                await _sync(comm)
                if rank == 0:
                    client = InferenceClient(
                        address,
                        deserialize=True,
                        block_size_tokens=engine.context.block_size_tokens,
                        prefix_caching_coordinator_policy=policy,
                    )
                    client.start(connect_timeout_seconds=10)
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(bi, "TARGET", 0)
                    baseline = bi.ForwardWitness(engine, patch)
                    if rank == 0:
                        reference = await client.add_request(
                            prompt, _params(), multi_modal_data=payload
                        )
                    await _sync(comm)
                owner = await _sync(comm, rank if baseline.steps else -1)
                assert owner == 0
                if rank == owner:
                    baseline.assert_active(version)
                    assert {s["physical"] for s in baseline.steps if s["decode"]} == {64}
                for step in baseline.steps:
                    for key in ("positions", "tokens", "logits"):
                        step[key] = step[key].cpu()
                baseline_steps = [None, None]
                torch.distributed.all_gather_object(baseline_steps, baseline.steps)
                if rank == 0:
                    client.pause_engines()
                await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), 30)
                if mode == "ttl":
                    await asyncio.sleep(0.1)
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(bi, "TARGET", 2)
                    witness = bi.ForwardWitness(engine, patch)
                    admitted = []
                    original = engine.add_request

                    def observe(request_id, *args, **kwargs):
                        result = original(request_id, *args, **kwargs)
                        admitted.append(request_id)
                        return result

                    patch.setattr(engine, "add_request", observe)
                    if rank == 0:
                        trigger = client.add_request([23] * (17 if media else 258), _params())
                    while await _sync(comm, len(admitted)) < 1:
                        await asyncio.sleep(0.01)
                    if rank == 0:
                        target = client.add_request(prompt, _params(), multi_modal_data=payload)
                    while await _sync(comm, int(2 in admitted)) < 1:
                        await asyncio.sleep(0.01)
                    target_owner = await _sync(comm, rank if 2 in admitted else -1)
                    assert target_owner == (owner if media else owner ^ 1), (mode, admitted)
                    if rank == owner and not media:
                        assert engine.context.kv_block_allocator.get_total_used() > 0
                    if rank == 0:
                        neighbors = [client.add_request([23], _params()) for _ in range(128)]
                    while await _sync(comm, -len(admitted)) != -65:
                        await asyncio.sleep(0.01)
                    assert len(admitted) == 65
                    if media:
                        before = len(engine._bi_media_evidence)
                        if rank == 0:
                            client.set_generation_epoch(1)
                        while engine._generation_epoch != 1:
                            await asyncio.sleep(0.01)
                        if rank == owner:
                            assert before == 1
                            assert len(engine._bi_media_evidence) == 1 + (mode == "media-fresh")
                    await _sync(comm)
                    if rank == 0:
                        client.unpause_engines()
                        results = await asyncio.wait_for(
                            asyncio.gather(target, trigger, *neighbors), 60
                        )
                        assert results[0].generated_tokens == reference.generated_tokens
                        if media:
                            assert [tuple(x) for x in results[0].policy_epoch] == [(0, 1)]
                    await _sync(comm)
                if rank == target_owner:
                    witness.assert_active(version)
                    for step in witness.steps:
                        for key in ("positions", "tokens", "logits"):
                            step[key] = step[key].cpu()
                    assert any(s["physical"] == 128 and s["requests"] == 65 for s in witness.steps)
                    assert min(int(s["positions"].min()) for s in witness.steps) == 0
                    bi.assert_same_target(
                        (None, SimpleNamespace(steps=baseline_steps[owner])),
                        (None, witness),
                        require_trajectory=False,
                    )
                    if media:
                        assert len(engine._bi_decoder_inputs) == 2
                        projected = engine._bi_media_evidence[0]["projected"].squeeze(1)
                        for embeddings, mask in engine._bi_decoder_inputs:
                            vlm._assert_projected(projected, embeddings[mask >= 0])
        finally:
            await _shutdown(engine, client, address, comm, context, rank)
