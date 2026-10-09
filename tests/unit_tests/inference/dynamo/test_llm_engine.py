# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
import zmq
import zmq.asyncio

pytest.importorskip("dynamo")

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.data_parallel_inference_coordinator.handlers import HANDLERS
from megatron.core.inference.disaggregation.handoff_ownership import HandoffOwnership
from megatron.core.inference.engine_endpoint import InferenceEngineEndpoint
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceRequestError
from megatron.inference.integrations.dynamo.args import Config
from megatron.inference.integrations.dynamo.llm_engine import (
    MegatronLLMEngine,
    build_sampling_params,
)


def _config(role="aggregated"):
    return Config(
        model="model",
        served_model_name="served",
        namespace="dynamo",
        component="prefill" if role == "prefill" else "backend",
        endpoint="generate",
        discovery_backend="etcd",
        request_plane="nats",
        event_plane="nats",
        role=role,
        nproc_per_node=1,
        coordinator_host=None,
        coordinator_port=None,
        worker_id_file=None,
        megatron_root="/opt/megatron-lm",
        drain_timeout=0.1,
        megatron_argv=["--load", "/checkpoint"],
    )


def _endpoint(address="tcp://127.0.0.1:5555"):
    return InferenceEngineEndpoint.from_dict(
        {
            "coordinator_address": address,
            "capabilities": {
                "context_length": 8192,
                "kv_cache_block_size": 64,
                "total_kv_blocks": 100,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 4096,
                "bos_token_id": 1,
                "enable_prefix_caching": True,
                "logical_data_parallel_size": 1,
                "prefix_caching_coordinator_policy": "first_prefix_block",
            },
        }
    )


def test_sampling_params_maps_greedy_and_limits():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "include_stop_str_in_output": True,
            },
            "stop_conditions": {"max_tokens": 7, "min_tokens": 0, "stop": ["END"]},
        }
    )
    assert params.top_k == 1
    assert params.top_p == 0.0
    assert params.num_tokens_to_generate == 7
    assert params.detokenize_stop_sequence
    assert params.stop_words == ["END"]
    assert not build_sampling_params({}).detokenize_stop_sequence
    with pytest.raises(ValueError, match="nonzero min_tokens"):
        build_sampling_params({"stop_conditions": {"min_tokens": 2}})

    selected_logprobs = build_sampling_params({"token_ids": [1], "output_options": {"logprobs": 0}})
    assert selected_logprobs.return_log_probs
    assert selected_logprobs.top_n_logprobs == 0
    assert selected_logprobs.skip_prompt_log_probs
    assert not selected_logprobs.detokenize_generations

    with pytest.raises(ValueError, match="selected-token logprobs only"):
        build_sampling_params({"token_ids": [1], "output_options": {"logprobs": 1}})

    with pytest.raises(ValueError, match="does not support prompt_logprobs"):
        build_sampling_params({"token_ids": [1], "output_options": {"prompt_logprobs": 0}})


@pytest.mark.asyncio
async def test_start_uses_parent_event_socket_and_base_client(tmp_path):
    config = _config()
    config.megatron_root = str(tmp_path)
    engine = MegatronLLMEngine(config, registration_model="/cache/model-metadata")
    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    stdout.feed_eof()
    stderr.feed_eof()
    process_exit = asyncio.Event()

    async def wait_for_process():
        await process_exit.wait()
        return 0

    process = SimpleNamespace(stdout=stdout, stderr=stderr, returncode=None, wait=wait_for_process)
    client = SimpleNamespace(start=MagicMock())
    event_receiver = SimpleNamespace(start=MagicMock(return_value="tcp://127.0.0.1:5556"))
    engine._wait_for_readiness = AsyncMock(return_value=_endpoint().to_dict())

    try:
        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)),
            patch(
                "megatron.inference.integrations.dynamo.llm_engine.InferenceClient",
                return_value=client,
            ) as client_class,
            patch(
                "megatron.inference.integrations.dynamo.llm_engine.EngineEventReceiver",
                return_value=event_receiver,
            ) as receiver_class,
        ):
            engine_config = await engine.start(worker_id=0)

        client_class.assert_called_once_with(
            "tcp://127.0.0.1:5555",
            deserialize=False,
            block_size_tokens=64,
            prefix_caching_coordinator_policy=PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK,
        )
        client.start.assert_called_once()
        receiver_class.assert_called_once_with(engine._on_engine_event, "127.0.0.1", bind_port=None)
        event_receiver.start.assert_called_once()
        assert engine_config.model == "/cache/model-metadata"
        assert engine_config.served_model_name == "served"
        assert engine_config.llm.context_length == 8192
        assert engine_config.llm.kv_cache_block_size == 64
        assert engine_config.llm.total_kv_blocks == 100
    finally:
        if engine._process_monitor is not None:
            engine._process_monitor.cancel()
            await asyncio.gather(engine._process_monitor, return_exceptions=True)
        await asyncio.gather(*engine._log_tasks)


@pytest.mark.asyncio
async def test_external_start_waits_without_launching_subprocess(tmp_path):
    config = _config()
    config.engine_launch_mode = "external"
    config.parent_event_host = "node-0"
    config.parent_event_port = 5556
    config.nproc_per_node = None
    config.megatron_root = str(tmp_path / "not-mounted-in-parent")
    engine = MegatronLLMEngine(config)
    client = SimpleNamespace(start=MagicMock())
    event_receiver = SimpleNamespace(start=MagicMock(return_value="tcp://node-0:5556"))
    engine._wait_for_readiness = AsyncMock(return_value=_endpoint().to_dict())
    create_subprocess = AsyncMock()

    with (
        patch("asyncio.create_subprocess_exec", create_subprocess),
        patch(
            "megatron.inference.integrations.dynamo.llm_engine.InferenceClient", return_value=client
        ),
        patch(
            "megatron.inference.integrations.dynamo.llm_engine.EngineEventReceiver",
            return_value=event_receiver,
        ) as receiver_class,
    ):
        await engine.start(worker_id=0)

    receiver_class.assert_called_once_with(engine._on_engine_event, "node-0", bind_port=5556)
    create_subprocess.assert_not_awaited()
    assert engine._process is None
    assert engine._process_monitor is None
    assert not engine._log_tasks


@pytest.mark.asyncio
async def test_start_cleans_up_after_readiness_failure(tmp_path):
    config = _config()
    config.megatron_root = str(tmp_path)
    engine = MegatronLLMEngine(config)
    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    stdout.feed_eof()
    stderr.feed_eof()
    process = SimpleNamespace(stdout=stdout, stderr=stderr, returncode=None)
    event_receiver = SimpleNamespace(start=MagicMock(return_value="tcp://127.0.0.1:5556"))
    engine._wait_for_readiness = AsyncMock(side_effect=TimeoutError("not ready"))
    engine.cleanup = AsyncMock()

    with (
        patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)),
        patch(
            "megatron.inference.integrations.dynamo.llm_engine.EngineEventReceiver",
            return_value=event_receiver,
        ),
        pytest.raises(TimeoutError, match="not ready"),
    ):
        await engine.start(worker_id=0)

    engine.cleanup.assert_awaited_once_with()
    await asyncio.gather(*engine._log_tasks)


@pytest.mark.asyncio
async def test_start_cleans_up_when_process_launch_is_cancelled(tmp_path):
    config = _config()
    config.megatron_root = str(tmp_path)
    engine = MegatronLLMEngine(config)
    event_receiver = SimpleNamespace(start=MagicMock(return_value="tcp://127.0.0.1:5556"))
    engine.cleanup = AsyncMock()

    with (
        patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=asyncio.CancelledError)),
        patch(
            "megatron.inference.integrations.dynamo.llm_engine.EngineEventReceiver",
            return_value=event_receiver,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await engine.start(worker_id=0)

    engine.cleanup.assert_awaited_once_with()


class _Context:
    def __init__(self, request_id="dynamo-request"):
        self.request_id = request_id

    def id(self):
        return self.request_id


def _stream(*items, request_id=1):
    stream = AsyncStream(request_id, MagicMock())
    for item in items:
        stream.put(item)
    stream.finish()
    return stream


@pytest.mark.asyncio
@pytest.mark.parametrize("responsive", [True, False])
async def test_decode_health_probe_checks_engine_without_handoff(responsive):
    engine = MegatronLLMEngine(_config("decode"))
    engine.config.engine_launch_mode = "external"
    engine.client = SimpleNamespace(
        add_request_streaming=MagicMock(side_effect=AssertionError),
        add_request_with_kv_handoff_streaming=MagicMock(side_effect=AssertionError),
    )
    request = {
        "token_ids": [1],
        "_HEALTH_CHECK": True,
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 1},
    }

    # A heartbeat from before the probe is not sufficient.
    engine._on_engine_event("progress", {})
    with patch("megatron.inference.integrations.dynamo.llm_engine._HEALTH_CHECK_TIMEOUT", 0.15):
        generation = engine.generate(request, _Context())
        probe = asyncio.create_task(anext(generation))
        await asyncio.sleep(0)
        if responsive:
            engine._on_engine_event("progress", {})
            assert (await probe)["finish_reason"] == "stop"
        else:
            with pytest.raises(TimeoutError):
                await probe
        await generation.aclose()
    engine.client.add_request_streaming.assert_not_called()
    engine.client.add_request_with_kv_handoff_streaming.assert_not_called()


@pytest.mark.asyncio
async def test_prefill_release_uses_registered_engine_endpoint():
    stream = _stream(
        {"final": {"request_id": 36, "disaggregated_params": {"block_ids": [4, 5]}}}, request_id=36
    )
    engine = MegatronLLMEngine(_config("prefill"))
    engine._engine_endpoint = _endpoint("tcp://prefill:5000")
    engine.client = SimpleNamespace(
        add_request_streaming=MagicMock(return_value=stream),
        coordinator_instance_id="prefill-instance",
    )
    request = {"token_ids": [1], "sampling_options": {}, "stop_conditions": {"max_tokens": 1}}

    chunks = [chunk async for chunk in engine.generate(request, _Context())]

    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["disaggregated_params"]["release"] == {
        "coordinator_addr": "tcp://prefill:5000",
        "coordinator_instance_id": "prefill-instance",
        "request_id": 36,
    }


@pytest.mark.asyncio
async def test_decode_uses_streaming_kv_handoff():
    handoff = MagicMock(return_value=_stream({"final": {"generated_tokens": [9]}}))
    engine = MegatronLLMEngine(_config("decode"))
    engine._claim_remote_handoff = AsyncMock()
    engine._release_completed_handoff = AsyncMock()
    engine.client = SimpleNamespace(add_request_with_kv_handoff_streaming=handoff)
    request = {"token_ids": [1], "sampling_options": {}, "stop_conditions": {"max_tokens": 1}}
    prefill = {"disaggregated_params": {"kv_meta": {"peer": "prefill"}, "block_ids": [4, 5]}}

    with patch(
        "megatron.inference.integrations.dynamo.llm_engine.require_prefill_result",
        return_value=prefill,
    ):
        chunks = [chunk async for chunk in engine.generate(request, _Context())]

    assert chunks[-1]["token_ids"] == [9]
    assert chunks[-1]["finish_reason"] == "length"
    assert handoff.call_args.args[2:] == ({"peer": "prefill"}, [4, 5])


@pytest.mark.asyncio
async def test_streaming_generation_forwards_selected_log_probs():
    stream = _stream(
        {"partial": {"new_tokens": [7], "new_log_probs": [-0.7]}},
        {
            "final": {
                "generated_tokens": ["tensor", [7, 8]],
                "generated_log_probs": ["tensor", [-0.7, -0.8]],
            }
        },
    )
    engine = MegatronLLMEngine(_config())
    engine.client = SimpleNamespace(add_request_streaming=MagicMock(return_value=stream))
    request = {
        "token_ids": [1],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 2},
        "output_options": {"logprobs": 0},
    }

    chunks = [chunk async for chunk in engine.generate(request, _Context())]

    assert [
        {key: value for key, value in chunk.items() if key != "log_probs"} for chunk in chunks
    ] == [
        {"token_ids": [7], "index": 0},
        {
            "token_ids": [8],
            "index": 0,
            "finish_reason": "length",
            "completion_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    ]
    assert [chunk["log_probs"][0] for chunk in chunks] == pytest.approx([-0.7, -0.8])


@pytest.mark.asyncio
async def test_streaming_generation_rejects_misaligned_log_probs():
    stream = _stream({"partial": {"new_tokens": [7], "new_log_probs": []}})
    engine = MegatronLLMEngine(_config())
    engine.client = SimpleNamespace(add_request_streaming=MagicMock(return_value=stream))
    request = {
        "token_ids": [1],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 1},
        "output_options": {"logprobs": 0},
    }

    with pytest.raises(RuntimeError, match="0 selected log probabilities for 1"):
        _ = [chunk async for chunk in engine.generate(request, _Context())]


@pytest.mark.asyncio
async def test_abort_uses_megatron_request_id_recorded_for_context():
    engine = MegatronLLMEngine(_config())
    engine.client = SimpleNamespace(abort_request_and_wait=AsyncMock(return_value=True))
    engine._request_ids["dynamo-request"] = 77

    await engine.abort(_Context())

    engine.client.abort_request_and_wait.assert_awaited_once_with(77)


@pytest.mark.asyncio
async def test_release_reregisters_after_coordinator_replacement():
    context = zmq.asyncio.Context()
    engines = [MegatronLLMEngine(_config("decode")) for _ in range(2)]
    address = None
    try:
        for instance_id in ("original", "replacement"):
            router = context.socket(zmq.ROUTER)
            if address is None:
                port = router.bind_to_random_port("tcp://127.0.0.1")
                address = f"tcp://127.0.0.1:{port}"
            else:
                router.bind(address)
            coordinator = SimpleNamespace(
                instance_id=instance_id,
                known_clients=set(),
                router_socket=router,
                _broadcast_to_engines=MagicMock(),
                handoff_ownership=HandoffOwnership(),
            )

            async def serve():
                while True:
                    identity, payload = await router.recv_multipart()
                    metadata = msgpack.unpackb(payload, raw=False)
                    HANDLERS[Headers(metadata[0])](coordinator, identity, metadata, [])

            server = asyncio.create_task(serve())
            try:
                for engine in engines:
                    await engine._release_remote_handoff(address, 7, instance_id)
                    socket = engine._release_sockets[address]
                    # Repeated CONNECT is also acknowledged by the same coordinator.
                    await engine._release_remote_handoff(address, 8, instance_id)
                    assert engine._release_sockets[address] is socket
                assert coordinator._broadcast_to_engines.call_count == 4
                assert len(coordinator.known_clients) == 2
                if instance_id == "replacement":
                    # Request 7 may be reused. An old generation's cleanup must
                    # not touch it, including a restart between CONNECT and RELEASE.
                    await engine._release_remote_handoff(address, 7, "original")
                    await engine._release_sockets[address].send(
                        msgpack.packb([Headers.RELEASE_KV.value, 7, "original"])
                    )
                    reply = msgpack.unpackb(
                        await engine._release_sockets[address].recv(), raw=False
                    )
                    assert reply == [Headers.RELEASE_KV_ACK.value, 7, "original"]
                    assert coordinator._broadcast_to_engines.call_count == 4
            finally:
                server.cancel()
                await asyncio.gather(server, return_exceptions=True)
                router.close(linger=0)
    finally:
        for engine in engines:
            sockets = list(engine._release_sockets.values())
            await engine.cleanup()
            assert all(socket.closed for socket in sockets)
        context.term()


@pytest.mark.asyncio
async def test_release_requires_acceptance_and_discards_timed_out_socket():
    engine = MegatronLLMEngine(_config("decode"))

    async def no_ack():
        await asyncio.Future()

    socket = MagicMock(send=AsyncMock())
    replies = iter([msgpack.packb([Headers.CONNECT_ACK.value, "instance"]), None])

    async def recv():
        reply = next(replies)
        return reply if reply is not None else await no_ack()

    socket.recv = recv
    engine._release_context = MagicMock(socket=MagicMock(return_value=socket))
    with patch("megatron.inference.integrations.dynamo.llm_engine._RELEASE_TIMEOUT", 0.05):
        with pytest.raises(TimeoutError):
            await engine._release_remote_handoff("tcp://prefill:5000", 7, "instance")
    assert not engine._release_sockets
    socket.close.assert_called_once_with(linger=0)
    engine._release_remote_handoff = AsyncMock(side_effect=[TimeoutError, None])
    assert await engine._release_handoff_from_meta_async(
        {
            "coordinator_addr": "tcp://prefill:5000",
            "request_id": 7,
            "coordinator_instance_id": "instance",
        }
    )
    assert engine._release_remote_handoff.await_count == 2
    await engine.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["consumer", "abort", "completed_abort"])
async def test_cancelled_prefill_releases_state_after_final_reply(cancellation):
    cancel = MagicMock()
    stream = AsyncStream(36, cancel)
    engine = MegatronLLMEngine(_config("prefill"))
    engine._engine_endpoint = _endpoint()
    engine.client = SimpleNamespace(
        add_request_streaming=MagicMock(return_value=stream), release_handoff=MagicMock()
    )
    generation = engine.generate({"token_ids": [1]}, _Context())
    consumer = asyncio.create_task(anext(generation))
    await asyncio.sleep(0)
    if cancellation == "completed_abort":
        # The shield can already have a result while its consumer has not yet
        # resumed. Cancelling a done future alone does not cancel that consumer.
        engine._request_waiters["dynamo-request"].set_result(
            {"status": "COMPLETED", "disaggregated_params": {"request_id": 900}}
        )
    if cancellation != "consumer":
        await engine.abort(_Context())
    else:
        consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    cancel.assert_not_called()
    engine.client.release_handoff.assert_not_called()
    # Engine request IDs differ from client stream IDs. The late final result
    # carries the identity needed to release both KV blocks and the SSM slot.
    stream.put({"final": {"status": "COMPLETED", "disaggregated_params": {"request_id": 900}}})
    stream.finish()
    await asyncio.wait_for(asyncio.gather(*engine._cleanup_tasks), timeout=1)
    engine.client.release_handoff.assert_called_once_with(900)
    assert not engine._request_waiters
    assert not engine._request_ids
    assert not engine._cleanup_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
async def test_failed_final_reply_propagates_engine_error(role):
    stream = AsyncStream(1, MagicMock())
    stream.put(
        {
            "final": {
                "status": "FAILED",
                "events": [
                    {
                        "type": "ERROR_NONTRANSIENT",
                        "payload": {
                            "type": "TokenOverflowError",
                            "message": "token budget exceeded",
                        },
                    }
                ],
            }
        }
    )
    stream.finish()
    engine = MegatronLLMEngine(_config(role))
    engine._engine_endpoint = _endpoint()
    engine._claim_remote_handoff = AsyncMock()
    engine._release_completed_handoff = AsyncMock()
    engine.client = SimpleNamespace(
        add_request_streaming=MagicMock(return_value=stream),
        add_request_with_kv_handoff_streaming=MagicMock(return_value=stream),
        release_handoff=MagicMock(),
    )
    with patch(
        "megatron.inference.integrations.dynamo.llm_engine.require_prefill_result", return_value={}
    ):
        with pytest.raises(InferenceRequestError, match="token budget exceeded"):
            _ = [chunk async for chunk in engine.generate({"token_ids": [1]}, _Context())]
    await asyncio.gather(*engine._cleanup_tasks)
    engine.client.release_handoff.assert_not_called()


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
def test_kv_startup_buffer_only_keeps_events_for_publishing_roles(role):
    engine = MegatronLLMEngine(_config(role))
    engine._on_engine_event("ready", _endpoint().to_dict())
    engine._on_engine_event("removed", {"block_hashes": [1]})
    assert engine._ready_messages.get_nowait() == _endpoint().to_dict()
    if role == "decode":
        assert engine._kv_queue.empty()
    else:
        publisher = MagicMock()
        engine._set_publisher(publisher)
        publisher.publish_removed.assert_called_once_with(block_hashes=[1])
        assert engine._kv_queue.empty()


@pytest.mark.asyncio
async def test_source_release_failure_does_not_block_or_fail_decode(caplog):
    stream = AsyncStream(1, MagicMock())
    stream.put({"partial": {"new_tokens": [2]}})
    stream.put({"final": {"status": "COMPLETED", "generated_tokens": [2, 3]}})
    stream.finish()
    engine = MegatronLLMEngine(_config("decode"))
    engine._claim_remote_handoff = AsyncMock()
    engine.client = SimpleNamespace(add_request_with_kv_handoff_streaming=lambda *args: stream)
    release_started = asyncio.Event()
    release_failed = asyncio.Event()

    async def release(*args):
        release_started.set()
        await release_failed.wait()
        raise TimeoutError("source unavailable")

    engine._release_remote_handoff = release
    prefill = {
        "disaggregated_params": {
            "release": {
                "coordinator_addr": "tcp://prefill:5000",
                "request_id": 9,
                "coordinator_instance_id": "prefill-instance",
            }
        }
    }
    with patch(
        "megatron.inference.integrations.dynamo.llm_engine.require_prefill_result",
        return_value=prefill,
    ):
        generation = engine.generate({"token_ids": [1]}, _Context())
        assert (await anext(generation))["token_ids"] == [2]
        await asyncio.wait_for(release_started.wait(), timeout=1)
        assert (await asyncio.wait_for(anext(generation), timeout=1))["token_ids"] == [3]
        with pytest.raises(StopAsyncIteration):
            await anext(generation)
    release_failed.set()
    await asyncio.gather(*engine._cleanup_tasks, return_exceptions=True)
    assert "Megatron handoff cleanup failed" in caplog.text


@pytest.mark.asyncio
async def test_unreachable_release_source_does_not_block_other_sources():
    engine = MegatronLLMEngine(_config("decode"))
    connecting = asyncio.Event()

    async def stalled_recv():
        connecting.set()
        await asyncio.Future()

    stalled = MagicMock(send=AsyncMock(), recv=stalled_recv)
    healthy = MagicMock(
        send=AsyncMock(),
        recv=AsyncMock(
            side_effect=[
                msgpack.packb([Headers.CONNECT_ACK.value, "healthy"]),
                msgpack.packb([Headers.RELEASE_KV_ACK.value, 2, "healthy"]),
            ]
        ),
    )
    engine._release_context = MagicMock(socket=MagicMock(side_effect=[stalled, healthy]))
    with patch("megatron.inference.integrations.dynamo.llm_engine._RELEASE_TIMEOUT", 0.1):
        pending = asyncio.create_task(
            engine._release_remote_handoff("tcp://stalled:1", 1, "stalled")
        )
        await asyncio.wait_for(connecting.wait(), timeout=1)
        await engine._release_remote_handoff("tcp://healthy:2", 2, "healthy")
        assert not pending.done()
        with pytest.raises(TimeoutError):
            await pending
    stalled.close.assert_called_once_with(linger=0)
    assert list(engine._release_sockets) == ["tcp://healthy:2"]
    await engine.cleanup()


@pytest.mark.asyncio
async def test_shutdown_drains_cancelled_prefill_before_stopping_engine():
    stream = AsyncStream(1, MagicMock())
    engine = MegatronLLMEngine(_config("prefill"))
    engine._engine_endpoint = _endpoint()
    client = MagicMock(add_request_streaming=MagicMock(return_value=stream))
    engine.client = client
    generation = engine.generate({"token_ids": [1]}, _Context())
    consumer = asyncio.create_task(anext(generation))
    await asyncio.sleep(0)

    async def finish_prefill():
        await asyncio.sleep(0.01)
        stream.put({"final": {"disaggregated_params": {"request_id": 99}}})
        stream.finish()

    completion = asyncio.create_task(finish_prefill())
    await engine.cleanup()
    await completion
    with pytest.raises(asyncio.CancelledError):
        await consumer
    names = [call[0] for call in client.mock_calls]
    assert names.index("release_handoff") < names.index("stop_engines")
    client.release_handoff.assert_called_once_with(99)
    assert not engine._cleanup_tasks
