from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest

pytest.importorskip("dynamo")

from megatron.core.inference.engine_endpoint import InferenceEngineEndpoint
from megatron.core.inference.headers import Headers
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
            },
        }
    )


def test_sampling_params_maps_greedy_and_limits():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {"temperature": 0.0, "top_p": 0.9},
            "stop_conditions": {"max_tokens": 7},
        }
    )
    assert params.top_k == 1
    assert params.top_p == 0.0
    assert params.num_tokens_to_generate == 7

    selected_logprobs = build_sampling_params({"token_ids": [1], "output_options": {"logprobs": 0}})
    assert selected_logprobs.return_log_probs
    assert selected_logprobs.top_n_logprobs == 0
    assert selected_logprobs.skip_prompt_log_probs

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

        client_class.assert_called_once_with("tcp://127.0.0.1:5555", deserialize=False)
        client.start.assert_called_once()
        receiver_class.assert_called_once_with(engine._on_engine_event, "127.0.0.1")
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


@pytest.mark.asyncio
async def test_decode_health_probe_bypasses_kv_handoff():
    handoff_called = False

    def add_request_with_kv_handoff(*_args, **_kwargs):
        nonlocal handoff_called
        handoff_called = True
        raise AssertionError("health probe must not import KV")

    engine = MegatronLLMEngine(_config("decode"))
    engine.client = SimpleNamespace(
        add_request_streaming=lambda *_args, **_kwargs: pytest.fail(
            "decode health probe must not enter the model engine"
        ),
        add_request_with_kv_handoff=add_request_with_kv_handoff,
    )
    request = {
        "token_ids": [1],
        "_HEALTH_CHECK": True,
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 1},
    }

    chunks = [chunk async for chunk in engine.generate(request, _Context())]

    assert chunks[-1]["token_ids"] == []
    assert chunks[-1]["finish_reason"] == "stop"
    assert not handoff_called


@pytest.mark.asyncio
async def test_prefill_release_uses_registered_engine_endpoint():
    class Stream:
        request_id = 36
        done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return {
                "final": {
                    "request_id": self.request_id,
                    "disaggregated_params": {"block_ids": [4, 5]},
                }
            }

        async def aclose(self):
            return None

    engine = MegatronLLMEngine(_config("prefill"))
    engine._engine_endpoint = _endpoint("tcp://prefill:5000")
    engine.client = SimpleNamespace(add_request_streaming=lambda *_args, **_kwargs: Stream())
    request = {"token_ids": [1], "sampling_options": {}, "stop_conditions": {"max_tokens": 1}}

    chunks = [chunk async for chunk in engine.generate(request, _Context())]

    assert chunks[-1]["disaggregated_params"]["release"] == {
        "coordinator_addr": "tcp://prefill:5000",
        "request_id": 36,
    }


@pytest.mark.asyncio
async def test_decode_uses_streaming_kv_handoff():
    class Stream:
        request_id = 37
        done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return {"final": {"generated_tokens": [9]}}

        async def aclose(self):
            return None

    handoff = MagicMock(return_value=Stream())
    engine = MegatronLLMEngine(_config("decode"))
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
    class Stream:
        request_id = 41

        def __init__(self):
            self.items = iter(
                [
                    {"partial": {"new_tokens": [7], "new_log_probs": [-0.7]}},
                    {
                        "final": {
                            "generated_tokens": ["tensor", [7, 8]],
                            "generated_log_probs": ["tensor", [-0.7, -0.8]],
                        }
                    },
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration as error:
                raise StopAsyncIteration from error

        async def aclose(self):
            return None

    engine = MegatronLLMEngine(_config())
    engine.client = SimpleNamespace(add_request_streaming=lambda *_args, **_kwargs: Stream())
    request = {
        "token_ids": [1],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 2},
        "output_options": {"logprobs": 0},
    }

    chunks = [chunk async for chunk in engine.generate(request, _Context())]

    assert chunks == [
        {"token_ids": [7], "index": 0, "log_probs": [-0.7]},
        {
            "token_ids": [8],
            "index": 0,
            "finish_reason": "length",
            "completion_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "log_probs": [-0.8],
        },
    ]


@pytest.mark.asyncio
async def test_streaming_generation_rejects_misaligned_log_probs():
    class Stream:
        request_id = 42

        def __aiter__(self):
            return self

        async def __anext__(self):
            if getattr(self, "done", False):
                raise StopAsyncIteration
            self.done = True
            return {"partial": {"new_tokens": [7], "new_log_probs": []}}

        async def aclose(self):
            return None

    engine = MegatronLLMEngine(_config())
    engine.client = SimpleNamespace(add_request_streaming=lambda *_args, **_kwargs: Stream())
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
    aborted = []

    def abort_request_and_wait(request_id):
        assert request_id == 77
        aborted.append(request_id)
        future = asyncio.get_running_loop().create_future()
        future.set_result(True)
        return future

    engine = MegatronLLMEngine(_config())
    engine.client = SimpleNamespace(abort_request_and_wait=abort_request_and_wait)
    engine._request_ids["dynamo-request"] = 77

    await engine.abort(_Context())

    assert aborted == [77]


@pytest.mark.asyncio
async def test_release_handoff_reuses_async_socket():
    engine = MegatronLLMEngine(_config("decode"))
    release = {"coordinator_addr": "tcp://prefill:5000", "request_id": 7}
    socket = SimpleNamespace(
        setsockopt=MagicMock(),
        connect=MagicMock(),
        send=AsyncMock(),
        recv=AsyncMock(return_value=msgpack.packb([Headers.CONNECT_ACK.value])),
        close=MagicMock(),
    )
    context = SimpleNamespace(socket=MagicMock(return_value=socket), term=MagicMock())
    engine._release_context = context

    assert await engine._release_handoff_from_meta_async(release)
    release["request_id"] = 8
    assert await engine._release_handoff_from_meta_async(release)
    await engine.cleanup()

    context.socket.assert_called_once()
    assert socket.send.await_count == 3
    socket.close.assert_called_once_with(linger=0)
