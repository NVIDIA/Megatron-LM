# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for configurable defaults on the dynamic text generation server."""

import asyncio
import os

import pytest

quart = pytest.importorskip("quart")
from quart import Quart

from megatron.core.inference.config import MultimodalPromptConfig
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    bp as chat_completions_blueprint,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.completions import (
    bp as completions_blueprint,
)


class _Tokenizer:
    chat_template = "test-template"
    bos = None

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return [10, 11]

    def tokenize(self, prompt):
        del prompt
        return [10, 11]


class _CapturingClient:
    def __init__(self):
        self.sampling_params = []
        self.offload_params = []

    def add_request_with_id(
        self, prompt_tokens, sampling_params, *, multi_modal_data=None, offload_params=None
    ):
        del prompt_tokens, multi_modal_data
        self.sampling_params.append(sampling_params)
        self.offload_params.append(offload_params)
        raise RuntimeError("stop after request submission")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "eval_mode",
        "request_overrides",
        "expected_temperature",
        "expected_top_p",
        "expected_top_k",
        "expected_prompt_tokens",
    ),
    [
        (False, {}, 0.7, 0.95, 20, True),
        (True, {}, 0.7, 0.95, 20, False),
        (False, {"prevent_retokenization": False}, 0.7, 0.95, 20, False),
        (False, {"temperature": 0.0}, 0.0, 0.0, 1, True),
        (True, {"temperature": None, "top_p": None, "top_k": None}, 0.7, 0.95, 20, False),
        (
            True,
            {"temperature": 0.4, "top_p": 0.8, "top_k": 5, "prevent_retokenization": True},
            0.4,
            0.8,
            5,
            True,
        ),
        (True, {"return_tokenized_data": True}, 0.7, 0.95, 20, True),
        (
            True,
            {
                "offload_params": {
                    "ng_capture": {
                        "mode": "text",
                        "rollout_id": "r0",
                        "model_call_id": "c1",
                        "prev_len": 0,
                    }
                }
            },
            0.7,
            0.95,
            20,
            False,
        ),
    ],
)
async def test_chat_request_uses_server_defaults(
    eval_mode,
    request_overrides,
    expected_temperature,
    expected_top_p,
    expected_top_k,
    expected_prompt_tokens,
):
    app = Quart(__name__)
    inference_client = _CapturingClient()
    app.config.update(
        client=inference_client,
        tokenizer=_Tokenizer(),
        parsers=[],
        verbose=False,
        multimodal_prompt_config=MultimodalPromptConfig(),
        default_temperature=0.7,
        default_top_p=0.95,
        default_top_k=20,
        eval_mode=eval_mode,
    )
    app.register_blueprint(chat_completions_blueprint)

    payload = {"messages": [{"role": "user", "content": "hello"}], **request_overrides}
    response = await app.test_client().post("/v1/chat/completions", json=payload)

    assert response.status_code == 500
    assert len(inference_client.sampling_params) == 1
    sampling_params = inference_client.sampling_params[0]
    assert sampling_params.temperature == expected_temperature
    assert sampling_params.top_p == expected_top_p
    assert sampling_params.top_k == expected_top_k
    assert sampling_params.return_prompt_tokens is expected_prompt_tokens
    assert inference_client.offload_params == [request_overrides.get("offload_params")]


def test_sampling_config_reaches_frontend_process(monkeypatch):
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        text_generation_server as server,
    )

    captured = {}

    async def fake_run_text_gen_server(*args):
        captured["run_args"] = args
        args[-1].send(True)  # the replica signals once it listens

    class FakeProcess:
        pid = 123

        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False
            self.sentinel = os.pipe()[0]  # never readable: the fake replica stays alive

        def start(self):
            self.started = True
            self.target(*self.args)

    class FakeSocket:
        def getsockname(self):
            return ("127.0.0.1", 4321)

        def close(self):
            captured["socket_closed"] = True

    class FakeProcessContext:
        Process = FakeProcess

    monkeypatch.setattr(server, "_SERVER_PROCESSES", [])
    monkeypatch.setattr(server, "_SERVER_PROCESS_CONTEXT", FakeProcessContext())
    monkeypatch.setattr(server, "_run_text_gen_server", fake_run_text_gen_server)
    monkeypatch.setattr(server.asyncio, "set_event_loop", lambda loop: None)

    tokenizer = object()
    multimodal_prompt_config = object()
    server.start_text_gen_server(
        coordinator_addr="tcp://coord:5555",
        tokenizer=tokenizer,
        rank=3,
        server_port=5000,
        parsers=["json"],
        verbose=True,
        num_replicas=1,
        hostname="127.0.0.1",
        sock=FakeSocket(),
        chat_template="template",
        multimodal_prompt_config=multimodal_prompt_config,
        default_temperature=0.4,
        default_top_p=0.8,
        default_top_k=5,
        eval_mode=True,
        max_inflight_requests=7,
        max_inflight_bytes=2**31,
    )

    # The port is taken from the socket; no fd is handed to the replica, which
    # binds its own listener on that port and reports readiness over the pipe.
    *run_args, _ready_conn = captured["run_args"]
    assert tuple(run_args) == (
        "tcp://coord:5555",
        tokenizer,
        3,
        4321,
        ["json"],
        True,
        "127.0.0.1",
        "template",
        multimodal_prompt_config,
        0.4,
        0.8,
        5,
        True,
        # block_size_tokens / prefix_caching_coordinator_policy: unset here, so the
        # frontend does not hash and the coordinator keeps doing it.
        None,
        None,
        7,
        2**31,
    )
    assert captured["socket_closed"] is True
    assert server._SERVER_PROCESSES[0].daemon is True
    assert server._SERVER_PROCESSES[0].started is True


def _patch_replica_launch(monkeypatch):
    """Record the byte budget each launched replica receives, without starting processes."""
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        text_generation_server as server,
    )

    budgets = []

    class FakeProcess:
        pid = 1

        def __init__(self, *, target, args, daemon) -> None:
            *_, replica_budget, self._ready_conn = args
            budgets.append(replica_budget)
            self.sentinel = os.pipe()[0]

        def start(self):
            # Report the replica as listening so start_text_gen_server returns.
            self._ready_conn.send(True)
            self._ready_conn.close()

    class FakeProcessContext:
        Process = FakeProcess

    monkeypatch.setattr(server, "_SERVER_PROCESSES", [])
    monkeypatch.setattr(server, "_SERVER_PROCESS_CONTEXT", FakeProcessContext())
    return server, budgets


def _start(server, **kwargs):
    return server.start_text_gen_server(
        coordinator_addr="tcp://coord:5555",
        tokenizer=object(),
        rank=0,
        server_port=5000,
        hostname="127.0.0.1",
        **kwargs,
    )


def test_inflight_byte_budget_is_split_across_replicas(monkeypatch):
    server, budgets = _patch_replica_launch(monkeypatch)

    _start(server, num_replicas=4, max_inflight_bytes=2**34)

    # The budget is per rank, so the host-wide total is unchanged by replica count.
    assert budgets == [2**32] * 4
    assert sum(budgets) == 2**34


@pytest.mark.parametrize("num_replicas", [1, 4, 16, 17, 32, 64])
def test_inflight_limits_are_off_by_default(monkeypatch, num_replicas):
    """No limit is configured unless the caller sets one, for any replica count."""
    server, budgets = _patch_replica_launch(monkeypatch)

    _start(server, num_replicas=num_replicas)

    assert budgets == [None] * num_replicas


def test_inflight_byte_budget_must_fit_a_max_size_request_per_replica(monkeypatch):
    server, budgets = _patch_replica_launch(monkeypatch)

    with pytest.raises(ValueError, match="max_inflight_bytes"):
        _start(server, num_replicas=4, max_inflight_bytes=3 * 2**30)
    with pytest.raises(ValueError, match="num_replicas"):
        _start(server, num_replicas=0, max_inflight_bytes=2**34)
    assert budgets == []

    # Exactly one maximum-size request per replica is allowed.
    _start(server, num_replicas=4, max_inflight_bytes=4 * 2**30)
    assert budgets == [2**30] * 4


@pytest.mark.asyncio
async def test_frontend_process_exposes_sampling_config_and_stops_client(monkeypatch):
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        text_generation_server as server,
    )

    captured = {}

    class FakeInferenceClient:
        def __init__(
            self,
            coordinator_addr,
            deserialize,
            block_size_tokens=None,
            prefix_caching_coordinator_policy=None,
        ):
            captured["client_init"] = (coordinator_addr, deserialize)
            captured["client_hashing"] = (block_size_tokens, prefix_caching_coordinator_policy)

        def start(self):
            captured["client_started"] = True

        def stop(self):
            captured["client_stopped"] = True

    async def fake_serve(app, config):
        captured["app"] = app
        captured["hypercorn_config"] = config

    class FakeListener:
        def fileno(self):
            return 23

        def listen(self, backlog):
            pass

        def close(self):
            captured["listener_closed"] = True

    monkeypatch.setattr(server, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(server, "serve", fake_serve)
    monkeypatch.setattr(server.endpoints, "__all__", [])
    # Each replica binds its own listener, so leaving this unpatched would make the
    # test take a real port and fail on whatever already holds it.
    monkeypatch.setattr(server, "_bind_reuseport_socket", lambda _port, _host: FakeListener())

    tokenizer = object()
    multimodal_prompt_config = object()
    await server._run_text_gen_server(
        coordinator_addr="tcp://coord:5555",
        tokenizer=tokenizer,
        rank=3,
        server_port=4321,
        parsers=["json"],
        verbose=True,
        hostname="127.0.0.1",
        chat_template="template",
        multimodal_prompt_config=multimodal_prompt_config,
        default_temperature=0.4,
        default_top_p=0.8,
        default_top_k=5,
        eval_mode=True,
        max_inflight_requests=7,
        max_inflight_bytes=2**31,
    )

    app_config = captured["app"].config
    assert captured["client_init"] == ("tcp://coord:5555", False)
    # Unset here, so this client does not hash and the coordinator keeps doing it.
    assert captured["client_hashing"] == (None, None)
    assert captured["client_started"] is True
    assert captured["client_stopped"] is True
    assert app_config["tokenizer"] is tokenizer
    assert app_config["parsers"] == ["json"]
    assert app_config["verbose"] is True
    assert app_config["chat_template"] == "template"
    assert app_config["multimodal_prompt_config"] is multimodal_prompt_config
    assert app_config["default_temperature"] == 0.4
    assert app_config["default_top_p"] == 0.8
    assert app_config["default_top_k"] == 5
    assert app_config["eval_mode"] is True
    assert isinstance(captured["app"].asgi_app, server._InflightHTTPLimit)
    assert captured["app"].asgi_app.max_inflight_requests == 7
    assert captured["app"].asgi_app.max_inflight_bytes == 2**31
    assert captured["app"].asgi_app.max_request_content_size == app_config["MAX_CONTENT_LENGTH"]
    assert captured["hypercorn_config"].h2_max_concurrent_streams == 7
    assert captured["hypercorn_config"].bind == ["fd://23"]
    assert captured["listener_closed"] is True


@pytest.mark.asyncio
async def test_frontend_process_has_no_inflight_limiter_by_default(monkeypatch):
    """The limiter is opt-in: without limits the app is served unwrapped."""
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        text_generation_server as server,
    )

    captured = {}

    class FakeInferenceClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    async def fake_serve(app, config):
        captured["app"] = app
        captured["hypercorn_config"] = config

    class FakeListener:
        def fileno(self):
            return 23

        def listen(self, backlog):
            pass

        def close(self):
            pass

    monkeypatch.setattr(server, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(server, "serve", fake_serve)
    monkeypatch.setattr(server.endpoints, "__all__", [])
    monkeypatch.setattr(server, "_bind_reuseport_socket", lambda _port, _host: FakeListener())

    await server._run_text_gen_server(
        coordinator_addr="tcp://coord:5555",
        tokenizer=object(),
        rank=0,
        server_port=4321,
        hostname="127.0.0.1",
    )

    assert not isinstance(captured["app"].asgi_app, server._InflightHTTPLimit)
    assert captured["hypercorn_config"].h2_max_concurrent_streams == 2**14


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "default_temperature",
        "default_top_p",
        "default_top_k",
        "request_overrides",
        "expected_temperature",
        "expected_top_p",
        "expected_top_k",
    ),
    [
        (0.6, 0.9, 0, {}, 0.6, 0.9, 0),
        (0.6, 0.9, 0, {"temperature": 0.4, "top_p": 0.8, "top_k": 5}, 0.4, 0.8, 5),
        (0.0, 0.9, 20, {}, 0.0, 0.0, 1),
    ],
)
async def test_completions_request_uses_sampling_defaults_and_overrides(
    default_temperature,
    default_top_p,
    default_top_k,
    request_overrides,
    expected_temperature,
    expected_top_p,
    expected_top_k,
):
    app = Quart(__name__)
    inference_client = _CapturingClient()
    app.config.update(
        client=inference_client,
        tokenizer=_Tokenizer(),
        verbose=False,
        default_temperature=default_temperature,
        default_top_p=default_top_p,
        default_top_k=default_top_k,
    )
    app.register_blueprint(completions_blueprint)

    payload = {"prompt": "hello", **request_overrides}
    response = await app.test_client().post("/v1/completions", json=payload)

    assert response.status_code == 500
    assert not inference_client.sampling_params[0].detokenize_generations
    sampling_params = inference_client.sampling_params[0]
    assert sampling_params.temperature == expected_temperature
    assert sampling_params.top_p == expected_top_p
    assert sampling_params.top_k == expected_top_k


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body_size", "max_request_content_size", "expected_status"),
    [(8, 2**30, 200), (64, 2**30, 503), (24, 16, 413)],
)
async def test_quart_request_over_byte_limit_is_rejected(
    body_size, max_request_content_size, expected_status
):
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        text_generation_server as server,
    )

    app = Quart(__name__)
    handled = []

    @app.post("/echo")
    async def echo():
        handled.append(len(await quart.request.get_data()))
        return "ok"

    app.asgi_app = server._InflightHTTPLimit(
        app.asgi_app,
        max_inflight_requests=10,
        max_inflight_bytes=32,
        max_request_content_size=max_request_content_size,
    )

    chunk = body_size // 2
    messages = [
        {"type": "http.request", "body": b"x" * chunk, "more_body": True},
        {"type": "http.request", "body": b"x" * chunk, "more_body": False},
    ]
    disconnected = asyncio.Event()

    async def receive():
        if messages:
            return messages.pop(0)
        await disconnected.wait()
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and not message.get("more_body"):
            disconnected.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/echo",
        "raw_path": b"/echo",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"localhost"), (b"content-length", str(body_size).encode())],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
        "extensions": {},
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)

    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert [m["status"] for m in starts] == [expected_status]
    assert handled == ([body_size] if expected_status == 200 else [])
    assert app.asgi_app.inflight_bytes == 0 and app.asgi_app.inflight == 0
