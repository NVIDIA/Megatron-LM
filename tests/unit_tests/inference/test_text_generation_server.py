# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    text_generation_server,
)


def test_frontend_processes_use_spawn_context():
    assert text_generation_server._SERVER_PROCESS_CONTEXT.get_start_method() == "spawn"


@pytest.mark.asyncio
@pytest.mark.parametrize("provide_config", [False, True])
async def test_server_exposes_multimodal_prompt_config(monkeypatch, provide_config):
    apps = []
    clients = []

    class FakeApp:
        def __init__(self, _name):
            self.config = {}
            self.blueprints = []
            self.asgi_app = object()
            apps.append(self)

        def register_blueprint(self, blueprint):
            self.blueprints.append(blueprint)

    class FakeClient:
        def __init__(
            self,
            address,
            deserialize,
            block_size_tokens=None,
            prefix_caching_coordinator_policy=None,
        ):
            self.address = address
            self.deserialize = deserialize
            self.block_size_tokens = block_size_tokens
            self.prefix_caching_coordinator_policy = prefix_caching_coordinator_policy
            self.started = False
            self.stopped = False
            clients.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    served = []

    async def fake_serve(app, config):
        served.append((app, config))

    closed_sockets = []

    class FakeListener:
        def fileno(self):
            return 19

        def close(self):
            closed_sockets.append(self)

    custom_config = MultimodalPromptConfig(video_spec=MediaPromptSpec(model_token="<video>"))
    supplied_config = custom_config if provide_config else None
    monkeypatch.setattr(text_generation_server, "HAS_BACKEND", True)
    monkeypatch.setattr(text_generation_server, "InferenceClient", FakeClient)
    monkeypatch.setattr(text_generation_server, "Quart", FakeApp, raising=False)
    monkeypatch.setattr(
        text_generation_server, "Config", lambda: SimpleNamespace(bind=None), raising=False
    )
    monkeypatch.setattr(text_generation_server, "serve", fake_serve, raising=False)
    monkeypatch.setattr(
        text_generation_server.endpoints, "__all__", ["completion-blueprint", "chat-blueprint"]
    )
    # Each replica binds its own listener, so leaving this unpatched would make the
    # test take a real port and fail on whatever already holds it.
    monkeypatch.setattr(
        text_generation_server, "_bind_reuseport_socket", lambda _port, _host: FakeListener()
    )

    await text_generation_server._run_text_gen_server(
        "coordinator:1234",
        tokenizer=object(),
        rank=0,
        server_port=8080,
        hostname="127.0.0.1",
        multimodal_prompt_config=supplied_config,
    )

    assert len(apps) == len(clients) == len(served) == 1
    app = apps[0]
    assert app.config["multimodal_prompt_config"] == (
        custom_config if provide_config else MultimodalPromptConfig()
    )
    assert app.blueprints == ["completion-blueprint", "chat-blueprint"]
    assert served[0][0] is app
    assert served[0][1].bind == ["fd://19"]
    assert len(closed_sockets) == 1, "the listener must be released once serve() returns"
    assert clients[0].address == "coordinator:1234"
    assert clients[0].deserialize is False
    assert clients[0].started is True
    assert clients[0].stopped is True


def _recorder():
    sent = []

    async def send(message):
        sent.append(message)

    return sent, send


@pytest.mark.asyncio
async def test_inflight_limit_rejects_requests_over_count():
    release = asyncio.Event()
    entered = []

    async def app(scope, receive, send):
        entered.append(scope["type"])
        await release.wait()

    limited = text_generation_server._InflightHTTPLimit(
        app, max_inflight_requests=2, max_inflight_bytes=100, max_request_content_size=100
    )

    async def call(scope_type="http"):
        sent, send = _recorder()
        scope = {
            "type": scope_type,
            "http_version": "1.1",
            "method": "POST",
            "headers": [(b"content-length", b"1")],
        }
        await limited(scope, None, send)
        return sent

    in_flight = [asyncio.create_task(call()) for _ in range(2)]
    while len(entered) < 2:
        await asyncio.sleep(0)

    rejected = await call()
    assert rejected[0]["status"] == 503
    assert (b"retry-after", b"1") in rejected[0]["headers"]
    assert len(entered) == 2, "the rejected request must not reach the app"

    # Lifespan and other non-HTTP scopes are never counted or rejected.
    lifespan = asyncio.create_task(call("lifespan"))
    while len(entered) < 3:
        await asyncio.sleep(0)

    release.set()
    await asyncio.gather(*in_flight, lifespan)
    assert limited.inflight == 0
    assert await call() == [], "slots must be freed once requests finish"


def _body(*chunks):
    messages = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ]

    async def receive():
        return messages.pop(0)

    return receive


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"transfer-encoding", b"chunked")],
        # h11 frames by Transfer-Encoding, so the declared length is not trusted.
        [(b"content-length", b"1"), (b"transfer-encoding", b"chunked")],
    ],
)
async def test_inflight_limit_counts_received_bytes(headers):
    received = []

    async def app(scope, receive, send):
        while (message := await receive())["type"] == "http.request":
            received.append(message["body"])
            if not message.get("more_body"):
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return

    limited = text_generation_server._InflightHTTPLimit(
        app, max_inflight_requests=10, max_inflight_bytes=10, max_request_content_size=100
    )
    scope = {"type": "http", "headers": headers}

    sent, send = _recorder()
    await limited(scope, _body(b"x" * 6, b"x" * 6), send)
    assert received == [b"x" * 6], "the chunk crossing the budget must not reach the app"
    assert [m.get("status") for m in sent] == [503, None]
    assert limited.inflight_bytes == 0 and limited.inflight == 0

    received.clear()
    sent, send = _recorder()
    await limited(scope, _body(b"x" * 5, b"x" * 5), send)
    assert sent[0]["status"] == 200, "a payload within budget is unaffected"
    assert limited.inflight_bytes == 0


@pytest.mark.asyncio
async def test_inflight_limit_counts_bytes_across_requests():
    release = asyncio.Event()
    entered = []

    async def app(scope, receive, send):
        await receive()
        entered.append(True)
        await release.wait()

    limited = text_generation_server._InflightHTTPLimit(
        app, max_inflight_requests=10, max_inflight_bytes=10, max_request_content_size=10
    )

    holding = asyncio.create_task(limited({"type": "http", "headers": []}, _body(b"x" * 10), None))
    while not entered:
        await asyncio.sleep(0)
    assert limited.inflight_bytes == 10

    sent, send = _recorder()
    await limited({"type": "http", "headers": []}, _body(b"x"), send)
    assert sent[0]["status"] == 503, "a full byte budget rejects new requests"
    assert len(entered) == 1

    release.set()
    await holding
    assert limited.inflight_bytes == 0 and limited.inflight == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("chunks", "expected_status"), [((5, 5), 200), ((6, 5), 413)])
async def test_inflight_limit_rejects_request_over_maximum(chunks, expected_status):
    received = []

    async def app(scope, receive, send):
        while (message := await receive())["type"] == "http.request":
            received.append(message["body"])
            if not message.get("more_body"):
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return

    limited = text_generation_server._InflightHTTPLimit(
        app, max_inflight_requests=10, max_inflight_bytes=100, max_request_content_size=10
    )
    sent, send = _recorder()
    # The declared length is not trusted; only received bytes count.
    scope = {"type": "http", "headers": [(b"content-length", b"1")]}
    await limited(scope, _body(*(b"x" * n for n in chunks)), send)

    assert sent[0]["status"] == expected_status
    if expected_status == 413:
        assert received == [b"x" * 6], "the chunk crossing the maximum must not reach the app"
        assert (b"retry-after", b"1") not in sent[0]["headers"]
    assert limited.inflight == 0 and limited.inflight_bytes == 0


def test_start_server_forwards_multimodal_prompt_config_to_worker(monkeypatch):
    processes = []

    class FakeSocket:
        def __init__(self):
            self.closed = False

        def getsockname(self):
            return "127.0.0.1", 8080

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.pid = 123
            processes.append(self)

        def start(self):
            pass

    prompt_config = MultimodalPromptConfig(video_spec=MediaPromptSpec(model_token="<video>"))
    monkeypatch.setattr(text_generation_server, "_SERVER_PROCESSES", [])
    monkeypatch.setattr(
        text_generation_server, "_SERVER_PROCESS_CONTEXT", SimpleNamespace(Process=FakeProcess)
    )

    handed_in_socket = FakeSocket()
    text_generation_server.start_text_gen_server(
        "coordinator:1234",
        tokenizer=object(),
        rank=0,
        server_port=0,
        num_replicas=1,
        sock=handed_in_socket,
        multimodal_prompt_config=prompt_config,
    )

    # The socket is read for its port and released: replicas bind that port
    # themselves so they each get their own accept queue.
    assert handed_in_socket.closed is True
    assert len(processes) == 1
    assert processes[0].target is text_generation_server._server_process_worker
    worker_call = inspect.signature(text_generation_server._server_process_worker).bind(
        *processes[0].args
    )
    assert worker_call.arguments["multimodal_prompt_config"] is prompt_config
    assert processes[0].daemon is True


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"max_inflight_requests": 0}, "max_inflight_requests"),
        ({"max_inflight_bytes": 2**30 - 1}, "max_inflight_bytes"),
    ],
)
def test_start_server_rejects_invalid_inflight_limits(monkeypatch, limits, message):
    monkeypatch.setattr(text_generation_server, "_SERVER_PROCESSES", [])
    monkeypatch.setattr(
        text_generation_server,
        "_SERVER_PROCESS_CONTEXT",
        SimpleNamespace(Process=lambda **_kwargs: pytest.fail("must not start a replica")),
    )

    with pytest.raises(ValueError, match=message):
        text_generation_server.start_text_gen_server(
            "coordinator:1234", tokenizer=object(), rank=0, server_port=8080, **limits
        )


def test_start_server_rejects_socket_without_real_port(monkeypatch):
    socket_without_port = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 0))
    monkeypatch.setattr(text_generation_server, "_SERVER_PROCESSES", [])

    with pytest.raises(ValueError, match="socket must be bound to a real port"):
        text_generation_server.start_text_gen_server(
            "coordinator:1234", tokenizer=object(), rank=0, server_port=0, sock=socket_without_port
        )

    assert text_generation_server._SERVER_PROCESSES == []


def test_start_server_is_noop_when_replicas_are_running(monkeypatch):
    existing_process = object()
    monkeypatch.setattr(text_generation_server, "_SERVER_PROCESSES", [existing_process])
    monkeypatch.setattr(
        text_generation_server,
        "_SERVER_PROCESS_CONTEXT",
        SimpleNamespace(Process=lambda **_kwargs: pytest.fail("must not create another process")),
    )

    text_generation_server.start_text_gen_server(
        "coordinator:1234", tokenizer=object(), rank=0, server_port=8080
    )

    assert text_generation_server._SERVER_PROCESSES == [existing_process]


def test_stop_server_cleans_up_processes(monkeypatch):
    class FakeProcess:
        def __init__(self, *, exits_on_terminate):
            self.alive = True
            self.exits_on_terminate = exits_on_terminate
            self.terminated = False
            self.killed = False
            self.join_timeouts = []

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            if self.exits_on_terminate:
                self.alive = False

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)

        def kill(self):
            self.killed = True
            self.alive = False

    graceful = FakeProcess(exits_on_terminate=True)
    stubborn = FakeProcess(exits_on_terminate=False)
    monkeypatch.setattr(text_generation_server, "_SERVER_PROCESSES", [graceful, stubborn])

    text_generation_server.stop_text_gen_server()

    assert graceful.terminated is stubborn.terminated is True
    assert graceful.killed is False
    assert stubborn.killed is True
    assert graceful.join_timeouts == [3]
    assert stubborn.join_timeouts == [3, None]
    assert text_generation_server._SERVER_PROCESSES == []
