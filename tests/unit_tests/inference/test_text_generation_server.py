# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    text_generation_server,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("provide_config", [False, True])
async def test_server_exposes_multimodal_prompt_config(monkeypatch, provide_config):
    apps = []
    clients = []

    class FakeApp:
        def __init__(self, _name):
            self.config = {}
            self.blueprints = []
            apps.append(self)

        def register_blueprint(self, blueprint):
            self.blueprints.append(blueprint)

    class FakeClient:
        def __init__(self, address, deserialize):
            self.address = address
            self.deserialize = deserialize
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
    assert served[0][1].bind == ["127.0.0.1:8080"]
    assert clients[0].address == "coordinator:1234"
    assert clients[0].deserialize is False
    assert clients[0].started is True
    assert clients[0].stopped is True


def test_start_server_forwards_multimodal_prompt_config_to_worker(monkeypatch):
    processes = []

    class FakeSocket:
        def getsockname(self):
            return "127.0.0.1", 8080

        def setblocking(self, _blocking):
            pass

        def set_inheritable(self, _inheritable):
            pass

        def fileno(self):
            return 12

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
    monkeypatch.setattr(text_generation_server, "_SHARED_SOCKET", None)
    monkeypatch.setattr(text_generation_server.mp, "Process", FakeProcess)

    text_generation_server.start_text_gen_server(
        "coordinator:1234",
        tokenizer=object(),
        rank=0,
        server_port=0,
        num_replicas=1,
        sock=FakeSocket(),
        multimodal_prompt_config=prompt_config,
    )

    assert len(processes) == 1
    assert processes[0].target is text_generation_server._server_process_worker
    worker_call = inspect.signature(text_generation_server._server_process_worker).bind(
        *processes[0].args
    )
    assert worker_call.arguments["multimodal_prompt_config"] is prompt_config
    assert processes[0].daemon is True


def test_start_server_rejects_socket_without_real_port(monkeypatch):
    socket_without_port = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 0))
    monkeypatch.setattr(text_generation_server, "_SERVER_PROCESSES", [])
    monkeypatch.setattr(text_generation_server, "_SHARED_SOCKET", None)

    with pytest.raises(ValueError, match="socket must be bound to a real port"):
        text_generation_server.start_text_gen_server(
            "coordinator:1234", tokenizer=object(), rank=0, server_port=0, sock=socket_without_port
        )

    assert text_generation_server._SHARED_SOCKET is None


def test_start_server_is_noop_when_replicas_are_running(monkeypatch):
    existing_process = object()
    monkeypatch.setattr(text_generation_server, "_SERVER_PROCESSES", [existing_process])
    monkeypatch.setattr(
        text_generation_server.mp,
        "Process",
        lambda **_kwargs: pytest.fail("must not create another process"),
    )

    text_generation_server.start_text_gen_server(
        "coordinator:1234", tokenizer=object(), rank=0, server_port=8080
    )

    assert text_generation_server._SERVER_PROCESSES == [existing_process]


def test_stop_server_cleans_up_processes_and_shared_socket(monkeypatch):
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
    shared_socket = SimpleNamespace(closed=False)

    def close_socket():
        shared_socket.closed = True

    shared_socket.close = close_socket
    monkeypatch.setattr(text_generation_server, "_SERVER_PROCESSES", [graceful, stubborn])
    monkeypatch.setattr(text_generation_server, "_SHARED_SOCKET", shared_socket)

    text_generation_server.stop_text_gen_server()

    assert graceful.terminated is stubborn.terminated is True
    assert graceful.killed is False
    assert stubborn.killed is True
    assert graceful.join_timeouts == [3]
    assert stubborn.join_timeouts == [3, None]
    assert shared_socket.closed is True
    assert text_generation_server._SERVER_PROCESSES == []
    assert text_generation_server._SHARED_SOCKET is None
