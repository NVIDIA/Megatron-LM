# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for configurable defaults on the dynamic text generation server."""

import pytest

quart = pytest.importorskip("quart")
from quart import Quart

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

    async def add_request(self, prompt_tokens, sampling_params, *, multi_modal_data=None):
        del prompt_tokens, multi_modal_data
        self.sampling_params.append(sampling_params)
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


def test_sampling_config_reaches_frontend_process(monkeypatch):
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        text_generation_server as server,
    )

    captured = {}

    async def fake_run_text_gen_server(*args):
        captured["run_args"] = args

    class FakeProcess:
        pid = 123

        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True
            self.target(*self.args)

    class FakeSocket:
        def getsockname(self):
            return ("127.0.0.1", 4321)

        def setblocking(self, blocking):
            captured["blocking"] = blocking

        def set_inheritable(self, inheritable):
            captured["inheritable"] = inheritable

        def fileno(self):
            return 17

    monkeypatch.setattr(server, "_SERVER_PROCESSES", [])
    monkeypatch.setattr(server, "_SHARED_SOCKET", None)
    monkeypatch.setattr(server.mp, "Process", FakeProcess)
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
    )

    assert captured["run_args"] == (
        "tcp://coord:5555",
        tokenizer,
        3,
        4321,
        ["json"],
        True,
        17,
        "127.0.0.1",
        "template",
        multimodal_prompt_config,
        0.4,
        0.8,
        5,
        True,
    )
    assert captured["blocking"] is False
    assert captured["inheritable"] is True
    assert server._SERVER_PROCESSES[0].daemon is True
    assert server._SERVER_PROCESSES[0].started is True


@pytest.mark.asyncio
async def test_frontend_process_exposes_sampling_config_and_stops_client(monkeypatch):
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        text_generation_server as server,
    )

    captured = {}

    class FakeInferenceClient:
        def __init__(self, coordinator_addr, deserialize):
            captured["client_init"] = (coordinator_addr, deserialize)

        def start(self):
            captured["client_started"] = True

        def stop(self):
            captured["client_stopped"] = True

    async def fake_serve(app, config):
        captured["app"] = app
        captured["hypercorn_config"] = config

    monkeypatch.setattr(server, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(server, "serve", fake_serve)
    monkeypatch.setattr(server.endpoints, "__all__", [])

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
    )

    app_config = captured["app"].config
    assert captured["client_init"] == ("tcp://coord:5555", False)
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
    assert captured["hypercorn_config"].bind == ["127.0.0.1:4321"]


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
    assert len(inference_client.sampling_params) == 1
    sampling_params = inference_client.sampling_params[0]
    assert sampling_params.temperature == expected_temperature
    assert sampling_params.top_p == expected_top_p
    assert sampling_params.top_k == expected_top_k
