# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for endpoints/common.py and for what /v1/completions and /v1/chat/completions share.

The shared behavior is pinned over HTTP: both handlers run under Quart's test client against the
fake inference client defined below, which test_completions.py and test_chat_completions.py reuse.
"""

import asyncio
import importlib
import json
import logging
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.config import MultimodalPromptConfig
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    abort_requests,
    apply_optional_sampling_default,
    generation_config_sampling_defaults,
    resolve_sampling_default,
)


@pytest.mark.parametrize(
    ("value", "expected_config"),
    [
        # The bug wdykas flagged on PR #7191: if startup set the key unconditionally (even to a
        # hardcoded default), tier 1 of resolve_sampling_default would always win and
        # generation_config.json would be dead code.
        pytest.param(None, {}, id="unset-leaves-no-key"),
        pytest.param(0.6, {"default_temperature": 0.6}, id="configured"),
        # 0 / 0.0 is a real, explicit configuration, not "unset". Only None means unset.
        pytest.param(0, {"default_temperature": 0}, id="configured-zero-is-explicit"),
    ],
)
def test_apply_optional_sampling_default(value, expected_config):
    app_config = {}
    apply_optional_sampling_default(app_config, "default_temperature", value)
    assert app_config == expected_config


@pytest.mark.parametrize(
    ("app_config", "generation_defaults", "expected"),
    [
        # A server started without --default-temperature must let the model's own
        # generation_config.json value win, not silently fall to the hardcoded 1.0.
        pytest.param({}, {"temperature": 0.6}, 0.6, id="generation-config-when-server-unset"),
        pytest.param({"default_temperature": 0.2}, {"temperature": 0.6}, 0.2, id="server-wins"),
        # `config_key in app_config` rather than `.get(config_key, default)`: an operator-configured
        # 0 is not mistaken for "unset".
        pytest.param({"default_temperature": 0}, {"temperature": 0.6}, 0, id="configured-zero"),
        pytest.param({}, {}, 1.0, id="hardcoded-fallback"),
    ],
)
def test_resolve_sampling_default_tiers(app_config, generation_defaults, expected):
    assert (
        resolve_sampling_default(
            app_config, generation_defaults, "temperature", "default_temperature", 1.0
        )
        == expected
    )


@pytest.mark.parametrize(
    ("tokenizer", "expected"),
    [
        pytest.param(SimpleNamespace(), {}, id="no-generation-config"),
        pytest.param(SimpleNamespace(generation_config="not-a-dict"), {}, id="not-a-dict"),
        pytest.param(
            SimpleNamespace(
                generation_config={
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "do_sample": True,
                }
            ),
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            id="numeric-sampling-fields",
        ),
        # bool is an int subclass; a stray `"top_k": true` is not a numeric sampling value.
        pytest.param(SimpleNamespace(generation_config={"top_k": True}), {}, id="bool-rejected"),
        pytest.param(
            SimpleNamespace(generation_config={"temperature": "warm", "top_p": 0.9}),
            {"top_p": 0.9},
            id="non-numeric-omitted",
        ),
    ],
)
def test_generation_config_sampling_defaults(tokenizer, expected):
    assert generation_config_sampling_defaults(tokenizer) == expected


@pytest.mark.parametrize(
    ("request_ids", "abort_results"),
    [
        pytest.param(
            [7, 8, 9], [None, RuntimeError("coordinator gone"), None], id="one-abort-fails"
        ),
        pytest.param([], None, id="nothing-in-flight"),
    ],
)
def test_abort_requests_is_best_effort(request_ids, abort_results):
    """One failing abort neither stops the rest nor raises: the caller is already unwinding."""
    client = MagicMock()
    client.abort_request.side_effect = abort_results

    abort_requests(client, request_ids, "client disconnected")

    assert [call.args[0] for call in client.abort_request.call_args_list] == request_ids


# --- HTTP harness, shared with the other endpoint test modules --------------------------------

CHAT_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"
CHAT_BODY = {"messages": [{"role": "user", "content": "hello"}]}
COMPLETIONS_BODY = {"prompt": "hello"}
BODIES = {CHAT_PATH: CHAT_BODY, COMPLETIONS_PATH: COMPLETIONS_BODY}
# Three submissions per request: n=3 choices for chat, a batch of three prompts for completions.
FAN_OUT_BODIES = {CHAT_PATH: {**CHAT_BODY, "n": 3}, COMPLETIONS_PATH: {"prompt": ["a", "b", "c"]}}
PATHS = pytest.mark.parametrize("path", [CHAT_PATH, COMPLETIONS_PATH], ids=["chat", "completions"])
NOT_A_NUMBER_ERROR = "Invalid sampling parameter: could not convert string to float: 'hot'"
_ENDPOINT_MODULES = {CHAT_PATH: "chat_completions", COMPLETIONS_PATH: "completions"}
_ENDPOINTS_PACKAGE = (
    "megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints"
)
_GENERATION_CONFIG_DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


class Tokenizer:
    """Fixed tokenization; each id detokenizes to "<id>" so expected strings stay readable."""

    chat_template = "test-template"
    bos = None
    eos_id = 2
    eod = 2

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return [10, 11]

    def tokenize(self, prompt):
        del prompt
        return [10, 11]

    def detokenize(self, token_ids, **kwargs):
        return "".join(f"<{tok}>" for tok in token_ids)


class GenerationConfigTokenizer(Tokenizer):
    """Carries the model's generation_config.json sampling defaults."""

    generation_config = _GENERATION_CONFIG_DEFAULTS


def completed_reply(uid, prompt_tokens, generated_tokens, **extra):
    """A completed, non-serialized reply dict of the shape the coordinator forwards."""
    reply = {
        "uid": uid,
        "status": "COMPLETED",
        "events": [],
        "prompt_tokens": list(prompt_tokens),
        "prompt_length": len(prompt_tokens),
        "generated_tokens": list(generated_tokens),
        "generated_log_probs": None,
        "routing_indices": None,
        "num_cached_tokens": 0,
        "sampling_params": {"num_tokens_to_generate": 16},
    }
    reply.update(extra)
    return reply


def failed_reply(*events):
    """A failed reply carrying the given engine events."""
    return {"uid": "req-failed", "status": "FAILED", "events": list(events)}


PENDING = object()  # A reply that never arrives: the request stays in flight.


class ReplyingClient:
    """Records each submission and answers it with the next canned reply.

    A reply that is an exception fails that request's future and ``PENDING`` leaves it unresolved;
    ``fail_admission_at`` is the zero-based submission that raises instead of being admitted.
    Without ``replies`` every submission gets a completed reply built from its own prompt tokens.
    ``drained`` is set once the last canned reply has been handed out.
    """

    def __init__(self, replies=None, *, fail_admission_at=None):
        self.replies = None if replies is None else list(replies)
        self.fail_admission_at = fail_admission_at
        self.drained = asyncio.Event()
        self.prompt_tokens = []
        self.sampling_params = []
        self.multi_modal_data = []
        self.offload_params = []
        self.aborted = []
        self.streamed = []

    def add_request_with_id(
        self, prompt_tokens, sampling_params, *, multi_modal_data=None, offload_params=None
    ):
        if len(self.prompt_tokens) == self.fail_admission_at:
            raise RuntimeError("zmq send failed")
        self.prompt_tokens.append(list(prompt_tokens))
        self.sampling_params.append(sampling_params)
        self.multi_modal_data.append(multi_modal_data)
        self.offload_params.append(offload_params)
        request_id = len(self.prompt_tokens)
        reply = (
            completed_reply(f"req-{request_id}", prompt_tokens, [12, 13])
            if self.replies is None
            else self.replies.pop(0)
        )
        if self.replies is not None and not self.replies:
            self.drained.set()
        future = asyncio.get_running_loop().create_future()
        if isinstance(reply, Exception):
            future.set_exception(reply)
        elif reply is not PENDING:
            future.set_result(reply)
        return request_id, future

    def add_request_streaming(
        self, prompt_tokens, sampling_params, *, multi_modal_data=None, offload_params=None
    ):
        """Answers with a stream that carries the canned reply as its only, final frame."""
        request_id, future = self.add_request_with_id(
            prompt_tokens,
            sampling_params,
            multi_modal_data=multi_modal_data,
            offload_params=offload_params,
        )
        self.streamed.append(request_id)
        stream = AsyncStream(request_id=request_id, cancel=lambda: self.aborted.append(request_id))
        stream.put({"final": future.result()})
        stream.finish()
        return stream

    def abort_request(self, request_id):
        self.aborted.append(request_id)


def build_app(path, client, **config):
    """A Quart app serving the endpoint at ``path`` with the test defaults; ``config`` overrides."""
    quart = pytest.importorskip("quart")
    blueprint = importlib.import_module(f"{_ENDPOINTS_PACKAGE}.{_ENDPOINT_MODULES[path]}").bp
    app = quart.Quart(__name__)
    app.config.update(
        client=client,
        tokenizer=Tokenizer(),
        parsers=[],
        verbose=False,
        multimodal_prompt_config=MultimodalPromptConfig(),
        eval_mode=True,
    )
    app.config.update(config)
    app.register_blueprint(blueprint)
    return app


# --- sampling parameters ----------------------------------------------------

# Request-controlled SamplingParams fields at their defaults, as both endpoints submit them. The
# temperature/top_p/top_k tiers and the greedy normalization are pinned end to end in
# test_dynamic_text_generation_server_config.py.
_DEFAULT_FIELDS = {
    "temperature": 1.0,
    "top_p": 0.0,  # SamplingParams normalizes the no-op 1.0 to the disabled sentinel.
    "top_k": 0,
    "return_log_probs": False,
    "top_n_logprobs": 0,
    "skip_prompt_log_probs": True,
    "num_tokens_to_generate": None,
    "stop_words": None,
    "add_BOS": False,
    "termination_id": None,
    "streaming_interval": 1,
    "detokenize_generations": False,
}
_CHAT_DEFAULTS = {**_DEFAULT_FIELDS, "return_prompt_tokens": False}  # eval_mode=True in build_app
_COMPLETIONS_DEFAULTS = {
    **_DEFAULT_FIELDS,
    "num_tokens_to_generate": 16,
    "return_prompt_tokens": True,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "body", "app_config", "expected_fields"),
    [
        pytest.param(CHAT_PATH, {}, {}, _CHAT_DEFAULTS, id="chat-defaults"),
        pytest.param(COMPLETIONS_PATH, {}, {}, _COMPLETIONS_DEFAULTS, id="completions-defaults"),
        pytest.param(
            CHAT_PATH,
            {
                "logprobs": None,
                "top_logprobs": None,
                "skip_prompt_log_probs": None,
                "max_tokens": None,
                "max_completion_tokens": None,
                "n": None,
                "stop": None,
                "add_BOS": None,
                "ignore_eos": None,
                "streaming_interval": None,
            },
            {},
            _CHAT_DEFAULTS,
            id="chat-null-fields-mean-default",
        ),
        pytest.param(
            COMPLETIONS_PATH,
            {"logprobs": None, "echo": None, "stop": None, "ignore_eos": None},
            {},
            _COMPLETIONS_DEFAULTS,
            id="completions-null-fields-mean-default",
        ),
        pytest.param(
            CHAT_PATH,
            {},
            {"tokenizer": GenerationConfigTokenizer()},
            _GENERATION_CONFIG_DEFAULTS,
            id="chat-model-generation-config",
        ),
        pytest.param(
            COMPLETIONS_PATH,
            {},
            {"tokenizer": GenerationConfigTokenizer()},
            _GENERATION_CONFIG_DEFAULTS,
            id="completions-model-generation-config",
        ),
        pytest.param(
            COMPLETIONS_PATH,
            {"logprobs": 5, "echo": True},
            {},
            {"return_log_probs": True, "top_n_logprobs": 5, "skip_prompt_log_probs": False},
            id="completions-logprobs-int-with-echo-scores-the-prompt",
        ),
        pytest.param(
            COMPLETIONS_PATH,
            {"logprobs": 0, "echo": True},
            {},
            {"return_log_probs": True, "top_n_logprobs": 0, "skip_prompt_log_probs": False},
            id="completions-logprobs-zero-scores-without-top-n",
        ),
        pytest.param(
            COMPLETIONS_PATH,
            {"logprobs": 3},
            {},
            {"return_log_probs": True, "top_n_logprobs": 3, "skip_prompt_log_probs": True},
            id="completions-logprobs-int-without-echo",
        ),
        # /v1/completions derives skip_prompt_log_probs from echo; the engine knob is not read.
        pytest.param(
            COMPLETIONS_PATH,
            {"logprobs": 5, "skip_prompt_log_probs": False},
            {},
            {"return_log_probs": True, "top_n_logprobs": 5, "skip_prompt_log_probs": True},
            id="completions-explicit-skip-prompt-log-probs-is-ignored",
        ),
        pytest.param(
            CHAT_PATH,
            {"logprobs": True, "top_logprobs": 10, "skip_prompt_log_probs": False},
            {},
            {"return_log_probs": True, "top_n_logprobs": 10, "skip_prompt_log_probs": False},
            id="chat-logprobs-bool-with-top-logprobs",
        ),
        pytest.param(
            CHAT_PATH,
            {"logprobs": False, "top_logprobs": 5},
            {},
            {"return_log_probs": False, "top_n_logprobs": 0},
            id="chat-logprobs-false-ignores-top-logprobs",
        ),
        pytest.param(
            CHAT_PATH,
            {"max_completion_tokens": 256, "max_tokens": 128},
            {},
            {"num_tokens_to_generate": 256},
            id="chat-max-completion-tokens-wins",
        ),
        pytest.param(
            CHAT_PATH,
            {"max_tokens": 128, "stop": ["A", "B"], "ignore_eos": True, "add_BOS": True},
            {},
            {
                "num_tokens_to_generate": 128,
                "stop_words": ["A", "B"],
                "termination_id": -1,
                "add_BOS": True,
            },
            id="chat-max-tokens-stop-ignore-eos-add-bos",
        ),
        pytest.param(
            CHAT_PATH,
            {"n": 4, "streaming_interval": 3},
            {},
            {"streaming_interval": 3},
            id="chat-n-fans-out-with-streaming-interval",
        ),
        pytest.param(
            CHAT_PATH, {"stop": "END"}, {}, {"stop_words": ["END"]}, id="chat-stop-string"
        ),
        pytest.param(
            COMPLETIONS_PATH,
            {"max_tokens": 32, "stop": "END", "ignore_eos": True, "streaming_interval": 2},
            {},
            {
                "num_tokens_to_generate": 32,
                "stop_words": ["END"],
                "termination_id": -1,
                "streaming_interval": 2,
            },
            id="completions-max-tokens-stop-string-ignore-eos",
        ),
        # Only a streaming request reads stream_options; junk there is ignored otherwise.
        pytest.param(
            CHAT_PATH,
            {"stream_options": ["include_usage"]},
            {},
            _CHAT_DEFAULTS,
            id="chat-non-dict-stream-options-ignored",
        ),
    ],
)
async def test_sampling_params_are_parsed_from_the_request(path, body, app_config, expected_fields):
    client = ReplyingClient()
    app = build_app(path, client, **app_config)

    response = await app.test_client().post(path, json={**BODIES[path], **body})

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert len(client.sampling_params) == (body.get("n") or 1)
    for sampling_params in client.sampling_params:
        assert {name: getattr(sampling_params, name) for name in expected_fields} == expected_fields


# Fields the HTTP layer sets from the request body (chat also reads add_BOS) ...
_REQUEST_CONTROLLED_FIELDS = {
    "temperature",
    "top_k",
    "top_p",
    "return_log_probs",
    "top_n_logprobs",
    "skip_prompt_log_probs",
    "num_tokens_to_generate",
    "stop_words",
    "termination_id",
    "streaming_interval",
}
# ... the fields each endpoint decides itself and never reads from the request ...
_FRONTEND_OWNED_FIELDS = {
    CHAT_PATH: {"return_prompt_tokens": False, "detokenize_generations": False},
    COMPLETIONS_PATH: {
        "return_prompt_tokens": True,
        "detokenize_generations": False,
        "add_BOS": False,
    },
}
# ... and the fields no HTTP client may set: engine-internal, derived, or forced by the client.
_ENGINE_OWNED_FIELDS = {
    "return_prompt_top_n_logprobs",
    "return_segments",
    "num_tokens_total",
    "detokenize_stop_sequence",
    "streaming",
    "do_kv_handoff",
}
_EVERY_FIELD_REQUEST = {
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 40,
    "logprobs": 5,  # an int for /v1/completions, truthy for chat
    "top_logprobs": 5,
    # Differs from the dataclass default and keeps the derived return_prompt_top_n_logprobs at its
    # own default.
    "skip_prompt_log_probs": True,
    "max_tokens": 200,
    "stop": ["END"],
    "add_BOS": True,
    "ignore_eos": True,
    "streaming_interval": 3,
    # Named by the client, but not the client's to set.
    **{name: True for name in _ENGINE_OWNED_FIELDS},
}


@pytest.mark.asyncio
@PATHS
async def test_every_sampling_params_field_is_classified(path):
    """A new SamplingParams field must be placed in one of the three groups above."""
    request_controlled = _REQUEST_CONTROLLED_FIELDS | ({"add_BOS"} if path == CHAT_PATH else set())
    frontend_owned = _FRONTEND_OWNED_FIELDS[path]
    assert {field.name for field in fields(SamplingParams)} == (
        request_controlled | set(frontend_owned) | _ENGINE_OWNED_FIELDS
    )
    client = ReplyingClient()
    app = build_app(path, client)

    response = await app.test_client().post(path, json={**BODIES[path], **_EVERY_FIELD_REQUEST})

    assert response.status_code == 200, await response.get_data(as_text=True)
    (sampling_params,) = client.sampling_params
    defaults = SamplingParams()
    for name in request_controlled:
        assert getattr(sampling_params, name) != getattr(defaults, name), name
    for name in _ENGINE_OWNED_FIELDS:
        assert getattr(sampling_params, name) == getattr(defaults, name), name
    assert {name: getattr(sampling_params, name) for name in frontend_owned} == frontend_owned


# --- failures before a response is formatted ---------------------------------


@pytest.mark.asyncio
@PATHS
@pytest.mark.parametrize(
    ("make_client", "expected_error", "expected_aborted"),
    [
        # A failure on admission k aborts the k-1 requests already in flight.
        pytest.param(
            lambda: ReplyingClient(fail_admission_at=2),
            "Error submitting request: zmq send failed",
            [1, 2],
            id="third-admission-fails",
        ),
        pytest.param(
            lambda: ReplyingClient(
                [
                    completed_reply("req-0", [10, 11], [12]),
                    ValueError("boom"),
                    completed_reply("req-2", [10, 11], [12]),
                ]
            ),
            "Error during inference: boom",
            [],
            id="engine-error",
        ),
    ],
)
async def test_failures_before_formatting_are_a_500(
    path, make_client, expected_error, expected_aborted
):
    client = make_client()
    app = build_app(path, client)

    response = await app.test_client().post(path, json=FAN_OUT_BODIES[path])

    assert response.status_code == 500
    assert await response.get_data(as_text=True) == expected_error
    assert client.aborted == expected_aborted


async def _post_then_disconnect(app, path, body, client):
    """POST over raw ASGI, wait until every request is admitted, then drop the connection.

    Quart's test client runs every request to completion; only an ``http.disconnect`` message
    makes its ASGI connection cancel the handler, which is the path under test.
    """
    payload = json.dumps(body).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"localhost"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("127.0.0.1", 45678),
        "server": ("localhost", 80),
    }
    incoming = asyncio.Queue()
    incoming.put_nowait({"type": "http.request", "body": payload, "more_body": False})
    sent = []

    async def receive():
        return await incoming.get()

    async def send(message):
        sent.append(message)

    connection = asyncio.create_task(app(scope, receive, send))
    # Disconnecting before admission would prove nothing: there would be nothing in flight.
    await asyncio.wait_for(client.drained.wait(), timeout=10)
    incoming.put_nowait({"type": "http.disconnect"})
    await asyncio.wait_for(connection, timeout=10)
    return sent


@pytest.mark.asyncio
@PATHS
async def test_disconnect_aborts_every_in_flight_request(path):
    """Every fan-out admission is aborted and nothing is written: the peer is gone."""
    client = ReplyingClient([PENDING] * 3)
    app = build_app(path, client)

    sent = await _post_then_disconnect(app, path, FAN_OUT_BODIES[path], client)

    # CancelledError is re-raised rather than turned into a 500; nobody is left to read one.
    assert client.aborted == [1, 2, 3]
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "expected_logs"),
    [
        # Chat also logs every reply, with its token-id lists redacted.
        (
            CHAT_PATH,
            [
                "Batch of 3 requests (n=3) processed in",
                "'prompt_tokens': '...truncated...'",
                "'generated_tokens': '...truncated...'",
            ],
        ),
        (COMPLETIONS_PATH, ["Batch of 3 requests processed in"]),
    ],
    ids=["chat", "completions"],
)
async def test_verbose_logs_the_batch_timing_and_redacted_replies(path, expected_logs, caplog):
    client = ReplyingClient()
    app = build_app(path, client, verbose=True)

    with caplog.at_level(logging.INFO):
        response = await app.test_client().post(path, json=FAN_OUT_BODIES[path])

    assert response.status_code == 200, await response.get_data(as_text=True)
    for expected_log in expected_logs:
        assert expected_log in caplog.text


# --- failed requests ----------------------------------------------------------

_COMPLETED = completed_reply("req-ok", [10, 11], [12])
_OVERFLOW = "MaxSequenceLengthOverflowError: prompt exceeds max_sequence_length"
# Nemo-RL matches on this exact message.
_NEMO_RL_OVERFLOW_BODY = (
    "This model's maximum context length was exceeded. Your messages resulted in 2 tokens. "
    f"Please reduce the length of the messages. Request 0: {_OVERFLOW}"
)
_PLAIN_OVERFLOW_BODY = f"Inference request(s) failed: Request 0: {_OVERFLOW}"


def _on_both_endpoints(expected):
    return {CHAT_PATH: expected, COMPLETIONS_PATH: expected}


@pytest.mark.asyncio
@PATHS
@pytest.mark.parametrize(
    ("replies", "expected_by_path"),
    [
        pytest.param(
            [
                failed_reply({"type": "ERROR_NONTRANSIENT", "payload": "bad"}),
                _COMPLETED,
                _COMPLETED,
            ],
            _on_both_endpoints(("Inference request(s) failed: Request 0: bad", 400)),
            id="nontransient-is-400",
        ),
        pytest.param(
            [
                _COMPLETED,
                failed_reply({"type": "ERROR_TRANSIENT", "payload": "timeout"}),
                _COMPLETED,
            ],
            _on_both_endpoints(("Inference request(s) failed: Request 1: timeout", 500)),
            id="transient-is-500",
        ),
        pytest.param(
            [
                failed_reply({"type": "ERROR_TRANSIENT", "payload": "t"}),
                _COMPLETED,
                failed_reply({"type": "ERROR_NONTRANSIENT", "payload": "nt"}),
            ],
            _on_both_endpoints(("Inference request(s) failed: Request 0: t; Request 2: nt", 400)),
            id="mixed-nontransient-wins",
        ),
        pytest.param(
            [failed_reply(), _COMPLETED, {"uid": "req-failed", "status": "FAILED"}],
            _on_both_endpoints(
                (
                    "Inference request(s) failed: Request 0: Unknown error; "
                    "Request 2: Unknown error",
                    500,
                )
            ),
            id="no-error-events",
        ),
        # /v1/completions reports a context overflow like any other failure, while
        # /v1/chat/completions reports the prompt length in the body Nemo-RL matches on, whatever
        # the event type.
        pytest.param(
            [
                failed_reply({"type": "ERROR_NONTRANSIENT", "payload": _OVERFLOW}),
                _COMPLETED,
                _COMPLETED,
            ],
            {
                CHAT_PATH: (_NEMO_RL_OVERFLOW_BODY, 400),
                COMPLETIONS_PATH: (_PLAIN_OVERFLOW_BODY, 400),
            },
            id="overflow-nontransient",
        ),
        pytest.param(
            [
                failed_reply({"type": "ERROR_TRANSIENT", "payload": _OVERFLOW}),
                _COMPLETED,
                _COMPLETED,
            ],
            {
                CHAT_PATH: (_NEMO_RL_OVERFLOW_BODY, 400),
                COMPLETIONS_PATH: (_PLAIN_OVERFLOW_BODY, 500),
            },
            id="overflow-transient",
        ),
    ],
)
async def test_failed_requests_are_reported(path, replies, expected_by_path):
    client = ReplyingClient(replies)
    app = build_app(path, client)

    response = await app.test_client().post(path, json=FAN_OUT_BODIES[path])

    assert (await response.get_data(as_text=True), response.status_code) == expected_by_path[path]


# --- streaming ----------------------------------------------------------------


class TextDetokenizer:
    """Stands in for the incremental detokenizer, which only accepts HF fast tokenizers."""

    def __init__(self, tokenizer, prompt_tokens):
        del prompt_tokens
        self._tokenizer = tokenizer
        self.text = ""

    def update(self, tokens):
        delta = self._tokenizer.detokenize(tokens)
        self.text += delta
        return delta

    @property
    def text_length(self):
        return len(self.text)


def build_streaming_app(path, client, monkeypatch, **config):
    """`build_app` with the endpoint's incremental detokenizer replaced by `TextDetokenizer`."""
    app = build_app(path, client, **config)
    module = importlib.import_module(f"{_ENDPOINTS_PACKAGE}.{_ENDPOINT_MODULES[path]}")
    monkeypatch.setattr(module, "HuggingFaceFastIncrementalDetokenizer", TextDetokenizer)
    return app


async def sse_payloads(response):
    """The JSON payloads of an SSE response, and whether it ended with `[DONE]`."""
    records = [r for r in (await response.get_data(as_text=True)).split("\n\n") if r]
    done = records[-1] == "data: [DONE]"
    payloads = [
        json.loads(record.removeprefix("data: ")) for record in records[: -1 if done else None]
    ]
    return payloads, done


@pytest.mark.asyncio
@PATHS
async def test_streaming_requires_a_fast_tokenizer(path):
    pytest.importorskip("transformers")
    client = ReplyingClient()
    app = build_app(path, client)

    response = await app.test_client().post(path, json={**BODIES[path], "stream": True})

    assert response.status_code == 400
    assert "Hugging Face fast tokenizers" in await response.get_data(as_text=True)
    assert client.prompt_tokens == []


@pytest.mark.asyncio
@PATHS
@pytest.mark.parametrize("include_usage", [False, True])
async def test_streaming_fans_out_and_reports_usage_on_request(path, include_usage, monkeypatch):
    offload_params = {"store": "x"}
    client = ReplyingClient()
    app = build_streaming_app(path, client, monkeypatch)
    body = {
        **FAN_OUT_BODIES[path],
        "stream": True,
        "stream_options": {"include_usage": include_usage},
        "offload_params": offload_params,
    }

    response = await app.test_client().post(path, json=body)

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert response.mimetype == "text/event-stream"
    assert client.streamed == [1, 2, 3]
    assert client.offload_params == [offload_params] * 3
    payloads, done = await sse_payloads(response)
    assert done
    finals = [
        p["choices"][0] for p in payloads if p["choices"] and "generated_text" in p["choices"][0]
    ]
    assert sorted(choice["index"] for choice in finals) == [0, 1, 2]
    assert all(choice["generated_text"] == "<12><13>" for choice in finals)
    usage = [p["usage"] for p in payloads if "usage" in p]
    assert usage == (
        [
            {
                "prompt_tokens": 2,
                "completion_tokens": 6,
                "total_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        ]
        if include_usage
        else []
    )
    assert client.aborted == []
