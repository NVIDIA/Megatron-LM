# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The completion endpoints abort in-flight requests when the client goes away.

The disconnect path is driven through Quart's ASGI interface rather than its
test client, because the cancellation under test only exists there:
ASGIHTTPConnection races handle_messages against handle_request and cancels the
loser, so an ``http.disconnect`` message is what raises CancelledError inside
the handler. A test-client request always runs to completion and never
exercises that path.
"""

import asyncio
import importlib
import json
from unittest.mock import MagicMock

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    abort_requests,
)

pytestmark = [pytest.mark.internal, pytest.mark.asyncio]

_ENDPOINTS_PACKAGE = (
    "megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints"
)


class _ToyTokenizer:
    def tokenize(self, text):
        return [len(text)]

    def detokenize(self, token_ids):
        return "toy prompt"


class _NeverCompletingClient:
    """Admits requests the engine never answers, and records the aborts.

    A future that never resolves is the situation the abort exists for: a
    non-streaming handler writes nothing to the socket while it generates, so
    the disconnect is never discovered as a broken pipe and the engine keeps
    the slot until the token limit.
    """

    def __init__(self, expected_admissions):
        self.expected_admissions = expected_admissions
        self.next_request_id = 0
        self.submitted = []
        self.aborted = []
        self.all_admitted = asyncio.Event()

    def add_request_with_id(self, prompt, sampling_params, *, multi_modal_data=None):
        request_id = self.next_request_id
        self.next_request_id += 1
        self.submitted.append(request_id)
        if len(self.submitted) >= self.expected_admissions:
            self.all_admitted.set()
        return request_id, asyncio.get_running_loop().create_future()

    def abort_request(self, request_id):
        self.aborted.append(request_id)


def _build_app(module_name, client):
    quart = pytest.importorskip("quart")
    blueprint = importlib.import_module(f"{_ENDPOINTS_PACKAGE}.{module_name}").bp

    app = quart.Quart(__name__)
    app.config.update(
        client=client,
        tokenizer=_ToyTokenizer(),
        verbose=False,
        parsers=[],
        multimodal_prompt_config=None,
        chat_template=None,
    )
    app.register_blueprint(blueprint)
    return app


async def _post_then_disconnect(app, path, body, client):
    """POST, wait until every request is admitted, then drop the connection."""
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
    # Disconnecting before admission would prove nothing -- there would be
    # nothing in flight to abort.
    await asyncio.wait_for(client.all_admitted.wait(), timeout=10)
    incoming.put_nowait({"type": "http.disconnect"})
    await asyncio.wait_for(connection, timeout=10)
    return sent


@pytest.mark.parametrize(
    ("module_name", "path", "body"),
    [
        pytest.param(
            "completions",
            "/v1/completions",
            {"prompt": [[1, 2, 3], [4, 5, 6]], "max_tokens": 128},
            id="completions",
        ),
        pytest.param(
            "chat_completions",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}], "n": 2, "max_tokens": 128},
            id="chat_completions",
        ),
    ],
)
async def test_disconnect_aborts_every_in_flight_request(module_name, path, body):
    """Both fan-out admissions are aborted, and no response is written.

    Two requests per call (two prompts for completions, n=2 for chat) so a
    handler that aborted only the first would fail here.
    """
    client = _NeverCompletingClient(expected_admissions=2)
    app = _build_app(module_name, client)

    sent = await _post_then_disconnect(app, path, body, client)

    assert client.submitted == [0, 1]
    assert client.aborted == [0, 1]
    # CancelledError is re-raised rather than converted to a 500: the peer is
    # gone, so there is nobody to send a status to.
    assert sent == []


async def test_abort_requests_helper_is_best_effort():
    """One failing abort must not stop the others, and must not raise.

    abort_requests runs on paths that are already unwinding -- the cancelled
    handler above, or one whose submission loop failed partway and is returning
    a 500 -- where letting a second exception escape would replace the real one.
    """
    client = MagicMock()
    client.abort_request.side_effect = [None, RuntimeError("coordinator gone"), None]

    abort_requests(client, [7, 8, 9], "client disconnected")

    assert [call.args[0] for call in client.abort_request.call_args_list] == [7, 8, 9]

    client.reset_mock(side_effect=True)
    abort_requests(client, [], "client disconnected")
    client.abort_request.assert_not_called()
