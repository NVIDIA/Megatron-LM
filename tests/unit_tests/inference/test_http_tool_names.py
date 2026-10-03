# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tool function names must be normalized before dispatch and schema lookup."""

import json

import pytest

from tests.unit_tests.inference.test_completions import (
    _build_app,
    _reply,
    _ReplyingClient,
    chat_completions_blueprint,
)

pytestmark = [pytest.mark.internal]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute",
            "parameters": {"type": "object", "properties": {"count": {"type": "integer"}}},
        },
    }
]


async def _generate(name, *, request_tools=True, parsers=None):
    text = f"<tool_call><function={name}><parameter=count>7</parameter>" "</function></tool_call>"
    app = _build_app(
        chat_completions_blueprint,
        _ReplyingClient([_reply("forced", [10, 11], [12], finish_reason="stop")]),
    )
    app.config["tokenizer"].detokenize = lambda *args, **kwargs: text
    app.config["parsers"] = ["qwen3-coder-tool"] if parsers is None else parsers
    body = {"messages": [{"role": "user", "content": "task"}]}
    if request_tools:
        body.update(tools=TOOLS, tool_choice="auto")
    response = await app.test_client().post("/v1/chat/completions", json=body)
    assert response.status_code == 200, await response.get_data(as_text=True)
    return text, (await response.get_json())["choices"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["execute", " execute", "execute ", " \texecute\n"])
async def test_http_normalizes_tool_name_before_argument_schema_lookup(name):
    _, choice = await _generate(name)
    call = choice["message"]["tool_calls"][0]["function"]
    assert call["name"] == "execute"
    assert json.loads(call["arguments"]) == {"count": 7}
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_http_does_not_emit_whitespace_only_function_name():
    text, choice = await _generate(" \t\n")
    assert not choice["message"].get("tool_calls")
    assert choice["message"]["content"] == text
    assert choice["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"request_tools": False}, {"parsers": []}])
async def test_http_plain_chat_preserves_tool_like_text(kwargs):
    text, choice = await _generate(" execute ", **kwargs)
    assert not choice["message"].get("tool_calls")
    assert choice["message"]["content"] == text
