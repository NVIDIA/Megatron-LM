# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reasoning-only response semantics must match the agent-facing OpenAI contract."""

import pytest

from tests.unit_tests.inference.test_completions import (
    _build_app,
    _reply,
    _ReplyingClient,
    chat_completions_blueprint,
)

pytestmark = [pytest.mark.internal]


async def _generate(
    text, *, finish_reason="length", template_kwargs=None, parsers=None, tools=None
):
    budget = 2 if finish_reason == "stop" else 1
    client = _ReplyingClient(
        [
            _reply(
                "forced",
                [10, 11],
                [12],
                finish_reason=finish_reason,
                sampling_params={"num_tokens_to_generate": budget},
            )
        ]
    )
    app = _build_app(chat_completions_blueprint, client)
    app.config["tokenizer"].detokenize = lambda *args, **kwargs: text
    app.config["parsers"] = (
        parsers if parsers is not None else ["nemotron-v3-reasoning", "qwen3-coder-tool"]
    )
    body = {
        "messages": [{"role": "user", "content": "task"}],
        "max_completion_tokens": budget,
        "chat_template_kwargs": template_kwargs or {"enable_thinking": True},
    }
    if tools:
        body.update(tools=tools, tool_choice="auto")
    response = await app.test_client().post("/v1/chat/completions", json=body)
    assert response.status_code == 200, await response.get_data(as_text=True)
    return (await response.get_json())["choices"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("text", ["still thinking", "thinking</think>"])
async def test_reasoning_only_content_is_null(text, finish_reason):
    choice = await _generate(text, finish_reason=finish_reason)
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning_content"] in ("still thinking", "thinking")
    assert not choice["message"].get("tool_calls")
    assert choice["finish_reason"] == finish_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"enable_thinking": False}, {"force_nonempty_content": True}])
async def test_explicit_reasoning_fallback_remains_content(kwargs):
    choice = await _generate("still thinking", template_kwargs=kwargs)
    assert choice["message"]["content"] == "still thinking"
    assert not choice["message"].get("reasoning_content")


@pytest.mark.asyncio
async def test_nonempty_final_content_is_preserved():
    choice = await _generate("thinking</think>answer")
    assert choice["message"]["content"] == "answer"
    assert choice["message"]["reasoning_content"] == "thinking"


@pytest.mark.asyncio
async def test_ordinary_empty_content_is_preserved():
    choice = await _generate("", parsers=[])
    assert choice["message"]["content"] == ""


@pytest.mark.asyncio
async def test_reasoning_followed_by_valid_tool_call_is_preserved():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
        }
    ]
    text = "thinking</think><tool_call>\n<function=finish>\n<parameter=message>done</parameter>\n</function>\n</tool_call>"
    choice = await _generate(text, finish_reason="stop", tools=tools)
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "finish"
    assert choice["message"]["content"] == ""
    assert choice["finish_reason"] == "tool_calls"
