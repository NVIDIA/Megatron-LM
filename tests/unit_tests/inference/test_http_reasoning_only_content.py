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
    text,
    *,
    finish_reason="length",
    template_kwargs=None,
    parsers=None,
    tools=None,
    generated_tokens=None,
):
    budget = 2 if finish_reason == "stop" else 1
    client = _ReplyingClient(
        [
            _reply(
                "forced",
                [10, 11],
                [12] if generated_tokens is None else generated_tokens,
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
@pytest.mark.parametrize("reasoning", ["thinking", ""])
async def test_reasoning_followed_by_valid_tool_call_is_preserved(reasoning):
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
    text = (
        reasoning
        + "</think><tool_call>\n<function=finish>\n<parameter=message>done</parameter>\n</function>\n</tool_call>"
    )
    choice = await _generate(text, finish_reason="stop", tools=tools)
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "finish"
    assert choice["message"]["content"] == ""
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("text", ["", "</think>", "<think></think>"])
async def test_empty_reasoning_content_is_null(text, finish_reason):
    choice = await _generate(text, finish_reason=finish_reason)
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning_content"] == ""
    assert not choice["message"].get("tool_calls")
    assert choice["finish_reason"] == finish_reason


@pytest.mark.asyncio
async def test_immediate_eos_with_empty_decoding_is_reasoning_only():
    choice = await _generate("", finish_reason="stop", generated_tokens=[2])
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning_content"] == ""
    assert choice["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("text", ["", "</think>", "<think></think>"])
@pytest.mark.parametrize("kwargs", [{"enable_thinking": False}, {"force_nonempty_content": True}])
async def test_empty_reasoning_explicit_fallback_is_ordinary_content(text, finish_reason, kwargs):
    choice = await _generate(text, finish_reason=finish_reason, template_kwargs=kwargs)
    assert choice["message"]["content"] == ""
    assert "reasoning_content" not in choice["message"]
    assert choice["finish_reason"] == finish_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("text", ["", "</think>", "<think></think>"])
async def test_no_parser_does_not_reinterpret_empty_reasoning(text, finish_reason):
    choice = await _generate(text, finish_reason=finish_reason, parsers=[])
    assert choice["message"]["content"] == text
    assert "reasoning_content" not in choice["message"]
    assert choice["finish_reason"] == finish_reason


@pytest.mark.asyncio
async def test_empty_reasoning_with_final_answer_preserves_answer():
    choice = await _generate("<think></think>answer")
    assert choice["message"]["content"] == "answer"
    assert choice["message"]["reasoning_content"] == ""
