# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parity tests for the `<think>`/`</think>` reasoning parsers.

Ground truth for `NemotronV3ReasoningParser` is derived from vLLM's actual
implementation:

- Base extraction: `BaseThinkingReasoningParser.extract_reasoning` in
  `vllm/reasoning/basic_parsers.py` (used unmodified by `DeepSeekR1ReasoningParser`
  for non-streaming extraction). Notably `final_content = content or None`, so an
  empty string after a closing `</think>` collapses to `None`, same as a missing
  closing tag entirely.
- Override: `SuperV3ReasoningParser`/`UltraV3ReasoningParser.extract_reasoning` in
  `super_v3_reasoning_parser.py`/`ultra_v3_reasoning_parser.py` (from
  huggingface.co/nvidia/NVIDIA-Nemotron-3-{Super,Ultra}-*), which swaps all text
  into content when `final_content is None` and either `enable_thinking is False`
  or `force_nonempty_content is True`.

"""

import pytest

from megatron.core.tokenizers.text.parsers import PARSER_MAPPING
from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)
from megatron.core.tokenizers.text.parsers.nemotron_v3_reasoning_parser import (
    NemotronV3ReasoningParser,
)

# (text, kwargs, expected_content, expected_info)
# `kwargs` is expanded into `parse(text, **kwargs)`; the override flags reach the
# parser inside `chat_template_kwargs`, exactly as the chat-completions endpoint
# forwards them from the request.
NEMOTRON_V3_CASES = [
    # No chat_template_kwargs override: behaves exactly like DeepSeekR1ReasoningParser.
    ("<think>hello", {}, "", {"reasoning": "hello"}),
    ("<think>hello</think>world", {}, "world", {"reasoning": "hello"}),
    # Closing tag present but nothing follows it: vLLM's `content or None` treats
    # this the same as a missing closing tag, so it is empty here too.
    ("<think>hello</think>", {}, "", {"reasoning": "hello"}),
    # No `<think>` tag at all: vLLM assumes the whole string is reasoning.
    ("just an answer", {}, "", {"reasoning": "just an answer"}),
    # enable_thinking=False surfaces would-be-empty content as the reasoning text,
    # for both the "unterminated" and "closes with nothing following" cases.
    ("<think>hello", {"chat_template_kwargs": {"enable_thinking": False}}, "hello", {}),
    ("<think>hello</think>", {"chat_template_kwargs": {"enable_thinking": False}}, "hello", {}),
    # force_nonempty_content=True has the same effect as enable_thinking=False.
    (
        "<think>hello</think>",
        {"chat_template_kwargs": {"force_nonempty_content": True}},
        "hello",
        {},
    ),
    ("<think>hello", {"chat_template_kwargs": {"force_nonempty_content": True}}, "hello", {}),
    # The override only fires when there would otherwise be no content.
    (
        "<think>hello</think>world",
        {"chat_template_kwargs": {"enable_thinking": False}},
        "world",
        {"reasoning": "hello"},
    ),
    # Text preceding `<think>` is discarded, override still applies past it.
    (
        "prefix<think>hello</think>",
        {"chat_template_kwargs": {"enable_thinking": False}},
        "hello",
        {},
    ),
    # enable_thinking=True (or omitted) must not trigger the override.
    (
        "<think>hello</think>",
        {"chat_template_kwargs": {"enable_thinking": True}},
        "",
        {"reasoning": "hello"},
    ),
]


@pytest.mark.parametrize("text,kwargs,expected_content,expected_info", NEMOTRON_V3_CASES)
def test_nemotron_v3_reasoning_parser_matches_vllm(text, kwargs, expected_content, expected_info):
    content, info = NemotronV3ReasoningParser.parse(text, **kwargs)
    assert content == expected_content
    assert info == expected_info


@pytest.mark.parametrize(
    "text", ["<think>hello", "<think>hello</think>world", "<think>hello</think>", "just an answer"]
)
def test_nemotron_v3_reasoning_parser_without_override_matches_deepseek_r1(text):
    """With no `enable_thinking`/`force_nonempty_content` kwargs, the Nemotron 3
    parser must be observably identical to the DeepSeek R1 parser it extends."""
    assert NemotronV3ReasoningParser.parse(text) == DeepSeekR1ReasoningParser.parse(text)


def test_parser_mapping_registers_nemotron_v3_reasoning():
    """Super and Ultra share identical reasoning-extraction logic upstream, so
    both models are served by a single consolidated parser and registry key."""
    assert PARSER_MAPPING["nemotron-v3-reasoning"] is NemotronV3ReasoningParser


def test_tool_call_marker_implicitly_ends_reasoning_for_downstream_parser():
    tool_text = (
        "<tool_call><function=bash><parameter=command>echo hi</parameter>"
        "</function></tool_call>"
    )
    model_output = f"I should inspect this first.\n{tool_text}"
    tool_parser = PARSER_MAPPING["qwen3-coder-tool"]

    content, reasoning_info = DeepSeekR1ReasoningParser.parse(
        model_output,
        implicit_reasoning_end_markers=tool_parser.implicit_reasoning_end_markers,
    )
    parsed_content, tool_info = tool_parser.parse(
        content,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ],
    )

    assert reasoning_info == {"reasoning": "I should inspect this first.\n"}
    assert parsed_content is None
    assert tool_info["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": '{"command": "echo hi"}',
    }


def test_tool_call_marker_does_not_end_reasoning_unless_configured():
    model_output = "reasoning<tool_call>not enabled</tool_call>"

    assert DeepSeekR1ReasoningParser.parse(model_output) == (
        "",
        {"reasoning": model_output},
    )
