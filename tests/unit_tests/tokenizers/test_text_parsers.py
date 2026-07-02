# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parity tests for the `<think>`/`</think>` reasoning parsers.

Ground truth for `NemotronV3ReasoningParser` is derived from vLLM's actual
implementation (not reimplemented from memory):

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

vLLM's `extract_reasoning` returns `(reasoning, content)` with `None` as the
"absent" sentinel; Megatron's `parse` returns `(content, info)` with `""` as the
"absent" sentinel and omits the `"reasoning"` key from `info` entirely when
reasoning is empty. Expected values below are translated accordingly.
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
    ("<think>hello", {"enable_thinking": False}, "hello", {}),
    ("<think>hello</think>", {"enable_thinking": False}, "hello", {}),
    # force_nonempty_content=True has the same effect as enable_thinking=False.
    ("<think>hello</think>", {"force_nonempty_content": True}, "hello", {}),
    ("<think>hello", {"force_nonempty_content": True}, "hello", {}),
    # The override only fires when there would otherwise be no content.
    (
        "<think>hello</think>world",
        {"enable_thinking": False},
        "world",
        {"reasoning": "hello"},
    ),
    # Text preceding `<think>` is discarded, override still applies past it.
    ("prefix<think>hello</think>", {"enable_thinking": False}, "hello", {}),
    # enable_thinking=True (or omitted) must not trigger the override.
    ("<think>hello</think>", {"enable_thinking": True}, "", {"reasoning": "hello"}),
]


@pytest.mark.parametrize("text,kwargs,expected_content,expected_info", NEMOTRON_V3_CASES)
def test_nemotron_v3_reasoning_parser_matches_vllm(text, kwargs, expected_content, expected_info):
    content, info = NemotronV3ReasoningParser.parse(text, **kwargs)
    assert content == expected_content
    assert info == expected_info


@pytest.mark.parametrize(
    "text",
    [
        "<think>hello",
        "<think>hello</think>world",
        "<think>hello</think>",
        "just an answer",
    ],
)
def test_nemotron_v3_reasoning_parser_without_override_matches_deepseek_r1(text):
    """With no `enable_thinking`/`force_nonempty_content` kwargs, the Nemotron 3
    parser must be observably identical to the DeepSeek R1 parser it extends."""
    assert NemotronV3ReasoningParser.parse(text) == DeepSeekR1ReasoningParser.parse(text)


def test_parser_mapping_registers_nemotron_v3_reasoning():
    """Super and Ultra share identical reasoning-extraction logic upstream, so
    both models are served by a single consolidated parser and registry key."""
    assert PARSER_MAPPING["nemotron-v3-reasoning"] is NemotronV3ReasoningParser
