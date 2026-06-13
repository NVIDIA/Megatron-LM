# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)


def test_deepseek_reasoning_parser_leaves_plain_content_unchanged():
    text = "Final answer without reasoning tags."

    content, metadata = DeepSeekR1ReasoningParser.parse(text)

    assert content == text
    assert metadata == {}


def test_deepseek_reasoning_parser_extracts_tagged_reasoning():
    content, metadata = DeepSeekR1ReasoningParser.parse(
        "prefix <think>hidden reasoning</think> visible answer"
    )

    assert content == "prefix  visible answer"
    assert metadata == {"reasoning": "hidden reasoning"}


def test_deepseek_reasoning_parser_infers_missing_opening_tag():
    content, metadata = DeepSeekR1ReasoningParser.parse("hidden reasoning</think> visible answer")

    assert content == " visible answer"
    assert metadata == {"reasoning": "hidden reasoning"}
