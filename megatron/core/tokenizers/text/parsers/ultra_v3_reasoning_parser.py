# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)


class UltraV3ReasoningParser(DeepSeekR1ReasoningParser):
    """Parser for NVIDIA Nemotron 3 Ultra reasoning output.

    Behaves like `DeepSeekR1ReasoningParser`, except when reasoning is disabled
    via `enable_thinking=False`, or the caller passes `force_nonempty_content=True`:
    in that case, an unterminated `<think>` block (no closing `</think>` tag, e.g.
    because reasoning exceeded the max length) is returned as content instead of
    being discarded as reasoning, so callers always get a non-empty response.
    """

    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, str]]:
        """Extract reasoning content delimited by `<think>...</think>` tags.

        Args:
            text (str): The text to parse.
            enable_thinking (bool, optional): Whether reasoning is enabled for
                this request. When `False`, unterminated reasoning is surfaced
                as content rather than discarded.
            force_nonempty_content (bool, optional): When `True`, unterminated
                reasoning is surfaced as content rather than discarded.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the unprocessed text
            and a dictionary with the extracted reasoning content.
        """
        # Discard anything before the first `<think>`.
        before, think_open, after = text.partition("<think>")
        remaining = after if think_open else before

        has_close_tag = "</think>" in remaining
        if has_close_tag:
            reasoning_content, _, content = remaining.partition("</think>")
        else:
            # No closing tag: treat the remaining text as unterminated reasoning.
            reasoning_content, content = remaining, ""

        if (
            not has_close_tag
            and reasoning_content
            and (kwargs.get("enable_thinking") is False or kwargs.get("force_nonempty_content") is True)
        ):
            content, reasoning_content = reasoning_content, ""

        info = {"reasoning": reasoning_content} if reasoning_content else {}
        return content, info
