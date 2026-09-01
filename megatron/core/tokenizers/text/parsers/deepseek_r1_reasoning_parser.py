# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from megatron.core.tokenizers.text.parsers.base_parser import BaseParser


class DeepSeekR1ReasoningParser(BaseParser):
    """Parser for DeepSeek R1 style reasoning output."""

    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, str]]:
        """Extract reasoning content delimited by `<think>...</think>` tags.

        Any text before the first `<think>` is discarded.
        When no `</think>` follows, the model is still "thinking": all text is reasoning,
        unless a configured implicit end marker (such as `<tool_call>`) appears.
        Otherwise the text is split at the first reasoning boundary.

        Args:
            text (str): The text to parse.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the unprocessed text
            and a dictionary with the extracted reasoning content.
        """
        # Discard anything before the first `<think>`.
        before, think_open, after = text.partition("<think>")
        remaining = after if think_open else before

        reasoning_end = remaining.find("</think>")
        content_start = reasoning_end + len("</think>")

        for marker in kwargs.get("implicit_reasoning_end_markers", ()):
            marker_index = remaining.find(marker)
            if marker_index >= 0 and (reasoning_end < 0 or marker_index < reasoning_end):
                reasoning_end = marker_index
                # Preserve the marker for the downstream tool parser.
                content_start = marker_index

        if reasoning_end < 0:
            # No closing tag: treat the remaining text as unterminated reasoning.
            reasoning_content, content = remaining, ""
        else:
            reasoning_content = remaining[:reasoning_end]
            content = remaining[content_start:]

        info = {"reasoning": reasoning_content} if reasoning_content else {}
        return content, info
