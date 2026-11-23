# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Any


class BaseParser:
    """Base class for text parsers."""

    # Tool parsers set this to strip the content of a non-streamed turn that carries tool calls,
    # as vLLM's qwen3_coder parser does (vLLM's DeepSeek V3.2/V4 and Kimi K2 parsers do not).
    # Whitespace-only content of such a turn is dropped for every parser.
    strip_content_with_tool_calls: bool = False

    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, Any]]:
        """
        Parses the text into a tuple containing extracted content
        and a dictionary of additional information.

        Args:
            text (str): The text to parse.

        Returns:
            tuple[str, dict[str, Any]]: A tuple containing the unprocessed text
            and a dictionary with the extracted information.
        """
        return text, {}
