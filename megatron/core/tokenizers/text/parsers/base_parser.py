# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Any


class BaseParser:
    """Base class for text parsers."""

    # Tool parsers set this to strip a non-streamed turn's content when it carries tool calls
    # (and drop it when only whitespace remains), as vLLM's tool parsers do.
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
