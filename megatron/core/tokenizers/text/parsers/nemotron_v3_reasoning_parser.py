# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)


class NemotronV3ReasoningParser(DeepSeekR1ReasoningParser):
    """Parser for NVIDIA Nemotron 3 (Super, Ultra) reasoning output.

    Behaves like `DeepSeekR1ReasoningParser`, except when reasoning is disabled
    via `enable_thinking=False`, or the caller passes `force_nonempty_content=True`:
    in that case, if no content would otherwise be returned (either because
    `</think>` never closes, e.g. reasoning exceeded the max length, or because
    it closes with nothing following it), the reasoning text is returned as
    content instead of being discarded, so callers always get a non-empty
    response.
    """

    @staticmethod
    def _should_force_content(chat_template_kwargs: "dict | None") -> bool:
        """Whether would-be-empty content should be backfilled from reasoning.

        Mirrors vLLM's `SuperV3ReasoningParser._should_force_content`: force
        content when reasoning is disabled (`enable_thinking is False`) or the
        caller explicitly requests it (`force_nonempty_content is True`). Both
        flags are supplied by the client inside `chat_template_kwargs`.
        """
        return bool(
            chat_template_kwargs
            and (
                chat_template_kwargs.get("enable_thinking") is False
                or chat_template_kwargs.get("force_nonempty_content") is True
            )
        )

    @staticmethod
    def parse(text: str, **kwargs) -> tuple[str, dict[str, str]]:
        """Extract reasoning content delimited by `<think>...</think>` tags.

        Delegates the `<think>`/`</think>` split to `DeepSeekR1ReasoningParser`,
        then surfaces the reasoning as content (instead of discarding it) when
        reasoning was disabled or the caller forced non-empty content.

        Args:
            text (str): The text to parse.
            chat_template_kwargs (dict, optional): The request's
                `chat_template_kwargs`. When it sets `enable_thinking=False` or
                `force_nonempty_content=True`, reasoning is surfaced as content
                rather than discarded if there would otherwise be no content.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the unprocessed text
            and a dictionary with the extracted reasoning content.
        """
        content, info = DeepSeekR1ReasoningParser.parse(text, **kwargs)
        if (
            content == ""
            and info.get("reasoning")
            and NemotronV3ReasoningParser._should_force_content(kwargs.get("chat_template_kwargs"))
        ):
            return info["reasoning"], {}
        return content, info
