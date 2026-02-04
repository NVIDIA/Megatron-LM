# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)
from megatron.core.tokenizers.text.parsers.qwen3_coder_tool_parser import Qwen3CoderToolParser

PARSER_MAPPING = {
    "deepseek-r1-reasoning": DeepSeekR1ReasoningParser,
    "qwen3-coder-tool": Qwen3CoderToolParser,
}

__all__ = ["PARSER_MAPPING"]
