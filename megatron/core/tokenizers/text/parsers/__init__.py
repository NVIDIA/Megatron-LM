# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)
from megatron.core.tokenizers.text.parsers.qwen3_coder_tool_parser import Qwen3CoderToolParser
from megatron.core.tokenizers.text.parsers.super_v3_reasoning_parser import SuperV3ReasoningParser
from megatron.core.tokenizers.text.parsers.ultra_v3_reasoning_parser import UltraV3ReasoningParser

PARSER_MAPPING = {
    "deepseek-r1-reasoning": DeepSeekR1ReasoningParser,
    "qwen3-coder-tool": Qwen3CoderToolParser,
    "super-v3-reasoning": SuperV3ReasoningParser,
    "ultra-v3-reasoning": UltraV3ReasoningParser,
}

__all__ = ["PARSER_MAPPING"]
