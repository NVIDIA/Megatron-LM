# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Container class for GPT and Bert tokenizers."""

from dataclasses import dataclass

from megatron.core.tokenizers import MegatronTokenizerBase


@dataclass
class RetroTokenizers:
    """Container class for GPT and Bert tokenizers."""

    gpt: MegatronTokenizerBase = None
    bert: MegatronTokenizerBase = None
