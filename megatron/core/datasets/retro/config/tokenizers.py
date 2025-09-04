# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Container class for GPT and Bert tokenizers."""

from dataclasses import dataclass

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer


@dataclass
class RetroTokenizers:
    """Container class for GPT and Bert tokenizers."""

    gpt: MegatronTokenizer = None
    bert: MegatronTokenizer = None
