# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Shared utilities for modelopt post-training scripts."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from megatron.training import get_tokenizer


def get_hf_tokenizer():
    """Return the underlying HuggingFace tokenizer, unwrapping Megatron-Core nesting.

    Megatron-Core tokenizers are nested (e.g. get_tokenizer()._tokenizer may itself
    have a .tokenizer or ._tokenizer attribute holding the actual HF tokenizer).
    This helper unwraps one level of that nesting.
    """
    tokenizer = get_tokenizer()._tokenizer
    tok_attrs = ["tokenizer", "_tokenizer"]
    for attr in tok_attrs:
        if hasattr(tokenizer, attr):
            tokenizer = getattr(tokenizer, attr)
            break
    return tokenizer
