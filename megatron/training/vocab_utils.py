# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import math
from functools import lru_cache

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer, vocab_size_with_padding
from megatron.training.config.training_config import TokenizerConfig


def calculate_padded_vocab_size(
    vocab_size: int,
    make_vocab_size_divisible_by: int,
    tensor_model_parallel_size: int,
    logging_enabled: bool = True,
) -> int:
    """Calculate padded vocab size for tensor parallelism.

    This function pads the vocabulary size to ensure it's divisible by the required
    multiple for efficient tensor parallel operations.

    Args:
        vocab_size: The original (unpadded) vocabulary size
        make_vocab_size_divisible_by: Base divisibility requirement (e.g., 128)
        tensor_model_parallel_size: Number of tensor parallel ranks
        logging_enabled: Whether to log the padding information

    Returns:
        int: The padded vocabulary size
    """
    padded_size = _calculate_padded_vocab_size_cached(
        vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size
    )

    # Handle logging separately to avoid affecting cache behavior
    if logging_enabled:
        print_rank_0(
            " > padded vocab (size: {}) with {} dummy tokens (new size: {})".format(
                vocab_size, padded_size - vocab_size, padded_size
            )
        )

    return padded_size


@lru_cache(maxsize=128)
def _calculate_padded_vocab_size_cached(
    vocab_size: int,
    make_vocab_size_divisible_by: int,
    tensor_model_parallel_size: int,
) -> int:
    """Cached computation of padded vocab size."""
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if make_vocab_size_divisible_by <= 0:
        raise ValueError(f"make_vocab_size_divisible_by must be positive, got {make_vocab_size_divisible_by}")
    if tensor_model_parallel_size <= 0:
        raise ValueError(f"tensor_model_parallel_size must be positive, got {tensor_model_parallel_size}")

    multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
    return int(math.ceil(vocab_size / multiple) * multiple)


def build_tokenizer_and_pad_vocab_size(config: TokenizerConfig, make_vocab_size_divisible_by: int | None = None, tensor_model_parallel_size: int = 1):
    from megatron.training.utils import print_rank_0

    tokenizer = build_tokenizer(config)
    
    if config.pad_vocab_size:
        if config.padded_vocab_size is None:
            assert isinstance(make_vocab_size_divisible_by, int), "must provide value for 'make_vocab_size_divisible_by' if 'pad_vocab_size' is set."
            config.padded_vocab_size = vocab_size_with_padding(
                tokenizer.vocab_size,
                make_vocab_size_divisible_by,
                tensor_model_parallel_size,
            )
        else:
            print_rank_0("padded_vocab_size already configured on TokenizerConfig. Skipping divisibility padding calculation and update.")

    return tokenizer
