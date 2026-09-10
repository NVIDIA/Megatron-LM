# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging

logger = logging.getLogger(__name__)


def init_gigatoken_from_hf(tokenizer: "AutoTokenizer", tokenizer_path: str) -> "HFCompat":
    """Initialize gigatoken tokenizer from Hugging Face."""
    try:
        import gigatoken as gt

        logger.info(f"Restoring {tokenizer_path} tokenizer with gigatoken.")
        return gt.Tokenizer(tokenizer).as_hf()
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "gigatoken library is not installed. "
            "Please, install gigatoken to use fast tokenizers: `pip install gigatoken`."
        )
