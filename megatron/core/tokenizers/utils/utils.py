# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging

logger = logging.getLogger(__name__)


def has_gigatoken_support() -> bool:
    """Check if gigatoken library is installed."""
    try:
        import gigatoken

        return True
    except ModuleNotFoundError:
        return False


def init_gigatoken_from_hf(tokenizer: "AutoTokenizer", tokenizer_path: str) -> "HFCompat":
    """Initialize gigatoken tokenizer from Hugging Face."""
    if has_gigatoken_support():
        import gigatoken as gt

        logger.info(f"Restoring {tokenizer_path} tokenizer with gigatoken.")
        return gt.Tokenizer(tokenizer).as_hf()
    else:
        raise ModuleNotFoundError(
            "gigatoken library is not installed. "
            "Please, install gigatoken to use fast tokenizers: `pip install gigatoken`."
        )
