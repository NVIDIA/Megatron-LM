# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


class MegatronNullMultimodalTokenizer:
    """Megatron Null Multimodal Tokenizer"""

    def __init__(self, vocab_size, image_token=None, image_token_id=None):
        """ """
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

        from megatron.core.models.multimodal.llava_model import (
            DEFAULT_IMAGE_TOKEN_INDEX,
            IMAGE_TOKEN,
        )

        self._image_token = image_token if image_token is not None else IMAGE_TOKEN
        self._image_token_id = (
            image_token_id if image_token_id is not None else DEFAULT_IMAGE_TOKEN_INDEX
        )

    def tokenize(self, text):
        """
        Text tokenization.

        Args:
            text (str | list): text to be tokenized.

        Returns:
            list: list of ids.
        """
        return [int(x) for x in text.split(' ')]

    def detokenize(self, ids):
        """
        Text detokenization.

        Args:
            ids (list): text to be tokenized.

        Returns:
            text: detokenized text.
        """
        text = [str(x) for x in ids]
        return ' '.join(text)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Offsets calculation."""
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to IDs."""
        ids = [
            (int(t) if t != self._image_token else self._image_token_id) for t in tokens.split('  ')
        ]
        return ids if len(ids) > 1 else ids[0]

    @property
    def vocab_size(self):
        """Vocab size."""
        return self._vocab_size_without_eod + 1

    @property
    def cls(self):
        """CLS token id."""
        return -1

    @property
    def sep(self):
        """SEP token id."""
        return -1

    @property
    def mask(self):
        """MASK token id."""
        return -1

    @property
    def eod(self):
        """EOD token id."""
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        """Returns IDs of additional special tokens."""
        return None
