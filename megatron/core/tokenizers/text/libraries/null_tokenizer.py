# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict


class NullTokenizer:
    """
    Synthetic tokenizer for performance benchmarking and debugging

    Args:
        vocab_size: vocabulary size for embedding
        eod_id: id of the end-of-document token. Defaults to ``vocab_size - 1``.
        pad_id: id of the padding token. Defaults to ``-1`` (no pad token).
    """

    def __init__(self, vocab_size, eod_id=None, pad_id=-1, **kwargs):
        """ """
        self._vocab_size = int(vocab_size)
        self._eod_id = int(eod_id) if eod_id is not None else self._vocab_size - 1
        self._pad_id = int(pad_id)

    def text_to_ids(self, text):
        """Converts text to ids."""
        return [int(x) for x in text.split(' ')]

    def ids_to_text(self, ids):
        """Converts ids to text."""
        text = [str(x) for x in ids]
        return ' '.join(text)

    def tokens_to_ids(self, tokens):
        """Converts tokens to ids."""
        return [int(x) for x in tokens]

    def ids_to_tokens(self, ids):
        """Converts ids to tokens."""
        return [str(x) for x in ids]

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Returns offsets."""
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    @property
    def unique_identifiers(self) -> OrderedDict:
        """Property required for use with megatron-core datasets."""
        return OrderedDict(
            {
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "vocab_size": self._vocab_size,
                "eod_id": self._eod_id,
                "pad_id": self._pad_id,
            }
        )

    @property
    def vocab_size(self):
        """Returns vocab size."""
        return self._vocab_size

    @property
    def vocab(self):
        """ """
        raise NotImplementedError

    @property
    def inv_vocab(self):
        """ """
        raise NotImplementedError

    @property
    def cls(self):
        """Returns cls token."""
        return -1

    @property
    def sep(self):
        """Returns sep token."""
        return -1

    @property
    def mask(self):
        """Returns mask token."""
        return -1

    @property
    def eod(self):
        """Returns eod token."""
        return self._eod_id

    @property
    def pad_id(self):
        """Returns pad token."""
        return self._pad_id

    @property
    def additional_special_tokens_ids(self):
        """ """
        return None
