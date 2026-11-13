# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""cls, sep and inv_vocab are read once per masked-LM sample."""

from types import SimpleNamespace

import numpy

from megatron.core.datasets.masked_dataset import MaskedWordPieceDataset

CLS, SEP, MASK = 1, 2, 3


class _CountingTokenizer:
    def __init__(self):
        self.inv = {i: (f"##w{i}" if i % 5 == 0 else f"w{i}") for i in range(64)}
        self.reads = {"cls": 0, "sep": 0, "inv_vocab": 0}

    @property
    def cls(self):
        self.reads["cls"] += 1
        return CLS

    @property
    def sep(self):
        self.reads["sep"] += 1
        return SEP

    @property
    def inv_vocab(self):
        self.reads["inv_vocab"] += 1
        return self.inv


class _Dataset(MaskedWordPieceDataset):
    def __init__(self, config):
        self.config = config

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_token_mask(self, numpy_random_state):
        return MASK


def test_tokenizer_fields_are_read_once_per_sample():
    tokenizer = _CountingTokenizer()
    dataset = _Dataset(
        SimpleNamespace(
            tokenizer=tokenizer,
            masking_probability=0.15,
            masking_max_ngram=3,
            masking_do_full_word=True,
            masking_do_permutation=False,
            masking_use_longer_ngrams=False,
            masking_use_geometric_distribution=False,
        )
    )
    token_ids = [CLS] + [4 + (i % 50) for i in range(200)] + [SEP]
    masked_token_ids, masked_positions, masked_labels, boundaries, _ = (
        dataset._create_masked_lm_predictions(
            token_ids, len(token_ids), numpy.random.RandomState(0)
        )
    )
    assert tokenizer.reads == {"cls": 1, "sep": 1, "inv_vocab": 1}
    assert len(boundaries) == len(token_ids)
    assert len(masked_token_ids) == len(token_ids)
    assert len(masked_positions) == len(masked_labels)
    assert masked_positions == sorted(masked_positions)
