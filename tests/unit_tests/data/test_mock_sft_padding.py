# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The mock SFT packer must never put an unembeddable id in the input stream.

``MockSFTDataset`` pads a packed row out to the full pack length. The padding id comes
from the tokenizer, and ``NullTokenizer`` -- which every ``--mock-data`` run uses --
defaults ``pad_id`` to -1. A negative id is fine in the *labels* (vocab-parallel
cross-entropy masks out-of-range targets) but not in the *tokens*: it reaches an
embedding gather and aborts the step with a device-side assert.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.training.datasets.sft_dataset import MockSFTDataset


class _LowLevel:
    """Stand-in for MockSFTLowLevelDataset: fixed-length rows of in-range ids."""

    def __init__(self, seq_len, vocab_size):
        self._row = np.arange(seq_len, dtype=np.int64) % vocab_size

    def __getitem__(self, _idx):
        return self._row


def _dataset(pad_id, *, pack_length=64, row_len=20, vocab_size=128):
    """A MockSFTDataset with its collaborators stubbed; __getitem__ is what we test."""
    ds = MockSFTDataset.__new__(MockSFTDataset)
    ds.dataset = _LowLevel(row_len, vocab_size)
    ds.indices = np.array([0])
    ds.num_samples = 1
    ds.padding_divisor = 1
    ds.config = SimpleNamespace(
        sequence_length=pack_length, tokenizer=SimpleNamespace(eod=vocab_size - 1, pad=pad_id)
    )
    return ds


@pytest.mark.parametrize("pad_id", [-1, 0, 100])
def test_mock_sft_tokens_are_always_embeddable(pad_id):
    sample = _dataset(pad_id)[0]
    assert int(sample['tokens'].min()) >= 0, "a negative token id aborts the embedding gather"
    assert torch.equal(sample['position_ids'], torch.arange(sample['tokens'].numel()))


def _padded_token_positions(sample, pad_id):
    """Input positions that hold padding.

    Tokens and labels are offset by one (next-token prediction), so the first padded
    *label* still pairs with the last real *token*; drop it.
    """
    return sample['labels'].eq(pad_id).nonzero().flatten()[1:]


def test_mock_sft_negative_pad_falls_back_to_eod_for_tokens_only():
    """A negative pad stays the label sentinel; only the token stream substitutes EOD."""
    vocab_size = 128
    sample = _dataset(-1, vocab_size=vocab_size)[0]
    padded_labels = sample['labels'] == -1
    assert padded_labels.any(), "the row should be padded out to the pack length"
    assert torch.all(sample['tokens'][_padded_token_positions(sample, -1)] == vocab_size - 1)
    assert torch.all(sample['loss_mask'][padded_labels] == 0.0), "padding must not add loss"


def test_mock_sft_explicit_pad_id_is_used_verbatim():
    # 100 is outside the 20 ids the stub row carries, so `labels == pad` selects padding only.
    sample = _dataset(100)[0]
    positions = _padded_token_positions(sample, 100)
    assert positions.numel() > 0
    assert torch.all(sample['tokens'][positions] == 100)
