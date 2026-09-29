# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Tests for evaluating pretraining-format validation sets during SFT."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.datasets.utils import Split
from megatron.training.datasets.sft_dataset import (
    IGNORE_INDEX,
    PretrainValidAsSFTDataset,
    SFTDataset,
)

SEQ_LENGTH = 16
EOD = 1
PAD = 99


class FakeGPTDataset:
    """Mimics the GPTDataset sample dict (without attention_mask)."""

    index_split = Split.valid

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.arange(idx, idx + SEQ_LENGTH, dtype=torch.int64)
        return {
            'tokens': tokens,
            'labels': tokens + 1,
            'loss_mask': torch.ones(SEQ_LENGTH, dtype=torch.float32),
            'position_ids': torch.arange(SEQ_LENGTH, dtype=torch.int64),
        }


@pytest.mark.parametrize("max_samples,expected_len", [(None, 10), (4, 4), (50, 10)])
def test_pretrain_valid_length_cap(max_samples, expected_len):
    ds = PretrainValidAsSFTDataset(FakeGPTDataset(10), max_samples)
    assert len(ds) == expected_len
    assert ds.index_split == Split.valid


def test_pretrain_valid_packed_sample_is_one_document():
    sample = PretrainValidAsSFTDataset(FakeGPTDataset(3), packed=True)[2]
    source = FakeGPTDataset(3)[2]
    for key in ('tokens', 'labels', 'loss_mask', 'position_ids'):
        assert torch.equal(sample[key], source[key])
    # One document covering the sequence, padded like SFTDataset's cu_seqlens.
    assert sample['cu_seqlens'].shape == (SEQ_LENGTH + 1,)
    assert sample['cu_seqlens'].dtype == torch.int32
    assert sample['cu_seqlens'][0] == 0
    assert torch.all(sample['cu_seqlens'][1:] == SEQ_LENGTH)
    assert sample['max_seqlen'] == SEQ_LENGTH


def test_pretrain_valid_unpacked_sample_has_no_thd_fields():
    sample = PretrainValidAsSFTDataset(FakeGPTDataset(3), packed=False)[0]
    assert set(sample) == {'tokens', 'labels', 'loss_mask', 'position_ids'}


class FakeTokenizer:
    """Tokenizes each message's content as a list of ints; prompt tokens are masked."""

    eod = EOD
    pad = PAD

    def tokenize_conversation(self, conversation, return_target, add_generation_prompt):
        tokens, targets = [], []
        for message in conversation:
            ids = message['content']
            tokens.extend(ids)
            masked = message['role'] in ('system', 'user')
            targets.extend([IGNORE_INDEX] * len(ids) if masked else ids)
        return np.array(tokens), np.array(targets)


def make_sft_dataset(conversations, cross_document_attention, num_samples=None):
    """Build an SFTDataset without the file-backed MegatronDataset constructor."""
    ds = SFTDataset.__new__(SFTDataset)
    ds.dataset = conversations
    ds.indices = np.arange(len(conversations))
    ds.num_samples = num_samples
    ds.padding_divisor = 1
    ds.config = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        sequence_length=SEQ_LENGTH,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
        sft_cross_document_attention=cross_document_attention,
    )
    return ds


PACK = [
    {'role': 'system', 'content': []},
    {'role': 'user', 'content': [10, 11, 12]},
    {'role': 'assistant', 'content': [20, 21, EOD]},
    {'role': 'system', 'content': []},
    {'role': 'user', 'content': [30, 31]},
    {'role': 'assistant', 'content': [40, EOD]},
]


@pytest.mark.parametrize("num_samples,expected_len", [(None, 2), (5, 5)])
def test_sft_len_defaults_to_one_pass(num_samples, expected_len):
    ds = make_sft_dataset([PACK, PACK], cross_document_attention=False, num_samples=num_samples)
    assert len(ds) == expected_len


def test_sft_cross_document_attention_sample():
    isolated = make_sft_dataset([PACK], cross_document_attention=False)[0]
    shared = make_sft_dataset([PACK], cross_document_attention=True)[0]

    # Plain causal attention: no THD fields, positions run on across conversations.
    assert set(shared) == {'tokens', 'labels', 'loss_mask', 'position_ids'}
    assert torch.equal(shared['position_ids'], torch.arange(SEQ_LENGTH))
    assert 'cu_seqlens' in isolated
    assert isolated['position_ids'][6] == 0  # the second conversation restarts positions

    # Tokens, labels and loss masking do not depend on the attention mode.
    for key in ('tokens', 'labels', 'loss_mask'):
        assert torch.equal(shared[key], isolated[key])
    # Only answer tokens are trained; the first conversation's last answer token predicts EOD,
    # while its EOD (predicting the next conversation's prompt) is masked.
    trained = shared['labels'][shared['loss_mask'].bool()].tolist()
    assert trained == [20, 21, EOD, 40, EOD]
