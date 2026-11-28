# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import numpy as np
import pytest

from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset


def make_fim_masking_dataset(fim_rate, append_eod=True):
    """Two source documents, the second cut off by the sample boundary."""
    documents = [np.array([1, 2, 99 if append_eod else 3]), np.arange(11, 21)]
    dataset = object.__new__(GPTFIMDataset)
    dataset.dataset = SimpleNamespace(
        get=lambda doc, offset=0, length=None: documents[doc][offset:][:length]
    )
    tokenizer = SimpleNamespace(
        ids_to_text=lambda ids, remove_special_tokens: "".join(map(chr, ids)),
        text_to_ids=lambda text: list(map(ord, text)),
    )
    dataset.config = SimpleNamespace(
        inter_document_masking=True,
        sequence_length=10,
        add_extra_token_to_sequence=True,
        tokenizer=SimpleNamespace(eod=99, _tokenizer=tokenizer),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
    )
    dataset.document_index = np.array([0, 1], dtype=np.int32)
    dataset.sample_index = np.array([[0, 0], [1, 7]], dtype=np.int32)
    dataset.shuffle_index = np.array([0], dtype=np.uint32)
    dataset.fim_rate = fim_rate
    dataset.fim_spm_rate = 0.0
    dataset.fim_split_sample = None
    dataset.no_fim_prefix = None
    dataset.prefix_tok_id, dataset.middle_tok_id, dataset.suffix_tok_id = 95, 96, 97
    dataset.pad_tok_id, dataset.eod_tok_id = 98, 99
    dataset.np_rng = np.random.RandomState(1234)
    dataset._pad_token_id = -1
    dataset.masks_and_position_ids_are_cacheable = False
    return dataset


@pytest.mark.parametrize("append_eod", [False, True])
@pytest.mark.parametrize("fim_rate", [0.0, 1.0])
def test_fim_document_lengths(fim_rate, append_eod):
    dataset = make_fim_masking_dataset(fim_rate, append_eod)
    tokens, document_ids, lengths = dataset._query_document_sample_shuffle_indices(0)
    assert tokens.shape == (11,)
    np.testing.assert_array_equal(document_ids, [0, 1])
    assert lengths == ([3, 8] if fim_rate == 0.0 else [6, 5])
    expected_first = [1, 2, 99 if append_eod else 3]
    if fim_rate == 1.0:
        expected_first += [95, 96, 97]
    assert sorted(tokens[: lengths[0]].tolist()) == sorted(expected_first)


@pytest.mark.parametrize("fim_rate", [0.0, 1.0])
def test_fim_packed_metadata(fim_rate):
    dataset = make_fim_masking_dataset(fim_rate)
    sample = dataset[0]
    boundary = 3 if fim_rate == 0.0 else 6
    assert sample["tokens"].shape == (10,)
    assert sample["cu_seqlens"].tolist() == [0, boundary] + [10] * 9
    assert sample["position_ids"].tolist() == list(range(boundary)) + list(range(10 - boundary))
    assert sample["max_seqlen"].item() == max(boundary, 10 - boundary)


def test_fim_truncation_at_document_boundary():
    dataset = make_fim_masking_dataset(1.0)
    dataset.sample_index[1, 1] = 2
    tokens, _, lengths = dataset._query_document_sample_shuffle_indices(0)
    assert tokens.shape == (6,)
    assert lengths == [6]
    assert sorted(tokens.tolist()) == [1, 2, 95, 96, 97, 99]


def test_fim_eod_only_document():
    dataset = make_fim_masking_dataset(1.0)
    dataset.sample_index[0, 1] = 2
    tokens, _, lengths = dataset._query_document_sample_shuffle_indices(0)
    assert tokens.shape == (9,)
    assert lengths == [1, 8]
    assert tokens[0] == 99
