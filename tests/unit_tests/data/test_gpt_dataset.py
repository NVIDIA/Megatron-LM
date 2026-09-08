# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

##
# Compile megatron.core.datasets.helpers_cpp dependencies before BlendedDataset import
##

import random

import numpy
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.utils import _merge_cu_seqlens_across_micro_batch
from tests.unit_tests.test_utilities import Utils

_MOCK_VOCAB_SIZE = 8192


def sample_N(dataset, N, randomize):
    if randomize:
        indices = [random.randint(0, len(dataset) - 1) for _ in range(N)]
    else:
        indices = list(range(N))
    samples = [dataset[index]["tokens"].numpy() for index in indices]
    return samples


def test_mock_gpt_dataset():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,9,1",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [100, 100, 100], lambda: True, config
    ).build()

    N = 10

    # Check iso-index variance by split
    subsets = [sample_N(dataset, N, randomize=False) for dataset in datasets]
    assert not numpy.allclose(subsets[0], subsets[1])
    assert not numpy.allclose(subsets[0], subsets[2])
    assert not numpy.allclose(subsets[1], subsets[2])

    # Check iso-split / iso-index identity
    subset_1A = sample_N(datasets[0], N, randomize=False)
    subset_1B = sample_N(datasets[0], N, randomize=False)
    assert numpy.allclose(subset_1A, subset_1B)

    # Check iso-split variance by index
    subset_1A = sample_N(datasets[0], N, randomize=True)
    subset_1B = sample_N(datasets[0], N, randomize=True)
    assert not numpy.allclose(subset_1A, subset_1B)

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,10,0",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        drop_last_partial_validation_sequence=False,
        add_extra_token_to_sequence=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [0, None, 0], lambda: True, config
    ).build()

    sample = datasets[1][datasets[1].shuffle_index.argmax()]
    argmax = sample['labels'].shape[0] - torch.flip(sample['labels'], [0]).argmax() - 1

    # Test add_extra_token_to_sequence
    assert sample['tokens'][argmax] != tokenizer.eod
    assert sample['labels'][argmax] == tokenizer.eod

    # Test eod_mask_loss, drop_last_partial_validation_sequence
    assert argmax < sample['labels'].shape[0] - 1
    assert torch.all(sample['labels'][argmax + 1 :] == 0)
    assert not torch.any(
        sample['loss_mask'][
            torch.logical_and(sample['labels'] == tokenizer.eod, sample['labels'] == 0)
        ]
    )

    sample = datasets[1][None]

    # Check handling of None index
    assert not torch.any(sample['loss_mask'])


def test_inter_document_masking():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )

    sequence_length = 1024

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=sequence_length,
        split="990,9,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
        inter_document_masking=True,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [100, 100, 100], lambda: True, config
    ).build()

    N = 20
    for idx in range(N):
        sample = datasets[0][idx]

        assert "cu_seqlens" in sample
        assert "max_seqlen" in sample
        assert "attention_mask" not in sample

        # Strip collation padding before validation.
        cu_seqlens = _merge_cu_seqlens_across_micro_batch(
            sample["cu_seqlens"].unsqueeze(0), sequence_length
        )
        max_seqlen = sample["max_seqlen"]
        tokens = sample["tokens"]
        position_ids = sample["position_ids"]

        assert tokens.shape[0] == sequence_length
        assert position_ids.shape[0] == sequence_length

        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0] == 0
        assert cu_seqlens[-1] == sequence_length

        # cu_seqlens must be strictly increasing.
        diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        assert torch.all(diffs > 0), f"cu_seqlens not strictly increasing: {cu_seqlens}"

        assert max_seqlen == diffs.max()

        # Position IDs must reset to 0 at each document boundary.
        for i in range(cu_seqlens.numel() - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            expected = torch.arange(end - start, dtype=torch.long)
            assert torch.equal(
                position_ids[start:end], expected
            ), f"position_ids mismatch in segment {i} [{start}:{end}]"

    # Verify that None index zeros out loss_mask.
    sample = datasets[0][None]
    assert not torch.any(sample["loss_mask"])
    assert "cu_seqlens" in sample


def _build_padded_mock_gpt_dataset(tokenizer):
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,10,0",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        drop_last_partial_validation_sequence=False,
        add_extra_token_to_sequence=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )
    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [0, None, 0], lambda: True, config
    ).build()
    return datasets[1]


def _padded_validation_sample(dataset):
    return dataset[dataset.shuffle_index.argmax()]


def _last_eod_label_index(labels, eod):
    eod_positions = (labels == eod).nonzero(as_tuple=True)[0]
    assert eod_positions.numel() > 0
    return int(eod_positions[-1].item())


def test_gpt_dataset_pad_token_id_remapping():
    """In-vocab pad ids stay unchanged; OOV sentinels remap to a safe embedding id."""
    unique_pad_id = 7
    in_vocab_tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"},
        vocab_size=_MOCK_VOCAB_SIZE,
        pad_id=unique_pad_id,
    )
    assert in_vocab_tokenizer.pad == unique_pad_id
    assert in_vocab_tokenizer.pad != in_vocab_tokenizer.eod
    assert 0 < unique_pad_id < in_vocab_tokenizer.vocab_size

    in_vocab_dataset = _build_padded_mock_gpt_dataset(in_vocab_tokenizer)
    assert in_vocab_dataset._pad_token_id == unique_pad_id

    in_vocab_sample = _padded_validation_sample(in_vocab_dataset)
    in_vocab_eod_idx = _last_eod_label_index(in_vocab_sample["labels"], in_vocab_tokenizer.eod)
    assert in_vocab_eod_idx < in_vocab_sample["labels"].shape[0] - 1

    in_vocab_pad_labels = in_vocab_sample["labels"][in_vocab_eod_idx + 1 :]
    assert torch.all(in_vocab_pad_labels == unique_pad_id)
    assert not torch.any(in_vocab_pad_labels == 0)
    assert torch.all(in_vocab_sample["loss_mask"][in_vocab_eod_idx + 1 :] == 0)
    assert in_vocab_sample["tokens"][in_vocab_eod_idx + 1] == in_vocab_tokenizer.eod
    assert torch.all(in_vocab_sample["tokens"][in_vocab_eod_idx + 2 :] == unique_pad_id)

    sentinel_tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE, pad_id=-1
    )
    sentinel_dataset = _build_padded_mock_gpt_dataset(sentinel_tokenizer)
    assert sentinel_dataset._pad_token_id == -1

    sentinel_sample = _padded_validation_sample(sentinel_dataset)
    sentinel_eod_idx = _last_eod_label_index(sentinel_sample["labels"], sentinel_tokenizer.eod)
    assert sentinel_eod_idx < sentinel_sample["labels"].shape[0] - 1

    sentinel_pad_labels = sentinel_sample["labels"][sentinel_eod_idx + 1 :]
    assert torch.all(sentinel_pad_labels == 0)
    assert torch.all(sentinel_sample["tokens"] >= 0)
    assert torch.all(sentinel_sample["labels"] >= 0)
    assert torch.all(sentinel_sample["tokens"] < sentinel_tokenizer.vocab_size)
    assert torch.all(sentinel_sample["labels"] < sentinel_tokenizer.vocab_size)
    assert torch.all(sentinel_sample["loss_mask"][sentinel_eod_idx + 1 :] == 0)


if __name__ == "__main__":
    test_mock_gpt_dataset()
