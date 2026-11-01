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


def test_mask_cache_does_not_leak_padding():
    """A sample's loss_mask must not depend on which sample was served first.

    The mask cache is populated with the first sample's mask, which is then masked
    in place for that sample's padding. Caching the tensor itself rather than a copy
    bakes that padding into every subsequent sample, so a dataset instance whose
    first served sample is padded returns wrong masks from then on.
    """
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

    def build_dataset():
        config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=1024,
            split="990,10,0",
            # All three False is what makes the masks cacheable.
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            create_attention_mask=False,
            # Keeps the padded trailing sequence, which is what poisons the cache.
            drop_last_partial_validation_sequence=False,
            add_extra_token_to_sequence=False,
            tokenizer=tokenizer,
            mid_level_dataset_surplus=0.005,
        )
        return BlendedMegatronDatasetBuilder(
            MockGPTDataset, [0, None, 0], lambda: True, config
        ).build()[1]

    # An instance served in index order, so the cache is filled by an unpadded sample.
    expected = build_dataset()[0]["loss_mask"]

    dataset = build_dataset()
    assert dataset.masks_and_position_ids_are_cacheable, "config no longer uses the cache"

    # Serve the padded trailing sequence first, filling the cache from it. Asserting
    # that it really is padded keeps the comparison below from passing vacuously.
    padded_index = int(dataset.shuffle_index.argmax())
    padded_mask = dataset[padded_index]["loss_mask"]
    assert int((padded_mask == 0).sum()) > 1, f"index {padded_index} is no longer padded"

    assert torch.equal(
        dataset[0]["loss_mask"], expected
    ), "padding from the first served sample leaked into a later sample's loss_mask"


if __name__ == "__main__":
    test_mock_gpt_dataset()
