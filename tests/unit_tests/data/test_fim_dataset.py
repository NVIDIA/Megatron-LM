# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.core.datasets.utils import compile_helpers, get_blend_from_list
from megatron.core.tokenizers import MegatronTokenizer
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("spm_rate", [0.0, 1.0])
@pytest.mark.parametrize("split_sample", [None, "python"])
@pytest.mark.usefixtures("ensure_test_data")
def test_fim_gpt_dataset(spm_rate, split_sample):
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path="/opt/data/tokenizers/huggingface",
        metadata_path={"library": "huggingface"},
        additional_special_tokens=["<prefix>", "<middle>", "<suffix>", "<pad>", "<eod>"],
        include_special_tokens=True,
    )
    blend = get_blend_from_list(["/opt/data/datasets/fim/fim_text_document"])
    extra_tokens = {
        "prefix": "<prefix>",
        "middle": "<middle>",
        "suffix": "<suffix>",
        "pad": "<pad>",
        "eod": "<eod>",
    }
    seq_length = 32
    rate = 1.0
    fragment_rate = 1.0
    config = GPTFIMDatasetConfig(
        blend=blend,
        random_seed=1234,
        sequence_length=seq_length,
        split="990,9,1",
        tokenizer=tokenizer,
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        fim_extra_tokens=extra_tokens,
        fim_rate=rate,
        fim_spm_rate=spm_rate,
        fim_fragment_rate=fragment_rate,
        fim_split_sample=split_sample,
    )

    datasets = BlendedMegatronDatasetBuilder(
        GPTFIMDataset, [10, 10, 10], lambda: True, config
    ).build()

    prefix_id = tokenizer.tokenize("<prefix>")[1]
    suffix_id = tokenizer.tokenize("<suffix>")[1]
    middle_id = tokenizer.tokenize("<middle>")[1]

    dataset = datasets[0]
    assert dataset.fim_rate == rate
    assert dataset.fim_spm_rate == spm_rate
    assert dataset.fragment_fim_rate == fragment_rate

    tokens = dataset[0]["tokens"].tolist()
    if split_sample:
        split_sample_id = tokenizer.tokenize(split_sample)[1]
        split_sample_index = tokens.index(split_sample_id)
        assert prefix_id == tokens[split_sample_index + 1]
    if spm_rate == 0.0:
        assert prefix_id == tokens[0]
        assert suffix_id in tokens
        assert middle_id in tokens
        assert tokens.index(suffix_id) < tokens.index(middle_id)
    else:
        assert prefix_id == tokens[0]
        assert suffix_id == tokens[1]
        assert middle_id in tokens


if __name__ == "__main__":
    test_fim_gpt_dataset()


@pytest.mark.parametrize(
    "add_extra_token_to_sequence,sample_index",
    [
        # The last sample of a 10-token document is partial (validation, no drop).
        (True, [[0, 0], [0, 4], [0, 8], [0, 9]]),
        (False, [[0, 0], [0, 4], [0, 8]]),
    ],
)
def test_fim_sample_length_matches_gpt_dataset(add_extra_token_to_sequence, sample_index):
    """With FIM disabled (fim_rate=0), GPTFIMDataset returns the same samples as GPTDataset."""
    document = np.arange(100, 110, dtype=np.int64)
    low_level_dataset = SimpleNamespace(
        get=lambda idx, offset=0, length=None: document[
            offset : None if length is None else offset + length
        ]
    )

    def make(cls):
        dataset = object.__new__(cls)
        dataset.dataset = low_level_dataset
        dataset.config = SimpleNamespace(
            sequence_length=4,
            add_extra_token_to_sequence=add_extra_token_to_sequence,
            tokenizer=None,
        )
        dataset.document_index = np.array([0], dtype=np.int32)
        dataset.sample_index = np.array(sample_index)
        dataset.shuffle_index = np.arange(len(sample_index) - 1)
        dataset._pad_token_id = -1
        dataset.pad_tok_id = -1
        dataset.eod_tok_id = 0
        dataset.fim_rate = 0.0
        dataset.fim_spm_rate = 0.0
        dataset.fim_split_sample = None
        dataset.no_fim_prefix = None
        dataset.prefix_tok_id = dataset.middle_tok_id = dataset.suffix_tok_id = 1
        dataset.np_rng = np.random.RandomState(0)
        return dataset

    gpt, fim = make(GPTDataset), make(GPTFIMDataset)
    for idx in range(len(sample_index) - 1):
        expected = gpt._query_document_sample_shuffle_indices(idx)[0]
        actual = fim._query_document_sample_shuffle_indices(idx)[0]
        assert len(expected) == 4 + add_extra_token_to_sequence
        np.testing.assert_array_equal(actual, expected)
