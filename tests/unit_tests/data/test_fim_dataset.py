# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.core.datasets.utils import compile_helpers, get_blend_from_list
from megatron.core.tokenizers import MegatronTokenizer
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("defer_mmap", [False, True])
def test_fim_cached_indices(defer_mmap, tmp_path):
    """Deferred FIM indices select the same shuffled samples as eager indices."""
    documents = [np.array([1, 2, 3, 4, 99]), np.array([11, 12, 13, 14, 99])]
    dataset = object.__new__(GPTFIMDataset)
    dataset.dataset = SimpleNamespace(
        get=lambda doc, offset=0, length=None: documents[doc][offset : offset + length]
    )
    dataset.config = SimpleNamespace(
        defer_npy_index_mmap=defer_mmap, path_to_cache=str(tmp_path), tokenizer=None
    )
    dataset.unique_description_hash = "fim-cache-test"
    dataset.index_split = SimpleNamespace(name="train")
    dataset.fim_rate = 0.0
    dataset.fim_spm_rate = 0.0
    dataset.fim_split_sample = None
    dataset.no_fim_prefix = None
    dataset.prefix_tok_id, dataset.middle_tok_id, dataset.suffix_tok_id = 95, 96, 97
    dataset.pad_tok_id, dataset.eod_tok_id = 98, 99
    dataset.np_rng = np.random.RandomState(1234)

    indices = {
        "document_index": np.array([1, 0], dtype=np.int32),
        "sample_index": np.array([[0, 0], [0, 2], [0, 4]], dtype=np.int32),
        "shuffle_index": np.array([1, 0], dtype=np.uint32),
    }
    for name, index in indices.items():
        np.save(tmp_path / f"fim-cache-test-GPTFIMDataset-train-{name}.npy", index)
    if defer_mmap:
        dataset.document_index, dataset.sample_index, dataset.shuffle_index = (
            dataset._build_document_sample_shuffle_indices()
        )
        assert dataset.shuffle_index is None
    else:
        dataset.document_index, dataset.sample_index, dataset.shuffle_index = indices.values()

    for sample_id, expected in [(0, [13, 14, 99]), (1, [11, 12, 13]), (0, [13, 14, 99])]:
        tokens, document_ids, lengths = dataset._query_document_sample_shuffle_indices(sample_id)
        np.testing.assert_array_equal(tokens, expected)
        np.testing.assert_array_equal(document_ids, [1])
        assert lengths == [3]
    if defer_mmap:
        for name in indices:
            assert isinstance(getattr(dataset, name), np.memmap)
            assert not getattr(dataset, name).flags.writeable


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
