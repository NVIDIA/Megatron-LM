# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.utils import Split, compile_helpers, get_blend_from_list
from megatron.core.tokenizers import MegatronTokenizer
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
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


@pytest.mark.parametrize(
    "text",
    [
        "<split>alpha beta",
        "alpha<split><split>beta",
        "<split><split><split>",
        "alpha<split>beta",
        "alpha beta<split>",
    ],
)
@pytest.mark.parametrize("fim_rate", [0.0, 1.0])
def test_fim_preserves_split_tokens(tmp_path, text, fim_rate):
    compile_helpers()
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path="/opt/data/tokenizers/huggingface",
        metadata_path={"library": "huggingface"},
        additional_special_tokens=["<prefix>", "<middle>", "<suffix>", "<pad>", "<eod>", "<split>"],
        include_special_tokens=False,
    )
    token_ids = tokenizer.tokenize(text)
    split_token = "<split>"
    split_id = tokenizer._tokenizer.tokens_to_ids(split_token)
    assert split_id in token_ids
    prefix = str(tmp_path / "sample")
    builder = IndexedDatasetBuilder(prefix + ".bin")
    builder.add_item(torch.tensor(token_ids, dtype=torch.int32))
    builder.end_document()
    builder.finalize(prefix + ".idx")
    config = GPTFIMDatasetConfig(
        random_seed=1234,
        sequence_length=len(token_ids) - 1,
        blend=([prefix], None),
        split="1,0,0",
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        fim_extra_tokens={
            "prefix": "<prefix>",
            "middle": "<middle>",
            "suffix": "<suffix>",
            "pad": "<pad>",
            "eod": "<eod>",
        },
        fim_rate=fim_rate,
        fim_spm_rate=0.0,
        fim_fragment_rate=0.0,
        fim_split_sample=split_token,
        path_to_cache=str(tmp_path / "cache"),
    )
    dataset = GPTFIMDataset(
        IndexedDataset(prefix), prefix, np.array([0], dtype=np.int32), 1, Split.train, config
    )
    sample = dataset[0]
    assert sample["tokens"].tolist() == token_ids[:-1]
    assert sample["labels"].tolist() == token_ids[1:]


if __name__ == "__main__":
    test_fim_gpt_dataset()
