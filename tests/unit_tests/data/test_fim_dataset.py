# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.training.datasets.fim_dataset import GPTFIMDatasetConfig, GPTFIMDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers import MegatronTokenizer
from tests.unit_tests.test_utilities import Utils
from megatron.core.datasets.utils import get_blend_from_list


def test_fim_gpt_dataset():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null"}, vocab_size=131072
    )
    blend = get_blend_from_list(["/opt/data/datasets/train/test_text_document"])
    extra_tokens = {"prefix": "777", "middle": "888", "suffix": "999", "pad": "666", "eod": "000"}
    seq_length = 8
    rate = 0.2
    spm_rate = 0.2
    fragment_rate = 0.5
    config = GPTFIMDatasetConfig(
        blend=blend,
        random_seed=1234,
        sequence_length=seq_length,
        split="990,9,1",
        tokenizer=tokenizer,
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        extra_tokens=extra_tokens,
        rate=rate,
        spm_rate=spm_rate,
        fragment_rate=fragment_rate,
        no_prefix="111214",
    )

    datasets = BlendedMegatronDatasetBuilder(
        GPTFIMDataset, [10, 10, 10], lambda: True, config
    ).build()

    dataset = datasets[0]
    assert dataset.fim_rate == rate
    assert dataset.fim_spm_rate == spm_rate
    assert dataset.fragment_fim_rate == 0.5
    assert dataset[0]["tokens"].tolist() == [343, 54365900, 77, 131072, 111214, 343, 54365900, 77]


if __name__ == "__main__":
    test_fim_gpt_dataset()
