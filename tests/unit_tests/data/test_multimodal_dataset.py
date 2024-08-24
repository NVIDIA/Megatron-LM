# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

##
# Compile megatron.core.datasets.helpers dependencies before BlendedDataset import
##

from types import SimpleNamespace

import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.multimodal_dataset import MockMultimodalDataset, MultimodalDatasetConfig
from megatron.core.datasets.utils import compile_helpers
from megatron.training.tokenizer.tokenizer import _NullTokenizer
from tests.unit_tests.test_utilities import Utils

_MOCK_VOCAB_SIZE = 8192


def test_mock_multimodal_dataset():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = MultimodalDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=True,
        image_h=336,
        image_w=336,
        split="990,9,1",
        tokenizer=_NullTokenizer(vocab_size=_MOCK_VOCAB_SIZE),
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockMultimodalDataset, [100, 100, 100], lambda: True, config
    ).build()

    for ds in datasets:
        sample = ds[0]
        assert "image" in sample
        assert sample["image"].shape == torch.Size([3, 336, 336])
        assert "tokens" in sample


if __name__ == "__main__":
    test_mock_multimodal_dataset()
