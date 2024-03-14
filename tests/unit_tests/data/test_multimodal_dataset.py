# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.multimodal_dataset import MockMultimodalDataset, MultimodalDatasetConfig


def test_mock_multimodal_dataset():
    config = MultimodalDatasetConfig(
        is_built_on_rank=lambda: True,
        random_seed=1234,
        sequence_length=1024,
        mock=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=True,
        tokenizer=SimpleNamespace(),
        image_h=336,
        image_w=336,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockMultimodalDataset, [100, 100, 100], config
    ).build()

    for ds in datasets:
        sample = ds[0]
        assert "image" in sample
        assert sample["image"].shape == torch.Size([3, 336, 336])
        assert "tokens" in sample
