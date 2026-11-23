"""The synthetic VLM input contract: aligned media, shifted text, reproducible samples."""

import pytest
import torch
from transformers import Qwen3_5MoeVisionConfig

from megatron.bridge.models.nemotron_3.super_vl_config import Nemotron35SuperVLModelConfig
from megatron.bridge.models.qwen_vl.qwen35_vl_config import (
    Qwen35VLModelConfig,
    Qwen35VLTransformerConfig,
)
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.training.vlm import MockVisionLanguageDataset
from megatron.core.datasets.utils import Split


@pytest.mark.parametrize("family", ["qwen", "nemotron"])
def test_mock_vlm_media_and_causal_targets_are_aligned(family):
    dimensions = dict(
        num_layers=2, hidden_size=64, num_attention_heads=4, max_seq_length_per_sample=128
    )
    if family == "qwen":
        model = Qwen35VLModelConfig(
            transformer=Qwen35VLTransformerConfig(**dimensions),
            vocab_size=250000,
            vision_config=Qwen3_5MoeVisionConfig(patch_size=16, spatial_merge_size=2),
        )
        prefix, media_id = 66, model.image_token_id
    else:
        model = Nemotron35SuperVLModelConfig(
            transformer=TransformerConfig(**dimensions), vocab_size=131072, image_token_index=128001
        )
        prefix, media_id = 64, model.image_token_index
    dataset = MockVisionLanguageDataset(model, num_samples=4, seed=1234, split=Split.train)
    first = dataset[0]
    assert first["input_ids"].shape == first["labels"].shape == (128,)
    assert (first["input_ids"] == media_id).sum().item() == 64
    assert torch.equal(first["labels"][prefix - 1 : -1], first["input_ids"][prefix:])
    assert torch.all(first["loss_mask"][: prefix - 1] == 0)
    assert torch.all(first["loss_mask"][prefix - 1 :] == 1)
    assert torch.all(first["labels"][: prefix - 1] == 0)
    if family == "qwen":
        assert first["input_ids"][0] == model.vision_start_token_id
        assert first["input_ids"][65] == model.vision_end_token_id
        assert first["pixel_values"].shape == (256, 1536)
        assert first["image_grid_thw"].tolist() == [1, 16, 16]
    else:
        assert first["images"].shape == (3, 256, 256)
        assert first["imgs_sizes"].tolist() == [256, 256]
        assert first["num_frames"].item() == 1
    # Global training RNG and access order must not change a sample across TP ranks/resume.
    torch.rand(8)
    second = dataset[1]
    for key, value in first.items():
        assert torch.equal(value, dataset[0][key])
    assert not torch.equal(first["input_ids"][prefix:], second["input_ids"][prefix:])
