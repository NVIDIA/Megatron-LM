# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.multimodal.utils import patchify_image, unpatchify_image
from megatron.core.models.vision.clip_vit_model import (
    CLIPViTModel,
    get_num_image_embeddings,
    get_num_video_embeddings,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestCLIPViTModel:
    """Test CLIP ViT model."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        self.model = CLIPViTModel(
            transformer_config, transformer_layer_spec, img_h=336, img_w=336, patch_dim=14
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, CLIPViTModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 174720

    def test_set_input_tensor(self):
        # [s, b, h] expected to the transformer.
        expected_shape = (577, 2, 64)
        input_tensor = torch.zeros(expected_shape)

        self.model.set_input_tensor(input_tensor)

        assert self.model.decoder.input_tensor.shape == torch.Size(expected_shape)

    def test_forward(self):
        self.model.cuda()

        img = torch.zeros((2, 3, 336, 336)).cuda()

        out = self.model.forward(img)
        assert out.shape == torch.Size([2, 577, 64])

    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))


@pytest.mark.internal
@pytest.mark.parametrize(
    "vision_model,pixel_shuffle,tile_tags,expected",
    [
        ("clip", False, False, 1024),
        ("internvit300M", False, False, 1024),
        ("clip", True, False, 256),
        ("internvit300M", True, True, 262),
    ],
)
def test_get_num_image_embeddings(vision_model, pixel_shuffle, tile_tags, expected):
    assert (
        get_num_image_embeddings(
            448, 448, 14, vision_model, True, 1, pixel_shuffle, tile_tags, 0, "nemotron5"
        )
        == expected
    )


@pytest.mark.parametrize("batched", [False, True])
def test_patchify_unpatchify_roundtrip(batched):
    image = torch.arange(3 * 4 * 6).reshape(3, 4, 6)
    if batched:
        image = torch.stack((image, image + image.numel()))

    patches = patchify_image(image, patch_dim=2)

    assert torch.equal(unpatchify_image(patches, img_H=4, img_W=6, patch_dim=2), image)


def test_get_num_video_embeddings_applies_temporal_and_attention_pooling():
    assert (
        get_num_video_embeddings(
            num_frames=4,
            video_temporal_patch_size=2,
            img_h=8,
            img_w=8,
            patch_dim=2,
            vision_model_type="siglip",
            disable_vision_class_token=True,
            class_token_len=0,
            pixel_shuffle=False,
            attn_pooling=True,
            attn_pooling_img_h=1,
            attn_pooling_img_w=1,
            attn_pooling_video_h=2,
            attn_pooling_video_w=2,
        )
        == 8
    )
