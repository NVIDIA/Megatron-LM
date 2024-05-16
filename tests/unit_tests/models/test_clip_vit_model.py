# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
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
        self.model = CLIPViTModel(transformer_config, transformer_layer_spec)

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
