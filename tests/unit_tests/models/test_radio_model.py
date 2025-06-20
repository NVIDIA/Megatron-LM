# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestRADIOViTModel:
    """Test RADIO ViT model."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        self.model = RADIOViTModel(
            transformer_config,
            transformer_layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, RADIOViTModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 1501824

    def test_set_input_tensor(self):
        # [s, b, h] expected to the transformer.
        expected_shape = (256, 2, 64)
        input_tensor = torch.zeros(expected_shape)

        self.model.set_input_tensor(input_tensor)

        assert self.model.decoder.input_tensor.shape == torch.Size(expected_shape)

    def test_forward(self):
        self.model.cuda()

        img = torch.zeros((2, 3, 224, 224)).cuda()

        out = self.model.forward(img)
        assert out.shape == torch.Size([2, 256, 64])

    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))
