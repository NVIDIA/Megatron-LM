# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from copy import deepcopy

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestLLaVAModel:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        language_config = TransformerConfig(
            num_layers=3, hidden_size=128, num_attention_heads=8, use_cpu_initialization=True
        )
        vision_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        vision_projection_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=72,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )

        language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        vision_layer_spec = deepcopy(language_layer_spec)
        vision_projection_spec = deepcopy(language_layer_spec.submodules.mlp.submodules)

        self.model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_layer_spec,
            vocab_size=2048,
            max_sequence_length=1024,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, LLaVAModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 1439432

    def test_set_input_tensor(self):
        expected_shape = (1, 2, 3, 4)
        input_tensor = torch.zeros(expected_shape)
        self.model.set_input_tensor(input_tensor)
        assert self.model.vision_model.transformer.input_tensor.shape == expected_shape

    def test_forward(self):
        self.model.cuda()

        img = torch.randn((2, 3, 336, 336)).cuda()
        input_ids = torch.randint(0, 2048, (2, 1024)).cuda()
        position_ids = torch.arange(0, 1024, dtype=torch.int).cuda()
        position_ids = position_ids.expand(2, 1024)
        # With default image and patch sizes of 336 and 14, respectively, and a class token, the combined sequence length is 1024 + (336/14) ** 2 + 1 = 1601.
        attention_mask = torch.tril(torch.ones((2, 1, 1601, 1601))).cuda()
        attention_mask = attention_mask < 0.5
        labels = torch.randint(0, 2048, (2, 1601)).cuda()

        # Try with and without labels.
        loss = self.model.forward(img, input_ids, position_ids, attention_mask, labels)
        assert loss.shape == torch.Size((2, 1601))

        logits = self.model.forward(img, input_ids, position_ids, attention_mask, labels=None)
        assert logits.shape == torch.Size((2, 1601, 2048))

    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))

    def test_freeze(self):
        self.model.freeze(
            freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=False
        )

        for module in [self.model.language_model, self.model.vision_model]:
            for param in module.parameters():
                assert not param.requires_grad

        for param in self.model.vision_projection.parameters():
            assert param.requires_grad
