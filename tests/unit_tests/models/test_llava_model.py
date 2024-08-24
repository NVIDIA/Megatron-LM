# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from copy import deepcopy
import os

from megatron.core.device_utils import get_current_device
import pytest
import torch

from megatron.core import InferenceParams
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

@pytest.mark.skipif(int(os.environ.get('ACCEL_MEMORY_GB', 40)) < 32 , reason="insufficient accelerator memory")
class TestLLaVAModel:
    @pytest.mark.internal  # The model is under active development and its methods may change.
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_device_manual_seed(123)

        language_config = TransformerConfig(
            num_layers=3, hidden_size=128, num_attention_heads=8, use_cpu_initialization=False
        )
        vision_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=False
        )
        vision_projection_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=72,
            num_attention_heads=1,
            use_cpu_initialization=False,
        )

        language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        vision_layer_spec = deepcopy(language_layer_spec)
        vision_projection_spec = deepcopy(language_layer_spec.submodules.mlp.submodules)

        self.model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_layer_spec,
            language_vocab_size=2048,
            language_max_sequence_length=1024,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
        )

    @pytest.mark.internal
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.model, LLaVAModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 1439304

    @pytest.mark.internal
    def test_set_input_tensor(self):
        expected_shape = (1, 2, 3, 4)
        input_tensor = torch.zeros(expected_shape)
        self.model.set_input_tensor(input_tensor)
        assert self.model.vision_model.decoder.input_tensor.shape == expected_shape

    def test_forward(self):
        self.model.to(device=get_current_device())

        img = torch.randn((2, 3, 336, 336)).to(device=get_current_device())
        input_ids = torch.randint(0, 2048, (2, 1024)).to(device=get_current_device())
        position_ids = torch.arange(0, 1024, dtype=torch.int).to(device=get_current_device())
        position_ids = position_ids.expand(2, 1024)
        # With default image and patch sizes of 336 and 14, respectively, and a class token, the combined sequence length is 1024 + (336/14) ** 2 + 1 = 1601.
        attention_mask = torch.tril(torch.ones((2, 1, 1601, 1601))).to(device=get_current_device())
        attention_mask = attention_mask < 0.5
        labels = torch.randint(0, 2048, (2, 1601)).to(device=get_current_device())

        # Try with labels.
        loss, new_loss_mask = self.model.forward(
            img,
            input_ids,
            position_ids,
            attention_mask,
            labels,
            loss_mask,
            num_image_tiles=num_image_tiles,
        )

        # The maximum sequence length is given by the sample with 2 images in 3 tiles, minus two image token indices, plus other text tokens.
        img_seq_len = 577
        max_seq_len = img_seq_len * 3 - 2 + 1024
        assert loss.shape == new_loss_mask.shape == torch.Size((5, max_seq_len))

        # Try without labels and without inference params.
        logits = self.model.forward(
            img,
            input_ids,
            position_ids,
            attention_mask,
            labels=None,
            loss_mask=None,
            num_image_tiles=num_image_tiles,
        )
        assert logits.shape == torch.Size((5, max_seq_len, 2048))

        # Try without labels and with inference params.
        inference_params = InferenceParams(5, max_seq_len)
        logits = self.model.forward(
            img,
            input_ids,
            position_ids,
            attention_mask,
            labels=None,
            loss_mask=None,
            num_image_tiles=num_image_tiles,
            inference_params=inference_params,
        )
        assert logits.shape == torch.Size((5, max_seq_len, 2048))

        # Check KV cache got populated correctly.
        kv_dict = inference_params.key_value_memory_dict

        assert kv_dict["image_tokens_count"] == 577 * 7
        for layer_no in range(1, 4):  # 3 layers in the model.
            layer_kv = kv_dict[layer_no]
            # Expected shape is [sequence_len, batch_size, num_heads, hidden_size_per_head]
            assert layer_kv[0].shape == layer_kv[1].shape == torch.Size((max_seq_len, 5, 8, 16))

    @pytest.mark.internal
    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))

    @pytest.mark.internal
    def test_freeze(self):
        self.model.freeze(
            freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=False
        )

        for module in [self.model.language_model, self.model.vision_model]:
            for param in module.parameters():
                assert not param.requires_grad

        for param in self.model.vision_projection.parameters():
            assert param.requires_grad
