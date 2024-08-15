# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from copy import deepcopy

import pytest
import torch

from megatron.core import InferenceParams
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestLLaVAModel:
    @pytest.mark.internal  # The model is under active development and its methods may change.
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

    @pytest.mark.internal
    def test_preprocess_data(self):
        self.model.cuda()

        image_embedding_value = torch.tensor(123.0)
        image_embeddings = image_embedding_value * torch.ones((577, 3, 128)).cuda()

        image_token_index = -200
        input_ids = torch.arange(0, 1024, dtype=torch.int).expand(4, 1024).cuda()
        input_ids[0, 0] = image_token_index  # image before text
        input_ids[1, 100] = image_token_index  # image in between
        input_ids[2, -1] = image_token_index  # image at the end
        # input_ids[3] - no image

        language_embedding_value = torch.tensor(999.0)
        language_embeddings = language_embedding_value * torch.ones((4, 1024, 128)).cuda()

        # Labels are input_ids shifted to left by one.
        labels = torch.arange(1, 1025, dtype=torch.int).expand(4, 1024).cuda()
        labels[1, 99] = image_token_index
        labels[2, -2] = image_token_index

        loss_mask = torch.ones((4, 1024), dtype=torch.int).cuda()
        # Mask some text inputs (the text mask should carry over)
        loss_mask[:2, :10] = 0
        loss_mask[:2, 110:120] = 0

        use_inference_kv_cache = False

        embeddings, labels, loss_mask = self.model._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            image_token_index,
        )

        assert embeddings.shape == torch.Size((1600, 4, 128))
        assert labels.shape == torch.Size((4, 1600))
        assert loss_mask.shape == labels.shape

        # First sample where image is before text (index 0).
        expected_embeddings = torch.empty(1600).cuda()
        expected_embeddings[:577] = image_embedding_value
        expected_embeddings[577:] = language_embedding_value

        expected_labels = torch.empty(1600, dtype=torch.int).cuda()
        expected_labels[:576] = -100
        expected_labels[576:] = torch.arange(1, 1025, dtype=torch.int)

        expected_loss_mask = torch.empty(1600, dtype=torch.int).cuda()
        expected_loss_mask[:577] = 0
        expected_loss_mask[577:586] = 0
        expected_loss_mask[586:686] = 1
        expected_loss_mask[686:696] = 0
        expected_loss_mask[696:] = 1

        assert torch.allclose(embeddings[:, 0], expected_embeddings.unsqueeze(1))
        assert torch.allclose(labels[0], expected_labels)
        assert torch.allclose(loss_mask[0], expected_loss_mask)

        # Second sample where image is in between (index 100).
        expected_embeddings = torch.empty(1600).cuda()
        expected_embeddings[:100] = language_embedding_value
        expected_embeddings[100:677] = image_embedding_value
        expected_embeddings[677:] = language_embedding_value

        expected_labels = torch.empty(1600, dtype=torch.int).cuda()
        expected_labels[:99] = torch.arange(1, 100)
        expected_labels[99:676] = -100
        expected_labels[676:] = torch.arange(101, 1025)

        expected_loss_mask = torch.empty(1600, dtype=torch.int).cuda()
        expected_loss_mask[:10] = 0
        expected_loss_mask[10:99] = 1
        expected_loss_mask[99] = (
            0  # Last text position before the image is not required to predict the first image embedding.
        )
        expected_loss_mask[100:677] = 0
        expected_loss_mask[677:686] = 1
        expected_loss_mask[686:696] = 0
        expected_loss_mask[696:] = 1

        assert torch.allclose(embeddings[:, 1], expected_embeddings.unsqueeze(1))
        assert torch.allclose(labels[1], expected_labels)
        assert torch.allclose(loss_mask[1], expected_loss_mask)

        # Third sample where image is at the end.
        expected_embeddings = torch.empty(1600).cuda()
        expected_embeddings[:1023] = language_embedding_value
        expected_embeddings[1023:] = image_embedding_value

        expected_labels = torch.empty(1600, dtype=torch.int).cuda()
        expected_labels[:1022] = torch.arange(1, 1023)
        expected_labels[1022:1599] = -100
        expected_labels[1599] = 1024

        expected_loss_mask = torch.empty(1600, dtype=torch.int).cuda()
        expected_loss_mask[:1022] = 1
        expected_loss_mask[1022] = (
            0  # Last text position before the image is not required to predict the first image embedding.
        )
        expected_loss_mask[1023:] = 0

        assert torch.allclose(embeddings[:, 2], expected_embeddings.unsqueeze(1))
        assert torch.allclose(labels[2], expected_labels)
        assert torch.allclose(loss_mask[2], expected_loss_mask)

        # Fourth sample where there is no image.
        expected_embeddings = torch.empty(1600).cuda()
        expected_embeddings[:1024] = language_embedding_value
        expected_embeddings[1024:] = 0  # padding

        expected_labels = torch.empty(1600, dtype=torch.int).cuda()
        expected_labels[:1024] = torch.arange(1, 1025)
        expected_labels[1024:] = -100

        expected_loss_mask = torch.empty(1600, dtype=torch.int).cuda()
        expected_loss_mask[:1024] = 1
        expected_loss_mask[1024:] = 0

        assert torch.allclose(embeddings[:, 3], expected_embeddings.unsqueeze(1))
        assert torch.allclose(labels[3], expected_labels)
        assert torch.allclose(loss_mask[3], expected_loss_mask)

    @pytest.mark.internal
    def test_forward(self):
        self.model.cuda()

        img = torch.randn((3, 3, 336, 336)).cuda()

        image_token_index = -200
        input_ids = torch.randint(0, 2048, (4, 1024)).cuda()
        input_ids[0, 0] = image_token_index  # image before text
        input_ids[1, 100] = image_token_index  # image in between
        input_ids[2, -1] = image_token_index  # image at the end
        # input_ids[3] - no image

        position_ids = torch.arange(0, 1024, dtype=torch.int).expand(4, 1024).cuda()

        loss_mask = torch.ones((4, 1024)).cuda()

        attention_mask = None  # Causal.

        labels = torch.randint(0, 2048, (4, 1024)).cuda()
        labels[1, 99] = image_token_index
        labels[2, -2] = image_token_index

        # Try with labels.
        loss, new_loss_mask = self.model.forward(
            img, input_ids, position_ids, attention_mask, labels, loss_mask
        )
        # The final sequence length 1600 comes from 577 image tokens and 1023 text tokens.
        assert loss.shape == new_loss_mask.shape == torch.Size((4, 1600))

        # Try without labels and without inference params.
        logits = self.model.forward(
            img, input_ids, position_ids, attention_mask, labels=None, loss_mask=None
        )
        assert logits.shape == torch.Size((4, 1600, 2048))

        # Try without labels and with inference params.
        inference_params = InferenceParams(4, 1600)
        logits = self.model.forward(
            img,
            input_ids,
            position_ids,
            attention_mask,
            labels=None,
            loss_mask=None,
            inference_params=inference_params,
        )
        assert logits.shape == torch.Size((4, 1600, 2048))

        # Check KV cache got populated correctly.
        kv_dict = inference_params.key_value_memory_dict

        assert kv_dict["image_tokens_count"] == 577
        for layer_no in range(1, 4):  # 3 layers in the model.
            layer_kv = kv_dict[layer_no]
            # Expected shape is [sequence_len, batch_size, num_heads, hidden_size_per_head]
            assert layer_kv[0].shape == layer_kv[1].shape == torch.Size((1600, 4, 8, 16))

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
