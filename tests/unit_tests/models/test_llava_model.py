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
            language_max_sequence_length=4096,
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
        assert num_weights == 1832520

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
        # 3 images with 1 tile and 2 image with 2 tiles = 7 tiles.
        image_embeddings = image_embedding_value * torch.ones((577, 7, 128)).cuda()

        image_token_index = -200
        input_ids = torch.arange(0, 1024, dtype=torch.int).expand(5, 1024).cuda()
        input_ids[0, 0] = image_token_index  # image before text
        input_ids[1, 100] = image_token_index  # image in between
        input_ids[2, -1] = image_token_index  # image at the end
        # input_ids[3] - no image
        input_ids[4, 50] = image_token_index  # two images in between
        input_ids[4, 150] = image_token_index

        language_embedding_value = torch.tensor(999.0)
        language_embeddings = language_embedding_value * torch.ones((5, 1024, 128)).cuda()

        # Labels are input_ids shifted to left by one.
        labels = torch.arange(1, 1025, dtype=torch.int).expand(5, 1024).cuda()
        labels[1, 99] = image_token_index
        labels[2, -2] = image_token_index
        labels[4, 49] = image_token_index
        labels[4, 149] = image_token_index

        loss_mask = torch.ones((5, 1024), dtype=torch.float).cuda()
        # Mask some text inputs (the text mask should carry over)
        loss_mask[:2, :10] = 0.0
        loss_mask[:2, 110:120] = 0.0

        # Number of tiles for each image in the batch.
        num_image_tiles = torch.tensor([1, 2, 1, 2, 1], dtype=torch.int).cuda()

        use_inference_kv_cache = False

        embeddings, labels, loss_mask = self.model._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            image_token_index,
            num_image_tiles,
        )

        img_seq_len = 577
        # The fifth sample has 2 images with 3 tiles and 1024 text tokens.
        max_seq_len = 3 * img_seq_len - 2 + 1024

        assert embeddings.shape == torch.Size((max_seq_len, 5, 128))
        assert labels.shape == torch.Size((5, max_seq_len))
        assert loss_mask.shape == labels.shape

        # First sample where image is before text (index 0).
        expected_embeddings = torch.empty(max_seq_len).cuda()
        expected_embeddings[:577] = image_embedding_value
        expected_embeddings[577:1600] = language_embedding_value
        expected_embeddings[1600:] = 0  # padding

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:576] = -100  # image
        expected_labels[576:1600] = torch.arange(1, 1025, dtype=torch.int)
        expected_labels[1600:] = -100  # padding

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:577] = 0
        expected_loss_mask[577:586] = 0
        expected_loss_mask[586:686] = 1
        expected_loss_mask[686:696] = 0
        expected_loss_mask[696:1600] = 1
        expected_loss_mask[1600:] = 0

        assert torch.allclose(embeddings[:, 0], expected_embeddings.unsqueeze(1))
        assert torch.allclose(labels[0], expected_labels)
        assert torch.allclose(loss_mask[0], expected_loss_mask)

        # Second sample where image is in between (index 100). The image has 2 tiles.
        expected_embeddings = torch.empty(max_seq_len).cuda()
        expected_embeddings[:100] = language_embedding_value
        expected_embeddings[100:1254] = image_embedding_value
        expected_embeddings[1254:2177] = language_embedding_value
        expected_embeddings[2177:] = 0  # padding

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:99] = torch.arange(1, 100)
        expected_labels[99:1253] = -100  # image
        expected_labels[1253:2177] = torch.arange(101, 1025)
        expected_labels[2177:] = -100  # padding

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:10] = 0
        expected_loss_mask[10:99] = 1
        # Last text position before the image is not required to predict the first image embedding.
        expected_loss_mask[99] = 0
        expected_loss_mask[100:1254] = 0
        expected_loss_mask[1254:1263] = 1
        expected_loss_mask[1263:1273] = 0
        expected_loss_mask[1273:2177] = 1
        expected_loss_mask[2177:] = 0  # padding

        assert torch.allclose(embeddings[:, 1], expected_embeddings.unsqueeze(1))
        assert torch.allclose(labels[1], expected_labels)
        assert torch.allclose(loss_mask[1], expected_loss_mask)

        # Third sample where image is at the end.
        expected_embeddings = torch.empty(max_seq_len).cuda()
        expected_embeddings[:1023] = language_embedding_value
        expected_embeddings[1023:1600] = image_embedding_value
        expected_embeddings[1600:] = 0  # padding

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:1022] = torch.arange(1, 1023)
        expected_labels[1022:1599] = -100
        expected_labels[1599] = 1024
        expected_labels[1600:] = -100  # padding

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:1022] = 1
        # Last text position before the image is not required to predict the first image embedding.
        expected_loss_mask[1022] = 0
        expected_loss_mask[1023:1600] = 0
        expected_loss_mask[1600:] = 0  # padding

        assert torch.allclose(embeddings[:, 2], expected_embeddings.unsqueeze(1))
        assert torch.allclose(labels[2], expected_labels)
        assert torch.allclose(loss_mask[2], expected_loss_mask)

        # Fourth sample where there is no image.
        expected_embeddings = torch.empty(max_seq_len).cuda()
        expected_embeddings[:1024] = language_embedding_value
        expected_embeddings[1024:] = 0  # padding

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:1024] = torch.arange(1, 1025)
        expected_labels[1024:] = -100  # padding

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:1024] = 1
        expected_loss_mask[1024:] = 0  # padding

        assert torch.allclose(embeddings[:, 3], expected_embeddings.unsqueeze(1))
        assert torch.allclose(labels[3], expected_labels)
        assert torch.allclose(loss_mask[3], expected_loss_mask)

        # Fifth sample has two images in between. The first image has two tiles.
        expected_embeddings = torch.empty(max_seq_len).cuda()
        expected_embeddings[:50] = language_embedding_value
        expected_embeddings[50:1204] = image_embedding_value  # two tiles
        expected_embeddings[1204:1303] = language_embedding_value
        expected_embeddings[1303:1880] = image_embedding_value
        expected_embeddings[1880:] = language_embedding_value

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:49] = torch.arange(1, 50)
        expected_labels[49:1203] = -100  # image
        expected_labels[1203:1302] = torch.arange(51, 150)
        expected_labels[1302:1879] = -100  # image
        expected_labels[1879:] = torch.arange(151, 1025)

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:49] = 1
        expected_loss_mask[49:1204] = 0
        expected_loss_mask[1204:1302] = 1
        expected_loss_mask[1302:1880] = 0
        expected_loss_mask[1880:] = 1

        assert torch.allclose(embeddings[:, 4], expected_embeddings.unsqueeze(1))
        assert torch.allclose(labels[4], expected_labels)
        assert torch.allclose(loss_mask[4], expected_loss_mask)

    @pytest.mark.internal
    def test_forward(self):
        self.model.cuda()

        # 3 images with 1 tile and 2 images with 2 tiles.
        img = torch.randn((7, 3, 336, 336)).cuda()

        image_token_index = -200
        input_ids = torch.randint(0, 2048, (5, 1024)).cuda()
        input_ids[0, 0] = image_token_index  # image before text
        input_ids[1, 100] = image_token_index  # image in between
        input_ids[2, -1] = image_token_index  # image at the end
        # input_ids[3] - no image
        input_ids[4, 50] = image_token_index
        input_ids[4, 150] = image_token_index

        position_ids = torch.arange(0, 1024, dtype=torch.int).expand(5, 1024).cuda()

        loss_mask = torch.ones((5, 1024)).cuda()

        attention_mask = None  # Causal.

        labels = torch.randint(0, 2048, (5, 1024)).cuda()
        labels[1, 99] = image_token_index
        labels[2, -2] = image_token_index

        num_image_tiles = torch.tensor([1, 2, 1, 2, 1], dtype=torch.int).cuda()

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

        # Try text-only input.
        loss, new_loss_mask = self.model.forward(
            torch.tensor([], dtype=torch.float).cuda(),
            torch.randint(0, 2048, (5, 1024)).cuda(),
            position_ids,
            attention_mask,
            torch.randint(0, 2048, (5, 1024)).cuda(),
            loss_mask,
            num_image_tiles=torch.tensor([], dtype=torch.int).cuda(),
        )

        assert loss.shape == new_loss_mask.shape == torch.Size((5, 1024))

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
