# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m pytest tests/unit_tests/models/test_mimo_model.py 
'''

import torch
import torch.nn as nn

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestMimoModel:
    """Test the MimoModel class."""

    def setup_method(self, method):
        '''setup env and model'''
        try:
            Utils.initialize_model_parallel(1, 1)
        except Exception as e:
            print(f"Warning: Could not initialize model parallel: {e}")

        # Set dimensions
        self.hidden_size = 64
        self.batch_size = 2
        self.seq_len = 2048
        self.img_h = 224
        self.img_w = 224
        self.patch_dim = 16
        self.vocab_size = 50304

        # Create transformer config for vision encoder
        self.vision_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        # Create transformer config for language model
        self.lm_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        # Create layer specs
        self.vision_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        self.language_layer_spec = get_gpt_layer_with_transformer_engine_spec()

        # Create language model spec
        self.language_model_spec = ModuleSpec(
            module=GPTModel,
            params={
                "config": self.lm_config,
                "transformer_layer_spec": self.language_layer_spec,
                "vocab_size": self.vocab_size,
                "max_sequence_length": self.seq_len,
                "pre_process": True,
                "post_process": True,
            },
        )

        # Create vision encoder spec
        self.vision_encoder_spec = ModuleSpec(
            module=CLIPViTModel,
            params={
                "transformer_config": self.vision_config,
                "transformer_layer_spec": self.vision_layer_spec,
                "img_h": self.img_h,
                "img_w": self.img_w,
                "patch_dim": self.patch_dim,
            },
        )

        # Create vision projection spec
        self.vision_projection_spec = ModuleSpec(
            module=nn.Linear,
            params={
                "in_features": self.vision_config.hidden_size,
                "out_features": self.vision_config.hidden_size,
            },
        )

        # Create vision modality spec
        self.vision_submodule_spec = ModuleSpec(
            module=VisionModalitySubmodules,
            submodules={
                "encoders": [self.vision_encoder_spec],
                "input_projections": [self.vision_projection_spec],
            },
        )

        # Define special token IDs
        self.special_token_ids = {"images": 50257}

        # Create MIMO model config
        self.mimo_config = MimoModelConfig(
            language_model_spec=self.language_model_spec,
            modality_submodules_spec={"images": self.vision_submodule_spec},
            special_token_ids=self.special_token_ids,
        )

        # Create MIMO model
        self.mimo_model = MimoModel(self.mimo_config)

        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mimo_model = self.mimo_model.to(self.device)

    def teardown_method(self, method):
        '''teardown env'''
        try:
            Utils.destroy_model_parallel()
        except Exception as e:
            print(f"Warning: Could not destroy model parallel: {e}")

    def test_constructor(self):
        """Test constructor initialization."""
        # Test that modality submodules were initialized correctly
        assert "images" in self.mimo_model.modality_submodules
        assert isinstance(self.mimo_model.modality_submodules["images"], VisionModalitySubmodules)

        # Test that language model was initialized
        assert hasattr(self.mimo_model, "language_model")
        assert isinstance(self.mimo_model.language_model, GPTModel)

        # Test that special token IDs were set correctly
        assert self.mimo_model.special_token_ids == self.special_token_ids

    def test_get_text_embeddings(self):
        """Test getting text embeddings."""
        # Create random input and position IDs (within vocab size range)
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device
        )
        position_ids = (
            torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        )

        # Get text embeddings
        text_embeddings = self.mimo_model.get_text_embeddings(
            input_ids, position_ids, self.special_token_ids
        )
        # Verify shape
        # [b*s, h]
        assert text_embeddings.shape == (self.batch_size * self.seq_len, self.hidden_size)

    def test_forward_text_only(self):
        """Test forward pass with only text input."""
        # Create data batch with only text (ensure input_ids are within vocab_size)
        data_batch = {
            "input_ids": torch.randint(
                0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device
            ),
            "position_ids": torch.arange(self.seq_len, device=self.device)
            .unsqueeze(0)
            .expand(self.batch_size, -1),
        }

        # Run forward pass
        outputs, _ = self.mimo_model(data_batch)
        assert outputs is not None

        # Verify output shape
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_forward_with_image_modality(self):
        """Test forward pass with text and image input."""
        # Calculate expected number of image tokens based on image size and patch dimension
        expected_img_seq_len = (self.img_h // self.patch_dim) * (
            self.img_w // self.patch_dim
        ) + 1  # +1 for CLS token

        # Create a fixed distribution of images: 3 in first sample, 2 in second sample
        num_images = 5
        images_per_sample = [3, 2]  # Must sum to num_images
        assert sum(images_per_sample) == num_images
        assert len(images_per_sample) == self.batch_size

        # Create data batch with text and images (ensure input_ids are within vocab_size)
        data_batch = {
            "input_ids": torch.randint(
                0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device
            ),
            "position_ids": torch.arange(self.seq_len, device=self.device)
            .unsqueeze(0)
            .expand(self.batch_size, -1),
            "images": torch.rand(
                num_images, 3, self.img_h, self.img_w, device=self.device
            ),  # [num_images, 3, h, w] format
        }

        # Include image special tokens in input IDs (ensure it's a valid ID)
        image_token_id = self.special_token_ids["images"]
        # Ensure we have enough space for image tokens in each sample
        start_pos = 5  # Start position for image tokens

        # Make sure there's enough space in the sequence for all image tokens in each sample
        for b in range(self.batch_size):
            tokens_needed = images_per_sample[b] * expected_img_seq_len
            assert (
                start_pos + tokens_needed <= self.seq_len
            ), f"Sequence length too short for image tokens in sample {b}"

        # Add image tokens to each batch sample according to its number of images
        for b in range(self.batch_size):
            tokens_in_this_batch = images_per_sample[b] * expected_img_seq_len
            if tokens_in_this_batch > 0:
                data_batch["input_ids"][
                    b, start_pos : start_pos + tokens_in_this_batch
                ] = image_token_id

        # Run forward pass
        outputs, _ = self.mimo_model(data_batch)
        assert outputs is not None

        # Verify output shape
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_state_dict(self):
        """Test state dict methods."""
        # Get state dict
        state_dict = self.mimo_model.state_dict()
        assert len(state_dict) > 0

        # Make sure we have keys for language model and modality submodules
        has_lm_keys = False
        has_modality_keys = False

        for key in state_dict.keys():
            if key.startswith("language_model."):
                has_lm_keys = True
            if key.startswith("modality_submodules."):
                has_modality_keys = True

        assert has_lm_keys
        assert has_modality_keys

        # Test checkpoint state dict
        checkpoint_dict = self.mimo_model.state_dict_for_save_checkpoint()
        assert len(checkpoint_dict) > 0
