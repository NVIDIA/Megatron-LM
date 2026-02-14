# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m pytest tests/unit_tests/models/test_mimo_model.py 
'''

import math

import pytest
import torch
import torch.nn as nn
from transformers import WhisperConfig, WhisperModel

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("modelopt", minversion="0.25")
# modelopt version < 0.27 breaks HF AutoModel.from_pretrained API
# so we need to skip the tests unitl versions are bumped in pyt LTS CI container


class AudioEncoderWrapper(torch.nn.Module):
    """Generic wrapper for audio encoder models that extracts last_hidden_state."""

    def __init__(self, config):
        super().__init__()
        # Use a local Whisper model (tiny config) to avoid checkpoint download
        self.encoder = WhisperModel(WhisperConfig()).encoder

    def forward(self, input_features):
        # Process through encoder and extract last_hidden_state
        with torch.no_grad():
            return self.encoder(input_features).last_hidden_state


def get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim):
    """Get the submodule spec for the vision modality."""
    vision_layer_spec = get_gpt_layer_with_transformer_engine_spec()

    vision_config = TransformerConfig(
        num_layers=1, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True
    )
    vision_encoder_spec = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": vision_layer_spec,
            "img_h": img_h,
            "img_w": img_w,
            "patch_dim": patch_dim,
        },
    )

    # Create vision projection spec
    vision_projection_spec = ModuleSpec(
        module=nn.Linear,
        params={
            "in_features": vision_config.hidden_size,
            "out_features": vision_config.hidden_size,
        },
    )

    # Create vision modality spec
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {"clip_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )

    return vision_submodule_spec


def get_audio_submodules_spec(hidden_size):
    """Get the submodule spec for the audio modality."""

    class AudioEncoderWrapper(torch.nn.Module):
        """Generic wrapper for audio encoder models that extracts last_hidden_state."""

        def __init__(self, model_name="openai/whisper-tiny"):
            super().__init__()
            # Local tiny Whisper model with random weights
            self.encoder = WhisperModel(WhisperConfig()).encoder

        def forward(self, input_features):
            # Process through encoder and extract last_hidden_state
            with torch.no_grad():
                return self.encoder(input_features).last_hidden_state

    # Audio modality configuration
    audio_encoder_spec = ModuleSpec(
        module=AudioEncoderWrapper, params={"model_name": "openai/whisper-tiny"}
    )

    audio_projection_spec = ModuleSpec(
        module=nn.Linear,
        params={"in_features": 384, "out_features": hidden_size},  # Whisper tiny hidden size
    )

    audio_submodule_spec = ModuleSpec(
        module=AudioModalitySubmodules,
        submodules={
            "encoders": {"whisper_encoder": audio_encoder_spec},
            "input_projections": [audio_projection_spec],
        },
    )

    return audio_submodule_spec


def get_language_model_spec(hidden_size, vocab_size, seq_len):
    """Get the language model spec."""
    lm_config = TransformerConfig(
        num_layers=2, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True
    )
    language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": language_layer_spec,
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": True,
            "post_process": True,
        },
    )
    return language_model_spec


def get_avlm_mimo_model(
    hidden_size, vocab_size, seq_len, img_h, img_w, patch_dim, special_token_ids
):
    language_model_spec = get_language_model_spec(hidden_size, vocab_size, seq_len)
    vision_submodule_spec = get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim)
    audio_submodule_spec = get_audio_submodules_spec(hidden_size)

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec, "audio": audio_submodule_spec},
        special_token_ids=special_token_ids,
    )

    # Create MIMO model
    mimo_model = MimoModel(mimo_config)
    return mimo_model


def get_vlm_mimo_model(
    hidden_size, vocab_size, seq_len, img_h, img_w, patch_dim, special_token_ids
):
    language_model_spec = get_language_model_spec(hidden_size, vocab_size, seq_len)
    vision_submodule_spec = get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim)

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids=special_token_ids,
    )

    # Create MIMO model
    mimo_model = MimoModel(mimo_config)
    return mimo_model


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
        self.vocab_size = 48000

        # Define special token IDs, not in LLM vocab
        self.special_token_ids = {"images": 50257, "audio": 50258}

    def teardown_method(self, method):
        '''teardown env'''
        try:
            Utils.destroy_model_parallel()
        except Exception as e:
            print(f"Warning: Could not destroy model parallel: {e}")

    def test_constructor(self):
        """Test constructor initialization."""

        mimo_model = get_avlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        )

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mimo_model = mimo_model.to(device)

        # Test that modality submodules were initialized correctly
        assert "images" in mimo_model.modality_submodules
        assert "audio" in mimo_model.modality_submodules
        assert isinstance(mimo_model.modality_submodules["images"], VisionModalitySubmodules)
        assert isinstance(mimo_model.modality_submodules["audio"], AudioModalitySubmodules)
        # Test that language model was initialized
        assert hasattr(mimo_model, "language_model")
        assert isinstance(mimo_model.language_model, GPTModel)

        # Test that special token IDs were set correctly
        assert mimo_model.special_token_ids == self.special_token_ids

    def test_get_text_embeddings(self):
        """Test getting text embeddings."""
        # Create random input and position IDs (within vocab size range)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=device
        )
        position_ids = (
            torch.arange(self.seq_len, device=device).unsqueeze(0).expand(self.batch_size, -1)
        )
        mimo_model = get_avlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        )
        mimo_model = mimo_model.to(device)
        # Get text embeddings
        text_embeddings = mimo_model.get_text_embeddings(
            input_ids, position_ids, self.special_token_ids
        )
        # Verify shape
        # [b*s, h]
        assert text_embeddings.shape == (self.batch_size * self.seq_len, self.hidden_size)

    def test_forward_text_only(self):
        """Test forward pass with only text input."""
        # Create inputs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=device
        )
        position_ids = (
            torch.arange(self.seq_len, device=device).unsqueeze(0).expand(self.batch_size, -1)
        )

        mimo_model = get_vlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        )
        mimo_model = mimo_model.to(device)
        # Run forward pass with explicit parameters
        outputs, _ = mimo_model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=None
        )
        assert outputs is not None

        # Verify output shape
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_forward_with_image_modality(self):
        """Test forward pass with text and image input."""
        # Calculate expected number of image tokens based on image size and patch dimension
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        expected_img_seq_len = (self.img_h // self.patch_dim) * (
            self.img_w // self.patch_dim
        ) + 1  # +1 for CLS token

        # Create a fixed distribution of images: 3 in first sample, 2 in second sample
        num_images = 5
        images_per_sample = [3, 2]  # Must sum to num_images
        assert sum(images_per_sample) == num_images
        assert len(images_per_sample) == self.batch_size

        # Create images tensor
        images = torch.rand(
            num_images, 3, self.img_h, self.img_w, device=device
        )  # [num_images, 3, h, w] format

        # Create input_ids with text tokens
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=device
        )

        # Create position_ids
        position_ids = (
            torch.arange(self.seq_len, device=device).unsqueeze(0).expand(self.batch_size, -1)
        )

        # Include image special tokens in input IDs
        image_token_id = self.special_token_ids["images"]
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
                input_ids[b, start_pos : start_pos + tokens_in_this_batch] = image_token_id

        # Create modality inputs using the new structure
        modality_inputs = {"images": {"clip_encoder": {"x": images}}}

        mimo_model = get_vlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        )
        mimo_model = mimo_model.to(device)

        # Run forward pass with new interface
        outputs, _ = mimo_model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=modality_inputs
        )
        assert outputs is not None

        # Verify output shape
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_forward_with_image_and_audio_modality(self):
        """Test forward pass with text, image, and audio input."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mimo_model = get_avlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        )
        mimo_model = mimo_model.to(device)

        # Calculate image sequence length
        img_seq_len = (self.img_h // self.patch_dim) * (self.img_w // self.patch_dim) + 1

        encoder_down_sampling = 2

        # Create simple audio input (30 sec)
        mel_bins = 80  # Whisper uses 80 mel bins
        time_bins = 3000  # 30 seconds of audio at 10ms per frame
        audio_features = torch.rand(2, mel_bins, time_bins, device=device)

        # Calculate audio sequence length using Whisper's formula
        audio_seq_len = math.ceil(time_bins / encoder_down_sampling)  # 1500 tokens

        # Create batch data
        batch_size = 2
        seq_len = self.seq_len

        # Create input_ids with special tokens
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Add special tokens at specific positions
        start_pos = 5
        image_token_id = self.special_token_ids["images"]
        audio_token_id = self.special_token_ids["audio"]

        # Place image tokens followed by audio tokens in each batch item
        for i in range(batch_size):
            # Add image tokens
            input_ids[i, start_pos : start_pos + img_seq_len] = image_token_id
            # Add audio tokens after a gap
            input_ids[
                i, start_pos + img_seq_len + 10 : start_pos + img_seq_len + 10 + audio_seq_len
            ] = audio_token_id

        # Prepare modality inputs
        modality_inputs = {
            "images": {
                "clip_encoder": {"x": torch.rand(2, 3, self.img_h, self.img_w, device=device)}
            },
            "audio": {"whisper_encoder": {"input_features": audio_features}},
        }

        # Run forward pass
        outputs, _ = mimo_model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=modality_inputs
        )

        # Verify output shape
        assert outputs is not None
        assert outputs.shape == (batch_size, seq_len, self.vocab_size)

    def test_state_dict(self):
        """Test state dict methods."""
        # Get state dict
        mimo_model = get_avlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        )
        state_dict = mimo_model.state_dict()
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
        checkpoint_dict = mimo_model.state_dict_for_save_checkpoint()
        assert len(checkpoint_dict) > 0


class MockProcessGroup:
    """Mock process group for testing."""
    def __init__(self, rank, world_size):
        self._rank = rank
        self._size = world_size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class MockGrid:
    """Mock grid with HyperCommGrid-compatible interface."""
    def __init__(self, rank_offset=0, size=1, dim_names=None, pp_rank=0, pp_size=1):
        self.rank_offset = rank_offset
        self.size = size
        self.dim_names = dim_names or []
        self._pp_group = MockProcessGroup(pp_rank, pp_size)

    def get_pg(self, dims):
        if dims == "pp":
            return self._pp_group
        raise KeyError(f"Process group for {dims} not found")


class TestMimoModelNonColocated:
    """Tests for non-colocated multi-module pipeline parallelism."""

    def setup_method(self, method):
        try:
            Utils.initialize_model_parallel(1, 1)
        except Exception:
            pass
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 64
        self.vocab_size = 48000
        self.seq_len = 256
        self.batch_size = 2

    def teardown_method(self, method):
        try:
            Utils.destroy_model_parallel()
        except Exception:
            pass

    def _make_config(self, encoder_in_grid=True, language_in_grid=True, pp_rank=0, pp_size=1):
        """Helper to create MimoModelConfig with mock grids."""
        language_model_spec = get_language_model_spec(self.hidden_size, self.vocab_size, self.seq_len)
        vision_submodule_spec = get_vision_submodules_spec(self.hidden_size, 224, 224, 16)

        encoder_offset = 0 if encoder_in_grid else 10  # rank 0 in grid if offset=0
        language_offset = 0 if language_in_grid else 10

        return MimoModelConfig(
            language_model_spec=language_model_spec,
            modality_submodules_spec={"images": vision_submodule_spec},
            special_token_ids={"images": 50257},
            module_to_grid_map={
                "images": MockGrid(rank_offset=encoder_offset, size=1, dim_names=["pp"] if pp_size > 1 else [], pp_rank=pp_rank, pp_size=pp_size),
                "language": MockGrid(rank_offset=language_offset, size=1, dim_names=["pp"] if pp_size > 1 else [], pp_rank=pp_rank, pp_size=pp_size),
            },
            language_module_key="language",
        )

    def test_grid_validation_rejects_mismatched_keys(self):
        """Test validation fails when grid_map keys don't match expected modules."""
        language_model_spec = get_language_model_spec(self.hidden_size, self.vocab_size, self.seq_len)
        vision_submodule_spec = get_vision_submodules_spec(self.hidden_size, 224, 224, 16)

        # Missing 'images' in grid_map
        mimo_config = MimoModelConfig(
            language_model_spec=language_model_spec,
            modality_submodules_spec={"images": vision_submodule_spec},
            special_token_ids={"images": 50257},
            module_to_grid_map={"language": MockGrid()},
            language_module_key="language",
        )

        with pytest.raises(ValueError, match="module_to_grid_map keys must match"):
            MimoModel(mimo_config)

    def test_role_determination(self):
        """Test role correctly identifies modules and stage positions."""
        # No grid map = no role
        model_no_grid = get_vlm_mimo_model(self.hidden_size, self.vocab_size, self.seq_len, 224, 224, 16, {"images": 50257})
        assert model_no_grid.role is None

        # Encoder-only rank (language grid excludes rank 0)
        model_encoder = MimoModel(self._make_config(encoder_in_grid=True, language_in_grid=False))
        assert model_encoder.role.has_modality_modules is True
        assert model_encoder.role.has_language_module is False

        # Language-only rank (encoder grid excludes rank 0)
        model_language = MimoModel(self._make_config(encoder_in_grid=False, language_in_grid=True))
        assert model_language.role.has_modality_modules is False
        assert model_language.role.has_language_module is True

        # Stage info with PP
        model_pp = MimoModel(self._make_config(encoder_in_grid=True, language_in_grid=True, pp_rank=1, pp_size=3))
        assert model_pp.role.is_first_stage("images") is False
        assert model_pp.role.is_last_stage("images") is False

    def test_selective_init_encoder_only(self):
        """Test encoder-only rank initializes encoder but not language model."""
        model = MimoModel(self._make_config(encoder_in_grid=True, language_in_grid=False))
        assert "images" in model.modality_submodules
        assert model.language_model is None

    def test_selective_init_language_only(self):
        """Test language-only rank initializes language model but not encoder."""
        model = MimoModel(self._make_config(encoder_in_grid=False, language_in_grid=True))
        assert "images" not in model.modality_submodules
        assert model.language_model is not None

    def test_forward_encoder_only(self):
        """Test encoder-only forward returns dict of embeddings."""
        model = MimoModel(self._make_config(encoder_in_grid=True, language_in_grid=False))
        model = model.to(self.device)

        images = torch.rand(2, 3, 224, 224, device=self.device)
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)

        outputs, _ = model(input_ids=input_ids, modality_inputs={"images": {"clip_encoder": {"x": images}}})

        assert isinstance(outputs, dict)
        assert "images" in outputs

    def test_forward_language_only(self):
        """Test language-only forward returns tensor."""
        model = MimoModel(self._make_config(encoder_in_grid=False, language_in_grid=True))
        model = model.to(self.device)

        img_seq_len = (224 // 16) * (224 // 16) + 1
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        input_ids[:, 5:5 + img_seq_len] = 50257  # image tokens
        position_ids = torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(self.batch_size, -1)

        # Simulate encoder output from previous stage
        encoder_embeddings = torch.randn(self.batch_size * img_seq_len, self.hidden_size, device=self.device)
        model.set_input_tensor({"images": encoder_embeddings})

        outputs, _ = model(input_ids=input_ids, position_ids=position_ids, modality_inputs=None)

        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)
