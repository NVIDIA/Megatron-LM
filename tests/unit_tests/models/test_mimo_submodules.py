# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m torch.distributed.run \
    --nproc_per_node=1 -m pytest \
    tests/unit_tests/models/test_mimo_submodules.py -v
'''

from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mimo.submodules.base import ModalitySubmodules
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class MockModalitySubmodule(ModalitySubmodules):
    """Concrete implementation of ModalitySubmodules for testing purposes."""

    def combine_embeddings(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        return

    def encode(self, data_batch: Dict) -> List[torch.Tensor]:
        return []

    def decode(self, embeddings: torch.Tensor, data_batch: Dict) -> torch.Tensor:
        return

    def project_embeddings(
        self, embeddings: List[torch.Tensor], is_input: bool = True
    ) -> Optional[torch.Tensor]:
        return None

    def forward(self, encoder_inputs: Dict[str, Any], seq_lengths: Optional[torch.Tensor] = None):
        return None


@pytest.mark.experimental
class TestBaseSubmodule:
    """Test the base ModalitySubmodules class initialization."""

    def setup_method(self, method):
        '''setup env'''
        # Initialize distributed environment
        try:
            Utils.initialize_model_parallel(1, 1)
        except Exception as e:
            print(f"Warning: Could not initialize model parallel: {e}")

        # Create transformer config for vision encoder
        self.vision_config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )

        # Create layer spec for transformer
        self.layer_spec = get_gpt_layer_with_transformer_engine_spec()

        # Define vision encoder parameters
        self.img_h = 224
        self.img_w = 224
        self.patch_dim = 16

        # Create encoder spec (using CLIP ViT model)
        self.encoder_spec = ModuleSpec(
            module=CLIPViTModel,
            params={
                "transformer_config": self.vision_config,
                "transformer_layer_spec": self.layer_spec,
                "img_h": self.img_h,
                "img_w": self.img_w,
                "patch_dim": self.patch_dim,
            },
        )

        # Create projection spec
        self.projection_spec = ModuleSpec(
            module=nn.Linear,
            params={
                "in_features": self.vision_config.hidden_size,
                "out_features": self.vision_config.hidden_size,
            },
        )

        # Create the main module spec
        self.module_spec = ModuleSpec(
            module=MockModalitySubmodule,
            submodules={
                "encoders": {"clip_encoder": self.encoder_spec},
                "input_projections": [self.projection_spec],
            },
        )

    def teardown_method(self, method):
        '''teardown env'''
        try:
            Utils.destroy_model_parallel()
        except Exception as e:
            print(f"Warning: Could not destroy model parallel: {e}")

    def test_initialize_with_modules(self):
        """Test constructor with pre-built modules."""
        # Create actual modules
        encoder = CLIPViTModel(
            transformer_config=self.vision_config,
            transformer_layer_spec=self.layer_spec,
            img_h=self.img_h,
            img_w=self.img_w,
            patch_dim=self.patch_dim,
        )

        projection = nn.Linear(
            in_features=self.vision_config.hidden_size, out_features=self.vision_config.hidden_size
        )

        # Create submodule with modules
        submodule = MockModalitySubmodule(
            encoders={"clip_encoder": encoder}, input_projections=[projection]
        )

        # Check modules are set correctly
        assert len(submodule.encoders) == 1
        assert len(submodule.decoders) == 0
        assert len(submodule.input_projections) == 1
        assert len(submodule.output_projections) == 0

        # Check the encoder module is of the right type
        assert isinstance(submodule.encoders['clip_encoder'], CLIPViTModel)

        # Check the projection module is of the right type
        assert isinstance(submodule.input_projections[0], nn.Linear)

    def test_initialize_from_spec(self):
        """Test creating a submodule from a ModuleSpec with real modules."""
        # Create from spec
        submodule_from_spec = MockModalitySubmodule.from_spec(self.module_spec)

        # Verify the submodule was created correctly
        assert len(submodule_from_spec.encoders) == 1
        assert len(submodule_from_spec.decoders) == 0
        assert len(submodule_from_spec.input_projections) == 1
        assert len(submodule_from_spec.output_projections) == 0

        # Check the encoder modules are of the right type
        assert isinstance(submodule_from_spec.encoders['clip_encoder'], CLIPViTModel)

        # Check the projection module is of the right type
        assert isinstance(submodule_from_spec.input_projections[0], nn.Linear)

        # Check parameters of the encoder
        encoder = submodule_from_spec.encoders['clip_encoder']
        assert encoder.img_h == self.img_h
        assert encoder.img_w == self.img_w
        assert encoder.patch_dim == self.patch_dim

        # Check parameters of the projection
        projection = submodule_from_spec.input_projections[0]
        assert projection.in_features == self.vision_config.hidden_size
        assert projection.out_features == self.vision_config.hidden_size


@pytest.mark.experimental
class TestVisionSubmodule:
    """Test the VisionModalitySubmodules class with forward passes."""

    def setup_method(self, method):
        '''setup env'''
        # Initialize distributed environment
        try:
            Utils.initialize_model_parallel(1, 1)
        except Exception as e:
            print(f"Warning: Could not initialize model parallel: {e}")

        model_parallel_cuda_manual_seed(123)

        self.hidden_size = 64
        self.vision_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        # Create layer spec for transformer
        self.layer_spec = get_gpt_layer_with_transformer_engine_spec()

        # Define vision parameters
        self.img_h = 224
        self.img_w = 224
        self.patch_dim = 16

        # Create vision encoder
        self.vision_encoder = CLIPViTModel(
            transformer_config=self.vision_config,
            transformer_layer_spec=self.layer_spec,
            img_h=self.img_h,
            img_w=self.img_w,
            patch_dim=self.patch_dim,
        )

        # Create projection layer
        self.input_projection = nn.Linear(self.hidden_size, self.hidden_size)

        # Create output projection
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)

        # Create VisionModalitySubmodules with encoder and projection
        self.vision_submodule = VisionModalitySubmodules(
            encoders={"clip_encoder": self.vision_encoder},
            input_projections=[self.input_projection],
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_submodule = self.vision_submodule.to(self.device)

        # Set all modules to eval mode to disable dropout and other stochastic layers
        # This makes tests more deterministic
        self.vision_submodule.eval()
        self.vision_encoder.eval()
        self.input_projection.eval()

    def teardown_method(self, method):
        '''teardown env'''
        try:
            Utils.destroy_model_parallel()
        except Exception as e:
            print(f"Warning: Could not destroy model parallel: {e}")

    def test_encode_with_random_data(self):
        """Test encoding with random image data."""
        # Create random batch of images
        num_images = 2
        images = torch.rand(num_images, 3, self.img_h, self.img_w, device=self.device)
        data_batch = {"clip_encoder": {"x": images}}

        # Test encode method
        embeddings = self.vision_submodule.encode(data_batch)

        # Verify embeddings shape and content
        assert len(embeddings) == 1  # One encoder
        embedding = embeddings[0]

        # Number of tokens depends on image size and patch size
        expected_seq_len = (self.img_h // self.patch_dim) * (
            self.img_w // self.patch_dim
        ) + 1  # +1 for cls token
        assert embedding.shape[0] == num_images * expected_seq_len
        assert embedding.shape[1] == self.hidden_size

    def test_combine_embeddings(self):
        """Test combining embeddings functionality."""
        # Create test embeddings with different sequence lengths
        num_images = 2
        seq_len1 = 10
        seq_len2 = 15

        # Create test embeddings
        embedding1 = torch.rand(num_images * seq_len1, self.hidden_size, device=self.device)
        embedding2 = torch.rand(num_images * seq_len2, self.hidden_size, device=self.device)
        embeddings = [embedding1, embedding2]

        # Test combining embeddings
        combined = self.vision_submodule.combine_embeddings(embeddings)
        assert combined.shape == (num_images * (seq_len1 + seq_len2), self.hidden_size)

        # Test combining a single embedding
        single_combined = self.vision_submodule.combine_embeddings([embedding1])
        assert single_combined.shape == (num_images * seq_len1, self.hidden_size)
        assert torch.all(single_combined == embedding1)

        # Test combining empty embeddings raises error
        with pytest.raises(ValueError):
            self.vision_submodule.combine_embeddings([])

    def test_forward_pass(self):
        """Test the complete forward pass."""
        # Create random batch of images
        num_images = 2
        images = torch.rand(num_images, 3, self.img_h, self.img_w, device=self.device)
        data_batch = {"clip_encoder": {"x": images}}

        # Test forward pass
        output = self.vision_submodule(data_batch)
        assert output is not None

        # Check output shape - flattened to [num_image_embeddings, hidden_dim]
        expected_seq_len = (self.img_h // self.patch_dim) * (self.img_w // self.patch_dim) + 1
        expected_total_embeddings = num_images * expected_seq_len
        assert output.shape == (expected_total_embeddings, self.hidden_size)

    def test_empty_data_batch(self):
        """Test forward pass with empty data batch."""
        # Create a data batch without images
        data_batch = {}

        # Test forward pass
        output = self.vision_submodule(data_batch)
        assert output is None
