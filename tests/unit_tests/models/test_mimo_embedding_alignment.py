# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m pytest tests/unit_tests/models/test_mimo_embedding_alignment.py
'''

from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.models.mimo.config import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.transformer.spec_utils import ModuleSpec


class TestEmbeddingAlignment:
    """Test the align_embeddings_by_token_positions method in MimoModel."""

    def setup_method(self):
        """Set up for each test."""
        # Create a minimal MimoModelConfig
        language_model_spec = ModuleSpec(module=MagicMock, params={'config': MagicMock()})
        self.mimo_config = MimoModelConfig(
            language_model_spec=language_model_spec,
            modality_submodules_spec={},
            special_token_ids={},
        )

        # Create MimoModel instance
        self.model = MimoModel(self.mimo_config)

        self.hidden_dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_marker_embeddings(self, num_embeddings, marker_positions=None, marker_values=None):
        """Create embeddings with marker values at specific positions.

        Args:
            num_embeddings: Number of embeddings to create
            marker_positions: List of positions to place markers. If None, uses range(num_embeddings)
            marker_values: List of values to use for markers. If None, uses [10.0, 20.0, ...]

        Returns:
            Tensor of shape [num_embeddings, hidden_dim] with markers at specified positions
        """
        if marker_positions is None:
            marker_positions = list(range(num_embeddings))

        embeddings = torch.zeros((num_embeddings, self.hidden_dim), device=self.device)

        # Set distinctive markers
        for i, pos in enumerate(marker_positions):
            # Use provided value or default pattern
            if marker_values is not None and i < len(marker_values):
                marker_value = marker_values[i]
            else:
                marker_value = float(i + 1) * 10.0  # Values like 10.0, 20.0, 30.0, etc.

            embeddings[i, pos % self.hidden_dim] = marker_value

        return embeddings

    def test_basic_alignment(self):
        """Test basic alignment with text and one modality."""
        # Create a simple batch
        batch_size = 2
        seq_length = 8
        hidden_dim = self.hidden_dim

        # Create input_ids with special tokens
        # Sequence 1: [text, image_token, text, text, text, text, text, text]
        # Sequence 2: [text, text, text, image_token, text, text, text, text]
        input_ids = torch.full((batch_size, seq_length), 100, dtype=torch.long, device=self.device)

        # Add image special tokens at different positions for each sequence
        image_token_id = 50
        input_ids[0, 1] = image_token_id  # Batch 0, position 1
        input_ids[1, 3] = image_token_id  # Batch 1, position 3

        # Create text embeddings (14 tokens total - 7 text tokens per sequence)
        # Instead of zeros, use a small distinct value for text embeddings
        text_embeddings = torch.full((14, hidden_dim), 0.01, device=self.device)

        # Create vision embeddings with distinctive markers
        # For batch 0: marker at position 0 with value 10.0
        # For batch 1: marker at position 1 with value 20.0
        vision_embeddings = self.create_marker_embeddings(2, marker_positions=[0, 1])

        # Define special token IDs
        special_token_ids = {"vision": image_token_id}

        # Align embeddings
        modality_embeddings = {"text": text_embeddings, "vision": vision_embeddings}

        combined = self.model.align_embeddings_by_token_positions(
            modality_embeddings=modality_embeddings,
            input_ids=input_ids,
            special_token_ids=special_token_ids,
        )

        # Check output shape
        assert combined.shape == (seq_length, batch_size, hidden_dim)

        # Check special token positions have the correct embeddings
        # First vision token (Batch 0, Seq 1) should have the first vision embedding
        assert combined[1, 0, 0] == 10.0  # First marker
        assert torch.all(combined[1, 0, 1:] == 0.0), "Non-zero values found after marker"

        # Second vision token (Batch 1, Seq 3) should have the second vision embedding
        assert combined[3, 1, 1] == 20.0  # Second marker
        assert torch.all(combined[3, 1, :1] == 0.0), "Non-zero values found before marker"
        assert torch.all(combined[3, 1, 2:] == 0.0), "Non-zero values found after marker"

        # Verify text positions have only zeros
        text_positions = [
            (0, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),  # Batch 0
            (0, 1),
            (1, 1),
            (2, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),  # Batch 1
        ]

        for s, b in text_positions:
            assert torch.all(combined[s, b] == 0.01)

    def test_multiple_modalities(self):
        """Test alignment with multiple modalities with special tokens at different positions."""
        batch_size = 2
        seq_length = 10
        hidden_dim = self.hidden_dim

        # Create input_ids with special tokens for multiple modalities
        # Sequence 1: [text, vision, text, text, audio, text, text, text, video, text]
        # Sequence 2: [text, text, vision, text, text, audio, text, video, text, text]
        input_ids = torch.full((batch_size, seq_length), 100, dtype=torch.long, device=self.device)

        # Define special token IDs
        vision_token_id = 50
        audio_token_id = 51
        video_token_id = 52

        # Add special tokens at different positions in each sequence
        # First sequence
        input_ids[0, 1] = vision_token_id  # Vision at pos 1 in seq 0
        input_ids[0, 4] = audio_token_id  # Audio at pos 4 in seq 0
        input_ids[0, 8] = video_token_id  # Video at pos 8 in seq 0

        # Second sequence
        input_ids[1, 2] = vision_token_id  # Vision at pos 2 in seq 1
        input_ids[1, 5] = audio_token_id  # Audio at pos 5 in seq 1
        input_ids[1, 7] = video_token_id  # Video at pos 7 in seq 1

        # Calculate text tokens: 7 tokens in each sequence
        # Create non-zero text embeddings for better verification
        text_embeddings = torch.full((14, hidden_dim), 0.01, device=self.device)

        # Create marker embeddings for each modality with specific positions and values
        # For vision: both embeddings have markers at position 0
        vision_embeddings = self.create_marker_embeddings(
            num_embeddings=2,
            marker_positions=[0, 0],  # Both markers at position 0
            marker_values=[10.0, 20.0],  # Batch 0 and Batch 1 markers
        )

        # For audio: both embeddings have markers at position 1
        audio_embeddings = self.create_marker_embeddings(
            num_embeddings=2,
            marker_positions=[1, 1],  # Both markers at position 1
            marker_values=[30.0, 40.0],  # Batch 0 and Batch 1 markers
        )

        # For video: both embeddings have markers at position 2
        video_embeddings = self.create_marker_embeddings(
            num_embeddings=2,
            marker_positions=[2, 2],  # Both markers at position 2
            marker_values=[50.0, 60.0],  # Batch 0 and Batch 1 markers
        )

        # Define special token mapping
        special_token_ids = {
            "vision": vision_token_id,
            "audio": audio_token_id,
            "video": video_token_id,
        }

        # Align embeddings
        modality_embeddings = {
            "text": text_embeddings,
            "vision": vision_embeddings,
            "audio": audio_embeddings,
            "video": video_embeddings,
        }

        combined = self.model.align_embeddings_by_token_positions(
            modality_embeddings=modality_embeddings,
            input_ids=input_ids,
            special_token_ids=special_token_ids,
        )

        # Check output shape
        assert combined.shape == (seq_length, batch_size, hidden_dim)

        # Check that special token positions have the correct markers and only at correct positions

        # Batch 0 markers
        assert torch.isclose(combined[1, 0, 0], torch.tensor(10.0, device=self.device))  # Vision
        assert torch.isclose(combined[4, 0, 1], torch.tensor(30.0, device=self.device))  # Audio
        assert torch.isclose(combined[8, 0, 2], torch.tensor(50.0, device=self.device))  # Video

        # Batch 1 markers
        assert torch.isclose(combined[2, 1, 0], torch.tensor(20.0, device=self.device))  # Vision
        assert torch.isclose(combined[5, 1, 1], torch.tensor(40.0, device=self.device))  # Audio
        assert torch.isclose(combined[7, 1, 2], torch.tensor(60.0, device=self.device))  # Video

        # Also check that markers are ONLY at their specific positions
        # For vision in batch 0 (position 1, value at index 0)
        assert torch.all(combined[1, 0, 1:] == 0.0), "Non-zero values found after marker"

        # For audio in batch 1 (position 5, value at index 1)
        assert torch.all(combined[5, 1, :1] == 0.0), "Non-zero values found before marker"
        assert torch.all(combined[5, 1, 2:] == 0.0), "Non-zero values found after marker"

    def test_multiple_images_with_variable_length(self):
        """Test handling multiple images per sample with variable sequence lengths.

        This test verifies that:
        1. Multiple image occurrences per batch sample are handled correctly
        2. Images with different sequence lengths are processed properly
        3. The batch-first ordering is preserved
        4. Embeddings are correctly placed at their corresponding positions
        """
        # Create a test case with 2 batches:
        # - Batch 0: 2 images with different sequence lengths (3 and 2 patches)
        # - Batch 1: 1 image with 4 patches
        batch_size = 2
        seq_length = 10
        hidden_dim = self.hidden_dim

        # Create input_ids with vision special tokens
        input_ids = torch.full((batch_size, seq_length), 100, dtype=torch.long, device=self.device)

        # Define vision token ID
        vision_token_id = 50

        # Place special tokens:
        # Batch 0: positions 1, 2, 3 (first image, 3 patches) and 5, 6 (second image, 2 patches)
        # Batch 1: positions 2, 3, 4, 5 (one image, 4 patches)
        # Batch 0 - first image (3 patches)
        input_ids[0, 1] = vision_token_id
        input_ids[0, 2] = vision_token_id
        input_ids[0, 3] = vision_token_id

        # Batch 0 - second image (2 patches)
        input_ids[0, 5] = vision_token_id
        input_ids[0, 6] = vision_token_id

        # Batch 1 - one image (4 patches)
        input_ids[1, 2] = vision_token_id
        input_ids[1, 3] = vision_token_id
        input_ids[1, 4] = vision_token_id
        input_ids[1, 5] = vision_token_id

        # Count text tokens (all non-vision tokens)
        # Batch 0: 5 text tokens (positions 0, 4, 7, 8, 9)
        # Batch 1: 6 text tokens (positions 0, 1, 6, 7, 8, 9)
        text_embeddings = torch.full((11, hidden_dim), 0.01, device=self.device)

        # Create the unflattened embeddings that would come from a vision encoder
        # First, create 3 tensors with different sequence lengths:

        # Batch 0, Image 1: 3 patches
        image_0_1 = self.create_marker_embeddings(
            num_embeddings=3,
            marker_positions=[0, 1, 2],
            marker_values=[101.0, 102.0, 103.0],  # Distinct values for each patch
        )

        # Batch 0, Image 2: 2 patches
        image_0_2 = self.create_marker_embeddings(
            num_embeddings=2, marker_positions=[3, 4], marker_values=[104.0, 105.0]
        )

        # Batch 1, Image 1: 4 patches
        image_1_1 = self.create_marker_embeddings(
            num_embeddings=4,
            marker_positions=[5, 6, 7, 8],
            marker_values=[201.0, 202.0, 203.0, 204.0],
        )

        # Flatten the images as the vision submodule would do
        # They should be concatenated in batch order
        vision_embeddings = torch.cat([image_0_1, image_0_2, image_1_1], dim=0)

        # Define special token IDs
        special_token_ids = {"vision": vision_token_id}

        # Create modality embeddings
        modality_embeddings = {"text": text_embeddings, "vision": vision_embeddings}

        # Align embeddings
        combined = self.model.align_embeddings_by_token_positions(
            modality_embeddings=modality_embeddings,
            input_ids=input_ids,
            special_token_ids=special_token_ids,
        )

        # Check output shape
        assert combined.shape == (seq_length, batch_size, hidden_dim)

        # Verify vision token embeddings are placed correctly

        # Batch 0, first image embeddings (3 patches)
        assert torch.isclose(combined[1, 0, 0], torch.tensor(101.0, device=self.device))
        assert torch.isclose(combined[2, 0, 1], torch.tensor(102.0, device=self.device))
        assert torch.isclose(combined[3, 0, 2], torch.tensor(103.0, device=self.device))

        # Batch 0, second image embeddings (2 patches)
        assert torch.isclose(combined[5, 0, 3], torch.tensor(104.0, device=self.device))
        assert torch.isclose(combined[6, 0, 4], torch.tensor(105.0, device=self.device))

        # Batch 1, image embeddings (4 patches)
        assert torch.isclose(combined[2, 1, 5], torch.tensor(201.0, device=self.device))
        assert torch.isclose(combined[3, 1, 6], torch.tensor(202.0, device=self.device))
        assert torch.isclose(combined[4, 1, 7], torch.tensor(203.0, device=self.device))
        assert torch.isclose(combined[5, 1, 8], torch.tensor(204.0, device=self.device))

        # Verify that each embedding only has one non-zero value
        for b in range(batch_size):
            # Check positions with special tokens
            positions = [(1, 2, 3, 5, 6), (2, 3, 4, 5)][b]
            for s in positions:
                emb = combined[s, b].clone()
                # Find the non-zero position
                nonzero_indices = torch.nonzero(emb)
                # Make sure we actually have non-zero values
                assert (
                    nonzero_indices.nelement() > 0
                ), f"No non-zero values found at position {s},{b}"
                nonzero_pos = nonzero_indices[0].item()
                # Check that all other positions are zero
                assert torch.all(
                    emb[:nonzero_pos] == 0.0
                ), f"Non-zero values found before marker at {s},{b}"
                assert torch.all(
                    emb[nonzero_pos + 1 :] == 0.0
                ), f"Non-zero values found after marker at {s},{b}"

    def test_validation_errors(self):
        """Test validation errors when token counts don't match embedding counts."""
        batch_size = 2
        seq_length = 5
        hidden_dim = self.hidden_dim

        # Create input_ids with different numbers of tokens
        input_ids = torch.full((batch_size, seq_length), 100, dtype=torch.long, device=self.device)

        # Add 3 special tokens for vision
        vision_token_id = 50
        input_ids[0, 1] = vision_token_id
        input_ids[0, 3] = vision_token_id
        input_ids[1, 2] = vision_token_id

        # Create text embeddings (non-zero for better verification)
        # We have 3 vision tokens, so we need:
        # (batch_size * seq_length) - num_vision_tokens = 2*5 - 3 = 7 text embeddings
        text_embeddings = torch.full((7, hidden_dim), 0.01, device=self.device)

        # Create vision embeddings with only 2 embeddings (not enough for 3 tokens)
        vision_embeddings = self.create_marker_embeddings(2)

        special_token_ids = {"vision": vision_token_id}

        modality_embeddings = {"text": text_embeddings, "vision": vision_embeddings}

        # Should raise a ValueError because we have 3 special tokens but only 2 embeddings
        with pytest.raises(ValueError, match="Number of vision tokens.*does not match"):
            self.model.align_embeddings_by_token_positions(
                modality_embeddings=modality_embeddings,
                input_ids=input_ids,
                special_token_ids=special_token_ids,
            )

        # Test with wrong number of text tokens
        input_ids = torch.full((batch_size, seq_length), 100, dtype=torch.long, device=self.device)

        # Add 1 special token in each batch
        input_ids[0, 1] = vision_token_id
        input_ids[1, 2] = vision_token_id

        # This would leave 8 text tokens (4 per batch), but we'll provide only 6
        text_embeddings = torch.full((6, hidden_dim), 0.01, device=self.device)

        # Create matching vision embeddings (correct count this time)
        vision_embeddings = self.create_marker_embeddings(2)

        modality_embeddings = {"text": text_embeddings, "vision": vision_embeddings}

        # Should raise a ValueError for mismatched text token count
        with pytest.raises(ValueError, match="Number of text tokens.*does not match"):
            self.model.align_embeddings_by_token_positions(
                modality_embeddings=modality_embeddings,
                input_ids=input_ids,
                special_token_ids=special_token_ids,
            )

    def test_missing_special_token_id(self):
        """Test error when a modality is missing from special_token_ids."""
        batch_size = 2
        seq_length = 5
        hidden_dim = self.hidden_dim

        # Create input_ids
        input_ids = torch.full((batch_size, seq_length), 100, dtype=torch.long, device=self.device)

        # Define text embeddings with non-zero value
        text_embeddings = torch.full(
            (batch_size * seq_length, hidden_dim), 0.01, device=self.device
        )

        # Create vision embeddings (not referenced in special_token_ids)
        vision_embeddings = self.create_marker_embeddings(1)

        # Empty special_token_ids
        special_token_ids = {}

        modality_embeddings = {
            "text": text_embeddings,
            "vision": vision_embeddings,  # Not in special_token_ids
        }

        # Should raise a ValueError because vision modality is missing from special_token_ids
        with pytest.raises(ValueError, match="No special token ID defined for modality vision"):
            self.model.align_embeddings_by_token_positions(
                modality_embeddings=modality_embeddings,
                input_ids=input_ids,
                special_token_ids=special_token_ids,
            )
