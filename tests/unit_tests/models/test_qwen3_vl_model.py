# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Test for Qwen3-VL multi-modal model.
from unittest.mock import patch

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.multimodal.qwen3_vl_model import (
    DEFAULT_IMAGE_TOKEN_INDEX,
    DEFAULT_VIDEO_TOKEN_INDEX,
    Qwen3VLModel,
    Qwen3VLTransformerBlock,
    Qwen3VLVisionEncoder,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class MockVisionEncoder(torch.nn.Module):
    """Mock vision encoder that avoids HuggingFace model loading."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.hidden_size = output_dim
        self.proj = torch.nn.Linear(input_dim, output_dim)
        self.image_token_id = DEFAULT_IMAGE_TOKEN_INDEX
        self.video_token_id = DEFAULT_VIDEO_TOKEN_INDEX

    def forward(self, pixel_values, grid_thw=None):
        output = self.proj(pixel_values)
        return output, []


class TestQwen3VLModel:

    @pytest.mark.internal
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        self.language_hidden_size = 64
        self.language_num_attention_heads = 4
        self.seq_len = 128
        self.vocab_size = 8192

        self.language_config = TransformerConfig(
            num_layers=3,
            hidden_size=self.language_hidden_size,
            num_attention_heads=self.language_num_attention_heads,
            use_cpu_initialization=False,
        )

        self.language_layer_spec = get_gpt_layer_with_transformer_engine_spec()

        # Create model without vision encoder to avoid HuggingFace dependency.
        # Vision encoder is mocked in tests that need it.
        self.model = Qwen3VLModel(
            language_transformer_config=self.language_config,
            language_transformer_layer_spec=self.language_layer_spec,
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=4096,
            add_encoder=False,
            add_decoder=True,
            image_token_index=DEFAULT_IMAGE_TOKEN_INDEX,
            video_token_index=DEFAULT_VIDEO_TOKEN_INDEX,
        )

    @pytest.mark.internal
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.model, Qwen3VLModel)
        assert self.model.language_model is not None
        assert self.model.visual is None
        assert self.model.add_decoder is True
        assert self.model.add_encoder is False
        assert self.model.image_token_index == DEFAULT_IMAGE_TOKEN_INDEX
        assert self.model.video_token_index == DEFAULT_VIDEO_TOKEN_INDEX

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights > 0

    @pytest.mark.internal
    def test_constructor_with_encoder(self):
        """Test constructor with add_encoder=True using mocked HF loading."""
        with patch.object(Qwen3VLVisionEncoder, '_load_hf_vision_model'):
            model = Qwen3VLModel(
                language_transformer_config=self.language_config,
                language_transformer_layer_spec=self.language_layer_spec,
                language_vocab_size=self.vocab_size,
                language_max_sequence_length=4096,
                add_encoder=True,
                add_decoder=True,
            )
        assert model.visual is not None
        assert isinstance(model.visual, Qwen3VLVisionEncoder)
        assert model.language_model is not None
        assert model.add_encoder is True

    @pytest.mark.internal
    def test_set_input_tensor(self):
        expected_shape = (1, 2, 3, 4)
        input_tensor = torch.zeros(expected_shape)
        self.model.set_input_tensor(input_tensor)
        assert self.model.language_model.decoder.input_tensor.shape == expected_shape

    @pytest.mark.internal
    def test_preprocess_data_no_images(self):
        """Test _preprocess_data returns text embeddings when no images are provided."""
        self.model.cuda()

        batch_size = 2
        seq_len = self.seq_len

        input_ids = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).expand(batch_size, seq_len).cuda()
        labels = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        loss_mask = torch.ones((batch_size, seq_len)).cuda()

        combined_embeds, pos_ids, visual_pos_mask, deepstack_embeds, new_labels, new_loss_mask = (
            self.model._preprocess_data(
                input_ids=input_ids,
                position_ids=position_ids,
                image_embeds=None,
                deepstack_embeds=None,
                labels=labels,
                loss_mask=loss_mask,
            )
        )

        # Sequence length unchanged, no visual mask.
        assert combined_embeds.shape == torch.Size(
            (seq_len, batch_size, self.language_hidden_size)
        )
        assert visual_pos_mask is None
        assert torch.equal(new_labels, labels)
        assert torch.equal(new_loss_mask, loss_mask)

    @pytest.mark.internal
    def test_preprocess_data_with_images(self):
        """Test _preprocess_data replaces image token positions via masked_scatter."""
        self.model.cuda()

        batch_size = 2
        seq_len = 64
        hidden_size = self.language_hidden_size
        num_img_tokens_per_sample = 5

        # Create input_ids with image tokens at known positions.
        input_ids = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        # Sample 0: image tokens at positions 10-14
        for i in range(num_img_tokens_per_sample):
            input_ids[0, 10 + i] = DEFAULT_IMAGE_TOKEN_INDEX
        # Sample 1: image tokens at positions 20-24
        for i in range(num_img_tokens_per_sample):
            input_ids[1, 20 + i] = DEFAULT_IMAGE_TOKEN_INDEX

        position_ids = torch.arange(seq_len).expand(batch_size, seq_len).cuda()
        labels = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        loss_mask = torch.ones((batch_size, seq_len)).cuda()

        total_image_tokens = num_img_tokens_per_sample * batch_size
        image_embeds = torch.randn(total_image_tokens, hidden_size).cuda()

        combined_embeds, pos_ids, visual_pos_mask, deepstack_embeds, new_labels, new_loss_mask = (
            self.model._preprocess_data(
                input_ids=input_ids,
                position_ids=position_ids,
                image_embeds=image_embeds,
                deepstack_embeds=None,
                labels=labels,
                loss_mask=loss_mask,
            )
        )

        # Sequence length is unchanged (masked_scatter, not expansion like LLaVA).
        assert combined_embeds.shape == torch.Size((seq_len, batch_size, hidden_size))

        # Visual position mask marks image token positions.
        assert visual_pos_mask is not None
        assert visual_pos_mask.shape == torch.Size((seq_len, batch_size))

        # Check mask is True at image positions and False elsewhere.
        for i in range(num_img_tokens_per_sample):
            assert visual_pos_mask[10 + i, 0].item() is True
            assert visual_pos_mask[20 + i, 1].item() is True
        assert visual_pos_mask[0, 0].item() is False
        assert visual_pos_mask[0, 1].item() is False

        # Verify image embeddings were scattered into the correct positions.
        # combined_embeds is [seq, batch, hidden]; transpose to [batch, seq, hidden].
        result = combined_embeds.transpose(0, 1)
        assert torch.allclose(
            result[0, 10:15], image_embeds[0:5].to(result.dtype), atol=1e-5
        )
        assert torch.allclose(
            result[1, 20:25], image_embeds[5:10].to(result.dtype), atol=1e-5
        )

        # Labels and loss_mask are unchanged.
        assert torch.equal(new_labels, labels)
        assert torch.equal(new_loss_mask, loss_mask)

    @pytest.mark.internal
    def test_preprocess_data_with_video_tokens(self):
        """Test _preprocess_data handles video tokens in the visual mask."""
        self.model.cuda()

        batch_size = 1
        seq_len = 32
        hidden_size = self.language_hidden_size

        input_ids = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        # Mix of image and video tokens.
        input_ids[0, 5] = DEFAULT_IMAGE_TOKEN_INDEX
        input_ids[0, 6] = DEFAULT_IMAGE_TOKEN_INDEX
        input_ids[0, 10] = DEFAULT_VIDEO_TOKEN_INDEX
        input_ids[0, 11] = DEFAULT_VIDEO_TOKEN_INDEX

        position_ids = torch.arange(seq_len).expand(batch_size, seq_len).cuda()

        # 4 total visual tokens (2 image + 2 video).
        image_embeds = torch.randn(4, hidden_size).cuda()

        combined_embeds, pos_ids, visual_pos_mask, _, _, _ = self.model._preprocess_data(
            input_ids=input_ids,
            position_ids=position_ids,
            image_embeds=image_embeds,
            deepstack_embeds=None,
        )

        # Visual mask includes both image and video positions.
        assert visual_pos_mask[5, 0].item() is True
        assert visual_pos_mask[6, 0].item() is True
        assert visual_pos_mask[10, 0].item() is True
        assert visual_pos_mask[11, 0].item() is True
        assert visual_pos_mask[0, 0].item() is False

    @pytest.mark.internal
    def test_forward_text_only(self):
        """Test forward pass with text-only input (no images)."""
        self.model.cuda()

        batch_size = 2
        seq_len = self.seq_len

        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).expand(batch_size, seq_len).cuda()
        labels = torch.randint(0, self.vocab_size, (batch_size, seq_len)).cuda()
        loss_mask = torch.ones((batch_size, seq_len)).cuda()

        output, new_loss_mask = self.model.forward(
            images=None,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
        )

        # With labels, output is per-token loss.
        assert output.shape == torch.Size((batch_size, seq_len))
        assert new_loss_mask.shape == torch.Size((batch_size, seq_len))

    @pytest.mark.internal
    def test_forward_text_only_no_labels(self):
        """Test forward pass without labels returns logits."""
        self.model.cuda()

        batch_size = 2
        seq_len = self.seq_len

        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).expand(batch_size, seq_len).cuda()

        output, new_loss_mask = self.model.forward(
            images=None,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=None,
            loss_mask=None,
        )

        # Without labels, output is logits.
        assert output.shape == torch.Size((batch_size, seq_len, self.vocab_size))
        assert new_loss_mask is None

    @pytest.mark.internal
    def test_forward_with_images(self):
        """Test forward pass with images using a mock vision encoder."""
        self.model.cuda()

        batch_size = 2
        seq_len = self.seq_len
        num_image_tokens = 10
        mock_input_dim = 16

        # Attach a mock vision encoder.
        mock_vision = MockVisionEncoder(mock_input_dim, self.language_hidden_size).cuda()
        self.model.visual = mock_vision
        self.model.add_encoder = True

        # Place image tokens in input_ids.
        input_ids = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        for i in range(num_image_tokens):
            input_ids[0, 5 + i] = DEFAULT_IMAGE_TOKEN_INDEX
            input_ids[1, 20 + i] = DEFAULT_IMAGE_TOKEN_INDEX

        position_ids = torch.arange(seq_len).expand(batch_size, seq_len).cuda()
        labels = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        loss_mask = torch.ones((batch_size, seq_len)).cuda()

        # Mock vision encoder expects [total_tokens, input_dim].
        total_image_tokens = num_image_tokens * batch_size
        images = torch.randn(total_image_tokens, mock_input_dim).cuda()

        output, new_loss_mask = self.model.forward(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
        )

        # Sequence length is unchanged (masked_scatter, not expansion).
        assert output.shape == torch.Size((batch_size, seq_len))
        assert new_loss_mask.shape == torch.Size((batch_size, seq_len))

    @pytest.mark.internal
    def test_forward_with_images_no_labels(self):
        """Test forward pass with images but no labels returns logits."""
        self.model.cuda()

        batch_size = 1
        seq_len = self.seq_len
        num_image_tokens = 8
        mock_input_dim = 16

        mock_vision = MockVisionEncoder(mock_input_dim, self.language_hidden_size).cuda()
        self.model.visual = mock_vision
        self.model.add_encoder = True

        input_ids = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        for i in range(num_image_tokens):
            input_ids[0, 10 + i] = DEFAULT_IMAGE_TOKEN_INDEX

        position_ids = torch.arange(seq_len).expand(batch_size, seq_len).cuda()
        images = torch.randn(num_image_tokens, mock_input_dim).cuda()

        output, new_loss_mask = self.model.forward(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=None,
            loss_mask=None,
        )

        assert output.shape == torch.Size((batch_size, seq_len, self.vocab_size))
        assert new_loss_mask is None

    @pytest.mark.internal
    def test_forward_frozen_vision(self):
        """Test forward pass with frozen vision encoder runs under torch.no_grad."""
        self.model.cuda()
        self.model.freeze_vision = True

        batch_size = 1
        seq_len = self.seq_len
        num_image_tokens = 4
        mock_input_dim = 16

        mock_vision = MockVisionEncoder(mock_input_dim, self.language_hidden_size).cuda()
        self.model.visual = mock_vision
        self.model.add_encoder = True

        input_ids = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        for i in range(num_image_tokens):
            input_ids[0, i] = DEFAULT_IMAGE_TOKEN_INDEX

        position_ids = torch.arange(seq_len).expand(batch_size, seq_len).cuda()
        labels = torch.randint(0, 100, (batch_size, seq_len)).cuda()
        loss_mask = torch.ones((batch_size, seq_len)).cuda()
        images = torch.randn(num_image_tokens, mock_input_dim).cuda()

        output, new_loss_mask = self.model.forward(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
        )

        assert output.shape == torch.Size((batch_size, seq_len))

    @pytest.mark.internal
    def test_freeze(self):
        """Test freezing language and vision model parameters."""
        self.model.cuda()

        mock_vision = MockVisionEncoder(16, self.language_hidden_size).cuda()
        self.model.visual = mock_vision

        # Initially all params require grad.
        for param in self.model.language_model.parameters():
            assert param.requires_grad
        for param in self.model.visual.parameters():
            assert param.requires_grad

        self.model.freeze(freeze_language_model=True, freeze_vision_model=True)

        for param in self.model.language_model.parameters():
            assert not param.requires_grad
        for param in self.model.visual.parameters():
            assert not param.requires_grad

    @pytest.mark.internal
    def test_freeze_partial(self):
        """Test freezing only one component."""
        self.model.cuda()

        mock_vision = MockVisionEncoder(16, self.language_hidden_size).cuda()
        self.model.visual = mock_vision

        self.model.freeze(freeze_language_model=True, freeze_vision_model=False)

        for param in self.model.language_model.parameters():
            assert not param.requires_grad
        for param in self.model.visual.parameters():
            assert param.requires_grad

    @pytest.mark.internal
    def test_freeze_embedding(self):
        """Test freezing only the language model embedding layer."""
        self.model.cuda()

        for param in self.model.language_model.embedding.parameters():
            assert param.requires_grad

        self.model.freeze_embedding()

        for param in self.model.language_model.embedding.parameters():
            assert not param.requires_grad
        assert self.model.freeze_lm_embedding is True

    @pytest.mark.internal
    def test_save_load(self, tmp_path):
        """Test saving and loading model state dict."""
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)
        self.model.load_state_dict(torch.load(path))


class TestQwen3VLTransformerBlock:
    """Tests for the DeepStack mechanism in Qwen3VLTransformerBlock."""

    @pytest.mark.internal
    def test_deepstack_process(self):
        """Test that _deepstack_process adds visual features at masked positions."""
        hidden_size = 64
        seq_len = 32
        batch_size = 2
        num_visual_tokens = 5

        # Only need _deepstack_process; skip full TransformerBlock construction.
        block = Qwen3VLTransformerBlock.__new__(Qwen3VLTransformerBlock)

        hidden_states = torch.zeros(seq_len, batch_size, hidden_size)
        visual_pos_masks = torch.zeros(seq_len, batch_size, dtype=torch.bool)
        # Mark positions 5-9 in batch 0 as visual tokens.
        visual_pos_masks[5:10, 0] = True
        visual_embeds = torch.ones(num_visual_tokens, hidden_size)

        result = block._deepstack_process(hidden_states, visual_pos_masks, visual_embeds)

        assert result.shape == hidden_states.shape
        # Visual positions: 0 + 1 = 1
        assert torch.allclose(result[5:10, 0], torch.ones(num_visual_tokens, hidden_size))
        # Non-visual positions in batch 0 remain zero.
        assert torch.allclose(result[0:5, 0], torch.zeros(5, hidden_size))
        assert torch.allclose(result[10:, 0], torch.zeros(seq_len - 10, hidden_size))
        # Batch 1 entirely unchanged (no visual tokens marked).
        assert torch.allclose(result[:, 1], torch.zeros(seq_len, hidden_size))

    @pytest.mark.internal
    def test_deepstack_process_residual(self):
        """Test that _deepstack_process adds (not replaces) visual features."""
        hidden_size = 32
        seq_len = 16
        batch_size = 1

        block = Qwen3VLTransformerBlock.__new__(Qwen3VLTransformerBlock)

        hidden_states = torch.full((seq_len, batch_size, hidden_size), 2.0)
        visual_pos_masks = torch.zeros(seq_len, batch_size, dtype=torch.bool)
        visual_pos_masks[0:3, 0] = True
        visual_embeds = torch.full((3, hidden_size), 5.0)

        result = block._deepstack_process(hidden_states, visual_pos_masks, visual_embeds)

        # Visual positions: 2 + 5 = 7
        assert torch.allclose(result[0:3, 0], torch.full((3, hidden_size), 7.0))
        # Non-visual positions unchanged at 2.
        assert torch.allclose(result[3:, 0], torch.full((seq_len - 3, hidden_size), 2.0))

    @pytest.mark.internal
    def test_deepstack_process_none_mask(self):
        """Test _deepstack_process returns hidden_states unchanged when mask is None."""
        block = Qwen3VLTransformerBlock.__new__(Qwen3VLTransformerBlock)

        hidden_states = torch.randn(16, 2, 64)
        result = block._deepstack_process(hidden_states, None, torch.randn(5, 64))
        assert torch.equal(result, hidden_states)

    @pytest.mark.internal
    def test_deepstack_process_none_embeds(self):
        """Test _deepstack_process returns hidden_states unchanged when embeds is None."""
        block = Qwen3VLTransformerBlock.__new__(Qwen3VLTransformerBlock)

        hidden_states = torch.randn(16, 2, 64)
        mask = torch.zeros(16, 2, dtype=torch.bool)
        result = block._deepstack_process(hidden_states, mask, None)
        assert torch.equal(result, hidden_states)
