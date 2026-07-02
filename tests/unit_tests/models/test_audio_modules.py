# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.audio import AudioProjection, PackedAudioEmbeddings
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestAudioProjection:
    def test_stack_features_mask(self):
        projector = object.__new__(AudioProjection)
        projector.input_size = 24
        projector.stack_factor = 2

        hidden_states = torch.randn(2, 5, 24)
        attention_mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]], dtype=torch.bool
        )

        stacked_states, output_mask = projector._stack_features(hidden_states, attention_mask)

        assert stacked_states.shape == torch.Size([2, 3, 48])
        assert output_mask.shape == torch.Size([2, 3])
        assert output_mask[0].tolist() == [True, True, True]
        assert output_mask[1].tolist() == [True, True, False]

    def test_forward_packed_projects_valid_tokens_only(self):
        class FakeProjector:
            def __call__(self, hidden_states):
                return torch.cat([hidden_states, hidden_states + 100.0], dim=-1)

        projector = object.__new__(AudioProjection)
        projector.input_size = 3
        projector.stack_factor = 1
        projector.projector = FakeProjector()

        embeddings = torch.arange(5 * 3, dtype=torch.float32).view(5, 3)
        lengths = torch.tensor([2, 3], dtype=torch.int32)
        projected = AudioProjection.forward_packed(
            projector, PackedAudioEmbeddings(embeddings=embeddings, lengths=lengths)
        )

        assert projected.lengths.tolist() == [2, 3]
        expected = torch.cat([embeddings, embeddings + 100.0], dim=-1)
        torch.testing.assert_close(projected.embeddings, expected)

    def test_forward_packed_rejects_stack_factor_greater_than_one(self):
        projector = object.__new__(AudioProjection)
        projector.input_size = 3
        projector.stack_factor = 2

        with pytest.raises(NotImplementedError, match="stack_factor == 1"):
            AudioProjection.forward_packed(
                projector,
                PackedAudioEmbeddings(
                    embeddings=torch.zeros(1, 3), lengths=torch.tensor([1], dtype=torch.int32)
                ),
            )

    def test_packed_audio_pad_to_lengths_adds_per_segment_zero_rows(self):
        packed = PackedAudioEmbeddings(
            embeddings=torch.arange(5 * 3, dtype=torch.float32).view(5, 3),
            lengths=torch.tensor([2, 3], dtype=torch.int32),
        )

        padded = packed.pad_to_lengths(torch.tensor([4, 3], dtype=torch.int32))

        assert padded.lengths.tolist() == [4, 3]
        assert padded.embeddings.shape == torch.Size([7, 3])
        torch.testing.assert_close(padded.embeddings[:2], packed.embeddings[:2])
        torch.testing.assert_close(padded.embeddings[2:4], torch.zeros(2, 3))
        torch.testing.assert_close(padded.embeddings[4:], packed.embeddings[2:])


class TestAudioProjectionWithModelParallel:
    def setup_method(self, method):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for TP projector test")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            ffn_hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        self.projector = AudioProjection(
            config=config,
            submodules=get_mlp_module_spec(use_te=False).submodules,
            projector_type="affine",
            input_size=24,
            stack_factor=2,
        )

    def teardown_method(self, method):
        if torch.cuda.is_available():
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TP projector test")
    def test_forward_shapes_and_mask(self):
        hidden_states = torch.randn(2, 5, 24)
        attention_mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]], dtype=torch.bool
        )

        projected_states, output_mask = self.projector(hidden_states, attention_mask)

        assert projected_states.shape == torch.Size([3, 2, 32])
        assert output_mask.shape == torch.Size([2, 3])
        assert output_mask[0].tolist() == [True, True, True]
        assert output_mask[1].tolist() == [True, True, False]
