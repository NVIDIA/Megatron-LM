# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for --freeze-base-for-mtp feature in model_builder."""

import torch
from packaging.version import Version

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_layer_specs,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.post_training.model_builder import _freeze_base_for_mtp
from tests.unit_tests.test_utilities import Utils


class TestFreezeBaseForMTP:
    """Test that _freeze_base_for_mtp correctly freezes base and keeps MTP trainable."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        self.config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            mtp_num_layers=1,
        )

        # Build model with modelopt spec (base layers) + MTP block spec (standard layers).
        modelopt_spec = get_gpt_modelopt_spec(self.config)
        decoder_layer_specs = get_gpt_decoder_layer_specs(
            self.config, use_transformer_engine=True,
        )
        mtp_block_spec = get_gpt_mtp_block_spec(
            self.config, decoder_layer_specs[-1], use_transformer_engine=True,
        )

        self.model = GPTModel(
            config=self.config,
            transformer_layer_spec=modelopt_spec,
            mtp_block_spec=mtp_block_spec,
            vocab_size=100,
            max_sequence_length=8,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_model_has_mtp(self):
        """Verify model was built with MTP layers."""
        assert hasattr(self.model, 'mtp'), "Model should have MTP attribute"
        mtp_params = [n for n, _ in self.model.named_parameters() if 'mtp.layers.' in n]
        assert len(mtp_params) > 0, "Model should have MTP parameters"

    def test_freeze_only_keeps_mtp_trainable(self):
        """After freezing, only mtp.layers.* params should have requires_grad=True."""
        _freeze_base_for_mtp(self.model)

        trainable_params = []
        frozen_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)

        # All trainable params must be MTP params.
        for name in trainable_params:
            assert 'mtp.layers.' in name, (
                f"Non-MTP param '{name}' should be frozen but has requires_grad=True"
            )

        # All MTP params must be trainable.
        for name, param in self.model.named_parameters():
            if 'mtp.layers.' in name:
                assert param.requires_grad, (
                    f"MTP param '{name}' should be trainable but has requires_grad=False"
                )

        # Sanity: we should have both frozen and trainable params.
        assert len(frozen_params) > 0, "Should have frozen base params"
        assert len(trainable_params) > 0, "Should have trainable MTP params"

    def test_base_params_are_frozen(self):
        """Embedding, decoder, and output_layer params should all be frozen."""
        _freeze_base_for_mtp(self.model)

        for name, param in self.model.named_parameters():
            if 'mtp.layers.' not in name:
                assert not param.requires_grad, (
                    f"Base param '{name}' should be frozen"
                )

    def test_freeze_is_idempotent(self):
        """Calling freeze twice should produce the same result."""
        _freeze_base_for_mtp(self.model)
        trainable_1 = {n for n, p in self.model.named_parameters() if p.requires_grad}

        _freeze_base_for_mtp(self.model)
        trainable_2 = {n for n, p in self.model.named_parameters() if p.requires_grad}

        assert trainable_1 == trainable_2
