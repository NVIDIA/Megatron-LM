# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for per-layer RoPE base (rotary_base_per_layer) wiring in SelfAttention."""

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from tests.unit_tests.test_utilities import Utils


SEQ_LEN = 16
BATCH_SIZE = 2
HIDDEN_SIZE = 128
NUM_HEADS = 4
NUM_LAYERS = 2
ROTARY_BASE_L1 = 10000.0
ROTARY_BASE_L2 = 5000.0


def _make_config(rotary_base_per_layer=None) -> TransformerConfig:
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        rotary_base_per_layer=rotary_base_per_layer,
    )
    # _build_per_layer_rotary_pos_emb reads these attributes from config; they are
    # normally injected by GPTModel but must be set manually in unit tests.
    config.position_embedding_type = 'rope'
    config.rotary_scaling_factor = None  # seq_len_interpolation_factor
    config.rotary_percent = 1.0
    config.rope_scaling = False
    config.rope_scaling_factor = 8.0
    return config


def _make_attention(config: TransformerConfig, layer_number: int = 1) -> SelfAttention:
    submodules = get_gpt_layer_local_spec().submodules.self_attention.submodules
    return SelfAttention(config, submodules, layer_number=layer_number)


class TestRotaryBasePerLayerInit:
    """Verify that SelfAttention builds the correct per-layer RotaryEmbedding."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    def test_rotary_pos_emb_is_rope_instance(self):
        """rotary_pos_emb is a RotaryEmbedding when rotary_base_per_layer is set."""
        config = _make_config([ROTARY_BASE_L1, ROTARY_BASE_L2])
        attn = _make_attention(config, layer_number=1)
        assert isinstance(attn.rotary_pos_emb, RotaryEmbedding)

    def test_rotary_pos_emb_none_without_per_layer_config(self):
        """rotary_pos_emb stays None when rotary_base_per_layer is not set."""
        config = TransformerConfig(
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        attn = _make_attention(config, layer_number=1)
        assert attn.rotary_pos_emb is None

    def test_different_bases_produce_different_inv_freq(self):
        """Layers with distinct bases must have different inv_freq tensors."""
        config = _make_config([ROTARY_BASE_L1, ROTARY_BASE_L2])
        attn1 = _make_attention(config, layer_number=1)
        attn2 = _make_attention(config, layer_number=2)
        assert not torch.allclose(attn1.rotary_pos_emb.inv_freq, attn2.rotary_pos_emb.inv_freq)

    def test_same_base_produces_identical_inv_freq(self):
        """Layers sharing the same base must have identical inv_freq tensors."""
        config = _make_config([ROTARY_BASE_L1, ROTARY_BASE_L1])
        attn1 = _make_attention(config, layer_number=1)
        attn2 = _make_attention(config, layer_number=2)
        torch.testing.assert_close(attn1.rotary_pos_emb.inv_freq, attn2.rotary_pos_emb.inv_freq)
