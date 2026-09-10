# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for rotary embedding sharing across layers (``maybe_share_rotary_pos_emb``).

Two layers of coverage:
  * a fast, device-free check that different rotary configs are not shared, and
  * a small two-layer multi-latent-attention proxy that runs one forward/backward step
    with sharing off and on and asserts the loss and gradient norm are unchanged --
    the feature must be numerically transparent (it only removes duplicate buffers).
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.embeddings import maybe_share_rotary_pos_emb
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 12


# ----------------------------------------------------------------------------------------------
# Helper key-distinction guard (fast, no distributed init required)
# ----------------------------------------------------------------------------------------------
def test_enabled_distinguishes_keys():
    """Different rotary configurations (keys) are not shared with each other."""
    config = SimpleNamespace(share_rotary_pos_emb=True)
    rope, yarn = object(), object()

    assert maybe_share_rotary_pos_emb(config, ("rope", 10000), rope) is rope
    assert maybe_share_rotary_pos_emb(config, ("yarn", 10000), yarn) is yarn


# ----------------------------------------------------------------------------------------------
# Numerical-transparency proxy (two MLA layers, one step, sharing off vs on)
# ----------------------------------------------------------------------------------------------
def _build_two_mla_layers(rope_type: str, share: bool):
    """Build a two-layer MLA stack sharing one config so the sharing spans both layers.

    Dropout is disabled so the forward pass is deterministic, which lets the sharing-off and
    sharing-on runs be compared bit-for-bit.
    """
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=128,
        v_head_dim=128,
        qk_pos_emb_head_dim=64,
        rope_type=rope_type,
        rotary_base=10000,
        original_max_position_embeddings=32,
        share_rotary_pos_emb=share,
    )
    layers = []
    for layer_number in (1, 2):
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            multi_latent_attention=True
        ).self_attention.submodules
        layers.append(
            MLASelfAttention(
                config, submodules, layer_number=layer_number, attn_mask_type=AttnMaskType.causal
            )
        )
    return layers


def _loss_and_grad_norm(layers, hidden_states, attention_mask):
    """Run one forward/backward over the stacked layers and return (loss, total grad norm)."""
    hidden = hidden_states
    for layer in layers:
        layer.cuda()
        output, bias = layer(hidden, attention_mask)
        hidden = output + bias if bias is not None else output
    loss = hidden.float().pow(2).mean()
    loss.backward()
    grad_sq = torch.zeros((), dtype=torch.float64, device=loss.device)
    for layer in layers:
        for param in layer.parameters():
            if param.grad is not None:
                grad_sq = grad_sq + param.grad.detach().double().pow(2).sum()
    return loss.detach(), grad_sq.sqrt()


@pytest.mark.parametrize("rope_type", ("rope", "yarn"))
class TestSharedRotaryNumericalTransparency:
    """Sharing the rotary module across layers must not change training numerics."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def test_loss_and_grad_norm_are_unchanged(self, rope_type):
        if not is_te_min_version("1.10.0"):
            pytest.skip("MLA attention GPU forward requires TransformerEngine >= 1.10.0")

        sequence_length, micro_batch_size = 32, 2
        torch.manual_seed(0)
        hidden_states = torch.randn(sequence_length, micro_batch_size, HIDDEN_SIZE).cuda()
        causal = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1
        )
        attention_mask = causal.view(1, 1, sequence_length, sequence_length).cuda()

        off = _build_two_mla_layers(rope_type, share=False)
        on = _build_two_mla_layers(rope_type, share=True)
        # Force bit-identical weights so any output difference can only come from rotary sharing.
        for layer_on, layer_off in zip(on, off):
            layer_on.load_state_dict(layer_off.state_dict(), strict=False)

        # The flag must actually change the wiring: off keeps two modules, on collapses to one.
        assert off[0].rotary_pos_emb is not off[1].rotary_pos_emb
        assert on[0].rotary_pos_emb is on[1].rotary_pos_emb

        loss_off, grad_norm_off = _loss_and_grad_norm(off, hidden_states, attention_mask)
        loss_on, grad_norm_on = _loss_and_grad_norm(on, hidden_states, attention_mask)

        # Forward is bit-identical; backward matches within fp32 kernel (atomic) tolerance.
        torch.testing.assert_close(loss_on, loss_off, rtol=0, atol=0)
        torch.testing.assert_close(grad_norm_on, grad_norm_off, rtol=1e-4, atol=1e-5)
