# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay SelfAttention with the Q/K RMSNorm fused into TE's single-QKV RoPE kernel.

With ``qk_layernorm`` and ``fused_single_qkv_rope`` the attention layer hands the norm weights
to Transformer Engine's fused QKV RoPE, whose Triton kernels normalize and rotate in one pass
and reduce the weight gradients in a fixed order. Forward, input gradient and the norm weight
gradients must therefore replay bit-exactly, also under side-stream contention.
"""

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_module_replays_bit_exact, seeded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Transformer Engine")
@pytest.mark.parametrize("contention", [False, True])
def test_fused_qkv_norm_rope_attention_replays(contention, monkeypatch):
    """Replay forward/backward of a QK-norm attention layer on the fused single-QKV path."""
    from megatron.core.extensions.transformer_engine import HAVE_TE

    if not HAVE_TE:
        pytest.skip("requires Transformer Engine")

    from megatron.core.transformer.attention import HAVE_FUSED_QKV_NORM_ROPE

    if not HAVE_FUSED_QKV_NORM_ROPE:
        pytest.skip("the installed Transformer Engine cannot fuse the QK norm into RoPE")

    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(123)
        seeded()
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        config = TransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=128,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            add_bias_linear=False,
            normalization="RMSNorm",
            qk_layernorm=True,
            fused_single_qkv_rope=True,
            deterministic_mode=True,
        )
        module = SelfAttention(
            config,
            get_gpt_layer_with_transformer_engine_submodules(
                qk_layernorm=True
            ).self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()
        assert module.qk_norm_fusable_into_qkv_rope()

        seq_length, batch_size = 512, 2
        hidden = torch.randn(
            seq_length, batch_size, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        rotary = RotaryEmbedding(kv_channels=config.kv_channels, rotary_percent=1.0)(seq_length)
        inputs = dict(
            hidden_states=hidden.requires_grad_(True), attention_mask=None, rotary_pos_emb=rotary
        )
        outputs, grads = assert_module_replays_bit_exact(
            module,
            inputs,
            replays=3,
            contention=contention,
            what="SelfAttention with the QK norm fused into single-QKV RoPE",
        )
        assert "in.hidden_states" in grads
        assert any("q_layernorm" in name for name in grads)
        assert any("k_layernorm" in name for name in grads)
        for tensors in (outputs, grads):
            assert all(torch.isfinite(t).all() for t in tensors.values())
    finally:
        Utils.destroy_model_parallel()
