# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dit_layer import DiTTransformerLayer


def test_dit_layer_identity_and_grad():
    hidden_size = 32
    cond_dim = 16

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    layer = DiTTransformerLayer(
        config=config,
        submodules=None,
        cond_dim=cond_dim,
    )

    x = torch.randn(2, 8, hidden_size)
    cond = torch.randn(2, cond_dim)
    attn_mask = torch.ones(2, 1, 8, 8, dtype=torch.bool)

    # Identity check (zero-init gates)
    with torch.no_grad():
        out = layer(x, attn_mask, cond)

    torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    # Gradient flow check
    cond.requires_grad_(True)
    out = layer(x, attn_mask, cond)
    loss = out.sum()
    loss.backward()

    assert cond.grad is not None
    assert torch.norm(cond.grad) > 0
