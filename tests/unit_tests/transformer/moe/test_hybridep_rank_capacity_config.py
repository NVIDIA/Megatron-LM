# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

import megatron.core.transformer.moe.fused_a2a as fused_a2a
from megatron.core.transformer.transformer_config import TransformerConfig


def test_hybridep_rank_capacity_does_not_require_te_op_fuser():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_expert_rank_capacity_factor=1.10,
        use_transformer_engine_op_fuser=False,
    )

    assert config.moe_expert_rank_capacity_factor == 1.10
    assert not config.use_transformer_engine_op_fuser


def test_rank_capacity_still_requires_hybridep_backend():
    with pytest.raises(ValueError, match="moe_flex_dispatcher_backend"):
        TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="deepep",
            moe_expert_rank_capacity_factor=1.10,
            use_transformer_engine_op_fuser=False,
        )


def test_hybridep_combine_backward_trims_bounded_workspace_gradient(monkeypatch):
    class FakeHybridEPBuffer:
        def combine_with_unpermute(self, *, hidden, handle, pad_multiple, **kwargs):
            return hidden * 2, None

        def dispatch_with_permute(self, *, hidden, handle, num_permuted_tokens, **kwargs):
            assert num_permuted_tokens == 5
            grad = hidden.new_ones((num_permuted_tokens, hidden.shape[1]))
            return grad, None, None, None, None

    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", FakeHybridEPBuffer())

    x = torch.randn(3, 4, requires_grad=True)
    y = fused_a2a.HybridEPCombine.apply(x, (), 5, None, False, 3)

    y.sum().backward()

    assert x.grad.shape == x.shape
    torch.testing.assert_close(x.grad, torch.ones_like(x))
