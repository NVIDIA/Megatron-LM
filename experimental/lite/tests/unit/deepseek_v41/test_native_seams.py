# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.lite.model.deepseek_v41.lite.components.attention import DS41SelfAttention
from megatron.lite.model.deepseek_v41.lite.moe import SortedTopKPolicy


@pytest.mark.parametrize("halo", [False, True])
def test_query_projection_parent_without_per_head_rms_is_bitwise(halo):
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as native,
    )

    class Projection(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    class Frequencies:
        def get_rotary_seq_len(self, *args):
            return 4

        def __call__(self, length, packed_seq):
            return (torch.arange(length).float() / 7).view(-1, 1, 1, 1).expand(-1, 1, 1, 2)

    module = DS41SelfAttention.__new__(DS41SelfAttention)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        apply_rope_fusion=False,
        qk_pos_emb_head_dim=2,
        qk_head_dim=2,
        q_lora_rank=4,
        layernorm_epsilon=1e-6,
        rotary_interleaved=False,
        multi_latent_attention=True,
        mrope_section=None,
    )
    module.recompute_up_proj = module._dsv4_uses_yarn_rope = False
    module.pg_collection = SimpleNamespace(
        cp=SimpleNamespace(rank=lambda: int(halo), size=lambda: 2 if halo else 1)
    )
    module.num_attention_heads_per_partition, module.q_head_dim = 1, 4
    module.linear_q_down_proj = Projection(4, 4, bias=False)
    module.linear_q_up_proj = Projection(4, 4, bias=False)
    module.linear_kv_proj = Projection(4, 4, bias=False)
    module.q_layernorm = torch.nn.RMSNorm(4, eps=1e-6)
    module.kv_layernorm = torch.nn.RMSNorm(4, eps=1e-6)
    module.rotary_pos_emb = Frequencies()
    rows = 4 if halo else 8
    x = torch.arange(rows * 4).float().reshape(rows, 1, 4).div(11).requires_grad_()
    boundary = torch.arange(8).float().reshape(2, 1, 4).div(13).requires_grad_() if halo else None
    prefix = torch.tensor([0, 4, 8], dtype=torch.int32)
    params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q_padded=prefix,
        cu_seqlens_kv_padded=prefix,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    actual = module.get_query_key_value_tensors(
        x, packed_seq_params=params, boundary_hidden=boundary
    )
    with patch.object(native, "_q_rms_norm", lambda query, eps: query):
        reference = native.DSv4HybridSelfAttention.get_query_key_value_tensors(
            module, x, packed_seq_params=params, boundary_hidden=boundary
        )
    assert len(actual) == len(reference) == (6 if halo else 5)
    for ours, core in zip(actual, reference):
        assert torch.equal(ours, core)


@pytest.mark.parametrize("score", ["sigmoid", "sqrtsoftplus", "softmax"])
@pytest.mark.parametrize("bias", [False, True])
def test_sorted_router_grad_and_no_grad_contract(score, bias):
    logits = torch.tensor([[0.1, 1.3, -0.7, 2.1], [1.7, -0.3, 0.4, 0.9]], requires_grad=True)
    expert_bias = torch.tensor([0.0, 0.2, -0.1, 0.3]) if bias else None
    original = torch.topk

    def unordered(tensor, *args, **kwargs):
        values, indices = original(tensor, *args, **{**kwargs, "sorted": True})
        if not kwargs.get("sorted", True):
            return values.flip(-1), indices.flip(-1)
        return values, indices

    def route(grad, policy):
        with torch.set_grad_enabled(grad):
            return topk_routing_with_score_function(
                logits,
                3,
                score_function=score,
                expert_bias=expert_bias,
                fused=False,
                router_replay=policy,
                dense_output=True,
            )

    with patch("torch.topk", unordered):
        train = route(True, SortedTopKPolicy())
        evaluate = route(False, SortedTopKPolicy())
        assert torch.equal(train[0], evaluate[0])
        assert torch.equal(train[1], evaluate[1])
        unprotected = route(False, None)
        assert not torch.equal(train[1], unprotected[1])
    train[0].sum().backward()
    assert torch.isfinite(logits.grad).all()
