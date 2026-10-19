# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Source-selection and routing contracts against the installed Core version."""
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    CSAIndexer,
)
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.moe.router import TopKRouter
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.components.attention import DS41SelfAttention
from megatron.lite.model.deepseek_v41.lite.components.config import (
    attach_ratio_two_sources,
    core_config,
)
from megatron.lite.model.deepseek_v41.lite.moe import SortedTopKPolicy, bind_sorted_selection
from megatron.lite.runtime.contracts import ParallelConfig


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


@pytest.mark.parametrize("unsupported", ["fusion", "quantile", "replay"])
def test_sorted_router_rejects_bypass(unsupported):
    router = SimpleNamespace(
        is_hash_layer=False,
        config=SimpleNamespace(moe_router_fusion=unsupported == "fusion"),
        use_quantile_balancing=unsupported == "quantile",
        router_replay=object() if unsupported == "replay" else None,
    )
    with pytest.raises(ValueError):
        bind_sorted_selection(router)


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
    signature = lambda cls: [
        (p.name, p.kind, p.default)
        for p in inspect.signature(cls.get_query_key_value_tensors).parameters.values()
    ]
    assert signature(DS41SelfAttention) == signature(native.DSv4HybridSelfAttention)
    assert not torch.allclose(actual[0].square().mean(-1), torch.ones_like(actual[0][..., 0]))
    assert actual[4].data_ptr() == x.data_ptr() and actual[4].shape == x.squeeze(1).shape
    assert torch.equal(actual[1], actual[2])
    (actual[0].square().mean() + actual[1].square().mean()).backward()
    assert torch.isfinite(x.grad).all()
    assert len(actual) == len(reference) == (6 if halo else 5)
    for ours, core in zip(actual, reference):
        assert torch.equal(ours, core)


def test_ratio_two_attachment_runs_native_csa_and_indexer_forward():
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnMaskType

    class CPUIndexSources(CSAIndexer):
        def forward_before_topk(self, x, qr, packed_seq_params=None):
            return (
                torch.ones(12, 1, 1, 4),
                torch.arange(24).float().reshape(6, 1, 4) / 8,
                torch.ones(12, 1, 1),
                torch.tensor([0, 3, 4, 6], dtype=torch.int32),
            )

    class CPUCompression(torch.nn.Module):
        compress_ratio = 2

        def forward(self, x, packed_seq_params):
            return x[::2], torch.tensor([0, 3, 4, 6], dtype=torch.int32)

    compressor = CPUCompression()
    config = core_config(DeepseekV41Config(num_hidden_layers=4), ParallelConfig())
    config.num_attention_heads, config.v_head_dim, config.qk_pos_emb_head_dim = 1, 4, 2
    config.csa_window_size = 2
    attention = CompressedSparseAttention(
        config,
        CompressedSparseAttentionSubmodules(),
        1,
        AttnMaskType.causal,
        "self",
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 1, rank=lambda: 0)),
        rotary_pos_emb=lambda length, packed_seq: torch.zeros(length, 1, 1, 2),
        compress_ratio=2,
    ).eval()
    assert attention.indexer is None and attention.compressor is None
    indexer = CPUIndexSources.__new__(CPUIndexSources)
    torch.nn.Module.__init__(indexer)
    indexer.compress_ratio, indexer.index_topk = 2, 1
    attach_ratio_two_sources(attention, compressor=compressor, indexer=indexer)
    assert attention.indexer is indexer and attention.compressor is compressor
    calls = []
    handle = indexer.register_forward_hook(lambda module, args, output: calls.append(output[1]))
    x = (torch.arange(48).reshape(12, 1, 4).float() / 32).to(torch.bfloat16)
    prefix = torch.tensor([0, 6, 8, 12], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=prefix, cu_seqlens_kv=prefix, max_seqlen_q=6, max_seqlen_kv=6
    )
    query, key = x.squeeze(1).unsqueeze(1), x.unsqueeze(2)
    with torch.no_grad():
        selected = attention(query, key, key, None, x=x, qr=x, packed_seq_params=params)
        assert len(calls) == 1 and calls[0].shape == (12, 1)
        attention.indexer = None
        all_compressed = attention(query, key, key, None, x=x, qr=x, packed_seq_params=params)
    handle.remove()
    assert selected.shape == all_compressed.shape == (12, 1, 4)
    assert torch.isfinite(selected).all()
    assert not torch.equal(selected, all_compressed)
