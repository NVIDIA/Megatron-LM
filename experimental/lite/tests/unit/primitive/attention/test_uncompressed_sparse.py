# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Uncompressed sparse attention composition and shared state, without model policy."""
import ast
from types import SimpleNamespace

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.lite.primitive.modules.attention.ratio_one import RatioOneAttention
from megatron.lite.primitive.parallel.source_state import KVSourceState


def test_ratio_one_native_unfused_attention_backward_and_source_reuse():
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.manual_seed(15)
    config = SimpleNamespace(
        num_attention_heads=1,
        qk_pos_emb_head_dim=2,
        csa_window_size=2,
        v_head_dim=4,
        dsa_indexer_topk=3,
        apply_rope_fusion=False,
        rotary_interleaved=False,
        multi_latent_attention=True,
        mrope_section=None,
    )
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    state = KVSourceState()
    rope = lambda length, packed_seq: torch.zeros(length, 1, 1, 2)
    owner = RatioOneAttention(
        config,
        group=group,
        rotary_pos_emb=rope,
        layer=3,
        index_owner=3,
        candidate_owner=3,
        candidate_blocks=2,
        block_size=2,
        state=state,
    )
    reader = RatioOneAttention(
        config,
        group=group,
        rotary_pos_emb=rope,
        layer=4,
        index_owner=3,
        candidate_owner=3,
        candidate_blocks=2,
        block_size=2,
        state=state,
    )
    prefix = torch.tensor([0, 5, 6, 12], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cp_partition_mode="contiguous",
        cu_seqlens_q=prefix,
        cu_seqlens_kv=prefix,
        max_seqlen_q=6,
        max_seqlen_kv=6,
    )
    q = torch.randn(12, 1, 4, requires_grad=True)
    window, main = [torch.randn(12, 4, requires_grad=True) for _ in range(2)]
    iq, ik, weights = torch.randn(12, 1, 4), torch.randn(12, 4), torch.randn(12, 1)
    with state.microbatch():
        output = owner(q, window, main, iq, ik, weights, params)
        other = reader(q, window, main, -iq, -ik, -weights, params)
        assert torch.equal(output, other)
        (output.square().sum() + other.square().sum()).backward()
    assert all(torch.isfinite(t.grad).all() for t in (q, window, main))
    assert (
        torch.isfinite(owner.attn_sink.grad).all() and torch.isfinite(reader.attn_sink.grad).all()
    )


def test_ratio_one_only_composes_core_functions():
    import inspect

    from megatron.core.transformer.experimental_attention_variant import csa, dsa
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
    from megatron.lite.primitive.modules.attention import ratio_one

    tree = ast.parse(inspect.getsource(ratio_one))
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "dist"
        for n in ast.walk(tree)
    )

    def bodies(module):
        return {
            ast.dump(ast.Module(body=n.body, type_ignores=[]), include_attributes=False)
            for n in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    assert not (bodies(ratio_one) & (bodies(csa) | bodies(dsa) | bodies(cp_utils)))
