# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Ratio-one layers: release RoPE and source ownership (40L release config)."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from torch import nn

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.components import attention
from megatron.lite.model.deepseek_v41.topology import layer_sources


def test_ratio_one_rotates_with_compress_theta_and_yarn(monkeypatch):
    built = []

    class Yarn(nn.Module):
        def __init__(self, dim, **kwargs):
            super().__init__()
            built.append(dict(dim=dim, **kwargs))

    monkeypatch.setattr(attention, "YarnRotaryEmbedding", Yarn)
    layer = attention.DS41SelfAttention.__new__(attention.DS41SelfAttention)
    nn.Module.__init__(layer)
    keys = ("rotary_scaling_factor", "original_max_position_embeddings", "beta_fast")
    layer.config = SimpleNamespace(
        qk_pos_emb_head_dim=64,
        csa_compress_rotary_base=160000,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=1,
        **dict(zip(keys, (16, 65536, 32))),
    )
    layer.pg_collection = SimpleNamespace(cp="cp")
    old = layer.rotary_pos_emb = nn.Identity()
    layer.core_attention = nn.Module()
    layer.core_attention.rotary = old
    layer.core_attention.native = nn.Module()
    layer.core_attention.native.rotary_pos_emb = old
    object.__setattr__(layer.core_attention.native, "view", old)  # unregistered holder
    layer._use_compressed_rope()
    (kwargs,) = built
    assert kwargs["rotary_base"] == 160000 and kwargs["scaling_factor"] == 16
    assert kwargs["original_max_position_embeddings"] == 65536 and kwargs["cp_group"] == "cp"
    holders = (
        layer.rotary_pos_emb,
        layer.core_attention.rotary,
        layer.core_attention.native.rotary_pos_emb,
        layer.core_attention.native.view,
    )
    assert all(isinstance(h, Yarn) for h in holders) and len({id(h) for h in holders}) == 1
    assert layer._dsv4_uses_yarn_rope and layer._dsv4_rope_base == 160000


def test_release_sources_and_candidate_publisher():
    release = DeepseekV41Config()  # 40 layers: kv (2,8,14,20), index (...,36), candidate 20
    sources = layer_sources(release)
    assert [(s.kv, s.index, s.candidate) for s in (sources[19], sources[20], sources[23])] == [
        (14, 14, False),
        (20, 20, True),
        (20, 20, True),
    ]
    assert (sources[24].index, sources[39].kv, sources[39].index) == (24, 20, 36)
    for candidate in (21, 14):  # a reader of index owner 20; a ratio-two index owner
        with pytest.raises(ValueError, match="candidate publisher"):
            layer_sources(replace(release, candidate_source_layer_id=candidate))
