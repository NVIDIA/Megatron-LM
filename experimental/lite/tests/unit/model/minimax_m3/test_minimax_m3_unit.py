# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for MiniMax-M3: config mapping, HF weight map, label packing, loss helpers."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
from megatron.lite.model.minimax_m3.lite.checkpoint import (
    MiniMaxM3WeightSpec,
    _pack_grouped_qkv,
    _unpack_grouped_qkv,
)
from megatron.lite.model.minimax_m3.lite.protocol import _shift_within_documents
from megatron.lite.model.registry import resolve_model_type_from_hf
from megatron.lite.primitive.modules.attention.msa import MsaV1Numerics, msa_v1_rank_kl_coeff


def _hf_config(num_layers: int = 6) -> dict:
    return {
        "model_type": "minimax_m3_vl",
        "text_config": {
            "hidden_size": 6144,
            "intermediate_size": 3072,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 200064,
            "rms_norm_eps": 1e-6,
            "use_gemma_norm": True,
            "rope_theta": 10000000,
            "rotary_dim": 64,
            "hidden_act": "swigluoai",
            "use_qk_norm": True,
            "qk_norm_type": "per_head",
            "dense_intermediate_size": 12288,
            "shared_intermediate_size": 3072,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "n_shared_experts": 1,
            "scoring_func": "sigmoid",
            "routed_scaling_factor": 2.0,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "moe_layer_freq": [0, 0, 0] + [1] * (num_layers - 3),
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 4,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128,
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_attention_freq": [1] * num_layers,
            },
        },
    }


def test_registry_resolves_hf_model_type():
    assert resolve_model_type_from_hf({"model_type": "minimax_m3_vl"}) == "minimax_m3"


def test_layer_selection_takes_per_layer_fields_at_the_selected_indices():
    config = MiniMaxM3Config._from_hf_dict(_hf_config(), layer_indices=[0, 1, 3, 4])
    assert config.num_hidden_layers == 4
    assert config.moe_layer_freq == [0, 0, 1, 1]
    assert [config.hf_layer_index(i) for i in range(4)] == [0, 1, 3, 4]
    assert config.n_routed_experts == 128 and config.moe_intermediate_size == 3072
    assert config.rope_theta == 10000000


def test_config_rejects_unsupported_architectures():
    hf = _hf_config()
    hf["text_config"]["hidden_act"] = "silu"
    with pytest.raises(ValueError, match="swigluoai"):
        MiniMaxM3Config._from_hf_dict(hf)
    hf = _hf_config()
    hf["text_config"]["sparse_attention_config"]["sparse_topk_blocks"] = 16
    hf["text_config"]["sparse_attention_config"]["sparse_local_block"] = 2
    with pytest.raises(ValueError, match="sparse_local_block"):
        MiniMaxM3Config._from_hf_dict(hf)


def test_weight_map_reads_selected_hf_layers_and_packs_qkv_per_kv_group():
    config = MiniMaxM3Config._from_hf_dict(_hf_config(), layer_indices=[0, 1, 3, 4])
    wm = MiniMaxM3WeightSpec(config).weight_map()
    assert wm["layers.2.mlp.router.gate.weight"] == [
        "language_model.model.layers.3.block_sparse_moe.gate.weight"
    ]
    assert wm["layers.0.mlp.fc1.weight"] == [
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
    ]
    assert wm["layers.3.mlp.experts._fc1_weight_127"] == [
        "language_model.model.layers.4.block_sparse_moe.experts.127.w1.weight",
        "language_model.model.layers.4.block_sparse_moe.experts.127.w3.weight",
    ]
    assert wm["head.col.linear.weight"] == ["language_model.lm_head.weight"]

    small = MiniMaxM3Config._from_hf_dict(_hf_config())
    small.num_attention_heads, small.num_key_value_heads, small.head_dim = 8, 2, 2
    q = torch.arange(16.0).view(16, 1)
    k = torch.arange(100.0, 104.0).view(4, 1)
    v = torch.arange(200.0, 204.0).view(4, 1)
    packed = _pack_grouped_qkv(q, k, v, small)
    assert packed.view(-1).tolist() == (
        list(range(0, 8)) + [100, 101, 200, 201] + list(range(8, 16)) + [102, 103, 202, 203]
    )
    for got, want in zip(_unpack_grouped_qkv(packed, small), (q, k, v)):
        assert torch.equal(got, want)


def test_labels_shift_within_documents_and_mask_document_ends():
    labels = torch.tensor([10, 11, 12, 20, 21])
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    shifted, shifted_mask = _shift_within_documents(labels, mask, cu)
    assert shifted.tolist() == [11, 12, -100, -100, -100]
    assert shifted_mask.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_kl_coefficient_scaling_per_rank():
    numerics = MsaV1Numerics(kl_loss_coeff=0.5, kl_temperature=3.0)
    coeff = msa_v1_rank_kl_coeff(
        numerics, tp_size=1, cp_size=2, total_tokens=8192, per_token_loss=True
    )
    assert coeff == 0.5 * 3.0**2 / 1 * (8192 / 2) * 2
    assert (
        msa_v1_rank_kl_coeff(
            numerics, tp_size=1, cp_size=2, total_tokens=8192, per_token_loss=False
        )
        == 4.5 * 2
    )


def test_loss_terms_are_off_by_default():
    config = MiniMaxM3Config._from_hf_dict(_hf_config())
    assert not (config.sparse_kl_loss_coeff or config.aux_loss_alpha)


def test_runtime_key_reads_the_minimax_config_through_a_ddp_wrapper(monkeypatch):
    """Under dist_opt the forward step gets the DDP-wrapped chunk, whose own ``config`` is a
    Megatron-Core TransformerConfig; the protocol must read the wrapped MiniMax model's."""
    from megatron.lite.model.minimax_m3.lite import protocol
    from megatron.lite.runtime.contracts.data import PackedBatch

    seen = {}

    def fake_key(cu_seqlens, group, **kwargs):
        seen.update(kwargs)
        return "key"

    monkeypatch.setattr(protocol, "build_msa_runtime_key", fake_key)
    inner = torch.nn.Module()
    inner.config = MiniMaxM3Config._from_hf_dict(_hf_config())
    inner.impl_options = {"per_token_loss": True, "deterministic": True, "msa_chunk_size": None}
    wrapper = torch.nn.Module()
    wrapper.config = object()
    wrapper.module = inner
    batch = PackedBatch(
        input_ids=torch.zeros(8, dtype=torch.long), labels=None, seq_lens=torch.tensor([5, 3])
    )
    assert protocol.get_msa_runtime_key(wrapper, batch) == "key"
    assert seen["numerics"] == protocol.msa_numerics(inner.config)
    assert seen["per_token_loss"] is True
    assert seen["dense_heads"] is None

    hf = _hf_config()
    hf["text_config"]["sparse_attention_config"]["sparse_attention_freq"] = [0] + [1] * 5
    inner.config = MiniMaxM3Config._from_hf_dict(hf)
    batch = PackedBatch(
        input_ids=torch.zeros(8, dtype=torch.long), labels=None, seq_lens=torch.tensor([8])
    )
    protocol.get_msa_runtime_key(wrapper, batch)
    assert seen["dense_heads"] == (64, 4)


def test_full_attention_layers_from_either_hf_field_and_their_weight_map():
    legacy = _hf_config(num_layers=4)
    legacy["text_config"]["sparse_attention_config"]["sparse_attention_freq"] = [0, 1, 1, 1]
    canonical = _hf_config(num_layers=4)
    del canonical["text_config"]["sparse_attention_config"]["sparse_attention_freq"]
    canonical["text_config"]["layer_types"] = ["full_attention"] + ["minimax_m3_sparse"] * 3
    for hf in (legacy, canonical):
        config = MiniMaxM3Config._from_hf_dict(hf)
        assert config.sparse_attention_freq == [0, 1, 1, 1]
        assert config.has_full_attention_layers
    wm = MiniMaxM3WeightSpec(config).weight_map()
    assert "layers.0.self_attention.index_q_proj.weight" not in wm
    assert wm["layers.0.self_attention.k_norm.weight"] == [
        "language_model.model.layers.0.self_attn.k_norm.weight"
    ]
    assert wm["layers.1.self_attention.index_k_norm.weight"] == [
        "language_model.model.layers.1.self_attn.index_k_norm.weight"
    ]
    assert not MiniMaxM3Config._from_hf_dict(_hf_config()).has_full_attention_layers


def test_layer_types_must_agree_with_legacy_fields_and_be_known():
    hf = _hf_config(num_layers=4)
    hf["text_config"]["layer_types"] = ["minimax_m3_sparse"] * 4
    hf["text_config"]["sparse_attention_config"]["sparse_attention_freq"] = [0, 1, 1, 1]
    with pytest.raises(ValueError, match="disagrees"):
        MiniMaxM3Config._from_hf_dict(hf)
    hf = _hf_config(num_layers=4)
    hf["text_config"]["mlp_layer_types"] = ["dense", "dense", "sparse", "shared"]
    with pytest.raises(ValueError, match="unsupported mlp_layer_types"):
        MiniMaxM3Config._from_hf_dict(hf)
