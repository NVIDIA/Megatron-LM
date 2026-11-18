# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage for fused MLA's optimizer checkpoint key reconciliation."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.transformer import multi_latent_attention as mla_module
from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention, MLASelfAttention


@pytest.mark.parametrize("wrapper_prefix", ["", "module.", "module.language_model."])
def test_optimizer_materializes_fused_mla_weight_from_unfused_keys(wrapper_prefix):
    layer = FusedMLASelfAttention.__new__(FusedMLASelfAttention)
    torch.nn.Module.__init__(layer)
    model = torch.nn.Module()
    model.add_module("attention", layer)
    prefix = wrapper_prefix + "attention."
    q = torch.arange(6, dtype=torch.float32).view(2, 3)
    kv = torch.arange(9, dtype=torch.float32).view(3, 3) + 20
    state = {
        prefix + "linear_q_down_proj.weight": q,
        prefix + "linear_kv_down_proj.weight": kv,
        prefix + "linear_q_down_proj.bias": torch.zeros(2),
        prefix + "linear_kv_down_proj.bias": torch.zeros(3),
    }
    DistributedOptimizer._synthesize_state_dict_params_for_model(state, model)
    key = prefix + "linear_qkv_down_proj.weight"
    assert set(state) == {key}
    torch.testing.assert_close(state[key][:2], q)
    torch.testing.assert_close(state[key][2:], kv)
    assert state[key].shape == (5, 3)


def test_fused_mla_checkpoint_keeps_layernorm_and_splits_only_projection(monkeypatch):
    layer = FusedMLASelfAttention.__new__(FusedMLASelfAttention)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(q_lora_rank=2, kv_lora_rank=2, qk_pos_emb_head_dim=1)
    layer.tp_group = None
    layer.linear_qkv_down_proj = torch.nn.Linear(3, 5, bias=False)
    prefix = "decoder.layers.0.self_attention."
    fused_prefix = prefix + "linear_qkv_down_proj."
    ln_weight = torch.ones(3)
    ln_bias = torch.arange(3.0)
    base = {
        fused_prefix + "weight": layer.linear_qkv_down_proj.weight,
        fused_prefix + "layer_norm_weight": ln_weight,
        fused_prefix + "layer_norm_bias": ln_bias,
        prefix + "other.weight": torch.ones(2),
    }
    monkeypatch.setattr(MLASelfAttention, "sharded_state_dict", lambda *_args: base.copy())
    monkeypatch.setattr(mla_module, "get_pg_size", lambda _group: 1)
    monkeypatch.setattr(
        mla_module, "make_tp_sharded_tensor_for_checkpoint", lambda tensor, **_kwargs: tensor
    )
    state = layer.sharded_state_dict(prefix=prefix)
    assert set(state) == {
        fused_prefix + "layer_norm_weight",
        fused_prefix + "layer_norm_bias",
        prefix + "linear_q_down_proj.weight",
        prefix + "linear_kv_down_proj.weight",
        prefix + "other.weight",
    }
    assert state[fused_prefix + "layer_norm_weight"] is ln_weight
    assert state[fused_prefix + "layer_norm_bias"] is ln_bias
    torch.testing.assert_close(
        torch.cat(
            [
                state[prefix + "linear_q_down_proj.weight"],
                state[prefix + "linear_kv_down_proj.weight"],
            ]
        ),
        layer.linear_qkv_down_proj.weight,
        rtol=0,
        atol=0,
    )
