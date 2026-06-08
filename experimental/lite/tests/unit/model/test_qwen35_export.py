from types import SimpleNamespace

import torch
import torch.nn as nn

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_5.lite.checkpoint import (
    Qwen35WeightSpec,
    _merge_gate_up_tp_shards,
    _merge_full_attn_qkvg,
    _merge_linear_attn_in_proj_tp_shards,
    export_hf_weights,
)
from megatron.lite.model.registry import TRAIN_RUNTIME_MODULES, resolve_runtime_model_name


def _tiny_config() -> Qwen35Config:
    return Qwen35Config(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        layer_types=["full_attention"],
        partial_rotary_factor=1.0,
    )


def _single_rank_parallel_state() -> SimpleNamespace:
    return SimpleNamespace(
        pp_size=1,
        tp_size=1,
        tp_group=None,
        ep_size=1,
        ep_group=None,
        etp_size=1,
        etp_group=None,
    )


def test_qwen35_protocol_registers_vllm_export_entrypoint() -> None:
    key = resolve_runtime_model_name("qwen3_5", "lite")
    module = __import__(TRAIN_RUNTIME_MODULES[key], fromlist=["export_hf_weights"])

    assert key == "qwen3_5"
    assert callable(module.export_hf_weights)


def test_qwen35_export_uses_vllm_loader_names_without_module_prefix() -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Module()
            self.embed.embedding = nn.Embedding(16, 8)
            self.norm = nn.LayerNorm(8)
            self.head = nn.Module()
            self.head.col = nn.Module()
            self.head.col.linear = nn.Linear(8, 16, bias=False)

    cfg = _tiny_config()
    model = TinyQwen35Module()

    exported = dict(export_hf_weights(model, cfg, _single_rank_parallel_state()))

    assert set(exported) == {
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    }
    assert all(not name.startswith("module.") for name in exported)
    assert all(not name.startswith(("embed.", "norm.", "head.")) for name in exported)


def test_qwen35_export_maps_top_level_and_layer_norm_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.arange(cfg.hidden_size)

    cases = {
        "embed.embedding.weight": "language_model.model.embed_tokens.weight",
        "norm.weight": "language_model.model.norm.weight",
        "head.col.linear.weight": "language_model.lm_head.weight",
        "layers.0.full_attn.qkv.linear.layer_norm_weight": (
            "language_model.model.layers.0.input_layernorm.weight"
        ),
        "layers.0.mlp_norm.weight": "language_model.model.layers.0.post_attention_layernorm.weight",
    }

    for native_name, hf_name in cases.items():
        exported = dict(spec.native_to_hf(native_name, tensor))
        assert set(exported) == {hf_name}
        assert torch.equal(exported[hf_name], tensor)


def test_qwen35_export_unpacks_full_attention_q_gate() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    hidden = cfg.hidden_size
    q_gate = torch.arange(
        cfg.num_attention_heads * 2 * cfg.head_dim * hidden,
    ).reshape(-1, hidden)
    key = torch.arange(
        q_gate.numel(),
        q_gate.numel() + cfg.num_key_value_heads * cfg.head_dim * hidden,
    ).reshape(
        -1,
        hidden,
    )
    value = torch.arange(
        key[-1, -1] + 1,
        key[-1, -1] + 1 + cfg.num_key_value_heads * cfg.head_dim * hidden,
    ).reshape(
        -1,
        hidden,
    )

    packed = _merge_full_attn_qkvg(q_gate, key, value, cfg=cfg)
    exported = dict(spec.native_to_hf("layers.0.full_attn.qkv.linear.weight", packed))

    assert set(exported) == {
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
    }
    assert torch.equal(exported["language_model.model.layers.0.self_attn.q_proj.weight"], q_gate)
    assert torch.equal(exported["language_model.model.layers.0.self_attn.k_proj.weight"], key)
    assert torch.equal(exported["language_model.model.layers.0.self_attn.v_proj.weight"], value)


def test_qwen35_export_maps_linear_attention_to_vllm_loader_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    rows = qk_dim * 2 + v_dim * 2 + cfg.linear_num_value_heads * 2
    tensor = torch.arange(rows * cfg.hidden_size).reshape(rows, cfg.hidden_size)

    exported = dict(spec.native_to_hf("layers.0.linear_attn.in_proj.linear.weight", tensor))

    assert set(exported) == {
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.0.linear_attn.in_proj_z.weight",
        "language_model.model.layers.0.linear_attn.in_proj_b.weight",
        "language_model.model.layers.0.linear_attn.in_proj_a.weight",
    }
    assert (
        exported["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"].shape[0]
        == qk_dim * 2 + v_dim
    )
    assert exported["language_model.model.layers.0.linear_attn.in_proj_z.weight"].shape[0] == v_dim
    assert (
        exported["language_model.model.layers.0.linear_attn.in_proj_b.weight"].shape[0]
        == cfg.linear_num_value_heads
    )
    assert (
        exported["language_model.model.layers.0.linear_attn.in_proj_a.weight"].shape[0]
        == cfg.linear_num_value_heads
    )


def test_qwen35_export_reorders_linear_attention_tp_shards_before_hf_split() -> None:
    cfg = _tiny_config()
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    hidden = cfg.hidden_size
    parts = [
        torch.arange(0, qk_dim * hidden).reshape(qk_dim, hidden),
        torch.arange(100, 100 + qk_dim * hidden).reshape(qk_dim, hidden),
        torch.arange(200, 200 + v_dim * hidden).reshape(v_dim, hidden),
        torch.arange(300, 300 + v_dim * hidden).reshape(v_dim, hidden),
        torch.arange(400, 400 + cfg.linear_num_value_heads * hidden).reshape(
            cfg.linear_num_value_heads,
            hidden,
        ),
        torch.arange(500, 500 + cfg.linear_num_value_heads * hidden).reshape(
            cfg.linear_num_value_heads,
            hidden,
        ),
    ]
    full = torch.cat(parts, dim=0)
    shards = [
        torch.cat([part.chunk(2, dim=0)[rank] for part in parts], dim=0)
        for rank in range(2)
    ]

    merged = _merge_linear_attn_in_proj_tp_shards(shards, cfg=cfg)

    assert torch.equal(merged, full)


def test_qwen35_export_reorders_shared_expert_gate_up_tp_shards() -> None:
    gate = torch.arange(0, 32).reshape(4, 8)
    up = torch.arange(100, 132).reshape(4, 8)
    full = torch.cat([gate, up], dim=0)
    shards = [
        torch.cat([gate.chunk(2, dim=0)[rank], up.chunk(2, dim=0)[rank]], dim=0)
        for rank in range(2)
    ]

    merged = _merge_gate_up_tp_shards(shards)

    assert torch.equal(merged, full)


def test_qwen35_export_restores_zero_centered_linear_attention_norm() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.tensor([-0.5, 0.0, 0.5])

    exported = dict(spec.native_to_hf("layers.0.linear_attn.norm.weight", tensor))

    assert set(exported) == {"language_model.model.layers.0.linear_attn.norm.weight"}
    assert torch.equal(
        exported["language_model.model.layers.0.linear_attn.norm.weight"],
        tensor + 1,
    )


def test_qwen35_export_maps_shared_expert_to_vllm_loader_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.arange(
        cfg.shared_expert_intermediate_size * 2 * cfg.hidden_size,
    ).reshape(-1, cfg.hidden_size)

    exported = dict(spec.native_to_hf("layers.0.moe.shared_expert.gate_up.linear.weight", tensor))

    assert set(exported) == {
        "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.0.mlp.shared_expert.up_proj.weight",
    }
    gate, up = tensor.chunk(2, dim=0)
    assert torch.equal(
        exported["language_model.model.layers.0.mlp.shared_expert.gate_proj.weight"],
        gate,
    )
    assert torch.equal(
        exported["language_model.model.layers.0.mlp.shared_expert.up_proj.weight"],
        up,
    )


def test_qwen35_export_maps_expert_fc1_to_individual_expert_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.arange(
        cfg.moe_intermediate_size * 2 * cfg.hidden_size,
    ).reshape(-1, cfg.hidden_size)

    exported = dict(spec.native_to_hf("layers.0.moe.experts.fc1.weight3", tensor))

    assert set(exported) == {
        "language_model.model.layers.0.mlp.experts.3.gate_proj.weight",
        "language_model.model.layers.0.mlp.experts.3.up_proj.weight",
    }
    gate, up = tensor.chunk(2, dim=0)
    assert torch.equal(
        exported["language_model.model.layers.0.mlp.experts.3.gate_proj.weight"],
        gate,
    )
    assert torch.equal(exported["language_model.model.layers.0.mlp.experts.3.up_proj.weight"], up)


def test_qwen35_export_maps_expert_fc2_and_expert_metadata() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.arange(
        cfg.hidden_size * cfg.moe_intermediate_size,
    ).reshape(cfg.hidden_size, cfg.moe_intermediate_size)
    native_name = "layers.0.moe.experts.fc2.weight2"

    exported = dict(spec.native_to_hf(native_name, tensor))

    assert set(exported) == {"language_model.model.layers.0.mlp.experts.2.down_proj.weight"}
    assert torch.equal(exported["language_model.model.layers.0.mlp.experts.2.down_proj.weight"], tensor)
    assert spec.is_expert(native_name)
    assert spec.expert_global_id(native_name) == 2
    assert spec.expert_local_name(native_name, 0) == "layers.0.moe.experts.fc2.weight0"
    assert spec.tp_spec(native_name) == (1, 1)
