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


def test_qwen35_export_uses_hf_checkpoint_names_without_module_prefix() -> None:
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
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    }
    assert all(not name.startswith("module.") for name in exported)
    assert all(not name.startswith(("embed.", "norm.", "head.")) for name in exported)


def test_qwen35_export_dtype_cast_is_opt_in() -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(8)
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for expert_idx in range(config.num_experts):
                tensor = torch.arange(rows * config.hidden_size, dtype=torch.float32).reshape(
                    rows,
                    config.hidden_size,
                )
                tensor = tensor + expert_idx * 1000
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{expert_idx}",
                    nn.Parameter(tensor),
                )

    cfg = _tiny_config()
    model = TinyQwen35Module(cfg)

    default_export = dict(export_hf_weights(model, cfg, _single_rank_parallel_state()))
    bf16_export = dict(
        export_hf_weights(
            model,
            cfg,
            _single_rank_parallel_state(),
            export_dtype="bfloat16",
        )
    )

    assert default_export["model.language_model.norm.weight"].dtype == torch.float32
    assert bf16_export["model.language_model.norm.weight"].dtype == torch.bfloat16
    assert default_export["model.language_model.layers.0.mlp.experts.gate_up_proj"].dtype == torch.float32
    assert bf16_export["model.language_model.layers.0.mlp.experts.gate_up_proj"].dtype == torch.bfloat16


def test_qwen35_export_preserves_runtime_parameter_dtype_by_default() -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(8).to(torch.bfloat16)
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for expert_idx in range(config.num_experts):
                tensor = torch.arange(rows * config.hidden_size, dtype=torch.bfloat16).reshape(
                    rows,
                    config.hidden_size,
                )
                tensor = tensor + expert_idx * 1000
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{expert_idx}",
                    nn.Parameter(tensor),
                )

    cfg = _tiny_config()
    model = TinyQwen35Module(cfg)

    exported = dict(export_hf_weights(model, cfg, _single_rank_parallel_state()))

    assert exported["model.language_model.norm.weight"].dtype == torch.bfloat16
    assert exported["model.language_model.layers.0.mlp.experts.gate_up_proj"].dtype == torch.bfloat16


def test_qwen35_export_batches_ep_expert_gather(monkeypatch) -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for local_idx in range(config.num_experts // 2):
                tensor = torch.arange(rows * config.hidden_size, dtype=torch.bfloat16).reshape(
                    rows,
                    config.hidden_size,
                )
                tensor = tensor + local_idx * 1000
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{local_idx}",
                    nn.Parameter(tensor),
                )

    cfg = _tiny_config()
    model = TinyQwen35Module(cfg)
    ps = SimpleNamespace(
        pp_size=1,
        tp_size=1,
        tp_group=None,
        ep_size=2,
        ep_group=object(),
        etp_size=1,
        etp_group=None,
    )
    gather_calls = []

    def fake_all_gather(outputs, tensor, group=None):
        del group
        gather_calls.append(tensor.clone())
        outputs[0].copy_(tensor)
        outputs[1].copy_(tensor + 2000)

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.dist.all_gather",
        fake_all_gather,
    )

    exported = dict(export_hf_weights(model, cfg, ps))

    assert len(gather_calls) == 1
    assert gather_calls[0].shape[0] == cfg.num_experts // ps.ep_size
    local_tensors = [
        model.layers[0].moe.experts.fc1.weight0.detach(),
        model.layers[0].moe.experts.fc1.weight1.detach(),
    ]
    expected = torch.stack(
        [
            local_tensors[0],
            local_tensors[1],
            local_tensors[0] + 2000,
            local_tensors[1] + 2000,
        ],
        dim=0,
    )
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.experts.gate_up_proj"],
        expected,
    )


def test_qwen35_export_rank0_only_still_participates_in_ep_gather(monkeypatch) -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for local_idx in range(config.num_experts // 2):
                tensor = torch.zeros(rows, config.hidden_size, dtype=torch.bfloat16) + local_idx
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{local_idx}",
                    nn.Parameter(tensor),
                )

    cfg = _tiny_config()
    ps = SimpleNamespace(
        pp_size=1,
        tp_size=1,
        tp_group=None,
        ep_size=2,
        ep_group=object(),
        etp_size=1,
        etp_group=None,
    )
    gather_calls = []

    def fake_all_gather(outputs, tensor, group=None):
        del group
        gather_calls.append(tensor.clone())
        outputs[0].copy_(tensor)
        outputs[1].copy_(tensor + 2)

    monkeypatch.setattr("megatron.lite.primitive.ckpt.hf_weights.dist.is_initialized", lambda: True)
    monkeypatch.setattr("megatron.lite.primitive.ckpt.hf_weights.dist.get_rank", lambda: 1)
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.dist.all_gather",
        fake_all_gather,
    )

    exported = list(export_hf_weights(TinyQwen35Module(cfg), cfg, ps, rank0_only=True))

    assert exported == []
    assert len(gather_calls) == 1


def test_qwen35_export_maps_top_level_and_layer_norm_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.arange(cfg.hidden_size)

    cases = {
        "embed.embedding.weight": "model.language_model.embed_tokens.weight",
        "norm.weight": "model.language_model.norm.weight",
        "head.col.linear.weight": "lm_head.weight",
        "layers.0.full_attn.qkv.linear.layer_norm_weight": (
            "model.language_model.layers.0.input_layernorm.weight"
        ),
        "layers.0.mlp_norm.weight": "model.language_model.layers.0.post_attention_layernorm.weight",
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
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    }
    assert torch.equal(exported["model.language_model.layers.0.self_attn.q_proj.weight"], q_gate)
    assert torch.equal(exported["model.language_model.layers.0.self_attn.k_proj.weight"], key)
    assert torch.equal(exported["model.language_model.layers.0.self_attn.v_proj.weight"], value)


def test_qwen35_export_maps_linear_attention_to_hf_checkpoint_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    rows = qk_dim * 2 + v_dim * 2 + cfg.linear_num_value_heads * 2
    tensor = torch.arange(rows * cfg.hidden_size).reshape(rows, cfg.hidden_size)

    exported = dict(spec.native_to_hf("layers.0.linear_attn.in_proj.linear.weight", tensor))

    assert set(exported) == {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
    }
    assert (
        exported["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"].shape[0]
        == qk_dim * 2 + v_dim
    )
    assert exported["model.language_model.layers.0.linear_attn.in_proj_z.weight"].shape[0] == v_dim
    assert (
        exported["model.language_model.layers.0.linear_attn.in_proj_b.weight"].shape[0]
        == cfg.linear_num_value_heads
    )
    assert (
        exported["model.language_model.layers.0.linear_attn.in_proj_a.weight"].shape[0]
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

    assert set(exported) == {"model.language_model.layers.0.linear_attn.norm.weight"}
    assert torch.equal(
        exported["model.language_model.layers.0.linear_attn.norm.weight"],
        tensor + 1,
    )


def test_qwen35_export_maps_shared_expert_to_hf_checkpoint_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.arange(
        cfg.shared_expert_intermediate_size * 2 * cfg.hidden_size,
    ).reshape(-1, cfg.hidden_size)

    exported = dict(spec.native_to_hf("layers.0.moe.shared_expert.gate_up.linear.weight", tensor))

    assert set(exported) == {
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
    }
    gate, up = tensor.chunk(2, dim=0)
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"],
        gate,
    )
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.shared_expert.up_proj.weight"],
        up,
    )


def test_qwen35_export_packs_base_expert_fc1_to_hf_gate_up_proj() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    base = torch.arange(
        cfg.moe_intermediate_size * 2 * cfg.hidden_size,
    ).reshape(-1, cfg.hidden_size)

    exported = {}
    expert_tensors = []
    for expert_idx in range(cfg.num_experts):
        tensor = base + expert_idx * 1000
        expert_tensors.append(tensor)
        exported.update(
            dict(spec.native_to_hf(f"layers.0.moe.experts.fc1.weight{expert_idx}", tensor))
        )

    assert set(exported) == {"model.language_model.layers.0.mlp.experts.gate_up_proj"}
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.experts.gate_up_proj"],
        torch.stack(expert_tensors, dim=0),
    )


def test_qwen35_export_matches_mbridge_qwen35_moe_packed_expert_contract() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    rows = cfg.moe_intermediate_size * 2
    fc1_tensors = [
        torch.arange(rows * cfg.hidden_size, dtype=torch.bfloat16).reshape(rows, cfg.hidden_size)
        + expert_idx * 1000
        for expert_idx in range(cfg.num_experts)
    ]
    fc2_tensors = [
        torch.arange(cfg.hidden_size * cfg.moe_intermediate_size, dtype=torch.bfloat16).reshape(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
        )
        + expert_idx * 1000
        for expert_idx in range(cfg.num_experts)
    ]

    fc1_exported = {}
    fc2_exported = {}
    for expert_idx, (fc1, fc2) in enumerate(zip(fc1_tensors, fc2_tensors, strict=True)):
        fc1_exported.update(
            dict(spec.native_to_hf(f"layers.0.moe.experts.fc1.weight{expert_idx}", fc1))
        )
        fc2_exported.update(
            dict(spec.native_to_hf(f"layers.0.moe.experts.fc2.weight{expert_idx}", fc2))
        )

    assert set(fc1_exported) == {"model.language_model.layers.0.mlp.experts.gate_up_proj"}
    assert set(fc2_exported) == {"model.language_model.layers.0.mlp.experts.down_proj"}
    assert torch.equal(
        fc1_exported["model.language_model.layers.0.mlp.experts.gate_up_proj"],
        torch.stack(fc1_tensors, dim=0),
    )
    assert torch.equal(
        fc2_exported["model.language_model.layers.0.mlp.experts.down_proj"],
        torch.stack(fc2_tensors, dim=0),
    )


def test_qwen35_export_packs_base_expert_fc2_and_expert_metadata() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    base = torch.arange(
        cfg.hidden_size * cfg.moe_intermediate_size,
    ).reshape(cfg.hidden_size, cfg.moe_intermediate_size)
    native_name = "layers.0.moe.experts.fc2.weight2"

    exported = {}
    expert_tensors = []
    for expert_idx in range(cfg.num_experts):
        tensor = base + expert_idx * 1000
        expert_tensors.append(tensor)
        exported.update(
            dict(spec.native_to_hf(f"layers.0.moe.experts.fc2.weight{expert_idx}", tensor))
        )

    assert set(exported) == {"model.language_model.layers.0.mlp.experts.down_proj"}
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.experts.down_proj"],
        torch.stack(expert_tensors, dim=0),
    )
    assert spec.is_expert(native_name)
    assert spec.expert_global_id(native_name) == 2
    assert spec.expert_local_name(native_name, 0) == "layers.0.moe.experts.fc2.weight0"
    assert spec.tp_spec(native_name) == (1, 1)
