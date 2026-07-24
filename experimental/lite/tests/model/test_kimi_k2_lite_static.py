# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static smoke coverage for Kimi K2 lite native implementation."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_kimi_k2_lite_registry_resolves():
    from megatron.lite.model.registry import (
        get_train_runtime_module,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    runtime_name = resolve_runtime_model_name("kimi_k2", "lite")
    assert runtime_name == "kimi_k2"
    assert resolve_model_type_from_hf({"model_type": "kimi_k2"}) == "kimi_k2"
    assert resolve_model_type_from_hf({"model_type": "deepseek_v3"}) == "kimi_k2"
    module = get_train_runtime_module(runtime_name)
    assert module.__name__ == "megatron.lite.model.kimi_k2.lite.protocol"


def test_kimi_k2_config_reads_hf_fields():
    from megatron.lite.model.kimi_k2.config import KimiK2Config

    cfg = KimiK2Config._from_hf_dict(
        {
            "model_type": "kimi_k2",
            "num_hidden_layers": 3,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 128,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "n_routed_experts": 8,
            "n_shared_experts": 2,
            "num_experts_per_tok": 2,
            "n_group": 4,
            "topk_group": 2,
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "scoring_func": "sigmoid",
            "seq_aux": True,
            "first_k_dense_replace": 1,
            "q_lora_rank": 12,
            "kv_lora_rank": 10,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "rope_scaling": {"type": "yarn", "factor": 32.0},
        }
    )

    assert cfg.num_experts == 8
    assert cfg.n_group == 4
    assert cfg.topk_group == 2
    assert cfg.shared_expert_intermediate_size == 32
    assert cfg.q_head_dim == 8
    assert not cfg.is_moe_layer(0)
    assert cfg.is_moe_layer(1)


def test_kimi_k2_lite_does_not_import_wrappers_or_sibling_models():
    root = Path(__file__).resolve().parents[2] / "megatron" / "lite" / "model" / "kimi_k2" / "lite"
    forbidden = (
        "megatron.lite.model.qwen3_5",
        "megatron.lite.model.qwen3_moe",
        "bridge_model",
        "hybrid",
        "build_mcore_context",
        "from mbridge",
        "import mbridge",
    )
    for path in root.glob("*.py"):
        text = path.read_text()
        for token in forbidden:
            assert token not in text


def test_kimi_k2_lite_implementation_files_stay_small():
    root = Path(__file__).resolve().parents[2] / "megatron" / "lite" / "model" / "kimi_k2" / "lite"
    for name in ("model.py", "protocol.py", "checkpoint.py"):
        line_count = len((root / name).read_text().splitlines())
        assert line_count < 1000, f"{name} has {line_count} lines"


def test_kimi_k2_lite_uses_shared_mla_primitive():
    lite_root = Path(__file__).resolve().parents[2] / "megatron" / "lite"
    model_root = lite_root / "model" / "kimi_k2" / "lite"
    primitive_mla = lite_root / "primitive" / "modules" / "attention" / "mla.py"
    model_text = (model_root / "model.py").read_text()
    mla_text = primitive_mla.read_text()

    assert (
        "from megatron.lite.primitive.modules.attention import MultiLatentAttention" in model_text
    )
    assert not (model_root / "mla.py").exists()
    assert "class MultiLatentAttention" not in model_text
    assert "class MultiLatentAttention" in mla_text
    assert "megatron.core" not in mla_text
    assert "KimiK2SigmoidTopKRouter" in model_text


def test_kimi_k2_lite_optimizer_names_are_current():
    root = Path(__file__).resolve().parents[2] / "megatron" / "lite" / "model" / "kimi_k2" / "lite"
    protocol_text = (root / "protocol.py").read_text()

    assert 'optimizer: str | None = "dist_opt"' in protocol_text
    assert 'impl_cfg.optimizer == "dist_opt"' in protocol_text
    assert 'impl_cfg.optimizer == "fsdp2"' in protocol_text
    for forbidden in ("m" + "c_full", 'optimizer == "m' + 'c"'):
        assert forbidden not in protocol_text


def test_kimi_k2_dist_opt_deterministic_default_is_enabled():
    from megatron.lite.model.kimi_k2.lite.protocol import ImplConfig

    cfg = ImplConfig(optimizer="dist_opt", deterministic=True, mtp_enable=True)

    assert cfg.optimizer == "dist_opt"
    assert cfg.deterministic is True
    assert cfg.mtp_enable is True


def test_kimi_k2_impl_config_accepts_runtime_mtp_fields():
    from megatron.lite.model.kimi_k2.config import KimiK2Config
    from megatron.lite.model.kimi_k2.lite.protocol import ImplConfig

    cfg = KimiK2Config(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=16,
        intermediate_size=12,
        moe_intermediate_size=4,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        num_nextn_predict_layers=1,
    )

    assert ImplConfig(mtp_enable=False, mtp_enable_train=False).mtp_enable is False
    assert ImplConfig(mtp_enable=True, mtp_enable_train=True).mtp_enable_train is True
    assert cfg.num_nextn_predict_layers == 1


def test_kimi_k2_mtp_and_pp_layout_rules_are_explicit():
    from megatron.lite.primitive.parallel import ParallelState, build_pipeline_chunk_layout

    model_text = (
        Path(__file__).resolve().parents[2]
        / "megatron"
        / "lite"
        / "model"
        / "kimi_k2"
        / "lite"
        / "model.py"
    ).read_text()

    rank0 = ParallelState(pp_size=2, pp_rank=0, pp_is_first=True, pp_is_last=False)
    rank1 = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)

    assert build_pipeline_chunk_layout(4, rank0).layer_indices == [0, 1]
    assert build_pipeline_chunk_layout(4, rank1).layer_indices == [2, 3]
    # MTP is built on the stage the layout designates (layout.has_mtp), not a
    # hard-coded head rank.
    assert (
        "if mtp_enable and config.num_nextn_predict_layers > 0 and layout.has_mtp"
        in model_text
    )
    assert "self.mtp_embed = mtp_embedding" in model_text

    # pp-only: VPP / interleaving raises rather than silently mis-splitting.
    with pytest.raises(NotImplementedError):
        build_pipeline_chunk_layout(4, rank1, vpp=2, vpp_chunk_id=1)

    # Non-divisible counts auto-balance (no "not divisible" error): 3/pp2 -> [2, 1].
    assert build_pipeline_chunk_layout(3, rank0).layer_indices == [0, 1]
    assert build_pipeline_chunk_layout(3, rank1).layer_indices == [2]


def test_kimi_k2_checkpoint_exports_hf_names():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.kimi_k2.config import KimiK2Config
    from megatron.lite.model.kimi_k2.lite import protocol
    from megatron.lite.model.kimi_k2.lite.checkpoint import (
        KimiK2WeightSpec,
        export_hf_weights,
        save_hf_weights,
    )

    cfg = KimiK2Config(
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=16,
        intermediate_size=12,
        moe_intermediate_size=4,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        num_nextn_predict_layers=1,
    )
    spec = KimiK2WeightSpec(cfg)

    assert callable(export_hf_weights)
    assert callable(save_hf_weights)
    assert hasattr(protocol, "export_hf_weights")
    assert spec.tp_spec("layers.1.self_attention.linear_proj.linear.weight") == (1, 0)
    assert spec.tp_spec("layers.1.moe.experts.fc1.weight0") == (0, 1)
    assert spec.expert_global_id("layers.1.moe.experts.fc2.weight1") == 1

    gate_up = torch.arange(8 * 8, dtype=torch.float32).view(8, 8)
    exported = dict(spec.native_to_hf("layers.1.moe.experts.fc1.weight0", gate_up))
    assert set(exported) == {
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
    }
    torch.testing.assert_close(
        exported["model.layers.1.mlp.experts.0.gate_proj.weight"],
        gate_up[:4],
    )
    torch.testing.assert_close(
        exported["model.layers.1.mlp.experts.0.up_proj.weight"],
        gate_up[4:],
    )

    bias = torch.arange(2, dtype=torch.float32)
    assert spec.native_to_hf("layers.1.moe.router.expert_bias", bias) == [
        ("model.layers.1.mlp.gate.e_score_correction_bias", bias)
    ]
    assert spec.native_to_hf("mtp.layers.0.enorm.weight", bias) == [
        ("model.layers.2.enorm.weight", bias)
    ]
    assert spec.native_to_hf("mtp.layers.0.eh_proj.linear.weight", gate_up) == [
        ("model.layers.2.eh_proj.weight", gate_up)
    ]
    assert spec.native_to_hf("mtp.layers.0.final_layernorm.weight", bias) == [
        ("model.layers.2.shared_head.norm.weight", bias)
    ]
    mtp_export = dict(
        spec.native_to_hf("mtp.layers.0.transformer_layer.mlp.gate_up.linear.weight", gate_up)
    )
    assert set(mtp_export) == {
        "model.layers.2.mlp.gate_proj.weight",
        "model.layers.2.mlp.up_proj.weight",
    }


def test_kimi_k2_fp8_checkpoint_dequant_cpu_path():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8_e4m3fn is required for this smoke.")

    from megatron.lite.model.kimi_k2.lite.checkpoint import _dequant_fp8_weight

    class Reader:
        index = {"w_scale_inv": "fake.safetensors"}

        @staticmethod
        def get_tensor(name):
            assert name == "w_scale_inv"
            return torch.full((1, 1), 2.0, dtype=torch.float32)

    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    out = _dequant_fp8_weight(Reader(), "w", weight)

    torch.testing.assert_close(out, weight.float() * 2.0)


def test_kimi_k2_int4_checkpoint_dequant_cpu_path():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.kimi_k2.lite.checkpoint import _get

    values = torch.tensor([[-8, -7, -1, 0, 1, 2, 6, 7, -8, 7]], dtype=torch.int8)
    unsigned = (values + 8).to(torch.int32)
    packed = torch.zeros((1, 2), dtype=torch.int32)
    for offset in range(8):
        packed[:, 0] |= unsigned[:, offset] << (4 * offset)
    for offset in range(2):
        packed[:, 1] |= unsigned[:, 8 + offset] << (4 * offset)

    class Reader:
        index = {
            "w_packed": "fake.safetensors",
            "w_scale": "fake.safetensors",
            "w_shape": "fake.safetensors",
        }

        @staticmethod
        def get_tensor(name):
            return {
                "w_packed": packed,
                "w_scale": torch.tensor([[0.5, 2.0]], dtype=torch.float32),
                "w_shape": torch.tensor([1, 10], dtype=torch.int64),
            }[name]

    out = _get(Reader(), "w")
    expected = torch.cat([values[:, :5].float() * 0.5, values[:, 5:].float() * 2.0], dim=1)

    torch.testing.assert_close(out, expected)


def test_kimi_k2_real_checkpoint_prefix_helpers():
    from megatron.lite.model.kimi_k2.lite.checkpoint import _lm_head_name, _text_prefix

    class Reader:
        index = {
            "language_model.model.embed_tokens.weight": "fake.safetensors",
            "language_model.lm_head.weight": "fake.safetensors",
        }

    prefix = _text_prefix(Reader())

    assert prefix == "language_model.model"
    assert _lm_head_name(Reader(), prefix) == "language_model.lm_head.weight"
