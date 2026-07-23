# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static and CPU smoke tests for native GLM-5 lite."""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_train_config(ps):
    from types import SimpleNamespace

    return SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        deterministic=True,
    )


def _make_glm5_model(cfg, ps=None, **kwargs):
    from megatron.lite.model.glm5.lite.model import Glm5Model
    from megatron.lite.primitive.parallel import ParallelState

    ps = ParallelState() if ps is None else ps
    return Glm5Model(cfg, _make_train_config(ps), ps, **kwargs)


def _tiny_config_kwargs():
    return dict(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=16,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )


def _glm52_indexer_types(num_layers=78):
    full_layers = {0, 1, 2, *range(6, num_layers, 4)}
    return ["full" if idx in full_layers else "shared" for idx in range(num_layers)]


def test_glm5_registry_resolves_lite():
    from megatron.lite.model.registry import (
        get_train_runtime_module,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    runtime_name = resolve_runtime_model_name("glm5", "lite")
    assert runtime_name == "glm5"
    module = get_train_runtime_module(runtime_name)
    assert module.__name__ == "megatron.lite.model.glm5.lite.protocol"
    assert resolve_model_type_from_hf({"model_type": "glm_moe_dsa"}) == "glm5"


def test_glm5_config_reads_hf_architecture_fields():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            "model_type": "glm_moe_dsa",
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_head_dim": 256,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 2048,
            "first_k_dense_replace": 3,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "vocab_size": 154880,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        }
    )

    assert cfg.q_lora_rank == 2048
    assert cfg.kv_lora_rank == 512
    assert cfg.index_topk == 2048
    assert cfg.num_nextn_predict_layers == 1
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.is_moe_layer(2) is False
    assert cfg.is_moe_layer(3) is True


def test_glm5_config_ignores_null_hf_optional_fields():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            "model_type": "glm_moe_dsa",
            "indexer_rope_first": None,
            "indexer_use_hadamard": None,
            "mlp_layer_types": None,
        }
    )

    assert cfg.indexer_rope_first is True
    assert cfg.indexer_use_hadamard is False
    assert cfg.mlp_layer_types is None


def test_glm52_config_validates_index_share_schedule():
    import pytest

    from megatron.lite.model.glm5.config import Glm5Config

    indexer_types = _glm52_indexer_types()
    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 78,
            "num_nextn_predict_layers": 1,
        },
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=indexer_types,
        index_share_for_mtp_iteration=True,
    )

    assert indexer_types.count("full") == 21
    assert indexer_types.count("shared") == 57
    assert cfg.uses_dsa_index_share is True
    assert cfg.dsa_indexer_type(2) == "full"
    assert cfg.dsa_indexer_type(3) == "shared"
    assert cfg.dsa_indexer_source_layer(3) == 2
    assert cfg.dsa_indexer_type(74) == "full"
    assert cfg.dsa_indexer_type(77) == "shared"
    assert cfg.dsa_indexer_source_layer(77) == 74
    # MTP layer 78 (0-based) is outside trunk indexer_types and is full by
    # the same global layer-number schedule.
    assert cfg.dsa_indexer_type(78) == "full"
    assert cfg.builds_dsa_indexer(78) is True
    assert cfg.index_share_for_mtp_iteration is True

    bad_types = list(indexer_types)
    bad_types[3] = "full"
    with pytest.raises(ValueError, match="indexer_types\\[3\\]"):
        Glm5Config(
            **{**_tiny_config_kwargs(), "num_hidden_layers": 78},
            index_topk_freq=4,
            index_skip_topk_offset=3,
            indexer_types=bad_types,
        )


def test_glm5_config_preserves_mtp_aliases_and_layer_types():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            **_tiny_config_kwargs(),
            "num_nextn_predict": 1,
            "mtp_loss_scaling_factor": 0.2,
            "mlp_layer_types": ["dense", "sparse", "sparse"],
        }
    )

    assert cfg.num_nextn_predict_layers == 1
    assert cfg.mtp_loss_scaling_factor == 0.2
    assert cfg.is_moe_layer(2) is True


def test_glm5_lite_does_not_import_wrappers_or_sibling_models():
    root = Path(__file__).resolve().parents[3] / "megatron" / "lite" / "model" / "glm5" / "lite"
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "megatron.lite.model.qwen" not in text
        assert "mbridge" not in text
        assert "MCore" not in text
        assert "megatron.core" not in text


def test_glm5_lite_uses_shared_mla_and_dsa_primitive():
    root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    model_text = (root / "model" / "glm5" / "lite" / "model.py").read_text()
    primitive_text = (root / "primitive" / "modules" / "attention" / "dsa.py").read_text()
    kernel_text = (root / "primitive" / "kernels" / "dsa_kernels.py").read_text()

    assert "DynamicSparseAttention" in model_text
    assert (
        "from megatron.lite.primitive.modules.attention.mla import MultiLatentAttention"
        in primitive_text
    )
    assert "class DynamicSparseAttention" in primitive_text
    assert "class MultiLatentAttention" not in primitive_text
    assert "class DSAIndexer" in primitive_text
    assert "megatron.core" not in primitive_text
    assert "dsa_kernels.fused_indexer_sparse_attn" in primitive_text
    assert "dsa_kernels.dsa_sparse_attn" in primitive_text
    assert "dsa_kernels.indexer_topk" in primitive_text
    assert "value_dim" in kernel_text
    assert "from cudnn.deepseek_sparse_attention import DSA" in kernel_text
    assert "from cudnn import DSA" in kernel_text
    assert "cudnn.deepseek_sparse_attention.indexer_forward._interface_sm90" in kernel_text
    assert "cudnn.deepseek_sparse_attention.indexer_forward._interface" in kernel_text
    assert "torch.cuda.get_device_capability(device)" in kernel_text
    assert "torch.topk" not in primitive_text
    assert "torch.softmax" not in primitive_text
    assert "torch.matmul" not in primitive_text


def test_glm5_dsa_kernel_routes_indexer_forward_by_sm(monkeypatch):
    from megatron.lite.primitive.kernels import dsa_kernels

    sm90_entry = object()
    sm100_entry = object()

    monkeypatch.setattr(dsa_kernels, "_load_indexer_fwd_sm90", lambda: sm90_entry)
    monkeypatch.setattr(dsa_kernels, "_load_indexer_fwd_sm100", lambda: sm100_entry)

    monkeypatch.setattr(dsa_kernels.torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert dsa_kernels._select_indexer_forward(None) is sm90_entry

    monkeypatch.setattr(dsa_kernels.torch.cuda, "get_device_capability", lambda device: (10, 0))
    assert dsa_kernels._select_indexer_forward(None) is sm100_entry

    monkeypatch.setattr(dsa_kernels.torch.cuda, "get_device_capability", lambda device: (8, 0))
    assert dsa_kernels._select_indexer_forward(None) is None


def test_glm5_dsa_training_forward_uses_fused_kernel(monkeypatch):
    import pytest
    import torch

    from megatron.lite.primitive.modules.attention import (
        DynamicSparseAttention,
        build_rope_cache,
        dsa,
    )

    if not torch.cuda.is_available():
        pytest.skip("GLM-5 native attention requires CUDA (Transformer Engine RMSNorm)")
    device = torch.device("cuda")

    calls = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        kv_offset=0,
        calculate_per_token_loss=False,
        value_dim=None,
    ):
        del attn_sink, q_indexer, k_indexer, weights, softmax_scale, indexer_softmax_scale
        calls["training"] = {
            "query_shape": tuple(query.shape),
            "kv_shape": tuple(kv_full.shape),
            "window_shape": tuple(window_idxs.shape),
            "indexer_topk": indexer_topk,
            "ratio": ratio,
            "loss_coeff": loss_coeff,
            "sparse_loss": sparse_loss,
            "kv_offset": kv_offset,
            "calculate_per_token_loss": calculate_per_token_loss,
            "value_dim": value_dim,
        }
        return query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        ), torch.zeros((), device=query.device, dtype=torch.float32)

    monkeypatch.setattr(
        dsa._dsa_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    attn = DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1e-5,
    )
    attn.to(device=device, dtype=torch.bfloat16)
    attn.train()
    x = torch.randn(1, 4, 16, device=device, dtype=torch.bfloat16)
    cos, sin = build_rope_cache(
        dim=4, max_position_embeddings=4, rope_theta=1_000_000.0, device=device
    )
    position_ids = torch.arange(4, device=device).unsqueeze(0)

    out = attn(x, cos=cos, sin=sin, position_ids=position_ids)

    assert out.shape == (1, 4, 16)
    assert calls["training"] == {
        "query_shape": (4, 1, 2, 8),
        "kv_shape": (4, 1, 8),
        "window_shape": (1, 4, 0),
        "indexer_topk": 2,
        "ratio": 1,
        "loss_coeff": 0.0,
        "sparse_loss": False,
        "kv_offset": 0,
        "calculate_per_token_loss": False,
        "value_dim": 4,
    }


def test_glm5_dsa_eval_forward_uses_fused_sparse_attention(monkeypatch):
    import pytest
    import torch

    from megatron.lite.primitive.modules.attention import (
        DynamicSparseAttention,
        build_rope_cache,
        dsa,
    )

    if not torch.cuda.is_available():
        pytest.skip("GLM-5 native attention requires CUDA (Transformer Engine RMSNorm)")
    device = torch.device("cuda")

    calls = {}

    def fake_indexer_topk(q_indexer, k_indexer, weights, topk, ratio, indexer_softmax_scale=1.0):
        del q_indexer, k_indexer, weights, indexer_softmax_scale
        calls["indexer"] = {"topk": topk, "ratio": ratio}
        idx = torch.zeros((1, 4, topk), dtype=torch.int32, device=device)
        return idx, torch.full((1, 4), topk, dtype=torch.int32, device=device)

    def fake_dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length=None,
        indexer_topk=0,
        value_dim=None,
    ):
        del kv_full, attn_sink, topk_idxs, softmax_scale, indexer_topk
        calls["sparse"] = {"topk_length_is_set": topk_length is not None, "value_dim": value_dim}
        return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", fake_indexer_topk)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", fake_dsa_sparse_attn)

    attn = DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1e-5,
    )
    attn.to(device=device, dtype=torch.bfloat16)
    attn.eval()
    x = torch.randn(1, 4, 16, device=device, dtype=torch.bfloat16)
    cos, sin = build_rope_cache(
        dim=4, max_position_embeddings=4, rope_theta=1_000_000.0, device=device
    )
    position_ids = torch.arange(4, device=device).unsqueeze(0)

    with torch.no_grad():
        out = attn(x, cos=cos, sin=sin, position_ids=position_ids)

    assert out.shape == (1, 4, 16)
    assert calls["indexer"] == {"topk": 2, "ratio": 1}
    assert calls["sparse"] == {"topk_length_is_set": True, "value_dim": 4}


def test_glm5_lite_model_exports_native_state_names():
    from megatron.lite.model.glm5.config import Glm5Config

    model = _make_glm5_model(Glm5Config(**_tiny_config_kwargs()))
    keys = set(model.state_dict())

    assert "embed.embedding.weight" in keys
    assert "layers.0.self_attention.self_attention.q_a_proj.weight" in keys
    assert "layers.0.mlp.gate_up.linear.weight" in keys
    assert "layers.1.moe.router.gate.weight" in keys
    assert "layers.1.moe.experts.fc1.weight0" in keys
    assert "layers.1.moe.shared_expert.gate_up.linear.weight" in keys
    assert "head.col.linear.weight" in keys


def test_glm52_index_share_shared_layers_omit_indexer_modules():
    import pytest

    try:
        import transformer_engine.pytorch  # noqa: F401
    except (ModuleNotFoundError, OSError) as exc:
        pytest.skip(f"Transformer Engine is not importable in this environment: {exc}")

    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 6,
            "num_nextn_predict_layers": 1,
        },
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=_glm52_indexer_types(num_layers=6),
    )
    model = _make_glm5_model(cfg, mtp_enable=True)
    attention_modules = [layer.self_attention.self_attention for layer in model.layers]

    assert [module.indexer is not None for module in attention_modules] == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert model.mtp is not None
    mtp_attention = model.mtp.layers[0].transformer_layer.self_attention.self_attention
    assert mtp_attention.layer_number == 7
    assert mtp_attention.indexer is not None

    keys = set(model.state_dict())
    assert "layers.2.self_attention.self_attention.indexer.wq_b.weight" in keys
    assert "layers.3.self_attention.self_attention.indexer.wq_b.weight" not in keys
    assert (
        "mtp.layers.0.transformer_layer.self_attention.self_attention.indexer.wq_b.weight" in keys
    )


def test_glm5_checkpoint_exports_and_saves_hf_style_weights(tmp_path):
    import torch
    from safetensors import safe_open

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import export_hf_weights, save_hf_weights
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Glm5Config(**_tiny_config_kwargs())
    ps = ParallelState()
    model = _make_glm5_model(cfg, ps=ps)
    model.layers[1].moe.router.expert_bias.copy_(torch.tensor([0.25, -0.5, 1.0]))

    exported = dict(export_hf_weights(model, cfg, ps))
    state = model.state_dict()

    assert torch.equal(
        exported["model.layers.1.mlp.experts.2.gate_proj.weight"].detach().cpu(),
        state["layers.1.moe.experts.fc1.weight2"][: cfg.moe_intermediate_size].detach().cpu(),
    )
    assert torch.equal(
        exported["model.layers.1.mlp.gate.e_score_correction_bias"].detach().cpu(),
        state["layers.1.moe.router.expert_bias"].detach().cpu(),
    )
    assert "model.layers.1.mlp.experts.gate_up_proj" not in exported

    hf_dir = tmp_path / "hf"
    save_hf_weights(model, str(hf_dir), cfg, ps)
    with safe_open(str(hf_dir / "model.safetensors"), framework="pt", device="cpu") as handle:
        assert torch.equal(
            handle.get_tensor("model.layers.1.mlp.experts.2.down_proj.weight"),
            state["layers.1.moe.experts.fc2.weight2"].detach().cpu(),
        )
        assert torch.equal(
            handle.get_tensor("model.layers.1.mlp.gate.e_score_correction_bias"),
            state["layers.1.moe.router.expert_bias"].detach().cpu(),
        )

    loaded = _make_glm5_model(cfg, ps=ps)
    from megatron.lite.model.glm5.lite.checkpoint import load_hf_weights

    load_hf_weights(loaded, str(hf_dir), cfg, ps)
    assert torch.equal(
        loaded.state_dict()["layers.1.moe.router.expert_bias"].detach().cpu(),
        state["layers.1.moe.router.expert_bias"].detach().cpu(),
    )

    hf_bf16_dir = tmp_path / "hf_bf16"
    save_hf_weights(model, str(hf_bf16_dir), cfg, ps, export_dtype=torch.bfloat16)
    with safe_open(str(hf_bf16_dir / "model.safetensors"), framework="pt", device="cpu") as handle:
        floating_dtypes = {
            handle.get_tensor(key).dtype
            for key in handle.keys()
            if handle.get_tensor(key).is_floating_point()
        }
        assert floating_dtypes == {torch.bfloat16}

    loaded_bf16 = _make_glm5_model(cfg, ps=ps)
    load_hf_weights(loaded_bf16, str(hf_bf16_dir), cfg, ps)
    assert torch.equal(
        loaded_bf16.state_dict()["layers.1.moe.experts.fc1.weight2"][cfg.moe_intermediate_size :]
        .detach()
        .cpu(),
        state["layers.1.moe.experts.fc1.weight2"][cfg.moe_intermediate_size :]
        .detach()
        .cpu()
        .to(torch.bfloat16)
        .to(torch.float32),
    )


def test_glm5_hf_export_rejects_missing_model_config():
    from megatron.lite.model.glm5.lite.checkpoint import export_hf_weights
    from megatron.lite.primitive.parallel import ParallelState

    with pytest.raises(ValueError, match="non-null model config"):
        next(export_hf_weights(None, None, ParallelState()))


def test_glm5_checkpoint_exports_and_loads_mtp_layers(tmp_path):
    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import export_hf_weights, load_hf_weights
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1)
    ps = ParallelState()
    model = _make_glm5_model(cfg, ps=ps, mtp_enable=True)
    state = model.state_dict()

    assert "mtp.layers.0.eh_proj.linear.weight" in state
    assert "mtp.layers.0.transformer_layer.input_layernorm.weight" in state

    exported = dict(export_hf_weights(model, cfg, ps))
    assert "model.layers.2.eh_proj.weight" in exported
    assert "model.layers.2.enorm.weight" in exported
    assert "model.layers.2.hnorm.weight" in exported
    assert "model.layers.2.shared_head.norm.weight" in exported
    assert "model.layers.2.input_layernorm.weight" in exported
    assert "model.layers.2.mlp.gate.weight" in exported

    save_safetensors(exported, str(tmp_path))
    loaded = _make_glm5_model(cfg, ps=ps, mtp_enable=True)
    load_hf_weights(loaded, str(tmp_path), cfg, ps)
    assert torch.equal(
        loaded.state_dict()["mtp.layers.0.eh_proj.linear.weight"],
        state["mtp.layers.0.eh_proj.linear.weight"],
    )


def test_glm52_checkpoint_mapping_skips_shared_indexer_without_te():
    import importlib.util

    import torch

    from megatron.lite.model.glm5.config import Glm5Config

    checkpoint_path = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "glm5"
        / "lite"
        / "checkpoint.py"
    )
    module_spec = importlib.util.spec_from_file_location("_glm5_checkpoint_test", checkpoint_path)
    assert module_spec is not None and module_spec.loader is not None
    checkpoint_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(checkpoint_module)

    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 6,
            "num_nextn_predict_layers": 1,
        },
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=_glm52_indexer_types(num_layers=6),
        index_share_for_mtp_iteration=True,
    )
    spec = checkpoint_module.Glm5WeightSpec(cfg)
    tensor = torch.ones(1)

    assert spec.native_to_hf(
        "layers.2.self_attention.self_attention.indexer.wq_b.weight", tensor
    ) == [("model.layers.2.self_attn.indexer.wq_b.weight", tensor)]
    assert (
        spec.native_to_hf("layers.3.self_attention.self_attention.indexer.wq_b.weight", tensor)
        == []
    )
    assert spec.native_to_hf(
        "mtp.layers.0.transformer_layer.self_attention.self_attention.indexer.wq_b.weight",
        tensor,
    ) == [("model.layers.6.self_attn.indexer.wq_b.weight", tensor)]

    base_names = {
        "model.layers.3.self_attn.q_a_proj.weight",
        "model.layers.3.self_attn.q_a_layernorm.weight",
        "model.layers.3.self_attn.q_b_proj.weight",
        "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.3.self_attn.kv_a_layernorm.weight",
        "model.layers.3.self_attn.kv_b_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
    }

    class Reader:
        index = {name: "model.safetensors" for name in base_names}

        def get_tensor(self, name):
            if name not in self.index:
                raise KeyError(name)
            return torch.ones(1)

    out = {}
    checkpoint_module._load_attention(
        out,
        local_prefix="layers.3",
        hf_prefix="model.layers.3.self_attn",
        reader=Reader(),
        ps=object(),
        load_indexer=False,
    )
    assert "layers.3.self_attention.self_attention.q_a_proj.weight" in out
    assert not any(".indexer." in name for name in out)


def test_glm52_checkpoint_skips_shared_indexer_weights_and_loads_full_layers(tmp_path):
    import pytest
    import torch

    try:
        import transformer_engine.pytorch  # noqa: F401
    except (ModuleNotFoundError, OSError) as exc:
        pytest.skip(f"Transformer Engine is not importable in this environment: {exc}")

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import export_hf_weights, load_hf_weights
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 6,
            "num_nextn_predict_layers": 1,
        },
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=_glm52_indexer_types(num_layers=6),
        index_share_for_mtp_iteration=True,
    )
    ps = ParallelState()
    model = _make_glm5_model(cfg, ps=ps, mtp_enable=True)
    state = model.state_dict()

    exported = dict(export_hf_weights(model, cfg, ps))
    assert "model.layers.2.self_attn.indexer.wq_b.weight" in exported
    assert "model.layers.3.self_attn.indexer.wq_b.weight" not in exported
    assert "model.layers.6.self_attn.indexer.wq_b.weight" in exported

    save_safetensors(exported, str(tmp_path))
    loaded = _make_glm5_model(cfg, ps=ps, mtp_enable=True)
    load_hf_weights(loaded, str(tmp_path), cfg, ps)
    loaded_state = loaded.state_dict()

    assert torch.equal(
        loaded_state["layers.2.self_attention.self_attention.indexer.wq_b.weight"],
        state["layers.2.self_attention.self_attention.indexer.wq_b.weight"],
    )
    assert "layers.3.self_attention.self_attention.indexer.wq_b.weight" not in loaded_state
    assert torch.equal(
        loaded_state[
            "mtp.layers.0.transformer_layer.self_attention.self_attention.indexer.wq_b.weight"
        ],
        state["mtp.layers.0.transformer_layer.self_attention.self_attention.indexer.wq_b.weight"],
    )


def test_glm5_router_modules_use_current_names_and_bias_buffers():
    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5SigmoidTopKRouter

    model = _make_glm5_model(
        Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1), mtp_enable=True
    )
    routers = [module for module in model.modules() if isinstance(module, Glm5SigmoidTopKRouter)]
    assert len(routers) == 2
    for router in routers:
        assert hasattr(router, "gate")
        assert hasattr(router, "expert_bias")
        assert torch.isfinite(router.gate.weight).all()
        assert torch.equal(router.expert_bias, torch.zeros_like(router.expert_bias))


def test_glm5_protocol_allows_cp_only_parallel_scope():
    import pytest

    from megatron.lite.model.glm5.lite.protocol import _validate_parallel_scope
    from megatron.lite.runtime.contracts import ParallelConfig

    # CP-only as well as PP/VPP/EP are supported and must validate cleanly.
    _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=1, cp=2, pp=1, vpp=1))
    _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=1, cp=1, pp=2, vpp=2))
    # GLM-5 native DSA attention rejects tensor / expert-tensor parallelism.
    with pytest.raises(NotImplementedError):
        _validate_parallel_scope(ParallelConfig(tp=2, ep=1, etp=1, cp=1, pp=1, vpp=1))
    with pytest.raises(NotImplementedError):
        _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=2, cp=1, pp=1, vpp=1))


def test_glm5_impl_config_accepts_runtime_mtp_fields():
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.protocol import ImplConfig

    cfg = Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1)

    assert ImplConfig(mtp_enable=False, mtp_enable_train=False).mtp_enable is False
    assert ImplConfig(mtp_enable=True, mtp_enable_train=True).mtp_enable_train is True
    assert cfg.num_nextn_predict_layers == 1


def test_glm5_dsa_execution_policy_lives_in_impl_config_only(
    transformer_engine_import_stub,
):
    from dataclasses import fields

    transformer_engine_import_stub()
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.protocol import ImplConfig

    architecture_fields = {field.name for field in fields(Glm5Config)}

    assert {
        "dsa_cp_mode",
        "dsa_indexer_loss_coeff",
        "dsa_indexer_use_sparse_loss",
        "calculate_per_token_loss",
    }.isdisjoint(architecture_fields)
    assert ImplConfig().dsa_cp_mode == "native"
    assert ImplConfig().dsa_indexer_loss_coeff == 0.0
    assert ImplConfig().dsa_indexer_use_sparse_loss is False
    assert ImplConfig().calculate_per_token_loss is False
    assert ImplConfig(dsa_cp_mode="legacy_gather_all").dsa_cp_mode == "legacy_gather_all"


def test_glm5_production_cp_path_has_no_zigzag_layout():
    lite_root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    model_text = (lite_root / "model" / "glm5" / "lite" / "model.py").read_text()
    dsa_text = (lite_root / "primitive" / "modules" / "attention" / "dsa.py").read_text()

    assert "zigzag" not in model_text
    assert "zigzag" not in dsa_text


def test_glm5_attention_receives_impl_dsa_policy(monkeypatch, transformer_engine_import_stub):
    import torch.nn as nn

    transformer_engine_import_stub()
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import model as model_module
    from megatron.lite.primitive.parallel import ParallelState

    captured = {}

    class FakeDSA(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(model_module, "DynamicSparseAttention", FakeDSA)
    model_module.Glm5DSAAttention(
        Glm5Config(**_tiny_config_kwargs()),
        ParallelState(cp_size=2, cp_rank=0),
        0,
        dsa_cp_mode="legacy_gather_all",
        dsa_indexer_loss_coeff=0.25,
        dsa_indexer_use_sparse_loss=True,
        calculate_per_token_loss=True,
    )

    assert captured["cp_mode"] == "legacy_gather_all"
    assert captured["indexer_loss_coeff"] == 0.25
    assert captured["indexer_use_sparse_loss"] is True
    assert captured["calculate_per_token_loss"] is True


def test_glm5_protocol_uses_mlite_optimizer_api():
    from megatron.lite.model.glm5.lite.protocol import ImplConfig

    protocol_path = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "glm5"
        / "lite"
        / "protocol.py"
    )
    protocol_text = protocol_path.read_text()

    assert ImplConfig().optimizer == "dist_opt"
    assert "build_dist_opt_training_optimizer" in protocol_text


def test_glm5_lite_tiny_forward_backward(monkeypatch):
    import pytest
    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.primitive.modules.attention import dsa

    if not torch.cuda.is_available():
        pytest.skip("GLM-5 native model requires CUDA (Transformer Engine RMSNorm)")
    device = torch.device("cuda")

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        kv_offset=0,
        calculate_per_token_loss=False,
        value_dim=None,
    ):
        del (
            kv_full,
            attn_sink,
            window_idxs,
            q_indexer,
            k_indexer,
            weights,
            indexer_topk,
            ratio,
            softmax_scale,
            indexer_softmax_scale,
            loss_coeff,
            sparse_loss,
            kv_offset,
            calculate_per_token_loss,
        )
        return query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        ), torch.zeros((), device=query.device, dtype=torch.float32)

    monkeypatch.setattr(
        dsa._dsa_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    torch.manual_seed(1234)
    model = _make_glm5_model(Glm5Config(**_tiny_config_kwargs())).to(
        device=device, dtype=torch.bfloat16
    )
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5), device=device)
    labels = torch.randint(0, model.config.vocab_size, (2, 5), device=device)

    output = model(input_ids=input_ids, labels=labels)

    # hidden_states stays in (seq, batch, hidden) layout; logits are transposed
    # back to (batch, seq, *) inside forward.
    assert output["hidden_states"].shape == (5, 2, model.config.hidden_size)
    assert output["loss"].ndim == 0
    output["loss"].backward()
    grad_norm = sum(
        param.grad.detach().float().norm() for param in model.parameters() if param.grad is not None
    )
    assert torch.isfinite(grad_norm)

    mtp_model = _make_glm5_model(
        Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1),
        mtp_enable=True,
        mtp_enable_train=True,
    ).to(device=device, dtype=torch.bfloat16)
    # Inference path (no labels) exposes the per-MTP-head logits.
    mtp_infer = mtp_model(input_ids=input_ids)
    assert len(mtp_infer["mtp_hidden_states"]) == 1
    assert len(mtp_infer["mtp_logits"]) == 1
    assert mtp_infer["mtp_hidden_states"][0].shape == (5, 2, mtp_model.config.hidden_size)
    assert mtp_infer["mtp_logits"][0].shape == (2, 5, mtp_model.config.vocab_size)

    # Training path (with labels) returns the MTP loss instead of logits.
    mtp_output = mtp_model(
        input_ids=input_ids, labels=labels, loss_mask=torch.ones_like(labels, dtype=torch.float32)
    )

    assert len(mtp_output["mtp_hidden_states"]) == 1
    assert mtp_output["mtp_hidden_states"][0].shape == (5, 2, mtp_model.config.hidden_size)
    assert mtp_output["mtp_loss"].ndim == 0
    mtp_output["loss"].backward()
    mtp_grad_norm = sum(
        param.grad.detach().float().norm()
        for param in mtp_model.parameters()
        if param.grad is not None
    )
    assert torch.isfinite(mtp_grad_norm)
