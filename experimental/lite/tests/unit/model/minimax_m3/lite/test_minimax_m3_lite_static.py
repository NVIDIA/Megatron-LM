# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static and CPU contracts of the native MiniMax-M3 lite implementation."""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_LITE = Path(__file__).resolve().parents[5]


def test_registry_resolves_lite():
    from megatron.lite.model.registry import (
        get_train_runtime_module,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    runtime_name = resolve_runtime_model_name("minimax_m3", "lite")
    assert runtime_name == "minimax_m3"
    assert get_train_runtime_module(runtime_name).__name__ == "megatron.lite.model.minimax_m3.lite.protocol"
    for model_type in ("minimax_m3_vl", "minimax_m3_vl_text", "minimax_m3"):
        assert resolve_model_type_from_hf({"model_type": model_type}) == "minimax_m3"


def test_lite_does_not_import_sibling_models_or_megatron_core():
    root = _LITE / "megatron" / "lite" / "model" / "minimax_m3"
    for path in root.rglob("*.py"):
        text = path.read_text()
        for forbidden in ("megatron.lite.model.qwen", "megatron.lite.model.glm5", "megatron.lite.model.kimi",
                          "megatron.lite.model.deepseek", "mbridge", "megatron.core"):
            assert forbidden not in text, (path, forbidden)


def test_config_maps_hf_text_config_strictly(tiny_hf_kwargs):
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    cfg = MiniMaxM3Config._from_hf_dict(tiny_hf_kwargs)  # strict: raises on unmapped fields
    assert cfg.num_hidden_layers == 4 and cfg.hidden_size == 256 and cfg.vocab_size == 256
    assert cfg.layer_types == ["full_attention"] * 2 + ["minimax_m3_sparse"] * 2
    assert cfg.mlp_layer_types == ["dense"] * 2 + ["sparse"] * 2
    assert cfg.num_experts == 4 and cfg.num_experts_per_tok == 2 and cfg.routed_scaling_factor == 2.0
    assert cfg.index_n_heads == 2 and cfg.index_head_dim == 128 and cfg.index_topk_blocks == 4
    assert cfg.index_block_size == 128 and cfg.index_local_blocks == 1
    assert cfg.rotary_dim == 64 and cfg.partial_rotary_factor == 0.5 and cfg.rope_theta == 5e6
    assert (cfg.swiglu_alpha, cfg.swiglu_limit, cfg.swiglu_up_offset) == (1.702, 7.0, 1.0)
    assert not cfg.tie_word_embeddings
    assert [cfg.is_sparse_attention_layer(i) for i in range(4)] == [False, False, True, True]
    assert [cfg.is_moe_layer(i) for i in range(4)] == [False, False, True, True]


@pytest.mark.optional
def test_config_from_hf_object_matches_dict_mapping(tiny_hf_kwargs):
    pytest.importorskip("transformers.models.minimax_m3_vl")
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    hf_cfg = MiniMaxM3VLTextConfig(**{k: v for k, v in tiny_hf_kwargs.items() if k != "model_type"})
    from_dict = MiniMaxM3Config._from_hf_dict(tiny_hf_kwargs)
    from_obj = MiniMaxM3Config.from_hf_config(hf_cfg, strict=False)
    assert from_obj.layer_types == from_dict.layer_types
    assert from_obj.mlp_layer_types == from_dict.mlp_layer_types
    assert from_obj.index_topk_blocks == from_dict.index_topk_blocks


def test_impl_config_defaults_to_the_magi_production_path():
    from megatron.lite.model.minimax_m3.lite import protocol as P

    cfg = P.ImplConfig()
    assert "msa_backend" not in {f.name for f in fields(P.ImplConfig)}
    assert cfg.deterministic is False
    assert cfg.use_thd is False
    assert (cfg.magi_chunk_size, cfg.magi_high_precision_reduce, cfg.magi_dense_kernel_backend) == (2048, False, "fa4")
    assert cfg.optimizer == "dist_opt"


def test_protocol_rejects_tensor_parallel_thd_and_vpp():
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(NotImplementedError, match="tp=1"):
        P.build_model(MiniMaxM3Config(), impl_cfg=P.ImplConfig(parallel=ParallelConfig(tp=2), optimizer=None))
    with pytest.raises(NotImplementedError, match="use_thd"):
        P.build_model(MiniMaxM3Config(), impl_cfg=P.ImplConfig(use_thd=True, optimizer=None))
    with pytest.raises(NotImplementedError, match="vpp=1"):
        P.build_model(MiniMaxM3Config(), impl_cfg=P.ImplConfig(parallel=ParallelConfig(pp=2, vpp=2), optimizer=None))


def test_protocol_builds_the_magi_backend(monkeypatch):
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
    from megatron.lite.model.minimax_m3.lite import protocol as P

    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, _cfg, _train_cfg, _ps, *, msa_backend, **_kwargs):
            super().__init__()
            seen["msa_backend"] = msa_backend
            self.layers = torch.nn.ModuleList()

        def to(self, *_args, **_kwargs):
            return self

        def cuda(self, *_args, **_kwargs):
            return self

    ps = SimpleNamespace(tp_size=1, ep_size=1, etp_size=1, pp_size=1, cp_size=1)
    fake_model_module = ModuleType("megatron.lite.model.minimax_m3.lite.model")
    fake_model_module.MiniMaxM3Model = FakeModel
    monkeypatch.setitem(sys.modules, fake_model_module.__name__, fake_model_module)
    monkeypatch.setattr(P, "init_parallel", lambda _cfg: ps)
    monkeypatch.setattr(P.magi_msa, "validate_device", lambda: None)
    monkeypatch.setattr(P.magi_msa, "validate_kernel_shapes", lambda **_kwargs: None)
    monkeypatch.setattr(P.magi_msa, "build_msa_config", lambda *_args, **_kwargs: object())

    bundle = P.build_model(MiniMaxM3Config(), impl_cfg=P.ImplConfig(optimizer=None))
    assert seen["msa_backend"] == "magi"
    assert bundle.forward_step is P._forward_step
    assert "magi_settings" not in bundle.extras
    assert bundle.chunks[0].magi_settings.deterministic is False


def test_forward_step_pads_and_dispatches_the_packed_batch(monkeypatch):
    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.runtime.contracts.data import PackedBatch

    ctx = SimpleNamespace(pad=3, cu_seqlens_host=(0, 2, 5, 8))
    monkeypatch.setattr(P, "_magi_plan", lambda _model, _batch: ctx)
    monkeypatch.setattr(P.magi_msa, "dispatch_tokens", lambda x, _ctx: x)
    monkeypatch.setattr(P.magi_msa, "undispatch_tokens", lambda x, _ctx: x)

    class Recorder(torch.nn.Module):
        cross_entropy_fusion = False

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return kwargs["input_ids"].unsqueeze(-1)

    model = Recorder()
    batch = PackedBatch(
        input_ids=torch.tensor([10, 11, 20, 21, 22]),
        labels=torch.tensor([11, 12, 21, 22, 23]),
        loss_mask=torch.tensor([1, 1, 1, 0, 1], dtype=torch.float32),
        seq_lens=torch.tensor([2, 3]),
    )
    output = P._forward_step(model, batch)
    assert output.shape == (1, 8, 1)
    assert model.kwargs["input_ids"].tolist() == [[10, 11, 20, 21, 22, 0, 0, 0]]
    assert model.kwargs["labels"].tolist() == [[11, 12, 21, 22, 23, 0, 0, 0]]
    assert model.kwargs["loss_mask"].tolist() == [[1, 1, 1, 0, 1, 0, 0, 0]]  # pad tokens never count
    assert model.kwargs["magi_ctx"] is ctx

    unpacked = P.unpack_forward_output(model, batch, output)
    assert [d.squeeze(-1).tolist() for d in unpacked.unbind()] == [[10, 11], [20, 21, 22]]


def test_forward_step_rejects_inconsistent_packed_batches():
    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.runtime.contracts.data import PackedBatch

    with pytest.raises(ValueError, match="seq_lens sums to 4"):
        P._validate_packed_batch(PackedBatch(input_ids=torch.tensor([10, 11, 20]), labels=torch.tensor([11, 12, 21]), seq_lens=torch.tensor([2, 2])))
    with pytest.raises(ValueError, match="document-local position_ids"):
        P._validate_packed_batch(
            PackedBatch(input_ids=torch.tensor([10, 11, 20, 21]), labels=torch.tensor([11, 12, 21, 22]),
                        seq_lens=torch.tensor([2, 2]), position_ids=torch.tensor([0, 1, 2, 3]))
        )
    P._validate_packed_batch(
        PackedBatch(input_ids=torch.tensor([10, 11, 20, 21]), labels=torch.tensor([11, 12, 21, 22]),
                    seq_lens=torch.tensor([2, 2]), position_ids=torch.tensor([0, 1, 0, 1]))
    )


def test_weight_spec_maps_native_names_to_hf_disk_names(tiny_hf_kwargs):
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
    from megatron.lite.model.minimax_m3.lite.checkpoint import (
        MiniMaxM3WeightSpec,
        disk_to_module_name,
    )

    cfg = MiniMaxM3Config._from_hf_dict(tiny_hf_kwargs)
    spec = MiniMaxM3WeightSpec(cfg)
    prefix = cfg.hf_text_prefix
    cases = {
        "embed.embedding.weight": [f"{prefix}.embed_tokens.weight"],
        "layers.2.attn.indexer.k_norm.weight": [f"{prefix}.layers.2.self_attn.index_k_norm.weight"],
        "layers.2.moe.router.gate.weight": [f"{prefix}.layers.2.block_sparse_moe.gate.weight"],
        "layers.2.moe.experts.fc1.weight3": [
            f"{prefix}.layers.2.block_sparse_moe.experts.3.w1.weight",
            f"{prefix}.layers.2.block_sparse_moe.experts.3.w3.weight",
        ],
        "layers.0.mlp.gate_up.linear.weight": [f"{prefix}.layers.0.mlp.gate_proj.weight", f"{prefix}.layers.0.mlp.up_proj.weight"],
        "norm.weight": [f"{prefix}.norm.weight"],
        "head.col.linear.weight": [f"{cfg.hf_head_name}.weight"],
    }
    for native, disk_names in cases.items():
        names = [hf_name for hf_name, _ in spec.native_to_hf(native, torch.zeros(4, 2))]
        assert names == disk_names, (native, names)
        assert spec.weight_map()[native] == disk_names, native
    assert "layers.0.attn.indexer.k_norm.weight" not in spec.weight_map()  # dense layers carry no indexer
    assert disk_to_module_name("language_model.model.layers.2.self_attn.index_k_norm.weight") == "model.layers.2.self_attn.indexer.k_norm.weight"
    assert disk_to_module_name("language_model.model.layers.2.block_sparse_moe.e_score_correction_bias") == "model.layers.2.mlp.gate.e_score_correction_bias"
