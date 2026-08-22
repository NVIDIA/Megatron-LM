# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 HF load must lift LOCAL pipeline layer indices to GLOBAL.

Under PP the model's ``self.layers`` ModuleDict is keyed by LOCAL pipeline
position, so a non-first stage's native ``state_dict`` keys carry local indices
(``layers.0`` ...). The HF release is keyed by GLOBAL layer index, so -- exactly
like the exporter -- ``load_hf_weights`` must map local->global via
``to_global_layer_name(name, layer_map)`` before resolving HF names. Without it a
non-first stage reads the wrong global layer's weights.

This is a CPU unit test: a minimal stand-in stage (no GPU/TE) whose layers are
keyed locally but carry ``layer_indices`` = the global ids it owns, plus a tiny
on-disk safetensors keyed by GLOBAL names. ``load_hf_weights`` must copy each
local layer the GLOBAL layer's tensor; pre-fix it resolves local names, finds
nothing, and leaves the params untouched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


def _checkpoint_module():
    checkpoint_path = (
        Path(__file__).resolve().parents[5]
        / "megatron"
        / "lite"
        / "model"
        / "deepseek_v4"
        / "lite"
        / "checkpoint.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "_deepseek_v4_checkpoint_test", checkpoint_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class _LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_layernorm = _LayerNorm(dim)


class _QATBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.self_attn = nn.Module()
        self.self_attn.self_attn.wq_a = nn.Linear(dim, dim, bias=False)


class _Stage(nn.Module):
    """A non-first PP stage: layers keyed by LOCAL position, ``layer_indices``
    gives the GLOBAL ids it owns (e.g. [4, 5] for the 3rd stage of pp)."""

    def __init__(self, global_ids: list[int], dim: int):
        super().__init__()
        self.layer_indices = list(global_ids)
        self.layers = nn.ModuleDict(
            {str(i): _Block(dim) for i in range(len(global_ids))}
        )


class _QATStage(nn.Module):
    def __init__(self, global_ids: list[int], dim: int):
        super().__init__()
        self.layer_indices = list(global_ids)
        self.layers = nn.ModuleDict(
            {str(i): _QATBlock(dim) for i in range(len(global_ids))}
        )


def test_ds4_dynamic_load_map_has_no_blanket_optional_escape_hatch():
    ckpt = _checkpoint_module()
    assert not hasattr(ckpt.DeepseekV4WeightSpec, "optional_for_load")


def test_ds4_load_hf_resolves_local_pp_layer_to_global(tmp_path):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from safetensors.torch import save_file

    ckpt = _checkpoint_module()

    dim = 4
    global_ids = [4, 5]  # this stage owns global layers 4 and 5, keyed local 0 and 1
    model = _Stage(global_ids, dim)
    cfg = DeepseekV4Config(num_hidden_layers=8, n_routed_experts=8)
    ps = SimpleNamespace(tp_size=1, etp_size=1, ep_size=1, ep_rank=0)

    # Real-release layout is keyed by GLOBAL layer index; input_layernorm maps to
    # the bare V4-Flash ``attn_norm.weight``.
    save_file(
        {
            f"layers.{g}.attn_norm.weight": torch.full((dim,), float(g))
            for g in global_ids
        },
        str(tmp_path / "model.safetensors"),
    )

    ckpt.load_hf_weights(model, str(tmp_path), cfg, ps)

    # local layer 0 -> global 4 -> filled with 4.0; local 1 -> global 5 -> 5.0.
    # Pre-fix, load resolved layers.0/layers.1 (local), found nothing, left zeros.
    torch.testing.assert_close(
        model.layers["0"].input_layernorm.weight.detach(), torch.full((dim,), 4.0)
    )
    torch.testing.assert_close(
        model.layers["1"].input_layernorm.weight.detach(), torch.full((dim,), 5.0)
    )


def test_ds4_load_hf_canonicalizes_qat_state_before_dynamic_mapping(tmp_path):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.primitive.quantization.qat import QATSpec, WeightFakeQuant
    from safetensors.torch import save_file

    ckpt = _checkpoint_module()
    dim = 4
    model = _QATStage([4], dim)
    linear = model.layers["0"].self_attn.self_attn.wq_a
    parametrize.register_parametrization(
        linear,
        "weight",
        WeightFakeQuant(
            QATSpec(enabled=True, format="int8", group_size=-1),
            linear.weight.shape,
        ),
    )
    master = linear.parametrizations.weight.original
    master.data.zero_()
    assert any(".parametrizations.weight.0.amax" in name for name in model.state_dict())

    expected = torch.arange(dim * dim, dtype=torch.float32).reshape(dim, dim)
    save_file(
        {"layers.4.attn.wq_a.weight": expected}, str(tmp_path / "model.safetensors")
    )

    cfg = DeepseekV4Config(num_hidden_layers=8, n_routed_experts=8)
    ps = SimpleNamespace(tp_size=1, etp_size=1, ep_size=1, ep_rank=0)
    ckpt.load_hf_weights(model, str(tmp_path), cfg, ps)

    torch.testing.assert_close(master, expected)


def test_ds4_shared_checkpoint_preserves_reversible_fp8_source_scale(tmp_path):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from safetensors.torch import save_file

    ckpt = _checkpoint_module()
    model = _QATStage([0], 128)
    qweight = torch.randn(128, 128).clamp(-4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.25]], dtype=torch.float32)
    shard = "model-00001-of-00001.safetensors"
    weight_name = "layers.0.attn.wq_a.weight"
    scale_name = "layers.0.attn.wq_a.scale"
    save_file({weight_name: qweight, scale_name: scale}, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_name: shard, scale_name: shard}})
    )

    cfg = DeepseekV4Config(num_hidden_layers=1, n_routed_experts=8)
    ps = SimpleNamespace(
        tp_size=1,
        etp_size=1,
        ep_size=1,
        ep_rank=0,
        dp_cp_group=None,
    )
    ckpt.load_hf_weights(model, str(tmp_path), cfg, ps)

    master = model.layers["0"].self_attn.self_attn.wq_a.weight
    restored = ckpt.requantize_block_fp8_weight(
        master.detach().to(torch.bfloat16), master._fp8_source_scales
    )
    assert torch.equal(restored.qweight, qweight)
    assert master._fp8_source_scale_version == master._version
    assert torch.equal(
        model.layers["0"].self_attn.self_attn.wq_a._fp8_source_scales_by_parameter[
            "weight"
        ],
        scale,
    )
    assert torch.equal(
        model._fp8_source_scales_by_name[
            "layers.0.self_attn.self_attn.wq_a.weight"
        ],
        scale,
    )


def _mock_dense_replica_receiver(monkeypatch, master, source_scale):
    from megatron.lite.primitive.ckpt import hf_weights

    broadcasts = 0

    def fake_broadcast(tensor, *, src, group):
        nonlocal broadcasts
        assert src == 0
        assert group is not None
        broadcasts += 1
        if broadcasts == 1:
            tensor.copy_(master)
        elif broadcasts == 2:
            tensor.fill_(source_scale is not None)
        elif broadcasts == 3 and source_scale is not None:
            tensor.copy_(source_scale)
        else:
            raise AssertionError(f"unexpected replica broadcast #{broadcasts}")

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(hf_weights.dist, "get_world_size", lambda _group: 4)
    monkeypatch.setattr(
        hf_weights.dist, "get_process_group_ranks", lambda _group: [0, 1, 2, 3]
    )
    monkeypatch.setattr(hf_weights.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(hf_weights.dist, "broadcast", fake_broadcast)
    return lambda: broadcasts


def _dense_ep4_parallel_state():
    return SimpleNamespace(
        tp_size=1,
        etp_size=1,
        ep_size=4,
        ep_rank=1,
        dp_cp_group=object(),
        ep_dp_group=None,
    )


def test_ds4_native_fp8_dense_replica_receives_source_scale(tmp_path, monkeypatch):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from safetensors.torch import save_file

    ckpt = _checkpoint_module()
    qweight = torch.randn(128, 128).clamp(-4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.25]], dtype=torch.float32)
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    master = (qweight.float() * expanded).to(torch.bfloat16)
    weight_name = "layers.0.attn.wq_a.weight"
    scale_name = "layers.0.attn.wq_a.scale"
    shard = "model-00001-of-00001.safetensors"
    save_file({weight_name: qweight, scale_name: scale}, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_name: shard, scale_name: shard}})
    )
    broadcast_count = _mock_dense_replica_receiver(monkeypatch, master, scale)
    model = _QATStage([0], 128)
    cfg = DeepseekV4Config(num_hidden_layers=1, n_routed_experts=8)

    ckpt.load_hf_weights(model, str(tmp_path), cfg, _dense_ep4_parallel_state())

    parameter = model.layers["0"].self_attn.self_attn.wq_a.weight
    assert torch.equal(parameter, master)
    assert torch.equal(parameter._fp8_source_scales, scale)
    assert parameter._fp8_source_scale_version == parameter._version
    restored = ckpt.requantize_block_fp8_weight(
        parameter.detach().to(torch.bfloat16), scale
    )
    assert torch.equal(restored.qweight, qweight)
    assert broadcast_count() == 3


def test_ds4_mxfp4_dense_replica_receives_identical_bf16_master(tmp_path, monkeypatch):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4
    from safetensors.torch import save_file

    ckpt = _checkpoint_module()
    source = torch.linspace(-3, 3, 128 * 128).reshape(128, 128)
    packed, scale = quantize_mxfp4(source)
    native_name = "layers.0.self_attn.self_attn.wq_a.weight"
    cfg = DeepseekV4Config(num_hidden_layers=1, n_routed_experts=8)
    source_spec = ckpt.DeepseekV4WeightSpec(cfg, source_block_fp8=True)
    master = source_spec.hf_to_native(native_name, [packed, scale])
    weight_name = "layers.0.attn.wq_a.weight"
    scale_name = "layers.0.attn.wq_a.scale"
    shard = "model-00001-of-00001.safetensors"
    save_file({weight_name: packed, scale_name: scale}, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_name: shard, scale_name: shard}})
    )
    broadcast_count = _mock_dense_replica_receiver(monkeypatch, master, None)
    model = _QATStage([0], 128)

    ckpt.load_hf_weights(model, str(tmp_path), cfg, _dense_ep4_parallel_state())

    parameter = model.layers["0"].self_attn.self_attn.wq_a.weight
    assert torch.equal(parameter, master)
    assert not hasattr(parameter, "_fp8_source_scales")
    assert broadcast_count() == 2


def test_ds4_source_scales_bind_global_expert_to_ep_local_parameter():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    ckpt = _checkpoint_module()
    model = nn.Module()
    model.layer_indices = [2]
    model.layers = nn.ModuleDict({"0": nn.Module()})
    layer = model.layers["0"]
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    layer.mlp.experts.fc1 = nn.Module()
    layer.mlp.experts.fc1.register_parameter(
        "weight0", nn.Parameter(torch.zeros(128, 128, dtype=torch.bfloat16))
    )
    cfg = DeepseekV4Config(num_hidden_layers=1, n_routed_experts=256)
    spec = ckpt.DeepseekV4WeightSpec(cfg, source_block_fp8=True)
    scale = torch.ones(1, 1, dtype=torch.float32)
    # The shared loader has already remapped the PP-stage layer to its local
    # index, while the expert suffix remains global until EP binding.
    spec.source_block_scales["layers.0.mlp.experts.fc1.weight128"] = scale
    ps = SimpleNamespace(ep_size=2, ep_rank=1)

    spec.bind_source_scales(model, ps)

    parameter = layer.mlp.experts.fc1.weight0
    assert torch.equal(parameter._fp8_source_scales, scale)
    assert torch.equal(
        layer.mlp.experts.fc1._fp8_source_scales_by_parameter["weight0"], scale
    )
    assert "layers.2.mlp.experts.fc1.weight128" in model._fp8_source_scales_by_name


def test_ds4_source_scale_export_gathers_remote_ep_experts(monkeypatch):
    ckpt = _checkpoint_module()
    local = {"layers.0.ffn.experts.0.w1.weight": torch.tensor([[1.0]])}
    remote = {"layers.0.ffn.experts.128.w1.weight": torch.tensor([[2.0]])}
    group = object()

    monkeypatch.setattr(ckpt.dist, "is_initialized", lambda: True)

    def fake_all_gather_object(output, value, *, group):
        output[:] = [value, remote]

    monkeypatch.setattr(ckpt.dist, "all_gather_object", fake_all_gather_object)

    combined = ckpt._gather_source_scale_registry(
        local, size=2, group=group, parallelism="EP"
    )

    assert set(combined) == set(local) | set(remote)
    assert torch.equal(combined[next(iter(remote))], next(iter(remote.values())))


def test_ds4_export_streams_router_buffers_from_every_pp_stage(monkeypatch):
    from megatron.lite.primitive.ckpt import hf_weights

    ckpt = _checkpoint_module()

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("expert_bias", torch.tensor([0.25, -0.5]))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.gate = Gate()

    class StageOne(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_indices = [1]
            self.layers = nn.ModuleDict({"0": Layer()})

    remote = torch.tensor([3, 1, 4], dtype=torch.int64)
    remote_headers = iter(
        [
            [
                (
                    "layers.0.mlp.gate.tid2eid",
                    tuple(remote.shape),
                    remote.dtype,
                )
            ],
            [],
        ]
    )

    def fake_broadcast_object_list(header, *, src, **_kwargs):
        if src == 0:
            header[0] = next(remote_headers)

    def fake_broadcast(tensor, *, src, **_kwargs):
        if src == 0:
            tensor.copy_(remote)

    monkeypatch.setattr(hf_weights.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        hf_weights.dist, "broadcast_object_list", fake_broadcast_object_list
    )
    monkeypatch.setattr(hf_weights.dist, "broadcast", fake_broadcast)

    ps = SimpleNamespace(
        pp_size=2,
        pp_rank=1,
        pp_global_ranks=[0, 1],
        pp_group=object(),
        tp_size=1,
        tp_group=None,
        ep_size=1,
        ep_group=None,
        etp_size=1,
        etp_group=None,
    )
    cfg = SimpleNamespace(num_hash_layers=1, n_routed_experts=1, vocab_size=8)

    exported = dict(ckpt._export_unquantized_weights(StageOne(), cfg, ps))

    assert torch.equal(
        exported["layers.0.ffn.gate.tid2eid"],
        remote.to(exported["layers.0.ffn.gate.tid2eid"].device),
    )
    assert torch.equal(
        exported["layers.1.ffn.gate.bias"],
        torch.tensor([0.25, -0.5], device=exported["layers.1.ffn.gate.bias"].device),
    )
