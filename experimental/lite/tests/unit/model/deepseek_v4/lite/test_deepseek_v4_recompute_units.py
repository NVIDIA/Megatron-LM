# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.lite.checkpoint import _map_block_attr
from megatron.lite.model.deepseek_v4.lite.protocol import (
    _cast_training_parameters,
    _is_native_fp32_control,
    _iter_transformer_units,
)


class _NativeChunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": nn.Linear(2, 2), "1": nn.Linear(2, 2)})
        self.mtp = nn.ModuleList([nn.Linear(2, 2)])


def test_iter_transformer_units_accepts_native_ds4_chunk() -> None:
    chunk = _NativeChunk()

    assert _iter_transformer_units(chunk) == [*chunk.layers.values(), *chunk.mtp]


def test_iter_transformer_units_accepts_wrapper_chunk() -> None:
    native = _NativeChunk()
    wrapper = nn.Module()
    wrapper.model = native

    assert _iter_transformer_units(wrapper) == [*native.layers.values(), *native.mtp]


def test_training_cast_preserves_native_fp32_controls() -> None:
    model = nn.Module()
    model.layers = nn.ModuleDict({"0": nn.Module()})
    layer = model.layers["0"]
    layer.self_attn = nn.Module()
    layer.self_attn.self_attn = nn.Module()
    layer.self_attn.self_attn.sinks = nn.Parameter(
        torch.tensor([1.0001, -0.5003], dtype=torch.float32)
    )
    layer.attn_hc = nn.Module()
    layer.attn_hc.fn = nn.Parameter(torch.tensor([1.0001], dtype=torch.float32))
    layer.attn_hc.base = nn.Parameter(torch.tensor([-0.5003], dtype=torch.float32))
    layer.attn_hc.scale = nn.Parameter(torch.tensor([0.7501], dtype=torch.float32))
    layer.ffn_hc = nn.Module()
    layer.ffn_hc.fn = nn.Parameter(torch.tensor([1.0001], dtype=torch.float32))
    layer.ffn_hc.base = nn.Parameter(torch.tensor([-0.5003], dtype=torch.float32))
    layer.ffn_hc.scale = nn.Parameter(torch.tensor([0.7501], dtype=torch.float32))
    model.hc_head = nn.Module()
    model.hc_head.hc_fn = nn.Parameter(torch.tensor([1.0001], dtype=torch.float32))
    model.hc_head.hc_base = nn.Parameter(torch.tensor([-0.5003], dtype=torch.float32))
    model.hc_head.hc_scale = nn.Parameter(torch.tensor([0.7501], dtype=torch.float32))
    model.mtp = nn.ModuleList([nn.Module()])
    model.mtp[0].hc_head = nn.Module()
    model.mtp[0].hc_head.hc_fn = nn.Parameter(torch.tensor([1.0001], dtype=torch.float32))
    model.mtp[0].hc_head.hc_base = nn.Parameter(torch.tensor([-0.5003], dtype=torch.float32))
    model.mtp[0].hc_head.hc_scale = nn.Parameter(torch.tensor([0.7501], dtype=torch.float32))
    layer.self_attn.self_attn.compressor = nn.Module()
    layer.self_attn.self_attn.compressor.ape = nn.Parameter(
        torch.tensor([[1.0001]], dtype=torch.float32)
    )
    layer.self_attn.self_attn.indexer = nn.Module()
    layer.self_attn.self_attn.indexer.compressor = nn.Module()
    layer.self_attn.self_attn.indexer.compressor.ape = nn.Parameter(
        torch.tensor([[-0.5003]], dtype=torch.float32)
    )
    layer.mlp = nn.Module()
    layer.mlp.gate = nn.Module()
    layer.mlp.gate.register_buffer(
        "expert_bias", torch.tensor([1.0001, -0.5003], dtype=torch.float32), persistent=True
    )
    layer.proj = nn.Linear(2, 2, bias=False)
    expected_sink = layer.self_attn.self_attn.sinks.detach().clone()
    expected_controls = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if _is_native_fp32_control(name)
    }

    _cast_training_parameters(model)

    actual_parameters = dict(model.named_parameters())
    assert set(expected_controls) == {
        "layers.0.self_attn.self_attn.sinks",
        "layers.0.self_attn.self_attn.compressor.ape",
        "layers.0.self_attn.self_attn.indexer.compressor.ape",
        "layers.0.attn_hc.fn",
        "layers.0.attn_hc.base",
        "layers.0.attn_hc.scale",
        "layers.0.ffn_hc.fn",
        "layers.0.ffn_hc.base",
        "layers.0.ffn_hc.scale",
        "hc_head.hc_fn",
        "hc_head.hc_base",
        "hc_head.hc_scale",
        "mtp.0.hc_head.hc_fn",
        "mtp.0.hc_head.hc_base",
        "mtp.0.hc_head.hc_scale",
    }
    for name, expected in expected_controls.items():
        assert actual_parameters[name].dtype == torch.float32
        assert torch.equal(actual_parameters[name], expected)
    assert layer.mlp.gate.expert_bias.dtype == torch.float32
    assert torch.equal(layer.self_attn.self_attn.sinks, expected_sink)
    assert not torch.equal(expected_sink, expected_sink.bfloat16().float())
    assert layer.proj.weight.dtype == torch.bfloat16
    assert _map_block_attr("self_attn.self_attn.sinks", "layers") == "attn.attn_sink"


def test_native_fp32_controls_map_to_release_fp32_controls() -> None:
    native_to_release = {
        "attn_hc.fn": "hc_attn_fn",
        "attn_hc.base": "hc_attn_base",
        "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.fn": "hc_ffn_fn",
        "ffn_hc.base": "hc_ffn_base",
        "ffn_hc.scale": "hc_ffn_scale",
        "self_attn.self_attn.sinks": "attn.attn_sink",
        "self_attn.self_attn.compressor.ape": "attn.compressor.ape",
        "self_attn.self_attn.indexer.compressor.ape": "attn.indexer.compressor.ape",
        "mlp.gate.expert_bias": "ffn.gate.bias",
        "hc_head.hc_fn": "hc_head_fn",
        "hc_head.hc_base": "hc_head_base",
        "hc_head.hc_scale": "hc_head_scale",
    }

    for native, release in native_to_release.items():
        assert _map_block_attr(native, "mtp" if native.startswith("hc_head.") else "layers") == release
