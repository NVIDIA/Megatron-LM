# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.lite.checkpoint import _map_block_attr
from megatron.lite.model.deepseek_v4.lite.protocol import (
    _cast_training_parameters,
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
    layer.attn_hc.hc_fn = nn.Parameter(torch.tensor([1.0001], dtype=torch.float32))
    layer.proj = nn.Linear(2, 2, bias=False)
    expected_sink = layer.self_attn.self_attn.sinks.detach().clone()

    _cast_training_parameters(model)

    assert layer.self_attn.self_attn.sinks.dtype == torch.float32
    assert torch.equal(layer.self_attn.self_attn.sinks, expected_sink)
    assert layer.attn_hc.hc_fn.dtype == torch.float32
    assert not torch.equal(expected_sink, expected_sink.bfloat16().float())
    assert layer.proj.weight.dtype == torch.bfloat16
    assert _map_block_attr("self_attn.self_attn.sinks", "layers") == "attn.attn_sink"
