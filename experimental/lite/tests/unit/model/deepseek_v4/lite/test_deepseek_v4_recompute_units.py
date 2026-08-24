# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn

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


def test_training_cast_preserves_release_fp32_controls() -> None:
    model = nn.Module()
    model.layers = nn.ModuleDict({"0": nn.Module()})
    model.layers["0"].attn = nn.Module()
    model.layers["0"].attn.attn_sink = nn.Parameter(
        torch.tensor([1.0001, -0.5003], dtype=torch.float32)
    )
    model.layers["0"].proj = nn.Linear(2, 2, bias=False)
    expected_sink = model.layers["0"].attn.attn_sink.detach().clone()

    _cast_training_parameters(model)

    assert model.layers["0"].attn.attn_sink.dtype == torch.float32
    assert torch.equal(model.layers["0"].attn.attn_sink, expected_sink)
    assert not torch.equal(expected_sink, expected_sink.bfloat16().float())
    assert model.layers["0"].proj.weight.dtype == torch.bfloat16
