# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch.nn as nn

from megatron.lite.model.deepseek_v4.lite.protocol import _iter_transformer_units


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
