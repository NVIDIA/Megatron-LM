# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch.nn as nn

from megatron.lite.model.deepseek_v4.lite.protocol import (
    ImplConfig,
    _configure_dsa_kernel_fusion,
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


class _CSAOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.apply_dsa_kernel_fusion = True
        self.child = nn.Linear(2, 2)


def test_dsa_kernel_fusion_runtime_config_is_explicit_and_propagated() -> None:
    chunk = nn.Module()
    chunk.csa = _CSAOwner()

    assert ImplConfig().apply_dsa_kernel_fusion is True
    assert ImplConfig(apply_dsa_kernel_fusion=False).apply_dsa_kernel_fusion is False

    _configure_dsa_kernel_fusion([chunk], enabled=False)
    assert chunk.csa.apply_dsa_kernel_fusion is False
    _configure_dsa_kernel_fusion([chunk], enabled=True)
    assert chunk.csa.apply_dsa_kernel_fusion is True
