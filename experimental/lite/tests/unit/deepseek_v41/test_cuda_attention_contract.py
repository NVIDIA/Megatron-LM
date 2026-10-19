# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA attention backend contract: Core-validated backend, FlashMLA indices, native FHT."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.native import core_config, require_native_hadamard
from megatron.lite.model.deepseek_v41.lite.ratio_one import flash_positions
from megatron.lite.runtime.contracts import ParallelConfig


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only backend validation")
def test_backend_is_set_at_construction_and_validated_by_core():
    release = DeepseekV41Config(num_hidden_layers=4)
    assert core_config(release, ParallelConfig()).dsa_kernel_backend == "none"
    # Core rejects the fused backend without CUDA when the config is built,
    # which assigning dsa_kernel_backend afterwards would skip.
    with pytest.raises(ValueError, match="requires a CUDA device"):
        core_config(
            DeepseekV41Config(num_hidden_layers=4), ParallelConfig(), attention_backend="cudnn"
        )


@pytest.mark.parametrize("width,alignment", [(1, 64), (64, 64), (65, 64), (130, 128)])
def test_flash_positions_pad_to_alignment_with_no_key(width, alignment):
    positions = torch.arange(width, dtype=torch.int32).repeat(3, 1)
    padded = flash_positions(positions, alignment)
    assert padded.shape[-1] % alignment == 0 and padded.shape[-1] - width < alignment
    assert torch.equal(padded[:, :width], positions)
    assert bool((padded[:, width:] == -1).all())
    assert padded.dtype == torch.int32 and padded.is_contiguous()


def test_cuda_index_selection_rejects_a_hadamard_stand_in(monkeypatch):
    from megatron.core.transformer.experimental_attention_variant import dsa

    monkeypatch.setattr(dsa, "hadamard_transform", lambda x, scale=1.0: x * scale)
    with pytest.raises(ValueError, match="native fast_hadamard_transform"):
        require_native_hadamard()
    monkeypatch.setattr(dsa, "hadamard_transform", None)
    with pytest.raises(ValueError, match="native fast_hadamard_transform"):
        require_native_hadamard()
    native = SimpleNamespace(__module__="fast_hadamard_transform.fast_hadamard_transform_interface")
    monkeypatch.setattr(dsa, "hadamard_transform", native)
    require_native_hadamard()
