# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA attention backend contract: Core-validated backend, FlashMLA indices, native FHT."""
import pytest
import torch

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.components.config import core_config
from megatron.lite.primitive.modules.attention.ratio_one import flash_positions
from megatron.lite.runtime.contracts import ParallelConfig


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only backend validation")
def test_backend_is_set_at_construction_and_validated_by_core():
    release = DeepseekV41Config(num_hidden_layers=4)
    assert core_config(release, ParallelConfig()).dsa_kernel_backend == "none"
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
