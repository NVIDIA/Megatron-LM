# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA attention backend contract: Core-validated backend, FlashMLA indices, native FHT."""
import pytest
import torch

from megatron.lite.primitive.modules.attention.ratio_one import flash_positions


@pytest.mark.parametrize("width,alignment", [(1, 64), (64, 64), (65, 64), (130, 128)])
def test_flash_positions_pad_to_alignment_with_no_key(width, alignment):
    positions = torch.arange(width, dtype=torch.int32).repeat(3, 1)
    padded = flash_positions(positions, alignment)
    assert padded.shape[-1] % alignment == 0 and padded.shape[-1] - width < alignment
    assert torch.equal(padded[:, :width], positions)
    assert bool((padded[:, width:] == -1).all())
    assert padded.dtype == torch.int32 and padded.is_contiguous()
