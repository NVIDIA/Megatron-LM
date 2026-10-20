# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.vision.vit_model import _dynamic_patch_grid_lists


@pytest.mark.parametrize(
    "imgs_sizes", [[(28, 28), (28, 56)], torch.tensor([[28, 28], [28, 56]], dtype=torch.int32)]
)
def test_dynamic_patch_grid_lists_keeps_metadata_on_cpu(imgs_sizes):
    patch_hw, seq_lens = _dynamic_patch_grid_lists(imgs_sizes, patch_dim=14)

    assert patch_hw == [(2, 2), (2, 4)]
    assert seq_lens == [4, 8]


def test_dynamic_patch_grid_lists_rejects_cuda_metadata():
    imgs_sizes = torch.tensor([[28, 28]], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="must remain on CPU"):
        _dynamic_patch_grid_lists(imgs_sizes, patch_dim=14)
