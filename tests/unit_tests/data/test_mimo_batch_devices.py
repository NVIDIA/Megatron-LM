# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MIMO batch placement follows vision host-metadata contracts."""

import torch

from examples.mimo.training.batch import map_batch_tensors, move_batch_to_cuda
from megatron.core.models.vision.vit_model import _dynamic_patch_grid_lists
from megatron.core.packed_seq_params import PackedSeqParams


def test_cuda_move_preserves_host_sizes_and_moves_nested_attention_indices(monkeypatch):
    sizes = torch.tensor([[32, 48]], dtype=torch.int32)
    pixels = torch.zeros(1, 6, 768)
    tokens = torch.tensor([[1, 2]])
    cu = torch.tensor([0, 6], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    moved = []

    def cuda(tensor, *, non_blocking):
        assert non_blocking is True
        moved.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "cuda", cuda)
    batch = {
        "input_ids": tokens,
        "modality_inputs": {
            "vision": [
                {"x": pixels, "imgs_sizes": sizes, "packed_seq_params": packed},
                (pixels, "unchanged"),
            ]
        },
    }
    result = move_batch_to_cuda(batch)
    nested = result["modality_inputs"]["vision"][0]
    assert nested["imgs_sizes"] is sizes
    assert nested["packed_seq_params"] is packed
    assert all(tensor is not sizes for tensor in moved)
    assert len(moved) == 5  # tokens, pixels twice, Q and KV indices
    assert any(tensor is tokens for tensor in moved)
    assert any(tensor is cu for tensor in moved)
    assert result["modality_inputs"]["vision"][1][1] == "unchanged"
    # Exercise the actual downstream CPU size reader, not a reconstructed shape rule.
    grids, lengths = _dynamic_patch_grid_lists(nested["imgs_sizes"], 16)
    assert grids == [(2, 3)]
    assert lengths == [6]


def test_generic_mapper_keeps_its_default_transform_all_behavior():
    sizes = torch.tensor([[32, 48]])
    result = map_batch_tensors({"imgs_sizes": sizes}, lambda tensor: tensor + 1)
    assert torch.equal(result["imgs_sizes"], sizes + 1)
    assert torch.equal(sizes, torch.tensor([[32, 48]]))
