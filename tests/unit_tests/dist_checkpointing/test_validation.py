# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock

import torch

from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.validation import determine_global_metadata


def test_determine_global_metadata_uses_explicit_process_group(monkeypatch):
    process_group = Mock()
    metadata = Mock()
    shard = Mock()
    shard.without_data.return_value = metadata
    get_world_size = Mock(return_value=2)
    all_gather_object = Mock()
    monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    local_metadata, global_metadata = determine_global_metadata(
        {"model": shard}, process_group=process_group
    )

    assert local_metadata == [metadata]
    assert global_metadata == [None, None]
    get_world_size.assert_called_once_with(group=process_group)
    all_gather_object.assert_called_once_with(global_metadata, local_metadata, group=process_group)


def test_diff_reports_tensors_with_broadcastable_shapes_as_mismatched():
    """Tensors whose shapes differ must not be compared through broadcasting."""
    for left, right in [
        (torch.ones(1, 4), torch.ones(8, 4)),
        (torch.zeros(4), torch.zeros(4, 1)),
        (torch.zeros(()), torch.zeros(3)),
    ]:
        only_left, only_right, mismatch = diff({"w": left}, {"w": right})

        assert not only_left
        assert not only_right
        assert mismatch == [(("w",), torch.Tensor, torch.Tensor, left.shape, right.shape)]

    assert diff({"w": torch.ones(2, 3)}, {"w": torch.ones(2, 3)}) == ([], [], [])
