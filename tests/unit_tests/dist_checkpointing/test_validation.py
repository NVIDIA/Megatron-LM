# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock

import torch

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
