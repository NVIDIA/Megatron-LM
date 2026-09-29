# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regressions for DCP's prepended-axis tensor protocol."""

from itertools import product

import pytest
import torch
import torch.distributed.checkpoint as torch_dcp
from torch.distributed.checkpoint.metadata import MetadataIndex

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.strategies.checkpointable import (
    CheckpointableShardedTensor,
    LocalShardsContainer,
)

pytestmark = pytest.mark.internal


def _checkpoint_shards(reference, prepend_axes, *, row_shards):
    """Represent every logical chunk locally, using deliberately noncontiguous buffers."""
    shards = []
    prepend_shape = reference.shape[:prepend_axes]
    for coordinates in product(*(range(size) for size in prepend_shape)):
        for row_rank, values in enumerate(reference[coordinates].chunk(row_shards, dim=0)):
            data = torch.empty(values.shape[::-1], dtype=values.dtype).t()
            data.copy_(values)
            assert not data.is_contiguous()
            offsets = tuple(
                (axis, coordinate, prepend_shape[axis])
                for axis, coordinate in enumerate(coordinates)
            )
            shard = ShardedTensor.from_rank_offsets(
                "weight",
                data,
                *offsets,
                (prepend_axes, row_rank, row_shards),
                prepend_axis_num=prepend_axes,
            )
            shard.validate_metadata_integrity()
            shards.append(CheckpointableShardedTensor.from_sh_ten(shard))
    return LocalShardsContainer(shards)


@pytest.mark.parametrize("prepend_axes", [0, 1, 2])
def test_checkpointable_prepended_axes_match_tensor_view(prepend_axes):
    """Chunk extents and the writable data view describe exactly the same axes."""
    shape = (2,) * prepend_axes + (4, 3)
    reference = torch.arange(torch.Size(shape).numel()).reshape(shape)
    container = _checkpoint_shards(reference, prepend_axes, row_shards=2)
    for item, chunk in zip(
        container.__create_write_items__("weight", container), container.__create_chunk_list__()
    ):
        assert item.tensor_data.chunk == chunk
        assert len(chunk.offsets) == len(chunk.sizes) == len(shape)
        assert tuple(chunk.sizes) == (1,) * prepend_axes + (2, 3)
        view = container.__get_tensor_shard__(item.index)
        assert view.shape == chunk.sizes
        index = MetadataIndex("weight", chunk.offsets)
        assert container.__get_tensor_shard__(index).data_ptr() == view.data_ptr()
        expected = reference[
            tuple(slice(offset, offset + size) for offset, size in zip(chunk.offsets, chunk.sizes))
        ]
        torch.testing.assert_close(view, expected)
        view.fill_(-7)
        assert torch.all(container._local_shards[item.index.index]._sh_ten.data == -7)


@pytest.mark.parametrize("prepend_axes", [0, 1, 2])
def test_checkpointable_prepended_axes_dcp_resharding(tmp_path, prepend_axes):
    """Run DCP's ordinary planner, writer and loader across different row chunking."""
    shape = (2,) * prepend_axes + (4, 3)
    reference = torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(shape)
    source = {"weight": _checkpoint_shards(reference, prepend_axes, row_shards=2)}
    # Every global shard is owned locally; no_dist selects this real single-process layout.
    metadata = torch_dcp.save(source, checkpoint_id=tmp_path, no_dist=True)
    assert metadata.state_dict_metadata["weight"].size == reference.shape
    dense = {"weight": torch.empty_like(reference)}
    torch_dcp.load(dense, checkpoint_id=tmp_path, no_dist=True)
    torch.testing.assert_close(dense["weight"], reference, rtol=0, atol=0)
    target = {
        "weight": _checkpoint_shards(torch.full_like(reference, -1), prepend_axes, row_shards=1)
    }
    torch_dcp.load(target, checkpoint_id=tmp_path, no_dist=True)
    for shard in target["weight"]._local_shards:
        expected = reference[shard._sh_ten.global_offset[:prepend_axes]]
        torch.testing.assert_close(shard._sh_ten.data, expected, rtol=0, atol=0)
