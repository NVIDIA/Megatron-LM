# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for owner gather/scatter using DBuffers.

These tests require `torchrun` (≥2 ranks); the stream test also requires CUDA. They gather
computed inputs from a separate DBuffer, then scatter results into remote-owned DBuffer shards.
"""

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import DBuffer, layout_builder
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.gather_scatter import (
    gather,
    scatter,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.owner_planning import (
    assign_owner_work,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    BlockAtomic,
    RowAtomic,
    TensorAtomic,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.range import intersect_ranges


def _known_full_tensors(dbuffer: DBuffer) -> list[torch.Tensor]:
    """Fill this rank's local views with known data; return the distinct full tensors."""
    full_tensors = []
    for i, shape in enumerate(dbuffer.layout.tensor_shapes):
        full = (
            torch.arange(shape.numel(), dtype=dbuffer.dtype, device=dbuffer.device).view(shape)
            + i * 100.0
        )
        full_tensors.append(full)
        local_view = dbuffer.get_tensor_view(i)
        if local_view.numel() > 0:
            tensor_range = dbuffer.layout.get_tensor_range(i)
            shard_range = intersect_ranges(
                tensor_range, dbuffer.layout.get_local_range(dbuffer.mesh, dbuffer.placements)
            )
            offset = shard_range.start - tensor_range.start
            local_view.copy_(
                full.flatten()[offset : offset + shard_range.numel].view(local_view.shape)
            )
    return full_tensors


@pytest.mark.parametrize("subgroup", [False, True])
@pytest.mark.parametrize("placement", [RowAtomic(), BlockAtomic(2), TensorAtomic()])
def test_gather_scatter_round_trip(subgroup, placement, distributed_setup):
    """Gather computed inputs into contiguous tensors and scatter the results."""
    if distributed_setup.world_size < 2:
        pytest.skip("Needs at least two ranks.")
    device = distributed_setup.device
    if subgroup:
        if distributed_setup.world_size < 4:
            pytest.skip("Needs four ranks to test the noncontiguous subgroup [1, 3].")
        mesh = DeviceMesh(device.type, [1, 3])
        if mesh.get_coordinate() is None:
            pytest.skip("Rank is outside the test subgroup [1, 3].")
    else:
        mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    tensor_shapes = [torch.Size((10, 4)), torch.Size((4, 4)), torch.Size((4,))]
    if isinstance(placement, TensorAtomic):
        layout = layout_builder.build_for_tensor_atomic(
            tensor_shapes, dp_size=mesh.size(), tensor_owners=[0, mesh.size() - 1, 0]
        )
    else:
        layout = layout_builder.build_for_row_atomic(
            tensor_shapes,
            dp_size=mesh.size(),
            block_size=placement.block_size if isinstance(placement, BlockAtomic) else 1,
        )
    dbuffer = DBuffer(mesh, [placement], layout, torch.float32, device)
    dbuffer.local_buffer.fill_(-1)
    full_tensors = _known_full_tensors(dbuffer)
    original = dbuffer.local_buffer.clone()
    this_rank = mesh.get_rank()
    (tensor_to_owner,) = assign_owner_work([dbuffer])
    assert set(tensor_to_owner) == {0, 1}
    # Reverse traversal on odd ranks to check that P2P is independent of dictionary order.
    if this_rank % 2:
        tensor_to_owner = dict(reversed(list(tensor_to_owner.items())))

    # Momentum/Nesterov-style computed inputs can use separate storage with the same layout.
    source = DBuffer.from_local(dbuffer.local_buffer + 2, mesh, [placement], layout)
    gather(source, {}, tensor_to_owner={})
    scatter({}, dbuffer, tensor_to_owner={})
    torch.testing.assert_close(dbuffer.local_buffer, original, atol=0, rtol=0)

    owned = {i for i, owner in tensor_to_owner.items() if owner == this_rank}
    destination = {
        i: torch.empty(full_tensors[i].shape, dtype=dbuffer.dtype, device=device) for i in owned
    }
    gather(source, destination, tensor_to_owner=tensor_to_owner)
    for i in owned:
        torch.testing.assert_close(destination[i], full_tensors[i] + 2, atol=0, rtol=0)
    assert set(destination) == owned
    torch.testing.assert_close(dbuffer.local_buffer, original, atol=0, rtol=0)

    for tensor in destination.values():
        tensor.add_(1)
    scatter(destination, dbuffer, tensor_to_owner=tensor_to_owner)

    expected_buffer = original.clone()
    for i in tensor_to_owner:
        if i in owned:
            continue
        shard_range = intersect_ranges(
            layout.get_tensor_range(i), layout.get_local_range(mesh, dbuffer.placements)
        )
        if shard_range.numel > 0:
            offset = shard_range.start - dbuffer.offset
            expected_buffer[offset : offset + shard_range.numel].add_(3)
    # Owner-local shards, excluded tensors, and padding retain their original values.
    torch.testing.assert_close(dbuffer.local_buffer, expected_buffer, atol=0, rtol=0)
    torch.testing.assert_close(source.local_buffer, original + 2, atol=0, rtol=0)


def test_gather_scatter_with_stream(distributed_setup):
    """Gather and scatter on a separate stream into a scratch DBuffer with explicit waits."""
    if distributed_setup.world_size < 2:
        pytest.skip("Needs at least two ranks.")
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    dbuffer = DBuffer.empty(mesh, [RowAtomic()], [(8, 4), (4, 4)], torch.float32, device)
    full_tensors = _known_full_tensors(dbuffer)
    (tensor_to_owner,) = assign_owner_work([dbuffer])
    this_rank = mesh.get_rank()
    owned = {i for i, owner in tensor_to_owner.items() if owner == this_rank}
    destination = {
        i: torch.empty(full_tensors[i].shape, dtype=dbuffer.dtype, device=device) for i in owned
    }
    scratch = DBuffer.from_local(
        torch.full_like(dbuffer.local_buffer, -1), mesh, dbuffer.placements, dbuffer.layout
    )

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gather(dbuffer, destination, tensor_to_owner=tensor_to_owner)
        scatter(destination, scratch, tensor_to_owner=tensor_to_owner)
    stream.synchronize()

    for i in owned:
        torch.testing.assert_close(destination[i], full_tensors[i], atol=0, rtol=0)
    for i in tensor_to_owner:
        local_view = dbuffer.get_tensor_view(i)
        result_view = scratch.get_tensor_view(i)
        expected = torch.full_like(local_view, -1) if i in owned else local_view
        torch.testing.assert_close(result_view, expected, atol=0, rtol=0)
