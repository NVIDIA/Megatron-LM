# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Distributed tests for the `gather_scatter` module.

These tests require `torchrun` (≥2 ranks and ≥1 GPU per rank). They create a `DBuffer` with known
data, gather full tensors to the owners via P2P, verify correctness, then scatter the results back
and verify that remote `DBuffer` views receive the changed results.
"""

import os

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.gather_scatter import (
    gather,
    scatter,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.owner_planning import (
    GroupOwnerLayout,
    assign_owner_work,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.range import intersect_ranges


def _setup() -> tuple[int, int, torch.device, DeviceMesh]:
    """Read torchrun env and return (rank, world_size, device, mesh)."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        pytest.skip("Needs at least two ranks.")
    if torch.cuda.is_available() and torch.cuda.device_count() < world_size:
        pytest.skip("Needs at least one GPU per rank when using CUDA.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    mesh = init_device_mesh(device.type, (world_size,))
    return rank, world_size, device, mesh


def _make_dbuffer(
    mesh: DeviceMesh, device: torch.device, tensor_shapes: list[torch.Size]
) -> DBuffer:
    """Create a `DBuffer` with all-`RowAtomic` placement."""
    return DBuffer.empty(
        mesh=mesh,
        placements=[RowAtomic()],
        tensor_shapes=tensor_shapes,
        dtype=torch.float32,
        device=device,
    )


def _owner_layout(dbuffer: DBuffer) -> GroupOwnerLayout:
    """Build owners directly from the DBuffer's existing layout and mesh."""
    indices = [i for i, shape in enumerate(dbuffer.layout.tensor_shapes) if len(shape) >= 2]
    tensor_to_owner = assign_owner_work(dbuffer.layout, dbuffer.mesh, indices)
    return GroupOwnerLayout(
        mesh=dbuffer.mesh, layout=dbuffer.layout, tensor_to_owner=tensor_to_owner
    )


def _known_full_tensors(
    dbuffer: DBuffer, owner_layout: GroupOwnerLayout, device: torch.device
) -> list[torch.Tensor]:
    """Fill this rank's local views with known data; return the distinct full tensors."""
    this_rank = dbuffer.mesh.get_rank()
    full_tensors = []
    for i, shape in enumerate(dbuffer.layout.tensor_shapes):
        full = (
            torch.arange(shape.numel(), dtype=torch.float32, device=device).view(shape) + i * 100.0
        )
        full_tensors.append(full)
        local_view = dbuffer.get_tensor_view(i)
        if local_view.numel() > 0:
            tensor_range = dbuffer.layout.get_tensor_range(i)
            shard_range = intersect_ranges(
                tensor_range, dbuffer.layout.get_rank_range(dbuffer.mesh, [RowAtomic()], this_rank)
            )
            offset = shard_range.start - tensor_range.start
            local_view.copy_(
                full.flatten()[offset : offset + shard_range.numel].view(local_view.shape)
            )
    return full_tensors


def _nonempty_local_tensors(
    dbuffer: DBuffer, owner_layout: GroupOwnerLayout
) -> dict[int, torch.Tensor]:
    """The gather source dict: the `DBuffer`'s local views for held parameters."""
    return {
        i: view
        for i in owner_layout.tensor_to_owner
        if (view := dbuffer.get_tensor_view(i)).numel() > 0
    }


@pytest.mark.parametrize("subgroup", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_gather_scatter_round_trip(subgroup, noncontiguous):
    """Gather tensors and scatter changed results into the remote DBuffer views."""
    rank, world_size, device, mesh = _setup()
    if subgroup:
        if world_size < 4:
            pytest.skip("Needs four ranks to test the noncontiguous subgroup [1, 3].")
        mesh = DeviceMesh(device.type, [1, 3])
        if rank not in (1, 3):
            return
    tensor_shapes = [torch.Size((8, 4)), torch.Size((4, 4))]
    dbuffer = _make_dbuffer(mesh, device, tensor_shapes)
    owner_layout = _owner_layout(dbuffer)
    full_tensors = _known_full_tensors(dbuffer, owner_layout, device)
    this_rank = mesh.get_rank()

    # --- Gather ---
    owned = {i for i, owner in owner_layout.tensor_to_owner.items() if owner == this_rank}
    destination = {
        i: torch.empty(full_tensors[i].shape, dtype=dbuffer.dtype, device=device) for i in owned
    }
    source = _nonempty_local_tensors(dbuffer, owner_layout)
    if noncontiguous:
        source = {
            i: torch.empty((*shard.shape, 2), dtype=shard.dtype, device=device)[..., 0].copy_(shard)
            for i, shard in source.items()
        }
    gather(source, destination, owner_layout=owner_layout)

    # Owners got the correct full tensors; nothing else was written.
    for i in owned:
        torch.testing.assert_close(destination[i], full_tensors[i], atol=0, rtol=0)
    assert set(destination) == owned

    # Scatter changed results to ensure every remote destination is written.
    for tensor in destination.values():
        tensor.add_(1)
    # Destination = the `DBuffer`'s local views, the natural update-application site.
    held_not_owned = [
        i
        for i in owner_layout.tensor_to_owner
        if i not in owned and dbuffer.get_tensor_view(i).numel() > 0
    ]
    scatter_destination = {i: dbuffer.get_tensor_view(i) for i in held_not_owned}
    scatter(destination, scatter_destination, owner_layout=owner_layout)

    for i in owner_layout.tensor_to_owner:
        local_view = dbuffer.get_tensor_view(i)
        if local_view.numel() == 0:
            continue
        tensor_range = dbuffer.layout.get_tensor_range(i)
        shard_range = intersect_ranges(
            tensor_range, dbuffer.layout.get_rank_range(mesh, [RowAtomic()], this_rank)
        )
        offset = shard_range.start - tensor_range.start
        expected = full_tensors[i].flatten()[offset : offset + shard_range.numel]
        if i in held_not_owned:
            expected = expected + 1
        torch.testing.assert_close(local_view.flatten(), expected, atol=0, rtol=0)


def test_gather_scatter_with_stream():
    """gather -> scatter on a separate stream with explicit waits produces correct results.

    The scatter destination is plain flat tensors (not `DBuffer` views), matching `scatter`'s
    caller-owned application contract.
    """
    _, _, device, mesh = _setup()
    if not torch.cuda.is_available():
        pytest.skip("Needs CUDA for stream testing.")

    tensor_shapes = [torch.Size((8, 4)), torch.Size((4, 4))]
    dbuffer = _make_dbuffer(mesh, device, tensor_shapes)
    owner_layout = _owner_layout(dbuffer)
    full_tensors = _known_full_tensors(dbuffer, owner_layout, device)
    this_rank = mesh.get_rank()

    stream = torch.cuda.Stream(device=device)

    owned = {i for i, owner in owner_layout.tensor_to_owner.items() if owner == this_rank}
    destination = {
        i: torch.empty(full_tensors[i].shape, dtype=dbuffer.dtype, device=device) for i in owned
    }
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gather(
            _nonempty_local_tensors(dbuffer, owner_layout), destination, owner_layout=owner_layout
        )
    stream.synchronize()

    for i in owned:
        torch.testing.assert_close(destination[i], full_tensors[i], atol=0, rtol=0)

    # Scatter into plain flat result-shard tensors.
    held_not_owned = [
        i
        for i in owner_layout.tensor_to_owner
        if i not in owned and dbuffer.get_tensor_view(i).numel() > 0
    ]
    scatter_destination = {
        i: torch.empty(dbuffer.get_tensor_view(i).numel(), dtype=torch.float32, device=device)
        for i in held_not_owned
    }
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scatter(destination, scatter_destination, owner_layout=owner_layout)
    stream.synchronize()

    for i in held_not_owned:
        tensor_range = dbuffer.layout.get_tensor_range(i)
        shard_range = intersect_ranges(
            tensor_range, dbuffer.layout.get_rank_range(mesh, [RowAtomic()], this_rank)
        )
        offset = shard_range.start - tensor_range.start
        expected = full_tensors[i].flatten()[offset : offset + shard_range.numel]
        torch.testing.assert_close(scatter_destination[i], expected, atol=0, rtol=0)
