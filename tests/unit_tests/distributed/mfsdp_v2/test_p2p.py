# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for the `p2p` owner-compute gather/scatter module.

These tests require `torchrun` (>=2 ranks and >=1 GPU per rank). They create a `DBuffer`
with known data, gather full tensors to the owners via P2P, verify correctness, then
scatter the results back and verify the `DBuffer` is unchanged (identity round-trip).

All dicts are keyed by tensor index; all communicated tensors are flat element ranges,
matching `owner_planning`.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import DBuffer, Flat
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.owner_planning import (
    GroupOwnerLayout,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.p2p import gather, scatter


def _setup():
    """Read torchrun env and return (rank, world_size, device, mesh)."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip("Needs >=2 ranks and >=1 GPU per rank.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    mesh = init_device_mesh(device.type, (world_size,))
    return rank, world_size, device, mesh


def _make_dbuffer(mesh, device, tensor_shapes):
    """Create a `DBuffer` with all-`Flat` placement, filled with known data.

    Also builds its `GroupOwnerLayout` by mocking the minimal `FsdpParameterGroup`
    surface that `GroupOwnerLayout.from_group` reads (default NS-5 cost balancing and
    default >=2D eligibility).
    """
    dbuffer = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        tensor_shapes=tensor_shapes,
        dtype=torch.float32,
        device=device,
    )
    params = tuple(nn.Parameter(torch.empty(shape)) for shape in tensor_shapes)
    group = SimpleNamespace(
        mesh=mesh,
        main_weight=SimpleNamespace(layout=dbuffer.layout),
        fsdp_parameters=tuple(SimpleNamespace(sharded=param) for param in params),
    )
    plan = GroupOwnerLayout.from_group(group)

    this_rank = mesh.get_local_rank()
    full_tensors = []
    for i, shape in enumerate(tensor_shapes):
        full = (
            torch.arange(shape.numel(), dtype=torch.float32, device=device).view(shape) + i * 100.0
        )
        full_tensors.append(full)
        local_view = dbuffer.get_local_tensor(i)
        if local_view.numel() > 0:
            layout = plan.layouts[i]
            offset = layout.rank_offset(this_rank)
            numel = layout.rank_numel(this_rank)
            local_view.copy_(full.flatten()[offset : offset + numel].view(local_view.shape))
    return dbuffer, full_tensors, plan


def test_gather_scatter_round_trip():
    """gather -> scatter identity round-trip: `DBuffer` data is unchanged.

    One boundary and one non-boundary parameter: the owners reconstruct the full
    tensors, and scattering them back into the `DBuffer`'s local views leaves the
    local buffer unchanged.
    """
    rank, world_size, device, mesh = _setup()
    tensor_shapes = [torch.Size((8, 4)), torch.Size((4, 4))]
    dbuffer, full_tensors, plan = _make_dbuffer(mesh, device, tensor_shapes)
    this_rank = mesh.get_local_rank()

    # --- Gather ---
    owned = {i for i in plan.layouts if plan.owners[i] == this_rank}
    destination = {
        i: torch.empty(full_tensors[i].shape, dtype=dbuffer.dtype, device=device) for i in owned
    }
    gather(dbuffer, destination, plan=plan)

    # Owners got the correct full tensors; nothing else was written.
    for i in owned:
        torch.testing.assert_close(destination[i], full_tensors[i], atol=0, rtol=0)
    assert set(destination) == owned

    # --- Scatter (identity: scatter the gathered full tensors back) ---
    # Destination = the `DBuffer`'s local views, the natural update-application site.
    held_not_owned = [
        i for i in plan.layouts if i not in owned and dbuffer.get_local_tensor(i).numel() > 0
    ]
    scatter_destination = {i: dbuffer.get_local_tensor(i) for i in held_not_owned}
    original_local = dbuffer.local_buffer.clone()
    scatter(destination, scatter_destination, plan=plan)

    # The `DBuffer` local buffer is unchanged (round-trip).
    torch.testing.assert_close(dbuffer.local_buffer, original_local, atol=0, rtol=0)


def test_gather_scatter_with_stream():
    """gather -> scatter on a side stream produces correct plain-tensor results.

    The scatter destination is plain flat tensors (not `DBuffer` views), matching
    `scatter`'s caller-owned application contract.
    """
    rank, world_size, device, mesh = _setup()
    if not torch.cuda.is_available():
        pytest.skip("Needs CUDA for stream testing.")

    tensor_shapes = [torch.Size((8, 4)), torch.Size((4, 4))]
    dbuffer, full_tensors, plan = _make_dbuffer(mesh, device, tensor_shapes)
    this_rank = mesh.get_local_rank()

    stream = torch.cuda.Stream(device=device)

    owned = {i for i in plan.layouts if plan.owners[i] == this_rank}
    destination = {
        i: torch.empty(full_tensors[i].shape, dtype=dbuffer.dtype, device=device) for i in owned
    }
    gather(dbuffer, destination, plan=plan, stream=stream)
    stream.synchronize()

    for i in owned:
        torch.testing.assert_close(destination[i], full_tensors[i], atol=0, rtol=0)

    # Scatter into plain flat result-shard tensors.
    held_not_owned = [
        i for i in plan.layouts if i not in owned and dbuffer.get_local_tensor(i).numel() > 0
    ]
    scatter_destination = {
        i: torch.empty(plan.layouts[i].rank_numel(this_rank), dtype=torch.float32, device=device)
        for i in held_not_owned
    }
    scatter(destination, scatter_destination, plan=plan, stream=stream)
    stream.synchronize()

    for i in held_not_owned:
        layout = plan.layouts[i]
        offset = layout.rank_offset(this_rank)
        numel = layout.rank_numel(this_rank)
        expected = full_tensors[i].flatten()[offset : offset + numel]
        torch.testing.assert_close(scatter_destination[i], expected, atol=0, rtol=0)
