# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP DBuffer."""

from collections.abc import Iterable

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.layout import GlobalLayout
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    BlockAtomic,
    Flat,
)


def _same_tensors_on_all_ranks(device: torch.device) -> list[torch.Tensor]:
    return [
        torch.arange(21, dtype=torch.float32, device=device).reshape(7, 3),
        torch.arange(10, dtype=torch.float32, device=device).reshape(2, 5) + 100,
        torch.arange(7, dtype=torch.float32, device=device) + 200,
    ]


def _assert_dbuffer_local_tensors_close(buffer: DBuffer, expected: Iterable[torch.Tensor]) -> None:
    for index, tensor in enumerate(expected):
        torch.testing.assert_close(buffer.get_local_tensor(index), tensor)


def test_dbuffer_layout_aligns_tensors_and_pads_to_lcm_times_dp_size(distributed_setup):
    """DBuffer layout aligns tensor starts and pads to LCM * DP size."""
    if distributed_setup.world_size < 2:
        pytest.skip("DBuffer layout test requires at least 2 ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (2,))
    shapes = [torch.Size((5, 4)), torch.Size((2, 6)), torch.Size((3,))]

    buffer = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        layout=GlobalLayout.build(shapes, dp_size=mesh.size()),
        dtype=torch.float32,
        device=distributed_setup.device,
    )

    assert buffer.layout.tensor_shapes == tuple(shapes)
    assert buffer.layout.tensor_to_offset == (0, 24, 20)
    assert buffer.layout.size == 48


def test_dbuffer_layout_aligns_fragment_offsets_to_rows(distributed_setup):
    """DBuffer layout keeps small tensors aligned to their non-leading dimensions."""
    if distributed_setup.world_size < 2:
        pytest.skip("DBuffer layout test requires at least 2 ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (2,))
    shapes = [torch.Size((4, 4)), torch.Size((1, 6))]

    buffer = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        layout=GlobalLayout.build(shapes, dp_size=mesh.size()),
        dtype=torch.float32,
        device=distributed_setup.device,
    )

    assert buffer.layout.tensor_to_offset == (0, 18)
    assert buffer.layout.size == 24


def test_block_atomic_layout_keeps_bf16_blocks_on_one_rank(distributed_setup):
    """BlockAtomic avoids the odd local row counts permitted by Flat."""
    if distributed_setup.world_size < 2:
        pytest.skip("Requires at least 2 ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (2,))
    tensors = [
        torch.arange(24, dtype=torch.bfloat16, device=distributed_setup.device).reshape(4, 6),
        torch.arange(32, dtype=torch.bfloat16, device=distributed_setup.device).reshape(8, 4),
    ]
    flat = DBuffer.distribute_tensors(tensors, mesh, [Flat()])
    block_atomic = DBuffer.distribute_tensors(tensors, mesh, [BlockAtomic(2)], block_size=2)

    # P0=(4, 6) has 24 elements and P1=(8, 4) has 32. With two ranks, Flat
    # uses 36-element shards: rank 0 owns P0 and 3 P1 rows, while rank 1 owns
    # the remaining 5 P1 rows. BlockAtomic(2) uses 48-element shards instead,
    # giving rank 0 6 P1 rows and rank 1 2 P1 rows—both whole row blocks.
    assert flat.placements != block_atomic.placements
    assert flat.get_local_tensor(1).shape[0] % 2 == 1
    assert block_atomic.get_local_tensor(1).shape[0] % 2 == 0


def test_compute_layout_preserves_rows_on_a_5_rank_flat_sharded_mesh(distributed_setup):
    """Flat layout preserves rows while tensor-aligned padding crosses rank boundaries."""
    if distributed_setup.world_size < 5:
        pytest.skip("Flat layout test requires at least 5 ranks.")

    # P0-P4 are zero-based logical tensor names matching tensor indices.
    shapes = [
        torch.Size((2, 6)),  # P0
        torch.Size((4, 4)),  # P1
        torch.Size((4, 4)),  # P2
        torch.Size((1, 2)),  # P3
        torch.Size((1, 6)),  # P4
    ]

    mesh = init_device_mesh(distributed_setup.device.type, (5,))
    if mesh.get_coordinate() is None:
        pytest.skip("Rank is outside the 5-rank DBuffer mesh.")

    buffer = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        layout=GlobalLayout.build(shapes, dp_size=mesh.size()),
        dtype=torch.float32,
        device=distributed_setup.device,
    )
    layout = buffer.layout
    expected_local_shapes_by_rank = [
        [(2, 6), (0, 4), (0, 4), (0, 2), (0, 6)],
        [(0, 6), (3, 4), (0, 4), (0, 2), (0, 6)],
        [(0, 6), (1, 4), (1, 4), (1, 2), (0, 6)],
        [(0, 6), (0, 4), (3, 4), (0, 2), (0, 6)],
        [(0, 6), (0, 4), (0, 4), (0, 2), (1, 6)],
    ]

    assert layout.tensor_shapes == tuple(shapes)
    assert layout.tensor_to_offset == (0, 12, 32, 28, 48)
    assert layout.size == 60
    expected_local_shapes = expected_local_shapes_by_rank[mesh.get_local_rank(0)]
    for index, expected_shape in enumerate(expected_local_shapes):
        assert buffer.get_local_tensor(index).shape == torch.Size(expected_shape)
        assert buffer.get_dtensor(index).shape == shapes[index]


def test_constructor_allocates_local_buffer(distributed_setup):
    """DBuffer allocates local storage from shape, mesh, placement, dtype, and device."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensor_shapes = [torch.Size((7, 3)), torch.Size((2, 5)), torch.Size((7,))]
    mesh_size = mesh.size()

    replicated_buffer = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        layout=GlobalLayout.build(tensor_shapes, dp_size=mesh.size()),
        dtype=torch.float32,
        device=distributed_setup.device,
    )
    sharded_buffer = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        layout=GlobalLayout.build(tensor_shapes, dp_size=mesh.size()),
        dtype=torch.float32,
        device=distributed_setup.device,
    )

    assert replicated_buffer.layout == sharded_buffer.layout
    assert replicated_buffer.layout.tensor_shapes == tuple(tensor_shapes)
    assert replicated_buffer.offset == 0
    expected_sharded_local_numel = replicated_buffer.layout.size // distributed_setup.world_size
    assert sharded_buffer.offset == distributed_setup.rank * expected_sharded_local_numel
    assert replicated_buffer.local_buffer.numel() == replicated_buffer.layout.size
    assert (
        sharded_buffer.local_buffer.numel()
        == replicated_buffer.layout.size // distributed_setup.world_size
    )
    assert sharded_buffer.layout.size % (15 * mesh_size) == 0
    assert replicated_buffer.dtype == torch.float32
    assert sharded_buffer.local_buffer.device == distributed_setup.device


def test_cast_to_same_dtype_returns_self(distributed_setup):
    """DBuffer.cast returns self when the dtype already matches."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(distributed_setup.device)
    buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])

    assert buffer.cast(torch.float32) is buffer


def test_cast_preserves_layout_and_casts_values(distributed_setup):
    """DBuffer.cast preserves layout metadata and casts local values."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(distributed_setup.device)
    buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])

    cast_buffer = buffer.cast(torch.bfloat16)

    assert cast_buffer is not buffer
    assert cast_buffer.mesh == buffer.mesh
    assert cast_buffer.placements == buffer.placements
    assert cast_buffer.layout == buffer.layout
    assert cast_buffer.device == buffer.device
    assert cast_buffer.dtype is torch.bfloat16
    _assert_dbuffer_local_tensors_close(
        cast_buffer, [tensor.to(dtype=torch.bfloat16) for tensor in tensors]
    )


def test_cast_with_out_reuses_destination_and_casts_values(distributed_setup):
    """DBuffer.cast writes casted values into an existing destination buffer."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(distributed_setup.device)
    buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])
    destination = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        layout=buffer.layout,
        dtype=torch.bfloat16,
        device=distributed_setup.device,
    )
    destination_data_ptr = destination.local_buffer.data_ptr()

    result = buffer.cast(torch.bfloat16, out=destination)

    assert result is destination
    assert destination.local_buffer.data_ptr() == destination_data_ptr
    _assert_dbuffer_local_tensors_close(
        destination, [tensor.to(dtype=torch.bfloat16) for tensor in tensors]
    )


def test_release_and_reallocate_storage_preserves_buffer_views(distributed_setup):
    """DBuffer storage can be released and reallocated without replacing existing views."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    buffer = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        layout=GlobalLayout.build([torch.Size((4, 4))], dp_size=mesh.size()),
        dtype=torch.float32,
        device=distributed_setup.device,
    )
    tensor_view = buffer.get_local_tensor(0)
    buffer_data_ptr = buffer.local_buffer.data_ptr()
    tensor_view_data_ptr = tensor_view.data_ptr()

    buffer.release_storage()
    assert buffer.local_buffer.untyped_storage().nbytes() == 0

    buffer.reallocate_storage()
    assert (
        buffer.local_buffer.untyped_storage().nbytes()
        == buffer.local_buffer.numel() * buffer.local_buffer.element_size()
    )
    assert buffer.local_buffer.data_ptr() == buffer_data_ptr
    assert tensor_view.data_ptr() == tensor_view_data_ptr
    buffer.local_buffer.fill_(7.0)
    torch.testing.assert_close(tensor_view, torch.full_like(tensor_view, 7.0))


def test_from_local_reuses_required_local_buffer(distributed_setup):
    """DBuffer.from_local reuses caller-provided local storage without allocation."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(distributed_setup.device)
    replicated_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])
    local_numel = replicated_buffer.layout.size // distributed_setup.world_size
    offset = distributed_setup.rank * local_numel
    local_buffer = replicated_buffer.local_buffer.narrow(0, offset, local_numel)

    sharded_buffer = DBuffer.from_local(
        local_buffer,
        mesh,
        iter([Flat()]),
        replicated_buffer.layout,
        allocation_stream=replicated_buffer.allocation_stream,
    )

    assert sharded_buffer.placements == (Flat(),)
    assert sharded_buffer.layout == replicated_buffer.layout
    assert sharded_buffer.offset == offset
    assert sharded_buffer.local_buffer.data_ptr() == local_buffer.data_ptr()
    _assert_dbuffer_local_tensors_close(sharded_buffer.allgather(0), tensors)


def test_replicate_get_local_tensor_and_dtensor(distributed_setup):
    """Replicated DBuffer returns full local tensors and replicated DTensors."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(distributed_setup.device)

    buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])

    _assert_dbuffer_local_tensors_close(buffer, tensors)
    dtensor = buffer.get_dtensor(0)
    torch.testing.assert_close(dtensor.to_local(), tensors[0], rtol=0, atol=0)


def test_distribute_tensors_moves_inputs_to_mesh_device(distributed_setup):
    """distribute_tensors moves full input tensors to the mesh device type."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(torch.device("cpu"))

    buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])

    assert buffer.local_buffer.device == distributed_setup.device
    _assert_dbuffer_local_tensors_close(
        buffer, [tensor.to(distributed_setup.device) for tensor in tensors]
    )


def test_distribute_tensors_detaches_and_contiguizes_inputs(distributed_setup):
    """distribute_tensors treats input tensors as detached contiguous values."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    parameter = torch.nn.Parameter(
        torch.arange(12, dtype=torch.float32, device=distributed_setup.device).view(3, 4).t()
    )

    buffer = DBuffer.distribute_tensors([parameter], mesh, [Replicate()])

    assert not parameter.is_contiguous()
    assert buffer.get_local_tensor(0).is_contiguous()
    assert not buffer.local_buffer.requires_grad
    torch.testing.assert_close(
        buffer.get_local_tensor(0), parameter.detach().contiguous(), rtol=0, atol=0
    )


def test_sharded_allgather_round_trip(distributed_setup):
    """Sharded buffers round-trip through all-gather as contiguous tensor fragments."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(distributed_setup.device)

    sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Flat()])
    layout = sharded_buffer.layout
    for index, tensor in enumerate(tensors):
        local_tensor = sharded_buffer.get_local_tensor(index)
        assert local_tensor.shape[1:] == tensor.shape[1:]
        assert local_tensor.is_contiguous()

    replicated_buffer = sharded_buffer.allgather(0)

    assert replicated_buffer.layout == layout
    _assert_dbuffer_local_tensors_close(replicated_buffer, tensors)


def test_sharded_allgather_into_existing_buffer(distributed_setup):
    """Sharded buffers can all-gather directly into a preallocated replicated buffer."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(distributed_setup.device)
    sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Flat()])
    destination = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        layout=sharded_buffer.layout,
        dtype=sharded_buffer.dtype,
        device=sharded_buffer.local_buffer.device,
    )
    destination_data_ptr = destination.local_buffer.data_ptr()

    result = sharded_buffer.allgather(0, out=destination)

    assert result is destination
    assert destination.local_buffer.data_ptr() == destination_data_ptr
    _assert_dbuffer_local_tensors_close(destination, tensors)


def test_replicate_scatter_round_trip(distributed_setup):
    """Replicated buffers locally chunk into sharded buffers and all-gather back."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(distributed_setup.device)

    replicated_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate()])
    sharded_buffer = replicated_buffer.scatter(0, Flat())
    redistribute_destination = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        layout=replicated_buffer.layout,
        dtype=replicated_buffer.dtype,
        device=replicated_buffer.local_buffer.device,
    )
    redistributed_sharded_buffer = replicated_buffer.redistribute(
        [Flat()], out=redistribute_destination
    )

    assert sharded_buffer.placements == (Flat(),)
    assert redistributed_sharded_buffer is redistribute_destination
    assert redistributed_sharded_buffer.placements == (Flat(),)
    expected_sharded_local_numel = replicated_buffer.layout.size // distributed_setup.world_size
    assert sharded_buffer.offset == distributed_setup.rank * expected_sharded_local_numel
    assert (
        sharded_buffer.local_buffer.untyped_storage()
        is replicated_buffer.local_buffer.untyped_storage()
    )
    torch.testing.assert_close(
        sharded_buffer.local_buffer, redistributed_sharded_buffer.local_buffer, rtol=0, atol=0
    )
    _assert_dbuffer_local_tensors_close(sharded_buffer.allgather(0), tensors)


def test_partial_allreduce(distributed_setup):
    """Partial buffers all-reduce into replicated buffers."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    rank_scale = float(distributed_setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=distributed_setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=distributed_setup.device),
    ]
    partial_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial()])

    replicated_buffer = partial_buffer.allreduce(0)

    scale_sum = float(distributed_setup.world_size * (distributed_setup.world_size + 1) // 2)
    expected = [
        torch.full((5, 3), scale_sum, dtype=torch.float32, device=distributed_setup.device),
        torch.full((4,), scale_sum * 10, dtype=torch.float32, device=distributed_setup.device),
    ]
    _assert_dbuffer_local_tensors_close(replicated_buffer, expected)


def test_partial_allreduce_average(distributed_setup):
    """Partial buffers can all-reduce with AVG."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    rank_scale = float(distributed_setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=distributed_setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=distributed_setup.device),
    ]
    partial_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial("avg")])

    destination = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        layout=partial_buffer.layout,
        dtype=partial_buffer.dtype,
        device=partial_buffer.local_buffer.device,
    )
    replicated_buffer = partial_buffer.allreduce(0, out=destination)

    assert replicated_buffer is destination
    scale_average = float(distributed_setup.world_size + 1) / 2.0
    expected = [
        torch.full((5, 3), scale_average, dtype=torch.float32, device=distributed_setup.device),
        torch.full((4,), scale_average * 10, dtype=torch.float32, device=distributed_setup.device),
    ]
    _assert_dbuffer_local_tensors_close(replicated_buffer, expected)


def test_partial_reduce_scatter_to_flat(distributed_setup):
    """Partial buffers reduce-scatter into sharded buffers."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    rank_scale = float(distributed_setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=distributed_setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=distributed_setup.device),
    ]
    partial_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial()])
    layout = partial_buffer.layout

    destination = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        layout=partial_buffer.layout,
        dtype=partial_buffer.dtype,
        device=partial_buffer.local_buffer.device,
    )
    sharded_buffer = partial_buffer.reduce_scatter(0, Flat(), out=destination)
    replicated_buffer = sharded_buffer.allgather(0)

    assert sharded_buffer is destination
    assert sharded_buffer.placements == (Flat(),)
    assert sharded_buffer.layout == layout
    assert replicated_buffer.layout == layout
    scale_sum = float(distributed_setup.world_size * (distributed_setup.world_size + 1) // 2)
    expected_tensors = [
        torch.full((5, 3), scale_sum, dtype=torch.float32, device=distributed_setup.device),
        torch.full((4,), scale_sum * 10, dtype=torch.float32, device=distributed_setup.device),
    ]
    _assert_dbuffer_local_tensors_close(replicated_buffer, expected_tensors)


def test_partial_reduce_scatter_to_flat_average(distributed_setup):
    """Partial buffers can reduce-scatter with AVG."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    rank_scale = float(distributed_setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=distributed_setup.device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=distributed_setup.device),
    ]
    partial_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial("avg")])
    layout = partial_buffer.layout

    sharded_buffer = partial_buffer.reduce_scatter(0, Flat())
    replicated_buffer = sharded_buffer.allgather(0)

    assert sharded_buffer.placements == (Flat(),)
    assert sharded_buffer.layout == layout
    assert replicated_buffer.layout == layout
    scale_average = float(distributed_setup.world_size + 1) / 2.0
    expected_tensors = [
        torch.full((5, 3), scale_average, dtype=torch.float32, device=distributed_setup.device),
        torch.full((4,), scale_average * 10, dtype=torch.float32, device=distributed_setup.device),
    ]
    _assert_dbuffer_local_tensors_close(replicated_buffer, expected_tensors)


def test_partial_reduce_scatter_to_flat_average_without_symm_mem_detector(
    distributed_setup, monkeypatch
):
    """Ordinary AVG remains available when PyTorch lacks the symmetric-memory detector."""
    device, world_size = distributed_setup.device, distributed_setup.world_size
    mesh = init_device_mesh(device.type, (world_size,))
    monkeypatch.delattr(symm_mem, "is_symm_mem_tensor")
    rank_scale = float(distributed_setup.rank + 1)
    partial_buffer = DBuffer.distribute_tensors(
        [torch.full((5, 3), rank_scale, dtype=torch.float32, device=device)], mesh, [Partial("avg")]
    )

    replicated_buffer = partial_buffer.reduce_scatter(0, Flat()).allgather(0)

    expected = torch.full((5, 3), (world_size + 1) / 2.0, dtype=torch.float32, device=device)
    _assert_dbuffer_local_tensors_close(replicated_buffer, [expected])


def test_symmetric_memory_partial_reduce_scatter_to_flat_average(distributed_setup):
    """Symmetric-memory reduce-scatter preserves AVG semantics."""
    device, world_size = distributed_setup.device, distributed_setup.world_size
    mesh = init_device_mesh(device.type, (world_size,))
    dist.barrier(device_ids=[device.index])
    rank_scale = float(distributed_setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=device),
    ]
    pool = symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool):
        partial_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial("avg")])
    assert symm_mem.is_symm_mem_tensor(partial_buffer.local_buffer)

    sharded_buffer = partial_buffer.reduce_scatter(0, Flat())
    replicated_buffer = sharded_buffer.allgather(0)

    scale_average = (world_size + 1) / 2.0
    expected_tensors = [
        torch.full((5, 3), scale_average, dtype=torch.float32, device=device),
        torch.full((4,), scale_average * 10, dtype=torch.float32, device=device),
    ]
    _assert_dbuffer_local_tensors_close(replicated_buffer, expected_tensors)


def test_symmetric_memory_partial_reduce_scatter_to_flat_sum(distributed_setup):
    """Symmetric-memory reduce-scatter preserves explicit SUM semantics."""
    device, world_size = distributed_setup.device, distributed_setup.world_size
    mesh = init_device_mesh(device.type, (world_size,))
    dist.barrier(device_ids=[device.index])
    rank_scale = float(distributed_setup.rank + 1)
    tensors = [
        torch.full((5, 3), rank_scale, dtype=torch.float32, device=device),
        torch.full((4,), rank_scale * 10, dtype=torch.float32, device=device),
    ]
    pool = symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool):
        partial_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial("sum")])
    assert symm_mem.is_symm_mem_tensor(partial_buffer.local_buffer)

    sharded_buffer = partial_buffer.reduce_scatter(0, Flat())
    replicated_buffer = sharded_buffer.allgather(0)

    scale_sum = float(world_size * (world_size + 1) // 2)
    expected_tensors = [
        torch.full((5, 3), scale_sum, dtype=torch.float32, device=device),
        torch.full((4,), scale_sum * 10, dtype=torch.float32, device=device),
    ]
    _assert_dbuffer_local_tensors_close(replicated_buffer, expected_tensors)


def test_get_dtensor_from_sharded_buffer(distributed_setup):
    """Sharded DBuffer exposes per-tensor local shards as DTensors."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    tensors = _same_tensors_on_all_ranks(distributed_setup.device)
    sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Flat()])

    dtensor = sharded_buffer.get_dtensor(0)

    torch.testing.assert_close(
        dtensor.to_local(), sharded_buffer.get_local_tensor(0), rtol=0, atol=0
    )
    assert dtensor.shape == tensors[0].shape
    assert dtensor.placements == (Flat(),)


def test_2d_mesh_replicate_flat_round_trip(distributed_setup):
    """A 2D mesh can replicate on one axis and flat-shard on the other."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(distributed_setup.device)
    mesh = init_device_mesh(
        distributed_setup.device.type,
        (2, distributed_setup.world_size // 2),
        mesh_dim_names=("replicate", "flat"),
    )

    sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate(), Flat()])
    replicated_buffer = sharded_buffer.allgather(1)

    _assert_dbuffer_local_tensors_close(replicated_buffer, tensors)


def test_2d_mesh_flat_before_replicate_is_rejected(distributed_setup):
    """Flat axes must be a suffix to keep every local buffer contiguous."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    mesh = init_device_mesh(
        distributed_setup.device.type,
        (2, distributed_setup.world_size // 2),
        mesh_dim_names=("flat", "replicate"),
    )

    with pytest.raises(ValueError, match="Flat placements must be a suffix"):
        DBuffer(
            mesh=mesh,
            placements=[Flat(), Replicate()],
            layout=GlobalLayout.build([torch.Size((6, 4))], dp_size=mesh.size()),
            dtype=torch.float32,
            device=distributed_setup.device,
        )


def test_2d_mesh_shards_across_all_ranks(distributed_setup):
    """Multiple Flat axes shard local storage by the product of their mesh sizes."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(distributed_setup.device)
    mesh = init_device_mesh(
        distributed_setup.device.type,
        (2, distributed_setup.world_size // 2),
        mesh_dim_names=("dp_outer", "dp_inner"),
    )
    fully_sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Flat(), Flat()])

    assert fully_sharded_buffer.layout.tensor_shapes == tuple(tensor.shape for tensor in tensors)
    expected_local_numel = fully_sharded_buffer.layout.size // mesh.size()
    expected_inner_axis_shard_numel = fully_sharded_buffer.layout.size // mesh.size(1)
    expected_offset = (
        mesh.get_local_rank(1) * expected_inner_axis_shard_numel
        + mesh.get_local_rank(0) * expected_local_numel
    )
    assert fully_sharded_buffer.offset == expected_offset
    assert (
        fully_sharded_buffer.local_buffer.numel() == fully_sharded_buffer.layout.size // mesh.size()
    )
    for index, _ in enumerate(tensors):
        assert fully_sharded_buffer.get_local_tensor(index).is_contiguous()


def test_2d_mesh_partial_flat_reduce_scatter_to_flat_flat(distributed_setup):
    """Partial+Flat reduce-scatter reduces the existing Flat local shard."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    mesh = init_device_mesh(
        distributed_setup.device.type,
        (2, distributed_setup.world_size // 2),
        mesh_dim_names=("dp_outer", "dp_inner"),
    )
    outer_scale = float(mesh.get_local_rank(0) + 1)
    tensors = [
        torch.full((6, 2), outer_scale, dtype=torch.float32, device=distributed_setup.device),
        torch.full((4,), outer_scale * 10, dtype=torch.float32, device=distributed_setup.device),
    ]

    partial_sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial(), Flat()])
    fully_sharded_buffer = partial_sharded_buffer.reduce_scatter(0, Flat())
    replicated_buffer = fully_sharded_buffer.allgather(0).allgather(1)

    assert fully_sharded_buffer.placements == (Flat(), Flat())
    expected_local_numel = fully_sharded_buffer.layout.size // mesh.size()
    expected_inner_axis_shard_numel = fully_sharded_buffer.layout.size // mesh.size(1)
    expected_offset = (
        mesh.get_local_rank(1) * expected_inner_axis_shard_numel
        + mesh.get_local_rank(0) * expected_local_numel
    )
    assert fully_sharded_buffer.offset == expected_offset
    assert (
        fully_sharded_buffer.local_buffer.numel()
        == partial_sharded_buffer.local_buffer.numel() // 2
    )

    outer_scale_sum = float(mesh.size(0) * (mesh.size(0) + 1) // 2)
    expected = [
        torch.full((6, 2), outer_scale_sum, dtype=torch.float32, device=distributed_setup.device),
        torch.full(
            (4,), outer_scale_sum * 10, dtype=torch.float32, device=distributed_setup.device
        ),
    ]
    _assert_dbuffer_local_tensors_close(replicated_buffer, expected)


def test_2d_mesh_replicate_flat_scatter_to_flat_flat(distributed_setup):
    """Replicate+Flat scatter chunks the existing Flat local shard."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(distributed_setup.device)
    mesh = init_device_mesh(
        distributed_setup.device.type,
        (2, distributed_setup.world_size // 2),
        mesh_dim_names=("dp_outer", "dp_inner"),
    )

    replicated_sharded_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate(), Flat()])
    fully_sharded_buffer = replicated_sharded_buffer.scatter(0, Flat())
    replicated_buffer = fully_sharded_buffer.allgather(0).allgather(1)

    assert fully_sharded_buffer.placements == (Flat(), Flat())
    expected_local_numel = fully_sharded_buffer.layout.size // mesh.size()
    expected_inner_axis_shard_numel = fully_sharded_buffer.layout.size // mesh.size(1)
    expected_offset = (
        mesh.get_local_rank(1) * expected_inner_axis_shard_numel
        + mesh.get_local_rank(0) * expected_local_numel
    )
    assert fully_sharded_buffer.offset == expected_offset
    assert (
        fully_sharded_buffer.local_buffer.numel()
        == replicated_sharded_buffer.local_buffer.numel() // 2
    )
    _assert_dbuffer_local_tensors_close(replicated_buffer, tensors)
