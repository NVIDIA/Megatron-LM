# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP DBuffer."""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.dbuffer import (
    DBuffer,
    Flat,
    Partial,
    Replicate,
    compute_layout,
)


@pytest.fixture(scope="module")
def distributed_setup():
    """Set up torch.distributed for DBuffer tests."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        device_type = "cuda"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device_type = "cpu"
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    yield {"rank": rank, "world_size": world_size, "device_type": device_type, "device": device}

    if dist.is_initialized():
        dist.destroy_process_group()


def _mesh(setup):
    return DeviceMesh(setup["device_type"], list(range(setup["world_size"])))


def _same_tensors_on_all_ranks(device: torch.device) -> list[torch.Tensor]:
    return [
        torch.arange(21, dtype=torch.float32, device=device).reshape(7, 3),
        torch.arange(10, dtype=torch.float32, device=device).reshape(2, 5) + 100,
        torch.arange(7, dtype=torch.float32, device=device) + 200,
    ]


def _rank_scaled_tensors(setup) -> list[torch.Tensor]:
    scale = float(setup["rank"] + 1)
    return [
        torch.full((5, 3), scale, dtype=torch.float32, device=setup["device"]),
        torch.full((4,), scale * 10, dtype=torch.float32, device=setup["device"]),
    ]


def _assert_tensors_equal(buffer: DBuffer, expected: list[torch.Tensor]) -> None:
    for index, tensor in enumerate(expected):
        assert torch.equal(buffer.get_tensor(index), tensor)


def _expected_flat_offset(mesh: DeviceMesh, flat_axes: tuple[int, ...], global_size: int) -> int:
    offset = 0
    numel = global_size
    for axis in reversed(flat_axes):
        numel //= mesh.size(axis)
        offset += mesh.get_local_rank(axis) * numel
    return offset


def test_compute_layout_pads_to_lcm_times_dp_size_and_fills_gaps():
    """compute_layout returns element offsets and pads to LCM * DP size."""
    shapes = [torch.Size((5, 4)), torch.Size((2, 6)), torch.Size((3,))]

    layout = compute_layout(shapes, dp_size=2)

    assert layout.tensor_shapes == tuple(shapes)
    assert layout.tensor_to_offset == (0, 24, 20)
    assert layout.size == 48


def test_compute_layout_aligns_fragment_offsets_to_rows():
    """compute_layout keeps small tensors aligned to their non-leading dimensions."""
    shapes = [torch.Size((4, 4)), torch.Size((1, 6))]

    layout = compute_layout(shapes, dp_size=2)

    assert layout.tensor_to_offset == (0, 18)
    assert layout.size == 24


@pytest.mark.distributed
def test_constructor_allocates_local_buffer(distributed_setup):
    """DBuffer allocates local storage from shape, mesh, placement, dtype, and device."""
    setup = distributed_setup
    mesh = _mesh(setup)
    tensor_shapes = [torch.Size((7, 3)), torch.Size((2, 5)), torch.Size((7,))]
    mesh_size = int(mesh.mesh.numel())
    expected_layout = compute_layout(tensor_shapes, dp_size=mesh_size)

    replicated_buffer = DBuffer(
        mesh=mesh,
        placements=[Replicate()],
        tensor_shapes=tensor_shapes,
        dtype=torch.float32,
        device=setup["device"],
    )
    flat_buffer = DBuffer(
        mesh=mesh,
        placements=[Flat()],
        tensor_shapes=tensor_shapes,
        dtype=torch.float32,
        device=setup["device"],
    )

    assert replicated_buffer.layout == expected_layout
    assert flat_buffer.layout == expected_layout
    assert replicated_buffer.offset == 0
    assert flat_buffer.offset == _expected_flat_offset(mesh, (0,), expected_layout.size)
    assert replicated_buffer.local_buffer.numel() == expected_layout.size
    assert flat_buffer.local_buffer.numel() == expected_layout.size // setup["world_size"]
    assert flat_buffer.layout.size % (15 * mesh_size) == 0
    assert replicated_buffer.local_buffer.dtype == torch.float32
    assert flat_buffer.local_buffer.device == setup["device"]


@pytest.mark.distributed
def test_replicate_get_tensor_and_dtensor(distributed_setup):
    """Replicated DBuffer returns full local tensors and replicated DTensors."""
    setup = distributed_setup
    tensors = _same_tensors_on_all_ranks(setup["device"])

    buffer = DBuffer.distribute_tensors(tensors, _mesh(setup), [Replicate()])

    _assert_tensors_equal(buffer, tensors)
    dtensor = buffer.get_dtensor(0)
    assert torch.equal(dtensor.to_local(), tensors[0])


@pytest.mark.distributed
def test_distribute_tensors_moves_inputs_to_mesh_device(distributed_setup):
    """distribute_tensors moves full input tensors to the mesh device type."""
    setup = distributed_setup
    tensors = _same_tensors_on_all_ranks(torch.device("cpu"))

    buffer = DBuffer.distribute_tensors(tensors, _mesh(setup), [Replicate()])

    assert buffer.local_buffer.device == setup["device"]
    _assert_tensors_equal(buffer, [tensor.to(setup["device"]) for tensor in tensors])


@pytest.mark.distributed
def test_flat_allgather_round_trip(distributed_setup):
    """Flat buffers round-trip through all-gather as contiguous tensor fragments."""
    setup = distributed_setup
    tensors = _same_tensors_on_all_ranks(setup["device"])

    flat_buffer = DBuffer.distribute_tensors(tensors, _mesh(setup), [Flat()])
    layout = flat_buffer.layout
    for index in range(len(tensors)):
        local_tensor = flat_buffer.get_tensor(index)
        assert local_tensor.shape[1:] == tensors[index].shape[1:]
        assert local_tensor.is_contiguous()

    replicated_buffer = flat_buffer.allgather(0)

    assert replicated_buffer.layout == layout
    _assert_tensors_equal(replicated_buffer, tensors)


@pytest.mark.distributed
def test_replicate_scatter_round_trip(distributed_setup):
    """Replicated buffers locally chunk into Flat buffers and all-gather back."""
    setup = distributed_setup
    tensors = _same_tensors_on_all_ranks(setup["device"])

    replicated_buffer = DBuffer.distribute_tensors(tensors, _mesh(setup), [Replicate()])
    flat_buffer = replicated_buffer.scatter(0, Flat())
    redistributed_flat_buffer = replicated_buffer.redistribute([Flat()])

    assert flat_buffer.placements == (Flat(),)
    assert redistributed_flat_buffer.placements == (Flat(),)
    assert flat_buffer.offset == _expected_flat_offset(
        _mesh(setup), (0,), replicated_buffer.layout.size
    )
    assert torch.equal(flat_buffer.local_buffer, redistributed_flat_buffer.local_buffer)
    _assert_tensors_equal(flat_buffer.allgather(0), tensors)


@pytest.mark.distributed
def test_partial_allreduce(distributed_setup):
    """Partial buffers all-reduce into replicated buffers."""
    setup = distributed_setup
    tensors = _rank_scaled_tensors(setup)
    partial_buffer = DBuffer.distribute_tensors(tensors, _mesh(setup), [Partial()])

    replicated_buffer = partial_buffer.allreduce(0)

    scale_sum = float(setup["world_size"] * (setup["world_size"] + 1) // 2)
    expected = [
        torch.full((5, 3), scale_sum, dtype=torch.float32, device=setup["device"]),
        torch.full((4,), scale_sum * 10, dtype=torch.float32, device=setup["device"]),
    ]
    _assert_tensors_equal(replicated_buffer, expected)


@pytest.mark.distributed
def test_partial_reduce_scatter_to_flat(distributed_setup):
    """Partial buffers reduce-scatter into Flat buffers."""
    setup = distributed_setup
    tensors = _rank_scaled_tensors(setup)
    partial_buffer = DBuffer.distribute_tensors(tensors, _mesh(setup), [Partial()])
    layout = partial_buffer.layout

    flat_buffer = partial_buffer.reduce_scatter(0, Flat())
    replicated_buffer = flat_buffer.allgather(0)

    assert flat_buffer.layout == layout
    assert replicated_buffer.layout == layout
    scale_sum = float(setup["world_size"] * (setup["world_size"] + 1) // 2)
    expected = [
        torch.full((5, 3), scale_sum, dtype=torch.float32, device=setup["device"]),
        torch.full((4,), scale_sum * 10, dtype=torch.float32, device=setup["device"]),
    ]
    _assert_tensors_equal(replicated_buffer, expected)


@pytest.mark.distributed
def test_get_dtensor_for_flat_buffer(distributed_setup):
    """Flat DBuffer exposes per-tensor local shards as DTensors."""
    setup = distributed_setup
    tensors = _same_tensors_on_all_ranks(setup["device"])
    flat_buffer = DBuffer.distribute_tensors(tensors, _mesh(setup), [Flat()])

    dtensor = flat_buffer.get_dtensor(0)

    assert torch.equal(dtensor.to_local(), flat_buffer.get_tensor(0))
    assert tuple(dtensor.shape) == tuple(tensors[0].shape)


@pytest.mark.distributed
def test_2d_mesh_replicate_flat_round_trip(distributed_setup):
    """A 2D mesh can replicate on one axis and flat-shard on the other."""
    setup = distributed_setup
    if setup["world_size"] < 4 or setup["world_size"] % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(setup["device"])
    mesh_ids = torch.arange(setup["world_size"]).reshape(2, setup["world_size"] // 2)
    mesh = DeviceMesh(setup["device_type"], mesh_ids, mesh_dim_names=("replicate", "flat"))

    flat_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate(), Flat()])
    replicated_buffer = flat_buffer.allgather("flat")

    _assert_tensors_equal(replicated_buffer, tensors)


@pytest.mark.distributed
def test_2d_mesh_flat_before_replicate_is_rejected(distributed_setup):
    """Flat axes must be a suffix to keep every local buffer contiguous."""
    setup = distributed_setup
    if setup["world_size"] < 4 or setup["world_size"] % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    mesh_ids = torch.arange(setup["world_size"]).reshape(2, setup["world_size"] // 2)
    mesh = DeviceMesh(setup["device_type"], mesh_ids, mesh_dim_names=("flat", "replicate"))

    with pytest.raises(ValueError, match="Flat placements must be a suffix"):
        DBuffer(
            mesh=mesh,
            placements=[Flat(), Replicate()],
            tensor_shapes=[torch.Size((6, 4))],
            dtype=torch.float32,
            device=setup["device"],
        )


@pytest.mark.distributed
def test_2d_mesh_two_flat_axes_use_product_shard_size(distributed_setup):
    """Multiple Flat axes shard local storage by the product of their mesh sizes."""
    setup = distributed_setup
    if setup["world_size"] < 4 or setup["world_size"] % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(setup["device"])
    mesh_ids = torch.arange(setup["world_size"]).reshape(2, setup["world_size"] // 2)
    mesh = DeviceMesh(setup["device_type"], mesh_ids, mesh_dim_names=("dp_outer", "dp_inner"))
    expected_layout = compute_layout(
        [tensor.shape for tensor in tensors], dp_size=int(mesh.mesh.numel())
    )

    flat_buffer = DBuffer.distribute_tensors(tensors, mesh, [Flat(), Flat()])

    assert flat_buffer.layout == expected_layout
    assert flat_buffer.offset == _expected_flat_offset(mesh, (0, 1), expected_layout.size)
    assert flat_buffer.local_buffer.numel() == expected_layout.size // int(mesh.mesh.numel())
    for index in range(len(tensors)):
        assert flat_buffer.get_tensor(index).is_contiguous()


@pytest.mark.distributed
def test_2d_mesh_partial_flat_reduce_scatter_uses_local_shard(distributed_setup):
    """Partial+Flat reduce-scatter reduces the existing Flat local shard."""
    setup = distributed_setup
    if setup["world_size"] < 4 or setup["world_size"] % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    mesh_ids = torch.arange(setup["world_size"]).reshape(2, setup["world_size"] // 2)
    mesh = DeviceMesh(setup["device_type"], mesh_ids, mesh_dim_names=("dp_outer", "dp_inner"))
    outer_scale = float(mesh.get_local_rank(0) + 1)
    tensors = [
        torch.full((6, 2), outer_scale, dtype=torch.float32, device=setup["device"]),
        torch.full((4,), outer_scale * 10, dtype=torch.float32, device=setup["device"]),
    ]

    partial_flat_buffer = DBuffer.distribute_tensors(tensors, mesh, [Partial(), Flat()])
    flat_flat_buffer = partial_flat_buffer.reduce_scatter("dp_outer", Flat())
    replicated_buffer = flat_flat_buffer.allgather("dp_outer").allgather("dp_inner")

    assert flat_flat_buffer.placements == (Flat(), Flat())
    assert flat_flat_buffer.offset == _expected_flat_offset(
        mesh, (0, 1), flat_flat_buffer.layout.size
    )
    assert flat_flat_buffer.local_buffer.numel() == partial_flat_buffer.local_buffer.numel() // 2

    outer_scale_sum = float(mesh.size(0) * (mesh.size(0) + 1) // 2)
    expected = [
        torch.full((6, 2), outer_scale_sum, dtype=torch.float32, device=setup["device"]),
        torch.full((4,), outer_scale_sum * 10, dtype=torch.float32, device=setup["device"]),
    ]
    _assert_tensors_equal(replicated_buffer, expected)


@pytest.mark.distributed
def test_2d_mesh_replicate_flat_scatter_uses_local_shard(distributed_setup):
    """Replicate+Flat scatter chunks the existing Flat local shard."""
    setup = distributed_setup
    if setup["world_size"] < 4 or setup["world_size"] % 2 != 0:
        pytest.skip("2D DBuffer test requires an even world size of at least 4.")

    tensors = _same_tensors_on_all_ranks(setup["device"])
    mesh_ids = torch.arange(setup["world_size"]).reshape(2, setup["world_size"] // 2)
    mesh = DeviceMesh(setup["device_type"], mesh_ids, mesh_dim_names=("dp_outer", "dp_inner"))

    replicate_flat_buffer = DBuffer.distribute_tensors(tensors, mesh, [Replicate(), Flat()])
    flat_flat_buffer = replicate_flat_buffer.scatter("dp_outer", Flat())
    replicated_buffer = flat_flat_buffer.allgather("dp_outer").allgather("dp_inner")

    assert flat_flat_buffer.placements == (Flat(), Flat())
    assert flat_flat_buffer.offset == _expected_flat_offset(
        mesh, (0, 1), flat_flat_buffer.layout.size
    )
    assert flat_flat_buffer.local_buffer.numel() == replicate_flat_buffer.local_buffer.numel() // 2
    _assert_tensors_equal(replicated_buffer, tensors)
