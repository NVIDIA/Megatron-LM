# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP global layout metadata."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.layout import GlobalLayout
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.range import Range


def test_layout_pads_to_lcm_times_dp_size_and_fills_gaps():
    """Layout offsets fill gaps and the total size is padded to LCM * DP size."""
    shapes = [torch.Size((5, 4)), torch.Size((2, 6)), torch.Size((3,))]
    layout = GlobalLayout.build_for_row_atomic(shapes, dp_size=2)

    assert layout.tensor_shapes == tuple(shapes)
    assert layout.tensor_to_offset == (0, 24, 20)
    assert layout.size == 48


def test_layout_aligns_fragment_offsets_to_rows():
    """Layout keeps small tensors aligned to their non-leading dimensions."""
    shapes = [torch.Size((4, 4)), torch.Size((1, 6))]
    layout = GlobalLayout.build_for_row_atomic(shapes, dp_size=2)

    assert layout.tensor_to_offset == (0, 18)
    assert layout.size == 24


def test_tensor_atomic_layout_preserves_tensor_ids_with_non_monotonic_owners():
    """Packing groups owners stably while keeping logical IDs and empty rank segments."""
    shapes = [torch.Size((4, 4)), torch.Size((3,)), torch.Size((2, 6)), torch.Size((7, 3))]
    tensor_owners = (1, 3, 0, 3)

    layout = GlobalLayout.build_for_tensor_atomic(shapes, dp_size=5, tensor_owners=tensor_owners)
    assert layout.tensor_shapes == tuple(shapes)
    assert layout.tensor_to_offset == (12, 28, 0, 31)
    assert layout.rank_to_offset == (0, 12, 28, 28, 52)
    assert layout.size == 52


@pytest.mark.parametrize("tensor_owners", [(), (0,), (0, 1, 0)])
def test_tensor_atomic_layout_requires_one_owner_per_tensor(tensor_owners):
    """The tensor-atomic builder requires exactly one owner per tensor."""
    with pytest.raises(ValueError, match="number of tensor owners"):
        GlobalLayout.build_for_tensor_atomic([(3,), (4,)], dp_size=2, tensor_owners=tensor_owners)


def test_tensor_atomic_layout_accepts_empty_assignments_for_no_tensors():
    """An explicit empty assignment produces empty rank segments."""
    layout = GlobalLayout.build_for_tensor_atomic([], dp_size=2, tensor_owners=())
    assert layout.tensor_shapes == ()
    assert layout.tensor_to_offset == ()
    assert layout.rank_to_offset == (0, 0)
    assert layout.size == 0


@pytest.mark.parametrize("owner", [-1, 2, 0.5, True, "0", None])
def test_tensor_atomic_layout_rejects_invalid_owners(owner):
    """Owner ranks must be integers within the layout's DP mesh."""
    with pytest.raises(ValueError, match="integer within the range"):
        GlobalLayout.build_for_tensor_atomic([(3,)], dp_size=2, tensor_owners=(owner,))


def test_get_local_range_rejects_mesh_size_mismatch(distributed_setup):
    "A layout planned for one DP size cannot be sharded over a mesh of another size"
    if distributed_setup.world_size < 4:
        pytest.skip("get_local_range mesh-size mismatch test requires at least 4 ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (4,))
    if mesh.get_coordinate() is None:
        pytest.skip("Rank is outside the 4-rank test mesh.")

    layout = GlobalLayout.build_for_row_atomic([torch.Size((4, 4)), torch.Size((2, 4))], dp_size=2)
    assert layout.rank_to_offset == (0, 12)
    assert layout.get_local_range(mesh, [Replicate()]) == Range(0, 24)
    with pytest.raises(ValueError, match="built for 2 shards"):
        layout.get_local_range(mesh, [RowAtomic()])

    # A layout planned for more shards than the mesh has ranks hands each rank a
    # contiguous run of segments, matching the original uniform split.
    layout = GlobalLayout.build_for_row_atomic([torch.Size((8, 4))], dp_size=8)
    assert layout.rank_to_offset == tuple(range(0, 32, 4))
    rank = mesh.get_local_rank(0)
    assert layout.get_local_range(mesh, [RowAtomic()]) == Range(rank * 8, 8)


def test_get_tensor_range():
    """Return the selected tensor's offset and full element count."""
    layout = GlobalLayout(
        tensor_shapes=(torch.Size((2, 3)), torch.Size((4, 2))),
        tensor_to_offset=(4, 12),
        size=24,
        rank_to_offset=(0, 12),
    )
    assert layout.get_tensor_range(1) == Range(12, 8)
