# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP global layout metadata."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import layout_builder
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.layout import GlobalLayout
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    RowAtomic,
    ShardLayout,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.range import Range


def test_layout_pads_to_lcm_times_dp_size_and_fills_gaps():
    """Layout offsets fill gaps and the total size is padded to LCM * DP size."""
    shapes = [torch.Size((5, 4)), torch.Size((2, 6)), torch.Size((3,))]
    layout = layout_builder.build_for_row_atomic(shapes, dp_size=2)

    assert layout.tensor_shapes == tuple(shapes)
    assert layout.tensor_to_offset == (0, 24, 20)
    assert layout.size == 48


def test_layout_aligns_fragment_offsets_to_rows():
    """Layout keeps small tensors aligned to their non-leading dimensions."""
    shapes = [torch.Size((4, 4)), torch.Size((1, 6))]
    layout = layout_builder.build_for_row_atomic(shapes, dp_size=2)

    assert layout.tensor_to_offset == (0, 18)
    assert layout.size == 24


def test_uneven_layout_packs_without_padding_and_snaps_to_row_boundaries():
    """The uneven builder stores exactly the data and rounds the even split to a row edge."""
    shapes = [torch.Size((2, 6)), torch.Size((4, 4)), torch.Size((1, 2))]
    layout = layout_builder.build_for_row_atomic_unpadding(shapes, dp_size=2)

    assert layout.tensor_shapes == tuple(shapes)
    assert layout.tensor_to_offset == (0, 12, 28)
    assert layout.size == 30
    # The even split at 15 lands inside P1 (row size 4) and snaps to its row boundary at 16.
    assert layout.rank_to_offset == (0, 16)
    assert not layout.has_equal_shard_sizes


def test_uneven_layout_keeps_blocks_whole():
    """Block boundaries, not row boundaries, are the snapping unit for BlockAtomic."""
    shapes = [torch.Size((4, 3)), torch.Size((6, 2))]
    layout = layout_builder.build_for_row_atomic_unpadding(shapes, dp_size=4, block_size=2)

    assert layout.size == 24
    # Even splits 6, 12, 18: P0 blocks are 6 elements, P1 blocks are 4 elements starting at 12,
    # so 18 is equidistant from 16 and 20 and rounds down.
    assert layout.rank_to_offset == (0, 6, 12, 16)
    assert [layout.rank_size(rank) for rank in range(4)] == [6, 6, 4, 8]


def test_uneven_layout_leaves_ranks_empty_when_boundaries_are_scarce():
    """Fewer row boundaries than ranks leaves some ranks with empty segments."""
    layout = layout_builder.build_for_row_atomic_unpadding([torch.Size((2, 8))], dp_size=4)

    assert layout.rank_to_offset == (0, 0, 8, 8)
    assert [layout.rank_size(rank) for rank in range(4)] == [0, 8, 0, 8]

    empty = layout_builder.build_for_row_atomic_unpadding([torch.Size((0, 8))], dp_size=3)
    assert empty.size == 0
    assert empty.rank_to_offset == (0, 0, 0)


@pytest.mark.parametrize(
    "build",
    [layout_builder.build_for_row_atomic, layout_builder.build_for_row_atomic_unpadding],
    ids=["padded", "unpadded"],
)
def test_row_atomic_builders_reject_invalid_shapes_and_sizes(build):
    """Both RowAtomic builders share the same shape and size validation."""
    with pytest.raises(ValueError, match="DP size must be positive"):
        build([torch.Size((4, 4))], dp_size=0)
    with pytest.raises(ValueError, match="Block size must be positive"):
        build([torch.Size((4, 4))], dp_size=2, block_size=0)
    with pytest.raises(ValueError, match="zero-sized non-leading dims"):
        build([torch.Size((4, 0))], dp_size=2)
    with pytest.raises(ValueError, match="must be divisible by block size"):
        build([torch.Size((6, 4))], dp_size=2, block_size=4)


def test_build_row_atomic_layout_dispatches_on_shard_layout():
    """The dispatcher selects the padded or unpadded builder and rejects anything else."""
    shapes = [torch.Size((2, 6)), torch.Size((4, 4)), torch.Size((1, 2))]
    padded = layout_builder.build_row_atomic_layout(
        shapes, dp_size=2, shard_layout=ShardLayout.PADDED
    )
    unpadded = layout_builder.build_row_atomic_layout(
        shapes, dp_size=2, shard_layout=ShardLayout.UNPADDED
    )
    assert padded == layout_builder.build_for_row_atomic(shapes, dp_size=2)
    assert unpadded == layout_builder.build_for_row_atomic_unpadding(shapes, dp_size=2)
    with pytest.raises(ValueError, match="Unsupported shard layout"):
        layout_builder.build_row_atomic_layout(shapes, dp_size=2, shard_layout="padded")


def test_validate_for_row_atomic_checks_boundaries_not_segment_sizes():
    """Unequal segments pass when every boundary is a whole number of rows into its tensor."""
    shapes = (torch.Size((4, 4)), torch.Size((2, 4)))
    GlobalLayout(
        tensor_shapes=shapes, tensor_to_offset=(0, 16), size=24, rank_to_offset=(0, 20)
    ).validate_for_row_atomic()
    # Tensor starts need not be globally aligned: the boundary at 15 is row-aligned
    # relative to the (3, 5) tensor that starts at 10.
    GlobalLayout(
        tensor_shapes=(torch.Size((2, 5)), torch.Size((3, 5))),
        tensor_to_offset=(0, 10),
        size=25,
        rank_to_offset=(0, 15),
    ).validate_for_row_atomic()

    split_row = GlobalLayout(
        tensor_shapes=shapes, tensor_to_offset=(0, 16), size=24, rank_to_offset=(0, 18)
    )
    with pytest.raises(ValueError, match="Rank offsets.*incompatible with alignment 4"):
        split_row.validate_for_row_atomic()

    split_block = GlobalLayout(
        tensor_shapes=shapes, tensor_to_offset=(0, 16), size=24, rank_to_offset=(0, 4)
    )
    split_block.validate_for_row_atomic()
    with pytest.raises(ValueError, match="Rank offsets.*incompatible with alignment 8"):
        split_block.validate_for_row_atomic(block_size=2)


def test_tensor_atomic_layout_preserves_tensor_ids_with_non_monotonic_owners():
    """Packing groups owners stably while keeping logical IDs and empty rank segments."""
    shapes = [torch.Size((4, 4)), torch.Size((3,)), torch.Size((2, 6)), torch.Size((7, 3))]
    tensor_owners = (1, 3, 0, 3)

    layout = layout_builder.build_for_tensor_atomic(shapes, dp_size=5, tensor_owners=tensor_owners)
    assert layout.tensor_shapes == tuple(shapes)
    assert layout.tensor_to_offset == (12, 28, 0, 31)
    assert layout.rank_to_offset == (0, 12, 28, 28, 52)
    assert layout.size == 52


@pytest.mark.parametrize("tensor_owners", [(), (0,), (0, 1, 0)])
def test_tensor_atomic_layout_requires_one_owner_per_tensor(tensor_owners):
    """The tensor-atomic builder requires exactly one owner per tensor."""
    with pytest.raises(ValueError, match="number of tensor owners"):
        layout_builder.build_for_tensor_atomic([(3,), (4,)], dp_size=2, tensor_owners=tensor_owners)


def test_tensor_atomic_layout_accepts_empty_assignments_for_no_tensors():
    """An explicit empty assignment produces empty rank segments."""
    layout = layout_builder.build_for_tensor_atomic([], dp_size=2, tensor_owners=())
    assert layout.tensor_shapes == ()
    assert layout.tensor_to_offset == ()
    assert layout.rank_to_offset == (0, 0)
    assert layout.size == 0


@pytest.mark.parametrize("owner", [-1, 2, 0.5, True, "0", None])
def test_tensor_atomic_layout_rejects_invalid_owners(owner):
    """Owner ranks must be integers within the layout's DP mesh."""
    with pytest.raises(ValueError, match="integer within the range"):
        layout_builder.build_for_tensor_atomic([(3,)], dp_size=2, tensor_owners=(owner,))


def test_get_local_range_rejects_mesh_size_mismatch(distributed_setup):
    "A layout planned for one DP size cannot be sharded over a mesh of another size"
    if distributed_setup.world_size < 4:
        pytest.skip("get_local_range mesh-size mismatch test requires at least 4 ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (4,))
    if mesh.get_coordinate() is None:
        pytest.skip("Rank is outside the 4-rank test mesh.")

    layout = layout_builder.build_for_row_atomic(
        [torch.Size((4, 4)), torch.Size((2, 4))], dp_size=2
    )
    assert layout.rank_to_offset == (0, 12)
    assert layout.get_local_range(mesh, [Replicate()]) == Range(0, 24)
    with pytest.raises(ValueError, match="built for 2 shards"):
        layout.get_local_range(mesh, [RowAtomic()])

    # A layout planned for more shards than the mesh has ranks hands each rank a
    # contiguous run of segments, matching the original uniform split.
    layout = layout_builder.build_for_row_atomic([torch.Size((8, 4))], dp_size=8)
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
