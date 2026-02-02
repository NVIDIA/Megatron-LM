# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""
Unit tests for Megatron-FSDP uneven_dtensor functions.

Run with torchrun:
    torchrun --nproc_per_node=4 pytest test_mfsdp_uneven_dtensor.py -v
    torchrun --nproc_per_node=8 pytest test_mfsdp_uneven_dtensor.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    split_dtensor,
    uneven_dtensor_to_full_tensor,
)


# Pytest fixtures for distributed setup
@pytest.fixture(scope="module")
def distributed_setup():
    """Setup distributed environment for pytest with proper CUDA device assignment."""
    # Check if running under torchrun
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running in distributed mode. Use torchrun to run this test.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Determine device type and set CUDA device
    if torch.cuda.is_available():
        device_type = "cuda"
        # Set CUDA device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device_type = "cpu"
        device = torch.device("cpu")
        backend = "gloo"

    # Initialize process group if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    yield {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device_type": device_type,
        "device": device,
    }

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------- Helper: distributed setup ----------


@pytest.fixture(scope="module")
def distributed_setup():
    """Setup torch.distributed and CUDA device for torchrun + pytest."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")

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

    yield {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device_type": device_type,
        "device": device,
    }

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------- Helper: broadcast-based global tensor creation ----------


def make_global_randn(shape, dtype=torch.float32, device=torch.device("cpu")):
    """
    Create the same random tensor on all ranks by generating on rank 0
    and broadcasting to everyone.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Make sure shape is known on all ranks (it is, since passed as arg)
    if rank == 0:
        tensor = torch.randn(*shape, dtype=dtype, device=device)
    else:
        # allocate empty tensor, then broadcast into it
        tensor = torch.empty(*shape, dtype=dtype, device=device)

    dist.broadcast(tensor, src=0)
    return tensor


def make_global_arange(shape, dtype=torch.float32, device=torch.device("cpu")):
    """Same idea as make_global_randn, but deterministic arange."""
    rank = dist.get_rank()
    if rank == 0:
        tensor = torch.arange(
            torch.prod(torch.tensor(shape)).item(), dtype=dtype, device=device
        ).reshape(*shape)
    else:
        tensor = torch.empty(*shape, dtype=dtype, device=device)
    dist.broadcast(tensor, src=0)
    return tensor


# ---------- Tests ----------

# ---------------------------------------------------------------------------
# uneven_dtensor_to_full_tensor tests
# ---------------------------------------------------------------------------


@pytest.mark.distributed
def test_basic_shard_gather(distributed_setup):
    """Basic 1D shard gather, world_size-agnostic."""
    setup = distributed_setup
    mesh = DeviceMesh(setup["device_type"], list(range(setup["world_size"])))

    global_tensor = make_global_arange((4, 3), dtype=torch.float32, device=setup["device"])
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor)


@pytest.mark.distributed
def test_replicated_dtensor(distributed_setup):
    """Replicated placement should reconstruct the same tensor."""
    setup = distributed_setup
    mesh = DeviceMesh(setup["device_type"], list(range(setup["world_size"])))

    global_tensor = make_global_randn((8, 4), device=setup["device"])
    dtensor = distribute_tensor(global_tensor, mesh, [Replicate()])

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor)


@pytest.mark.distributed
def test_uneven_sharding_dim0(distributed_setup):
    """Uneven sharding on dim 0 using manual split + DTensor.from_local."""
    setup = distributed_setup
    world_size = setup["world_size"]
    mesh = DeviceMesh(setup["device_type"], list(range(world_size)))

    # size intentionally not divisible by world_size
    rows = world_size * 3 + 2
    global_tensor = make_global_arange((rows, 4), dtype=torch.float32, device=setup["device"])

    shard = Shard(0)
    local_list, _ = shard._split_tensor(
        global_tensor, world_size, with_padding=False, contiguous=True
    )

    local = local_list[setup["rank"]]

    dtensor = DTensor.from_local(
        local, mesh, (Shard(0),), shape=global_tensor.size(), stride=global_tensor.stride()
    )

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor)


@pytest.mark.distributed
def test_uneven_sharding_dim1(distributed_setup):
    """Uneven sharding on dim 1 using manual split + DTensor.from_local."""
    setup = distributed_setup
    world_size = setup["world_size"]
    mesh = DeviceMesh(setup["device_type"], list(range(world_size)))

    cols = world_size * 2 + 1
    global_tensor = make_global_randn((8, cols), device=setup["device"])

    shard = Shard(1)
    local_list, _ = shard._split_tensor(
        global_tensor, world_size, with_padding=False, contiguous=True
    )
    local = local_list[setup["rank"]]

    dtensor = DTensor.from_local(
        local, mesh, (Shard(1),), shape=global_tensor.size(), stride=global_tensor.stride()
    )

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor)


@pytest.mark.distributed
def test_2d_mesh_shard_and_replicate(distributed_setup):
    """2D mesh with Shard + Replicate, for world_size=4 or 8."""
    setup = distributed_setup
    world_size = setup["world_size"]

    if world_size == 4:
        mesh_shape = (2, 2)
    elif world_size == 8:
        mesh_shape = (2, 4)
    else:
        pytest.skip(f"2D mesh test expects world_size 4 or 8, got {world_size}")

    mesh_ids = torch.arange(world_size).reshape(mesh_shape)
    mesh = DeviceMesh(setup["device_type"], mesh_ids)

    global_tensor = make_global_randn((16, 12), device=setup["device"])
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0), Replicate()])

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_multiple_sharded_dims_even(distributed_setup):
    """Shard on two dimensions with even splits, 2D mesh."""
    setup = distributed_setup
    world_size = setup["world_size"]

    if world_size == 4:
        mesh_shape = (2, 2)
    elif world_size == 8:
        mesh_shape = (2, 4)
    else:
        pytest.skip(f"2D mesh test expects world_size 4 or 8, got {world_size}")

    mesh_ids = torch.arange(world_size).reshape(mesh_shape)
    mesh = DeviceMesh(setup["device_type"], mesh_ids)

    global_tensor = make_global_randn((16, 24), device=setup["device"])
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0), Shard(1)])

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_multiple_sharded_dims_uneven(distributed_setup):
    """Shard on two dimensions with uneven sizes using manual splitting."""
    setup = distributed_setup
    world_size = setup["world_size"]

    if world_size == 4:
        mesh_shape = (2, 2)
        dim0 = 13
        dim1 = 15
    elif world_size == 8:
        mesh_shape = (2, 4)
        dim0 = 13
        dim1 = 23
    else:
        pytest.skip(f"2D mesh test expects world_size 4 or 8, got {world_size}")

    mesh_ids = torch.arange(world_size).reshape(mesh_shape)
    mesh = DeviceMesh(setup["device_type"], mesh_ids)

    global_tensor = make_global_randn((dim0, dim1), device=setup["device"])

    shard0 = Shard(0)
    shard1 = Shard(1)

    list0, _ = shard0._split_tensor(
        global_tensor, mesh_shape[0], with_padding=False, contiguous=True
    )
    rank0 = setup["rank"] // mesh_shape[1]
    intermediate = list0[rank0]

    list1, _ = shard1._split_tensor(
        intermediate, mesh_shape[1], with_padding=False, contiguous=True
    )
    rank1 = setup["rank"] % mesh_shape[1]
    local = list1[rank1]

    dtensor = DTensor.from_local(
        local, mesh, (Shard(0), Shard(1)), shape=global_tensor.size(), stride=global_tensor.stride()
    )

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_3d_tensor_two_shards(distributed_setup):
    """3D tensor with sharding on two dims on 2D mesh."""
    setup = distributed_setup
    world_size = setup["world_size"]

    if world_size == 4:
        mesh_shape = (2, 2)
    elif world_size == 8:
        mesh_shape = (2, 4)
    else:
        pytest.skip(f"2D mesh test expects world_size 4 or 8, got {world_size}")

    mesh_ids = torch.arange(world_size).reshape(mesh_shape)
    mesh = DeviceMesh(setup["device_type"], mesh_ids)

    global_tensor = make_global_randn((16, 8, 24), device=setup["device"])
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0), Shard(2)])

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_different_dtypes(distributed_setup):
    """Verify correctness across several dtypes."""
    setup = distributed_setup
    mesh = DeviceMesh(setup["device_type"], list(range(setup["world_size"])))

    for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
        global_tensor = make_global_arange((4, 4), dtype=dtype, device=setup["device"])
        dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])

        gathered = uneven_dtensor_to_full_tensor(dtensor)

        assert gathered.dtype == dtype
        assert torch.equal(gathered, global_tensor)


@pytest.mark.distributed
def test_large_tensor(distributed_setup):
    """Scalability: larger tensor, sharded on dim 0."""
    setup = distributed_setup
    mesh = DeviceMesh(setup["device_type"], list(range(setup["world_size"])))

    global_tensor = make_global_randn((1024, 512), device=setup["device"])
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_3d_tensor_single_shard(distributed_setup):
    """3D tensor with sharding on dim 0 only."""
    setup = distributed_setup
    mesh = DeviceMesh(setup["device_type"], list(range(setup["world_size"])))

    global_tensor = make_global_randn((8, 6, 4), device=setup["device"])
    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor)


@pytest.mark.distributed
def test_error_on_invalid_input(distributed_setup):
    """Non-DTensor input should raise TypeError."""
    x = torch.randn(4, 4)
    with pytest.raises(TypeError):
        uneven_dtensor_to_full_tensor(x)


@pytest.mark.distributed
def test_backward_compatibility(distributed_setup):
    """Check gathered tensor can participate in autograd."""
    setup = distributed_setup
    mesh = DeviceMesh(setup["device_type"], list(range(setup["world_size"])))

    global_tensor = make_global_randn((8, 4), device=setup["device"])
    global_tensor.requires_grad_(True)

    dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])
    gathered = uneven_dtensor_to_full_tensor(dtensor)

    loss = gathered.sum()
    assert loss is not None


@pytest.mark.distributed
def test_strided_shard_2d_mesh(distributed_setup):
    """
    Test _StridedShard on a 2D mesh, sharding the same dimension across two mesh dims.
    This is similar to TP + DP style strided sharding.
    """
    setup = distributed_setup
    world_size = setup["world_size"]

    if world_size == 4:
        mesh_shape = (2, 2)
    elif world_size == 8:
        mesh_shape = (2, 4)
    else:
        pytest.skip(f"2D mesh test expects world_size 4 or 8, got {world_size}")

    mesh_ids = torch.arange(world_size).reshape(mesh_shape)
    mesh = DeviceMesh(setup["device_type"], mesh_ids)

    rows = 8
    cols = 8
    global_tensor = make_global_randn((rows, cols), device=setup["device"])

    # Shard dim 0 over both mesh dims; one of them is encoded as _StridedShard.
    # Example pattern: [Shard(0), _StridedShard(0, split_factor=mesh_shape[0])]
    # so that combined, dim 0 is split across mesh_shape[0] * mesh_shape[1] ranks. [web:146]
    placements = [Shard(0), _StridedShard(0, split_factor=mesh_shape[0])]

    dtensor = distribute_tensor(global_tensor, mesh, placements)

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == global_tensor.shape
    assert torch.allclose(gathered, global_tensor, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_wild_random_uneven_shards(distributed_setup):
    """
    Wild random uneven sharding:
    - We choose a global shape.
    - We randomly assign each rank a slice length (including 0) along a sharded dimension.
    - Each rank holds arbitrary local data of its length.
    - We build a DTensor via from_local + explicit offsets metadata (through your metadata updater),
      and then use uneven_dtensor_to_full_tensor to reconstruct.
    """
    setup = distributed_setup
    rank = setup["rank"]
    world_size = setup["world_size"]
    mesh = DeviceMesh(setup["device_type"], list(range(world_size)))

    # Global logical shape on dim 1 (sharded dim)
    rows = 2
    max_local = 8

    if rank == 0:
        # Random lengths per rank, allowing zero
        lengths = torch.randint(
            low=0, high=max_local + 1, size=(world_size,), device=setup["device"]
        )
        # Ensure at least one element globally to avoid degenerate zero-sized tensor
        if lengths.sum().item() == 0:
            lengths[0] = 1
    else:
        lengths = torch.empty(world_size, dtype=torch.int64, device=setup["device"])

    # Broadcast lengths and total from rank 0
    dist.broadcast(lengths, src=0)

    total = int(lengths.sum().item())

    # Now we know global shape: rows x total
    global_cols = total

    # Build a “reference” tensor on all ranks via broadcast from rank 0
    ref_global = make_global_arange(
        (rows, global_cols), dtype=torch.float32, device=setup["device"]
    )

    # Local slice for this rank is a contiguous segment in dim 1
    start = int(lengths[:rank].sum().item())
    length = int(lengths[rank].item())
    end = start + length

    if length > 0:
        local = ref_global[:, start:end].clone()
    else:
        # zero-sized local shard
        local = torch.empty((rows, 0), dtype=ref_global.dtype, device=ref_global.device)

    # Construct DTensor with explicit shape/stride; offsets handled by your metadata helper
    dtensor = DTensor.from_local(
        local, mesh, (Shard(1),), shape=(rows, global_cols), stride=(global_cols, 1)
    )

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == ref_global.shape
    assert torch.allclose(gathered, ref_global, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_wild_random_uneven_shards_multi_dim(distributed_setup):
    """
    Wild uneven shards across a 2D mesh and 2 sharded dims, including zero-sized shards.
    This stresses:
      - multiple mesh dims
      - multiple sharded dims
      - varying per-rank shapes
    """
    setup = distributed_setup
    rank = setup["rank"]
    world_size = setup["world_size"]

    if world_size == 4:
        mesh_shape = (2, 2)
    elif world_size == 8:
        mesh_shape = (2, 4)
    else:
        pytest.skip(f"2D mesh test expects world_size 4 or 8, got {world_size}")

    mesh_ids = torch.arange(world_size).reshape(mesh_shape)
    mesh = DeviceMesh(setup["device_type"], mesh_ids)

    # Logical global shape
    base_rows = 3
    base_cols = 5

    # Each mesh row gets its own random row-count; each mesh col gets its own random col-count
    if rank == 0:
        row_chunks = torch.randint(
            low=0, high=base_rows + 2, size=(mesh_shape[0],), device=setup["device"]
        )
        col_chunks = torch.randint(
            low=0, high=base_cols + 3, size=(mesh_shape[1],), device=setup["device"]
        )
        if row_chunks.sum().item() == 0:
            row_chunks[0] = 1
        if col_chunks.sum().item() == 0:
            col_chunks[0] = 1
    else:
        row_chunks = torch.empty(mesh_shape[0], dtype=torch.int64, device=setup["device"])
        col_chunks = torch.empty(mesh_shape[1], dtype=torch.int64, device=setup["device"])

    dist.broadcast(row_chunks, src=0)
    dist.broadcast(col_chunks, src=0)

    total_rows = int(row_chunks.sum().item())
    total_cols = int(col_chunks.sum().item())

    # Global reference tensor
    ref_global = make_global_arange(
        (total_rows, total_cols), dtype=torch.float32, device=setup["device"]
    )

    # Determine which row/col block this rank owns
    mesh_row = rank // mesh_shape[1]
    mesh_col = rank % mesh_shape[1]

    row_start = int(row_chunks[:mesh_row].sum().item())
    row_len = int(row_chunks[mesh_row].item())
    row_end = row_start + row_len

    col_start = int(col_chunks[:mesh_col].sum().item())
    col_len = int(col_chunks[mesh_col].item())
    col_end = col_start + col_len

    if row_len > 0 and col_len > 0:
        local = ref_global[row_start:row_end, col_start:col_end].clone()
    else:
        local = torch.empty((row_len, col_len), dtype=ref_global.dtype, device=ref_global.device)

    dtensor = DTensor.from_local(
        local, mesh, (Shard(0), Shard(1)), shape=(total_rows, total_cols), stride=(total_cols, 1)
    )

    gathered = uneven_dtensor_to_full_tensor(dtensor)

    assert gathered.shape == ref_global.shape
    assert torch.allclose(gathered, ref_global, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# split_dtensor tests
# ---------------------------------------------------------------------------


@pytest.mark.distributed
def test_split_dtensor_even_shard_dim0(distributed_setup):
    """Even split along sharded dim 0, verify each split matches torch.split of global tensor."""
    setup = distributed_setup
    mesh = DeviceMesh(setup["device_type"], list(range(setup["world_size"])))

    global_tensor = make_global_arange((16, 4), dtype=torch.float32, device=setup["device"])
    dt = distribute_tensor(global_tensor, mesh, [Shard(0)])

    # Split evenly into size 4 along dim 0
    splits = list(split_dtensor(dt, 4, dim=0, update_uneven_dtensor_chunk_meta=True))

    assert len(splits) == 4

    # Reference splits on full tensor
    ref_splits = torch.split(global_tensor, 4, dim=0)
    for i, (chunk_dt, ref) in enumerate(zip(splits, ref_splits)):
        gathered = uneven_dtensor_to_full_tensor(chunk_dt)
        ref = ref.to(gathered.device)
        assert gathered.shape == ref.shape, f"split {i} shape mismatch"
        assert torch.allclose(gathered, ref), f"split {i} content mismatch"


@pytest.mark.distributed
def test_split_dtensor_uneven_sections_dim1(distributed_setup):
    """List-of-sections split along dim 1 on a sharded DTensor."""
    setup = distributed_setup
    mesh = DeviceMesh(setup["device_type"], list(range(setup["world_size"])))

    global_tensor = make_global_randn((8, 13), device=setup["device"])
    dt = distribute_tensor(global_tensor, mesh, [Shard(0)])

    sections = [3, 5, 5]  # sum == 13
    splits = list(split_dtensor(dt, sections, dim=1, update_uneven_dtensor_chunk_meta=True))
    assert len(splits) == len(sections)

    ref_splits = torch.split(global_tensor, sections, dim=1)
    for i, (chunk_dt, ref) in enumerate(zip(splits, ref_splits)):
        gathered = uneven_dtensor_to_full_tensor(chunk_dt)
        ref = ref.to(gathered.device)
        assert gathered.shape == ref.shape
        assert torch.allclose(gathered, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_split_dtensor_replicate_placement(distributed_setup):
    """Splitting a replicated DTensor should behave like splitting the global tensor, no redistribution."""
    setup = distributed_setup
    mesh = DeviceMesh(setup["device_type"], list(range(setup["world_size"])))

    global_tensor = make_global_randn((6, 10), device=setup["device"])
    dt = distribute_tensor(global_tensor, mesh, [Replicate()])

    splits = list(split_dtensor(dt, 4, dim=1, update_uneven_dtensor_chunk_meta=False))
    ref_splits = torch.split(global_tensor, 4, dim=1)

    assert len(splits) == len(ref_splits)
    for i, (chunk_dt, ref) in enumerate(zip(splits, ref_splits)):
        # Replicated placement: local == global slice
        local = chunk_dt.to_local()
        ref = ref.to(local.device)
        assert local.shape == ref.shape
        assert torch.allclose(local, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_split_dtensor_uneven_shard_with_metadata(distributed_setup):
    """Split along dim 0 on an unevenly sharded DTensor and verify correctness."""
    setup = distributed_setup
    world_size = setup["world_size"]
    mesh = DeviceMesh(setup["device_type"], list(range(world_size)))

    rows = world_size * 3 + 1  # uneven vs world_size
    global_tensor = make_global_arange((rows, 4), dtype=torch.float32, device=setup["device"])

    shard = Shard(0)
    local_list, _ = shard._split_tensor(
        global_tensor, world_size, with_padding=False, contiguous=True
    )
    local = local_list[setup["rank"]]

    dt = DTensor.from_local(
        local, mesh, (Shard(0),), shape=global_tensor.size(), stride=global_tensor.stride()
    )

    # Split into size 2 along dim 0
    splits = list(split_dtensor(dt, 2, dim=0, update_uneven_dtensor_chunk_meta=True))
    ref_splits = torch.split(global_tensor, 2, dim=0)

    assert len(splits) == len(ref_splits)
    for i, (chunk_dt, ref) in enumerate(zip(splits, ref_splits)):
        gathered = uneven_dtensor_to_full_tensor(chunk_dt)
        ref = ref.to(gathered.device)
        assert gathered.shape == ref.shape
        assert torch.allclose(gathered, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.distributed
def test_split_dtensor_zero_local_shard(distributed_setup):
    """
    Split DTensor where some ranks have zero local data (after an uneven manual layout),
    ensuring split_dtensor yields correct empty locals but correct global slices.
    """
    setup = distributed_setup
    rank = setup["rank"]
    world_size = setup["world_size"]
    mesh = DeviceMesh(setup["device_type"], list(range(world_size)))

    # Create a manual uneven sharding along dim 1 with possible zero-length local on some ranks
    # Similar style to your "wild random uneven" gather test.
    if rank == 0:
        # random but deterministic lengths
        lengths = torch.tensor(
            [0] + [4] * (world_size - 1), dtype=torch.int64, device=setup["device"]
        )
        if lengths.sum().item() == 0:
            lengths[0] = 1  # fallback
    else:
        lengths = torch.empty(world_size, dtype=torch.int64, device=setup["device"])

    dist.broadcast(lengths, src=0)
    total = int(lengths.sum().item())

    rows = 4
    cols = total
    global_tensor = make_global_arange((rows, cols), dtype=torch.float32, device=setup["device"])

    start = int(lengths[:rank].sum().item())
    length = int(lengths[rank].item())
    end = start + length

    if length > 0:
        local = global_tensor[:, start:end].clone()
    else:
        local = torch.empty((rows, 0), dtype=global_tensor.dtype, device=global_tensor.device)

    dt = DTensor.from_local(
        local, mesh, (Shard(1),), shape=global_tensor.size(), stride=global_tensor.stride()
    )

    # Split dim 1 into chunks of size 5
    splits = list(split_dtensor(dt, 5, dim=1, update_uneven_dtensor_chunk_meta=True))
    ref_splits = torch.split(global_tensor, 5, dim=1)

    assert len(splits) == len(ref_splits)
    for i, (chunk_dt, ref) in enumerate(zip(splits, ref_splits)):
        gathered = uneven_dtensor_to_full_tensor(chunk_dt)
        ref = ref.to(gathered.device)
        assert gathered.shape == ref.shape
        assert torch.allclose(gathered, ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
