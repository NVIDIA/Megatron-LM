# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the hierarchical NCCL M2N ReFIT copy service."""

import ctypes
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.resharding.copy_services.base import RecvOp, SendOp
from megatron.core.resharding.copy_services.nccl_m2n_copy_service import (
    NCCLM2NCopyService,
    _align_up,
    _M2NTopology,
    _NcclDistTensor,
    _NcclMesh,
    _ordered_ops_by_peer,
    _validate_mesh_limits,
    _validate_role_roster,
)
from tests.unit_tests.test_utilities import Utils


def test_ctypes_structs_match_nccl_m2n_abi():
    """The public C structs have stable natural-layout sizes on 64-bit hosts."""
    assert ctypes.sizeof(_NcclMesh) == 20
    assert ctypes.sizeof(_NcclDistTensor) == 48


@pytest.mark.parametrize(("value", "expected"), [(0, 0), (1, 32), (32, 32), (49, 64)])
def test_align_up_for_m2n_chunking(value, expected):
    assert _align_up(value, 32) == expected


def test_validate_role_roster_accepts_source_first_disjoint_meshes():
    topology = _validate_role_roster([(True, False), (True, False), (False, True), (False, True)])
    assert topology.src_ranks == (0, 1)
    assert topology.dst_ranks == (2, 3)


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        ([(True, True), (False, True)], "non-collocated"),
        ([(True, False), (False, False), (False, True)], "idle ranks"),
        ([(True, False), (False, True), (True, False), (False, True)], "source-first"),
        ([(True, False)], "at least one source and one destination"),
    ],
)
def test_validate_role_roster_rejects_unsupported_topologies(roles, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_role_roster(roles)


@pytest.mark.parametrize(
    ("algorithm", "src_count", "dst_count", "should_pass"),
    [
        ("AUTO", 16, 64, True),
        ("RING", 17, 1, False),
        ("DIRECT", 32, 64, True),
        ("DIRECT", 33, 1, False),
        ("DIRECT", 1, 65, False),
    ],
)
def test_validate_mesh_limits(algorithm, src_count, dst_count, should_pass):
    topology = _M2NTopology(
        src_ranks=tuple(range(src_count)), dst_ranks=tuple(range(src_count, src_count + dst_count))
    )
    if should_pass:
        _validate_mesh_limits(topology, algorithm)
    else:
        with pytest.raises(RuntimeError, match="static limits"):
            _validate_mesh_limits(topology, algorithm)


def test_ordered_ops_group_by_peer_and_keep_duplicate_task_order():
    first_duplicate = SendOp(task_id=4, tensor=torch.tensor([1]), dest_rank=3)
    second_duplicate = SendOp(task_id=4, tensor=torch.tensor([2]), dest_rank=3)
    lower_id = SendOp(task_id=1, tensor=torch.tensor([3]), dest_rank=3)
    other_peer = SendOp(task_id=0, tensor=torch.tensor([4]), dest_rank=2)

    grouped = _ordered_ops_by_peer(
        [first_duplicate, other_peer, second_duplicate, lower_id], is_send=True
    )

    assert grouped[2] == [other_peer]
    assert grouped[3] == [lower_id, first_duplicate, second_duplicate]


def test_ordered_ops_requires_task_ids():
    op = RecvOp(task_id=None, tensor=torch.zeros(1), src_rank=0)
    with pytest.raises(RuntimeError, match="requires a task_id"):
        _ordered_ops_by_peer([op], is_send=False)


@pytest.mark.skipif(
    not os.getenv("NCCL_M2N_LIBRARY"),
    reason="set NCCL_M2N_LIBRARY to run the NCCL M2N multi-GPU integration test",
)
def test_nccl_m2n_moves_variable_mixed_dtype_payloads():
    """Exercise cross-dimension packing, M2N transfer, and unpacking on GPUs."""
    Utils.initialize_distributed()
    world_size = dist.get_world_size()
    if world_size < 2 or world_size % 2:
        pytest.skip("NCCL M2N integration test requires an even distributed world size >= 2")

    rank = dist.get_rank()
    src_count = world_size // 2
    src_ranks = range(src_count)
    dst_ranks = range(src_count, world_size)
    is_source = rank < src_count

    service = NCCLM2NCopyService()
    service.set_model_roles(is_source=is_source, is_destination=not is_source)
    local_ok = True
    for iteration in range(2):
        received: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        if is_source:
            for dst_rank in dst_ranks:
                task_id = rank * world_size + dst_rank
                length = 13 + rank * 7 + (dst_rank - src_count) * 5
                byte_value = (rank * 31 + dst_rank * 17 + iteration) % 251
                service.submit_send(
                    torch.full((length,), byte_value, dtype=torch.uint8, device="cuda"),
                    dst_rank,
                    task_id=task_id,
                )
                # Duplicate task IDs model transforms that emit more than one tensor
                # for a single logical plan operation.
                service.submit_send(
                    torch.tensor(
                        [rank, dst_rank, length, iteration], dtype=torch.int64, device="cuda"
                    ),
                    dst_rank,
                    task_id=task_id,
                )
        else:
            for src_rank in src_ranks:
                task_id = src_rank * world_size + rank
                length = 13 + src_rank * 7 + (rank - src_count) * 5
                bytes_out = torch.empty(length, dtype=torch.uint8, device="cuda")
                metadata_out = torch.empty(4, dtype=torch.int64, device="cuda")
                service.submit_recv(bytes_out, src_rank, task_id=task_id)
                service.submit_recv(metadata_out, src_rank, task_id=task_id)
                received.append((src_rank, bytes_out, metadata_out, length))

        service.run()
        torch.cuda.synchronize()

        for src_rank, bytes_out, metadata_out, length in received:
            byte_value = (src_rank * 31 + rank * 17 + iteration) % 251
            expected_bytes = torch.full_like(bytes_out, byte_value)
            expected_metadata = torch.tensor(
                [src_rank, rank, length, iteration], dtype=torch.int64, device="cuda"
            )
            local_ok &= torch.equal(bytes_out, expected_bytes)
            local_ok &= torch.equal(metadata_out, expected_metadata)

    # close() deregisters the symmetric window collectively, so every rank
    # reaches it before surfacing a validation failure.
    service.close()
    status = torch.tensor(int(local_ok), dtype=torch.int32, device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    assert status.item() == 1
