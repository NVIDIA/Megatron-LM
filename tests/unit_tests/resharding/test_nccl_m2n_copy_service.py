# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the hierarchical NCCL M2N ReFIT copy service."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.resharding.copy_services.base import RecvOp, SendOp
from megatron.core.resharding.copy_services.nccl_m2n_copy_service import (
    NCCLM2NCopyService,
    _ordered_ops_by_peer,
    _validate_nccl_version,
    _validate_role_roster,
)
from tests.unit_tests.test_utilities import Utils


def _nccl_with_version(*release: int):
    version = SimpleNamespace(release=release)
    version_info = SimpleNamespace(libnccl=SimpleNamespace(version=version))
    return SimpleNamespace(get_version=lambda: version_info)


def test_validate_nccl_version():
    _validate_nccl_version(_nccl_with_version(2, 30, 5))
    with pytest.raises(RuntimeError, match=r"NCCL >= 2\.30\.5"):
        _validate_nccl_version(_nccl_with_version(2, 30, 4))


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


def test_model_roles_cannot_change_while_reusing_service():
    service = object.__new__(NCCLM2NCopyService)
    service._is_source = None
    service._is_destination = None

    service.set_model_roles(is_source=True, is_destination=False)
    service.set_model_roles(is_source=True, is_destination=False)

    with pytest.raises(RuntimeError, match="roles cannot change"):
        service.set_model_roles(is_source=False, is_destination=True)


def test_topology_is_collected_once_but_slot_size_is_reduced_each_run(monkeypatch):
    service = object.__new__(NCCLM2NCopyService)
    service._device = torch.device("cpu")
    service._is_source = True
    service._is_destination = False
    service._topology = None
    service._slot_bytes_tensor = torch.empty((), dtype=torch.int64)
    service.world_size = 4
    service.group = None
    calls = {"all_gather": 0, "all_reduce": 0}

    def fake_all_gather(outputs, _state, group):
        assert group is None
        calls["all_gather"] += 1
        states = ((1, 0, 11), (1, 0, 17), (0, 1, 13), (0, 1, 19))
        for output, state in zip(outputs, states):
            output.copy_(torch.tensor(state))

    def fake_all_reduce(value, op, group):
        assert op == dist.ReduceOp.MAX
        assert group is None
        calls["all_reduce"] += 1
        value.fill_(23)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    topology, first_slot_bytes = service._get_topology_and_slot_bytes(11)
    cached_topology, second_slot_bytes = service._get_topology_and_slot_bytes(7)

    assert topology is cached_topology
    assert topology.src_ranks == (0, 1)
    assert topology.dst_ranks == (2, 3)
    assert first_slot_bytes == 19
    assert second_slot_bytes == 23
    assert calls == {"all_gather": 1, "all_reduce": 1}


def test_pack_only_zeros_active_source_buffer():
    service = object.__new__(NCCLM2NCopyService)
    service._buffer = torch.full((12,), 7, dtype=torch.uint8)
    topology = _validate_role_roster([(True, False), (False, True), (False, True)])

    service._pack_sends(topology, sends={}, slot_bytes=3)

    assert torch.equal(service._buffer[:6], torch.zeros(6, dtype=torch.uint8))
    assert torch.equal(service._buffer[6:], torch.full((6,), 7, dtype=torch.uint8))


def _has_nccl_m2n_python_package() -> bool:
    try:
        return (
            importlib.util.find_spec("nccl.core") is not None
            and importlib.util.find_spec("nccl.m2n") is not None
        )
    except (ImportError, ModuleNotFoundError):
        return False


@pytest.mark.skipif(
    not _has_nccl_m2n_python_package(),
    reason="install NVIDIA/nccl-extensions and NCCL4Py to run the M2N integration test",
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
                length = 13 + rank * 7 + (dst_rank - src_count) * 5 + iteration * 3
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
                length = 13 + src_rank * 7 + (rank - src_count) * 5 + iteration * 3
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

    # close() drains M2N collectively, so every rank reaches it before
    # surfacing a validation failure.
    service.close()
    status = torch.tensor(int(local_ok), dtype=torch.int32, device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    assert status.item() == 1
