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
    _operation_layout,
    _ordered_ops_by_peer,
    _validate_nccl_version,
    _validate_pair_layouts,
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


def test_operation_layout_detects_ordered_tensor_disagreement():
    first = SendOp(task_id=1, tensor=torch.zeros(1, dtype=torch.float32), dest_rank=2)
    second = SendOp(task_id=2, tensor=torch.zeros(4, dtype=torch.uint8), dest_rank=2)
    matching_layout = [
        RecvOp(task_id=1, tensor=torch.zeros(1, dtype=torch.float32), src_rank=0),
        RecvOp(task_id=2, tensor=torch.zeros(4, dtype=torch.uint8), src_rank=0),
    ]
    same_bytes_different_layout = [
        RecvOp(task_id=1, tensor=torch.zeros(4, dtype=torch.uint8), src_rank=0),
        RecvOp(task_id=2, tensor=torch.zeros(4, dtype=torch.uint8), src_rank=0),
    ]

    send_layout = _operation_layout([first, second])
    recv_layout = _operation_layout(same_bytes_different_layout)

    assert send_layout == _operation_layout(matching_layout)
    assert send_layout[:2] == recv_layout[:2] == (8, 2)
    assert send_layout[2] != recv_layout[2]


def test_validate_pair_layouts_returns_largest_agreed_pair():
    topology = _validate_role_roster([(True, False), (True, False), (False, True)])
    layouts = [[[8, 2, 11], [0, 0, 0]], [[5, 1, 12], [0, 0, 0]], [[8, 2, 11], [5, 1, 12]]]

    assert _validate_pair_layouts(topology, layouts) == 8


@pytest.mark.parametrize(
    ("recv_layout", "message"),
    [([7, 2, 11], "source submitted 8 bytes"), ([8, 2, 99], "ordered task/dtype layouts differ")],
)
def test_validate_pair_layouts_rejects_disagreement(recv_layout, message):
    topology = _validate_role_roster([(True, False), (False, True)])
    layouts = [[[8, 2, 11]], [recv_layout]]

    with pytest.raises(RuntimeError, match=message):
        _validate_pair_layouts(topology, layouts)


def test_model_roles_cannot_change_while_reusing_service():
    service = object.__new__(NCCLM2NCopyService)
    service._is_source = None
    service._is_destination = None

    service.set_model_roles(is_source=True, is_destination=False)
    service.set_model_roles(is_source=True, is_destination=False)

    with pytest.raises(RuntimeError, match="roles cannot change"):
        service.set_model_roles(is_source=False, is_destination=True)


def test_topology_is_collected_once(monkeypatch):
    service = object.__new__(NCCLM2NCopyService)
    service._device = torch.device("cpu")
    service._is_source = True
    service._is_destination = False
    service._topology = None
    service.world_size = 4
    service.group = None
    calls = 0

    def fake_all_gather(outputs, _roles, group):
        nonlocal calls
        assert group is None
        calls += 1
        roles = ((1, 0), (1, 0), (0, 1), (0, 1))
        for output, role in zip(outputs, roles):
            output.copy_(torch.tensor(role))

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    topology = service._get_topology()
    cached_topology = service._get_topology()

    assert topology is cached_topology
    assert topology.src_ranks == (0, 1)
    assert topology.dst_ranks == (2, 3)
    assert calls == 1


def test_pair_layout_is_reused_until_plan_changes(monkeypatch):
    service = object.__new__(NCCLM2NCopyService)
    service._active_plan = None
    service._active_transform = None
    service._slot_bytes = None
    topology = _validate_role_roster([(True, False), (False, True)])
    calls = 0

    def fake_exchange(_topology, _sends, _recvs):
        nonlocal calls
        calls += 1
        return calls * 8

    monkeypatch.setattr(service, "_exchange_pair_layouts", fake_exchange)

    first_plan = object()
    service.set_plan(first_plan)
    assert service._get_slot_bytes(topology, {}, {}) == 8
    service.set_plan(first_plan)
    assert service._get_slot_bytes(topology, {}, {}) == 8

    second_plan = object()
    service.set_plan(second_plan)
    assert service._get_slot_bytes(topology, {}, {}) == 16
    service.set_plan(second_plan, transform=object())
    assert service._get_slot_bytes(topology, {}, {}) == 24
    assert calls == 3


def test_unbound_pair_layout_is_collected_every_run(monkeypatch):
    service = object.__new__(NCCLM2NCopyService)
    service._active_plan = None
    service._active_transform = None
    service._slot_bytes = None
    topology = _validate_role_roster([(True, False), (False, True)])
    calls = 0

    def fake_exchange(_topology, _sends, _recvs):
        nonlocal calls
        calls += 1
        return 8

    monkeypatch.setattr(service, "_exchange_pair_layouts", fake_exchange)

    assert service._get_slot_bytes(topology, {}, {}) == 8
    assert service._get_slot_bytes(topology, {}, {}) == 8
    assert calls == 2


def test_pack_leaves_padding_untouched():
    service = object.__new__(NCCLM2NCopyService)
    buffer = torch.full((6,), 7, dtype=torch.uint8)
    topology = _validate_role_roster([(True, False), (False, True), (False, True)])
    op = SendOp(task_id=1, tensor=torch.tensor([1, 2], dtype=torch.uint8), dest_rank=1)

    service._pack_sends(buffer, topology, sends={1: [op]}, slot_bytes=3)

    assert torch.equal(buffer, torch.tensor([1, 2, 7, 7, 7, 7], dtype=torch.uint8))


def test_close_is_local_and_idempotent_after_destroy_failure(monkeypatch):
    calls = []

    class Resource:
        def __init__(self, name, raises=False):
            self.name = name
            self.raises = raises

        def destroy(self):
            calls.append(self.name)
            if self.raises:
                raise RuntimeError(f"{self.name} destroy failed")

    service = object.__new__(NCCLM2NCopyService)
    service._closed = False
    service._device = torch.device("cpu")
    service._handle = Resource("handle", raises=True)
    service._comm = Resource("comm")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: calls.append("synchronize"))

    with pytest.raises(RuntimeError, match="handle destroy failed"):
        service.close()
    service.close()

    assert calls == ["synchronize", "handle", "comm"]
    assert service._closed
    assert service._handle is None
    assert service._comm is None


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
    service.set_plan(object())
    exchange_calls = 0
    exchange_pair_layouts = service._exchange_pair_layouts

    def counted_exchange_pair_layouts(topology, sends, recvs):
        nonlocal exchange_calls
        exchange_calls += 1
        return exchange_pair_layouts(topology, sends, recvs)

    service._exchange_pair_layouts = counted_exchange_pair_layouts
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

    local_ok &= exchange_calls == 1
    service.close()
    status = torch.tensor(int(local_ok), dtype=torch.int32, device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    assert status.item() == 1


@pytest.mark.skipif(
    not _has_nccl_m2n_python_package(),
    reason="install NVIDIA/nccl-extensions and NCCL4Py to run the M2N integration test",
)
def test_nccl_m2n_rejects_pair_layout_mismatch_before_transfer():
    """A missing or wrong-sized operation must fail on every rank before M2N runs."""
    Utils.initialize_distributed()
    world_size = dist.get_world_size()
    if world_size < 2 or world_size % 2:
        pytest.skip("NCCL M2N integration test requires an even distributed world size >= 2")

    rank = dist.get_rank()
    first_dst = world_size // 2
    is_source = rank < first_dst
    service = NCCLM2NCopyService()
    service.set_model_roles(is_source=is_source, is_destination=not is_source)
    if rank == 0:
        service.submit_send(torch.zeros(8, dtype=torch.uint8, device="cuda"), first_dst, task_id=1)
    elif rank == first_dst:
        service.submit_recv(torch.empty(7, dtype=torch.uint8, device="cuda"), 0, task_id=1)

    with pytest.raises(RuntimeError, match="transfer layout mismatch"):
        service.run()
    service.close()
