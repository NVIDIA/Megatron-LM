# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the hierarchical NCCL M2N ReFIT copy service."""

import importlib.util
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from megatron.core.resharding.copy_services.base import RecvOp, SendOp
from megatron.core.resharding.copy_services.nccl_m2n_copy_service import (
    NCCLM2NCopyService,
    _M2NTopology,
    _OfficialM2NRuntime,
    _ordered_ops_by_peer,
    _validate_mesh_limits,
    _validate_role_roster,
    _version_tuple,
)
from tests.unit_tests.test_utilities import Utils


@dataclass(frozen=True)
class _FakeConfig:
    pass


@dataclass(frozen=True)
class _FakeMesh:
    dims: tuple[int, ...]
    start_rank: int = 0


@dataclass(frozen=True)
class _FakeShard:
    dim: int


@dataclass(frozen=True)
class _FakeReplicate:
    pass


class _FakeHandle:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events

    def destroy(self) -> None:
        self.events.append(("handle_destroy",))


class _FakeM2N:
    Config = _FakeConfig
    Mesh = _FakeMesh
    Replicate = _FakeReplicate
    Shard = _FakeShard

    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []

    def init(self, config: _FakeConfig) -> _FakeHandle:
        self.events.append(("m2n_init", config))
        return _FakeHandle(self.events)

    def reshard(self, **kwargs) -> None:
        self.calls.append(kwargs)
        self.events.append(("reshard",))


class _FakeUniqueId:
    def __init__(self, value: bytes) -> None:
        self.value = value

    @classmethod
    def from_bytes(cls, value: bytes) -> "_FakeUniqueId":
        return cls(value)

    def __bytes__(self) -> bytes:
        return self.value


class _FakeComm:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events

    def destroy(self) -> None:
        self.events.append(("comm_destroy",))


class _FakeNccl:
    UniqueId = _FakeUniqueId

    def __init__(self, events: list[tuple[Any, ...]], version: object = "2.30.5") -> None:
        self.events = events
        self.version = version
        owner = self

        class Communicator:
            @staticmethod
            def init(nranks: int, rank: int, unique_id: _FakeUniqueId) -> _FakeComm:
                owner.events.append(("comm_init", nranks, rank, unique_id.value))
                return _FakeComm(owner.events)

        self.Communicator = Communicator

    def get_version(self) -> object:
        return self.version

    def get_unique_id(self, *, empty: bool = False) -> _FakeUniqueId:
        value = bytes(128) if empty else b"u" * 128
        self.events.append(("get_unique_id", empty))
        return _FakeUniqueId(value)


def _fake_runtime(version: object = "2.30.5"):
    events: list[tuple[Any, ...]] = []
    m2n = _FakeM2N(events)
    nccl = _FakeNccl(events, version=version)
    runtime = _OfficialM2NRuntime(_m2n_module=m2n, _nccl_module=nccl)
    return runtime, m2n, nccl, events


@pytest.mark.parametrize(
    ("version", "expected"),
    [("2.30.5", (2, 30, 5)), ("NCCL 2.31", (2, 31, 0)), ("2.30.5-1", (2, 30, 5))],
)
def test_version_tuple(version, expected):
    assert _version_tuple(version) == expected


def test_official_runtime_uses_nccl4py_and_nccl_m2n_objects():
    runtime, m2n, _nccl, events = _fake_runtime()
    assert runtime.get_unique_id(empty=False) == b"u" * 128
    assert runtime.get_unique_id(empty=True) == bytes(128)

    comm = runtime.init_comm(4, 2, b"x" * 128)
    topology = _M2NTopology(src_ranks=(0, 1), dst_ranks=(2, 3))
    src = torch.empty((1, 2, 17), dtype=torch.uint8)
    stream = object()
    runtime.reshard(comm, src, None, topology, 17, stream)

    call = m2n.calls[0]
    assert call["src"] is src
    assert call["dst"] is None
    assert call["comm"] is comm
    assert call["stream"] is stream
    assert call["src_mesh"] == _FakeMesh((2,), start_rank=0)
    assert call["src_placements"] == (_FakeShard(0),)
    assert call["src_local_shape"] == (1, 2, 17)
    assert call["src_dtype"] is torch.uint8
    assert call["dst_mesh"] == _FakeMesh((2,), start_rank=2)
    assert call["dst_placements"] == (_FakeShard(1),)
    assert call["dst_local_shape"] == (2, 1, 17)
    assert call["dst_dtype"] is torch.uint8

    runtime.finalize()
    runtime.destroy_comm(comm)
    assert events[-2:] == [("handle_destroy",), ("comm_destroy",)]


def test_official_runtime_rejects_old_nccl():
    events: list[tuple[Any, ...]] = []
    with pytest.raises(RuntimeError, match=r"NCCL >= 2\.30\.5"):
        _OfficialM2NRuntime(
            _m2n_module=_FakeM2N(events), _nccl_module=_FakeNccl(events, version="2.30.4")
        )
    assert not any(event[0] == "m2n_init" for event in events)


def test_official_runtime_checks_loaded_library_in_structured_version_info():
    events: list[tuple[Any, ...]] = []
    version_info = SimpleNamespace(
        nccl4py="99.0.0",
        nccl_bindings="99.0.0",
        libnccl=SimpleNamespace(version=SimpleNamespace(release=(2, 30, 4))),
    )
    with pytest.raises(RuntimeError, match=r"NCCL >= 2\.30\.5"):
        _OfficialM2NRuntime(
            _m2n_module=_FakeM2N(events), _nccl_module=_FakeNccl(events, version=version_info)
        )
    assert not any(event[0] == "m2n_init" for event in events)


def test_official_runtime_sets_explicit_native_library(tmp_path, monkeypatch):
    library = tmp_path / "libnccl_m2n.so"
    library.touch()
    monkeypatch.delenv("NCCL_M2N_LIBRARY", raising=False)
    events: list[tuple[Any, ...]] = []

    runtime = _OfficialM2NRuntime(
        library, _m2n_module=_FakeM2N(events), _nccl_module=_FakeNccl(events)
    )

    assert runtime.library_path == str(library.resolve())
    assert runtime.explicit_library_path == str(library.resolve())
    runtime.finalize()


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
    ("src_count", "dst_count", "should_pass"), [(16, 64, True), (17, 1, False), (1, 65, False)]
)
def test_validate_mesh_limits(src_count, dst_count, should_pass):
    topology = _M2NTopology(
        src_ranks=tuple(range(src_count)), dst_ranks=tuple(range(src_count, src_count + dst_count))
    )
    if should_pass:
        _validate_mesh_limits(topology)
    else:
        with pytest.raises(RuntimeError, match="copy/staging limits"):
            _validate_mesh_limits(topology)


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

    # close() drains M2N collectively, so every rank reaches it before
    # surfacing a validation failure.
    service.close()
    status = torch.tensor(int(local_ok), dtype=torch.int32, device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    assert status.item() == 1
