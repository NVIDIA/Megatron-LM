# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for shared copy_services utilities.

Covers:
- ``match_local_ops_by_task_id``: pairing semantics and every error branch.
- ``CopyService.close()`` default no-op contract.
- Explicit opt-in for multiple runs per plan.
- NCCL communicator initialization on ranks with no local operations.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.resharding.copy_services.base import (
    CopyService,
    RecvOp,
    SendOp,
    match_local_ops_by_task_id,
)
from megatron.core.resharding.copy_services.gloo_copy_service import GlooCopyService
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.copy_services.nixl_copy_service import NixlCopyService
from megatron.core.resharding.copy_services.nvshmem_copy_service import NVSHMEMCopyService
from tests.unit_tests.test_utilities import Utils


def _t():
    return torch.zeros(4)


class TestMatchLocalOps:
    """match_local_ops_by_task_id pairs by task_id and rejects malformed inputs."""

    def test_single_pair(self):
        sends = [SendOp(task_id=1, tensor=_t(), dest_rank=0)]
        recvs = [RecvOp(task_id=1, tensor=_t(), src_rank=0)]
        pairs = match_local_ops_by_task_id(sends, recvs, "Test", rank=0)
        assert len(pairs) == 1
        send_op, recv_op = pairs[0]
        assert send_op is sends[0]
        assert recv_op is recvs[0]

    def test_pairs_match_across_order(self):
        """Order of sends vs recvs doesn't matter; pairing is by task_id."""
        sends = [
            SendOp(task_id=1, tensor=_t(), dest_rank=0),
            SendOp(task_id=2, tensor=_t(), dest_rank=0),
        ]
        recvs = [
            RecvOp(task_id=2, tensor=_t(), src_rank=0),
            RecvOp(task_id=1, tensor=_t(), src_rank=0),
        ]
        pairs = match_local_ops_by_task_id(sends, recvs, "Test", rank=0)
        pair_ids = {(s.task_id, r.task_id) for s, r in pairs}
        assert pair_ids == {(1, 1), (2, 2)}

    def test_empty_lists(self):
        pairs = match_local_ops_by_task_id([], [], "Test", rank=0)
        assert pairs == []

    def test_none_send_task_id_raises(self):
        sends = [SendOp(task_id=None, tensor=_t(), dest_rank=0)]
        recvs = [RecvOp(task_id=1, tensor=_t(), src_rank=0)]
        with pytest.raises(RuntimeError, match="requires a task_id"):
            match_local_ops_by_task_id(sends, recvs, "TestBackend", rank=0)

    def test_none_recv_task_id_raises(self):
        sends = [SendOp(task_id=1, tensor=_t(), dest_rank=0)]
        recvs = [RecvOp(task_id=None, tensor=_t(), src_rank=0)]
        with pytest.raises(RuntimeError, match="requires a task_id"):
            match_local_ops_by_task_id(sends, recvs, "TestBackend", rank=0)

    def test_more_sends_than_recvs_raises(self):
        sends = [
            SendOp(task_id=1, tensor=_t(), dest_rank=0),
            SendOp(task_id=2, tensor=_t(), dest_rank=0),
        ]
        recvs = [RecvOp(task_id=1, tensor=_t(), src_rank=0)]
        with pytest.raises(RuntimeError, match="unmatched local ops"):
            match_local_ops_by_task_id(sends, recvs, "TestBackend", rank=7)

    def test_more_recvs_than_sends_raises(self):
        sends = [SendOp(task_id=1, tensor=_t(), dest_rank=0)]
        recvs = [
            RecvOp(task_id=1, tensor=_t(), src_rank=0),
            RecvOp(task_id=2, tensor=_t(), src_rank=0),
        ]
        with pytest.raises(RuntimeError, match="unmatched local ops"):
            match_local_ops_by_task_id(sends, recvs, "TestBackend", rank=7)

    def test_duplicate_send_task_ids_raises(self):
        """Duplicate task_ids in sends collapse the dict — triggers count-mismatch raise."""
        sends = [
            SendOp(task_id=1, tensor=_t(), dest_rank=0),
            SendOp(task_id=1, tensor=_t(), dest_rank=0),  # duplicate
        ]
        recvs = [
            RecvOp(task_id=1, tensor=_t(), src_rank=0),
            RecvOp(task_id=2, tensor=_t(), src_rank=0),
        ]
        with pytest.raises(RuntimeError, match="unmatched local ops"):
            match_local_ops_by_task_id(sends, recvs, "TestBackend", rank=7)

    def test_duplicate_recv_task_ids_raises(self):
        sends = [
            SendOp(task_id=1, tensor=_t(), dest_rank=0),
            SendOp(task_id=2, tensor=_t(), dest_rank=0),
        ]
        recvs = [
            RecvOp(task_id=1, tensor=_t(), src_rank=0),
            RecvOp(task_id=1, tensor=_t(), src_rank=0),  # duplicate
        ]
        with pytest.raises(RuntimeError, match="unmatched local ops"):
            match_local_ops_by_task_id(sends, recvs, "TestBackend", rank=7)

    def test_mismatched_task_ids_raises(self):
        """Equal counts but task_ids don't overlap — should raise missing-send."""
        sends = [SendOp(task_id=1, tensor=_t(), dest_rank=0)]
        recvs = [RecvOp(task_id=99, tensor=_t(), src_rank=0)]
        with pytest.raises(RuntimeError, match="missing local send"):
            match_local_ops_by_task_id(sends, recvs, "TestBackend", rank=0)

    def test_error_message_includes_backend_and_rank(self):
        sends = [SendOp(task_id=None, tensor=_t(), dest_rank=0)]
        recvs = [RecvOp(task_id=1, tensor=_t(), src_rank=0)]
        with pytest.raises(RuntimeError) as exc_info:
            match_local_ops_by_task_id(sends, recvs, "TestBackend", rank=0)
        assert "TestBackend" in str(exc_info.value)


class TestCopyServiceClose:
    """CopyService.close() default is a no-op; subclasses may override."""

    def test_default_close_is_noop_and_returns_none(self):
        class _Stub(CopyService):
            def __init__(self):  # bypass dist requirement
                pass

            def submit_send(self, *args, **kwargs):
                pass

            def submit_recv(self, *args, **kwargs):
                pass

            def run(self):
                pass

        svc = _Stub()
        result = svc.close()
        assert result is None

    def test_subclass_can_override_close(self):
        class _ClosingService(CopyService):
            def __init__(self):
                self.closed = False

            def submit_send(self, *args, **kwargs):
                pass

            def submit_recv(self, *args, **kwargs):
                pass

            def run(self):
                pass

            def close(self):
                self.closed = True

        svc = _ClosingService()
        assert svc.closed is False
        svc.close()
        assert svc.closed is True


def test_nixl_service_skips_redundant_process_group_barrier():
    """NIXL's ready/data protocol provides its own peer completion."""
    assert NixlCopyService.requires_process_group_barrier is False


@pytest.mark.parametrize("service_cls", [NCCLCopyService, GlooCopyService])
def test_local_copy_waits_for_current_stream(service_cls):
    """A local copy must observe writes already queued on the caller's stream."""
    Utils.initialize_distributed()
    service = service_cls()
    source = torch.zeros(1, device="cuda")
    destination = torch.zeros_like(source)
    torch.cuda.synchronize()

    producer_gate = torch.cuda.Event()
    release_stream = torch.cuda.Stream()
    with torch.cuda.stream(release_stream):
        torch.cuda._sleep(100_000_000)
        producer_gate.record()

    torch.cuda.current_stream().wait_event(producer_gate)
    source.fill_(1)

    service.submit_send(source, dist.get_rank(), task_id=0)
    service.submit_recv(destination, dist.get_rank(), task_id=0)
    service.run()

    torch.cuda.synchronize()

    torch.testing.assert_close(destination, source)


def test_multiple_runs_per_plan_require_explicit_backend_support():
    """Third-party services stay model-wide unless they accept the stream contract."""
    assert CopyService.supports_multiple_runs_per_plan is False
    assert GlooCopyService.supports_multiple_runs_per_plan is True
    assert NCCLCopyService.supports_multiple_runs_per_plan is True
    assert NVSHMEMCopyService.supports_multiple_runs_per_plan is True
    assert NixlCopyService.supports_multiple_runs_per_plan is False


def test_nccl_service_eagerly_connects_before_first_run(monkeypatch):
    """Idle ranks must join NCCL setup before active ranks issue their first P2P."""
    device = torch.device("cuda", 1)

    class Backend:
        def __init__(self):
            self.connected_devices = []

        def eager_connect_single_device(self, connected_device):
            self.connected_devices.append(connected_device)

    class Group:
        def __init__(self, backend):
            self.backend = backend
            self.requested_devices = []

        def rank(self):
            return 1

        def size(self):
            return 4

        def _get_backend(self, requested_device):
            self.requested_devices.append(requested_device)
            return self.backend

    class Stream:
        def wait_stream(self, _stream):
            pass

    backend = Backend()
    group = Group(backend)
    stream = Stream()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: device.index)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)

    service = NCCLCopyService(group=group)
    service.run()
    service.run()

    assert group.requested_devices == [device]
    assert backend.connected_devices == [device]
