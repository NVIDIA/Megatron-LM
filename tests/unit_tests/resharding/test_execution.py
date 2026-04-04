# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for execute_reshard_plan using a mock CopyService.

These test the send/recv submission logic, writeback paths, and
non-collocated mode handling.  Requires CUDA (uses torch.cuda.synchronize).
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from megatron.core.resharding.copy_services.base import CopyService
from megatron.core.resharding.execution import execute_reshard_plan
from megatron.core.resharding.transforms import ReshardTransform
from megatron.core.resharding.utils import ReshardPlan, TransferOp

_HAS_CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")


# ---------------------------------------------------------------------------
# Mock CopyService that records calls and performs local copies
# ---------------------------------------------------------------------------


class MockCopyService(CopyService):
    """CopyService that records submitted ops and copies send→recv on run()."""

    def __init__(self):
        self.sends = []  # [(tensor, dest_rank, task_id)]
        self.recvs = []  # [(tensor, src_rank, task_id)]

    def submit_send(self, src_tensor, dest_rank, task_id=None):
        self.sends.append((src_tensor.clone(), dest_rank, task_id))

    def submit_recv(self, dest_tensor, src_rank, task_id=None):
        self.recvs.append((dest_tensor, src_rank, task_id))

    def run(self):
        """Match sends to recvs by task_id and copy data."""
        sends_by_id = {tid: t for t, _, tid in self.sends if tid is not None}
        for recv_tensor, _, tid in self.recvs:
            if tid is not None and tid in sends_by_id:
                src = sends_by_id[tid]
                recv_tensor.copy_(src[: recv_tensor.numel()].reshape(recv_tensor.shape))
        self.sends.clear()
        self.recvs.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module_with_params(params_dict):
    """Create a simple nn.Module with given named parameters."""
    module = torch.nn.Module()
    for name, tensor in params_dict.items():
        parts = name.split('.')
        parent = module
        for part in parts[:-1]:
            if not hasattr(parent, part):
                child = torch.nn.Module()
                parent.add_module(part, child)
            parent = getattr(parent, part)
        parent.register_parameter(parts[-1], torch.nn.Parameter(tensor))
    return module


def _full_slice(ndim):
    return tuple(slice(None) for _ in range(ndim))


def _make_transfer_op(param_name, peer_rank, is_send, my_slice, peer_slice, task_id):
    return TransferOp(
        param_name=param_name,
        peer_rank=peer_rank,
        is_send=is_send,
        my_slice=my_slice,
        peer_slice=peer_slice,
        task_id=task_id,
    )


def _run(plan, src_module, dst_module, service, transform=None):
    """Execute a reshard plan, initializing a temporary process group if needed."""
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)
    execute_reshard_plan(plan, src_module, dst_module, service, transform=transform)


# ===========================================================================
# Basic send/recv
# ===========================================================================


class TestExecuteReshard:
    """Test basic execute_reshard_plan functionality."""

    def test_simple_full_copy(self):
        """Full tensor copy from src to dst via a matched send/recv pair."""
        src_data = torch.randn(4, 8, device="cuda")
        dst_data = torch.zeros(4, 8, device="cuda")
        src_module = _make_module_with_params({"weight": src_data.clone()})
        dst_module = _make_module_with_params({"weight": dst_data.clone()})

        s = _full_slice(2)
        plan = ReshardPlan(
            send_ops=[_make_transfer_op("weight", peer_rank=1, is_send=True, my_slice=s, peer_slice=s, task_id=0)],
            recv_ops=[_make_transfer_op("weight", peer_rank=0, is_send=False, my_slice=s, peer_slice=s, task_id=0)],
        )

        service = MockCopyService()
        _run(plan, src_module, dst_module, service)

        assert torch.equal(
            dict(dst_module.named_parameters())["weight"].data,
            dict(src_module.named_parameters())["weight"].data,
        )

    def test_slice_copy(self):
        """Copy a row-slice of a parameter."""
        src_data = torch.randn(8, 4, device="cuda")
        dst_data = torch.zeros(4, 4, device="cuda")
        src_module = _make_module_with_params({"weight": src_data.clone()})
        dst_module = _make_module_with_params({"weight": dst_data.clone()})

        src_slice = (slice(0, 4), slice(None))
        dst_slice = (slice(None), slice(None))
        plan = ReshardPlan(
            send_ops=[_make_transfer_op("weight", 1, True, src_slice, dst_slice, task_id=0)],
            recv_ops=[_make_transfer_op("weight", 0, False, dst_slice, src_slice, task_id=0)],
        )

        service = MockCopyService()
        _run(plan, src_module, dst_module, service)

        assert torch.equal(
            dict(dst_module.named_parameters())["weight"].data,
            src_data[:4],
        )


# ===========================================================================
# Non-collocated mode
# ===========================================================================


class TestNonCollocated:
    """Test non-collocated mode where src_module or dst_module is None."""

    def test_source_only_rank(self):
        """Source-only rank: has src_module, dst_module=None. Should only submit sends."""
        src_data = torch.randn(4, 8, device="cuda")
        src_module = _make_module_with_params({"weight": src_data.clone()})

        s = _full_slice(2)
        plan = ReshardPlan(
            send_ops=[_make_transfer_op("weight", 1, True, s, s, task_id=0)],
            recv_ops=[],
        )

        service = MockCopyService()
        _run(plan, src_module, None, service)

    def test_destination_only_rank(self):
        """Destination-only rank: src_module=None, has dst_module. Should only submit recvs."""
        dst_data = torch.zeros(4, 8, device="cuda")
        dst_module = _make_module_with_params({"weight": dst_data.clone()})

        s = _full_slice(2)
        plan = ReshardPlan(
            send_ops=[],
            recv_ops=[_make_transfer_op("weight", 0, False, s, s, task_id=0)],
        )

        service = MockCopyService()
        _run(plan, None, dst_module, service)

    def test_idle_rank(self):
        """Idle rank: both src_module and dst_module are None."""
        plan = ReshardPlan(send_ops=[], recv_ops=[])
        service = MockCopyService()
        _run(plan, None, None, service)



# ===========================================================================
# Transform path
# ===========================================================================


class TestTransformPath:
    """Test that the transform hooks are properly invoked."""

    def test_transform_prepare_send_called(self):
        """Transform's prepare_send should be called for matching params."""
        src_data = torch.randn(4, 8, device="cuda")
        src_module = _make_module_with_params({"weight": src_data.clone()})

        transform = MagicMock(spec=ReshardTransform)
        transform.should_transform.return_value = True
        transform.prepare_send.return_value = [torch.randn(4, 8, device="cuda")]

        s = _full_slice(2)
        plan = ReshardPlan(
            send_ops=[_make_transfer_op("weight", 1, True, s, s, task_id=0)],
            recv_ops=[],
        )

        service = MockCopyService()
        _run(plan, src_module, None, service, transform=transform)
        transform.prepare_send.assert_called_once()

    def test_transform_prepare_recv_and_finalize(self):
        """Transform's prepare_recv and finalize_recv should be called for matching params."""
        dst_data = torch.zeros(4, 8, device="cuda")
        dst_module = _make_module_with_params({"weight": dst_data.clone()})

        recv_buf = torch.randn(4, 8, device="cuda")
        transform = MagicMock(spec=ReshardTransform)
        transform.should_transform.return_value = True
        transform.prepare_recv.return_value = [recv_buf]

        s = _full_slice(2)
        plan = ReshardPlan(
            send_ops=[],
            recv_ops=[_make_transfer_op("weight", 0, False, s, s, task_id=0)],
        )

        service = MockCopyService()
        _run(plan, None, dst_module, service, transform=transform)
        transform.prepare_recv.assert_called_once()
        transform.finalize_recv.assert_called_once()

    def test_transform_not_called_for_non_matching(self):
        """Transform should NOT be called for params where should_transform returns False."""
        src_data = torch.randn(4, 8, device="cuda")
        src_module = _make_module_with_params({"weight": src_data.clone()})

        transform = MagicMock(spec=ReshardTransform)
        transform.should_transform.return_value = False

        s = _full_slice(2)
        plan = ReshardPlan(
            send_ops=[_make_transfer_op("weight", 1, True, s, s, task_id=0)],
            recv_ops=[],
        )

        service = MockCopyService()
        _run(plan, src_module, None, service, transform=transform)
        transform.prepare_send.assert_not_called()


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases in execute_reshard_plan."""

    def test_empty_plan(self):
        """Empty plan (no ops) should complete without error."""
        plan = ReshardPlan(send_ops=[], recv_ops=[])
        service = MockCopyService()
        _run(plan, None, None, service)

    def test_missing_param_in_src(self):
        """Send op referencing nonexistent param should be silently skipped."""
        src_module = _make_module_with_params({"other": torch.randn(4, 8, device="cuda")})

        s = _full_slice(2)
        plan = ReshardPlan(
            send_ops=[_make_transfer_op("weight", 1, True, s, s, task_id=0)],
            recv_ops=[],
        )

        service = MockCopyService()
        _run(plan, src_module, None, service)

    def test_missing_param_in_dst(self):
        """Recv op referencing nonexistent param should be silently skipped."""
        dst_module = _make_module_with_params({"other": torch.zeros(4, 8, device="cuda")})

        s = _full_slice(2)
        plan = ReshardPlan(
            send_ops=[],
            recv_ops=[_make_transfer_op("weight", 0, False, s, s, task_id=0)],
        )

        service = MockCopyService()
        _run(plan, None, dst_module, service)
