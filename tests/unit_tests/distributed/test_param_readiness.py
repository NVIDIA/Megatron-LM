# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parameter-readiness contracts for direct parameter consumers."""

from types import SimpleNamespace

import torch

import megatron.core.distributed.distributed_data_parallel as ddp_module
from megatron.core.distributed.distributed_data_parallel import (
    DistributedDataParallel,
    _BucketParamReadyCallback,
)
from megatron.core.utils import PARAM_READY_CALLBACK_ATTR, ensure_params_ready


class _FakeBucketGroup:
    def __init__(self):
        self.finished = []
        self.param_gather_dispatched = False
        self.param_gather_handle = None

    def finish_param_sync(self, skip_next_bucket_dispatch=False):
        self.finished.append(skip_next_bucket_dispatch)
        self.param_gather_dispatched = True
        self.param_gather_handle = None


def _fake_ddp(align_param_gather=False, hooks_enabled=True):
    ddp = object.__new__(DistributedDataParallel)
    ddp.ddp_config = SimpleNamespace(align_param_gather=align_param_gather)
    ddp.overlap_param_gather_with_optimizer_step = False
    ddp.remove_forward_pre_hook_handles = {object(): object()} if hooks_enabled else {}
    return ddp


class TestParamReadinessProtocol:
    def test_no_marker_is_a_noop(self):
        param = torch.nn.Parameter(torch.zeros(2))
        assert not hasattr(param, PARAM_READY_CALLBACK_ATTR)
        ensure_params_ready([param])

    def test_callbacks_are_deduplicated_per_bucket(self):
        calls = []
        shared = lambda: calls.append("shared")  # noqa: E731
        other = lambda: calls.append("other")  # noqa: E731

        first, second, third = (torch.nn.Parameter(torch.zeros(1)) for _ in range(3))
        setattr(first, PARAM_READY_CALLBACK_ATTR, shared)
        setattr(second, PARAM_READY_CALLBACK_ATTR, shared)
        setattr(third, PARAM_READY_CALLBACK_ATTR, other)
        plain = torch.nn.Parameter(torch.zeros(1))

        ensure_params_ready([first, second, plain, first, third])

        assert calls == ["shared", "other"]


class TestBucketParamReadiness:
    def test_cuda_graph_capture_defers_publication(self, monkeypatch):
        ddp = _fake_ddp()
        bucket_group = _FakeBucketGroup()
        ready_callback = _BucketParamReadyCallback(ddp, bucket_group)
        capturing = [True]
        monkeypatch.setattr(ddp_module, "is_graph_capturing", lambda: capturing[0])

        ready_callback()
        assert bucket_group.finished == []

        capturing[0] = False
        ready_callback()
        assert bucket_group.finished == [False]

    def test_publishes_once_then_is_idempotent(self):
        ddp = _fake_ddp()
        bucket_group = _FakeBucketGroup()
        ready_callback = _BucketParamReadyCallback(ddp, bucket_group)

        ready_callback()
        ready_callback()

        assert bucket_group.finished == [False]

    def test_align_param_gather_skips_next_bucket_dispatch(self):
        ddp = _fake_ddp(align_param_gather=True)
        bucket_group = _FakeBucketGroup()

        _BucketParamReadyCallback(ddp, bucket_group)()

        assert bucket_group.finished == [True]

    def test_disabled_hooks_do_not_start_a_gather(self):
        ddp = _fake_ddp(hooks_enabled=False)
        bucket_group = _FakeBucketGroup()

        _BucketParamReadyCallback(ddp, bucket_group)()

        assert bucket_group.finished == []

    def test_in_flight_gather_is_drained_with_hooks_disabled(self):
        ddp = _fake_ddp(hooks_enabled=False)
        bucket_group = _FakeBucketGroup()
        bucket_group.param_gather_dispatched = True
        bucket_group.param_gather_handle = object()

        _BucketParamReadyCallback(ddp, bucket_group)()

        assert bucket_group.finished == [True]
