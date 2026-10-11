# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections import deque
from unittest.mock import Mock, patch

from megatron.core.distributed.distributed_data_parallel import WgradStream


def _wgrad_stream():
    """A WgradStream without a CUDA stream: the bookkeeping only."""
    wgrad_stream = WgradStream.__new__(WgradStream)
    wgrad_stream.stream = Mock()
    wgrad_stream.stream.record_event.side_effect = lambda: Mock(name="event")
    wgrad_stream._dispatch_events = deque()
    return wgrad_stream


def test_compute_stream_waits_for_the_dispatch_two_bucket_groups_earlier():
    wgrad_stream = _wgrad_stream()
    compute_stream = Mock()
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        for _ in range(WgradStream.BACKLOG_BUCKET_GROUPS):
            wgrad_stream.limit_backlog(compute_stream)
        compute_stream.wait_event.assert_not_called()
        wgrad_stream.limit_backlog(compute_stream)
    compute_stream.wait_event.assert_called_once()
    assert len(wgrad_stream._dispatch_events) == WgradStream.BACKLOG_BUCKET_GROUPS


def test_events_do_not_cross_an_iteration_or_a_capture_boundary():
    wgrad_stream = _wgrad_stream()
    compute_stream = Mock()
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        for _ in range(WgradStream.BACKLOG_BUCKET_GROUPS):
            wgrad_stream.limit_backlog(compute_stream)
    # Next iteration starts: nothing from the previous one is waited on.
    wgrad_stream.reset()
    assert not wgrad_stream._dispatch_events
    # An eagerly recorded event that did survive is never waited on by a capturing stream.
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        for _ in range(WgradStream.BACKLOG_BUCKET_GROUPS):
            wgrad_stream.limit_backlog(compute_stream)
    with patch("torch.cuda.is_current_stream_capturing", return_value=True):
        wgrad_stream.limit_backlog(compute_stream)
        compute_stream.wait_event.assert_not_called()
        for _ in range(WgradStream.BACKLOG_BUCKET_GROUPS):
            wgrad_stream.limit_backlog(compute_stream)
    compute_stream.wait_event.assert_called_once()
