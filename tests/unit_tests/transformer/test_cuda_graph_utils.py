# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from unittest.mock import Mock

import torch

import megatron.core.transformer.cuda_graph_utils as cuda_graph_utils


def test_default_stream_allocation_is_noop_on_default_stream(monkeypatch):
    default_stream = Mock()
    stream_context = Mock(side_effect=AssertionError("unexpected stream context"))
    event = Mock(side_effect=AssertionError("unexpected allocation event"))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: default_stream)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda: default_stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "Event", event)

    with cuda_graph_utils.default_stream_allocation():
        pass

    stream_context.assert_not_called()
    event.assert_not_called()


def test_default_stream_allocation_preserves_capture_stream(monkeypatch):
    capture_stream = Mock()
    default_stream = Mock(side_effect=AssertionError("unexpected default-stream query"))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: capture_stream)
    monkeypatch.setattr(torch.cuda, "default_stream", default_stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with cuda_graph_utils.default_stream_allocation():
        pass

    default_stream.assert_not_called()


def test_default_stream_allocation_orders_caller(monkeypatch):
    caller_stream = Mock()
    allocation_stream = Mock()
    allocation_ready_event = Mock()
    stream_context = Mock(return_value=nullcontext())

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller_stream)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda: allocation_stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=allocation_ready_event))

    with cuda_graph_utils.default_stream_allocation():
        pass

    stream_context.assert_called_once_with(allocation_stream)
    allocation_ready_event.record.assert_called_once_with(allocation_stream)
    caller_stream.wait_event.assert_called_once_with(allocation_ready_event)
