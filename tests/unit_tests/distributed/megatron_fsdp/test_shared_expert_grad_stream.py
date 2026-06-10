# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import megatron_fsdp


class _FakeStream:
    def __init__(self, name, cuda_stream, events, device="cuda:0"):
        self.name = name
        self.cuda_stream = cuda_stream
        self.device = device
        self._events = events

    def wait_stream(self, stream):
        self._events.append((self.name, "wait_stream", stream.name))


class _FakeStreamContext:
    def __init__(self, stream, events, active_streams):
        self.stream = stream
        self.events = events
        self.active_streams = active_streams

    def __enter__(self):
        self.events.append(("enter", self.stream.name))
        self.active_streams.append(self.stream)
        return self.stream

    def __exit__(self, exc_type, exc, tb):
        self.events.append(("exit", self.stream.name))
        self.active_streams.pop()
        return False


def test_shared_expert_grad_processing_bridges_to_default_stream(monkeypatch):
    """Regression guard for the FSDP shared-expert backward stream ordering bug.

    Before the fix, shared expert post-backward grad handling ran directly on the
    producing shared-expert stream. The fixed path must wait from default stream,
    process on default stream, then make the producing stream wait for default.
    """
    events = []
    active_streams = []
    shared_stream = _FakeStream("shared", 7, events)
    default_stream = _FakeStream("default", 0, events)

    monkeypatch.setattr(megatron_fsdp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_fsdp, "_get_shared_expert_stream", lambda: shared_stream)
    monkeypatch.setattr(megatron_fsdp.torch.cuda, "current_stream", lambda: shared_stream)
    monkeypatch.setattr(
        megatron_fsdp.torch.cuda, "default_stream", lambda device=None: default_stream
    )
    monkeypatch.setattr(
        megatron_fsdp.torch.cuda,
        "stream",
        lambda stream: _FakeStreamContext(stream, events, active_streams),
    )

    shared_param = torch.nn.Parameter(torch.empty(1))
    other_param = torch.nn.Parameter(torch.empty(1))
    param_to_name = {
        shared_param: "layers.0.mlp.shared_experts.linear_fc1.weight",
        other_param: "layers.0.mlp.experts.local_experts.weight",
    }

    def process_grads():
        assert active_streams == [default_stream]
        events.append(("process", active_streams[-1].name))

    routed_streams = megatron_fsdp._run_shared_expert_grad_processing_on_default_stream(
        param_to_name, [shared_param, other_param], process_grads
    )

    assert routed_streams == (shared_stream, default_stream)
    assert events == [
        ("default", "wait_stream", "shared"),
        ("enter", "default"),
        ("process", "default"),
        ("exit", "default"),
        ("shared", "wait_stream", "default"),
    ]


def test_shared_expert_grad_processing_leaves_non_shared_params_on_caller_stream(monkeypatch):
    events = []
    active_streams = []
    source_stream = _FakeStream("source", 7, events)
    default_stream = _FakeStream("default", 0, events)

    monkeypatch.setattr(megatron_fsdp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_fsdp, "_get_shared_expert_stream", lambda: source_stream)
    monkeypatch.setattr(megatron_fsdp.torch.cuda, "current_stream", lambda: source_stream)
    monkeypatch.setattr(
        megatron_fsdp.torch.cuda, "default_stream", lambda device=None: default_stream
    )
    monkeypatch.setattr(
        megatron_fsdp.torch.cuda,
        "stream",
        lambda stream: _FakeStreamContext(stream, events, active_streams),
    )

    param = torch.nn.Parameter(torch.empty(1))

    def process_grads():
        events.append(("process", "unexpected"))

    routed_streams = megatron_fsdp._run_shared_expert_grad_processing_on_default_stream(
        {param: "layers.0.mlp.experts.local_experts.weight"}, [param], process_grads
    )

    assert routed_streams is None
    assert events == []
