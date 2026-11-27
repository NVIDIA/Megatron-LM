# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from contextlib import contextmanager
from itertools import count
from types import SimpleNamespace

import pytest
import torch

from megatron.core.activation_policy.offload import fine_grained_activation_offload as offload
from megatron.core.activation_policy.recompute import checkpointing
from megatron.core.extensions import transformer_engine
from megatron.core.tensor_parallel import random as tp_random


def make_offload_manager(monkeypatch):
    monkeypatch.setattr(offload.PipelineOffloadManager, "OFFLOAD_MGR", None)
    return offload.PipelineOffloadManager.get_instance()


@pytest.fixture
def cpu_cuda(monkeypatch):
    trace, pinned = [], set()
    stream_ids = count()

    class Stream:
        def __init__(self, stream_id=None) -> None:
            self.stream_id = next(stream_ids) if stream_id is None else stream_id

        def __eq__(self, other):
            return isinstance(other, Stream) and self.stream_id == other.stream_id

        def __hash__(self):
            return hash(self.stream_id)

        def wait_event(self, event):
            event.wait(self)

        def wait_stream(self, stream):
            assert isinstance(stream, Stream)
            trace.append(("wait_stream", self, stream))

    class Event:
        def __init__(self) -> None:
            self.stream = None

        def record(self, stream=None):
            self.stream = current if stream is None else stream
            assert isinstance(self.stream, Stream)
            trace.append(("record", self.stream, self))

        def wait(self, stream=None):
            assert self.stream is not None
            trace.append(("wait", current if stream is None else stream, self))

    compute = current = Stream()

    @contextmanager
    def use_stream(stream=None):
        nonlocal current
        previous = current
        if stream is not None:
            assert isinstance(stream, Stream)
            current = stream
        try:
            yield
        finally:
            current = previous

    original_empty = torch.empty
    original_copy = torch.Tensor.copy_

    def empty(*size, device=None, **kwargs):
        if device is None or torch.device(device).type != "cuda":
            return original_empty(*size, device=device, **kwargs)
        tensor = original_empty(*size, device="cpu", **kwargs)
        trace.append(("allocate", current, tensor.data_ptr()))
        return tensor

    def copy_(destination, source, non_blocking=False):
        dst_pinned, src_pinned = destination.is_pinned(), source.is_pinned()
        if dst_pinned != src_pinned:
            direction = "d2h" if dst_pinned else "h2d"
            trace.append((direction, current, destination.data_ptr(), non_blocking))
        return original_copy(destination, source, non_blocking=non_blocking)

    def record_stream(tensor, stream):
        assert not tensor.is_pinned()
        assert isinstance(stream, Stream)
        trace.append(("retain", stream, tensor.data_ptr()))

    class TracedTensors(dict):
        def pop(self, tag, *default):
            tensor = super().pop(tag, *default)
            if isinstance(tensor, torch.Tensor):
                trace.append(("pop", current, tensor.data_ptr()))
            return tensor

        def __delitem__(self, tag):
            tensor = self[tag]
            if isinstance(tensor, torch.Tensor):
                trace.append(("drop", tensor.data_ptr()))
            super().__delitem__(tag)

        def clear(self):
            for tag in list(self):
                del self[tag]

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: Stream(current.stream_id))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    monkeypatch.setattr(
        torch.Tensor, "is_pinned", lambda tensor, device=None: tensor.untyped_storage() in pinned
    )
    return SimpleNamespace(trace=trace, compute=compute, pinned=pinned, tensors=TracedTensors)


@pytest.fixture
def cpu_offload(monkeypatch, cpu_cuda):
    monkeypatch.setattr(transformer_engine, "cpu_offload", SimpleNamespace(CPUOffloadEnabled=False))
    monkeypatch.setattr(
        offload.ChunkOffloadHandler, "_can_manage_tensor_for_offload", staticmethod(lambda _: True)
    )
    monkeypatch.setattr(offload, "nvtx_range_push", lambda _: None)
    monkeypatch.setattr(offload, "nvtx_range_pop", lambda _: None)
    original_offload = offload.ChunkOffloadHandler.offload

    def offload_to_cpu(handler, *args, **kwargs):
        _, backup, use_pool, view = original_offload(handler, *args, **kwargs)
        return torch.device("cuda"), backup, use_pool, view

    monkeypatch.setattr(offload.ChunkOffloadHandler, "offload", offload_to_cpu)
    manager = make_offload_manager(monkeypatch)
    pool = manager.cpu_tensor_pool
    pool.pin_memory = False
    allocate = pool.allocate

    def allocate_backup(*args, **kwargs):
        backup = allocate(*args, **kwargs)
        cpu_cuda.pinned.add(backup.untyped_storage())
        return backup

    monkeypatch.setattr(pool, "allocate", allocate_backup)
    return manager


def init_chunk(manager):
    manager.init_model_chunk_offload_handler(0, None, None, min_offloaded_tensor_size=2)
    return manager.cur_forward_chunk()


def save_group(chunk, *tensors):
    group = chunk.on_group_start_forward("qkv_linear")
    tags = [chunk.tensor_push(tensor) for tensor in tensors]
    chunk.on_group_commit_forward("qkv_linear", [])
    return group, tags


def assert_reload_trace(trace, h2d, allocation_stream, copy_count):
    copies = [entry for entry in trace if entry[0] == "h2d"]
    assert len(copies) == copy_count
    assert all(stream == h2d and non_blocking for _, stream, _, non_blocking in copies)
    records = [entry for entry in trace if entry[:2] == ("record", h2d)]
    assert len(records) == 1
    (record,) = records
    for copy in copies:
        allocation = ("allocate", allocation_stream, copy[2])
        assert allocation in trace, "Reload destination was not allocated on the consumer stream"
        assert trace.index(allocation) < trace.index(copy) < trace.index(record)
    return copies, record


def test_mla_recompute_waits_before_unpack(monkeypatch, cpu_cuda, cpu_offload):
    trace, compute = cpu_cuda.trace, cpu_cuda.compute
    monkeypatch.setattr(tp_random, "_get_cuda_rng_state", lambda **kwargs: None)
    monkeypatch.setattr(tp_random, "_set_cuda_rng_state", lambda *args, **kwargs: None)

    def restore_storage(dst, src):
        source = src.untyped_storage()
        dst.untyped_storage().resize_(source.nbytes()).copy_(source)

    monkeypatch.setattr(checkpointing, "_get_share_storage", lambda: restore_storage)

    def up_project(resident, q_compressed, kv_compressed):
        if torch.is_grad_enabled():
            trace.append(("recompute", torch.cuda.current_stream()))
        return q_compressed * 2 + resident, kv_compressed * 3, kv_compressed * 4

    class Attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value):
            ctx.save_for_backward(query, key)
            return query * key + value

        @staticmethod
        def backward(ctx, grad):
            trace.append(("attention backward", torch.cuda.current_stream()))
            query, key = ctx.saved_tensors
            return grad * key, grad * query, grad

    chunk = init_chunk(cpu_offload)
    hidden = torch.tensor([2.0, 3.0], device="cpu", requires_grad=True)
    checkpoint = checkpointing.CheckpointWithoutOutput(fp8=None)
    scope = offload.FineGrainedActivationOffloadingInterface(True, hidden, "qkv_linear")
    with scope as activation:
        query, key, value = checkpoint.checkpoint(
            up_project, activation[0], activation + 1, activation + 2
        )
    query = scope.group_offload(query)
    output = Attention.apply(query, key, value)
    attention_output_grad = ("attention output gradient", compute)
    output.register_hook(lambda _: trace.append(attention_output_grad))
    checkpoint.discard_output_and_register_recompute(output)

    # The projection's Start prefetches QKV before the attention output hook runs.
    scope = offload.FineGrainedActivationOffloadingInterface(True, output, "attn_proj")
    with scope as activation:
        output = activation + 1
    output = scope.group_offload(output)
    qkv = chunk.offload_groups[0]
    monkeypatch.setattr(qkv, "_tensors", cpu_cuda.tensors(qkv._tensors))
    output.sum().backward()

    d2h, h2d = cpu_offload.d2h_stream, cpu_offload.h2d_stream
    d2h_copies = [entry for entry in trace if entry[0] == "d2h"]
    h2d_copies, record_reload = assert_reload_trace(trace, h2d, compute, copy_count=2)
    assert len(d2h_copies) == 2
    assert all(stream == d2h and non_blocking for _, stream, _, non_blocking in d2h_copies)
    pops = [("pop", compute, address) for _, _, address, _ in h2d_copies]
    (record_offload,) = [entry for entry in trace if entry[:2] == ("record", d2h)]
    wait_offload = ("wait", h2d, record_offload[2])
    wait = ("wait", compute, record_reload[2])
    assert trace.index(d2h_copies[-1]) < trace.index(record_offload) < trace.index(wait_offload)
    assert trace.index(wait_offload) < trace.index(h2d_copies[0])
    assert trace.count(wait) == 1
    assert trace.index(record_reload) < trace.index(attention_output_grad) < trace.index(wait)
    assert trace.index(wait) < min(map(trace.index, pops)), "QKV unpack preceded the H2D wait"
    assert max(map(trace.index, pops)) < trace.index(("recompute", compute))
    assert trace.index(("recompute", compute)) < trace.index(("attention backward", compute))
    assert ("wait_stream", compute, d2h) not in trace
    assert ("wait", compute, record_offload[2]) not in trace
    assert not any(entry[:2] == ("retain", compute) for entry in trace)
    torch.testing.assert_close(hidden.grad, torch.tensor([79.0, 64.0], device="cpu"))


def test_auxiliary_streams_wait_once_and_retain_each_tensor(cpu_cuda, cpu_offload):
    chunk = init_chunk(cpu_offload)
    group, tags = save_group(chunk, *(torch.ones(2, device="cpu") for _ in range(4)))
    chunk.bulk_reload_group()

    trace, compute = cpu_cuda.trace, cpu_cuda.compute
    h2d = cpu_offload.h2d_stream
    _, record = assert_reload_trace(trace, h2d, compute, copy_count=len(tags))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for index, (tag, stream) in enumerate(zip(tags, streams * 2)):
        start = len(trace)
        with torch.cuda.stream(stream):
            tensor = chunk.tensor_pop(tag)
        unpack = trace[start:]
        expected_waits = [("wait", stream, record[2])] if index < len(streams) else []
        waits = [entry for entry in unpack if entry[0] in {"wait", "wait_stream"}]
        assert waits == expected_waits, "Each consuming stream must wait once for the reload"
        retain = ("retain", stream, tensor.data_ptr())
        assert retain in unpack, "The reload destination must be retained on its consuming stream"
        torch.testing.assert_close(tensor, torch.ones_like(tensor))
    assert not any(entry[:2] == ("wait", compute) for entry in trace)

    with torch.cuda.stream(streams[0]):
        chunk.on_group_commit_backward("qkv_linear")
        chunk.on_group_start_backward(group)
    assert group._reload_allocation_stream is None, "Group completion kept the allocation stream"
    assert not group._reload_waiters, "Group completion kept the reload waiters"

    chunk.reset()
    chunk.is_warmup = False
    reused_group, (tag,) = save_group(chunk, torch.ones(2, device="cpu"))
    assert reused_group is group
    start = len(trace)
    chunk.bulk_reload_group()
    with torch.cuda.stream(streams[0]):
        chunk.tensor_pop(tag)
    reload_trace = trace[start:]
    _, record = assert_reload_trace(reload_trace, h2d, compute, copy_count=1)
    assert ("wait", streams[0], record[2]) in reload_trace[reload_trace.index(record) + 1 :]


@pytest.mark.parametrize("capture", ["megatron", "stream"])
def test_unpack_skips_reload_wait_during_capture(monkeypatch, cpu_cuda, cpu_offload, capture):
    chunk = init_chunk(cpu_offload)
    group, (tag,) = save_group(chunk, torch.ones(2, device="cpu"))
    chunk.bulk_reload_group()
    trace = cpu_cuda.trace
    copies, _ = assert_reload_trace(trace, cpu_offload.h2d_stream, cpu_cuda.compute, copy_count=1)
    start = len(trace)

    monkeypatch.setattr(offload, "is_graph_capturing", lambda: capture == "megatron")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture == "stream")
    with torch.cuda.stream(torch.cuda.Stream()):
        tensor = chunk.tensor_pop(tag)

    assert not trace[start:], "Unpack issued a stream operation during capture"
    assert tag not in group._tensors
    assert tensor.data_ptr() == copies[0][2]
    torch.testing.assert_close(tensor, torch.ones_like(tensor))


def test_reset_releases_unused_prefetch_on_h2d_stream(monkeypatch, cpu_cuda, cpu_offload):
    chunk = init_chunk(cpu_offload)
    group, (consumed, unused) = save_group(chunk, *(torch.ones(2, device="cpu") for _ in range(2)))
    chunk.bulk_reload_group()
    h2d, trace = cpu_offload.h2d_stream, cpu_cuda.trace
    assert_reload_trace(trace, h2d, cpu_cuda.compute, copy_count=2)
    with torch.cuda.stream(torch.cuda.Stream()):
        chunk.tensor_pop(consumed)
    address = group._tensors[unused].data_ptr()
    monkeypatch.setattr(group, "_tensors", cpu_cuda.tensors(group._tensors))
    start = len(trace)
    chunk.reset()

    assert not group._tensors, "Reset retained an unused reload destination"
    assert trace[start:] == [("retain", h2d, address), ("drop", address)]
    assert not group._prefetched_tags, "Reset kept the prefetched tags"
    assert group._reload_allocation_stream is None, "Reset kept the allocation stream"
    assert not group._reload_waiters, "Reset kept the reload waiters"


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "resident"])
def test_non_offloaded_tensors_need_no_reload_wait(monkeypatch, cpu_cuda, cpu_offload, enabled):
    chunk = init_chunk(cpu_offload)
    hidden = torch.ones(1, device="cpu", requires_grad=True)
    scope = offload.FineGrainedActivationOffloadingInterface(enabled, hidden, "qkv_linear")
    with scope as activation:
        output = activation.square()
    if enabled:
        group = chunk.offload_groups[0]
        monkeypatch.setattr(group, "_tensors", cpu_cuda.tensors(group._tensors))
    scope.group_offload(output).sum().backward()
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2))
    h2d = cpu_offload.h2d_stream
    assert not any(
        entry[0] in {"d2h", "h2d", "wait", "retain"}
        or entry[:2] == ("record", h2d)
        or (entry[0] == "wait_stream" and entry[2] == h2d)
        for entry in cpu_cuda.trace
    )
    if enabled:
        assert ("pop", cpu_cuda.compute, activation.data_ptr()) in cpu_cuda.trace
    else:
        assert not cpu_cuda.trace


@pytest.fixture
def cuda_offload(monkeypatch):
    with torch.cuda.device(int(os.environ.get("LOCAL_RANK", 0))):
        manager = make_offload_manager(monkeypatch)
        try:
            yield manager
        finally:
            torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA activation offload")
def test_cuda_mla_recompute_matches_without_offload(monkeypatch, cuda_offload):
    interface = offload.FineGrainedActivationOffloadingInterface
    compute, h2d = torch.cuda.current_stream(), cuda_offload.h2d_stream
    prefetch = ("prefetch", h2d, None)
    trace = []
    event_record = torch.cuda.Event.record
    event_wait = torch.cuda.Event.wait
    stream_wait_event = torch.cuda.Stream.wait_event
    reload_tensor = offload.ChunkOffloadHandler.reload

    def record(event, stream=None):
        event_record(event, stream)
        stream = torch.cuda.current_stream() if stream is None else stream
        trace.append(("record", stream, event))

    def wait(event, stream=None):
        event_wait(event, stream)
        stream = torch.cuda.current_stream() if stream is None else stream
        trace.append(("wait", stream, event))

    def wait_event(stream, event):
        stream_wait_event(stream, event)
        trace.append(("wait", stream, event))

    def reload(handler, *args, **kwargs):
        tensor = reload_tensor(handler, *args, **kwargs)
        if torch.cuda.current_stream() == h2d:
            trace.append(prefetch)
        return tensor

    monkeypatch.setattr(torch.cuda.Event, "record", record)
    monkeypatch.setattr(torch.cuda.Event, "wait", wait)
    monkeypatch.setattr(torch.cuda.Stream, "wait_event", wait_event)
    monkeypatch.setattr(offload.ChunkOffloadHandler, "reload", reload)

    class LatentAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.compress = torch.nn.Linear(16, 8)
            self.query_up = torch.nn.Linear(8, 16)
            self.key_value_up = torch.nn.Linear(8, 32)
            self.output_bias = torch.nn.Parameter(torch.zeros(16))

        def up_project(self, latent, enabled):
            stream = torch.cuda.current_stream()
            step = "recompute" if torch.is_grad_enabled() else "up-projection"
            if enabled and step == "recompute":
                assert prefetch in trace, "MLA recompute preceded the H2D prefetch"
                before_read = trace[trace.index(prefetch) + 1 :]
                assert any(
                    operation == "record"
                    and producer == h2d
                    and ("wait", stream, event) in before_read[index + 1 :]
                    for index, (operation, producer, event) in enumerate(before_read)
                ), "MLA recompute must wait for the H2D event on the compute stream"
            trace.append((step, stream, None))
            return self.query_up(latent), self.key_value_up(latent)

        def forward(self, hidden, enabled):
            compressed = self.compress(hidden)
            scope = interface(enabled, compressed, "qkv_linear")
            checkpoint = checkpointing.CheckpointWithoutOutput(fp8=None)
            with scope as latent:
                query, key_value = checkpoint.checkpoint(self.up_project, latent, enabled)
            query = scope.group_offload(query)
            key, value = key_value.chunk(2, dim=-1)
            weights = (query @ key.transpose(-2, -1) * 0.25).softmax(dim=-1)
            attention = weights @ value
            checkpoint.discard_output_and_register_recompute(attention)

            # The bias addition saves nothing, so its backward can prefetch QKV.
            scope = interface(enabled, attention, "attn_proj")
            with scope as activation:
                output = activation + self.output_bias
            return scope.group_offload(output)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = LatentAttention().cuda()
    source = torch.linspace(-1, 1, 256, device="cuda").view(2, 8, 16)

    def run(enabled):
        model.zero_grad(set_to_none=True)
        hidden = source.detach().requires_grad_()
        if enabled:
            interface.init_chunk_handler(
                pp_rank=0,
                vp_size=None,
                vp_stage=None,
                min_offloaded_tensor_size=2,
                delta_offload_bytes_across_pp_ranks=0,
                activation_offload_fraction=1.0,
            )
        output = model(hidden, enabled)
        output.square().mean().backward()
        gradients = {"input": hidden.grad}
        gradients.update((name, parameter.grad) for name, parameter in model.named_parameters())
        assert all(grad is not None for grad in gradients.values())
        return output.detach(), gradients

    expected_output, expected_gradients = run(enabled=False)
    steps = [("up-projection", compute, None), ("recompute", compute, None)]
    assert [entry for entry in trace if entry[2] is None] == steps
    trace.clear()
    output, gradients = run(enabled=True)
    assert [entry for entry in trace if entry[2] is None] == [steps[0], prefetch, steps[1]]
    torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(gradients, expected_gradients, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires asynchronous CUDA transfers")
def test_cuda_unpack_waits_before_consumer_read(monkeypatch, cuda_offload):
    chunk = init_chunk(cuda_offload)
    source = torch.arange(2**20, device="cuda", dtype=torch.float32)
    _, (tag,) = save_group(chunk, source)
    producer, consumer = cuda_offload.h2d_stream, torch.cuda.Stream()
    recorded, waited = [], []
    event_record = torch.cuda.Event.record
    event_wait = torch.cuda.Event.wait

    def record(event, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        event_record(event, stream)
        if stream == producer:
            recorded.append(event)

    def wait(event, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        event_wait(event, stream)
        if stream == consumer:
            waited.append(event)

    monkeypatch.setattr(torch.cuda.Event, "record", record)
    # Stream.wait_event delegates to Event.wait.
    monkeypatch.setattr(torch.cuda.Event, "wait", wait)
    with torch.cuda.stream(consumer):
        chunk.bulk_reload_group()
        assert len(recorded) == 1
        assert not waited, "Prefetch made the consumer wait before unpack"
        tensor = chunk.tensor_pop(tag)
        assert waited == recorded, "Unpack must wait for the H2D event on the consumer stream"
        observed = tensor.clone()
    consumer.synchronize()
    torch.testing.assert_close(observed, source, rtol=0, atol=0)
