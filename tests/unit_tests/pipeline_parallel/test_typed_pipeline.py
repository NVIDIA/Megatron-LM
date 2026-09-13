# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed boundary protocol and autograd checks independent of model semantics."""

import gc
import os
import weakref
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    PipelinePayload,
    PipelinePayloadPlan,
    PipelinePayloadSpec,
    PipelineTensorSpec,
)
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    deallocate_output_tensor,
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.pipeline_parallel.typed_p2p_communication import (
    TypedP2PCommunicator,
    _pack_header,
    _unpack_header,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass(frozen=True)
class _Payload(PipelinePayload):
    tensors: tuple[torch.Tensor, ...]

    @property
    def metadata(self):
        return self.tensors[0].shape[0], self.tensors[1].shape[0]

    @property
    def tensor_specs(self):
        return tuple(
            PipelineTensorSpec(str(i), tuple(t.shape), t.dtype, t.requires_grad)
            for i, t in enumerate(self.tensors)
        )


def test_joint_backward_sums_local_and_relay_contributions():
    hidden = torch.tensor([1.0, 2.0], requires_grad=True)
    shared = torch.tensor([3.0, 4.0], requires_grad=True)
    indexer = torch.tensor([5.0, 6.0], requires_grad=True)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    incoming = _Payload((hidden, shared, indexer, indices))
    weight = torch.tensor(2.0, requires_grad=True)
    outgoing = _Payload((hidden * weight + shared.square(), shared, indexer, indices))
    original_shapes = tuple(t.shape for t in outgoing.tensors)
    deallocate_output_tensor(outgoing, True)
    assert tuple(t.shape for t in outgoing.tensors) == original_shapes
    gradients = backward_step(
        incoming,
        outgoing,
        (torch.tensor([7.0, 8.0]), torch.tensor([9.0, 10.0]), None, None),
        SimpleNamespace(
            timers=None, grad_scale_func=lambda loss: loss * 100, deallocate_pipeline_outputs=True
        ),
    )
    torch.testing.assert_close(gradients[0], torch.tensor([14.0, 16.0]))
    torch.testing.assert_close(gradients[1], torch.tensor([51.0, 74.0]))
    torch.testing.assert_close(gradients[2], torch.zeros(2))
    assert gradients[3] is None
    torch.testing.assert_close(weight.grad, torch.tensor(23.0))


def test_terminal_payload_backward_scales_loss_once():
    hidden = torch.tensor([2.0, 3.0], requires_grad=True)
    state = torch.tensor([4.0, 5.0], requires_grad=True)
    incoming = _Payload((hidden, state))
    grads = backward_step(
        incoming,
        (hidden * state).sum(),
        None,
        SimpleNamespace(
            timers=None, grad_scale_func=lambda loss: loss * 3, deallocate_pipeline_outputs=True
        ),
    )
    torch.testing.assert_close(grads[0], torch.tensor([12.0, 15.0]))
    torch.testing.assert_close(grads[1], torch.tensor([6.0, 9.0]))


def test_header_preserves_mixed_dtypes_empty_shapes_and_host_metadata():
    specs = (
        PipelineTensorSpec("hidden", (9, 1, 128), torch.bfloat16, True),
        PipelineTensorSpec("mix", (9, 1, 4), torch.float32, True),
        PipelineTensorSpec("empty", (0, 1, 16), torch.bfloat16, True),
        PipelineTensorSpec("ids", (9, 3), torch.int32, False),
        PipelineTensorSpec("prefixes", (5,), torch.int64, False),
        PipelineTensorSpec("scalar", (), torch.float64, False),
    )
    values = _pack_header(specs, (12345, 4), 7)
    decoded, metadata = _unpack_header(values, 7)
    assert _pack_header(decoded, metadata, 7) == values
    assert metadata == (12345, 4)
    with pytest.raises(ValueError, match="microbatch mismatch"):
        _unpack_header(values, 8)
    with pytest.raises(ValueError, match="chunk mismatch"):
        _unpack_header(values, 7, chunk_id=1)


@pytest.mark.parametrize(
    "spec",
    [
        PipelineTensorSpec("bad", (-1, 3), torch.float32, True),
        PipelineTensorSpec("bad", (1,) * 4, torch.float32, True),
        PipelineTensorSpec("bad", (1,), torch.int32, True),
    ],
)
def test_invalid_wire_descriptors(spec):
    with pytest.raises(ValueError, match="[Ii]nvalid"):
        _pack_header((spec,), (0, 0), 0)


def _make_payload(tensors, metadata):
    payload = _Payload(tensors)
    assert payload.metadata == metadata
    return payload


@pytest.fixture
def delayed_transport(monkeypatch):
    """Control request completion without letting the fake Work retain wire buffers."""
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(dist, "get_global_rank", lambda group, rank: rank)
    config = TransformerConfig(
        num_layers=4,
        hidden_size=8,
        num_attention_heads=1,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        virtual_pipeline_model_parallel_size=2,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
    )
    communicator = TypedP2PCommunicator(
        group, config, [_make_payload, _make_payload], device=torch.device("cpu")
    )
    requests = []

    class DelayedWork:
        def __init__(self, name, tensor):
            self.name = name
            self.buffer = weakref.ref(tensor)
            self.wait_count = 0
            self.received_value = None

        def wait(self):
            assert self.buffer() is not None, "Communication buffer was released before wait"
            self.wait_count += 1
            if self.received_value is not None:
                with torch.no_grad():
                    self.buffer().copy_(self.received_value)

    def post(**kwargs):
        result = {}
        for key, tensor in kwargs.items():
            if key.startswith("tensor_") and tensor is not None:
                name = key.removeprefix("tensor_")
                request = DelayedWork(name, tensor)
                requests.append(request)
                result[name] = request
        return result

    monkeypatch.setattr("megatron.core.pipeline_parallel.typed_p2p_communication._p2p_ops", post)
    return communicator, requests, post


def test_overlap_waits_each_direction_and_retains_wire_copies(delayed_transport):
    communicator, requests, _ = delayed_transport
    view = torch.arange(12.0, requires_grad=True).reshape(3, 4).t()
    expected = view.detach().clone()
    handles = communicator._transfer(
        send_next=(view, torch.empty(0)),
        send_prev=(torch.ones(2),),
        recv_prev=(torch.empty(3),),
        recv_next=(torch.empty(4),),
        overlap=True,
    )
    assert set(handles) == {"send_next", "send_prev", "recv_prev", "recv_next"}
    assert len(requests) == 4  # The empty field has no wire operation.
    assert all(request.wait_count == 0 for request in requests)
    sent = next(request for request in requests if request.name == "send_next")
    assert sent.buffer().is_contiguous() and not sent.buffer().requires_grad
    del view
    gc.collect()
    torch.testing.assert_close(sent.buffer(), expected)

    handles["recv_prev"].wait()
    assert all(request.wait_count == int(request.name == "recv_prev") for request in requests)
    assert len(communicator._pending) == 3
    handles["send_next"].wait()
    handles["send_next"].wait()  # Idempotent even if schedule cleanup waits again.
    assert sent.wait_count == 1 and sent.buffer() is None
    del handles  # finish() must retain and drain requests not kept by the caller.
    communicator.finish()
    assert not communicator._pending
    assert all(request.wait_count == 1 and request.buffer() is None for request in requests)


def test_overlap_constructs_payload_after_header_and_waits_before_data_use(
    delayed_transport, monkeypatch
):
    communicator, requests, post = delayed_transport
    source = _Payload((torch.ones(3, requires_grad=True), torch.ones(0, dtype=torch.bfloat16)))
    wire_header = torch.tensor(_pack_header(source.tensor_specs, source.metadata, 0, chunk_id=1))

    def post_receive(**kwargs):
        result = post(**kwargs)
        for name, request in result.items():
            if name == "recv_prev":
                request.received_value = wire_header if len(requests) == 1 else 7
        return result

    def factory(tensors, metadata):
        assert requests[0].wait_count == 1  # Header is parsed synchronously.
        assert all(request.wait_count == 0 for request in requests[1:])
        return _make_payload(tensors, metadata)

    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.typed_p2p_communication._p2p_ops", post_receive
    )
    communicator.make_payloads[1] = factory
    payload, handles = communicator.send_forward_recv_forward(
        None, True, None, overlap_p2p_comm=True, recv_chunk_id=1
    )
    assert payload.metadata == (3, 0)
    assert len(requests) == 2 and requests[1].wait_count == 0
    handles["recv_prev"].wait()
    torch.testing.assert_close(payload.tensors[0], torch.full((3,), 7.0))
    assert payload.tensors[1].numel() == 0

    # Gradients use the same chunk's FIFO; missing contributions become zeros.
    _, handles = communicator.send_backward_recv_backward(
        (None, None), False, None, overlap_p2p_comm=True, send_chunk_id=1
    )
    assert requests[-1].wait_count == 0
    torch.testing.assert_close(requests[-1].buffer(), torch.zeros(3))
    communicator.finish()
    assert all(request.wait_count == 1 for request in requests)
    assert not communicator._pending


def test_overlap_empty_directions_still_have_wait_handles(delayed_transport):
    communicator, requests, _ = delayed_transport
    handles = communicator._transfer(
        send_next=(torch.empty(0),), recv_next=(None, torch.empty(0)), overlap=True
    )
    assert set(handles) == {"send_next", "recv_next"}
    assert not requests
    for handle in handles.values():
        handle.wait()
    communicator.finish()
    assert not communicator._pending


@pytest.mark.parametrize("next_is_backward", [False, True])
def test_warmup_prefetch_defers_header_and_preserves_peer_order(
    delayed_transport, monkeypatch, next_is_backward
):
    communicator, requests, post = delayed_transport
    communicator.config.overlap_p2p_comm_warmup_flush = True
    sources = [
        _Payload(
            (torch.ones(3, requires_grad=True), torch.empty(0), torch.ones(2, dtype=torch.int32))
        ),
        _Payload((torch.ones(5, requires_grad=True), torch.ones(1, dtype=torch.bfloat16))),
    ]
    headers = iter(
        torch.tensor(_pack_header(source.tensor_specs, source.metadata, i, chunk_id=1))
        for i, source in enumerate(sources)
    )
    posted, factories = [], []

    def post_receive(**kwargs):
        result = post(**kwargs)
        for name, request in result.items():
            tensor = request.buffer()
            is_header = name == "recv_prev" and tensor.dtype == torch.int64
            posted.append((name, "header" if is_header else tuple(tensor.shape)))
            if name.startswith("recv_"):
                request.received_value = next(headers) if is_header else 7
        return result

    def factory(tensors, metadata):
        factories.append(metadata)
        return _make_payload(tensors, metadata)

    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.typed_p2p_communication._p2p_ops", post_receive
    )
    communicator.make_payloads[1] = factory
    first, first_handles = communicator.send_forward_recv_forward(
        None, True, None, overlap_p2p_comm=True, recv_chunk_id=1
    )
    assert posted == [("recv_prev", "header")]
    assert requests[0].wait_count == 0 and not factories
    assert not isinstance(first, PipelinePayload)

    if next_is_backward:
        # A gradient receive sharing PP=2's peer must not consume forward data.
        communicator._sent[0].append(sources[0].tensor_specs)
        _, next_handles = communicator.send_backward_recv_backward(
            None, True, None, overlap_p2p_comm=True, recv_chunk_id=0
        )
        next_direction, next_shape = "recv_next", (3,)
    else:
        second, next_handles = communicator.send_forward_recv_forward(
            None, True, None, overlap_p2p_comm=True, recv_chunk_id=1
        )
        next_direction, next_shape = "recv_prev", "header"
    assert posted == [
        ("recv_prev", "header"),
        ("recv_prev", (3,)),
        ("recv_prev", (2,)),
        (next_direction, next_shape),
    ]
    assert requests[0].wait_count == 1
    assert all(request.wait_count == 0 for request in requests[1:])
    assert factories == [(3, 0)]

    first_handles["recv_prev"].wait()
    payload = communicator.resolve_forward(first)
    torch.testing.assert_close(payload.tensors[0], torch.full((3,), 7.0))
    reference = weakref.ref(payload)
    del payload
    assert reference() is None  # A retained wait handle must not retain model inputs.
    with pytest.raises(RuntimeError, match="already been consumed"):
        communicator.resolve_forward(first)
    next_handles[next_direction].wait()
    if not next_is_backward:
        payload = communicator.resolve_forward(second)
        torch.testing.assert_close(payload.tensors[0], torch.full((5,), 7.0))
        assert factories == [(3, 0), (5, 1)]
    for source in sources[: 1 if next_is_backward else 2]:
        communicator.send_backward_recv_backward(
            (None,) * len(source.tensors), False, None, overlap_p2p_comm=True, send_chunk_id=1
        )
    communicator.finish()
    assert not communicator._pending and communicator._forward_header is None
    assert all(request.wait_count == 1 for request in requests)


def test_warmup_send_retains_header_and_all_fields_until_wait(delayed_transport):
    communicator, requests, _ = delayed_transport
    communicator.config.overlap_p2p_comm_warmup_flush = True
    communicator.forward_only = True
    source = _Payload(
        (torch.ones(2, 3, requires_grad=True).t(), torch.ones(2, dtype=torch.bfloat16))
    )
    _, handles = communicator.send_forward_recv_forward(source, False, None, overlap_p2p_comm=True)
    assert len(requests) == 3 and all(request.wait_count == 0 for request in requests)
    assert requests[0].buffer().dtype == torch.int64
    assert all(request.buffer().is_contiguous() for request in requests)
    del source
    handles["send_next"].wait()
    communicator.finish()
    assert all(request.wait_count == 1 and request.buffer() is None for request in requests)


def test_retiring_send_advances_prefetched_data_before_wait(delayed_transport, monkeypatch):
    communicator, requests, post = delayed_transport
    communicator.config.overlap_p2p_comm_warmup_flush = True
    communicator.forward_only = True
    source = _Payload((torch.ones(3), torch.empty(0)))
    wire_header = torch.tensor(_pack_header(source.tensor_specs, source.metadata, 0, chunk_id=1))

    def post_receive(**kwargs):
        result = post(**kwargs)
        for name, request in result.items():
            if name == "recv_prev":
                request.received_value = wire_header if request.buffer().dtype == torch.int64 else 9
        return result

    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.typed_p2p_communication._p2p_ops", post_receive
    )
    received, _ = communicator.send_forward_recv_forward(
        None, True, None, overlap_p2p_comm=True, recv_chunk_id=1
    )
    empty = communicator._transfer(send_prev=(torch.empty(0),), overlap=True)
    empty["send_prev"].wait()
    assert len(requests) == 1 and requests[0].wait_count == 0
    _, handles = communicator.send_forward_recv_forward(source, False, None, overlap_p2p_comm=True)
    assert len(requests) == 3 and all(request.wait_count == 0 for request in requests)
    handles["send_next"].wait()
    # Retiring a send must first post the prefetched payload's data. Otherwise
    # neighbors can both block retiring sends that have no posted data receives.
    assert len(requests) == 4 and requests[0].wait_count == 1
    assert requests[-1].name == "recv_prev" and requests[-1].wait_count == 0
    payload = communicator.resolve_forward(received)
    torch.testing.assert_close(payload.tensors[0], torch.full((3,), 9.0))
    communicator.finish()
    assert not communicator._pending


@pytest.mark.parametrize("warmup_flush", [False, True])
def test_prepared_receive_posts_all_fields_without_waiting_for_a_header(
    delayed_transport, monkeypatch, warmup_flush
):
    communicator, requests, post = delayed_transport
    communicator.config.overlap_p2p_comm_warmup_flush = warmup_flush
    communicator.forward_only = True
    source = _Payload((torch.ones(3), torch.empty(0), torch.ones(2, dtype=torch.int64)))
    descriptor = PipelinePayloadSpec(source.tensor_specs, source.metadata)
    communicator.payload_plans = [
        PipelinePayloadPlan((None,), (descriptor,)),
        PipelinePayloadPlan((descriptor,), (None,)),
    ]

    def no_header(*args, **kwargs):
        pytest.fail("Prepared communication must never build or parse a header")

    def post_receive(**kwargs):
        result = post(**kwargs)
        for name, request in result.items():
            if name == "recv_prev":
                request.received_value = 7
        return result

    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.typed_p2p_communication._pack_header", no_header
    )
    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.typed_p2p_communication._unpack_header", no_header
    )
    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.typed_p2p_communication._p2p_ops", post_receive
    )
    received, handles = communicator.send_forward_recv_forward(
        source, True, None, overlap_p2p_comm=True, send_chunk_id=0, recv_chunk_id=1
    )
    assert isinstance(received, PipelinePayload)
    assert len(requests) == 4  # Two nonempty fields in each direction; no header.
    assert all(request.wait_count == 0 for request in requests)
    assert communicator._forward_header is None
    handles["recv_prev"].wait()
    torch.testing.assert_close(received.tensors[0], torch.full((3,), 7.0))
    assert received.tensors[1].numel() == 0
    communicator.finish()
    assert all(request.wait_count == 1 for request in requests)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "grad", "metadata"])
def test_prepared_sender_rejects_stale_plan_before_posting(delayed_transport, mismatch):
    communicator, requests, _ = delayed_transport
    source = _Payload((torch.ones(3, requires_grad=True), torch.empty(0)))
    spec = source.tensor_specs[0]
    replacement = PipelineTensorSpec(
        spec.name,
        (5,) if mismatch == "shape" else spec.shape,
        torch.bfloat16 if mismatch == "dtype" else spec.dtype,
        False if mismatch == "grad" else spec.requires_grad,
    )
    descriptor = PipelinePayloadSpec(
        (replacement, source.tensor_specs[1]), (5, 0) if mismatch == "metadata" else source.metadata
    )
    communicator.payload_plans = [PipelinePayloadPlan((None,), (descriptor,))] * 2
    with pytest.raises(ValueError, match="disagrees"):
        communicator.send_forward_recv_forward(source, False, None, overlap_p2p_comm=True)
    assert not requests


class _Timer:
    def __init__(self):
        self.started = self.stopped = 0

    def start(self, **kwargs):
        assert self.started == self.stopped
        self.started += 1

    def stop(self):
        assert self.started == self.stopped + 1
        self.stopped += 1


class _Timers(dict):
    def __call__(self, name, **kwargs):
        if name not in self:
            self[name] = _Timer()
        return self[name]


class _Stage(torch.nn.Module):
    def __init__(self, config, rank, size, device):
        super().__init__()
        self.config = config
        self.model_type = ModelType.encoder_or_decoder
        self.rank = rank
        self.size = size
        self.weight = torch.nn.Parameter(torch.tensor((rank + 1) / 2, device=device))
        self.pipeline_payload_factory = _make_payload
        self.input_tensor = None
        self.references = []

    def set_input_tensor(self, inputs):
        self.input_tensor = inputs[0] if isinstance(inputs, list) else inputs

    def forward(self, x):
        incoming, self.input_tensor = self.input_tensor, None
        if self.rank == 0:
            hidden = x * self.weight
            rows = x.shape[0] // 2
            # A non-contiguous differentiable view, including zero capacity.
            shared = (self.weight * torch.ones(2, rows, device=x.device)).to(torch.bfloat16).t()
            unused = self.weight * torch.ones(x.shape[0], device=x.device)
            indices = torch.zeros(x.shape[0], 2, device=x.device, dtype=torch.int32)
            prefixes = torch.arange(x.shape[0] % 3 + 2, device=x.device, dtype=torch.int64)
        else:
            self.references.append(weakref.ref(incoming))
            hidden, shared, unused, indices, prefixes = incoming.tensors
            hidden = hidden * self.weight + shared.float().sum() * self.weight
        if self.rank == self.size - 1:
            return hidden.sum()
        payload = _Payload((hidden, shared, unused, indices, prefixes))
        self.references.append(weakref.ref(payload))
        return payload


@pytest.fixture
def transport_groups():
    """Use NCCL on GPUs; Gloo also exercises the real protocol on CPU-only hosts."""
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size < 2:
        pytest.skip("Launch with torch.distributed.run and at least two processes")
    rank = int(os.environ["RANK"])
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=60)
        )
    singleton_groups = [dist.new_group([i]) for i in range(size)]
    groups = ProcessGroupCollection()
    groups.pp = dist.group.WORLD
    groups.tp = groups.cp = singleton_groups[rank]
    dist.barrier()
    yield groups, device
    dist.barrier()
    dist.destroy_process_group(singleton_groups[rank])


@pytest.mark.parametrize(
    "batch_p2p,overlap,warmup_flush",
    [(False, False, False), (True, False, False), (False, True, False), (False, True, True)],
    ids=["blocking", "batched", "overlap", "warmup-flush"],
)
@pytest.mark.parametrize(
    "vp_size,count,forward_only,group_multiple",
    [
        (1, 1, False, 1),
        (1, 4, False, 1),
        (1, 3, False, 1),
        (1, 4, True, 1),
        (2, 1, False, 1),
        (2, 3, False, 1),
        (2, 3, False, 2),
        (2, 3, True, 2),
        (3, 3, False, 1),
        (3, 3, True, 2),
    ],
)
@pytest.mark.parametrize("planned", [False, True], ids=["dynamic-header", "prepared"])
def test_typed_1f1b_actual_transport(
    transport_groups,
    batch_p2p,
    overlap,
    warmup_flush,
    vp_size,
    count,
    forward_only,
    group_multiple,
    planned,
    monkeypatch,
):
    """Exercise ordinary/VPP P2P, rank wraparound, dynamic FIFO shapes and release."""
    if overlap and vp_size == 1:
        pytest.skip("P2P overlap requires VPP")
    groups, device = transport_groups
    rank, size = groups.pp.rank(), groups.pp.size()
    if vp_size > 1:
        count *= size
    config = TransformerConfig(
        num_layers=size * vp_size,
        hidden_size=8,
        num_attention_heads=1,
        pipeline_model_parallel_size=size,
        virtual_pipeline_model_parallel_size=vp_size if vp_size > 1 else None,
        microbatch_group_size_per_vp_stage=size * group_multiple,
        pipeline_dtype=torch.float32,
        batch_p2p_comm=batch_p2p,
        overlap_p2p_comm=overlap,
        overlap_p2p_comm_warmup_flush=warmup_flush,
        deallocate_pipeline_outputs=True,
        timers=_Timers(),
    )
    models = [_Stage(config, rank + vp * size, size * vp_size, device) for vp in range(vp_size)]
    reference = [_Stage(config, i, size * vp_size, device) for i in range(size * vp_size)]
    batches = [
        torch.arange((1, 4, 7)[i % 3] * 3, device=device).float().reshape(-1, 1, 3) / 8
        for i in range(count)
    ]
    if planned:

        def no_header(*args, **kwargs):
            pytest.fail("Prepared transport must not use dynamic headers")

        monkeypatch.setattr(
            "megatron.core.pipeline_parallel.typed_p2p_communication._pack_header", no_header
        )
        monkeypatch.setattr(
            "megatron.core.pipeline_parallel.typed_p2p_communication._unpack_header", no_header
        )

    for _ in range(2):
        expected = []
        for model in models:
            model.zero_grad(set_to_none=True)
        for stage in reference:
            stage.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(not forward_only):
            for x in batches:
                value = None
                for stage in reference:
                    stage.set_input_tensor(value)
                    value = stage(x)
                expected.append(value.detach().clone())
                if not forward_only:
                    (value / count).backward()
            actual = []

            def forward_step(iterator, stage):
                output = stage(next(iterator))

                def loss_func(loss):
                    actual.append(loss.detach().clone())
                    return loss, {"loss": loss.detach().clone()}

                return output, loss_func

            iterators = [iter(batches) for _ in models]
            if planned:
                descriptors = []
                for x in batches:
                    rows = x.shape[0]
                    fields = [
                        (tuple(x.shape), torch.float32, not forward_only),
                        ((rows // 2, 2), torch.bfloat16, not forward_only),
                        ((rows,), torch.float32, not forward_only),
                        ((rows, 2), torch.int32, False),
                        ((rows % 3 + 2,), torch.int64, False),
                    ]
                    descriptors.append(
                        PipelinePayloadSpec(
                            tuple(
                                PipelineTensorSpec(str(i), *field) for i, field in enumerate(fields)
                            ),
                            (rows, rows // 2),
                        )
                    )
                iterators = [
                    PipelineDataIterator(
                        batches,
                        PipelinePayloadPlan(
                            tuple(descriptors) if stage.rank else (None,) * count,
                            tuple(descriptors) if stage.rank < stage.size - 1 else (None,) * count,
                        ),
                    )
                    for stage in models
                ]

                def prepare(iterator, stage, count, *, forward_only):
                    timer = stage.config.timers('forward-backward')
                    assert timer.started == timer.stopped + 1
                    assert count == len(iterator.pipeline_payload_plan.incoming)
                    return iterator

                forward_step.prepare_pipeline_inputs = prepare

            schedule = (
                forward_backward_pipelining_with_interleaving
                if vp_size > 1
                else forward_backward_pipelining_without_interleaving
            )
            schedule(
                forward_step_func=forward_step,
                data_iterator=iterators if vp_size > 1 else iterators[0],
                model=models if vp_size > 1 else models[0],
                num_microbatches=count,
                seq_length=99,  # The wire shapes deliberately differ from the configured shape.
                micro_batch_size=1,
                forward_only=forward_only,
                p2p_communicator=P2PCommunicator(groups.pp, config),
                pg_collection=groups,
            )
        if rank == size - 1:
            torch.testing.assert_close(torch.stack(actual), torch.stack(expected), atol=0, rtol=0)
        gc.collect()
        for model in models:
            if not forward_only:
                torch.testing.assert_close(
                    model.weight.grad, reference[model.rank].weight.grad, atol=1e-3, rtol=5e-3
                )
            else:
                assert model.weight.grad is None
            assert model.input_tensor is None and all(ref() is None for ref in model.references)
            model.references.clear()
        assert all(timer.started == timer.stopped for timer in config.timers.values())
        if rank < size - 1 or not forward_only:
            assert any("send" in name for name in config.timers)
        if rank > 0 or not forward_only:
            assert any("recv" in name for name in config.timers)
