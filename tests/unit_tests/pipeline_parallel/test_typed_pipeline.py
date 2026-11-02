# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed boundary protocol and autograd checks independent of model semantics."""

import gc
import os
import weakref
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    PipelinePayloadPlan,
    PipelinePayloadSpec,
    TensorStatePayload,
    backward_pipeline_payload,
)
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.pipeline_parallel.typed_p2p_communication import TypedP2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.state_boundary import TensorField, TensorSchema
from megatron.core.transformer.transformer_config import TransformerConfig

_CONFIG = SimpleNamespace(timers=None, grad_scale_func=None, deallocate_pipeline_outputs=True)


def _field(key, shape, dtype, differentiable, layout="local", present=True):
    return TensorField(key, tuple(shape), dtype, layout, differentiable, present)


def _payload(*tensors, differentiable=None):
    """A received-style payload over the given tensors; the first is hidden."""
    fields = tuple(
        _field(
            "__hidden__" if i == 0 else f"field{i}",
            t.shape,
            t.dtype,
            t.requires_grad if differentiable is None else differentiable[i],
        )
        for i, t in enumerate(tensors)
    )
    return TensorStatePayload(tensors, PipelinePayloadSpec(fields))


@pytest.mark.parametrize("release", [False, True])
def test_joint_backward_sums_local_and_relay_contributions(release):
    hidden = torch.tensor([1.0, 2.0], requires_grad=True)
    shared = torch.tensor([3.0, 4.0], requires_grad=True)
    indexer = torch.tensor([5.0, 6.0], requires_grad=True)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    incoming = _payload(hidden, shared, indexer, indices)
    weight = torch.tensor(2.0, requires_grad=True)
    outgoing = _payload(hidden * weight + shared.square(), shared, indexer, indices)
    if release:
        outgoing.release_output()
    assert bool(outgoing.tensors) is not release
    gradients = backward_step(
        incoming,
        outgoing,
        (torch.tensor([7.0, 8.0]), torch.tensor([9.0, 10.0]), None, None),
        SimpleNamespace(**{**vars(_CONFIG), "grad_scale_func": lambda loss: loss * 100}),
    )
    torch.testing.assert_close(gradients[0], torch.tensor([14.0, 16.0]))
    torch.testing.assert_close(gradients[1], torch.tensor([51.0, 74.0]))
    assert gradients[2] is None and gradients[3] is None
    torch.testing.assert_close(weight.grad, torch.tensor(23.0))


def test_release_shrinks_owned_roots_and_keeps_saved_values():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    saved = x.sigmoid()  # sigmoid saves this very output for its own backward
    expected = saved.detach().clone()
    schema = TensorSchema(
        (_field("saved", (2, 2), torch.float32, True), _field("empty", (0,), torch.float32, True))
    )
    payload = TensorStatePayload.from_state(
        x * 2, {"saved": saved.view(2, 2), "empty": x[:0]}, schema
    )
    payload.release_output()
    payload.release_output()  # The schedule may retire the same output more than once.
    assert not payload.tensors and all(root.numel() == 1 for root in payload.backward_roots())
    torch.testing.assert_close(saved, expected)  # The caller's alias keeps its storage.
    backward_step(None, payload, (torch.ones(4), torch.ones(2, 2), torch.empty(0)), _CONFIG)
    torch.testing.assert_close(x.grad, 2 + expected * (1 - expected))
    with pytest.raises(ValueError, match="unconsumed forward payload"):
        backward_pipeline_payload(None, payload, (torch.ones(4), torch.ones(2, 2), None))


def test_terminal_payload_backward_scales_loss_once():
    hidden = torch.tensor([2.0, 3.0], requires_grad=True)
    state = torch.tensor([4.0, 5.0], requires_grad=True)
    grads = backward_step(
        _payload(hidden, state),
        (hidden * state).sum(),
        None,
        SimpleNamespace(**{**vars(_CONFIG), "grad_scale_func": lambda loss: loss * 3}),
    )
    torch.testing.assert_close(grads[0], torch.tensor([12.0, 15.0]))
    torch.testing.assert_close(grads[1], torch.tensor([6.0, 9.0]))


def test_spec_is_generic_and_fingerprints_every_qualification():
    fields = tuple(
        _field(f"feature/value:L{i}", (2, 1, 3, 1), torch.float32, i % 3 == 0, present=i % 2 == 0)
        for i in range(12)
    )
    spec = PipelinePayloadSpec(fields, (0, 1, 2, 3, 4, 5), "decoder/cut:20")
    assert [f.key for f in spec.schema.packed_fields] == [
        f"feature/value:L{i}" for i in range(0, 12, 2)
    ]
    assert spec == PipelinePayloadSpec(fields, (0, 1, 2, 3, 4, 5), "decoder/cut:20")
    assert (
        spec.fingerprint
        == PipelinePayloadSpec(fields, (0, 1, 2, 3, 4, 5), "decoder/cut:20").fingerprint
    )
    variants = [
        PipelinePayloadSpec(fields, (0, 1, 2, 3, 4, 6), "decoder/cut:20"),
        PipelinePayloadSpec(fields, (0, 1, 2, 3, 4, 5), "decoder/cut:21"),
        PipelinePayloadSpec((_field("x", (2,), torch.float32, True),) + fields[1:], spec.metadata),
    ]
    for change in ("shape", "dtype", "differentiable", "present"):
        value = {
            "shape": (2, 1, 3, 2),
            "dtype": torch.bfloat16,
            "differentiable": False,
            "present": False,
        }[change]
        replaced = TensorField(**{**vars(fields[0]), change: value})
        variants.append(
            PipelinePayloadSpec((replaced, *fields[1:]), spec.metadata, spec.boundary_id)
        )
    assert len({spec.fingerprint, *(v.fingerprint for v in variants)}) == len(variants) + 1
    for bad in [((), ()), (fields, (1.5,)), ((object(),), ())]:
        with pytest.raises((ValueError, TypeError)):
            PipelinePayloadSpec(*bad)


@pytest.mark.parametrize("activity", ["kv", "zero_hidden", "missing"])
def test_missing_and_zero_gradients_differ_for_auxiliary_losses(monkeypatch, activity):
    """Why the transport never zero-fills: a zero still runs auxiliary-loss hooks."""
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
    x = torch.ones(3, requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor(2.0))
    auxiliary = torch.nn.Parameter(torch.tensor(3.0))
    hidden = DSAIndexerLossAutoScaler.apply(x.square(), auxiliary.square())
    payload = TensorStatePayload.from_state(
        hidden, {"kv": x * weight}, TensorSchema((_field("kv", (3,), torch.float32, True),))
    )
    payload.release_output()
    grads = (
        torch.zeros(3) if activity == "zero_hidden" else None,
        None if activity == "missing" else torch.ones(3),
    )
    backward_pipeline_payload(None, payload, grads)
    assert (auxiliary.grad is not None) == (activity == "zero_hidden")
    assert (weight.grad is not None) == (activity != "missing")


def test_same_tensor_in_several_fields_accumulates_each_contribution():
    x = torch.ones(3, requires_grad=True)
    shared = x.square()
    schema = TensorSchema(
        (_field("a", (3,), torch.float32, True), _field("b", (3,), torch.float32, True))
    )
    payload = TensorStatePayload.from_state(shared, {"a": shared, "b": shared}, schema)
    payload.release_output()
    backward_pipeline_payload(
        None, payload, (torch.ones(3), torch.full((3,), 2.0), torch.full((3,), 3.0))
    )
    torch.testing.assert_close(x.grad, torch.full((3,), 12.0))
    y = torch.ones(3, requires_grad=True)
    backward_pipeline_payload(None, _payload(y, y), (torch.ones(3), torch.full((3,), 2.0)))
    torch.testing.assert_close(y.grad, torch.full((3,), 3.0))


@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_owned_release_preserves_saved_aliases_and_joint_gradients(view, empty):
    leaf = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    expected_leaf = leaf.detach().clone().requires_grad_()
    hidden, expected_hidden = leaf.sigmoid(), expected_leaf.sigmoid()
    if view:
        hidden, expected_hidden = hidden.t(), expected_hidden.t()
    if empty:
        hidden, expected_hidden = hidden[:0], expected_hidden[:0]
    schema = TensorSchema((_field("shared", hidden.shape, hidden.dtype, True),))
    payload = TensorStatePayload.from_state(hidden, {"shared": hidden}, schema)
    wire = payload.tensors[0].detach().contiguous()
    snapshot = hidden.detach().clone()
    payload.release_output()
    assert payload.backward_roots()[0].numel() == 1
    torch.testing.assert_close(hidden, snapshot)
    torch.testing.assert_close(wire, snapshot)
    backward_pipeline_payload(
        None, payload, (torch.ones_like(snapshot), torch.full_like(snapshot, 2))
    )
    (expected_hidden.sum() * 3).backward()
    torch.testing.assert_close(leaf.grad, expected_leaf.grad)


def test_owned_release_does_not_retain_an_unaliased_output_storage():
    leaf = torch.ones(4096, requires_grad=True)
    hidden = leaf * 2
    reference = weakref.ref(hidden)
    payload = TensorStatePayload.from_state(hidden, {}, TensorSchema(()))
    del hidden
    gc.collect()
    assert reference() is None
    payload.release_output()
    root = payload.backward_roots()[0]
    assert root.numel() == 1 and root.untyped_storage().nbytes() == root.element_size()
    backward_pipeline_payload(None, payload, (torch.ones_like(leaf),))
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2))


def test_release_drops_readonly_outputs_after_the_wire_owns_them():
    leaf = torch.ones(4, requires_grad=True)
    ids = torch.arange(4)
    reference = weakref.ref(ids)
    payload = TensorStatePayload.from_state(
        leaf * 2, {"ids": ids}, TensorSchema((_field("ids", (4,), torch.int64, False),))
    )
    wire = payload.tensors[1].detach()
    del ids
    payload.release_output()
    gc.collect()
    assert reference() is None and payload.backward_roots()[1] is None
    torch.testing.assert_close(wire, torch.arange(4))
    backward_pipeline_payload(None, payload, (torch.ones_like(leaf), None))
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2))


def test_released_identity_payload_still_returns_input_gradients():
    hidden = torch.ones(4, requires_grad=True)
    incoming = _payload(hidden, torch.arange(4))
    incoming.release_output()  # The stage returns exactly its received payload.
    gradients = backward_pipeline_payload(incoming, incoming, (torch.full_like(hidden, 3), None))
    torch.testing.assert_close(gradients[0], torch.full_like(hidden, 3))
    assert gradients[1] is None


def test_gradient_wire_requires_every_declared_gradient():
    spec = PipelinePayloadSpec(
        (
            _field("__hidden__", (3,), torch.float32, True),
            _field("empty", (0, 2), torch.float32, True),
            _field("ids", (3,), torch.int64, False),
        )
    )
    owner = SimpleNamespace(device=torch.device("cpu"))
    wire = TypedP2PCommunicator._gradient_wire(owner, (torch.ones(3), None, None), spec)
    assert len(wire) == 1  # Empty fields carry no data; integers carry no gradient.
    with pytest.raises(ValueError, match="received no gradient"):
        TypedP2PCommunicator._gradient_wire(owner, (None, None, None), spec)
    with pytest.raises(ValueError, match="Unexpected gradient"):
        TypedP2PCommunicator._gradient_wire(owner, (torch.ones(3), None, torch.ones(3)), spec)
    with pytest.raises(ValueError, match="incoming field slots"):
        TypedP2PCommunicator._gradient_wire(owner, (), spec)


_TRANSFER_SPEC = PipelinePayloadSpec(
    (
        _field("__hidden__", (3,), torch.float32, True, "strided"),
        _field("empty", (0,), torch.bfloat16, True),
        _field("ids", (2,), torch.int64, False),
    ),
    (3, 0),
)


@pytest.fixture
def fake_transport(monkeypatch):
    """Two-stage VPP communicator whose requests complete only when waited."""
    requests = []

    class DelayedWork:
        def __init__(self, operation, tensor):
            self.operation, self.buffer, self.wait_count = operation, weakref.ref(tensor), 0
            self.value = None

        def wait(self):
            assert self.buffer() is not None, "Communication buffer was released before wait"
            self.wait_count += 1
            if self.value is not None:
                with torch.no_grad():
                    self.buffer().fill_(self.value)

    def post(operation):
        def issue(tensor, peer, group=None):
            requests.append(DelayedWork(operation, tensor))
            return requests[-1]

        return issue

    monkeypatch.setattr(dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(dist, "isend", post("send"))
    monkeypatch.setattr(dist, "irecv", post("recv"))
    monkeypatch.setattr(TypedP2PCommunicator, "_validate_plan_peers", lambda self: None)
    config = TransformerConfig(
        num_layers=4,
        hidden_size=8,
        num_attention_heads=1,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        virtual_pipeline_model_parallel_size=2,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        deallocate_pipeline_outputs=True,
    )
    plans = [
        PipelinePayloadPlan((None,), (_TRANSFER_SPEC,)),
        PipelinePayloadPlan((_TRANSFER_SPEC,), (None,)),
    ]
    communicator = TypedP2PCommunicator(
        SimpleNamespace(size=lambda: 2, rank=lambda: 0),
        config,
        [TensorStatePayload] * 2,
        plans,
        device=torch.device("cpu"),
    )
    return communicator, requests


def test_overlap_waits_each_direction_and_retains_wire_copies(fake_transport):
    communicator, requests = fake_transport
    view = torch.arange(12.0, requires_grad=True).reshape(3, 4).t()
    expected = view.detach().clone()
    handles = communicator._transfer(
        send_next=(view, torch.empty(0)),
        send_prev=(torch.ones(2),),
        recv_prev=(torch.empty(3),),
        recv_next=(None, torch.empty(4)),
    )
    assert set(handles) == {"send_next", "send_prev", "recv_prev", "recv_next"}
    assert len(requests) == 4  # Empty and absent fields have no wire operation.
    sent = requests[0]  # Even stages post sends first.
    assert sent.operation == "send" and sent.buffer().is_contiguous()
    assert not sent.buffer().requires_grad
    del view
    gc.collect()
    torch.testing.assert_close(sent.buffer(), expected)
    handles["send_next"].wait()
    handles["send_next"].wait()  # Idempotent even if schedule cleanup waits again.
    assert sent.wait_count == 1 and sent.buffer() is None
    assert all(request.wait_count == 0 for request in requests[1:])
    del handles  # finish() must drain requests not kept by the caller.
    communicator.payload_plans = [PipelinePayloadPlan((), ())] * 2
    communicator.finish()
    assert all(request.wait_count == 1 and request.buffer() is None for request in requests)


def test_overlap_empty_directions_still_have_wait_handles(fake_transport):
    communicator, requests = fake_transport
    handles = communicator._transfer(send_next=(torch.empty(0),), recv_next=(None, torch.empty(0)))
    assert set(handles) == {"send_next", "recv_next"} and not requests
    for handle in handles.values():
        handle.wait()
    communicator.payload_plans = [PipelinePayloadPlan((), ())] * 2
    communicator.finish()


def test_batched_work_owns_every_direction_until_completion(fake_transport, monkeypatch):
    """NCCL may return a single Work for a batch containing several field operations."""
    communicator, _ = fake_transport
    communicator.config.batch_p2p_comm = True
    send, receive = torch.ones(3), torch.empty(3)
    references = []

    class BatchWork:
        done = False

        def wait(self):
            if not self.done:
                assert all(ref() is not None for ref in references)
                self.done = True

    request = BatchWork()
    monkeypatch.setattr(dist, "P2POp", lambda *args: args)
    monkeypatch.setattr(dist, "batch_isend_irecv", lambda ops: [request])
    handles = communicator._transfer(send_next=[send], recv_next=[receive])
    references.extend(weakref.ref(handles[name].buffers[0]) for name in ("send_next", "recv_next"))
    del send, receive
    # Waiting one direction must wait the common Work, not drop the other's buffer early.
    handles["recv_next"].wait()
    assert request.done and references[0]() is not None and references[1]() is None
    handles["send_next"].wait()
    assert all(ref() is None for ref in references)
    communicator.payload_plans = [PipelinePayloadPlan((), ())] * 2
    communicator.finish()


@pytest.mark.parametrize("release", [False, True])
def test_planned_exchange_posts_only_tensor_data(fake_transport, release):
    communicator, requests = fake_transport
    communicator.config.deallocate_pipeline_outputs = release
    hidden = torch.ones(3, requires_grad=True) * 2
    source = TensorStatePayload.from_state(
        hidden,
        {"empty": torch.empty(0, dtype=torch.bfloat16), "ids": torch.arange(2)},
        TensorSchema(_TRANSFER_SPEC.fields[1:]),
        boundary_id=_TRANSFER_SPEC.boundary_id,
        metadata=_TRANSFER_SPEC.metadata,
    )
    received, handles = communicator.send_forward_recv_forward(
        source, True, None, overlap_p2p_comm=True, send_chunk_id=0, recv_chunk_id=1
    )
    assert bool(source.tensors) is not release
    assert received.spec == _TRANSFER_SPEC and received.tensors[0].requires_grad
    assert len(requests) == 4  # hidden and ids each way; the empty field has no data.
    assert all(request.wait_count == 0 for request in requests)
    for request in requests:
        request.value = 7 if request.operation == "recv" else None
    handles["recv_prev"].wait()
    torch.testing.assert_close(received.tensors[0], torch.full((3,), 7.0))
    gradients = backward_pipeline_payload(received, received.tensors[0].sum(), None)
    _, handles = communicator.send_backward_recv_backward(
        gradients, True, None, overlap_p2p_comm=True, send_chunk_id=1, recv_chunk_id=0
    )
    handles["recv_next"].wait()
    communicator.finish()
    assert all(request.wait_count == 1 for request in requests)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "grad", "metadata"])
@pytest.mark.parametrize("forward_only", [False, True])
def test_planned_sender_rejects_stale_plan_before_posting(fake_transport, mismatch, forward_only):
    communicator, requests = fake_transport
    communicator.forward_only = forward_only
    field = _TRANSFER_SPEC.fields[0]
    replaced = TensorField(
        field.key,
        (5,) if mismatch == "shape" else field.shape,
        torch.bfloat16 if mismatch == "dtype" else field.dtype,
        field.layout,
        mismatch != "grad",
    )
    stale = PipelinePayloadSpec(
        (replaced, *_TRANSFER_SPEC.fields[1:]),
        (5, 0) if mismatch == "metadata" else _TRANSFER_SPEC.metadata,
    )
    communicator.payload_plans = [PipelinePayloadPlan((None,), (stale,))] * 2
    source = TensorStatePayload(
        (torch.ones(3), torch.empty(0, dtype=torch.bfloat16), torch.arange(2)), _TRANSFER_SPEC
    )
    with pytest.raises(ValueError, match="disagrees"):
        communicator.send_forward_recv_forward(source, False, None, overlap_p2p_comm=True)
    assert not requests


def test_bad_gradient_length_is_rejected_without_consuming_forward(fake_transport):
    communicator, requests = fake_transport
    communicator._received[0].append(_TRANSFER_SPEC)
    with pytest.raises(ValueError, match="match the incoming field slots"):
        communicator.send_backward((), is_first_stage=False)
    assert communicator._received[0][0] is _TRANSFER_SPEC and not requests


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


def _boundary_spec(x, microbatch, train):
    """Per-microbatch plan: shapes vary, and 'aux' is differentiable on even microbatches."""
    rows = x.shape[0]
    return PipelinePayloadSpec(
        (
            _field("__hidden__", x.shape, torch.float32, True, "strided"),
            _field("shared", (rows // 2, 2), torch.bfloat16, train),
            _field("aux", (rows,), torch.float32, train and microbatch % 2 == 0),
            _field("indices", (rows, 2), torch.int32, False),
            _field("prefixes", (rows % 3 + 2,), torch.int64, False),
            _field("absent", (rows,), torch.float32, False, present=False),
        ),
        (rows, rows // 2),
        "test",
    )


class _Stage(torch.nn.Module):
    pipeline_payload_factory = TensorStatePayload

    def __init__(self, config, rank, size, device):
        super().__init__()
        self.config, self.rank, self.size = config, rank, size
        self.model_type = ModelType.encoder_or_decoder
        self.weight = torch.nn.Parameter(torch.tensor((rank + 1) / 2, device=device))
        self.input_tensor, self.calls, self.references = None, 0, []

    def set_input_tensor(self, inputs):
        self.input_tensor = inputs[0] if isinstance(inputs, list) else inputs

    def forward(self, x):
        incoming, self.input_tensor = self.input_tensor, None
        spec = _boundary_spec(x, self.calls, torch.is_grad_enabled())
        self.calls += 1
        if self.rank == 0:
            hidden = x * self.weight
            rows = x.shape[0] // 2
            state = {
                # A non-contiguous differentiable view, including zero capacity.
                "shared": (self.weight * torch.ones(2, rows, device=x.device)).bfloat16().t(),
                "aux": self.weight * torch.ones(x.shape[0], device=x.device),
                "indices": torch.zeros(x.shape[0], 2, device=x.device, dtype=torch.int32),
                "prefixes": torch.arange(x.shape[0] % 3 + 2, device=x.device),
            }
        else:
            self.references.append(weakref.ref(incoming))
            hidden, state = incoming.restore()
            hidden = hidden * self.weight + state["shared"].float().sum() * self.weight
            if incoming.spec.fields[2].differentiable:
                hidden = hidden + state["aux"].sum() * self.weight
        if self.rank == self.size - 1:
            return hidden.sum()
        payload = TensorStatePayload.from_state(
            hidden,
            state,
            TensorSchema(spec.fields[1:]),
            boundary_id=spec.boundary_id,
            metadata=spec.metadata,
        )
        self.references.append(weakref.ref(payload))
        return payload


@pytest.fixture(scope="module")
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
    groups.pp = dist.new_group(list(range(size)))
    groups.tp = groups.cp = singleton_groups[rank]
    # Initialize the full PP communicator before mixing batched and unbatched P2P.
    # Lazy pairwise NCCL communicators can hang when this fixture shuts them down.
    dist.barrier(group=groups.pp)
    yield groups, device
    dist.barrier()
    dist.destroy_process_group(groups.pp)
    dist.destroy_process_group(singleton_groups[rank])


@pytest.mark.parametrize("interleaved", [False, True])
def test_typed_adapter_rejects_custom_communicator(transport_groups, interleaved):
    """Model opt-in must not silently discard a caller's transport overrides."""
    groups, device = transport_groups
    config = TransformerConfig(
        num_layers=groups.pp.size() * 2,
        hidden_size=8,
        num_attention_heads=1,
        pipeline_model_parallel_size=groups.pp.size(),
        pipeline_dtype=torch.float32,
        virtual_pipeline_model_parallel_size=2 if interleaved else None,
        overlap_p2p_comm=False,
        batch_p2p_comm=False,
    )

    class CustomCommunicator(P2PCommunicator):
        pass

    stages = [_Stage(config, i, 2, device) for i in range(2)]
    schedule = (
        forward_backward_pipelining_with_interleaving
        if interleaved
        else forward_backward_pipelining_without_interleaving
    )
    with pytest.raises(ValueError, match="custom P2P communicator"):
        schedule(
            forward_step_func=lambda *args: pytest.fail("must fail before forward"),
            data_iterator=[iter(()) for _ in stages] if interleaved else iter(()),
            model=stages if interleaved else stages[0],
            num_microbatches=2,
            seq_length=8,
            micro_batch_size=1,
            p2p_communicator=CustomCommunicator(groups.pp, config),
            pg_collection=groups,
        )


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
def test_typed_1f1b_actual_transport(
    transport_groups, batch_p2p, overlap, warmup_flush, vp_size, count, forward_only, group_multiple
):
    """Ordinary/VPP P2P, rank wraparound, per-microbatch shapes/eligibility and release."""
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
    if warmup_flush:
        with pytest.raises(ValueError, match="warmup/flush prefetch"):
            TypedP2PCommunicator(groups.pp, config, [TensorStatePayload] * vp_size, [])
        return
    models = [_Stage(config, rank + vp * size, size * vp_size, device) for vp in range(vp_size)]
    reference = [_Stage(config, i, size * vp_size, device) for i in range(size * vp_size)]
    batches = [
        torch.arange((1, 4, 7)[i % 3] * 3, device=device).float().reshape(-1, 1, 3) / 8
        for i in range(count)
    ]
    for _ in range(2):
        expected, actual = [], []
        for stage in (*models, *reference):
            stage.zero_grad(set_to_none=True)
            stage.calls = 0
        with torch.set_grad_enabled(not forward_only):
            for x in batches:
                value = None
                for stage in reference:
                    stage.set_input_tensor(value)
                    value = stage(x)
                expected.append(value.detach().clone())
                if not forward_only:
                    (value / count).backward()
            for stage in models:
                stage.calls = 0

            def forward_step(iterator, stage):
                output = stage(next(iterator))

                def loss_func(loss):
                    actual.append(loss.detach().clone())
                    return loss, {"loss": loss.detach().clone()}

                return output, loss_func

            descriptors = tuple(
                _boundary_spec(x, i, not forward_only) for i, x in enumerate(batches)
            )
            iterators = [
                PipelineDataIterator(
                    batches,
                    PipelinePayloadPlan(
                        descriptors if stage.rank else (None,) * count,
                        descriptors if stage.rank < stage.size - 1 else (None,) * count,
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


def test_schedule_requires_prepared_plans(transport_groups):
    groups, device = transport_groups
    config = TransformerConfig(
        num_layers=groups.pp.size(),
        hidden_size=8,
        num_attention_heads=1,
        pipeline_model_parallel_size=groups.pp.size(),
        pipeline_dtype=torch.float32,
        batch_p2p_comm=False,
    )
    with pytest.raises(ValueError, match="PipelinePayloadPlan"):
        forward_backward_pipelining_without_interleaving(
            forward_step_func=lambda *args: pytest.fail("must fail before forward"),
            data_iterator=iter(()),
            model=_Stage(config, groups.pp.rank(), groups.pp.size(), device),
            num_microbatches=2,
            seq_length=8,
            micro_batch_size=1,
            p2p_communicator=P2PCommunicator(groups.pp, config),
            pg_collection=groups,
        )
