# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Event ordering, alias safety, capacity, and explicit release contracts."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core  # noqa: F401


@pytest.fixture
def ep(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap

    return moe_ep_chunk_overlap


@pytest.mark.parametrize("caller_owned", [False, True])
@pytest.mark.parametrize("rows", [0, 2])
def test_chunked_transport_owns_external_metadata_and_waits_before_finish(
    ep, monkeypatch, caller_owned, rows
):
    from megatron.lite.primitive.modules import chunked_ep_dispatcher as transport
    from megatron.lite.primitive.modules import dispatcher as shared_transport

    buffer = Mock()
    monkeypatch.setattr(shared_transport, "deep_ep", object())
    monkeypatch.setattr(shared_transport, "_build_deepep_buffer", lambda *args: buffer)
    ps = SimpleNamespace(ep_size=2, tp_ep_group=object())
    dispatcher = transport.ChunkedDispatcher(4, 3, ps)
    assert dispatcher.buffer is buffer
    event = SimpleNamespace(event=object(), current_stream_wait=Mock())
    hidden = torch.arange(rows * 3.0).reshape(rows, 3).requires_grad_()
    state = {
        "recv_hidden": hidden,
        "recv_indices": torch.tensor([[1], [0]])[:rows],
        "recv_probs": torch.tensor([[0.2], [0.8]])[:rows],
        "recv_per_expert": [rows // 2, rows // 2],
        "handle": object(),
        "event": event,
    }
    target = torch.empty_like(hidden)
    allocate = Mock(return_value=target) if caller_owned else None
    output, counts, probs, metadata = dispatcher.finish_deepep_dispatch_for_backward(
        state, output_allocation=allocate
    )
    if caller_owned:
        allocate.assert_called_once_with("fc1_input", (rows, 3))
        assert output is target and not output.requires_grad
    else:
        torch.testing.assert_close(
            torch.autograd.grad(output.sum(), hidden)[0], torch.ones_like(hidden)
        )
    event.current_stream_wait.assert_called_once()
    torch.testing.assert_close(output, hidden.flip(0))
    torch.testing.assert_close(probs, state["recv_probs"].flatten().flip(0))
    assert counts is None and metadata["local_tpe_list"] == state["recv_per_expert"]
    assert metadata["handle"] is state["handle"]
    assert dispatcher._handle is None
    normal, counts, normal_probs = dispatcher.finish_deepep_dispatch(state)
    torch.testing.assert_close(normal, output)
    torch.testing.assert_close(normal_probs, probs)
    assert counts is None and dispatcher._handle is state["handle"]
    completion = {"combined": hidden, "event": event}
    assert dispatcher.finish_deepep_combine(completion) is hidden
    assert completion == {}
    assert event.current_stream_wait.call_count == 3
    for returned_event in (None, SimpleNamespace(event=None, current_stream_wait=Mock()), event):
        buffer.dispatch.return_value = (hidden, None, None, None, None, returned_event)
        buffer.combine.return_value = (hidden, None, returned_event)
        buffer.get_dispatch_layout.return_value = (None, None, None, None, event)
        for submit, args in (
            (dispatcher.submit_deepep_combine_prepared, (hidden, state["handle"])),
            (
                dispatcher.submit_deepep_dispatch,
                (hidden, state["recv_probs"], state["recv_indices"]),
            ),
            (dispatcher.submit_deepep_combine_backward, (hidden, state["handle"])),
            (dispatcher.submit_deepep_dispatch_backward, (hidden, None, state["handle"])),
        ):
            if returned_event is event:
                assert submit(*args)["event"] is event
            else:
                with pytest.raises(RuntimeError, match="requires a completion event"):
                    submit(*args)
    if caller_owned:
        with pytest.raises(RuntimeError, match="Invalid caller-owned"):
            dispatcher.finish_deepep_dispatch_for_backward(
                state, output_allocation=lambda *args: torch.empty(1)
            )


def test_qwen_release_visits_only_chunked_modules_once(ep, monkeypatch):
    from megatron.lite.model.qwen3_moe.lite.chunked_ep import release_chunked_ep

    execution = Mock()
    monkeypatch.setattr(ep, "EPChunkExecution", lambda **kwargs: execution)
    moe = ep.ChunkedMoE(router=torch.nn.Linear(4, 2), experts=torch.nn.Linear(4, 4))
    native = torch.nn.Linear(4, 4)
    native.chunked_ep = Mock()  # A coincidental attribute must not trigger cleanup.
    parent = torch.nn.ModuleList([moe, native])
    release_chunked_ep([parent, moe, parent])
    execution.release.assert_called_once_with(stream=None)
    native.chunked_ep.release.assert_not_called()


@pytest.mark.parametrize("with_probs", [False, True])
def test_shared_expert_backward_preserves_gradients_and_input_storage(ep, with_probs):
    x = torch.arange(6.0).reshape(3, 2).requires_grad_()
    probs = torch.full_like(x, 0.5, requires_grad=True) if with_probs else None
    output = x.square() if probs is None else x.square() * probs
    chunk = SimpleNamespace(dispatched=x, probs=probs, expert_out=output, expert_out_edge=None)
    dx, dp, storage = ep._backward_expert(
        chunk, torch.ones_like(x), SimpleNamespace(check_active=lambda: None)
    )
    torch.testing.assert_close(dx, 2 * x if probs is None else 2 * x * probs)
    if probs is None:
        assert dp is None
    else:
        torch.testing.assert_close(dp, x.square())
    assert storage.data_ptr() == x.data_ptr() and not storage.requires_grad
    assert chunk.dispatched is None and chunk.probs is None and chunk.expert_out is None


@pytest.mark.parametrize("edge", [False, True])
@pytest.mark.parametrize("missing_scores", [False, True])
def test_shared_router_backward(ep, edge, missing_scores):
    x = torch.ones(2, requires_grad=True)
    weight = torch.tensor(3.0, requires_grad=True)
    scores = x * weight
    chunk = SimpleNamespace(
        x=x,
        scores=None if edge else scores,
        scores_edge=torch.autograd.graph.get_gradient_edge(scores) if edge else None,
        scores_shape=scores.shape,
        scores_dtype=scores.dtype,
    )
    accum = [torch.tensor(5.0)]
    dx = ep._backward_router(
        chunk,
        torch.ones_like(x),
        None if missing_scores else torch.ones_like(scores),
        (weight,),
        accum,
    )
    torch.testing.assert_close(dx, torch.full_like(x, 1.0 if missing_scores else 4.0))
    torch.testing.assert_close(accum[0], torch.tensor(5.0 if missing_scores else 7.0))


@pytest.mark.parametrize("retain_output", [False, True])
def test_context_capture_keeps_output_only_when_requested(ep, retain_output):
    x = torch.ones(2, 2, requires_grad=True)
    scores, output = x.sigmoid(), x.square()
    state = dict(handle=object(), recv_hidden=x, recv_probs=scores)
    metadata = dict(manual_row_id_map=torch.arange(2), manual_prob_flat_indices=torch.arange(2))
    chunk = ep._BackwardChunk.from_dispatch(
        state,
        metadata,
        scores,
        output,
        retain_output=retain_output,
        idx=0,
        start=0,
        end=2,
        x=x,
        dispatched=x,
        probs=scores,
        dispatcher=None,
        workspace_lease=None,
    )
    assert (chunk.expert_out is output) == retain_output
    assert chunk.scores is None and chunk.recv_probs_base is scores
    (dx,) = torch.autograd.grad(chunk.expert_out_edge, x, torch.ones_like(output))
    torch.testing.assert_close(dx, 2 * x)


def test_stream_cache_separates_roles_and_devices(ep, monkeypatch):
    monkeypatch.setattr(ep, "_EP_CHUNK_STREAMS", {})
    monkeypatch.setattr(ep, "_make_stream", lambda device: object())
    comm = ep._shared_stream(0, "comm")
    assert comm is ep._shared_stream(torch.device("cuda:0"), "comm")
    assert comm is not ep._shared_stream(0, "wgrad")
    assert comm is not ep._shared_stream(1, "comm")


class Event:
    def query(self):
        return False


@pytest.mark.parametrize("chunk_count", [2, 3, 4])
def test_saved_backward_retires_two_slots_before_reuse(ep, monkeypatch, chunk_count):
    """Exercise the real backward schedule and workspace, with CPU compute doubles."""
    stream, event = Mock(), Mock(query=lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(ep, "_shared_stream", lambda *args: stream)
    monkeypatch.setattr(ep, "_queue_backward_stream_wait", lambda *args: None)
    monkeypatch.setattr(ep._EPChunkOperationBase, "_streams", lambda *args: (stream, stream))
    x = torch.ones(1, 2)
    dispatcher = Mock()
    dispatcher.submit_deepep_combine_backward.return_value = {"event": event}
    dispatcher.submit_deepep_dispatch_backward.return_value = {"event": event}
    dispatcher.finish_deepep_dispatch_backward.return_value = (x, None)
    monkeypatch.setattr(ep, "_manual_unpermute_backward", lambda *args: x)
    monkeypatch.setattr(ep, "_backward_expert", lambda *args: (x, None, x))
    monkeypatch.setattr(ep, "_dispatch_local_backward", lambda *args, **kwargs: (x, x))
    monkeypatch.setattr(ep, "_backward_router", lambda chunk, *args: x * (chunk.idx + 1))
    profile = ep.EPChunkShapeProfile(8, 2, 1, 2, chunk_count=chunk_count)
    workspace = ep.EPChunkWorkspace(
        ep.EPChunkWorkspaceKey("backward", "cpu", None, 0, torch.float32, profile),
        lambda slot: dispatcher,
    )
    # CPU streams have no CUDA device; the real scratch workspace stays on CPU.
    stream.device = None
    chunks = []
    for idx in range(chunk_count):
        values = {item.name: None for item in ep.fields(ep._ChunkContext)}
        values.update(idx=idx, start=idx, end=idx + 1, x=x, dispatcher=dispatcher)
        chunks.append(ep._ForwardChunkContext(**values, recv_consumed_event=event))
    experts = Mock(parameters=lambda: ())
    op = ep.EPChunkBackwardOp(router=torch.nn.Identity(), experts=experts, workspace=workspace)
    grad, _, _ = op._saved_context_backward(
        ep._SavedForwardContext(chunks, torch.Size((chunk_count, 2))), torch.ones(chunk_count, 2)
    )
    torch.testing.assert_close(grad, torch.arange(1.0, chunk_count + 1)[:, None].expand(-1, 2))
    assert [
        call.kwargs["num_contexts"] for call in experts.flush_delayed_weight_grads.call_args_list
    ] == [min(2, remaining) for remaining in range(chunk_count, 0, -2)]
    assert not any(slot.in_use for slot in workspace._slots)


@pytest.mark.parametrize("retain_backward", [False, True])
def test_execution_owns_lifecycle_without_registering_parameters(ep, monkeypatch, retain_backward):
    spaces = {}

    def acquire(key, factory):
        return spaces.setdefault(key.op, Mock(key=key))

    monkeypatch.setattr(ep, "get_ep_chunk_workspace", acquire)
    release = Mock()
    monkeypatch.setattr(ep, "release_ep_chunk_workspace", release)
    router, experts = torch.nn.Linear(4, 2), torch.nn.Linear(4, 4)
    model = torch.nn.Module()
    model.router, model.experts = router, experts
    keys = tuple(model.state_dict())
    execution = ep.EPChunkExecution(
        router=router,
        experts=experts,
        dispatcher_factory=Mock(),
        max_input_rows=16,
        hidden_size=4,
        expert_intermediate_size=3,
        topk=2,
        ep_size=2,
        ep_group=object(),
        retain_backward=retain_backward,
    )
    model.chunked_ep = execution
    assert tuple(model.state_dict()) == keys
    assert execution.forward_op.router is router
    assert (execution.backward_op is not None) == retain_backward
    assert (execution.fused_op is not None) != retain_backward
    backward = spaces["backward" if retain_backward else "fused_forward_backward"]
    execution.materialize(phase="backward", expert_activation_max_rows=4)
    if retain_backward:
        backward.prepare_scratch.assert_called_once_with(device=None)
        backward.materialize.assert_not_called()
    else:
        backward.materialize.assert_called_once_with(device=None)
    backward.reserve_expert_activations.assert_called_once_with(max_expert_rows=4, device=None)
    execution.finish_forward(torch.zeros(1))
    spaces["forward"].reset_tensors.assert_called_once_with(stream=None)
    execution.finish_backward(torch.zeros(1))
    backward.park_expert_activations.assert_called_once_with(stream=None)
    execution.release()
    assert [call.args[0] for call in release.call_args_list] == [
        spaces["forward"].key,
        backward.key,
    ]


class Stream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


@pytest.fixture
def workspaces(ep, monkeypatch):
    monkeypatch.setattr(ep, "_EXPERT_ACTIVATION_SIZE_CLASS_BYTES", 1)
    registry = ep.EPChunkWorkspaceRegistry()
    profile = ep.EPChunkShapeProfile(16, 4, 2, 2, expert_intermediate_size=3)
    result = [
        registry.get_or_create(
            ep.EPChunkWorkspaceKey(op, "cpu", None, 1, torch.float32, profile),
            lambda slot: SimpleNamespace(use_deepep=True),
        )
        for op in ("forward", "backward", "fused_forward_backward")
    ]
    return registry, result


def test_cross_op_reuse_waits_for_consumer_and_rejects_live_owner(workspaces):
    _, (forward, backward, fused) = workspaces
    stream = Stream()
    first = forward.acquire_expert_activation(stream=stream)
    x = first.tensor("fc1_input", (3, 4), dtype=torch.float32, device="cpu")
    out = first.tensor("fc2_output", (3, 4), dtype=torch.float32, device="cpu")
    assert x.data_ptr() == out.data_ptr()
    with pytest.raises(RuntimeError):
        fused.acquire_expert_activation(stream=stream)
    event = Event()
    first.release(event)
    next_lease = fused.acquire_expert_activation(stream=stream)
    assert event in stream.waited
    assert (
        next_lease.tensor("fc1_input", (3, 4), dtype=torch.float32, device="cpu").data_ptr()
        == x.data_ptr()
    )
    out = next_lease.tensor("fc2_output", (3, 4), dtype=torch.float32, device="cpu")
    dgrad = next_lease.tensor("fc1_dgrad", (3, 4), dtype=torch.float32, device="cpu")
    fc2_dgrad = next_lease.tensor("fc2_dgrad", (3, 3), dtype=torch.float32, device="cpu")
    assert out.data_ptr() != dgrad.data_ptr() == fc2_dgrad.data_ptr()
    out.fill_(1)
    fc2_dgrad.fill_(2)
    dgrad.fill_(3)
    torch.testing.assert_close(out, torch.ones_like(out))
    next_lease.release(Event())
    normal = backward.acquire_expert_activation(stream=stream)
    normal_out = normal.tensor("fc2_output", (3, 4), dtype=torch.float32, device="cpu")
    normal_grad = normal.tensor("fc2_dgrad", (3, 3), dtype=torch.float32, device="cpu")
    assert normal_out.data_ptr() != normal_grad.data_ptr()
    normal.release(Event())


def test_lazy_growth_reuses_capacity_across_ops(workspaces):
    _, (forward, _, fused) = workspaces
    stream = Stream()
    arena = forward._activation_arena
    pointers = []
    capacities = []
    for workspace, rows in ((forward, 2), (fused, 2), (forward, 4), (fused, 3)):
        lease = workspace.acquire_expert_activation(stream=stream)
        tensor = lease.tensor("fc1_input", (rows, 4), dtype=torch.float32, device="cpu")
        assert tensor.shape == (rows, 4)
        pointers.append(tensor.data_ptr())
        capacities.append(dict(arena.capacity_bytes))
        lease.release(Event())
    assert pointers[0] == pointers[1]
    assert pointers[2] == pointers[3]
    assert capacities[0] == capacities[1]
    assert capacities[2] == capacities[3]
    assert sum(capacities[2].values()) > sum(capacities[0].values())
    assert not arena.frozen


def test_reserve_park_release_and_rematerialize(workspaces):
    registry, (forward, _, fused) = workspaces
    stream = Stream()
    forward.reserve_expert_activations(max_expert_rows=4)
    arena = forward._activation_arena
    capacity = dict(arena.capacity_bytes)
    first = forward.acquire_expert_activation(stream=stream)
    first.tensor("fc1_input", (4, 4), dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError, match="Frozen"):
        first.tensor("fc1_input", (5, 4), dtype=torch.float32, device="cpu")
    first.release(Event())
    forward.reset_tensors(stream=stream)
    assert not arena.tensors and not arena.backing_tensors
    assert arena.capacity_bytes == capacity
    restored = fused.acquire_expert_activation(stream=stream)
    restored.tensor("fc1_input", (4, 4), dtype=torch.float32, device="cpu")
    restored.release(Event())
    for workspace in workspaces[1]:
        registry.release(workspace.key, stream=stream)
    assert not arena.capacity_bytes and not arena.tensors
    forward.materialize()
    assert forward.dispatcher(0) is not forward.dispatcher(1)


@pytest.mark.parametrize("replacement_index", [0, 2])
def test_released_workspace_rejects_replaced_identity(workspaces, replacement_index):
    registry, spaces = workspaces
    for workspace in spaces:
        registry.release(workspace.key)
    registry.get_or_create(spaces[replacement_index].key, lambda slot: Mock(use_deepep=True))
    with pytest.raises(RuntimeError, match="key was reused|replaced activation arena"):
        spaces[0].materialize()


@pytest.mark.parametrize("mtp", [False, True])
def test_mtp_composition_is_explicit(mtp):
    from megatron.lite.model.qwen3_moe.lite.chunked_ep import validate_chunked_ep_mtp

    if mtp:
        with pytest.raises(ValueError, match="MTP"):
            validate_chunked_ep_mtp(enable_ep_chunk_overlap=True, mtp_enable=True)
    else:
        validate_chunked_ep_mtp(enable_ep_chunk_overlap=True, mtp_enable=False)


@pytest.mark.parametrize("te_base", [SimpleNamespace(), SimpleNamespace(get_workspace=object)])
def test_production_router_wgrad_does_not_round_each_chunk(ep, monkeypatch, te_base):
    from megatron.lite.primitive.utils import moe

    calls = []

    def gemm(a, b, out_dtype, *, layout, out=None, accumulate=False, **kwargs):
        assert ("workspace" in kwargs) == hasattr(te_base, "get_workspace")
        value = {
            "TN": lambda: b.float() @ a.float().T,
            "NN": lambda: b.float() @ a.float(),
            "NT": lambda: b.float().T @ a.float(),
        }[layout]()
        value = value.to(out_dtype)
        if out is not None:
            calls.append((out_dtype, accumulate, out.data_ptr()))
            out.add_(value) if accumulate else out.copy_(value)
            value = out
        return (value,)

    monkeypatch.setattr(moe, "general_gemm", gemm)
    monkeypatch.setattr(moe, "te_module_base", te_base)
    torch.manual_seed(31)
    weight = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    x = torch.randn(12, 8, dtype=torch.bfloat16)
    grad = torch.randn(12, 3, dtype=torch.bfloat16)
    accum = [None]
    for xx, gg in zip(x.chunk(2), grad.chunk(2)):
        xx = xx.detach().requires_grad_(True)
        scores = moe.router_gating_linear(xx, weight, None, torch.bfloat16)
        chunk = SimpleNamespace(x=xx, scores=scores, scores_edge=None, scores_dtype=scores.dtype)
        dx = ep._backward_router(chunk, torch.zeros_like(xx), gg, (weight,), accum)
        assert torch.equal(dx, (gg.float() @ weight.detach().float()).bfloat16())
        assert not hasattr(weight, "_wgrad_accumulator")
    expected = (grad.float().T @ x.float()).bfloat16()
    assert torch.equal(ep._materialize((weight,), accum)[0], expected)
    assert len(calls) == 2 and all(c[0] == torch.float32 and c[1] for c in calls)
    assert calls[0][2] == calls[1][2]
    native = moe.router_gating_linear(x, weight, None, torch.bfloat16)
    assert torch.equal(torch.autograd.grad(native, weight, grad)[0], expected)
    assert weight.grad is None and len(calls) == 2


def test_double_router_is_not_downcast(ep):
    from megatron.lite.primitive.utils.moe import router_gating_linear

    weight = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[1.00000000001]], dtype=torch.float64, requires_grad=True)
    scores = router_gating_linear(x, weight, None, torch.float64)
    chunk = SimpleNamespace(x=x, scores=scores, scores_edge=None, scores_dtype=scores.dtype)
    accum = [None]
    ep._backward_router(chunk, torch.zeros_like(x), torch.ones_like(scores), (weight,), accum)
    assert torch.equal(ep._materialize((weight,), accum)[0], x.detach())


def test_router_accumulator_binding_cleans_up_on_error(ep, monkeypatch):
    weight = torch.ones(2, 2, requires_grad=True)
    x = torch.ones(2, 2, requires_grad=True)
    chunk = SimpleNamespace(x=x, scores=x, scores_edge=None, scores_dtype=x.dtype)

    def fail(*args, **kwargs):
        assert weight._wgrad_accumulator.dtype == torch.float32
        raise RuntimeError("injected backward failure")

    monkeypatch.setattr(torch.autograd, "grad", fail)
    with pytest.raises(RuntimeError, match="injected"):
        ep._backward_router(chunk, x, x, (weight,), [None])
    assert not hasattr(weight, "_wgrad_accumulator")


def test_router_accumulator_rejects_overlapping_owner(ep):
    weight = torch.ones(2, 2, requires_grad=True)
    owner = torch.zeros_like(weight)
    weight._wgrad_accumulator = owner
    x = torch.ones(2, 2, requires_grad=True)
    chunk = SimpleNamespace(x=x, scores=x, scores_edge=None, scores_dtype=x.dtype)
    with pytest.raises(RuntimeError, match="already leased"):
        ep._backward_router(chunk, x, x, (weight,), [None])
    assert weight._wgrad_accumulator is owner
    del weight._wgrad_accumulator
