# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Event ordering, alias safety, capacity, and explicit release contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core  # noqa: F401


def test_chunked_transport_owns_external_metadata_and_waits_before_finish(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import chunked_ep_dispatcher as transport

    buffer = Mock()
    monkeypatch.setattr(transport, "deep_ep", object())
    monkeypatch.setattr(
        transport,
        "_build_deepep_buffer",
        lambda *args: transport._DeepEPBufferAllocation(buffer, 8, 16),
    )
    ps = SimpleNamespace(ep_size=2, tp_ep_group=object())
    dispatcher = transport.ChunkedDispatcher(4, 3, ps)
    assert dispatcher.deepep_buffer_resident_bytes == 24
    event = SimpleNamespace(event=object(), current_stream_wait=Mock())
    hidden = torch.arange(6.0).reshape(2, 3)
    state = {
        "recv_hidden": hidden,
        "recv_indices": torch.tensor([[1], [0]]),
        "recv_probs": torch.tensor([[0.2], [0.8]]),
        "recv_per_expert": [1, 1],
        "handle": object(),
        "event": event,
    }
    output, counts, probs, metadata = dispatcher.finish_deepep_dispatch_external_with_options(
        state, force_manual_map=True, force_direct_permute=True, materialize_local_tpe=False
    )
    event.current_stream_wait.assert_called_once()
    torch.testing.assert_close(output, hidden.flip(0))
    torch.testing.assert_close(probs, state["recv_probs"].flatten().flip(0))
    assert counts is None and metadata["local_tpe_list"] == [1, 1]
    assert metadata["handle"] is state["handle"]
    assert dispatcher._handle is None
    completion = {"combined": hidden, "event": event}
    assert dispatcher.finish_deepep_combine(completion) is hidden
    assert completion == {}
    assert event.current_stream_wait.call_count == 2


class Event:
    def query(self):
        return False


@pytest.mark.parametrize("retain_backward", [False, True])
def test_execution_owns_lifecycle_without_registering_parameters(
    transformer_engine_import_stub, monkeypatch, retain_backward
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as ep

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
def workspaces(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as ep

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
    assert out.data_ptr() == dgrad.data_ptr()
    next_lease.release(Event())
    normal = backward.acquire_expert_activation(stream=stream)
    normal_out = normal.tensor("fc2_output", (3, 4), dtype=torch.float32, device="cpu")
    normal_grad = normal.tensor("fc2_dgrad", (3, 4), dtype=torch.float32, device="cpu")
    assert normal_out.data_ptr() != normal_grad.data_ptr()
    normal.release(Event())


def test_reserve_park_release_and_rematerialize(workspaces):
    registry, (forward, _, fused) = workspaces
    stream = Stream()
    forward.reserve_expert_activations(max_expert_rows=4)
    arena = forward._expert_activation_owner.coordinator
    capacity = dict(arena.capacity_bytes)
    first = forward.acquire_expert_activation(stream=stream)
    first.tensor("fc1_input", (4, 4), dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError, match="Frozen"):
        first.tensor("fc1_input", (5, 4), dtype=torch.float32, device="cpu")
    first.release(Event())
    forward.reset_tensors(stream=stream)
    assert not arena.arena.tensors and not arena.backing_tensors
    assert arena.capacity_bytes == capacity
    restored = fused.acquire_expert_activation(stream=stream)
    restored.tensor("fc1_input", (4, 4), dtype=torch.float32, device="cpu")
    restored.release(Event())
    for workspace in workspaces[1]:
        registry.release(workspace.key, stream=stream)
    assert not arena.capacity_bytes and not arena.arena.tensors
    forward.materialize()
    assert forward.dispatcher(0) is not forward.dispatcher(1)


@pytest.mark.parametrize("mtp", [False, True])
def test_mtp_composition_is_explicit(mtp):
    from megatron.lite.model.qwen3_moe.lite.chunked_ep import validate_chunked_ep_mtp

    if mtp:
        with pytest.raises(ValueError, match="MTP"):
            validate_chunked_ep_mtp(enable_ep_chunk_overlap=True, mtp_enable=True)
    else:
        validate_chunked_ep_mtp(enable_ep_chunk_overlap=True, mtp_enable=False)
