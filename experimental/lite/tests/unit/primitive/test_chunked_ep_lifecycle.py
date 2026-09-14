# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Event ordering, alias safety, capacity, and explicit release contracts."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core  # noqa: F401


class Event:
    def query(self):
        return False


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
    from megatron.lite.model.qwen3_moe.lite.head_loss import validate_chunked_ep_mtp

    if mtp:
        with pytest.raises(ValueError, match="MTP"):
            validate_chunked_ep_mtp(enable_ep_chunk_overlap=True, mtp_enable=True)
    else:
        validate_chunked_ep_mtp(enable_ep_chunk_overlap=True, mtp_enable=False)
