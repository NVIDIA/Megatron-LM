# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared state parity across eager, recompute and real CUDA graph execution."""

import os
from copy import deepcopy

import pytest
import torch
from torch import nn

from megatron.core.pipeline_parallel.pipeline_payload import TensorStatePayload
from megatron.core.tensor_parallel import random as rng
from megatron.core.transformer.state_boundary import TensorField, TensorSchema
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule


@pytest.fixture(autouse=True)
def local_device():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _fields(dtype=torch.float32):
    return (
        TensorField("memory", (4, 1, 8), dtype, "contiguous", True),
        TensorField("ids", (4,), torch.int64, "contiguous", False),
        TensorField("optional", (1,), dtype, "local", True, present=False),
    )


class _Branches(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.main = nn.Parameter(torch.linspace(0.25, 0.5, 8, device="cuda", dtype=dtype))
        self.side = nn.Parameter(torch.linspace(0.5, 0.75, 8, device="cuda", dtype=dtype))

    def forward(self, hidden, state):
        memory = hidden * self.side
        return hidden * self.main, {
            "memory": memory,
            "ids": torch.arange(hidden.shape[0], device=hidden.device),
            "optional": None,
        }


class _Consumer(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.1, 0.4, 8, device="cuda", dtype=dtype))

    def forward(self, hidden, state):
        return hidden + state["memory"] * self.weight, state


def _loss(hidden, state, activity):
    loss = hidden.float().square().sum()
    if activity != "unused":
        loss = loss + state["memory"].float().square().sum() * (0.0 if activity == "zero" else 1.0)
    return loss


@pytest.mark.parametrize("activity", ["used", "unused", "zero"])
@pytest.mark.parametrize("input_grad", [False, True])
def test_checkpoint_preserves_parameter_only_and_absent_gradients(activity, input_grad):
    eager, replay = _Branches(), _Branches()
    a = StatefulModule(eager, output_fields=_fields())
    b = StatefulModule(replay, output_fields=_fields())
    x = torch.randn(4, 1, 8, device="cuda", requires_grad=input_grad)
    y = x.detach().clone().requires_grad_(input_grad)
    h1, s1 = a.run(x, {})
    h2, s2 = b.run(y, {}, recompute=True)
    torch.testing.assert_close(h1, h2)
    torch.testing.assert_close(s1["memory"], s2["memory"])
    assert s2["optional"] is None and not s2["ids"].requires_grad
    _loss(h1, s1, activity).backward()
    _loss(h2, s2, activity).backward()
    for p, q in zip(eager.parameters(), replay.parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad)
    if input_grad:
        torch.testing.assert_close(x.grad, y.grad)
    if activity == "zero":
        assert replay.side.grad is not None and not replay.side.grad.any()


@pytest.mark.parametrize("recompute", [False, True])
def test_two_pending_forwards_accumulate_shared_producer_gradients(recompute):
    producer = StatefulModule(_Branches(), output_fields=_fields())
    consumer = StatefulModule(_Consumer(), _fields(), _fields())
    reference_producer, reference_consumer = deepcopy(producer), deepcopy(consumer)
    pending, expected = [], []
    for _ in range(2):
        x = torch.randn(4, 1, 8, device="cuda", requires_grad=True)
        y = x.detach().clone().requires_grad_()
        hidden, state = producer.run(x, {}, recompute=recompute)
        first_memory = state["memory"]
        hidden, state = consumer.run(hidden, state, recompute=recompute)
        hidden, state = consumer.run(hidden, state, recompute=recompute)
        ref_hidden, ref_state = reference_producer.run(y, {})
        for _ in range(2):
            ref_hidden, ref_state = reference_consumer.run(ref_hidden, ref_state)
        torch.testing.assert_close(hidden, ref_hidden)
        assert state["memory"] is first_memory
        pending.append((x, _loss(hidden, state, "used")))
        expected.append((y, _loss(ref_hidden, ref_state, "used")))
    for (x, loss), (y, ref_loss) in zip(pending, expected):
        loss.backward()
        ref_loss.backward()
        torch.testing.assert_close(x.grad, y.grad)
    for a, b in [(producer, reference_producer), (consumer, reference_consumer)]:
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad)


class _RngRegion(nn.Module):
    def forward(self, hidden, state):
        with rng.get_cuda_rng_tracker().fork():
            result = torch.nn.functional.dropout(hidden, p=0.25, training=True)
        return result, {"memory": result.square()}


@pytest.mark.parametrize("tracker_kind", ["native", "native_graphsafe", "te"])
def test_checkpoint_replays_model_parallel_rng_and_restores_tracker(monkeypatch, tracker_kind):
    if tracker_kind == "te":
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

        tracker = TECudaRNGStatesTracker()
    else:
        tracker = rng.CudaRNGStatesTracker(use_cudagraphable_rng=tracker_kind == "native_graphsafe")
    monkeypatch.setattr(rng, "_CUDA_RNG_STATE_TRACKER", tracker)
    monkeypatch.setattr(rng, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", True)

    def snapshot(states):
        return {
            k: v.clone_state() if isinstance(v, torch.Generator) else v.clone()
            for k, v in states.items()
        }

    tracker.add(rng._MODEL_PARALLEL_RNG_TRACKER_NAME, 91)
    initial = snapshot(tracker.get_states())
    region = StatefulModule(_RngRegion(), output_fields=(_fields()[0],))
    x = torch.ones(4, 1, 8, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    h1, s1 = region.run(x, {})
    next_state = snapshot(tracker.get_states())
    tracker.set_states(initial)
    h2, s2 = region.run(y, {}, recompute=True)
    _loss(h1, s1, "used").backward()
    _loss(h2, s2, "used").backward()
    torch.testing.assert_close(x.grad, y.grad)
    for key, value in tracker.get_states().items():
        expected = next_state[key]
        if isinstance(value, torch.Generator):
            value, expected = value.get_state(), expected.get_state()
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert not rng.is_checkpointing()


@pytest.mark.parametrize("backend", ["torch", "transformer_engine"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_graph_slots_preserve_two_outstanding_microbatches(backend, dtype):
    if backend == "transformer_engine":
        pytest.importorskip("transformer_engine.pytorch")
    reference = StatefulModule(_Branches(dtype), output_fields=_fields(dtype))
    actual = deepcopy(reference)
    sample = torch.randn(4, 1, 8, device="cuda", dtype=dtype, requires_grad=True)
    graphs = StatefulGraphs(actual, sample, {}, slots=2, backend=backend)
    for _ in range(2):
        reference.zero_grad(set_to_none=True)
        actual.zero_grad(set_to_none=True)
        pending = []
        for slot in range(2):
            x = torch.randn_like(sample, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            hidden, state = graphs.run(x, {}, slot=slot)
            expected, ref_state = reference.run(y, {})
            torch.testing.assert_close(hidden, expected)
            torch.testing.assert_close(state["memory"], ref_state["memory"])
            assert not state["ids"].requires_grad and state["optional"] is None
            with pytest.raises(RuntimeError, match="outstanding"):
                graphs.run(x, {}, slot=slot)
            pending.append((x, _loss(hidden, state, "used"), y, _loss(expected, ref_state, "used")))
        # FIFO backward is deliberately different from a shared graph pool's LIFO lifetime.
        for x, loss, y, ref_loss in pending:
            loss.backward()
            ref_loss.backward()
            torch.testing.assert_close(x.grad, y.grad)
        for p, q in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad)


def test_graph_rejects_changed_profile_and_unused_differentiable_output():
    region = StatefulModule(_Branches(), output_fields=_fields())
    x = torch.randn(4, 1, 8, device="cuda", requires_grad=True)
    graphs = StatefulGraphs(region, x, {})
    with pytest.raises(ValueError, match="profile"):
        graphs.run(x[:2], {})
    hidden, _ = graphs.run(x, {})
    with pytest.raises(RuntimeError, match="Every differentiable"):
        hidden.sum().backward()


def test_schema_payload_roundtrip_preserves_aliases_and_optional_fields():
    hidden = torch.randn(4, 1, 8, requires_grad=True)
    fields = (
        TensorField("shared", hidden.shape, hidden.dtype, "local", True),
        TensorField("readonly", hidden.shape, hidden.dtype, "local", False),
        TensorField("absent", (1,), hidden.dtype, "local", True, False),
    )
    schema = TensorSchema(fields)
    payload = TensorStatePayload.from_state(
        hidden, {"shared": hidden, "readonly": hidden, "absent": None}, schema
    )
    restored, state = payload.restore()
    assert restored is hidden and state["shared"] is hidden
    assert not state["readonly"].requires_grad
    assert state["readonly"].data_ptr() == hidden.data_ptr()
    assert state["absent"] is None
    assert schema.fingerprint == TensorSchema(fields).fingerprint
    with pytest.raises(ValueError, match="Absent"):
        schema.pack({"shared": hidden, "readonly": hidden, "absent": torch.ones(1)})
    with pytest.raises(ValueError, match="duplicate"):
        TensorSchema((fields[0], fields[0]))


def test_last_consumer_retires_inputs_without_dropping_another_components_state():
    producer = StatefulModule(_Branches(), output_fields=_fields())
    consumer = StatefulModule(_Consumer(), _fields())
    x = torch.randn(4, 1, 8, device="cuda", requires_grad=True)
    peer = torch.ones(1, device="cuda")
    hidden, state = producer.run(x, {"peer/metadata": peer})
    output, remaining = consumer.run(hidden, state, recompute=True)
    assert set(remaining) == {"peer/metadata"} and remaining["peer/metadata"] is peer
    assert "memory" in state  # The caller's snapshot is not mutated.
    output.sum().backward()
    assert producer.module.side.grad is not None


@pytest.mark.parametrize("backend", ["torch", "transformer_engine"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_graph_with_real_te_normalized_projection_and_shared_output(backend, dtype):
    te = pytest.importorskip("transformer_engine.pytorch")

    class Region(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = te.LayerNormLinear(
                16,
                16,
                bias=False,
                params_dtype=dtype,
                device="cuda",
                return_layernorm_output=True,
                normalization="RMSNorm",
            )

        def forward(self, hidden, state):
            output, normalized = self.projection(hidden)
            return output, {"memory": normalized}

    fields = (TensorField("memory", (4, 2, 16), dtype, "contiguous", True),)
    actual = StatefulModule(Region(), output_fields=fields)
    expected = deepcopy(actual)
    x = torch.randn(4, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    graphs = StatefulGraphs(actual, x, {}, backend=backend)
    output, state = graphs.run(x, {})
    reference, reference_state = expected.run(y, {})
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(state["memory"], reference_state["memory"])
    _loss(output, state, "used").backward()
    _loss(reference, reference_state, "used").backward()
    torch.testing.assert_close(x.grad, y.grad)
    for p, q in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    del output, reference, state, reference_state
    graphs.close()
