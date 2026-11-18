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


@pytest.fixture
def te_rng_tracker(monkeypatch):
    """Provide seeded graph-safe RNG without leaking MCore or TE tracker state."""
    te_distributed = pytest.importorskip("transformer_engine.pytorch.distributed")
    from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
    from megatron.core.tensor_parallel import random as rng

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    previous = te_distributed.get_all_rng_states()
    tracker = TECudaRNGStatesTracker()
    monkeypatch.setattr(rng, "_CUDA_RNG_STATE_TRACKER", tracker)
    monkeypatch.setattr(rng, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", True)
    try:
        tracker.add(rng._MODEL_PARALLEL_RNG_TRACKER_NAME, 123)
        yield tracker
    finally:
        te_distributed.set_all_rng_states(previous)


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
def test_checkpoint_replays_model_parallel_rng_and_restores_tracker(
    monkeypatch, tracker_kind, request
):
    if tracker_kind == "te":
        from transformer_engine.pytorch.distributed import get_all_rng_states, set_all_rng_states

        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

        previous = get_all_rng_states()
        request.addfinalizer(lambda: set_all_rng_states(previous))
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


@pytest.mark.usefixtures("te_rng_tracker")
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
    pending.clear()
    del hidden, state, expected, ref_state, loss, ref_loss
    graphs.close()


def test_graph_rejects_changed_profile_and_unused_differentiable_output():
    region = StatefulModule(_Branches(), output_fields=_fields())
    x = torch.randn(4, 1, 8, device="cuda", requires_grad=True)
    graphs = StatefulGraphs(region, x, {})
    with pytest.raises(ValueError, match="profile"):
        graphs.run(x[:2], {})
    hidden, _ = graphs.run(x, {})
    with pytest.raises(RuntimeError, match="Every differentiable"):
        hidden.sum().backward()
    graphs.abort()
    del hidden, _
    graphs.close()


def test_te_graph_rejects_legacy_rng_before_capture_and_recovers(te_rng_tracker):
    """A bad RNG profile must not poison later TE attention or graph captures."""
    from transformer_engine.pytorch import graph as te_graph
    from transformer_engine.pytorch.distributed import get_all_rng_states, set_all_rng_states

    region = StatefulModule(_Branches(), output_fields=_fields())
    sample = torch.randn(4, 1, 8, device="cuda", requires_grad=True)
    original_call = StatefulModule.__call__
    registered = get_all_rng_states()
    assert not te_graph.is_graph_capturing()
    try:
        set_all_rng_states({"legacy": torch.cuda.get_rng_state()})
        with pytest.raises(ValueError, match="graph-safe RNG states"):
            StatefulGraphs(region, sample, {}, backend="transformer_engine")
        assert not te_graph.is_graph_capturing()
        assert StatefulModule.__call__ is original_call
    finally:
        set_all_rng_states(registered)

    graphs = StatefulGraphs(region, sample, {}, backend="transformer_engine")
    hidden, state = graphs.run(sample, {})
    _loss(hidden, state, "used").backward()
    assert region.module.main.grad is not None and region.module.side.grad is not None
    del hidden, state
    graphs.close()


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
    assert restored is not hidden and restored.data_ptr() == hidden.data_ptr()
    assert state["shared"] is hidden
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


@pytest.mark.usefixtures("te_rng_tracker")
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


@pytest.mark.usefixtures("te_rng_tracker")
@pytest.mark.parametrize("backend", ["torch", "transformer_engine"])
@pytest.mark.parametrize("mode", ["no_grad", "frozen"])
def test_graph_outputs_survive_forward_only_slot_reuse(backend, mode):
    region = StatefulModule(_Branches(), output_fields=_fields())
    if mode == "frozen":
        region.requires_grad_(False)
    sample = torch.ones(4, 1, 8, device="cuda", requires_grad=mode != "frozen")
    graphs = StatefulGraphs(region, sample, {}, backend=backend)
    try:
        with torch.set_grad_enabled(mode != "no_grad"):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                hidden, state = graphs.run(sample, {})
                wire = tuple(
                    t.detach().contiguous() for t in (hidden, state["memory"], state["ids"])
                )
                expected = tuple(t.clone() for t in wire)
            second = (sample.detach() * 3).requires_grad_(sample.requires_grad)
            next_hidden, next_state = graphs.run(second, {})
            torch.cuda.current_stream().wait_stream(stream)
            for original, snapshot in zip(wire, expected):
                torch.testing.assert_close(original, snapshot)
            assert hidden.data_ptr() != next_hidden.data_ptr()
            assert state["memory"].data_ptr() != next_state["memory"].data_ptr()
            assert state["ids"].data_ptr() != next_state["ids"].data_ptr()
            del hidden, state, next_hidden, next_state, wire
    finally:
        graphs.close()


@pytest.mark.usefixtures("te_rng_tracker")
@pytest.mark.parametrize("backend", ["torch", "transformer_engine"])
def test_abort_rejects_old_backward_without_retiring_new_invocation(backend):
    region = StatefulModule(_Branches(), output_fields=_fields())
    sample = torch.ones(4, 1, 8, device="cuda", requires_grad=True)
    graphs = StatefulGraphs(region, sample, {}, backend=backend)
    old, old_state = graphs.run(sample, {})
    old_loss = _loss(old, old_state, "used")
    graphs.abort()
    fresh, state = graphs.run(sample, {})
    with pytest.raises(RuntimeError, match="aborted"):
        old_loss.backward()
    with pytest.raises(RuntimeError, match="outstanding"):
        graphs.run(sample, {})
    _loss(fresh, state, "used").backward()
    del old, old_state, old_loss, fresh, state
    graphs.close()


def test_graph_module_validation_can_run_outside_replay():
    region = StatefulModule(_Branches(), output_fields=_fields())
    sample = torch.ones(4, 1, 8, device="cuda", requires_grad=True)
    graphs = StatefulGraphs(region, sample, {}, debug_checks=True)
    region.eval()
    with pytest.raises(ValueError, match="training mode"):
        graphs.validate_module()
    with pytest.raises(ValueError, match="training mode"):
        graphs.run(sample, {})
    graphs.close()


def test_layout_typo_is_rejected_at_schema_construction():
    with pytest.raises(ValueError, match="Unsupported state layout"):
        TensorField("memory", (4, 1, 8), torch.float32, "contigous", True)


@pytest.mark.usefixtures("te_rng_tracker")
@pytest.mark.parametrize("backend", ["torch", "transformer_engine"])
def test_capture_failure_restores_process_settings(backend):
    from transformer_engine.pytorch import graph as te_graph

    from megatron.core.transformer import cuda_graphs

    class Broken(nn.Module):
        def forward(self, hidden, state):
            raise RuntimeError("intentional warmup failure")

    before_call = StatefulModule.__call__
    before_mcore = cuda_graphs.is_graph_capturing()
    before_te = te_graph.is_graph_capturing()
    get_override = getattr(torch._C, "_override_stale_capture_stream", None)
    before_override = None if get_override is None else get_override()
    sample = torch.ones(4, 1, 8, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="intentional warmup failure"):
        StatefulGraphs(StatefulModule(Broken()), sample, {}, backend=backend)
    assert StatefulModule.__call__ is before_call
    assert cuda_graphs.is_graph_capturing() == before_mcore
    assert te_graph.is_graph_capturing() == before_te
    if get_override is not None:
        assert get_override() == before_override


@pytest.mark.usefixtures('te_rng_tracker')
@pytest.mark.parametrize('backend', ['torch', 'transformer_engine'])
def test_graph_capture_preserves_registered_statistics(backend):
    """Graph warmup must not inflate router-like counters, including across slots."""

    class Counter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8, device='cuda'))
            self.register_buffer('calls', torch.tensor(17, device='cuda'))

        def forward(self, hidden, state):
            self.calls.add_(1)
            return hidden * self.weight, {}

    module = Counter()
    region = StatefulModule(module)
    sample = torch.ones(4, 1, 8, device='cuda', requires_grad=True)
    graphs = StatefulGraphs(region, sample, {}, slots=2, backend=backend)
    pre_hooks = []
    graphs.setup_manual_hooks(lambda: lambda owner: pre_hooks.append(owner))
    try:
        assert module.calls.item() == 17
        outputs = [graphs.run(sample.clone(), {}, slot=i)[0] for i in range(2)]
        for output in outputs:
            output.sum().backward()
        assert module.calls.item() == 19
        assert pre_hooks == [module, module]
    finally:
        graphs.abort()
        graphs.close()


@pytest.mark.usefixtures('te_rng_tracker')
def test_scheduled_graphs_reuse_backward_workspace_and_preserve_gradients():
    """Temporary backward storage must be shared across layers, not pinned per layer."""
    scratch_elements = 16 * 1024 * 1024

    class Projection(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight):
            ctx.save_for_backward(hidden, weight)
            return hidden * weight

        @staticmethod
        def backward(ctx, gradient):
            hidden, weight = ctx.saved_tensors
            scratch = torch.empty(scratch_elements, device=hidden.device)
            scratch.fill_(1)
            return gradient * weight * scratch[0], (gradient * hidden).sum((0, 1))

    class Region(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.linspace(0.5, 1, 8, device='cuda'))

        def forward(self, hidden, state):
            return Projection.apply(hidden, self.weight), {}

    regions = [StatefulModule(Region()) for _ in range(4)]
    reference = deepcopy(regions)
    sample = torch.ones(4, 1, 8, device='cuda', requires_grad=True)
    grouped, independent = [], []
    try:
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_reserved()
        grouped = StatefulGraphs.capture_many(
            [(region, sample, {}, None) for region in regions],
            order=[1, -1],
            num_layers_per_chunk=[4],
            capture_kwargs=dict(num_warmup_iters=1),
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        shared_bytes = torch.cuda.memory_reserved() - baseline
        baseline = torch.cuda.memory_reserved()
        independent = [
            StatefulGraphs(region, sample, {}, backend='transformer_engine', num_warmup_iters=1)
            for region in deepcopy(regions)
        ]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        independent_bytes = torch.cuda.memory_reserved() - baseline
        assert independent_bytes - shared_bytes >= 2 * scratch_elements * sample.element_size()
        for _ in range(2):
            x = torch.randn_like(sample, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            actual, expected = x, y
            for graph, region in zip(grouped, reference):
                actual, _ = graph.run(actual, {})
                expected, _ = region.run(expected, {})
            torch.testing.assert_close(actual, expected)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(x.grad, y.grad)
            for graph, region in zip(grouped, reference):
                for p, q in zip(graph.region.parameters(), region.parameters()):
                    torch.testing.assert_close(p.grad, q.grad)
                    with torch.no_grad():
                        p.add_(p.grad, alpha=-0.01)
                        q.add_(q.grad, alpha=-0.01)
                    p.grad = q.grad = None
    finally:
        for graph in (*grouped, *independent):
            graph.abort()
            graph.close()


@pytest.mark.usefixtures('te_rng_tracker')
def test_scheduled_graphs_preserve_fifo_microbatches_and_shared_state():
    """A pipeline schedule can reuse inputs only after the matching FIFO backward."""
    regions = [StatefulModule(_Branches(), output_fields=_fields()) for _ in range(2)]
    reference = deepcopy(regions)
    sample = torch.ones(4, 1, 8, device='cuda', requires_grad=True)
    order = [1, 1, 2, -1, 2, -1, -2, -2]
    graphs = StatefulGraphs.capture_many(
        [(region, sample, {}, None) for region in regions],
        order=order,
        num_layers_per_chunk=[1, 1],
        capture_kwargs=dict(num_warmup_iters=1),
    )
    try:
        for _ in range(2):
            pending = [[], []]
            slots = [0, 0]
            for operation in order:
                chunk = abs(operation) - 1
                if operation > 0:
                    x = torch.randn_like(sample, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    actual, state = graphs[chunk].run(x, {}, slot=slots[chunk])
                    expected, ref_state = reference[chunk].run(y, {})
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(state['memory'], ref_state['memory'])
                    pending[chunk].append((actual, state, expected, ref_state, x, y))
                    slots[chunk] += 1
                else:
                    actual, state, expected, ref_state, x, y = pending[chunk].pop(0)
                    _loss(actual, state, 'used').backward()
                    _loss(expected, ref_state, 'used').backward()
                    torch.testing.assert_close(x.grad, y.grad)
            for graph, region in zip(graphs, reference):
                for p, q in zip(graph.region.parameters(), region.parameters()):
                    torch.testing.assert_close(p.grad, q.grad)
                    p.grad = q.grad = None
    finally:
        for graph in graphs:
            graph.abort()
            graph.close()


@pytest.mark.usefixtures('te_rng_tracker')
@pytest.mark.parametrize('scheduled', [False, True])
def test_mxfp8_graph_weight_cache_refreshes_after_parameter_update(scheduled):
    """Reused graph slots must refresh updated weights once per training step."""
    te = pytest.importorskip('transformer_engine.pytorch')
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.fp8 import check_mxfp8_support

    supported, reason = check_mxfp8_support()
    if not supported:
        pytest.skip(reason)

    class Region(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = te.Linear(
                64, 64, bias=False, params_dtype=torch.bfloat16, device='cuda'
            )

        def forward(self, hidden, state):
            return self.projection(hidden), state

    region = StatefulModule(Region())
    weight = region.module.projection.weight
    with torch.no_grad():
        weight.fill_(1 / 64)
    sample = torch.ones(32, 1, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    options = dict(fp8_enabled=True, fp8_recipe=MXFP8BlockScaling(), fp8_weight_caching=True)
    graph = None
    try:
        if scheduled:
            graph = StatefulGraphs.capture_many(
                [(region, sample, {}, None)],
                order=[1, -1],
                num_layers_per_chunk=[1],
                capture_kwargs=dict(options, num_warmup_iters=1),
            )[0]
        else:
            graph = StatefulGraphs(
                region,
                sample,
                {},
                backend='transformer_engine',
                num_warmup_iters=1,
                capture_kwargs=options,
            )
        # Powers of two make the output, input grad and weight grad exactly
        # representable. A stale quantized weight changes both sign and scale.
        for weight_value in (1 / 64, -1 / 32):
            with torch.no_grad():
                weight.fill_(weight_value)
            region.zero_grad(set_to_none=True)
            for microbatch in range(3):
                hidden = torch.full_like(sample, microbatch + 1, requires_grad=True)
                output, _ = graph.run(hidden, {}, is_first_microbatch=microbatch == 0)
                torch.testing.assert_close(
                    output,
                    torch.full_like(output, 64 * weight_value * (microbatch + 1)),
                    rtol=0,
                    atol=0,
                )
                output.sum().backward()
                torch.testing.assert_close(
                    hidden.grad, torch.full_like(hidden, 64 * weight_value), rtol=0, atol=0
                )
            torch.testing.assert_close(
                weight.grad, torch.full_like(weight, 32 * (1 + 2 + 3)), rtol=0, atol=0
            )
    finally:
        if graph is not None:
            graph.abort()
            graph.close()


@pytest.mark.usefixtures('te_rng_tracker')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_shared_graph_replays_seven_and_eight_outputs_with_same_gradient_shapes(dtype):
    """Different integer state layouts must not collide in TE's gradient cache."""
    names = ('cu', 'cu_padded', 'memory', 'key', 'global_indices', 'candidate_ids', 'counts')

    class Region(nn.Module):
        def __init__(self, names):
            super().__init__()
            self.names = names
            self.weight = nn.Parameter(torch.linspace(0.25, 0.5, 8, device='cuda', dtype=dtype))

        def forward(self, hidden, state):
            output = hidden * self.weight
            values = {
                name: (
                    output * (2 if name == 'memory' else 3)
                    if name in ('memory', 'key')
                    else torch.arange(4, device=hidden.device, dtype=torch.int32)
                )
                for name in self.names
            }
            return output, values

    def region(names):
        fields = tuple(
            TensorField(
                name,
                (4, 1, 8) if name in ('memory', 'key') else (4,),
                dtype if name in ('memory', 'key') else torch.int32,
                'contiguous',
                name in ('memory', 'key'),
            )
            for name in names
        )
        return StatefulModule(Region(names), output_fields=fields)

    regions = [region(names), region(tuple(name for name in names if name != 'global_indices'))]
    reference = deepcopy(regions)
    sample = torch.ones(4, 1, 8, device='cuda', dtype=dtype, requires_grad=True)
    graphs = StatefulGraphs.capture_many(
        [(module, sample, {}, None) for module in regions],
        order=[1, 1, -1, -1],
        num_layers_per_chunk=[2],
        capture_kwargs=dict(num_warmup_iters=1),
    )
    try:
        for _ in range(2):
            pending = []
            for slot in range(2):
                x = torch.randn_like(sample, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                actual, expected = x, y
                actual_loss, expected_loss = 0, 0
                for graph, module in zip(graphs, reference):
                    actual, state = graph.run(actual, {}, slot=slot)
                    expected, ref_state = module.run(expected, {})
                    assert tuple(state) == tuple(ref_state)
                    torch.testing.assert_close(actual, expected)
                    for name in state:
                        torch.testing.assert_close(state[name], ref_state[name])
                        assert state[name].requires_grad == ref_state[name].requires_grad
                    actual_loss = actual_loss + state['memory'].sum() + state['key'].sum()
                    expected_loss = (
                        expected_loss + ref_state['memory'].sum() + ref_state['key'].sum()
                    )
                pending.append((x, y, actual_loss + actual.sum(), expected_loss + expected.sum()))
            for x, y, actual_loss, expected_loss in pending:
                actual_loss.backward()
                expected_loss.backward()
                torch.testing.assert_close(x.grad, y.grad)
            for module, ref in zip(regions, reference):
                torch.testing.assert_close(module.module.weight.grad, ref.module.weight.grad)
                module.zero_grad(set_to_none=True)
                ref.zero_grad(set_to_none=True)
    finally:
        for graph in graphs:
            graph.abort()
            graph.close()
