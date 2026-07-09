# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid import shortcut_block as shortcut_block_module
from megatron.core.models.hybrid.shortcut_block import (
    ShortcutExecutionMode,
    ShortcutMoEBlock,
    _OutputProjSharedExperts,
    _RecordCombineGradReady,
    _RouteInputCompute,
)
from megatron.core.transformer.moe.shortcut_cudagraph import (
    AsyncCombineToPersistentBuffer as _AsyncCombineToPersistentBuffer,
    AsyncDispatchToPersistentGradBuffers as _AsyncDispatchToPersistentGradBuffers,
    PersistentBuffer,
)
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture
def cuda_graph_mempool():
    """Provide the graph pool normally created by CudaGraphManager in production."""
    from megatron.core.transformer.cuda_graphs import CudaGraphManager

    previous_mempool = CudaGraphManager.global_mempool
    if previous_mempool is None:
        CudaGraphManager.global_mempool = torch.cuda.graph_pool_handle()
    yield
    CudaGraphManager.global_mempool = previous_mempool


def test_shortcut_composite_modules_propagate_layer_boundaries():
    config = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=1)

    class FakeCompute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 1
            self.is_first_layer = True
            self.is_last_layer = False

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_experts = torch.nn.Identity()

        def shortcut_graph_participants(self):
            return [], []

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 2
            self.is_first_layer = False
            self.is_last_layer = True
            self.shortcut_pre_mlp_layernorm = torch.nn.Identity()
            self.pre_mlp_layernorm = torch.nn.Identity()
            self.mlp = FakeMLP()

    block = ShortcutMoEBlock(
        FakeCompute(),
        FakeMoE(),
        is_mamba=False,
        enable_cudagraph=False,
        overlap_a2a=False,
    )

    assert block.route_input_compute.is_first_layer
    assert not block.route_input_compute.is_last_layer
    assert not block.output_shared.is_first_layer
    assert block.output_shared.is_last_layer


def test_persistent_buffer_releases_strong_reference_once(monkeypatch):
    from megatron.core.transformer import cuda_graphs

    strong_tensor = torch.empty(1)
    strong_tensor.is_from_global_mempool = True
    weak_tensor = torch.empty(1)
    calls = []
    monkeypatch.setattr(
        cuda_graphs,
        "make_weakref",
        lambda tensor, inplace: calls.append((tensor, inplace)) or weak_tensor,
    )

    buffer = PersistentBuffer("test")
    buffer._tensor = strong_tensor
    buffer.release_strong_reference()
    buffer.release_strong_reference()

    assert buffer.tensor is weak_tensor
    assert calls == [(strong_tensor, False)]


def test_output_capture_releases_only_forward_persistent_buffers(monkeypatch):
    releases = []
    target = object.__new__(_OutputProjSharedExperts)
    target._forward_persistent_buffers = (
        SimpleNamespace(release_strong_reference=lambda: releases.append("route_input")),
        SimpleNamespace(release_strong_reference=lambda: releases.append("route_probs")),
        SimpleNamespace(release_strong_reference=lambda: releases.append("combined_output")),
    )
    monkeypatch.setattr(shortcut_block_module, "is_graph_capturing", lambda: True)
    monkeypatch.setattr(shortcut_block_module, "is_graph_warmup", lambda: False)

    target._weakref_forward_persistent_buffers()

    assert releases == ["route_input", "route_probs", "combined_output"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_persistent_dispatch_gradients_flow_into_route_backward(
    monkeypatch, cuda_graph_mempool
):
    calls = []

    class RecordAttentionBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            return tensor

        @staticmethod
        def backward(ctx, grad):
            calls.append("attention_backward")
            return grad

    class FakeCompute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(5.0))

        def input_proj_attn(self, hidden_states, **kwargs):
            output = RecordAttentionBackward.apply(hidden_states * self.scale)
            return output, hidden_states

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.prob_scale = torch.nn.Parameter(torch.tensor(3.0))
            self.dispatch_args = None

        def shortcut_route_preprocess(self, shortcut_hidden, padding_mask=None):
            return shortcut_hidden * self.input_scale, shortcut_hidden * self.prob_scale

        def _restore_token_dispatcher_attrs_for_dispatch(self, probs):
            self.restored_probs = probs

        def shortcut_launch_dispatch(self, route_input, route_probs, event, **kwargs):
            self.dispatch_args = (route_input, route_probs, event, kwargs)

    class FakeStream:
        def wait_event(self, event):
            calls.append(("route_grad_wait", event))

    compute = FakeCompute().cuda()
    moe = FakeMoE().cuda()
    stream = FakeStream()
    route_ready_event = SimpleNamespace(
        record=lambda actual_stream: calls.append(("route_ready_record", actual_stream))
    )
    route_grad_ready_event = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)

    route_compute = object.__new__(_RouteInputCompute)
    torch.nn.Module.__init__(route_compute)
    object.__setattr__(route_compute, 'moe_layer', moe)
    route_compute.route_input_buffer = PersistentBuffer("route input", requires_grad=True)
    route_compute.route_probs_buffer = PersistentBuffer(
        "route probabilities", requires_grad=True
    )
    route_compute.route_input_grad_buffer = PersistentBuffer("route input gradient")
    route_compute.route_probs_grad_buffer = PersistentBuffer("route probability gradient")
    object.__setattr__(
        route_compute, '_execution_mode', ShortcutExecutionMode.CUDA_GRAPH_OVERLAP
    )
    route_compute.route_ready_event = route_ready_event
    route_compute.route_grad_ready_event = route_grad_ready_event
    route_compute.compute_layer = compute
    route_compute._is_mamba = False

    block = object.__new__(ShortcutMoEBlock)
    block.route_input_compute = route_compute
    block.moe_layer = moe

    hidden = torch.ones(4, device="cuda", requires_grad=True)
    paired_state = route_compute(hidden)
    block.launch_dispatch_with_partial_cudagraph()

    route_input, route_probs, actual_event, dispatch_kwargs = moe.dispatch_args
    assert route_input is route_compute.route_input_buffer.tensor
    assert route_probs is route_compute.route_probs_buffer.tensor
    assert actual_event is route_ready_event
    assert moe.restored_probs is route_probs
    route_grad_buffers = dispatch_kwargs["route_grad_buffers"]
    assert route_grad_buffers[0] is route_compute.route_input_grad_buffer.tensor
    assert route_grad_buffers[1] is route_compute.route_probs_grad_buffer.tensor
    assert dispatch_kwargs["route_grad_ready_event"] is route_grad_ready_event

    route_compute.route_input_grad_buffer.tensor.fill_(7.0)
    route_compute.route_probs_grad_buffer.tensor.fill_(11.0)
    paired_state[0].sum().backward()

    torch.testing.assert_close(compute.scale.grad, torch.tensor(4.0, device="cuda"))
    torch.testing.assert_close(moe.input_scale.grad, torch.tensor(28.0, device="cuda"))
    torch.testing.assert_close(moe.prob_scale.grad, torch.tensor(44.0, device="cuda"))
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 52.0))
    assert calls.index("attention_backward") < calls.index(
        ("route_grad_wait", route_grad_ready_event)
    )


def test_output_shared_uses_method_level_graph_manager(monkeypatch):
    from megatron.core.transformer import cuda_graphs

    target = object.__new__(_OutputProjSharedExperts)
    torch.nn.Module.__init__(target)
    object.__setattr__(
        target, '_execution_mode', ShortcutExecutionMode.CUDA_GRAPH_OVERLAP
    )
    manager = object()
    monkeypatch.setattr(cuda_graphs, "CudaGraphManager", lambda *args, **kwargs: manager)

    target.create_mcore_cudagraph_manager(SimpleNamespace(cuda_graph_impl="local"))

    assert target.cudagraph_manager_output_shared_postprocess is manager
    assert not hasattr(target, 'cudagraph_manager')
    block = object.__new__(ShortcutMoEBlock)
    block.output_shared = target
    assert block.cudagraph_manager is manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_persistent_buffer_reuses_storage_and_validates_metadata(cuda_graph_mempool):
    persistent = PersistentBuffer(
        "test",
        requires_grad=True,
        detach_on_reuse=True,
    )
    source = torch.randn(4, 8, device="cuda")

    first = persistent.copy_from(source)
    first_ptr = first.data_ptr()
    second = persistent.acquire_like(source)

    assert second.data_ptr() == first_ptr
    assert second.requires_grad
    torch.testing.assert_close(second, source)
    with pytest.raises(AssertionError, match="metadata changed"):
        persistent.acquire_like(torch.empty(2, 8, device="cuda"))


def test_persistent_buffer_rejects_cpu_template():
    with pytest.raises(ValueError, match="requires a CUDA tensor template"):
        PersistentBuffer("test").acquire_like(torch.empty(2, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_output_shared_owns_one_persistent_combine_buffer(cuda_graph_mempool):
    target = object.__new__(_OutputProjSharedExperts)
    object.__setattr__(
        target, '_execution_mode', ShortcutExecutionMode.CUDA_GRAPH_OVERLAP
    )
    target.combined_output_buffer = PersistentBuffer(
        "combined output",
        prebound_graph_input=True,
        detach_on_reuse=True,
    )

    template = torch.empty_strided((8, 16), (16, 1), device="cuda")
    buffer = target.get_persistent_combined_output_buffer(template)
    buffer_ptr = buffer.data_ptr()

    # Reuse detaches the prior autograd wrapper, but the graph-visible allocation is fixed.
    reused_buffer = target.get_persistent_combined_output_buffer(template)
    assert reused_buffer.data_ptr() == buffer_ptr
    assert target.combined_output_buffer.tensor.data_ptr() == buffer_ptr

    with pytest.raises(AssertionError, match="metadata changed"):
        target.get_persistent_combined_output_buffer(torch.empty(4, 16, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_dispatch_bridge_publishes_both_route_gradients():
    class StatefulManager:
        def __init__(self, stale_probs):
            self.token_probs = stale_probs
            self.dispatched_probs = stale_probs
            self.tokens_per_expert = torch.ones(1, dtype=torch.int64, device="cuda")

        def dispatch(self, hidden_states, async_finish, allocate_on_comm_stream):
            self.dispatch_token_probs = self.token_probs
            self.dispatched_probs = 3 * self.token_probs
            return 2 * hidden_states

        def get_permuted_hidden_states_by_experts(self, hidden_states):
            return hidden_states, self.dispatched_probs

        def get_number_of_tokens_per_expert(self):
            return self.tokens_per_expert

    route_input = torch.randn(8, 16, device="cuda", requires_grad=True)
    route_probs = torch.randn(8, 4, device="cuda", requires_grad=True)
    route_input_grad_buffer = torch.empty_like(route_input)
    route_probs_grad_buffer = torch.empty_like(route_probs)
    dispatch_stream = torch.cuda.Stream()
    route_grad_ready_event = torch.cuda.Event(external=True)
    manager = StatefulManager(torch.zeros_like(route_probs))
    dispatcher = object.__new__(MoEFlexTokenDispatcher)
    dispatcher.shared_experts = None
    dispatcher._comm_manager = manager
    module = SimpleNamespace(dispatch=dispatcher.token_dispatch)

    dispatch_stream.wait_stream(torch.cuda.current_stream())
    dispatched_input, dispatched_probs = _AsyncDispatchToPersistentGradBuffers.apply(
        route_input,
        route_probs,
        module,
        dispatch_stream,
        route_input_grad_buffer,
        route_probs_grad_buffer,
        route_grad_ready_event,
    )
    torch.cuda.current_stream().wait_stream(dispatch_stream)
    expert_input, _, expert_probs = dispatcher.dispatch_postprocess(
        dispatched_input, dispatched_probs
    )

    assert manager.dispatch_token_probs is not route_probs
    assert manager.dispatch_token_probs.data_ptr() == route_probs.data_ptr()
    assert manager.dispatched_probs is dispatched_probs
    assert expert_probs is dispatched_probs

    grad_input = torch.randn_like(expert_input)
    grad_probs = torch.randn_like(expert_probs)
    torch.autograd.backward(
        (expert_input, expert_probs),
        (grad_input, grad_probs),
    )
    torch.cuda.synchronize()

    assert route_grad_ready_event.query()
    torch.testing.assert_close(route_input_grad_buffer, 2 * grad_input)
    torch.testing.assert_close(route_probs_grad_buffer, 3 * grad_probs)
    assert route_input.grad is None
    assert route_probs.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_dispatch_bridge_zero_fills_an_unused_private_input():
    route_input = torch.randn(8, 4, device="cuda", requires_grad=True)
    route_probs = torch.randn_like(route_input, requires_grad=True)
    route_input_grad_buffer = torch.empty_like(route_input)
    route_probs_grad_buffer = torch.empty_like(route_probs)
    dispatch_stream = torch.cuda.Stream()
    route_grad_ready_event = torch.cuda.Event(external=True)
    # Both outputs remain differentiable, but this dispatch variant does not consume route_probs.
    module = SimpleNamespace(
        dispatch=lambda inputs, probs: (2 * inputs, inputs + 0)
    )

    dispatch_stream.wait_stream(torch.cuda.current_stream())
    dispatched_input, dispatched_probs = _AsyncDispatchToPersistentGradBuffers.apply(
        route_input,
        route_probs,
        module,
        dispatch_stream,
        route_input_grad_buffer,
        route_probs_grad_buffer,
        route_grad_ready_event,
    )
    torch.cuda.current_stream().wait_stream(dispatch_stream)

    grad_input = torch.randn_like(dispatched_input)
    grad_probs = torch.randn_like(dispatched_probs)
    torch.autograd.backward(
        (dispatched_input, dispatched_probs),
        (grad_input, grad_probs),
    )
    torch.cuda.synchronize()

    assert route_grad_ready_event.query()
    torch.testing.assert_close(route_input_grad_buffer, 2 * grad_input + grad_probs)
    torch.testing.assert_close(route_probs_grad_buffer, torch.zeros_like(route_probs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_combine_bridge_preserves_gradient_and_event_order():
    source = torch.randn(8, 16, device="cuda", requires_grad=True)
    combine_stream = torch.cuda.Stream()
    ready_event = torch.cuda.Event(external=True)
    grad_ready_event = torch.cuda.Event(external=True)
    destination = None

    def persistent_output_factory(like):
        nonlocal destination
        destination = torch.empty_strided(
            like.size(), like.stride(), dtype=like.dtype, device=like.device
        )
        return destination

    module = SimpleNamespace(combine=lambda tensor: 2 * tensor)
    output = _AsyncCombineToPersistentBuffer.apply(
        source,
        module,
        combine_stream,
        persistent_output_factory,
        ready_event,
        grad_ready_event,
    )
    assert output.data_ptr() == destination.data_ptr()

    torch.cuda.current_stream().wait_event(ready_event)
    output = _RecordCombineGradReady.apply(output, grad_ready_event)
    grad = torch.randn_like(output)
    output.backward(grad)
    torch.cuda.synchronize()
    assert ready_event.query()
    assert grad_ready_event.query()
    torch.testing.assert_close(source.grad, 2 * grad)


@pytest.mark.parametrize("is_mamba", [True, False], ids=["mamba", "attention"])
def test_eager_overlap_matches_serial_forward_and_gradients(monkeypatch, is_mamba):
    class FakeCompute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.output_scale = torch.nn.Parameter(torch.tensor(3.0))

        def input_proj_ssm(self, hidden_states, **kwargs):
            return hidden_states * self.input_scale, hidden_states

        def input_proj_attn(self, hidden_states, **kwargs):
            return hidden_states * self.input_scale, hidden_states

        def output_proj(self, projected, residual, **kwargs):
            return projected * self.output_scale

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dispatch_scale = torch.nn.Parameter(torch.tensor(5.0))
            self.expert_scale = torch.nn.Parameter(torch.tensor(7.0))
            self.combine_scale = torch.nn.Parameter(torch.tensor(11.0))
            self._combined_output = None

        def dispatch(self, route_input, route_probs):
            return route_input * self.dispatch_scale, route_probs

        def routed_experts_compute(self, dispatched_input, dispatched_probs):
            return dispatched_input + dispatched_probs * self.expert_scale, None

        def combine(self, routed_output):
            return routed_output * self.combine_scale

        def wait_combine(self):
            return self._combined_output

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 2
            self.route_scale = torch.nn.Parameter(torch.tensor(13.0))
            self.prob_scale = torch.nn.Parameter(torch.tensor(17.0))
            self.shared_scale = torch.nn.Parameter(torch.tensor(19.0))
            self.mlp = FakeMLP()
            self._dispatch_output = None

        def shortcut_route_preprocess(self, shortcut_hidden, padding_mask=None):
            return shortcut_hidden * self.route_scale, shortcut_hidden * self.prob_scale

        def shortcut_launch_dispatch(self, route_input, route_probs, ready_event):
            self._dispatch_output = self.mlp.dispatch(route_input, route_probs)

        def shortcut_wait_dispatch_and_launch_combine(self, backward_dependency):
            dispatched_input, dispatched_probs = self._dispatch_output
            routed_output, _ = self.mlp.routed_experts_compute(
                dispatched_input, dispatched_probs
            )
            self.mlp._combined_output = self.mlp.combine(routed_output)

        def _shortcut_shared_experts(self, hidden_states):
            return hidden_states * self.shared_scale

        def shortcut_postprocess_with_combined_output(
            self, hidden_states, combined_output, shared_expert_output
        ):
            return hidden_states + combined_output + shared_expert_output

    @contextmanager
    def quant_context_factory(config, layer_number):
        yield

    def run(mode):
        compute = FakeCompute()
        moe = FakeMoE()

        route_input_compute = object.__new__(_RouteInputCompute)
        torch.nn.Module.__init__(route_input_compute)
        object.__setattr__(route_input_compute, 'moe_layer', moe)
        object.__setattr__(route_input_compute, '_execution_mode', mode)
        route_input_compute.route_ready_event = (
            SimpleNamespace(record=lambda stream: None)
            if mode == ShortcutExecutionMode.EAGER_OVERLAP
            else None
        )
        route_input_compute.route_grad_ready_event = None
        route_input_compute.compute_layer = compute
        route_input_compute._is_mamba = is_mamba

        output_shared = object.__new__(_OutputProjSharedExperts)
        torch.nn.Module.__init__(output_shared)
        object.__setattr__(output_shared, 'moe_layer', moe)
        object.__setattr__(output_shared, '_execution_mode', mode)
        output_shared.compute_layer = compute
        output_shared._is_mamba = is_mamba

        block = object.__new__(ShortcutMoEBlock)
        block.compute_layer = compute
        block.moe_layer = moe
        block.execution_mode = mode
        block.route_input_compute = route_input_compute
        block.output_shared = output_shared

        hidden_states = torch.arange(1.0, 5.0, requires_grad=True)
        output = block.forward(
            hidden_states=hidden_states,
            attention_mask=None,
            inference_context=None,
            rotary_pos_emb=None,
            sequence_len_offset=None,
            packed_seq_params=None,
            padding_mask=None,
            quant_context_factory=quant_context_factory,
            quant_config=None,
        )
        output.sum().backward()
        gradients = [hidden_states.grad]
        gradients.extend(parameter.grad for parameter in compute.parameters())
        gradients.extend(parameter.grad for parameter in moe.parameters())
        return output, gradients

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    serial_output, serial_gradients = run(ShortcutExecutionMode.EAGER_SERIAL)
    overlap_output, overlap_gradients = run(ShortcutExecutionMode.EAGER_OVERLAP)

    torch.testing.assert_close(overlap_output, serial_output)
    assert len(overlap_gradients) == len(serial_gradients)
    for overlap_gradient, serial_gradient in zip(overlap_gradients, serial_gradients):
        torch.testing.assert_close(overlap_gradient, serial_gradient)


@pytest.mark.parametrize(
    ("is_mamba", "compute_method"),
    [(True, "input_proj_ssm"), (False, "input_proj_attn")],
)
@pytest.mark.parametrize(
    ("execution_mode", "enable_cudagraph"),
    [
        (ShortcutExecutionMode.EAGER_SERIAL, False),
        (ShortcutExecutionMode.EAGER_OVERLAP, False),
        (ShortcutExecutionMode.CUDA_GRAPH_OVERLAP, True),
    ],
    ids=["eager-serial", "eager-overlap", "graph-overlap"],
)
def test_shortcut_block_selects_a2a_schedule(
    monkeypatch,
    is_mamba,
    compute_method,
    execution_mode,
    enable_cudagraph,
):
    calls = []
    quant_contexts = []

    hidden_states = object()
    attention_mask = object()
    inference_context = object()
    rotary_pos_emb = object()
    packed_seq_params = object()
    padding_mask = object()
    quant_config = object()

    permuted_input = object()
    dispatch_probs = object()
    compute_dependency = object()
    compute_aux = object()
    dispatched_input = object()
    dispatched_probs = object()
    routed_output = object()
    projected_hidden = object()
    shared_expert_output = object()
    combined_output = object()
    expected_combined_output = combined_output
    final_output = object()

    def route_preprocess(*, shortcut_hidden, padding_mask):
        assert shortcut_hidden is hidden_states
        assert padding_mask is expected_padding_mask
        calls.append("route")
        calls.append("prepare_dispatch")
        return permuted_input, dispatch_probs

    def input_projection(**kwargs):
        assert kwargs == {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "inference_context": expected_inference_context,
            "rotary_pos_emb": rotary_pos_emb,
            "sequence_len_offset": 17,
            "packed_seq_params": packed_seq_params,
            "padding_mask": expected_padding_mask,
        }
        calls.append(compute_method)
        return compute_dependency, compute_aux

    def route_input_compute(actual_hidden, **kwargs):
        assert actual_hidden is hidden_states
        route_preprocess(shortcut_hidden=actual_hidden, padding_mask=kwargs["padding_mask"])
        if execution_mode != ShortcutExecutionMode.EAGER_SERIAL:
            calls.append("record_event")
        paired_state = input_projection(hidden_states=actual_hidden, **kwargs)
        if execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            return paired_state
        return permuted_input, dispatch_probs, paired_state

    route_ready_event = SimpleNamespace(record=lambda stream: calls.append("record_event"))

    def launch_dispatch(actual_input, actual_probs, actual_event):
        assert actual_input is permuted_input
        assert actual_probs is dispatch_probs
        assert actual_event is route_ready_event
        calls.append("launch_dispatch")

    combine_ready_event = object()
    combine_grad_ready_event = object()

    def wait_dispatch_and_launch_combine(
        backward_dependency,
        persistent_output_factory=None,
        ready_event=None,
        grad_ready_event=None,
    ):
        assert backward_dependency is compute_dependency
        calls.append("wait_dispatch_and_launch_combine")
        if enable_cudagraph:
            assert callable(persistent_output_factory)
            assert ready_event is combine_ready_event
            assert grad_ready_event is combine_grad_ready_event
            return combined_output
        assert persistent_output_factory is None
        assert ready_event is None
        assert grad_ready_event is None

    def dispatch(actual_input, actual_probs):
        assert actual_input is permuted_input
        assert actual_probs is dispatch_probs
        calls.append("dispatch")
        return dispatched_input, dispatched_probs

    def routed_experts(actual_input, actual_probs):
        assert actual_input is dispatched_input
        assert actual_probs is dispatched_probs
        calls.append("routed_experts")
        return routed_output, None

    def combine(actual_output):
        assert actual_output is routed_output
        calls.append("combine")
        return combined_output

    def wait_combine():
        calls.append("wait_combine")
        return combined_output

    def output_shared(
        *paired_state,
        combined_output=None,
        inference_context,
        padding_mask,
    ):
        assert paired_state == (compute_dependency, compute_aux)
        assert inference_context is expected_inference_context
        assert padding_mask is expected_padding_mask
        calls.append("output_shared")
        if combined_output is None:
            assert execution_mode == ShortcutExecutionMode.EAGER_OVERLAP
            combined_output = wait_combine()
        assert combined_output is expected_combined_output
        return postprocess(projected_hidden, combined_output, shared_expert_output)

    def postprocess(actual_hidden, actual_combined, actual_shared):
        assert actual_hidden is projected_hidden
        assert actual_combined is combined_output
        assert actual_shared is shared_expert_output
        calls.append("postprocess")
        return final_output

    moe_layer = SimpleNamespace(
        layer_number=8,
        mlp=SimpleNamespace(
            dispatch=dispatch,
            routed_experts_compute=routed_experts,
            combine=combine,
            wait_combine=wait_combine,
        ),
        shortcut_route_preprocess=route_preprocess,
        shortcut_launch_dispatch=launch_dispatch,
        shortcut_wait_dispatch_and_launch_combine=wait_dispatch_and_launch_combine,
        shortcut_postprocess_with_combined_output=postprocess,
    )
    compute_layer = SimpleNamespace()
    setattr(compute_layer, compute_method, input_projection)

    block = object.__new__(ShortcutMoEBlock)
    block.compute_layer = compute_layer
    block.moe_layer = moe_layer
    block.execution_mode = execution_mode
    route_input_compute.route_ready_event = route_ready_event
    block.route_input_compute = route_input_compute
    output_shared.get_persistent_combined_output_buffer = lambda tensor: tensor
    output_shared.combine_ready_event = combine_ready_event
    output_shared.combine_grad_ready_event = combine_grad_ready_event
    block.output_shared = output_shared
    if execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:

        def launch_unified_dispatch():
            calls.append("launch_dispatch")

        monkeypatch.setattr(
            block, "launch_dispatch_with_partial_cudagraph", launch_unified_dispatch
        )

    @contextmanager
    def quant_context_factory(actual_config, layer_number):
        assert actual_config is quant_config
        assert layer_number == 7
        quant_contexts.append("enter")
        yield
        quant_contexts.append("exit")

    expected_inference_context = inference_context
    expected_padding_mask = padding_mask
    output = block.forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        sequence_len_offset=17,
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
        quant_context_factory=quant_context_factory,
        quant_config=quant_config,
    )

    assert output is final_output
    if execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
        expected_calls = [
            "route",
            "prepare_dispatch",
            "record_event",
            compute_method,
            "launch_dispatch",
            "wait_dispatch_and_launch_combine",
            "output_shared",
            "postprocess",
        ]
        expected_num_contexts = 2
    elif execution_mode == ShortcutExecutionMode.EAGER_OVERLAP:
        expected_calls = [
            "route",
            "prepare_dispatch",
            "record_event",
            compute_method,
            "launch_dispatch",
            "wait_dispatch_and_launch_combine",
            "output_shared",
            "wait_combine",
            "postprocess",
        ]
        expected_num_contexts = 2
    else:
        expected_calls = [
            "route",
            "prepare_dispatch",
            compute_method,
            "dispatch",
            "routed_experts",
            "combine",
            "output_shared",
            "postprocess",
        ]
        expected_num_contexts = 3
    assert calls == expected_calls
    assert quant_contexts == ["enter", "exit"] * expected_num_contexts


def test_shortcut_execution_mode_resolution():
    assert (
        ShortcutExecutionMode.resolve(enable_cudagraph=False, overlap_a2a=False)
        == ShortcutExecutionMode.EAGER_SERIAL
    )
    assert (
        ShortcutExecutionMode.resolve(enable_cudagraph=False, overlap_a2a=True)
        == ShortcutExecutionMode.EAGER_OVERLAP
    )
    assert (
        ShortcutExecutionMode.resolve(enable_cudagraph=True, overlap_a2a=True)
        == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP
    )
    with pytest.raises(ValueError, match="require moe_shortcut_parallel"):
        ShortcutExecutionMode.resolve(enable_cudagraph=True, overlap_a2a=False)


def test_shortcut_rejects_shared_expert_overlap():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            num_moe_experts=2,
            moe_shortcut_connection=True,
            moe_shared_expert_overlap=True,
        )
