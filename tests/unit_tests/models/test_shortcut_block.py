# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.shortcut_block import (
    AsyncCombineToPersistentBuffer,
    AsyncDispatchToPersistentGradBuffers,
    PersistentBuffer,
    RecordCombineGradReady,
    ShortcutMoEBlock,
    _GraphState,
    _PersistentSlot,
    group_layers_into_shortcut_blocks,
)
from megatron.core.transformer.module import SplitOutputProjection
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


def test_group_layers_into_registered_shortcut_blocks():
    config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        mtp_use_repeated_layer=False,
        mtp_num_layers=None,
        moe_shortcut_parallel=True,
        fp32_residual_connection=False,
        hidden_size=8,
        layernorm_epsilon=1e-5,
        sequence_parallel=False,
        cuda_graph_modules=[],
    )

    class FakeCompute(torch.nn.Module, SplitOutputProjection):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 1
            self.is_first_layer = True
            self.is_last_layer = False

        def forward_pre_output_proj(self, hidden_states, **kwargs):
            return hidden_states

        def forward_output_proj(self, hidden_states, **kwargs):
            return hidden_states

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tp_group = None

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 2
            self.is_first_layer = False
            self.is_last_layer = False
            self.mlp = FakeMLP()

    compute = FakeCompute()
    moe = FakeMoE()
    standalone = torch.nn.Identity()
    config.moe_shortcut_parallel = False
    with pytest.raises(ValueError, match="requires moe_shortcut_parallel"):
        group_layers_into_shortcut_blocks(
            torch.nn.ModuleList([compute, moe]),
            [LayerSymbols.MAMBA, LayerSymbols.MOE],
            config,
        )
    config.moe_shortcut_parallel = True
    grouped = group_layers_into_shortcut_blocks(
        torch.nn.ModuleList([compute, moe, standalone]),
        [LayerSymbols.MAMBA, LayerSymbols.MOE, LayerSymbols.ATTENTION],
        config,
    )

    assert len(grouped) == 2
    assert isinstance(grouped[0], ShortcutMoEBlock)
    assert grouped[0].compute_layer is compute
    assert grouped[0].moe_layer is moe
    assert grouped[1] is standalone
    registered_modules = dict(grouped.named_modules())
    assert registered_modules["0.compute_layer"] is compute
    assert registered_modules["0.moe_layer"] is moe
    assert registered_modules["1"] is standalone


@pytest.fixture
def cuda_graph_mempool():
    """Provide the graph pool normally created by CudaGraphManager in production."""
    from megatron.core.transformer.cuda_graphs import CudaGraphManager

    previous_mempool = CudaGraphManager.global_mempool
    if previous_mempool is None:
        CudaGraphManager.global_mempool = torch.cuda.graph_pool_handle()
    yield
    CudaGraphManager.global_mempool = previous_mempool


def test_shortcut_block_registers_pair_and_post_norm():
    config = TransformerConfig(
        num_layers=2, hidden_size=8, num_attention_heads=1, sequence_parallel=True
    )
    config.moe_shortcut_parallel = True

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
            self.tp_group = None
            self.shared_experts = torch.nn.Identity()

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 2
            self.is_first_layer = False
            self.is_last_layer = True
            self.pre_mlp_layernorm = torch.nn.Identity()
            self.mlp = FakeMLP()

    compute = FakeCompute()
    moe = FakeMoE()
    block = ShortcutMoEBlock(compute, moe, enable_cudagraph=False)

    assert block.compute_layer is compute
    assert block.moe_layer is moe
    assert isinstance(block.shortcut_post_norm, torch.nn.RMSNorm)
    assert block.shortcut_post_norm.eps == config.layernorm_epsilon
    assert all(
        getattr(parameter, 'sequence_parallel', False)
        for parameter in block.shortcut_post_norm.parameters()
    )
    assert block.is_first_layer
    assert block.is_last_layer
    assert block._graph_state is None
    assert list(dict(block.named_children())) == [
        "compute_layer",
        "moe_layer",
        "shortcut_post_norm",
    ]


def test_shortcut_blocks_share_one_parallel_stream_per_device(monkeypatch):
    created_streams = []

    def make_stream(*, priority):
        stream = SimpleNamespace(priority=priority)
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(ShortcutMoEBlock, "_parallel_streams", {})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    first_block = object.__new__(ShortcutMoEBlock)
    second_block = object.__new__(ShortcutMoEBlock)

    first_stream = first_block._get_parallel_stream()
    second_stream = second_block._get_parallel_stream()

    assert first_stream is second_stream
    assert first_stream.priority == -1
    assert created_streams == [first_stream]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple CUDA devices required")
def test_shortcut_blocks_use_distinct_parallel_streams_across_devices(monkeypatch):
    monkeypatch.setattr(ShortcutMoEBlock, "_parallel_streams", {})

    with torch.cuda.device(0):
        first_stream = ShortcutMoEBlock._get_parallel_stream()
    with torch.cuda.device(1):
        second_stream = ShortcutMoEBlock._get_parallel_stream()

    assert first_stream is not second_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_persistent_dispatch_gradients_flow_into_route_backward(monkeypatch, cuda_graph_mempool):
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

        def forward_pre_output_proj(self, hidden_states, **kwargs):
            output = RecordAttentionBackward.apply(hidden_states * self.scale)
            return output, hidden_states

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.prob_scale = torch.nn.Parameter(torch.tensor(3.0))
            self.dispatch_args = None
            self._local_cudagraph_attr_names = ()

        def shortcut_route_preprocess(self, shortcut_hidden, padding_mask=None):
            return shortcut_hidden * self.input_scale, shortcut_hidden * self.prob_scale

        def _restore_token_dispatcher_attrs(self, attrs):
            self.restored_attrs = attrs

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

    block = object.__new__(ShortcutMoEBlock)
    torch.nn.Module.__init__(block)
    block.compute_layer = compute
    block.moe_layer = moe
    block.enable_cudagraph = True
    route_slot = SimpleNamespace(
        route_input_buffer=PersistentBuffer("route input", requires_grad=True),
        route_probs_buffer=PersistentBuffer("route probabilities", requires_grad=True),
        route_input_grad_buffer=PersistentBuffer("route input gradient"),
        route_probs_grad_buffer=PersistentBuffer("route probability gradient"),
        route_ready_event=route_ready_event,
        route_grad_ready_event=route_grad_ready_event,
    )
    block._graph_state = SimpleNamespace(get_slot=lambda index: route_slot)
    dispatch_output = (object(), object())

    def launch_dispatch(route_input, route_probs, event, **kwargs):
        moe.dispatch_args = (route_input, route_probs, event, kwargs)
        return dispatch_output

    monkeypatch.setattr(block, "_shortcut_route_preprocess", moe.shortcut_route_preprocess)
    monkeypatch.setattr(block, "_launch_dispatch_async", launch_dispatch)

    hidden = torch.ones(4, device="cuda", requires_grad=True)
    route_outputs = ShortcutMoEBlock.route_input_compute(block, hidden)
    paired_state, actual_dispatch_output = block._launch_dispatch(route_outputs, 0)
    assert len(paired_state) == len(route_outputs)
    assert all(actual is expected for actual, expected in zip(paired_state, route_outputs))
    assert actual_dispatch_output is dispatch_output

    route_input, route_probs, actual_event, dispatch_kwargs = moe.dispatch_args
    assert route_input is route_slot.route_input_buffer.tensor
    assert route_probs is route_slot.route_probs_buffer.tensor
    assert actual_event is route_ready_event
    assert moe.restored_attrs == ()
    route_grad_buffers = dispatch_kwargs["route_grad_buffers"]
    assert route_grad_buffers[0] is route_slot.route_input_grad_buffer.tensor
    assert route_grad_buffers[1] is route_slot.route_probs_grad_buffer.tensor
    assert dispatch_kwargs["route_grad_ready_event"] is route_grad_ready_event

    route_slot.route_input_grad_buffer.tensor.fill_(7.0)
    route_slot.route_probs_grad_buffer.tensor.fill_(11.0)
    paired_state[0].sum().backward()

    torch.testing.assert_close(compute.scale.grad, torch.tensor(4.0, device="cuda"))
    torch.testing.assert_close(moe.input_scale.grad, torch.tensor(28.0, device="cuda"))
    torch.testing.assert_close(moe.prob_scale.grad, torch.tensor(44.0, device="cuda"))
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 52.0))
    assert calls.index("attention_backward") < calls.index(
        ("route_grad_wait", route_grad_ready_event)
    )


def test_shortcut_block_uses_two_method_level_graph_managers(monkeypatch):
    from megatron.core.transformer import cuda_graphs

    target = object.__new__(ShortcutMoEBlock)
    torch.nn.Module.__init__(target)
    target.enable_cudagraph = True
    target.config = SimpleNamespace(moe_latent_size=None)
    target._graph_state = SimpleNamespace(route_manager=None, output_manager=None)
    target.compute_layer = torch.nn.Identity()
    target.compute_layer.is_first_layer = True
    target.compute_layer.is_last_layer = False
    target.moe_layer = SimpleNamespace(
        is_first_layer=False,
        is_last_layer=True,
        pre_mlp_layernorm=torch.nn.Identity(),
        mlp=SimpleNamespace(shared_experts=torch.nn.Identity()),
    )
    target.shortcut_post_norm = torch.nn.Identity()
    calls = []

    def make_manager(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.nn.Identity()

    monkeypatch.setattr(cuda_graphs, "CudaGraphManager", make_manager)

    target.create_mcore_cudagraph_manager(
        SimpleNamespace(cuda_graph_impl="local", moe_latent_size=None)
    )

    assert len(calls) == 2
    assert calls[0][1] == {
        "function_name": "route_input_compute",
        "is_first_layer": True,
        "is_last_layer": False,
        "participant_modules": (
            target.compute_layer,
            target.moe_layer.pre_mlp_layernorm,
            target.moe_layer.mlp,
        ),
    }
    assert calls[1][1] == {
        "function_name": "output_shared",
        "is_first_layer": False,
        "is_last_layer": True,
        "participant_modules": [
            target.compute_layer,
            target.moe_layer.pre_mlp_layernorm,
            target.moe_layer.mlp.shared_experts,
            target.shortcut_post_norm,
        ],
    }
    assert target.cudagraph_manager is target._graph_state.output_manager
    assert not hasattr(target, "cudagraph_manager_route_input_compute")
    assert not hasattr(target, "cudagraph_manager_output_shared_postprocess")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_persistent_buffer_reuses_storage_and_validates_metadata(cuda_graph_mempool):
    persistent = PersistentBuffer("test", requires_grad=True, detach_on_reuse=True)
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
def test_persistent_slot_owns_one_persistent_combine_buffer(cuda_graph_mempool):
    slot = _PersistentSlot(0)

    template = torch.empty_strided((8, 16), (16, 1), device="cuda")
    buffer = slot.acquire_combined_output(template)
    buffer_ptr = buffer.data_ptr()

    # Reuse detaches the prior autograd wrapper, but the graph-visible allocation is fixed.
    reused_buffer = slot.acquire_combined_output(template)
    assert reused_buffer.data_ptr() == buffer_ptr
    assert slot.combined_output_buffer.tensor.data_ptr() == buffer_ptr

    with pytest.raises(AssertionError, match="metadata changed"):
        slot.acquire_combined_output(torch.empty(4, 16, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_state_rotates_persistent_slots():
    graph_state = _GraphState(3)

    acquired = [graph_state.acquire_slot() for _ in range(5)]

    assert [index for index, _ in acquired] == [0, 1, 2, 0, 1]
    assert [slot for _, slot in acquired] == [
        graph_state.slots[0],
        graph_state.slots[1],
        graph_state.slots[2],
        graph_state.slots[0],
        graph_state.slots[1],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_dispatch_bridge_publishes_both_route_gradients(monkeypatch):
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
    module = SimpleNamespace(dispatch=dispatcher.token_dispatch, token_dispatcher=dispatcher)
    block = object.__new__(ShortcutMoEBlock)
    block.moe_layer = SimpleNamespace(mlp=module)
    backward_dependency = torch.zeros((), device="cuda", requires_grad=True)

    monkeypatch.setattr(
        ShortcutMoEBlock,
        "_parallel_streams",
        {torch.cuda.current_device(): dispatch_stream},
    )
    dispatch_output = block._launch_dispatch_async(
        route_input,
        route_probs,
        route_grad_buffers=(route_input_grad_buffer, route_probs_grad_buffer),
        route_grad_ready_event=route_grad_ready_event,
        backward_dependency=backward_dependency,
    )
    dispatched_input, dispatched_probs = block._wait_dispatch(dispatch_output)
    expert_input, _, expert_probs = dispatcher.dispatch_postprocess(
        dispatched_input, dispatched_probs
    )

    assert manager.dispatch_token_probs is not route_probs
    assert manager.dispatch_token_probs.data_ptr() == route_probs.data_ptr()
    assert manager.dispatched_probs is dispatched_probs
    assert expert_probs is dispatched_probs

    grad_input = torch.randn_like(expert_input)
    grad_probs = torch.randn_like(expert_probs)
    torch.autograd.backward((expert_input, expert_probs), (grad_input, grad_probs))
    torch.cuda.synchronize()

    assert route_grad_ready_event.query()
    torch.testing.assert_close(route_input_grad_buffer, 2 * grad_input)
    torch.testing.assert_close(route_probs_grad_buffer, 3 * grad_probs)
    assert route_input.grad is None
    assert route_probs.grad is None
    assert backward_dependency.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_dispatch_bridge_zero_fills_an_unused_private_input():
    route_input = torch.randn(8, 4, device="cuda", requires_grad=True)
    route_probs = torch.randn_like(route_input, requires_grad=True)
    route_input_grad_buffer = torch.empty_like(route_input)
    route_probs_grad_buffer = torch.empty_like(route_probs)
    dispatch_stream = torch.cuda.Stream()
    route_grad_ready_event = torch.cuda.Event(external=True)
    # Both outputs remain differentiable, but this dispatch variant does not consume route_probs.
    module = SimpleNamespace(dispatch=lambda inputs, probs: (2 * inputs, inputs + 0))
    backward_dependency = torch.zeros((), device="cuda", requires_grad=True)

    dispatch_stream.wait_stream(torch.cuda.current_stream())
    dispatched_input, dispatched_probs = AsyncDispatchToPersistentGradBuffers.apply(
        route_input,
        route_probs,
        backward_dependency,
        module,
        dispatch_stream,
        route_input_grad_buffer,
        route_probs_grad_buffer,
        route_grad_ready_event,
    )
    torch.cuda.current_stream().wait_stream(dispatch_stream)

    grad_input = torch.randn_like(dispatched_input)
    grad_probs = torch.randn_like(dispatched_probs)
    torch.autograd.backward((dispatched_input, dispatched_probs), (grad_input, grad_probs))
    torch.cuda.synchronize()

    assert route_grad_ready_event.query()
    torch.testing.assert_close(route_input_grad_buffer, 2 * grad_input + grad_probs)
    torch.testing.assert_close(route_probs_grad_buffer, torch.zeros_like(route_probs))
    assert backward_dependency.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_dispatch_bridge_orders_paired_backward_after_dispatch_backward():
    calls = []

    class RecordDispatchBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            return 2 * tensor

        @staticmethod
        def backward(ctx, grad):
            calls.append("dispatch_backward")
            return 2 * grad

    class RecordPairedBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            return tensor.clone()

        @staticmethod
        def backward(ctx, grad):
            calls.append("paired_backward")
            return grad

    route_input = torch.randn(8, 4, device="cuda", requires_grad=True)
    route_probs = torch.randn_like(route_input, requires_grad=True)
    paired_leaf = torch.randn((), device="cuda", requires_grad=True)
    backward_dependency = RecordPairedBackward.apply(paired_leaf)
    route_input_grad_buffer = torch.empty_like(route_input)
    route_probs_grad_buffer = torch.empty_like(route_probs)
    dispatch_stream = torch.cuda.Stream()
    route_grad_ready_event = torch.cuda.Event(external=True)
    module = SimpleNamespace(
        dispatch=lambda inputs, probs: (RecordDispatchBackward.apply(inputs), 3 * probs)
    )

    dispatch_stream.wait_stream(torch.cuda.current_stream())
    dispatched_input, dispatched_probs = AsyncDispatchToPersistentGradBuffers.apply(
        route_input,
        route_probs,
        backward_dependency,
        module,
        dispatch_stream,
        route_input_grad_buffer,
        route_probs_grad_buffer,
        route_grad_ready_event,
    )
    torch.cuda.current_stream().wait_stream(dispatch_stream)

    # The ordinary paired path supplies its real gradient. The bridge contributes None,
    # but its edge prevents the paired node from running until dispatch backward returns.
    loss = dispatched_input.sum() + dispatched_probs.sum() + backward_dependency
    loss.backward()
    torch.cuda.synchronize()

    assert calls == ["dispatch_backward", "paired_backward"]
    assert route_grad_ready_event.query()


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
    output = AsyncCombineToPersistentBuffer.apply(
        source, module, combine_stream, persistent_output_factory, ready_event, grad_ready_event
    )
    assert output.data_ptr() == destination.data_ptr()

    torch.cuda.current_stream().wait_event(ready_event)
    output = RecordCombineGradReady.apply(output, grad_ready_event)
    grad = torch.randn_like(output)
    output.backward(grad)
    torch.cuda.synchronize()
    assert ready_event.query()
    assert grad_ready_event.query()
    torch.testing.assert_close(source.grad, 2 * grad)


def test_eager_overlap_produces_gradients(monkeypatch):
    config = TransformerConfig(num_layers=2, hidden_size=4, num_attention_heads=1)
    config.moe_shortcut_parallel = True

    class FakeCompute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 1
            self.is_first_layer = True
            self.is_last_layer = False
            self.input_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.output_scale = torch.nn.Parameter(torch.tensor(3.0))

        def forward_pre_output_proj(self, hidden_states, **kwargs):
            return hidden_states * self.input_scale, hidden_states

        def forward_output_proj(self, projected, residual, **kwargs):
            return projected * self.output_scale

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tp_group = None
            self.dispatch_scale = torch.nn.Parameter(torch.tensor(5.0))
            self.expert_scale = torch.nn.Parameter(torch.tensor(7.0))
            self.combine_scale = torch.nn.Parameter(torch.tensor(11.0))

        def dispatch(self, route_input, route_probs):
            return route_input * self.dispatch_scale, route_probs

        def routed_experts_compute(self, dispatched_input, dispatched_probs):
            return dispatched_input + dispatched_probs * self.expert_scale, None

        def combine(self, routed_output):
            return routed_output * self.combine_scale

        def wait_combine(self, combined_output):
            return combined_output

        def postprocess(self, combined_output, shared_expert_output):
            return combined_output + shared_expert_output

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 2
            self.is_first_layer = False
            self.is_last_layer = True
            self.pre_mlp_layernorm = torch.nn.Identity()
            self.route_scale = torch.nn.Parameter(torch.tensor(13.0))
            self.prob_scale = torch.nn.Parameter(torch.tensor(17.0))
            self.shared_scale = torch.nn.Parameter(torch.tensor(19.0))
            self.mlp = FakeMLP()

        def shortcut_route_preprocess(self, shortcut_hidden, padding_mask=None):
            return shortcut_hidden * self.route_scale, shortcut_hidden * self.prob_scale

        def launch_dispatch(self, route_input, route_probs, ready_event):
            return self.mlp.dispatch(route_input, route_probs)

        def wait_dispatch_and_launch_combine(self, dispatch_output):
            dispatched_input, dispatched_probs = dispatch_output
            routed_output, _ = self.mlp.routed_experts_compute(
                dispatched_input, dispatched_probs
            )
            return self.mlp.combine(routed_output)

        def _shortcut_shared_experts(self, hidden_states):
            return hidden_states * self.shared_scale

        def _forward_post_mlp(self, output_with_bias, residual):
            return output_with_bias[0] + residual

    @contextmanager
    def quant_context_factory(config, layer_number):
        yield

    def run():
        compute = FakeCompute()
        moe = FakeMoE()
        block = ShortcutMoEBlock(compute, moe, enable_cudagraph=False)
        monkeypatch.setattr(block, "_shortcut_route_preprocess", moe.shortcut_route_preprocess)
        monkeypatch.setattr(block, "_shortcut_shared_experts", moe._shortcut_shared_experts)
        monkeypatch.setattr(block, "_launch_dispatch_async", moe.launch_dispatch)
        monkeypatch.setattr(
            block,
            "_wait_dispatch_and_launch_combine",
            moe.wait_dispatch_and_launch_combine,
        )
        monkeypatch.setattr(block, "_wait_combine", moe.mlp.wait_combine)

        hidden_states = torch.arange(1.0, 5.0, requires_grad=True)
        output = block(
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

    monkeypatch.setattr(
        torch.cuda, "Event", lambda *args, **kwargs: SimpleNamespace(record=lambda stream: None)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    output, gradients = run()

    assert output.isfinite().all()
    assert all(gradient is not None and gradient.isfinite().all() for gradient in gradients)


@pytest.mark.parametrize(
    "enable_cudagraph", [False, True], ids=["eager-overlap", "graph-overlap"]
)
def test_shortcut_block_selects_a2a_schedule(monkeypatch, enable_cudagraph):
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
    dispatcher_attr_output = object()
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
        calls.append("forward_pre_output_proj")
        return compute_dependency, compute_aux

    def route_input_compute(actual_hidden, **kwargs):
        assert actual_hidden is hidden_states
        route_preprocess(shortcut_hidden=actual_hidden, padding_mask=kwargs["padding_mask"])
        calls.append("record_event")
        compute_kwargs = dict(kwargs)
        compute_kwargs.pop("persistent_slot", None)
        paired_state = input_projection(hidden_states=actual_hidden, **compute_kwargs)
        if enable_cudagraph:
            return (*paired_state, dispatcher_attr_output)
        return permuted_input, dispatch_probs, paired_state

    route_ready_event = SimpleNamespace(record=lambda stream: calls.append("record_event"))

    combine_ready_event = object()
    combine_grad_ready_event = object()

    def wait_dispatch_and_launch_combine(
        actual_dispatch_output,
        persistent_output_factory=None,
        ready_event=None,
        grad_ready_event=None,
    ):
        assert actual_dispatch_output == (dispatched_input, dispatched_probs)
        calls.append("wait_dispatch_and_launch_combine")
        if enable_cudagraph:
            assert callable(persistent_output_factory)
            assert ready_event is combine_ready_event
            assert grad_ready_event is combine_grad_ready_event
            return combined_output
        assert persistent_output_factory is None
        assert ready_event is None
        assert grad_ready_event is None
        return combined_output

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

    def wait_combine(actual_combined_output):
        assert actual_combined_output is combined_output
        calls.append("wait_combine")
        return actual_combined_output

    def output_shared(
        *paired_state,
        combined_output,
        inference_context,
        padding_mask,
        persistent_slot=0,
    ):
        assert persistent_slot == 0
        assert paired_state == (compute_dependency, compute_aux)
        assert inference_context is expected_inference_context
        assert padding_mask is expected_padding_mask
        calls.append("output_projection")
        calls.append("shared_experts")
        assert combined_output is expected_combined_output
        if not enable_cudagraph:
            combined_output = wait_combine(combined_output)
        return postprocess(projected_hidden, combined_output, shared_expert_output)

    def postprocess(actual_hidden, actual_combined, actual_shared):
        assert actual_hidden is projected_hidden
        assert actual_combined is combined_output
        assert actual_shared is shared_expert_output
        calls.append("postprocess")
        return final_output

    moe_layer = SimpleNamespace(
        layer_number=8,
        _local_cudagraph_attr_names=("dispatcher_attr",),
        mlp=SimpleNamespace(
            dispatch=dispatch, routed_experts_compute=routed_experts, combine=combine
        ),
        shortcut_route_preprocess=route_preprocess,
    )
    compute_layer = SimpleNamespace(forward_pre_output_proj=input_projection)

    block = object.__new__(ShortcutMoEBlock)
    torch.nn.Module.__init__(block)
    block.compute_layer = compute_layer
    block.moe_layer = moe_layer
    block.enable_cudagraph = enable_cudagraph
    block.route_ready_event = route_ready_event
    monkeypatch.setattr(block, "route_input_compute", route_input_compute)
    monkeypatch.setattr(block, "output_shared", output_shared)
    monkeypatch.setattr(block, "_postprocess", postprocess)
    monkeypatch.setattr(
        block,
        "_wait_dispatch_and_launch_combine",
        wait_dispatch_and_launch_combine,
    )
    monkeypatch.setattr(block, "_wait_combine", wait_combine)
    output_slot = SimpleNamespace(
        acquire_combined_output=lambda tensor: tensor,
        combine_ready_event=combine_ready_event,
        combine_grad_ready_event=combine_grad_ready_event,
    )

    block._graph_state = (
        SimpleNamespace(acquire_slot=lambda: (0, output_slot)) if enable_cudagraph else None
    )

    def launch_unified_dispatch(route_outputs, persistent_slot):
        assert persistent_slot == 0
        if enable_cudagraph:
            assert route_outputs == (compute_dependency, compute_aux, dispatcher_attr_output)
        else:
            assert route_outputs == (
                permuted_input,
                dispatch_probs,
                (compute_dependency, compute_aux),
            )
        calls.append("launch_dispatch")
        return (compute_dependency, compute_aux), (dispatched_input, dispatched_probs)

    monkeypatch.setattr(block, "_launch_dispatch", launch_unified_dispatch)

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
    if enable_cudagraph:
        expected_calls = [
            "route",
            "prepare_dispatch",
            "record_event",
            "forward_pre_output_proj",
            "launch_dispatch",
            "wait_dispatch_and_launch_combine",
            "output_projection",
            "shared_experts",
            "postprocess",
        ]
    else:
        expected_calls = [
            "route",
            "prepare_dispatch",
            "record_event",
            "forward_pre_output_proj",
            "launch_dispatch",
            "wait_dispatch_and_launch_combine",
            "output_projection",
            "shared_experts",
            "wait_combine",
            "postprocess",
        ]
    assert calls == expected_calls
    assert quant_contexts == ["enter", "exit"] * 2


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
