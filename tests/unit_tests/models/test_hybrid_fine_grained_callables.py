# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core.models.common.utils as node_utils
import megatron.core.models.hybrid.fine_grained_callables as hybrid_callables
import megatron.core.pipeline_parallel.utils as schedule_utils
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.moe.moe_layer import MoELayer


@pytest.fixture
def cpu_streams(monkeypatch):
    """Exercise real schedule-node autograd while replacing only CUDA stream operations."""
    streams = SimpleNamespace(comp=Mock(), comm=Mock())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: streams.comp)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    monkeypatch.setattr(schedule_utils, "get_comm_stream", lambda: streams.comm)
    monkeypatch.setattr(hybrid_callables, "get_comm_stream", lambda: streams.comm)
    for module in (node_utils, schedule_utils):
        monkeypatch.setattr(module, "nvtx_range_push", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "nvtx_range_pop", lambda *args, **kwargs: None)
    return streams


def _config(**kwargs):
    values = dict(
        fp8=None,
        fp4=None,
        fp32_residual_connection=False,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="ncclep",
        moe_ncclep_zero_copy=False,
        moe_latent_size=None,
        overlap_moe_expert_parallel_comm=True,
        cuda_graph_modules=[],
        bias_dropout_fusion=False,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def _moe_layer(config, *, shared=False):
    mlp = SimpleNamespace(
        config=config,
        num_local_experts=1,
        use_shared_expert=shared,
        shared_expert_overlap=False,
        experts=torch.nn.Linear(2, 2, bias=False),
        shared_experts=torch.nn.Linear(2, 2, bias=False) if shared else None,
        backward_dw=Mock(),
    )
    return SimpleNamespace(config=config, mlp=mlp, recompute_pre_mlp_layernorm=False)


def _chunk_state():
    return SimpleNamespace(
        model=SimpleNamespace(decoder=SimpleNamespace(final_norm=None)),
        padding_mask=None,
        attention_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    )


def _node(layer, layer_state, function, name):
    return hybrid_callables.HybridStackNode(
        stream=object(),
        event=Mock(),
        layer_state=layer_state,
        chunk_state=_chunk_state(),
        submodule=function,
        name=name,
        extra_args={"config": layer.config, "is_moe": True, "num_local_experts": 1},
    )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("latent", [False, True])
def test_delayed_wgrad_hooks_belong_to_the_slot_that_computes_them(cpu_streams, shared, latent):
    """Stale gradients must not cause the routed slot to accumulate shared-expert gradients."""
    events = []
    cpu_streams.comp.wait_stream.side_effect = lambda stream: events.append("wait_comm")

    class Expert(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.weight = torch.nn.Parameter(torch.ones(1))
            # A prior microbatch can leave a gradient before this slot runs.
            self.weight.grad = torch.ones_like(self.weight)
            self.weight.post_wgrad_grad_acc_hook = lambda: events.append(f"{name}:hook")

        def backward_dw(self):
            events.append(f"{self.name}:wgrad")

    layer = _moe_layer(_config(moe_latent_size=2 if latent else None), shared=shared)
    mlp = layer.mlp
    mlp.experts = Expert("routed")
    mlp.shared_experts = Expert("shared") if shared else None
    if latent:
        mlp.fc1_latent_proj = Expert("down")
        mlp.fc2_latent_proj = Expert("up")
    mlp.backward_dw = partial(MoELayer.backward_dw, mlp)
    _, backward_dw, _, _ = hybrid_callables.build_hybrid_stack_callables(layer, Symbols.MOE)

    def run_wgrad(modules):
        node = SimpleNamespace(
            delay_wgrad_compute=True,
            stream=object(),
            name="test slot",
            bwd_dw_callables=modules,
            post_wgrad_grad_acc_hooks=None,
            is_layer_first_node=False,
        )
        node_utils.TransformerLayerNode.backward_dw(node)
        assert node.bwd_dw_callables is None

    run_wgrad([backward_dw["mlp"]])
    routed_names = ["routed", "up"] if latent else ["routed"]
    assert events == (
        [f"{name}:wgrad" for name in routed_names]
        + (["wait_comm"] if latent else [])
        + [f"{name}:hook" for name in routed_names]
    )
    if latent:
        cpu_streams.comp.wait_stream.assert_called_once_with(cpu_streams.comm)
    else:
        cpu_streams.comp.wait_stream.assert_not_called()

    events.clear()
    pre_names = (["shared"] if shared else []) + (["down"] if latent else [])
    if pre_names:
        run_wgrad(backward_dw["pre_dispatch_computation"])
    else:
        assert "pre_dispatch_computation" not in backward_dw
    assert events == [f"{name}:wgrad" for name in pre_names] + [
        f"{name}:hook" for name in pre_names
    ]


@pytest.mark.parametrize(
    "symbol", [Symbols.ATTENTION, Symbols.DS_ATTENTION, Symbols.MLA, Symbols.GDN]
)
def test_attention_half_layer_forward_and_wgrad_are_scheduled(symbol):
    """All attention symbols, including MLA '+', preserve their forward and backward work."""
    backward_dw_wrapper = Mock()
    layer = SimpleNamespace(
        config=_config(),
        _forward_attention=Mock(
            side_effect=lambda hidden_states, **kwargs: (hidden_states.sin(), None)
        ),
        backward_dw_wrapper=backward_dw_wrapper,
        init_backward_dw_wrapper=Mock(),
    )
    forward, backward_dw, is_moe, _ = hybrid_callables.build_hybrid_stack_callables(layer, symbol)
    node = SimpleNamespace(chunk_state=_chunk_state(), is_mtp=False, is_last_layer=False)
    hidden_states = torch.tensor([0.2, 0.4], requires_grad=True)
    output = forward[0](node, hidden_states)
    output.sum().backward()

    torch.testing.assert_close(output, hidden_states.sin())
    torch.testing.assert_close(hidden_states.grad, hidden_states.cos())
    layer.init_backward_dw_wrapper.assert_called_once_with()
    assert backward_dw["pre_dispatch_computation"] == [backward_dw_wrapper]
    assert not is_moe


@pytest.mark.parametrize("zero_copy", [False, True])
def test_ncclep_probabilities_do_not_reconnect_schedule_graphs(cpu_streams, zero_copy):
    """Backward across all three real schedule nodes must match the unsplit computation."""
    layer = _moe_layer(_config(moe_ncclep_zero_copy=zero_copy))
    manager = SimpleNamespace(
        token_probs=None,
        dispatched_probs=None,
        get_number_of_tokens_per_expert=Mock(return_value=torch.tensor([2])),
        _zc_bwd_token_buf=torch.empty(2),
    )
    dispatch_grad_ptrs = []

    class Dispatch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tokens, probs):
            return tokens * 3, probs * 4

        @staticmethod
        def backward(ctx, token_grad, prob_grad):
            dispatch_grad_ptrs.append(token_grad.data_ptr())
            return token_grad * 3, prob_grad * 4

    def dispatch(tokens, probs):
        # Flex dispatch consumes the saved manager state, not its explicit probs argument.
        output, manager.dispatched_probs = Dispatch.apply(tokens, manager.token_probs)
        return output, manager.dispatched_probs

    layer.mlp.token_dispatcher = SimpleNamespace(_comm_manager=manager)
    layer.mlp.dispatch = dispatch
    layer.mlp.routed_experts_compute = lambda tokens, probs: (
        tokens * manager.dispatched_probs,
        None,
    )
    forward, _, _, _ = hybrid_callables.build_hybrid_stack_callables(layer, Symbols.MOE)

    def preprocess(node, hidden_states):
        manager.token_probs = hidden_states.square()
        return hidden_states * 2, manager.token_probs

    state = SimpleNamespace()
    pre_node = _node(layer, state, preprocess, "pre_dispatch_computation")
    dispatch_node = _node(layer, state, forward[1], "moe_dispatch")
    expert_node = _node(layer, state, forward[2], "mlp")
    hidden_states = torch.tensor([0.2, 0.4], requires_grad=True)
    output = expert_node.forward(dispatch_node.forward(pre_node.forward(hidden_states)))
    grad = expert_node.backward(torch.ones_like(output))
    grad = dispatch_node.backward(grad)
    grad = pre_node.backward(grad)

    torch.testing.assert_close(output, 24 * hidden_states.pow(3))
    torch.testing.assert_close(grad, 72 * hidden_states.square())
    assert state.tokens_per_expert is manager.get_number_of_tokens_per_expert.return_value
    if zero_copy:
        assert dispatch_grad_ptrs == [manager._zc_bwd_token_buf.data_ptr()]


@pytest.mark.parametrize("recompute", [False, True])
def test_norm_offload_uses_its_microbatch_manager_after_bda(cpu_streams, recompute):
    """Two in-flight microbatches keep distinct offload managers and single recompute hooks."""
    layer = _moe_layer(_config())
    layer.recompute_pre_mlp_layernorm = recompute
    layer.offload_mlp_norm = True
    layer.training = True
    layer.hidden_dropout = 0.0
    layer.bias_dropout_add_exec_handler = nullcontext
    layer.mlp_bda = lambda *args: lambda output, residual, dropout: output[0] + residual
    layer.mlp.shared_experts_compute = lambda hidden: None
    layer.mlp.route = lambda hidden, mask: (hidden, None)
    layer.mlp.preprocess = lambda hidden, probs, routing: (hidden, probs)
    layer.mlp.routed_experts_compute = lambda hidden, probs: (hidden * 3, None)
    layer.mlp.combine = lambda output: output
    layer.mlp.postprocess = lambda output, shared: output
    layer.mlp.token_dispatcher = SimpleNamespace(
        _comm_manager=SimpleNamespace(get_number_of_tokens_per_expert=lambda: torch.tensor([2]))
    )
    managers, checkpoints = [], []

    def normalize(hidden_states):
        manager = SimpleNamespace(group_offload=Mock(side_effect=lambda output, **kwargs: output))
        layer.mlp_norm_manager = manager
        managers.append(manager)
        layer.pre_mlp_norm_checkpoint = Mock()
        checkpoints.append(layer.pre_mlp_norm_checkpoint)
        return hidden_states * 2

    layer._forward_pre_mlp_layernorm = normalize
    microbatches = []
    for value in (1.0, 2.0):
        hidden_states = torch.full((2,), value, requires_grad=True)
        node = SimpleNamespace(
            layer_state=SimpleNamespace(),
            chunk_state=_chunk_state(),
            detach=lambda tensor: tensor.detach(),
            is_mtp=False,
            is_last_layer=False,
        )
        tokens, probs = hybrid_callables._run_moe_preprocess(layer, node, hidden_states)
        node.layer_state.dispatched_probs = probs
        expert_output = hybrid_callables._run_moe_experts(layer, node, tokens)
        microbatches.append((node, hidden_states, expert_output))
        assert layer.mlp_norm_manager is None

    for index, (node, hidden_states, expert_output) in enumerate(microbatches):
        residual = node.layer_state.residual
        output = hybrid_callables._run_moe_combine(layer, node, expert_output)
        torch.testing.assert_close(output, hidden_states * 7)
        managers[index].group_offload.assert_called_once_with(
            output, forced_released_tensors=[residual]
        )
        assert node.layer_state.mlp_norm_manager is None
        assert node.layer_state.residual is None
        if recompute:
            checkpoints[index].discard_output_and_register_recompute.assert_called_once_with(
                expert_output
            )
        else:
            checkpoints[index].discard_output_and_register_recompute.assert_not_called()
