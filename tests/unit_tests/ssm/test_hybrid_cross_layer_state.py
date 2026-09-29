# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the cross-layer contract with shared memory independent of CSA2.

CPU modules use the production Hybrid/Transformer layers and checkpoint engine.
These tests cover autograd and replay contracts, not distributed or CUDA kernels.
"""

from dataclasses import dataclass, replace

import pytest
import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid import hybrid_stack_adapter as boundary_runtime
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import get_hybrid_state_components
from megatron.core.models.hybrid.hybrid_state import HybridStateDeclaration
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import random as checkpoint_runtime
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.state_boundary import (
    BoundarySchema,
    StateRegion,
    TensorField,
    TensorSchema,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


@dataclass
class _MemoryState:
    memory: torch.Tensor | None = None
    last_layer: int = 0

    def attention_kwargs(self):
        return {"memory_state": self}

    def recompute_boundary_tensors(self):
        return () if self.memory is None else (self.memory,)


class _MemoryDeclaration(HybridStateDeclaration):
    """Native memory fields and codec using the same contract as CSA2 and mHC."""

    context_attribute = "memory_owner"

    def __init__(self, config, *, layer_type_list, pp_layer_offset, **kwargs):
        self.config = config
        self.pattern, self.offset = layer_type_list, pp_layer_offset

    def initial_state(self, hidden, packed_seq_params):
        return _MemoryState()

    def layer_kwargs(self, layer, state):
        if isinstance(layer, TransformerLayer):
            return {"attention": state.attention_kwargs()}
        return {}

    def recompute_boundary_tensors(self, state):
        return state.recompute_boundary_tensors()

    @staticmethod
    def _key(last_layer):
        source = 4 if last_layer >= 4 else int(last_layer > 0)
        return f"memory/shared:L{source}"

    def _field(self, hidden, last_layer):
        return TensorField(
            self._key(last_layer),
            (*hidden.shape[:2], self.config.hidden_size),
            self.config.params_dtype,
            "sbhd",
            True,
            present=last_layer > 0,
        )

    def checkpoint_region(self, start, end, hidden, state):
        readers = [self.offset + i + 1 for i in range(start, end) if self.pattern[i] != "-"]
        last_layer = readers[-1] if readers else state.last_layer
        return StateRegion(
            BoundarySchema(
                f"memory/checkpoint:{start}:{end}",
                (self._field(hidden, state.last_layer),),
                (self._field(hidden, last_layer),),
            ),
            self,
            (state.last_layer,),
            (last_layer,),
        )

    def export(self, state, fields):
        assert len(fields) == 1 and fields[0].key == self._key(state.last_layer)
        assert fields[0].present == (state.memory is not None)
        tensors = (state.memory,) if fields[0].present else ()
        TensorSchema(fields).validate(tensors)
        return tensors

    def restore(self, fields, tensors, metadata):
        TensorSchema(fields).validate(tensors)
        state = _MemoryState(tensors[0] if tensors else None, metadata[0])
        self.export(state, fields)
        return state


class _Norm(nn.LayerNorm):
    def __init__(self, config, hidden_size, eps, **kwargs):
        super().__init__(hidden_size, eps=eps, dtype=config.params_dtype)


class _MemoryAttention(nn.Module):
    def __init__(self, config, layer_number, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, *, memory_state, **kwargs):
        assert memory_state.last_layer < self.layer_number
        if self.layer_number == 1:
            assert memory_state.memory is None
        projected = self.projection(hidden_states)
        if self.layer_number in (1, 4):
            memory_state.memory = projected.sin()
        memory_state.last_layer = self.layer_number
        return projected.tanh() + 0.2 * memory_state.memory.square(), None


class _MLPMemoryDeclaration(_MemoryDeclaration):
    context_attribute = "mlp_owner"

    @staticmethod
    def _key(last_layer):
        return "mlp/" + _MemoryDeclaration._key(last_layer)

    def layer_kwargs(self, layer, state):
        return {"mlp": state.attention_kwargs()}


def _config(mhc=False):
    return TransformerConfig(
        num_layers=5,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.float32,
        enable_hyper_connections=mhc,
        num_residual_streams=2,
        use_fused_mhc=False,
        bias_dropout_fusion=False,
    )


def _stack(config):
    groups = ProcessGroupCollection()
    groups.tp = groups.pp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    stack = HybridStack(
        config,
        HybridStackSubmodules(
            state_components=get_hybrid_state_components(
                config, (_MemoryDeclaration, _MLPMemoryDeclaration)
            ),
            attention_layer=ModuleSpec(
                TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=_Norm,
                    self_attention=_MemoryAttention,
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=_Norm,
                    mlp=_MemoryAttention,
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
        ),
        layer_type_list=list("*****"),
        post_layer_norm=False,
        pg_collection=groups,
    )
    finalize = stack.forward_adapter.finalize_forward

    def observe(hidden, params, context):
        return (
            finalize(hidden, params, context),
            context.memory_owner.memory,
            context.mlp_owner.memory,
        )

    stack.forward_adapter.finalize_forward = observe
    return stack


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(checkpoint_runtime, "_get_cuda_rng_state", lambda **kwargs: None)
        monkeypatch.setattr(checkpoint_runtime, "_set_cuda_rng_state", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda message: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)


def _assert_grad(actual, expected):
    if expected is None:
        assert actual is None or torch.count_nonzero(actual) == 0
    else:
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("mhc", [False, True])
@pytest.mark.parametrize("mode", ["full", "layernorm"])
@pytest.mark.parametrize("objective", ["side", "both"])
def test_attention_and_mlp_bind_the_same_native_name_independently(mhc, mode, objective):
    """Ordinary TransformerLayer routes two memory_state arguments to distinct consumers."""
    config = _config(mhc)

    reference = _stack(config)
    runtime = (
        replace(
            config, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2
        )
        if mode == "full"
        else replace(config, recompute_granularity="selective", recompute_modules=["layernorm"])
    )
    actual = _stack(runtime)
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(3, 1, config.hidden_size, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    output, expected = actual(x, None), reference(rx, None)
    torch.testing.assert_close(output, expected)
    for tensors in (output, expected):
        sum(t.square().sum() for t in (tensors if objective == "both" else tensors[1:])).backward()
    _assert_grad(x.grad, rx.grad)
    for p, q in zip(actual.parameters(), reference.parameters()):
        _assert_grad(p.grad, q.grad)


def test_components_reject_duplicate_native_arguments_within_one_consumer():
    stack = _stack(_config())
    adapter = stack.forward_adapter
    adapter.components = (adapter.components[0], adapter.components[0])
    context = boundary_runtime.HybridStackForwardContext(memory_owner=_MemoryState())
    with pytest.raises(ValueError, match="duplicate attention arguments.*memory_state"):
        adapter.layer_kwargs(stack.layers[0], context)


@pytest.mark.parametrize(
    "backend,recompute,message",
    [
        ("local", None, "Hybrid host:.*Transformer Engine CUDA Graphs only"),
        ("transformer_engine", None, "must declare graph_state"),
        ("transformer_engine", "full", "Hybrid TE backend:.*full recompute"),
    ],
)
def test_state_graph_setup_rejects_missing_runtime_capabilities(backend, recompute, message):
    config = _config()
    config.cuda_graph_impl = backend
    config.recompute_granularity = recompute
    with pytest.raises(ValueError, match=message):
        _stack(config)


@pytest.mark.parametrize("mode", ["mlp_recompute", "mlp_chunking"])
def test_stateful_mlp_rejects_undeclared_internal_boundaries(mode):
    stack = _stack(_config())
    layer = stack.layers[0]
    layer.mlp = _MemoryAttention(stack.config, layer_number=1)
    if mode == "mlp_recompute":
        layer.recompute_mlp = True
    else:
        layer.config.mlp_chunks_for_training = 2
    stack.forward_adapter.components = (
        _MLPMemoryDeclaration(stack.config, layer_type_list=["*"], pp_layer_offset=0),
    )
    context = boundary_runtime.HybridStackForwardContext(mlp_owner=_MemoryState())
    kwargs = stack.forward_adapter.layer_kwargs(layer, context)
    with pytest.raises(ValueError, match="Stateful MLP arguments require.*state boundary"):
        layer._forward_mlp_output_with_bias(torch.randn(3, 1, stack.config.hidden_size), **kwargs)
