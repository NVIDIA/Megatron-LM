# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the cross-layer contract with shared memory independent of CSA2.

CPU modules use the production Hybrid/Transformer layers and checkpoint engine.
These tests cover autograd and replay contracts, not distributed or CUDA kernels.
"""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import MethodType

import pytest
import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid import hybrid_stack_adapter as boundary_runtime
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import get_hybrid_state_components
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateAdapter
from megatron.core.models.hybrid.hybrid_state import HybridStateDeclaration
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import random as checkpoint_runtime
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.state_boundary import (
    BoundarySchema,
    StateRegion,
    TensorField,
    TensorMappingCodec,
    TensorSchema,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.ssm.test_hybrid_state_adapter import cpu_graph_slots as cpu_graph_slots
from tests.unit_tests.tensor_parallel.test_boundary_checkpoint import (
    cpu_te_checkpoint as cpu_te_checkpoint,
)


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
        return {"layer": {"memory_state": state}} if isinstance(layer, _MemoryLayer) else {}

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


class _MemoryLayer(_MemoryAttention):
    def forward(
        self, hidden_states, attention_mask, inference_context, packed_seq_params, *, memory_state
    ):
        # Deliberately narrow: full replay must use the same kwargs as eager.
        delta, _ = super().forward(hidden_states, memory_state=memory_state)
        return hidden_states + delta


class _MLPMemoryDeclaration(_MemoryDeclaration):
    context_attribute = "mlp_owner"

    @staticmethod
    def _key(last_layer):
        return "mlp/" + _MemoryDeclaration._key(last_layer)

    def layer_kwargs(self, layer, state):
        return {"mlp": state.attention_kwargs()}


class _PlainLayer(nn.Module):
    def __init__(self, config, layer_number, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask, inference_context, packed_seq_params):
        # Unrelated layers must retain their existing interface.
        return hidden_states * 0.9


def _config(kind):
    return TransformerConfig(
        num_layers=5,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.float32,
        enable_hyper_connections=kind.startswith("mhc_"),
        mhc_single_pass=kind.startswith("mhc_single_"),
        num_residual_streams=2,
        use_fused_mhc=False,
        bias_dropout_fusion=False,
    )


def _stack(config, kind):
    groups = ProcessGroupCollection()
    groups.tp = groups.pp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    submodules = HybridStackSubmodules(
        state_components=get_hybrid_state_components(config, (_MemoryDeclaration,)),
        mamba_layer=_MemoryLayer,
        mlp_layer=_PlainLayer,
        attention_layer=ModuleSpec(
            TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=_Norm,
                self_attention=_MemoryAttention,
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
    )
    # A layer that does not consume memory separates its producer and consumers.
    stack = HybridStack(
        config,
        submodules,
        layer_type_list=list(
            "*****"
            if kind.startswith("mhc_single_")
            else "*-***" if kind.endswith("attention") else "M-MMM"
        ),
        post_layer_norm=False,
        pg_collection=groups,
    )
    finalize = stack.forward_adapter.finalize_forward

    def observe_memory(output, packed_seq_params, context):
        # Expose a native side output only to test independent backward roots.
        # Model state initialization and checkpoint replay use the shared executor.
        return finalize(output, packed_seq_params, context), context.memory_owner.memory

    stack.forward_adapter.finalize_forward = observe_memory
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
    config = _config("mhc_attention" if mhc else "attention")
    groups = ProcessGroupCollection()
    groups.tp = groups.pp = groups.cp = torch.distributed.ProcessGroup(0, 1)

    def make_stack(config):
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

    reference = make_stack(config)
    runtime = (
        replace(
            config, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2
        )
        if mode == "full"
        else replace(config, recompute_granularity="selective", recompute_modules=["layernorm"])
    )
    actual = make_stack(runtime)
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


@pytest.mark.parametrize("stateful_attention", [False, True])
@pytest.mark.parametrize("objective", ["side", "both"])
@pytest.mark.parametrize(
    "scope,mhc",
    [
        ("attn", None),
        ("mlp", None),
        ("moe_router", None),
        ("moe_router", "single"),
        ("attn", "standard"),
        ("attn", "attention_split"),
        ("attn+moe", None),
        ("attn+mlp", None),
    ],
)
def test_partial_graph_preserves_eager_state(
    cpu_graph_slots, stateful_attention, objective, scope, mhc
):
    """Graph publication and the eager MLP share the invocation's original state."""
    from megatron.core.transformer.cuda_graphs import (
        _layer_is_graphable,
        _set_capture_end,
        _set_capture_start,
        is_graph_capturing,
    )
    from megatron.core.transformer.enums import CudaGraphModule
    from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.module import GraphableMegatronModule
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.moe_utils import MoECudaGraphTensorStore
    from tests.unit_tests.ssm.test_hybrid_state_adapter import _Attention

    if mhc == "single" and stateful_attention:
        pytest.skip("Single-pass mHC uses separate attention and FFN layers")
    if mhc == "attention_split" and not stateful_attention:
        pytest.skip("The explicit split case exercises a stateful attention consumer")

    class GraphMemory(_MemoryDeclaration):
        consumer = "attention"

        def layer_kwargs(self, layer, state):
            bindings = {self.consumer: {"memory_state": state}}
            if self.consumer == "attention" and scope != "attn" and mhc is None:
                # Eager attention's context is not an output of the MLP graph.
                bindings["layer"] = {"context": state.memory}
            return bindings

        def graph_state(self, layer, symbol, region):
            if self.consumer not in region.branches:
                return None
            return type(self)(
                self.config, layer_type_list=[symbol], pp_layer_offset=layer.layer_number - 1
            )

        def region(self, hidden):
            return StateRegion(
                BoundarySchema(
                    "memory/graph",
                    (self._field(hidden, self.offset),),
                    (self._field(hidden, self.offset + 1),),
                ),
                self,
                (self.offset,),
                (self.offset + 1,),
            )

        def get_static_inputs(self, inputs):
            owned = {}
            if self.offset:
                hidden = inputs["hidden_states"]
                owned[self.consumer + "_graph_memory"] = hidden.new_ones(
                    (*hidden.shape[:2], self.config.hidden_size)
                )
            return owned

        def restore_inputs(self, hidden, kwargs):
            region = self.region(hidden)
            tensors = (kwargs[self.consumer + "_graph_memory"],) if self.offset else ()
            return region, self.restore(region.schema.inputs, tensors, region.input_metadata)

        def prepare_replay(self, hidden, kwargs):
            region = self.region(hidden)
            state = kwargs["cross_layer_state"].branch_kwargs(self.consumer)["memory_state"]
            tensors = self.export(state, region.schema.inputs)
            owned = {self.consumer + "_graph_memory": tensors[0]} if tensors else {}

            def publish(result, restored):
                state.memory, state.last_layer = restored.memory, restored.last_layer
                return result

            return region, owned, publish

    class EagerMLPMemory(GraphMemory):
        consumer, context_attribute = "mlp", "mlp_owner"
        _key = staticmethod(_MLPMemoryDeclaration._key)

    class MemoryMLP(MLP):
        def __init__(self, config, layer_number, **kwargs):
            nn.Module.__init__(self)
            self.config, self.layer_number = config, layer_number
            self.projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        forward = _MemoryAttention.forward

    class MemoryMoE(MoELayer):
        def __init__(self, config, layer_number, **kwargs):
            nn.Module.__init__(self)
            self.config, self.layer_number = config, layer_number
            self.projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.expert = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.router = nn.Identity()
            self.cudagraph_tensor_store = MoECudaGraphTensorStore()

        def set_layer_number(self, layer_number):
            self.layer_number = layer_number

        def forward(self, hidden, memory_state=None, **kwargs):
            if not self.cudagraph_tensor_store.is_empty():
                # The real TransformerLayer reaches this expert continuation only
                # after the graph region's memory output has been published.
                assert memory_state is not None and memory_state.last_layer == self.layer_number
                return self.expert(self.cudagraph_tensor_store.hidden_states).tanh(), None
            routed, _ = _MemoryAttention.forward(self, hidden, memory_state=memory_state)
            if is_graph_capturing():
                return routed, routed.sigmoid(), torch.ones_like(routed, dtype=torch.bool)
            return self.expert(routed).tanh(), None

    config = _config(
        "mhc_single_attention" if mhc == "single" else "mhc_attention" if mhc else "attention"
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.pp = groups.cp = torch.distributed.ProcessGroup(0, 1)

    def build(config):
        stack = HybridStack(
            config,
            HybridStackSubmodules(
                state_components=get_hybrid_state_components(
                    config,
                    (
                        *((GraphMemory,) if stateful_attention else ()),
                        *((EagerMLPMemory,) if mhc != "attention_split" else ()),
                    ),
                ),
                attention_layer=ModuleSpec(
                    TransformerLayer,
                    submodules=TransformerLayerSubmodules(
                        input_layernorm=IdentityOp if mhc == "single" else _Norm,
                        self_attention=(
                            IdentityOp
                            if mhc == "single"
                            else _MemoryAttention if stateful_attention else _Attention
                        ),
                        self_attn_bda=IdentityFuncOp if mhc == "single" else get_bias_dropout_add,
                        pre_mlp_layernorm=IdentityOp if mhc == "attention_split" else _Norm,
                        mlp=(
                            IdentityOp
                            if mhc == "attention_split"
                            else MemoryMoE if scope in ("moe_router", "attn+mlp") else MemoryMLP
                        ),
                        mlp_bda=(
                            IdentityFuncOp if mhc == "attention_split" else get_bias_dropout_add
                        ),
                    ),
                ),
            ),
            layer_type_list=list("*****"),
            post_layer_norm=False,
            pg_collection=groups,
        )
        finalize = stack.forward_adapter.finalize_forward

        def observe(hidden, params, context):
            state = context.memory_owner if mhc == "attention_split" else context.mlp_owner
            return finalize(hidden, params, context), state.memory

        stack.forward_adapter.finalize_forward = observe
        return stack

    reference = build(config)
    runtime = replace(config)
    runtime.cuda_graph_impl = "transformer_engine"
    runtime.cuda_graph_modules = [getattr(CudaGraphModule, part) for part in scope.split("+")]
    if mhc == "attention_split":
        runtime.mhc_recompute_attn_cuda_graph_split = True
        runtime.recompute_granularity = "selective"
        runtime.recompute_modules = ["mhc"]
        runtime.mhc_recompute_layer_num = 2
        runtime.is_hybrid_model = True
    actual = build(runtime)
    actual.load_state_dict(reference.state_dict())
    graph_calls = []
    for layer in actual.layers:
        assert _layer_is_graphable(layer, runtime)
        adapter = layer._te_cuda_graph_adapter
        samples = adapter.get_static_inputs(
            {
                "hidden_states": torch.ones(
                    3,
                    1,
                    config.hidden_size
                    * (config.num_residual_streams if mhc and mhc != "attention_split" else 1),
                )
            }
        )
        sample_hidden = samples.pop("hidden_states")
        adapter.finalize_sample_inputs((sample_hidden,), samples)
        if mhc == "attention_split":
            layer.set_te_cuda_graph_static_hidden_inputs(
                [sample_hidden.detach().clone() for _ in range(2)]
            )
        layer._get_te_cuda_graph_replay_args = MethodType(
            GraphableMegatronModule._get_te_cuda_graph_replay_args, layer
        )

        def graph(*args, layer=layer, adapter=adapter, **kwargs):
            kwargs.pop("is_first_microbatch", None)
            # No mutable state object is allowed through the actual TE graph boundary.
            assert all(
                value is None or isinstance(value, torch.Tensor) for value in kwargs.values()
            )
            graph_calls.append(layer.layer_number)
            if mhc == "attention_split":
                # CPU graph bodies retain their own activations; the actual arena
                # still validates and writes the per-microbatch static input.
                args = (args[0].clone(), *args[1:])
            if scope == "moe_router":
                _set_capture_start()
            try:
                return adapter.capture(layer._te_cuda_graph_capture, *args, **kwargs)
            finally:
                if scope == "moe_router":
                    _set_capture_end()

        layer.cuda_graphs = [graph, graph]

    batches = []
    for microbatch in range(2):
        for layer in actual.layers:
            layer.current_microbatch = microbatch
        x = torch.randn(3, 1, config.hidden_size, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        output, expected = actual(x, None), reference(rx, None)
        torch.testing.assert_close(output, expected)
        batches.append((x, rx, output, expected))
    # Two outstanding forwards must keep their graph-external states independent.
    for x, rx, output, expected in reversed(batches):
        for tensors in (output, expected):
            selected = tensors if objective == "both" else tensors[1:]
            sum(t.square().sum() for t in selected).backward()
        _assert_grad(x.grad, rx.grad)
    for p, q in zip(actual.parameters(), reference.parameters()):
        _assert_grad(p.grad, q.grad)
    assert graph_calls == list(range(1, 6)) * 2


def test_components_reject_duplicate_native_arguments_within_one_consumer():
    stack = _stack(_config("attention"), "attention")
    adapter = stack.forward_adapter
    adapter.components = (adapter.components[0], adapter.components[0])
    context = boundary_runtime.HybridStackForwardContext(memory_owner=_MemoryState())
    with pytest.raises(ValueError, match="duplicate attention arguments.*memory_state"):
        adapter.layer_kwargs(stack.layers[0], context)


@pytest.mark.parametrize("frozen_input", [False, True])
def test_receiving_stage_prepares_local_lookup_before_full_recompute(frozen_input):
    """Stage-local trainable inputs merge with wire state and bypass replay preparation."""
    from megatron.core.models.hybrid.hybrid_state import build_hybrid_state_pipeline_plan

    class LocalLookup(HybridStateDeclaration):
        context_attribute = "local_owner"

        def __init__(self, config, lookup, *, pp_layer_offset, **kwargs):
            self.lookup, self.offset = lookup, pp_layer_offset
            self.calls = []

        def initial_state(self, hidden, packed_seq_params):
            return {"wire": torch.zeros_like(hidden)}

        def prepare_local_state(self, hidden, state, inputs):
            assert "wire" in state
            if self.offset:
                assert state["wire"].requires_grad
            with pytest.raises(TypeError):
                inputs["input_ids"] = None
            self.calls.append(inputs["input_ids"])
            return dict(state, lookup=self.lookup(inputs["input_ids"]))

        def layer_kwargs(self, layer, state):
            return {"layer": {"memory_state": state}}

        def recompute_boundary_tensors(self, state):
            return tuple(state.values())

        def field(self, hidden, name, version):
            return TensorField(
                f"local/{name}:{version}", tuple(hidden.shape), hidden.dtype, "sbhd", True
            )

        def pipeline_region(self, boundary, hidden, params, *, requires_grad=True):
            field = replace(
                self.field(hidden, "wire", boundary.layer_offset), differentiable=requires_grad
            )
            return StateRegion(BoundarySchema("local/pp", (field,), (field,)), self)

        def checkpoint_region(self, start, end, hidden, state):
            prepared = self.field(hidden, "lookup", self.offset)
            start, end = start + self.offset, end + self.offset
            return StateRegion(
                BoundarySchema(
                    "local/checkpoint",
                    (self.field(hidden, "wire", start), prepared),
                    (self.field(hidden, "wire", end),),
                ),
                self,
                retained_fields=(prepared,),
            )

        def export(self, state, fields):
            values = tuple(state[f.key.split("/")[1].split(":")[0]] for f in fields)
            TensorSchema(fields).validate(values)
            return values

        def restore(self, fields, tensors, metadata):
            TensorSchema(fields).validate(tensors)
            return {f.key.split("/")[1].split(":")[0]: t for f, t in zip(fields, tensors)}

    class Consumer(nn.Module):
        def __init__(self, config, lookup, layer_number, **kwargs):
            super().__init__()
            self.layer_number = layer_number
            self.lookup = lookup  # Parameters are registered on the native model module.
            self.weight = nn.Parameter(torch.tensor(0.3))

        def forward(self, hidden_states, *, memory_state, **kwargs):
            memory_state["wire"] = (
                memory_state["wire"] + (hidden_states * self.weight + memory_state["lookup"]).tanh()
            )
            return hidden_states + memory_state["wire"]

    config = replace(
        _config("attention"),
        num_layers=2,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
    )
    plan = build_hybrid_state_pipeline_plan(config, "M|M", pp_size=2)
    groups = ProcessGroupCollection()
    groups.tp = groups.pp = groups.cp = torch.distributed.ProcessGroup(0, 1)

    def build(config):
        stacks = []
        for chunk in plan:
            lookup = nn.Embedding(13, config.hidden_size)
            stack = HybridStack(
                config,
                HybridStackSubmodules(
                    state_components=(ModuleSpec(LocalLookup, params={"lookup": lookup}),),
                    mamba_layer=ModuleSpec(Consumer, params={"lookup": lookup}),
                ),
                layer_type_list=list(chunk.layer_pattern),
                pp_layer_offset=chunk.layer_offset,
                pre_process=chunk.incoming is None,
                post_process=chunk.outgoing is None,
                post_layer_norm=False,
                pg_collection=groups,
            )
            stack.forward_adapter.configure_pipeline(chunk)
            stacks.append(stack)
        return stacks

    reference = build(config)
    actual = build(
        replace(
            config, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
        )
    )
    for left, right in zip(actual, reference):
        left.load_state_dict(right.state_dict())
    pairs = []
    for _ in range(2):
        ids = [torch.randint(0, 13, (3, 1)) for _ in plan]
        x = torch.randn(3, 1, config.hidden_size, requires_grad=not frozen_input)
        rx = x.detach().clone().requires_grad_(not frozen_input)
        outputs = []
        for stacks, hidden in ((actual, x), (reference, rx)):
            payload = stacks[0](hidden, None, input_ids=ids[0])
            assert all("lookup" not in field.field.key for field in payload.tensor_specs)
            stacks[1].set_input_tensor(payload)
            outputs.append(stacks[1](None, None, input_ids=ids[1]))
        torch.testing.assert_close(*outputs)
        pairs.append((x, rx, outputs))
    for x, rx, outputs in reversed(pairs):
        for output in outputs:
            output.square().sum().backward()
        _assert_grad(x.grad, rx.grad)
    for left, right in zip(actual, reference):
        assert len(left.forward_adapter.components[0].calls) == 2
        for p, q in zip(left.parameters(), right.parameters()):
            _assert_grad(p.grad, q.grad)


def test_graph_requirements_are_checked_before_capture():
    class NeedsOutputActivity(_MemoryDeclaration):
        def validate_runtime(self, runtime):
            if runtime.graph_region is not None and not runtime.graph_output_activity:
                raise ValueError("This objective requires independent graph output activity")

    stack = _stack(_config("attention"), "attention")
    stack.forward_adapter.components = (
        NeedsOutputActivity(stack.config, layer_type_list=["*"], pp_layer_offset=0),
    )
    stack.config.cuda_graph_impl = "transformer_engine"
    with pytest.raises(ValueError, match="requires independent graph output activity"):
        stack.forward_adapter.configure_cuda_graphs(stack.layers)
    assert not hasattr(stack.layers[0], "_te_cuda_graph_adapter")


@pytest.mark.parametrize(
    "backend,recompute,message",
    [
        ("local", None, "Hybrid host:.*Transformer Engine CUDA Graphs only"),
        ("transformer_engine", None, "must declare graph_state"),
        ("transformer_engine", "full", "Hybrid TE backend:.*full recompute"),
    ],
)
def test_state_graph_setup_rejects_missing_runtime_capabilities(backend, recompute, message):
    config = _config("attention")
    config.cuda_graph_impl = backend
    config.recompute_granularity = recompute
    with pytest.raises(ValueError, match=message):
        _stack(config, "attention")


@pytest.mark.parametrize("mode", ["mlp_recompute", "mlp_chunking"])
def test_stateful_mlp_rejects_undeclared_internal_boundaries(mode):
    stack = _stack(_config("attention"), "attention")
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


@pytest.mark.parametrize("backend", ["mcore", "te"])
@pytest.mark.parametrize("execution", ["checkpoint", "frozen-input", "block-eager"])
@pytest.mark.parametrize("differentiable", [False, True])
def test_hybrid_checkpoint_input_qualification(
    cpu_te_checkpoint, backend, execution, differentiable
):
    """Prepared lookup eligibility also applies to retained state and eager groups."""
    from megatron.core.transformer.state_boundary import compose_state_regions

    config = replace(
        _config("attention"),
        num_layers=1,
        recompute_granularity="full",
        recompute_method="block" if execution == "block-eager" else "uniform",
        recompute_num_layers=0 if execution == "block-eager" else 1,
    )
    if backend == "te":
        config.quant_recipe = object()
    hidden = torch.ones(3, 1, config.hidden_size, requires_grad=execution != "frozen-input")
    lookup = torch.full_like(hidden, 3.0, requires_grad=True)
    weight = nn.Parameter(torch.tensor(2.0))
    layer = nn.Identity()
    layer.layer_number = 1
    key = "prepared/lookup:batch"
    field = TensorField(key, tuple(hidden.shape), hidden.dtype, "sbhd", differentiable)
    context = boundary_runtime.HybridStackForwardContext(lookup={key: lookup})

    def boundary(*args):
        return compose_state_regions(
            "lookup/checkpoint",
            (
                (
                    "lookup",
                    StateRegion(
                        BoundarySchema("lookup", (field,), ()),
                        TensorMappingCodec(),
                        retained_fields=(field,),
                    ),
                ),
            ),
            boundary_runtime.HybridStackForwardContext,
        )

    output = boundary_runtime.checkpointed_hybrid_forward(
        config,
        nn.ModuleList([layer]),
        hidden,
        context,
        tp_group=None,
        quantization_context=lambda *_: nullcontext(),
        layer_forward=lambda layer, hidden, working: hidden * weight + 5 * working.lookup[key],
        boundary_factory=boundary,
    )
    output.sum().backward()
    assert context.lookup[key].requires_grad == differentiable
    if differentiable:
        torch.testing.assert_close(lookup.grad, torch.full_like(lookup, 5.0))
    else:
        assert lookup.grad is None
    torch.testing.assert_close(weight.grad, torch.tensor(float(hidden.numel())))
    if hidden.requires_grad:
        torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2.0))
    checkpointed = execution != "block-eager" and (hidden.requires_grad or differentiable)
    assert cpu_te_checkpoint == ([True] if backend == "te" and checkpointed else [])


@pytest.mark.parametrize("backend", ["mcore", "te"])
@pytest.mark.parametrize("hidden_active", [False, True])
def test_hybrid_checkpoint_preserves_unused_hidden_gradient(
    cpu_te_checkpoint, backend, hidden_active
):
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )

    config = replace(
        _config("attention"),
        num_layers=1,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    if backend == "te":
        config.quant_recipe = object()
    declaration = _MemoryDeclaration(config, layer_type_list=["*"], pp_layer_offset=0)
    x = torch.ones(3, 1, config.hidden_size, requires_grad=True)
    indexer = nn.Parameter(torch.tensor(3.0))
    layer = nn.Identity()
    layer.layer_number = 1
    context = boundary_runtime.HybridStackForwardContext(memory_owner=_MemoryState())

    def forward(layer, hidden, working):
        working.memory_owner.memory = hidden.sin()
        working.memory_owner.last_layer = 1
        return DSAIndexerLossAutoScaler.apply(hidden.square(), indexer.square())

    def region(start, end, hidden, working):
        from megatron.core.transformer.state_boundary import compose_state_regions

        return compose_state_regions(
            "memory",
            (
                (
                    "memory_owner",
                    declaration.checkpoint_region(start, end, hidden, working.memory_owner),
                ),
            ),
            boundary_runtime.HybridStackForwardContext,
        )

    hidden = boundary_runtime.checkpointed_hybrid_forward(
        config,
        nn.ModuleList([layer]),
        x,
        context,
        tp_group=None,
        quantization_context=lambda *_: nullcontext(),
        layer_forward=forward,
        boundary_factory=region,
    )
    loss = context.memory_owner.memory.sum()
    if hidden_active:
        loss = loss + hidden.sum() * 0
    loss.backward()
    torch.testing.assert_close(x.grad, x.detach().cos())
    if hidden_active:
        torch.testing.assert_close(indexer.grad, torch.tensor(6.0))
    else:
        assert indexer.grad is None
    assert cpu_te_checkpoint == ([True] if backend == "te" else [])


@pytest.mark.parametrize("backend", ["mcore", "te"])
@pytest.mark.parametrize("execution", ["checkpoint", "frozen-input", "block-eager", "no-grad"])
def test_hybrid_checkpoint_output_qualification(cpu_te_checkpoint, backend, execution):
    """Every group applies the side-output policy, including uncheckpointed groups."""
    config = replace(
        _config("attention"),
        num_layers=1,
        recompute_granularity="full",
        recompute_method="block" if execution == "block-eager" else "uniform",
        recompute_num_layers=0 if execution == "block-eager" else 1,
    )
    if backend == "te":
        config.quant_recipe = object()
    declaration = _MemoryDeclaration(config, layer_type_list=["*"], pp_layer_offset=0)
    x = torch.ones(3, 1, config.hidden_size, requires_grad=execution != "frozen-input")
    hidden_weight = nn.Parameter(torch.tensor(2.0))
    side_weight = nn.Parameter(torch.tensor(3.0))
    layer = nn.Identity()
    layer.layer_number = 1
    context = boundary_runtime.HybridStackForwardContext(memory_owner=_MemoryState())

    def forward(layer, hidden, working):
        working.memory_owner.memory = hidden * side_weight
        working.memory_owner.last_layer = 1
        return hidden * hidden_weight

    def region(start, end, hidden, working):
        from megatron.core.transformer.state_boundary import compose_state_regions

        native = declaration.checkpoint_region(start, end, hidden, working.memory_owner)
        native = replace(
            native,
            schema=replace(
                native.schema,
                outputs=tuple(replace(f, differentiable=False) for f in native.schema.outputs),
            ),
        )
        return compose_state_regions(
            "memory", (("memory_owner", native),), boundary_runtime.HybridStackForwardContext
        )

    with torch.set_grad_enabled(execution != "no-grad"):
        output = boundary_runtime.checkpointed_hybrid_forward(
            config,
            nn.ModuleList([layer]),
            x,
            context,
            tp_group=None,
            quantization_context=lambda *_: nullcontext(),
            layer_forward=forward,
            boundary_factory=region,
        )
    (output.sum() + context.memory_owner.memory.sum() + hidden_weight * 0).backward()
    assert side_weight.grad is None
    assert not context.memory_owner.memory.requires_grad
    expected = float(x.numel()) if execution != "no-grad" else 0.0
    torch.testing.assert_close(hidden_weight.grad, torch.tensor(expected))
    assert cpu_te_checkpoint == ([True] if backend == "te" and execution == "checkpoint" else [])


@pytest.mark.parametrize("backend", ["mcore", "te"])
@pytest.mark.parametrize("objective", ["memory", "both"])
def test_checkpoint_codec_with_arbitrary_context_attribute(cpu_te_checkpoint, backend, objective):
    """Both checkpoint backends carry state with no built-in context name or snapshot API."""
    torch.manual_seed(852)
    config = replace(
        _config("attention"),
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=2,
    )
    declaration = _MemoryDeclaration(config, layer_type_list=["*"] * 5, pp_layer_offset=0)
    declaration.context_attribute = "memory_owner"
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    adapter = HybridStateAdapter(
        config,
        components=(declaration,),
        hidden_size=config.hidden_size,
        hidden_dtype=config.params_dtype,
        layer_type_list=["*"] * 5,
        pp_layer_offset=0,
        pre_process=True,
        post_process=True,
        is_mtp_layer=False,
        pg_collection=groups,
    )
    actual = nn.ModuleList(_MemoryAttention(config, i + 1) for i in range(5))
    reference = nn.ModuleList(_MemoryAttention(config, i + 1) for i in range(5))
    reference.load_state_dict(actual.state_dict())
    if backend == "te":
        config.quant_recipe = object()

    def forward(layer, hidden, working):
        return layer(hidden, memory_state=working.memory_owner)[0]

    forwards = []
    for length in (3, 5):
        x = torch.randn(length, 2, config.hidden_size, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        _, _, context = adapter.prepare_forward(x, None, None)
        _, _, ref_context = adapter.prepare_forward(ref_x, None, None)
        assert set(vars(context)) == {"memory_owner"}
        hidden = boundary_runtime.checkpointed_hybrid_forward(
            config,
            actual,
            x,
            context,
            tp_group=groups.tp,
            quantization_context=lambda *_: nullcontext(),
            layer_forward=forward,
            boundary_factory=adapter.checkpoint_region,
        )
        ref_hidden = ref_x
        for layer in reference:
            ref_hidden = forward(layer, ref_hidden, ref_context)
        output = (hidden, context.memory_owner.memory)
        expected = (ref_hidden, ref_context.memory_owner.memory)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        forwards.append((x, ref_x, output, expected))
    for x, ref_x, output, expected in reversed(forwards):
        for values in (output, expected):
            selected = values if objective == "both" else values[1:]
            sum(t.square().sum() for t in selected).backward()
        _assert_grad(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_grad(parameter.grad, ref_parameter.grad)
    assert len(cpu_te_checkpoint) == (6 if backend == "te" else 0)


@pytest.mark.parametrize("kind", ["attention", "custom", "mhc_custom", "mhc_single_attention"])
@pytest.mark.parametrize("method,count", [("uniform", 1), ("uniform", 2), ("block", 2)])
@pytest.mark.parametrize("objective", ["hidden", "memory", "both"])
def test_memory_full_recompute_and_outstanding_microbatches(kind, method, count, objective):
    """Owner replacement and reverse-order backward preserve every consumer's gradient."""
    torch.manual_seed(419)
    config = _config(kind)
    reference = _stack(config, kind)
    actual = _stack(
        replace(
            config,
            recompute_granularity="full",
            recompute_method=method,
            recompute_num_layers=count,
        ),
        kind,
    )
    actual.load_state_dict(reference.state_dict())
    forwards = []
    for length in (3, 5):
        x = torch.randn(length, 2, config.hidden_size, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        output, ref_output = actual(x, None), reference(ref_x, None)
        torch.testing.assert_close(output, ref_output, atol=0, rtol=0)
        forwards.append((x, ref_x, output, ref_output))
    for x, ref_x, output, ref_output in reversed(forwards):
        for values in (output, ref_output):
            selected = values if objective == "both" else (values[objective == "memory"],)
            sum(value.square().sum() for value in selected).backward()
        _assert_grad(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_grad(parameter.grad, ref_parameter.grad)


@pytest.mark.parametrize("kind", ["attention", "mhc_attention"])
def test_memory_only_backward_restores_norm_output(kind):
    """A shared-memory loss must restore the producer norm without a local output loss."""
    torch.manual_seed(527)
    config = _config(kind)
    reference = _stack(config, kind)
    actual = _stack(
        replace(config, recompute_granularity="selective", recompute_modules=["layernorm"]), kind
    )
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(3, 2, config.hidden_size, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    memory, ref_memory = actual(x, None)[1], reference(ref_x, None)[1]
    torch.testing.assert_close(memory, ref_memory, atol=0, rtol=0)
    memory.square().sum().backward()
    ref_memory.square().sum().backward()
    _assert_grad(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_grad(parameter.grad, ref_parameter.grad)
