# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Argument packing, layer contracts, and graph dispatch boundaries."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from megatron.core.recompute import checkpointed_forward
from megatron.core.transformer.layer_interface import (
    AuxInputs,
    LayerAuxOutputs,
    LayerInterface,
    bind_aux_inputs,
    bind_layer_input_stage,
    bind_layer_output_stage,
)
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from tests.unit_tests.determinism.utils import assert_bit_exact


def bare(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


@pytest.mark.parametrize("direct", [False, True])
def test_argument_packing_preserves_wrapper_output(direct):
    class Wrapper(MegatronModule):
        @bind_aux_inputs
        def forward(self, hidden_states, gain=1, *, aux_inputs):
            assert vars(aux_inputs) == {"context": context}
            return result

    wrapper = bare(Wrapper)
    x, context, result = torch.ones(1), object(), {"output": object()}
    call = wrapper.forward if direct else wrapper
    for invoke in (
        lambda: call(x, AuxInputs(gain=2, context=context)),
        lambda: call(x, gain=2, context=context),
    ):
        assert invoke() is result
    with pytest.raises(TypeError, match="gain"):
        call(x, AuxInputs(gain=2), gain=3)


@pytest.mark.parametrize("direct", [False, True])
def test_layer_inheritance_binds_overrides_and_preserves_hooks(direct):
    class Core(MegatronModule, LayerInterface):
        def forward(self, hidden_states, bias=None):
            events.append("forward")
            return hidden_states * 2, bias

    class Child(Core):
        def forward(self, hidden_states, bias=None):
            return super().forward(hidden_states, bias=bias)

    layer = bare(Child)
    events = []
    layer.register_forward_pre_hook(lambda *args: events.append("pre"))
    layer.register_forward_hook(lambda *args: events.append("post"))
    x = torch.ones(2, requires_grad=True)
    bias = torch.ones(2, requires_grad=True)
    call = layer.forward if direct else layer
    output, aux = call(x, AuxInputs(bias=bias, unused=object()))
    assert events == (["forward"] if direct else ["pre", "forward", "post"])
    assert isinstance(aux, LayerAuxOutputs) and aux.bias is bias
    (output + aux.bias).sum().backward()
    assert_bit_exact(x.grad, {}, torch.full_like(x, 2), {})
    assert_bit_exact(bias.grad, {}, torch.ones_like(bias), {})
    _, legacy_bias = call(x, bias=bias)
    assert legacy_bias is bias


def test_inputs_follow_signature_and_remain_read_only():
    class Core(MegatronModule, LayerInterface):
        def forward(self, hidden_states, weights, gain=2):
            return hidden_states * weights * gain

    layer = bare(Core)
    x = torch.ones(2)
    for inputs in (AuxInputs(weights=x), layer.forward.Inputs(weights=x)):
        output, _ = layer(x, inputs)
        assert_bit_exact(output, {}, x * 2, {})
        with pytest.raises(FrozenInstanceError):
            inputs.weights = None
        with pytest.raises(TypeError, match="weights"):
            inputs.update(weights=None)
        extended = inputs.update(extra=x)
        assert "extra" not in vars(inputs)
        leaves, spec = tree_flatten(extended)
        restored = tree_unflatten(leaves, spec)
        assert restored.weights is x and restored.extra is x
    with pytest.raises(TypeError, match="weights"):
        layer(x, AuxInputs())


def test_split_stages_preserve_intermediates_and_bind_final_output():
    class Core(MegatronModule, LayerInterface):
        def forward(self, hidden_states):
            return hidden_states

        @bind_layer_input_stage
        def prepare(self, hidden_states, bias=None):
            return hidden_states, bias

        @bind_layer_output_stage
        def finish(self, hidden_states, bias):
            return hidden_states, bias

    class Child(Core):
        pass

    layer = bare(Child)
    x, bias = torch.ones(2), torch.zeros(2)
    state = layer.prepare(x, AuxInputs(bias=bias))
    assert state[0] is x and state[1] is bias
    assert layer.finish(*state)[1] is bias
    output, aux = layer.finish(*state, aux_inputs=AuxInputs())
    assert output is x and aux.bias is bias


@pytest.mark.parametrize("decorator", [bind_layer_input_stage, bind_layer_output_stage])
@pytest.mark.parametrize("shared", [False, True])
def test_stage_decorators_require_actual_inheritance(decorator, shared):
    class Lookalike(MegatronModule):
        bind_layer_outputs = LayerInterface.bind_layer_outputs

        @decorator
        def stage(self, hidden_states):
            pytest.fail("Reject invalid stages before execution")

    module = bare(Lookalike)
    assert isinstance(module, LayerInterface)
    with pytest.raises(TypeError, match="requires LayerInterface inheritance"):
        module.stage(torch.ones(1), **({"aux_inputs": AuxInputs()} if shared else {}))


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("extract", [False, True])
def test_checkpoint_helper_preserves_legacy_return_format(shared, extract):
    class Core(MegatronModule, LayerInterface):
        def forward(self, hidden_states, gain=1, context=None, **kwargs):
            contexts.append(context)
            return hidden_states * gain, None if context is None else context + 1

    block = bare(MegatronModule)
    block.config = SimpleNamespace(
        fp8=False, fp4=False, recompute_method="block", recompute_num_layers=0
    )
    block.layers = nn.ModuleList([bare(Core), bare(Core)])
    block.num_layers_per_pipeline_rank = 2
    contexts = []
    x = torch.ones(2, requires_grad=True)
    args, kwargs = (
        ((AuxInputs(gain=3),), {}) if shared else ((), {"gain": 3, "context": torch.zeros(2)})
    )
    result = checkpointed_forward(
        block,
        x,
        *args,
        use_inner_quantization_context=False,
        extract_layer_indices={1} if extract else None,
        **kwargs,
    )
    if extract:
        result, intermediates = result
        assert intermediates[0] is result
    assert_bit_exact(result, {}, x * 9, {})
    if not shared:
        assert_bit_exact(contexts[1], {}, contexts[0] + 1, {})
    result.sum().backward()
    assert_bit_exact(x.grad, {}, torch.full_like(x, 9), {})


def test_graph_processing_preserves_wrapper_outputs():
    class Wrapper(GraphableMegatronModule):
        @bind_aux_inputs
        def forward(self, hidden_states, *, aux_inputs=None):
            return hidden_states

    wrapper = bare(Wrapper)
    x = torch.ones(1)
    kwargs = {"aux_inputs": AuxInputs()}
    args, graph_kwargs = wrapper.process_aux_inputs((x,), kwargs)
    assert not args and graph_kwargs == {"hidden_states": x}
    result = (x,)
    assert wrapper.process_aux_outputs(result, kwargs, local=True) is result
    assert "aux_inputs" in kwargs


@pytest.mark.parametrize("backend", ["local", "transformer_engine"])
@pytest.mark.parametrize("carry_inputs", [False, True])
def test_graph_adapter_preserves_flat_calls_and_packed_outputs(backend, carry_inputs):
    seen_inputs = []

    class Layer(GraphableMegatronModule, LayerInterface):
        if carry_inputs:

            def forward(
                self, hidden_states, padding_mask=None, inference_context=None, *, aux_inputs=None
            ):
                seen_inputs.append(aux_inputs)
                return hidden_states + padding_mask

        else:

            def forward(self, hidden_states, padding_mask=None, inference_context=None):
                return hidden_states + padding_mask

    layer = bare(Layer)
    layer.config = SimpleNamespace(fine_grained_activation_offloading=False)
    layer.cuda_graph_manual_hooks = []
    layer._should_call_te_cudagraph = lambda *args, **kwargs: backend == "transformer_engine"

    def local_graph(module, args, kwargs):
        assert not args and "hidden_states" in kwargs
        assert "aux_inputs" not in kwargs
        assert kwargs["inference_context"] is None
        return (module.forward(*args, **kwargs),)

    if backend == "local":
        layer.cudagraph_manager = local_graph

    x, mask = torch.randn(2, 4), torch.randn(2, 4)
    inputs = AuxInputs(padding_mask=mask, context=object())

    def replay(hidden_states, *, padding_mask, is_first_microbatch):
        assert padding_mask is mask and is_first_microbatch
        return hidden_states + padding_mask

    for packed in (False, True):
        layer.cuda_graphs = []
        for capture in (True, False):
            if not capture:
                layer.cuda_graphs = [replay]
            result = (
                layer(x, inputs)
                if packed
                else layer(hidden_states=x, padding_mask=mask, inference_context=None)
            )
            if packed:
                result, outputs = result
                assert isinstance(outputs, LayerAuxOutputs) and outputs.bias is None
            assert_bit_exact(result, {}, x + mask, {})
    assert all(value is None for value in seen_inputs)
    if backend == "transformer_engine":
        with pytest.raises(AssertionError, match="only Tensor inputs"):
            layer(x, inputs.update(inference_context=object()))

    layer._should_call_local_cudagraph = lambda *args, **kwargs: False
    layer._should_call_te_cudagraph = lambda *args, **kwargs: False
    layer(x, inputs)
    if carry_inputs:
        assert vars(seen_inputs[-1]) == {"context": inputs.context}
        assert inputs.padding_mask is mask

    if backend == "local":
        # Legacy model/block callers still unwrap the manager's tuple themselves.
        layer.forward = layer.forward.__wrapped__.__get__(layer)
        layer._should_call_local_cudagraph = lambda *args, **kwargs: True
        result = layer(hidden_states=x, padding_mask=mask, inference_context=None)
        assert isinstance(result, tuple) and len(result) == 1
        assert_bit_exact(result[0], {}, x + mask, {})
