# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared binding mechanics; real-layer numerical equivalence lives under ssm/."""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.core.transformer.layer_interface import (
    LayerAuxInputs,
    LayerAuxOutputs,
    bind_layer_input_stage,
    bind_layer_interface,
    bind_layer_output_stage,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.determinism.utils import assert_bit_exact


def bare(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def test_aux_inputs_update_preserves_the_original():
    inputs = LayerAuxInputs(input_ids=torch.ones(2, 4))
    with pytest.raises(FrozenInstanceError):
        inputs.context = object()
    assert inputs.update(input_ids=inputs.input_ids) is inputs
    updated = inputs.update(context=object())
    assert inputs.context is None and updated.context is not None
    assert updated.input_ids is inputs.input_ids


@pytest.mark.parametrize("direct", [False, True])
def test_two_binding_hooks_own_input_and_output_adaptation(direct):
    class Layer(MegatronModule):
        def bind_layer_inputs(self, hidden_states, aux_inputs, **kwargs):
            events.append("bind_inputs")
            return super().bind_layer_inputs(hidden_states * 2, aux_inputs, **kwargs)

        def bind_layer_outputs(self, result):
            events.append("bind_outputs")
            output, context = result
            return output, LayerAuxOutputs(bias=context)

        @bind_layer_interface
        def forward(self, hidden_states, context=None):
            events.append("forward")
            return hidden_states.sin(), context

    events = []
    module = bare(Layer)
    x = torch.randn(2, 4, requires_grad=True)
    context = torch.randn(2, 4, requires_grad=True)
    module.register_forward_pre_hook(lambda *args: events.append("pre_hook"))
    module.register_forward_hook(lambda *args: events.append("post_hook"))
    call = module.forward if direct else module
    output, aux_outputs = call(x, LayerAuxInputs(context=context))
    assert events == (
        ["bind_inputs", "forward", "bind_outputs"]
        if direct
        else ["bind_inputs", "pre_hook", "forward", "post_hook", "bind_outputs"]
    )
    assert aux_outputs.bias is context
    (output.sum() + aux_outputs.bias.sum()).backward()
    assert_bit_exact(x.grad, {}, 2 * (2 * x.detach()).cos(), {})
    assert_bit_exact(context.grad, {}, torch.ones_like(context), {})

    class NativeLayer(MegatronModule):
        @bind_layer_interface
        def forward(self, hidden_states, aux_inputs):
            assert aux_inputs.context is context
            return hidden_states.sin()

    native = bare(NativeLayer)
    call = native.forward if direct else native
    assert_bit_exact(call(x, context=context), {}, x.sin(), {})
    result = call(x, aux_inputs=LayerAuxInputs(context=context))
    output, aux_outputs = (result, LayerAuxOutputs()) if direct else result
    assert_bit_exact(output, {}, x.sin(), {})
    assert aux_outputs.bias is None
    with pytest.raises(TypeError, match="cannot be mixed"):
        call(x, LayerAuxInputs(context=context), context=context)
    if not direct:
        with pytest.raises(TypeError, match="without legacy arguments"):
            native(x, aux_inputs=LayerAuxInputs(context=context), context=context)


def test_split_stages_use_the_same_two_binding_hooks():
    class Layer(MegatronModule):
        def bind_layer_inputs(self, hidden_states, aux_inputs, **kwargs):
            events.append("bind_inputs")
            return super().bind_layer_inputs(hidden_states * 2, aux_inputs, **kwargs)

        def bind_layer_outputs(self, result):
            events.append("bind_outputs")
            output, context = result
            return output, LayerAuxOutputs(bias=context)

        @bind_layer_interface
        def forward(self, hidden_states, context=None):
            raise AssertionError("Split execution must select the stage signature")

        @bind_layer_input_stage
        def forward_pre_attn_and_core_attn(self, hidden_states, attention_mask=None):
            return hidden_states, hidden_states.sin(), attention_mask

        @bind_layer_output_stage
        def forward_post_core_attn(self, hidden_states, residual, context):
            return hidden_states + residual, context

    events = []
    module = bare(Layer)
    x = torch.randn(2, 4)
    mask = object()
    state = module.forward_pre_attn_and_core_attn(
        x, LayerAuxInputs(attention_mask=mask, context=object())
    )
    assert len(state) == 3 and state[2] is mask
    assert events == ["bind_inputs"]
    legacy_output, legacy_context = module.forward_post_core_attn(*state)
    assert events == ["bind_inputs"]
    assert legacy_context is mask
    output, aux_outputs = module.forward_post_core_attn(*state, aux_inputs=LayerAuxInputs())
    assert events == ["bind_inputs", "bind_outputs"]
    assert_bit_exact(legacy_output, {}, output, {})
    assert_bit_exact(output, {}, 2 * x + (2 * x).sin(), {})
    assert aux_outputs.bias is mask


def test_transformer_split_stage_does_not_bind_context_as_bias():
    layer = bare(TransformerLayer)
    x, context = torch.randn(2, 4), torch.randn(2, 4)
    layer.supports_two_stage_attention = Mock(return_value=True)
    layer.self_attention = nn.Module()
    layer.self_attention.forward_post_core_attn = Mock(return_value=(x, None))
    layer.attention_bda_and_cross_attention = Mock(return_value=(x, context))
    assert layer.forward_post_core_attn(x, x, context=context) is x
    output, aux_outputs = layer.forward_post_core_attn(
        x, x, context=context, aux_inputs=LayerAuxInputs()
    )
    assert output is x and aux_outputs.bias is None
