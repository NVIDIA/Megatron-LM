# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Input adapter and arithmetic equivalence checks, using mocked cores on CPU."""

import copy
import inspect
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.recompute import checkpointed_forward
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNet2
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.cuda_graphs import (
    _CudagraphReplayNode,
    _CudaGraphRunner,
    _GraphStatus,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.layer_interface import (
    LayerAuxInputs,
    LayerAuxOutputs,
    LayerInterface,
    bind_layer_input_stage,
    bind_layer_interface,
    bind_layer_output_stage,
)
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.determinism.utils import assert_bit_exact


class DeferredBiasLinear(nn.Linear):

    def forward(self, x):
        return (nn.functional.linear(x, self.weight), self.bias)


def bare(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def apply_output_bias(output):
    assert isinstance(output, tuple) and len(output) == 2
    hidden_states, aux_outputs = output
    bias = aux_outputs.bias if isinstance(aux_outputs, LayerAuxOutputs) else aux_outputs
    assert isinstance(hidden_states, torch.Tensor)
    assert bias is None or isinstance(bias, torch.Tensor)
    return hidden_states if bias is None else hidden_states + bias


def test_frozen_inputs_and_interface():
    aux_inputs = LayerAuxInputs()
    with pytest.raises(FrozenInstanceError):
        aux_inputs.input_ids = torch.ones(1)
    assert not isinstance(object(), LayerInterface)
    for cls in (
        SelfAttention,
        MLASelfAttention,
        MLP,
        MoELayer,
        MambaMixer,
        GatedDeltaNet,
        GatedDeltaNet2,
        GatedDeltaProductMixer,
    ):
        assert LayerInterface in cls.__mro__


def assert_equivalent(module, legacy, aux_inputs=LayerAuxInputs()):
    other = copy.deepcopy(module)
    x = torch.randn(3, 2, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    old = legacy(module, x)
    new = apply_output_bias(other(y, aux_inputs))
    assert_bit_exact(new, {}, old, {})
    old.square().sum().backward()
    new.square().sum().backward()
    assert_bit_exact(y.grad, {}, x.grad, {})
    for (name, p), (other_name, q) in zip(module.named_parameters(), other.named_parameters()):
        assert name == other_name
        assert_bit_exact(q.grad, {}, p.grad, {})


def test_mlp_forward_backward_with_and_without_bias():
    for bias in (False, True):
        module = bare(MLP)
        module.config = SimpleNamespace(
            use_te_activation_func=False,
            bias_activation_fusion=False,
            activation_func_tanh_clamp_scale=None,
            gated_linear_unit=False,
        )
        module.activation_func = nn.functional.gelu
        module.linear_fc1 = DeferredBiasLinear(4, 8, bias=bias)
        module.linear_fc2 = DeferredBiasLinear(8, 4, bias=bias)
        assert_equivalent(module, lambda m, x: apply_output_bias(m(x)))


def test_attention_adapter_threads_every_execution_input():
    module = bare(SelfAttention)
    output, bias = (torch.randn(2, 4), torch.randn(4))
    module.forward_pre_attn_and_core_attn = Mock(return_value=output)
    module.forward_post_core_attn = Mock(return_value=(output, bias))
    mask, rope, metadata, packed, context = (object() for _ in range(5))
    aux_inputs = LayerAuxInputs(
        attention_mask=mask,
        rotary_pos_emb=rope,
        packed_sequence_cp_metadata=metadata,
        packed_seq_params=packed,
        inference_context=context,
        sequence_len_offset=3,
    )
    result = module(output, aux_inputs)
    assert result[0] is output
    assert result[1].bias is bias
    kwargs = module.forward_pre_attn_and_core_attn.call_args.kwargs
    for key, value in dict(
        attention_mask=mask,
        rotary_pos_emb=rope,
        packed_sequence_cp_metadata=metadata,
        packed_seq_params=packed,
        inference_context=context,
        sequence_len_offset=3,
    ).items():
        assert kwargs[key] is value
    module.forward_post_core_attn.assert_called_once_with(output)
    assert module(hidden_states=output, inputs=aux_inputs)[0] is output
    assert module(x=output, inputs=aux_inputs)[0] is output
    with pytest.raises(TypeError):
        module(output, aux_inputs, attention_mask=mask)


@pytest.mark.parametrize(
    "cls",
    [
        MLASelfAttention,
        AbsorbedMLASelfAttention,
        DSv4HybridSelfAttention,
        MambaMixer,
        GatedDeltaNet,
        GatedDeltaNet2,
        GatedDeltaProductMixer,
    ],
)
def test_core_entry_points_delegate_to_legacy_inference_dispatch(cls):
    x, bias = (torch.randn(2, 4), torch.randn(4))
    aux_inputs = LayerAuxInputs(
        attention_mask=object(),
        packed_seq_params=object(),
        inference_context=object(),
        sequence_len_offset=7,
    )
    module = bare(cls)
    module._call_with_bound_args = Mock(return_value=(x, bias))
    result = module(x, aux_inputs)
    assert result[0] is x
    assert result[1].bias is bias
    kwargs = module._call_with_bound_args.call_args.kwargs
    assert kwargs['inference_context'] is aux_inputs.inference_context
    assert kwargs['packed_seq_params'] is aux_inputs.packed_seq_params


def test_new_core_implements_interface_without_legacy_adapter_or_layer_wrapper():

    class Core(GraphableMegatronModule, LayerInterface):

        def __init__(self):
            super().__init__(TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1))
            self.proj = nn.Linear(4, 4)

        def forward(self, x, aux_inputs: LayerAuxInputs):
            return self.proj(x), LayerAuxOutputs()

    core = Core()
    assert isinstance(core, LayerInterface)
    assert not hasattr(core, 'forward_layer')
    x = torch.randn(3, 2, 4, requires_grad=True)
    aux_inputs = LayerAuxInputs(input_ids=torch.ones(2, 3, dtype=torch.long))
    result = apply_output_bias(core(x, aux_inputs))
    assert_bit_exact(core(x=x, aux_inputs=aux_inputs)[0], {}, result, {})
    expected = nn.functional.linear(x, core.proj.weight, core.proj.bias)
    assert_bit_exact(result, {}, expected, {})
    result.sum().backward()
    assert x.grad is not None
    assert core.proj.weight.grad is not None


def test_gdp_training_forward_backward_and_packed_metadata():
    for bias in (False, True):
        core = bare(GatedDeltaProductMixer)
        core.chunkwise_context_parallel = True
        core._gdp_chunk_forward = lambda x, **kwargs: x.sin()
        core.out_proj = DeferredBiasLinear(4, 4, bias=bias)
        packed, metadata = (object(), object())
        aux_inputs = LayerAuxInputs(packed_seq_params=packed, packed_sequence_cp_metadata=metadata)
        assert_equivalent(
            core,
            lambda m, x: apply_output_bias(
                m(x, packed_seq_params=packed, packed_sequence_cp_metadata=metadata)
            ),
            aux_inputs,
        )
        core._gdp_chunk_forward = Mock(side_effect=lambda x, **kwargs: x.sin())
        x = torch.randn(3, 2, 4)
        updates = core(x, aux_inputs)
        kwargs = core._gdp_chunk_forward.call_args.kwargs
        assert kwargs['packed_seq_params'] is packed
        assert kwargs['packed_sequence_cp_metadata'] is metadata
        assert len(updates) == 2
        assert updates[1].bias is (core.out_proj.bias if bias else None)


def test_gdp_preserves_static_and_dynamic_inference_dispatch():
    x = torch.randn(3, 2, 4)
    for dynamic in (False, True):
        core = bare(GatedDeltaProductMixer)
        core.chunkwise_context_parallel = False
        core.config = SimpleNamespace(
            gdp_cutedsl_kernel=False, batch_invariant_mode=False, sequence_parallel=False
        )
        core.cp = SimpleNamespace(cp_size=1)
        context = SimpleNamespace(
            is_dynamic_batching=lambda: dynamic,
            is_static_batching=lambda: not dynamic,
            seqlen_offset=1,
        )
        conv_state, ssm_state = (object(), object())
        core._get_states_from_cache = Mock(return_value=(conv_state, ssm_state))
        core._gdp_chunk_forward = Mock(side_effect=AssertionError('scan called twice'))
        core.out_proj = Mock(side_effect=AssertionError('projection called twice'))
        for bias in (None, torch.randn(4)):
            core._static_decode = Mock(return_value=(x, bias))
            core.ssm_dynamic_inference = Mock(return_value=(x, bias))
            with patch(
                'megatron.core.ssm.gated_delta_product.check_fla_sequence_packing_support',
                return_value=(True, ''),
            ):
                legacy = core(x, inference_context=context)
                updates = core(x, LayerAuxInputs(inference_context=context))
            assert updates[0] is legacy[0]
            assert len(updates) == 2
            assert updates[1].bias is bias
            if dynamic:
                core.ssm_dynamic_inference.assert_called_with(x, context)
                core._static_decode.assert_not_called()
            else:
                core._static_decode.assert_called_with(x, conv_state, ssm_state)
                core.ssm_dynamic_inference.assert_not_called()
        core._gdp_chunk_forward.assert_not_called()
        core.out_proj.assert_not_called()


def test_gdp_shared_inputs_preserve_chunkwise_inference_rejection():
    core = bare(GatedDeltaProductMixer)
    core.chunkwise_context_parallel = True
    with pytest.raises(NotImplementedError, match='does not support inference'):
        core(torch.randn(3, 2, 4), LayerAuxInputs(inference_context=object()))


def test_skeleton_adapters_preserve_recompute_and_mhc_state():
    x = torch.randn(2, 4)
    replay, manager, metadata = (object(), object(), object())
    aux_inputs = LayerAuxInputs(
        packed_sequence_cp_metadata=metadata,
        input_ids=object(),
        residual_stream_recompute_context=replay,
    )
    for cls in (TransformerLayer, MambaLayer):
        module = bare(cls)
        module.config = SimpleNamespace(cuda_graph_impl='none', linear_cp_mode='chunkwise')
        module.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=True))
        module._call_with_bound_args = Mock(
            return_value=(x, LayerAuxOutputs()) if cls is TransformerLayer else x
        )
        assert module(x, aux_inputs) == (x, LayerAuxOutputs())
        kwargs = module._call_with_bound_args.call_args.kwargs
        assert kwargs['aux_inputs'] is aux_inputs
    wrapper = bare(HyperConnectionHybridLayer)
    wrapper.inner_layer = SimpleNamespace()
    wrapper._call_with_bound_args = Mock(return_value=x)
    assert wrapper(x, LayerAuxInputs(mhc_recompute_manager=manager)) == (x, LayerAuxOutputs())
    assert (
        wrapper._call_with_bound_args.call_args.kwargs['aux_inputs'].mhc_recompute_manager
        is manager
    )
    with pytest.raises(TypeError, match='mhc'):
        wrapper(x, aux_inputs)


def test_graph_adapter_limits_shared_inputs_to_layer_execution_fields():
    module = bare(TransformerLayer)
    module.config = SimpleNamespace(cuda_graph_impl='local')
    x = torch.randn(2, 4)
    module.cudagraph_manager = Mock(return_value=(x, LayerAuxOutputs()))
    inputs = LayerAuxInputs(attention_mask=x, execution_context=object(), embedding=object())
    assert module(x, inputs) == (x, LayerAuxOutputs())
    args = module.cudagraph_manager.call_args.args
    assert args[1] == ()
    assert args[2]['hidden_states'] is x
    graph_inputs = args[2]['aux_inputs']
    assert graph_inputs.attention_mask is x
    assert graph_inputs.execution_context is None
    assert graph_inputs.embedding is None
    assert inputs.execution_context is not None


@pytest.mark.parametrize("mode", ["headwise", "chunkwise"])
@pytest.mark.parametrize("has_metadata", [False, True])
def test_mamba_binding_preserves_cp_metadata_independently_of_mode(mode, has_metadata):
    module = bare(MambaLayer)
    module.config = SimpleNamespace(linear_cp_mode=mode)
    x = torch.randn(2, 4)
    metadata = object() if has_metadata else None
    module._call_with_bound_args = Mock(return_value=x)

    inputs = LayerAuxInputs(packed_sequence_cp_metadata=metadata)
    assert module(x, inputs) == (x, LayerAuxOutputs())
    assert module._call_with_bound_args.call_args.kwargs["aux_inputs"] is inputs


@pytest.mark.parametrize(
    "mixer_cp", [None, False, True], ids=["unsupported", "headwise", "chunkwise"]
)
@pytest.mark.parametrize("mode", ["headwise", "chunkwise"])
@pytest.mark.parametrize("shared", [False, True])
def test_mamba_forward_carries_aux_inputs_to_its_mixer(mixer_cp, mode, shared):
    module = bare(MambaLayer)
    module.config = SimpleNamespace(linear_cp_mode=mode)
    module.layer_number = 1
    x = torch.randn(2, 4)
    metadata, packed = object(), object()
    module._prepare_mixer_state = Mock(return_value=(x, x, None, None))
    module._apply_mixer_update = Mock(return_value=x)
    module.mixer = Mock(side_effect=lambda h, inputs: (x, LayerAuxOutputs()))
    if mixer_cp is None:
        del module.mixer.chunkwise_context_parallel
    else:
        module.mixer.chunkwise_context_parallel = mixer_cp

    inputs = LayerAuxInputs(packed_seq_params=packed, packed_sequence_cp_metadata=metadata)
    if shared:
        output = module.forward(x, inputs)
        assert output == (x, LayerAuxOutputs())
    else:
        assert module.forward(hidden_states=x, aux_inputs=inputs) is x
    assert module.mixer.call_args.args[1] is inputs


@pytest.mark.parametrize(
    ("cls", "mode"),
    [(TransformerLayer, "headwise"), (MambaLayer, "headwise"), (MambaLayer, "chunkwise")],
)
def test_pre_core_forwarding_selects_metadata_and_preserves_stage_state(cls, mode):
    module = bare(cls)
    module.config = SimpleNamespace(linear_cp_mode=mode)
    x, residual, intermediate = (torch.randn(2, 4) for _ in range(3))
    metadata, packed, recompute, connection_state = (object() for _ in range(4))
    core = Mock()
    core.forward_pre_attn_and_core_attn.return_value = intermediate
    if cls is MambaLayer:
        core.chunkwise_context_parallel = mode == "chunkwise"
        module.mixer = core
        module._prepare_mixer_state = Mock(return_value=(x, residual, connection_state, None))
    else:
        module.self_attention = core
        module.supports_two_stage_attention = Mock(return_value=True)
        module._run_input_layernorm = Mock(return_value=(x, residual, connection_state))

    state = module.forward_pre_attn_and_core_attn(
        x,
        LayerAuxInputs(
            packed_seq_params=packed,
            packed_sequence_cp_metadata=metadata,
            residual_stream_recompute_context=recompute,
        ),
    )
    assert state[0] is intermediate
    assert state[1] is residual
    assert state[-1] is connection_state
    kwargs = core.forward_pre_attn_and_core_attn.call_args.kwargs
    assert kwargs["packed_seq_params"] is packed
    if cls is MambaLayer:
        assert kwargs["packed_sequence_cp_metadata"] is (metadata if mode == "chunkwise" else None)
    else:
        assert "packed_sequence_cp_metadata" not in kwargs
    prepare = module._prepare_mixer_state if cls is MambaLayer else module._run_input_layernorm
    assert prepare.call_args.kwargs["residual_stream_recompute_context"] is recompute


@pytest.mark.parametrize("chunks", [1, 2])
@pytest.mark.parametrize("recompute", [False, True])
def test_mlp_shared_binding_preserves_bias_and_checkpoint_tensor_boundary(
    monkeypatch, chunks, recompute
):
    class Core(MegatronModule):
        def __init__(self):
            super().__init__(config=None)
            self.bias = nn.Parameter(torch.ones(4))
            self.calls = []

        def bind_layer_inputs(self, hidden_states, aux_inputs, **kwargs):
            return super().bind_layer_inputs(hidden_states * 2, aux_inputs, **kwargs)

        @bind_layer_interface
        def forward(self, hidden_states, padding_mask=None, input_ids=None, packed_seq_params=None):
            self.calls.append((hidden_states.shape, padding_mask, input_ids, packed_seq_params))
            return hidden_states.square(), self.bias

    def tensor_checkpoint(function, distribute_saved_activations, *args):
        output = checkpoint(function, *args, use_reentrant=False)
        assert isinstance(output[0], torch.Tensor)
        assert isinstance(output[1], torch.Tensor)
        return output

    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.tensor_parallel.checkpoint", tensor_checkpoint
    )
    core = Core()
    layer = bare(TransformerLayer)
    layer.mlp = core
    layer.is_moe_layer = True
    layer.recompute_mlp = recompute
    layer.config = SimpleNamespace(
        cuda_graph_impl="none",
        mlp_chunks_for_prefill=1,
        mlp_chunks_for_training=chunks,
        transformer_impl="local",
        inference_fuse_tp_communication=False,
        fp8=False,
        fp4=False,
    )
    x = torch.randn(4, 2, 4, requires_grad=True)
    ids = torch.arange(8).reshape(2, 4)
    mask = ids % 2 == 0
    packed = object()
    output, bias = layer._run_mlp(
        x,
        x,
        mask,
        None,
        input_ids=ids,
        packed_seq_params=packed,
        aux_inputs=LayerAuxInputs(padding_mask=mask, input_ids=ids, packed_seq_params=packed),
    )
    assert bias is core.bias
    torch.testing.assert_close(output, (x * 2).square())
    expected_chunks = 1 if recompute else chunks
    assert len(core.calls) == expected_chunks
    for index, (shape, seen_mask, seen_ids, seen_packed) in enumerate(core.calls):
        assert shape[0] == 4 // expected_chunks
        assert torch.equal(seen_mask, mask.chunk(expected_chunks, dim=1)[index])
        assert torch.equal(seen_ids, ids.chunk(expected_chunks, dim=1)[index])
        assert seen_packed is packed
    (output + bias).sum().backward()
    torch.testing.assert_close(x.grad, 8 * x)
    torch.testing.assert_close(core.bias.grad, torch.full_like(core.bias, 8))


def test_mlp_te_partial_capture_preserves_intermediate_outputs(monkeypatch):
    layer = bare(TransformerLayer)
    layer.is_moe_layer = True
    layer.recompute_mlp = False
    layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.moe_router],
        mlp_chunks_for_prefill=1,
        mlp_chunks_for_training=1,
        inference_fuse_tp_communication=False,
    )
    x = torch.randn(2, 4)
    intermediates = (x, x + 1, x + 2, None)
    layer.mlp = nn.Module()
    layer.mlp.forward = Mock(return_value=intermediates)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.is_graph_capturing", lambda: True
    )
    assert layer._run_mlp(x, x, None, None) is intermediates


def test_shortcut_atomic_call_accepts_any_layer_with_shared_inputs():
    x = torch.randn(2, 4)
    layer = Mock(side_effect=lambda h, inputs: (x, LayerAuxOutputs()))
    unit = SimpleNamespace(compute_layer=layer)
    aux_inputs = LayerAuxInputs(packed_sequence_cp_metadata=object())
    assert ShortcutMoEBlock._forward_compute_atomic(unit, x, aux_inputs) == (x, aux_inputs)
    assert layer.call_args.args == (x, aux_inputs)


@pytest.mark.parametrize("config_cls", [MambaLayerConfig, GDNLayerConfig, TransformerConfig])
@pytest.mark.parametrize("method", ["uniform", "block"])
@pytest.mark.parametrize("shared", [False, True])
def test_recompute_preserves_call_style_and_replays_gradients(config_cls, method, shared):

    class Layer(nn.Module):

        def __init__(self):
            super().__init__()
            self.layer_number = 1
            self.config = config_cls(
                num_layers=1,
                hidden_size=256,
                num_attention_heads=4,
                linear_cp_mode='chunkwise',
                linear_cp_layout='contiguous',
            )
            self.inputs = []

        def forward(self, hidden_states, aux_inputs=None, **kwargs):
            self.inputs.append(aux_inputs if aux_inputs is not None else kwargs)
            return hidden_states.sin(), LayerAuxOutputs() if aux_inputs is not None else None

    layer = Layer()
    stack = SimpleNamespace(
        layers=[layer],
        num_layers_per_pipeline_rank=1,
        config=SimpleNamespace(
            fp8=False,
            fp4=False,
            recompute_method=method,
            recompute_num_layers=1,
            distribute_saved_activations=False,
        ),
    )
    x = torch.randn(2, 4, requires_grad=True)
    ids, metadata = (object(), object())
    inputs = LayerAuxInputs(
        rotary_pos_emb=(None, None), input_ids=ids, packed_sequence_cp_metadata=metadata
    )
    with patch(
        'megatron.core.recompute.tensor_parallel.checkpoint',
        side_effect=lambda fn, distribute, *args: checkpoint(fn, *args, use_reentrant=True),
    ):
        y, _ = checkpointed_forward(
            stack,
            x,
            use_inner_quantization_context=False,
            **({"aux_inputs": inputs} if shared else vars(inputs)),
        )
        y.sum().backward()
    assert_bit_exact(y, {}, x.sin(), {})
    assert_bit_exact(x.grad, {}, x.detach().cos(), {})
    assert len(layer.inputs) == 2
    for received in layer.inputs:
        assert isinstance(received, LayerAuxInputs) == shared
        fields = vars(received) if shared else received
        assert fields["input_ids"] is ids
        assert fields["packed_sequence_cp_metadata"] is metadata


@pytest.mark.parametrize("call_style", ["legacy", "shared", "aux_keyword"])
def test_transformer_preserves_shared_inputs_and_legacy_core_calls(call_style):
    shared = call_style != "legacy"
    layer = bare(TransformerLayer)
    layer.layer_number = 1
    layer.config = SimpleNamespace(inference_fuse_tp_communication=False)
    layer.is_moe_layer = False
    layer._padding_mask_shape_seen = None
    x = torch.randn(2, 1, 4)
    mask, ids = torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.long)
    context = torch.randn_like(x)
    inputs = LayerAuxInputs(padding_mask=mask, input_ids=ids, context=context)
    layer._run_input_layernorm = Mock(return_value=(x, x, ()))
    layer.self_attention = nn.Module()
    layer.self_attention.forward = Mock(return_value=(x, LayerAuxOutputs() if shared else None))
    layer.attention_bda_and_cross_attention = Mock(return_value=(x, context))
    layer._pre_mlp_layernorm_and_residual = Mock(return_value=(x, x, ()))
    layer._maybe_unflatten_for_moe = Mock(return_value=(x, mask, ids, None))
    layer._run_mlp = Mock(return_value=(x, None))
    layer._apply_mlp_bda_step = Mock(return_value=x)

    if call_style == "shared":
        output, auxiliary = layer.forward(x, inputs)
    else:
        output, auxiliary = layer.forward(
            x,
            padding_mask=mask,
            input_ids=ids,
            context=context,
            aux_inputs=inputs if shared else None,
        )
    if shared:
        assert isinstance(auxiliary, LayerAuxOutputs)
        assert auxiliary.bias is None
        assert layer.self_attention.forward.call_args.args == (x, inputs)
        assert layer.self_attention.forward.call_args.kwargs == {}
        assert layer._run_mlp.call_args.kwargs["aux_inputs"] is inputs
    else:
        assert auxiliary is context
        assert layer.self_attention.forward.call_args.args == (x,)
        assert "aux_inputs" not in layer.self_attention.forward.call_args.kwargs
        assert layer._run_mlp.call_args.kwargs["aux_inputs"] is None
    assert output is x
    assert layer._run_mlp.call_args.args[2] is mask
    assert layer._run_mlp.call_args.kwargs["input_ids"] is ids


def test_hybrid_stack_eager_and_full_recompute_use_the_same_call():

    class TensorLayer(nn.Module):

        def __init__(self, config, layer_number, **kwargs):
            super().__init__()
            self.config, self.layer_number = (config, layer_number)

        def forward(self, hidden_states, aux_inputs):
            self.received_inputs = aux_inputs
            return hidden_states.sin(), LayerAuxOutputs()

    for method in (None, 'uniform', 'block'):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=4,
            num_attention_heads=1,
            use_cpu_initialization=True,
            recompute_granularity='full' if method else None,
            recompute_method=method,
            recompute_num_layers=1 if method else None,
        )
        group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
        with patch(
            'megatron.core.models.hybrid.hybrid_block.build_module',
            side_effect=lambda spec, **kwargs: TensorLayer(**kwargs),
        ):
            stack = HybridStack(
                config,
                hybrid_stack_spec.submodules,
                layer_config_list=validate_segment_layers('*-', config),
                post_process=False,
                post_layer_norm=False,
                pg_collection=SimpleNamespace(tp=group, cp=group, pp=group, tp_cp=group),
            )
        # Simulate metadata prepared once for a mixed stack; each layer owns its filtering.
        stack._has_linear_layer_with_chunkwise_cp = True
        metadata = object()
        packed = SimpleNamespace(seq_idx=object())
        inputs = LayerAuxInputs(
            packed_seq_params=packed,
            packed_seq_params_by_layout={"default": packed},
            cp_layout_plan=object(),
        )
        stack._cp_layout_manager = Mock()
        stack._cp_layout_manager.build_forward_state.return_value = None
        x = torch.randn(3, 2, 4, requires_grad=True)
        with (
            patch(
                'megatron.core.models.hybrid.hybrid_block.build_packed_sequence_cp_metadata',
                return_value=metadata,
            ),
            patch(
                'megatron.core.recompute.tensor_parallel.checkpoint',
                side_effect=lambda fn, distribute, *args: checkpoint(fn, *args, use_reentrant=True),
            ),
        ):
            result = stack(hidden_states=x, aux_inputs=inputs)
            result.sum().backward()
        stack._cp_layout_manager.build_forward_state.assert_called_once_with(
            packed,
            packed_seq_params_by_layout=inputs.packed_seq_params_by_layout,
            thd_plan=inputs.cp_layout_plan,
        )
        assert inputs.packed_seq_params_by_layout is not None
        assert inputs.cp_layout_plan is not None
        reference = x.detach().clone().requires_grad_(True)
        expected = reference.sin().sin()
        expected.sum().backward()
        assert_bit_exact(result, {}, expected, {})
        assert_bit_exact(x.grad, {}, reference.grad, {})
        for layer in stack.layers:
            assert isinstance(layer.received_inputs, LayerAuxInputs)
            assert layer.received_inputs.packed_sequence_cp_metadata is metadata
            assert layer.received_inputs.packed_seq_params is packed
            assert layer.received_inputs.packed_seq_params_by_layout is None
            assert layer.received_inputs.cp_layout_plan is None


def test_mtp_depth_invokes_the_shared_call_shape():
    seen = []

    def depth(hidden_states, aux_inputs):
        seen.append(aux_inputs)
        return hidden_states.sin(), LayerAuxOutputs()

    layer = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=False, fp8=False, fp4=False),
        mhc_enabled=False,
        mtp_model_layer=depth,
        _concat_embeddings=lambda h, embedding: h + embedding,
        _postprocess=lambda h: h,
    )
    x, embedding = (torch.randn(3, 2, 4), torch.randn(3, 2, 4))
    layout_plan = object()
    aux_inputs = LayerAuxInputs(cp_layout_plan=layout_plan)
    result = MultiTokenPredictionLayer._proj_and_transformer_layer(
        layer, x, aux_inputs, decoder_input=embedding
    )
    assert_bit_exact(result, {}, (x + embedding).sin(), {})
    assert seen[0] is aux_inputs
    assert seen[0].cp_layout_plan is layout_plan


def test_mtp_checkpoint_rebuilds_inputs_from_restored_tensors():
    x, embedding, mask, rope = (torch.randn(2, 4) for _ in range(4))
    packed, layouts = object(), object()
    aux_inputs = LayerAuxInputs(
        attention_mask=mask,
        rotary_pos_emb=rope,
        packed_seq_params=packed,
        packed_seq_params_by_layout=layouts,
    )
    layer = SimpleNamespace(
        config=SimpleNamespace(
            fp8=False,
            fp4=False,
            recompute_method="uniform",
            recompute_num_layers=1,
            distribute_saved_activations=False,
        ),
        mtp_layer_pattern=None,
        _proj_and_transformer_layer=Mock(return_value=x),
    )
    restored = []

    def checkpoint_forward(fn, distribute, *args):
        assert all(value is None or torch.is_tensor(value) for value in args)
        restored.extend(value.clone() if value is not None else None for value in args)
        return fn(*restored)

    with patch(
        "megatron.core.transformer.multi_token_prediction.tensor_parallel.checkpoint",
        side_effect=checkpoint_forward,
    ):
        assert (
            MultiTokenPredictionLayer._checkpointed_forward(
                layer, x, aux_inputs, decoder_input=embedding
            )
            is x
        )
    states, replay_inputs = layer._proj_and_transformer_layer.call_args.args
    assert states is restored[0]
    assert layer._proj_and_transformer_layer.call_args.kwargs["decoder_input"] is restored[1]
    assert replay_inputs.attention_mask is restored[2]
    assert replay_inputs.rotary_pos_emb is restored[6]
    assert replay_inputs.packed_seq_params is packed
    assert replay_inputs.packed_seq_params_by_layout is layouts
    assert aux_inputs.attention_mask is mask and aux_inputs.rotary_pos_emb is rope


def test_mtp_stack_adapter_preserves_layout_inputs():
    stack = bare(HybridStack)
    x = torch.randn(2, 4)
    stack._call_with_bound_args = Mock(return_value=x)
    aux_inputs = LayerAuxInputs(packed_seq_params_by_layout=object(), cp_layout_plan=object())
    assert stack(x, aux_inputs) == (x, LayerAuxOutputs())
    assert stack._call_with_bound_args.call_args.kwargs['aux_inputs'] is aux_inputs


def test_moe_forward_backward_combined_output():
    for shared in (False, True):
        for recompute in (False, True):
            for latent in (False, True):
                module = bare(MoELayer)
                module.config = SimpleNamespace(
                    sequence_parallel=False,
                    moe_latent_size=latent,
                    moe_use_norm_before_up_proj=False,
                    fp8=False,
                    fp4=False,
                )
                module.attn_tp_group = SimpleNamespace(size=lambda: 1)
                module.fwd_execution_map = {'route', 'expert_compute', 'postprocess'}
                module.moe_layer_recompute = recompute
                module.select_token_dispatcher = lambda: None
                module.shared_experts_compute = lambda x: x.sigmoid() if shared else None
                module.route = lambda x, *args, **kwargs: (None, None)
                module.preprocess = lambda x, *args: (x, None)
                module.dispatch = lambda x, probs: (x, probs)
                module.routed_experts_compute = lambda x, probs: (x.sin(), None)
                module.combine = lambda x: x
                module.token_dispatcher = SimpleNamespace(combine_postprocess=lambda x: x)
                if latent:
                    module.fc2_latent_proj = DeferredBiasLinear(4, 4, bias=False)
                with patch(
                    'megatron.core.transformer.moe.moe_layer.tensor_parallel.checkpoint',
                    side_effect=lambda fn, distribute, *args: checkpoint(
                        fn, *args, use_reentrant=True
                    ),
                ):
                    assert_equivalent(module, lambda m, x: m(x)[0])
                module.moe_layer_recompute = False
                x = torch.randn(3, 2, 4)
                expected = module(x)[0]
                assert_bit_exact(apply_output_bias(module(x, LayerAuxInputs())), {}, expected, {})
                assert_bit_exact(
                    apply_output_bias(module.forward(x, LayerAuxInputs())), {}, expected, {}
                )


def test_flex_dispatcher_combines_shared_output_and_preserves_gradients():
    x = torch.randn(3, 4, requires_grad=True)
    shared = torch.randn(3, 4, requires_grad=True)
    expert = SimpleNamespace(
        linear_fc2_forward=Mock(), post_forward_comm=Mock(), get_output=Mock(return_value=shared)
    )
    dispatcher = SimpleNamespace(shared_experts=expert, hidden_shape=(3, 1, 4))
    result = MoEFlexTokenDispatcher.combine_postprocess(dispatcher, x.clone())
    assert_bit_exact(result, {}, (x + shared).view(3, 1, 4), {})
    result.sum().backward()
    assert_bit_exact(x.grad, {}, torch.ones_like(x), {})
    assert_bit_exact(shared.grad, {}, torch.ones_like(shared), {})
    expert.linear_fc2_forward.assert_called_once()
    expert.post_forward_comm.assert_called_once()
    expert.get_output.assert_called_once()


@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("wrapper", [False, True])
def test_input_ids_reach_transformer_layers_independently_of_routing(hash_routing, wrapper):
    x = torch.randn(2, 4)
    inner = bare(TransformerLayer)
    inner.config = SimpleNamespace(cuda_graph_impl="none")
    inner.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=hash_routing))
    inner._call_with_bound_args = Mock(return_value=(x, LayerAuxOutputs()))
    module = mock_mhc_boundary(inner, x) if wrapper else inner
    if wrapper:
        module._call_inner_transformer_layer_without_local_bda = Mock(return_value=None)
    ids = torch.ones(1, 2, dtype=torch.long)
    assert_bit_exact(apply_output_bias(module(x, LayerAuxInputs(input_ids=ids))), {}, x, {})
    assert inner._call_with_bound_args.call_args.kwargs["aux_inputs"].input_ids is ids


def mock_mhc_boundary(inner, x):
    wrapper = bare(HyperConnectionHybridLayer)
    wrapper.inner_layer = inner
    wrapper.config = SimpleNamespace(fp32_residual_connection=False)
    wrapper.hyper_connection = Mock(return_value=(x, None, None, x))
    wrapper.hyper_connection.fused_h_res_h_post_bda.side_effect = (
        lambda h_res, residual, h_post, output, **kwargs: residual + output[0]
    )
    return wrapper


@pytest.mark.parametrize("call_style", ["inputs_keyword", "shared", "direct_forward"])
def test_mhc_fallback_accepts_native_aux_inputs_and_preserves_gradients(call_style):

    class NativeLayer(nn.Module):

        def forward(self, x, aux_inputs):
            self.inputs = aux_inputs
            return x + x.sin(), LayerAuxOutputs()

    inner = NativeLayer()
    x = torch.randn(3, 2, 4, requires_grad=True)
    wrapper = mock_mhc_boundary(inner, x)
    aux_inputs = LayerAuxInputs(
        attention_mask=object(),
        inference_context=object(),
        rotary_pos_emb=object(),
        sequence_len_offset=7,
        packed_seq_params=object(),
        packed_sequence_cp_metadata=object(),
        padding_mask=object(),
        input_ids=object(),
        mhc_recompute_manager=object(),
    )
    call = wrapper.forward if call_style == "direct_forward" else wrapper
    if call_style == "inputs_keyword":
        output, aux_outputs = call(x, inputs=aux_inputs)
    else:
        output, aux_outputs = call(x, aux_inputs)
    assert aux_outputs.bias is None
    assert vars(inner.inputs) == {**vars(aux_inputs), "mhc_recompute_manager": None}
    assert aux_inputs.mhc_recompute_manager is not None
    reference = x.detach().clone().requires_grad_(True)
    expected = reference + ((reference + reference.sin()) - reference)
    output.square().sum().backward()
    expected.square().sum().backward()
    assert_bit_exact(output, {"input": x.grad}, expected, {"input": reference.grad})


@pytest.mark.parametrize("hash_routing", [False, True])
def test_mhc_fast_path_leaves_routing_input_selection_to_the_core(hash_routing):
    x = torch.randn(3, 2, 4)
    inner = bare(TransformerLayer)
    inner.config = SimpleNamespace(bias_dropout_fusion=False)
    inner.self_attention = IdentityOp()
    inner.cross_attention = IdentityOp()
    inner.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=hash_routing))
    inner.hidden_dropout = 0.0
    inner.recompute_pre_mlp_layernorm = False
    inner.mlp_norm_manager = None
    inner._forward_mlp_output_with_bias = Mock(return_value=((x, None), x))
    wrapper = mock_mhc_boundary(inner, x)
    ids = torch.ones(2, 3, dtype=torch.long)
    wrapper(x, LayerAuxInputs(input_ids=ids))
    assert inner._forward_mlp_output_with_bias.call_args.args[1].input_ids is ids


@pytest.mark.parametrize("direct", [False, True])
def test_cross_attention_binds_context_and_preserves_bias(direct):
    from megatron.core.transformer.attention import CrossAttention

    module = bare(CrossAttention)
    x, context, mask, bias = (torch.randn(2, 4) for _ in range(4))
    module.forward_pre_attn_and_core_attn = Mock(return_value=x)
    module.forward_post_core_attn = Mock(return_value=(x, bias))
    call = module.forward if direct else module
    output, aux_outputs = call(
        x, LayerAuxInputs(context=context, context_mask=mask, attention_mask=torch.zeros_like(mask))
    )
    assert output is x and aux_outputs.bias is bias
    kwargs = module.forward_pre_attn_and_core_attn.call_args.kwargs
    assert kwargs["key_value_states"] is context
    assert kwargs["attention_mask"] is mask

    layer = bare(TransformerLayer)
    layer.config = SimpleNamespace(fp32_residual_connection=False, bias_dropout_fusion=False)
    layer.hidden_dropout = 0.0
    layer.pre_cross_attn_layernorm = IdentityOp()
    layer._pre_cross_attn_layernorm_returns_residual = False
    layer.cross_attention = module
    layer.bias_dropout_add_exec_handler = torch.enable_grad
    bda = Mock(side_effect=lambda pair, residual, dropout: residual + pair[0] + pair[1])
    layer.cross_attn_bda = Mock(return_value=bda)

    output, output_context = layer._run_cross_attention(x, context, mask, None)
    assert_bit_exact(output, {}, x + x + bias, {})
    assert output_context is context
    assert bda.call_args.args[0][1] is bias


@pytest.mark.parametrize("cls", [MambaMixer, GatedDeltaNet, GatedDeltaNet2])
@pytest.mark.parametrize("direct_forward", [False, True])
def test_mixer_bindings_ignore_unconsumed_cp_metadata(cls, direct_forward):
    module = bare(cls)
    x = torch.randn(2, 1, 4)
    packed = object()
    if cls is MambaMixer:
        core = module._mamba_chunk = Mock(return_value=x)
        module.out_proj = Mock(return_value=(x, None))
    else:
        core = module.forward_pre_attn_and_core_attn = Mock(return_value=x)
        module.forward_post_core_attn = Mock(return_value=(x, None))
    call = module.forward if direct_forward else module
    output, aux_outputs = call(
        x, LayerAuxInputs(packed_seq_params=packed, packed_sequence_cp_metadata=object())
    )
    assert output is x
    assert aux_outputs.bias is None
    assert core.call_args.kwargs["packed_seq_params"] is packed
    assert "packed_sequence_cp_metadata" not in core.call_args.kwargs


class ExplicitSignatureLayer(MegatronModule):
    """Exercise binding without implementing any LayerAuxInputs-specific method."""

    @bind_layer_interface
    def forward(
        self, hidden_states, attention_mask=None, *, packed_sequence_cp_metadata=None, **kwargs
    ):
        self.seen = (attention_mask, packed_sequence_cp_metadata, kwargs)
        return hidden_states.sin()


@pytest.mark.parametrize("call_style", ["positional", "mixed", "x", "hidden_states", "forward"])
def test_signature_binding_preserves_hooks_gradients_and_legacy_calls(call_style):
    class CustomLayer(ExplicitSignatureLayer):

        def bind_layer_inputs(self, hidden_states, aux_inputs, **kwargs):
            events.append("shared")
            return super().bind_layer_inputs(hidden_states, aux_inputs, **kwargs)

        def bind_layer_outputs(self, result):
            events.append("outputs")
            return super().bind_layer_outputs(result)

    module = bare(CustomLayer)
    x = torch.randn(2, 4, requires_grad=True)
    mask, metadata, unrelated = object(), object(), object()
    aux_inputs = LayerAuxInputs(
        attention_mask=mask, packed_sequence_cp_metadata=metadata, input_ids=unrelated
    )
    events = []

    def before(module, args, kwargs):
        events.append("before")
        assert len(args) == 1 and args[0] is x
        assert kwargs["attention_mask"] is mask
        assert kwargs["packed_sequence_cp_metadata"] is metadata
        assert "input_ids" not in kwargs  # **kwargs does not opt into unrelated fields.
        return (x * 2,), kwargs

    def after(module, args, kwargs, output):
        events.append("after")
        assert isinstance(output, torch.Tensor)  # Normalization happens after module hooks.
        return output * 3

    pre = module.register_forward_pre_hook(before, with_kwargs=True)
    post = module.register_forward_hook(after, with_kwargs=True)
    # No signature inspection or field enumeration is allowed in the hot path.
    with patch("inspect.signature", side_effect=AssertionError("runtime inspection")):
        if call_style == "positional":
            result = module(x, aux_inputs)
        elif call_style == "mixed":
            result = module(x, inputs=aux_inputs)
        elif call_style == "forward":
            result = module.forward(x, aux_inputs)
        else:
            result = module(**{call_style: x, "inputs": aux_inputs})
    assert len(result) == 2 and result[1].bias is None
    assert module.seen == (mask, metadata, {})
    expected = x.sin() if call_style == "forward" else (x * 2).sin() * 3
    assert_bit_exact(result[0], {}, expected, {})
    assert events == (
        ["shared", "outputs"]
        if call_style == "forward"
        else ["shared", "before", "after", "outputs"]
    )
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    result[0].sum().backward()
    assert_bit_exact(x.grad, {}, expected_grad, {})
    pre.remove()
    post.remove()
    events.clear()
    # Legacy calls retain positional arguments, arbitrary kwargs, and the tensor result.
    old = module(x, mask, packed_sequence_cp_metadata=metadata, unrelated=unrelated)
    assert isinstance(old, torch.Tensor)
    assert events == []
    assert_bit_exact(old, {}, x.sin(), {})
    assert module.seen == (mask, metadata, {"unrelated": unrelated})
    assert inspect.signature(module.forward) == inspect.signature(
        module.forward.__wrapped__
    ).replace(
        parameters=list(inspect.signature(module.forward.__wrapped__).parameters.values())[1:]
    )


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((torch.ones(1), LayerAuxInputs(), None), {}),
        ((torch.ones(1), LayerAuxInputs()), {"attention_mask": None}),
        ((torch.ones(1), LayerAuxInputs()), {"inputs": LayerAuxInputs()}),
        ((torch.ones(1),), {"inputs": object()}),
        ((torch.ones(1),), {"inputs": None}),
        ((), {"inputs": LayerAuxInputs()}),
        ((), {"x": torch.ones(1), "hidden_states": torch.ones(1), "inputs": LayerAuxInputs()}),
    ],
)
def test_shared_call_rejects_ambiguous_arguments_before_execution(args, kwargs):
    module = bare(ExplicitSignatureLayer)
    module._call_with_bound_args = Mock()
    with pytest.raises(TypeError):
        module(*args, **kwargs)
    module._call_with_bound_args.assert_not_called()


def test_adapter_validates_declarations_and_requires_a_forwarding_schema():
    with pytest.raises(TypeError, match="explicit hidden-state"):
        bind_layer_interface(lambda self, *args, **kwargs: None)
    with pytest.raises(ValueError, match="Required parameter"):
        bind_layer_interface(lambda self, x, required: x)


@pytest.mark.parametrize("mode", ["eager", "local", "te_capture", "te_replay"])
@pytest.mark.parametrize("output_kind", ["tensor", "pair", "bias", "aux_outputs"])
def test_shared_binding_surrounds_graph_dispatch_without_recursion(mode, output_kind):
    bias = torch.randn(4, requires_grad=True) if output_kind == "bias" else None

    class Layer(GraphableMegatronModule):

        def bind_layer_inputs(self, hidden_states, aux_inputs, **kwargs):
            if aux_inputs.context is not None:
                raise ValueError("Unsupported context")
            return super().bind_layer_inputs(hidden_states, aux_inputs, **kwargs)

        @bind_layer_interface
        def forward(self, hidden_states, attention_mask=None):
            output = hidden_states.sin()
            if output_kind == "aux_outputs":
                return output, LayerAuxOutputs(mhc_multistream=hidden_states.cos())
            return output if output_kind == "tensor" else (output, bias)

    module = bare(Layer)
    x = torch.randn(2, 4, requires_grad=True)
    mask = torch.ones(2, 2)
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        assert not args
        assert kwargs["hidden_states"] is x
        assert kwargs["attention_mask"] is mask
        assert "inputs" not in kwargs
        return nn.Module.__call__(module, *args, **kwargs)

    module._should_call_local_cudagraph = Mock(return_value=mode == "local")
    module._should_call_te_cudagraph = Mock(return_value=mode.startswith("te_"))
    module.cuda_graphs = [object()] if mode == "te_replay" else []
    module.cudagraph_manager = Mock(side_effect=lambda mod, a, kw: execute(*a, **kw))
    module._te_cuda_graph_capture = Mock(side_effect=execute)
    module._te_cuda_graph_replay = Mock(side_effect=execute)
    seen = []
    module.register_forward_hook(lambda mod, args, result: seen.append(result))
    for _ in range(2):
        result = module(x, LayerAuxInputs(attention_mask=mask))
        assert (
            len(result) == 2 and result[1].bias is bias
        )  # Both eager and graph output obey the shared contract.
        assert_bit_exact(result[0], {}, x.sin(), {})
        if output_kind == "tensor":
            assert result[0] is seen[-1]
        elif output_kind == "aux_outputs":
            assert result[1] is seen[-1][1]
            assert_bit_exact(result[1].mhc_multistream, {}, x.cos(), {})
        else:
            assert result[0] is seen[-1][0]
            assert result[1].bias is seen[-1][1]
    assert len(seen) == 2
    assert len(calls) == (0 if mode == "eager" else 2)
    with pytest.raises(ValueError, match="Unsupported context"):
        module(x, LayerAuxInputs(attention_mask=mask, context=object()))
    assert len(seen) == 2
    assert len(calls) == (0 if mode == "eager" else 2)
    loss = apply_output_bias(result).sum()
    expected_grad = x.detach().cos()
    if result[1].mhc_multistream is not None:
        loss = loss + result[1].mhc_multistream.sum()
        expected_grad = expected_grad - x.detach().sin()
    loss.backward()
    assert_bit_exact(x.grad, {}, expected_grad, {})
    if bias is not None:
        assert_bit_exact(bias.grad, {}, torch.full_like(bias, x.shape[0]), {})
    if mode == "local":
        module._should_call_te_cudagraph.assert_not_called()
    else:
        module.cudagraph_manager.assert_not_called()


def test_transformer_local_graph_inputs_exclude_unconsumed_scheduling_state():
    module = bare(TransformerLayer)
    x, rope = torch.randn(2, 4), object()
    inputs = LayerAuxInputs(rotary_pos_emb=rope, embedding=object(), execution_context=object())
    module.cudagraph_manager = Mock(return_value=(x, LayerAuxOutputs()))
    module(x, inputs)
    _, args, kwargs = module.cudagraph_manager.call_args.args
    assert args == ()
    assert kwargs["hidden_states"] is x
    assert kwargs["rotary_pos_emb"] is rope
    assert "aux_inputs" not in kwargs
    assert "embedding" not in kwargs
    assert "execution_context" not in kwargs
    assert inputs.embedding is not None and inputs.execution_context is not None


def test_mamba_local_graph_carries_prepared_aux_inputs():
    module = bare(MambaLayer)
    x = torch.randn(2, 4)
    inputs = LayerAuxInputs(
        rotary_pos_emb=torch.randn(2, 4), input_ids=torch.ones(2, dtype=torch.long)
    )
    module.cudagraph_manager = Mock(return_value=x)
    module(x, inputs)
    _, args, kwargs = module.cudagraph_manager.call_args.args
    assert args == ()
    assert set(kwargs) == {"hidden_states", "aux_inputs"}
    assert kwargs["hidden_states"] is x
    assert kwargs["aux_inputs"] is inputs


@pytest.mark.parametrize("call_style", ["module", "forward", "inputs_keyword"])
def test_binding_passes_whole_container_without_reentering_override(call_style):
    events = []

    class Wrapper(MegatronModule):

        def bind_layer_inputs(self, hidden_states, aux_inputs, **kwargs):
            events.append(aux_inputs)
            return super().bind_layer_inputs(hidden_states, aux_inputs, **kwargs)

        @bind_layer_interface
        def forward(self, hidden_states, aux_inputs: LayerAuxInputs):
            self.received_inputs = aux_inputs
            return hidden_states.sin()

    module = bare(Wrapper)
    x = torch.randn(2, 4, requires_grad=True)
    aux_inputs = LayerAuxInputs(context=object(), input_ids=object())
    hooks = []
    module.register_forward_pre_hook(
        lambda mod, args, kwargs: hooks.append(kwargs["aux_inputs"]), with_kwargs=True
    )
    call = module.forward if call_style == "forward" else module
    output, aux_outputs = (
        call(x, inputs=aux_inputs) if call_style == "inputs_keyword" else call(x, aux_inputs)
    )
    assert module.received_inputs is aux_inputs
    assert len(events) == 1 and events[0] is aux_inputs
    assert hooks == ([] if call_style == "forward" else [aux_inputs])
    assert aux_outputs.bias is None
    output.sum().backward()
    assert_bit_exact(x.grad, {}, x.detach().cos(), {})


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
    output, aux_outputs = module.forward_post_core_attn(*state)
    assert events == ["bind_inputs", "bind_outputs"]
    assert_bit_exact(output, {}, 2 * x + (2 * x).sin(), {})
    assert aux_outputs.bias is mask


def test_inherited_adapter_and_override_keep_distinct_signatures():
    class Inherited(ExplicitSignatureLayer):
        pass

    class Override(ExplicitSignatureLayer):

        @bind_layer_interface
        def forward(self, hidden_states, input_ids=None):
            self.ids = input_ids
            return hidden_states, None

    x, ids = torch.ones(2, 4), object()
    assert len(bare(Inherited)(x, LayerAuxInputs(input_ids=ids))) == 2
    module = bare(Override)
    assert module(x, LayerAuxInputs(input_ids=ids)) == (x, LayerAuxOutputs())
    assert module.ids is ids


@pytest.mark.parametrize("inplace", [False, True])
def test_decorated_layer_compiles_with_shared_inputs(inplace):
    module = bare(ExplicitSignatureLayer)
    x = torch.randn(2, 4, requires_grad=True)
    if inplace:
        module.compile(backend="eager", fullgraph=True)
        compiled = module
    else:
        compiled = torch.compile(module, backend="eager", fullgraph=True)
    output = compiled(x, LayerAuxInputs())[0]
    assert_bit_exact(output, {}, x.sin(), {})
    output.sum().backward()
    assert_bit_exact(x.grad, {}, x.detach().cos(), {})


def test_cross_attention_preserves_legacy_execution_and_two_stage_capability():
    from megatron.core.transformer.attention import CrossAttention

    module = bare(CrossAttention)
    x = torch.randn(2, 4)
    module.forward_pre_attn_and_core_attn = Mock(return_value=x)
    module.forward_post_core_attn = Mock(return_value=(x, None))
    assert module.supports_two_stage_attention()
    assert module(x, attention_mask=None) == (x, None)


def test_shared_core_preserves_full_backward_input_hooks():
    module = bare(ExplicitSignatureLayer)
    x = torch.randn(2, 4, requires_grad=True)
    seen = []

    def backward(module, grad_inputs, grad_outputs):
        assert len(grad_inputs) == 1
        assert grad_inputs[0] is not None
        seen.append(grad_inputs[0].clone())
        return (grad_inputs[0] * 2,)

    module.register_full_backward_hook(backward)
    module(x, LayerAuxInputs())[0].sum().backward()
    assert len(seen) == 1
    assert_bit_exact(seen[0], {}, x.detach().cos(), {})
    assert_bit_exact(x.grad, {}, x.detach().cos() * 2, {})


def test_mhc_per_call_setup_runs_before_graph_dispatch():
    from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer

    module = bare(HyperConnectionTransformerLayer)
    x, manager = torch.randn(2, 4), object()
    seen = []

    def graph(module, args, kwargs):
        seen.append(module._mhc_recompute_manager)
        assert "mhc_recompute_manager" not in kwargs
        assert "aux_inputs" not in kwargs
        return x, LayerAuxOutputs()

    module.cudagraph_manager = Mock(side_effect=graph)
    assert module(hidden_states=x, aux_inputs=LayerAuxInputs(mhc_recompute_manager=manager)) == (
        x,
        LayerAuxOutputs(),
    )
    assert module(x, LayerAuxInputs(mhc_recompute_manager=manager)) == (x, LayerAuxOutputs())
    assert module(x, LayerAuxInputs()) == (x, LayerAuxOutputs())
    assert seen == [manager, manager, None]

    def attention(*args, **kwargs):
        assert module._mhc_recompute_manager is manager
        assert "mhc_recompute_manager" not in kwargs
        return x, None

    module.layer_number = 1
    module._forward_attention = Mock(side_effect=attention)
    module._forward_mlp = Mock(return_value=x)
    inputs = LayerAuxInputs(mhc_recompute_manager=manager)
    assert module.forward(x, inputs) == (x, LayerAuxOutputs())
    assert module._forward_attention.call_count == 1
    assert module._forward_attention.call_args.kwargs["aux_inputs"] is inputs
    assert module._forward_mlp.call_count == 1
    assert module._forward_mlp.call_args.kwargs["aux_inputs"] is inputs
    assert seen == [manager, manager, None]


def test_te_capture_and_replay_use_legacy_tensors_and_manual_hooks():
    class Layer(GraphableMegatronModule):

        @bind_layer_interface
        def forward(self, hidden_states, attention_mask=None):
            self.forward_count += 1
            return hidden_states.sin()

    module = bare(Layer)
    module.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine", fine_grained_activation_offloading=False
    )
    module.forward_count = 0
    module.cuda_graphs = []
    x = torch.randn(2, 4)
    aux_inputs = LayerAuxInputs(attention_mask=torch.ones(2, 2))
    # Exercise the real TE dispatch and capture methods with an ordinary CPU forward.
    with patch("megatron.core.transformer.cuda_graphs.is_graph_capturing", return_value=True):
        assert_bit_exact(module(x, aux_inputs)[0], {}, x.sin(), {})
    hook = Mock()
    module.cuda_graph_manual_hooks = [(hook, (module,))]
    graph = Mock(side_effect=lambda hidden_states, **kwargs: hidden_states.sin())
    module.cuda_graphs = [graph]
    for microbatch in (0, 1):
        module.current_microbatch = microbatch
        result = module(x, aux_inputs)
        assert len(result) == 2 and result[1].bias is None
        assert_bit_exact(result[0], {}, x.sin(), {})
        assert graph.call_args.args == (x,)
        assert graph.call_args.kwargs == {
            "attention_mask": aux_inputs.attention_mask,
            "is_first_microbatch": microbatch == 0,
        }
    assert module.forward_count == 1
    assert hook.call_count == 2


@pytest.mark.parametrize(
    "cls", [TransformerLayer, MambaLayer, HybridStack, HyperConnectionHybridLayer]
)
def test_complete_layer_output_does_not_return_inputs(cls):
    module = bare(cls)
    if cls is MambaLayer:
        module.config = SimpleNamespace(linear_cp_mode="headwise")
    x, context = torch.randn(2, 4), torch.randn(2, 4)
    if cls in (MambaLayer, HybridStack, HyperConnectionHybridLayer):
        legacy_output = x
    else:
        legacy_output = (x, context)
    module._call_with_bound_args = Mock(return_value=legacy_output)
    result = module(x, LayerAuxInputs())
    assert isinstance(result, tuple) and len(result) == 2
    assert result[0] is x and result[1].bias is None
    assert module(x) is legacy_output


@pytest.mark.parametrize(
    "cls",
    [
        MLP,
        MoELayer,
        SelfAttention,
        MLASelfAttention,
        AbsorbedMLASelfAttention,
        DSv4HybridSelfAttention,
        MambaMixer,
        GatedDeltaNet,
        GatedDeltaNet2,
        GatedDeltaProductMixer,
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_core_output_preserves_hidden_state_and_bias_identity(cls, with_bias):
    module = bare(cls)
    x = torch.randn(2, 4)
    bias = torch.randn(4) if with_bias else None
    legacy_output = (x, bias)
    module._call_with_bound_args = Mock(return_value=legacy_output)
    result = module(x, LayerAuxInputs())
    assert result[0] is x and result[1].bias is bias


@pytest.mark.parametrize("with_bias", [False, True])
def test_offload_preserves_deferred_bias(with_bias):
    x = torch.randn(2, 4, requires_grad=True)
    bias = torch.randn(4, requires_grad=True) if with_bias else None
    manager = SimpleNamespace(group_offload=Mock(side_effect=lambda value, **kw: value.clone()))
    output = TransformerLayer._group_offload_output_with_bias((x, bias), manager, [x])
    assert output[1] is bias
    manager.group_offload.assert_called_once_with(x, forced_released_tensors=[x])
    apply_output_bias(output).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    if with_bias:
        assert torch.equal(bias.grad, torch.full_like(bias, 2))


def test_stack_preserves_mhc_stream_in_named_aux_outputs():
    module = bare(HybridStack)
    x, stream = torch.randn(2, 4), torch.randn(2, 16)
    inputs = LayerAuxInputs()
    natural_output = (x, stream)
    module._call_with_bound_args = Mock(return_value=natural_output)
    result = module(x, inputs)
    assert result[0] is x
    assert result[1].bias is None and result[1].mhc_multistream is stream
    assert module(hidden_states=x, aux_inputs=inputs) is natural_output


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("routing", ["dense", "learned", "hash"])
def test_transformer_te_replay_filters_none_after_binding(present, routing):
    module = bare(TransformerLayer)
    module.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[],
        fine_grained_activation_offloading=False,
        delay_offload_until_cuda_graph=False,
        overlap_moe_expert_parallel_comm=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
    )
    module.is_moe_layer = routing != "dense"
    module.mlp = (
        SimpleNamespace(router=SimpleNamespace(is_hash_layer=routing == "hash"))
        if module.is_moe_layer
        else SimpleNamespace()
    )
    x = torch.randn(2, 1, 4)
    mask = torch.zeros(1, 1, 2, 2)
    value = torch.randn(2, 1, 4) if present else None
    optional = dict.fromkeys(
        (
            "context",
            "context_mask",
            "rotary_pos_cos",
            "rotary_pos_sin",
            "rotary_pos_cos_sin",
            "attention_bias",
            "residual_stream_recompute_context",
            "input_ids",
        ),
        value,
    )
    inputs = LayerAuxInputs(attention_mask=mask if present else None, **optional)
    graph = Mock(return_value=(x, value) if present else (x,))
    module.cuda_graphs = [graph]
    module.cuda_graph_manual_hooks = []
    module._te_cuda_graph_padding_mask = Mock(return_value=None)
    module.bind_layer_inputs = Mock(wraps=module.bind_layer_inputs)
    module._te_cuda_graph_replay = Mock(wraps=module._te_cuda_graph_replay)
    te = Mock()
    with (
        patch.dict(
            "sys.modules",
            {"transformer_engine": Mock(pytorch=te), "transformer_engine.pytorch": te},
        ),
        patch("megatron.core.transformer.transformer_layer.is_te_min_version", return_value=True),
        patch("megatron.core.transformer.transformer_layer.torch.zeros", return_value=mask),
    ):
        output, aux_outputs = module(x, inputs)
    assert output is x and aux_outputs.bias is None
    module.bind_layer_inputs.assert_called_once()
    assert module._te_cuda_graph_replay.call_args.kwargs["aux_inputs"] is inputs
    assert graph.call_args.args == (x,)
    kwargs = graph.call_args.kwargs
    assert kwargs["attention_mask"] is mask
    assert "aux_inputs" not in kwargs
    assert all(value is not None for value in kwargs.values())
    for name in optional:
        if present and (name != "input_ids" or routing == "hash"):
            assert kwargs[name] is value
        else:
            assert name not in kwargs


def test_mamba_te_capture_and_replay_adapt_only_at_graph_boundary():
    class Layer(MambaLayer):
        @bind_layer_interface
        def forward(self, hidden_states, aux_inputs):
            self.received_inputs = aux_inputs
            return hidden_states.sin()

    module = bare(Layer)
    module.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine", fine_grained_activation_offloading=False
    )
    module.cuda_graphs = []
    module.cuda_graph_manual_hooks = []
    x = torch.randn(2, 4)
    with patch("megatron.core.transformer.cuda_graphs.is_graph_capturing", return_value=True):
        # TE's static capture signature supplies only hidden states.
        captured = module(x)
    torch.testing.assert_close(captured, x.sin())
    assert isinstance(module.received_inputs, LayerAuxInputs)

    inputs = LayerAuxInputs(embedding=object(), execution_context=object())
    graph = Mock(return_value=captured)
    module.cuda_graphs = [graph]
    output, aux_outputs = module(x, inputs)
    assert output is captured and aux_outputs.bias is None
    assert graph.call_args.args == (x,)
    assert graph.call_args.kwargs == {"is_first_microbatch": True}
    assert inputs.embedding is not None and inputs.execution_context is not None


@pytest.mark.parametrize(
    "field",
    [
        "inference_context",
        "packed_seq_params",
        "packed_sequence_cp_metadata",
        "residual_stream_recompute_context",
    ],
)
def test_mamba_te_replay_rejects_non_tensor_execution_state(field):
    module = bare(MambaLayer)
    with pytest.raises(AssertionError, match="CUDA graph accepts only Tensor inputs"):
        module._te_cuda_graph_replay(
            hidden_states=torch.randn(2, 4), aux_inputs=LayerAuxInputs(**{field: object()})
        )


@pytest.mark.parametrize("with_context", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_transformer_te_capture_preserves_raw_intermediates(with_context, partial):
    module = bare(TransformerLayer)
    module.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router] if partial else [],
    )
    module.cuda_graphs = []
    module.is_moe_layer = partial
    module.offload_module_in_cuda_graph = False
    x = torch.randn(2, 4)
    context = torch.randn(3, 4) if with_context else None
    intermediates = (x, torch.randn(2, 4), torch.randn(2, 4)) if partial else (x,)
    module._forward_attention = Mock(return_value=(x, context))
    module._forward_mlp = Mock(return_value=intermediates if partial else x)

    # TE's capture helper invokes the module with explicit tensor arguments.
    with patch("megatron.core.transformer.cuda_graphs.is_graph_capturing", return_value=True):
        captured = module(x, context=context, aux_inputs=LayerAuxInputs(context=context))
    assert "aux_inputs" not in module._forward_attention.call_args.kwargs
    assert all(value is not None for value in module._forward_attention.call_args.kwargs.values())
    expected = intermediates + ((context,) if with_context else ())
    assert len(captured) == len(expected)
    assert all(actual is wanted for actual, wanted in zip(captured, expected))


@pytest.mark.parametrize("with_context", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_transformer_te_replay_returns_named_layer_outputs(with_context, partial):
    module = bare(TransformerLayer)
    module.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn] if partial else [],
        delay_offload_until_cuda_graph=False,
        overlap_moe_expert_parallel_comm=False,
        fine_grained_activation_offloading=False,
    )
    module.is_moe_layer = False
    module.cuda_graph_manual_hooks = [(Mock(), ())]
    module._te_cuda_graph_padding_mask = Mock(return_value=None)
    # Use the tensor argument preparation without importing TE in this mocked check.
    module._get_te_cuda_graph_replay_args = (
        GraphableMegatronModule._get_te_cuda_graph_replay_args.__get__(module)
    )
    x, completed = torch.randn(2, 4), torch.randn(2, 4)
    context = torch.randn(3, 4) if with_context else None
    captured = (x,) + ((context,) if with_context else ())
    module.cuda_graphs = [Mock(return_value=captured)]
    module._forward_mlp = Mock(return_value=completed)

    output, aux_outputs = module(x, LayerAuxInputs(context=context))
    assert output is (completed if partial else x)
    assert aux_outputs.bias is None
    module.cuda_graph_manual_hooks[0][0].assert_called_once_with()
    if partial:
        module._forward_mlp.assert_called_once_with(x, padding_mask=None, input_ids=None)
    else:
        module._forward_mlp.assert_not_called()


@pytest.mark.parametrize("output_kind", ["tensor", "singleton", "bias", "none", "aux_outputs"])
def test_local_graph_record_and_replay_preserve_return_structure(output_kind):
    x = torch.randn(2, 4)
    bias = torch.randn(4)
    stream = torch.randn(2, 16)
    legacy_output = {
        "tensor": x,
        "singleton": (x,),
        "bias": (x, bias),
        "none": (x, None),
        "aux_outputs": (x, LayerAuxOutputs(bias=bias, mhc_multistream=stream)),
    }[output_kind]
    runner = object.__new__(_CudaGraphRunner)
    runner.func = lambda: legacy_output
    runner.status = _GraphStatus.FWD_READY
    runner.fwd_graph_recorded = True
    runner._make_pipeline_output_viewless = lambda output: output
    recorded = runner.record_graph_capture((), {})

    runner.fwd_graph_outputs = legacy_output
    runner._gtp_fwd_params_to_ensure_ready = ()
    runner.grad_enabled = False
    runner.get_mismatch_errors = lambda args, kwargs: []
    runner.get_tensors = lambda *args, **kwargs: ()
    tensors = (x, bias) if output_kind == "bias" else (x,)
    if output_kind == "aux_outputs":
        tensors = (x, bias, stream)
    with patch.object(_CudagraphReplayNode, "apply", return_value=tensors):
        replayed = runner.replay_graph_capture(False, (), {})

    assert type(recorded) is type(replayed) is type(legacy_output)
    if output_kind == "tensor":
        assert_bit_exact(recorded, {}, replayed, {})
    else:
        assert len(recorded) == len(replayed) == len(legacy_output)
        for actual, expected in zip(recorded, replayed):
            if expected is None:
                assert actual is None
            elif isinstance(expected, LayerAuxOutputs):
                for name in ("bias", "mhc_multistream"):
                    assert_bit_exact(getattr(actual, name), {}, getattr(expected, name), {})
            else:
                assert_bit_exact(actual, {}, expected, {})


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_default_binding_preserves_inputs_and_returns_only_outputs(direct, with_bias):
    bias = torch.randn(4) if with_bias else None

    class Layer(MegatronModule):
        @bind_layer_interface
        def forward(self, hidden_states):
            return (hidden_states.sin(), bias) if with_bias else hidden_states.sin()

    layer = bare(Layer)
    inputs = LayerAuxInputs(context=torch.randn(2, 4), input_ids=torch.ones(2, 4))
    x = torch.randn(2, 4)
    call = layer.forward if direct else layer
    output, aux_outputs = call(x, inputs)
    assert inputs.context is not None
    assert aux_outputs.bias is bias
    assert_bit_exact(output, {}, x.sin(), {})
    assert inputs.update(context=inputs.context) is inputs
    updated = inputs.update(context=None)
    assert updated is not inputs and updated.context is None
    assert inputs.context is not None and updated.input_ids is inputs.input_ids


@pytest.mark.parametrize("method", ["uniform", "block"])
def test_checkpoint_reuses_inputs_between_chunks(method):
    class Layer(nn.Module):
        layer_number = 1

        def forward(self, hidden_states, aux_inputs):
            output = hidden_states + aux_inputs.attention_bias
            return (output, LayerAuxOutputs())

    stack = SimpleNamespace(
        layers=[Layer(), Layer()],
        num_layers_per_pipeline_rank=2,
        config=SimpleNamespace(
            fp8=False,
            fp4=False,
            recompute_method=method,
            recompute_num_layers=1,
            distribute_saved_activations=False,
        ),
    )
    x = torch.randn(2, 4, requires_grad=True)
    bias = torch.randn(2, 4, requires_grad=True)
    inputs = LayerAuxInputs(attention_bias=bias)
    with patch(
        "megatron.core.recompute.tensor_parallel.checkpoint",
        side_effect=lambda fn, distribute, *args: checkpoint(fn, *args, use_reentrant=True),
    ):
        output, extracted = checkpointed_forward(
            stack, x, inputs, use_inner_quantization_context=False
        )
        output.sum().backward()
    assert not extracted
    assert inputs.attention_bias is bias
    assert_bit_exact(output, {}, x + bias + bias, {})
    assert_bit_exact(x.grad, {}, torch.ones_like(x), {})
    assert_bit_exact(bias.grad, {}, torch.full_like(bias, 2), {})


@pytest.mark.parametrize("triple", [False, True])
def test_default_binding_rejects_returned_inputs(triple):
    class Layer(MegatronModule):
        @bind_layer_interface
        def forward(self, hidden_states, aux_inputs):
            if triple:
                return hidden_states, LayerAuxOutputs(), aux_inputs
            return hidden_states, aux_inputs

    with pytest.raises((TypeError, ValueError)):
        bare(Layer)(torch.ones(2, 4), LayerAuxInputs())
