# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Input adapter and arithmetic equivalence checks, using mocked cores on CPU."""

import copy
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
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from megatron.core.transformer.layer_interface import LayerInputs, LayerInterface, as_addends
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule
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


def add_updates(updates):
    result = updates[0]
    for update in updates[1:]:
        result = result + update
    return result


def test_frozen_inputs_and_bias_identity():
    inputs = LayerInputs()
    with pytest.raises(FrozenInstanceError):
        inputs.input_ids = torch.ones(1)
    x, bias = (torch.randn(2, 4), torch.randn(4))
    assert len(as_addends((x, None))) == 1
    assert as_addends((x, bias))[0] is x
    assert as_addends((x, bias))[1] is bias
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


def assert_equivalent(module, legacy, inputs=LayerInputs()):
    other = copy.deepcopy(module)
    x = torch.randn(3, 2, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    old = legacy(module, x)
    new = add_updates(other(y, inputs))
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
        assert_equivalent(module, lambda m, x: add_updates(as_addends(m(x))))


def test_attention_adapter_threads_every_execution_input():
    module = bare(SelfAttention)
    output, bias = (torch.randn(2, 4), torch.randn(4))
    module.forward_pre_attn_and_core_attn = Mock(return_value=output)
    module.forward_post_core_attn = Mock(return_value=(output, bias))
    mask, rope, metadata, packed, context = (object() for _ in range(5))
    inputs = LayerInputs(
        attention_mask=mask,
        rotary_pos_emb=rope,
        cp_metadata=metadata,
        packed_seq_params=packed,
        inference_context=context,
        sequence_len_offset=3,
    )
    result = module(output, inputs)
    assert result[0] is output
    assert result[1] is bias
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
    assert module(hidden_states=output, inputs=inputs)[0] is output
    assert module(x=output, inputs=inputs)[0] is output
    with pytest.raises(TypeError):
        module(output, inputs, attention_mask=mask)


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
    inputs = LayerInputs(
        attention_mask=object(),
        packed_seq_params=object(),
        inference_context=object(),
        sequence_len_offset=7,
    )
    module = bare(cls)
    module.forward = Mock(return_value=(x, bias))
    result = module.forward_layer(x, inputs)
    assert result[0] is x
    assert result[1] is bias
    kwargs = module.forward.call_args.kwargs
    assert kwargs['inference_context'] is inputs.inference_context
    assert kwargs['packed_seq_params'] is inputs.packed_seq_params


def test_new_core_implements_interface_without_legacy_adapter_or_layer_wrapper():

    class Core(GraphableMegatronModule, LayerInterface):

        def __init__(self):
            super().__init__(TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1))
            self.proj = nn.Linear(4, 4)

        def forward(self, x, inputs: LayerInputs):
            return (self.proj(x),)

    core = Core()
    assert isinstance(core, LayerInterface)
    assert not hasattr(core, 'forward_layer')
    x = torch.randn(3, 2, 4, requires_grad=True)
    inputs = LayerInputs(input_ids=torch.ones(2, 3, dtype=torch.long))
    result = add_updates(core(x, inputs))
    assert_bit_exact(core(x=x, inputs=inputs)[0], {}, result, {})
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
        inputs = LayerInputs(packed_seq_params=packed, cp_metadata=metadata)
        assert_equivalent(
            core,
            lambda m, x: add_updates(
                as_addends(m(x, packed_seq_params=packed, packed_sequence_cp_metadata=metadata))
            ),
            inputs,
        )
        core._gdp_chunk_forward = Mock(side_effect=lambda x, **kwargs: x.sin())
        x = torch.randn(3, 2, 4)
        updates = core(x, inputs)
        kwargs = core._gdp_chunk_forward.call_args.kwargs
        assert kwargs['packed_seq_params'] is packed
        assert kwargs['packed_sequence_cp_metadata'] is metadata
        assert len(updates) == (2 if bias else 1)
        if bias:
            assert updates[1] is core.out_proj.bias


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
                updates = core(x, LayerInputs(inference_context=context))
            assert updates[0] is legacy[0]
            assert len(updates) == (1 if bias is None else 2)
            if bias is not None:
                assert updates[1] is bias
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
        core(torch.randn(3, 2, 4), LayerInputs(inference_context=object()))


def test_skeleton_adapters_preserve_recompute_and_mhc_state():
    x = torch.randn(2, 4)
    replay, manager, metadata = (object(), object(), object())
    inputs = LayerInputs(
        cp_metadata=metadata, input_ids=object(), residual_stream_recompute_context=replay
    )
    for cls in (TransformerLayer, MambaLayer):
        module = bare(cls)
        module.config = SimpleNamespace(cuda_graph_impl='none')
        module.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=True))
        module.forward = Mock(return_value=(x, None) if cls is TransformerLayer else x)
        assert module(x, inputs) is x
        kwargs = module.forward.call_args.kwargs
        assert kwargs['residual_stream_recompute_context'] is replay
        if cls is MambaLayer:
            assert kwargs['packed_sequence_cp_metadata'] is metadata
        else:
            assert kwargs['input_ids'] is inputs.input_ids
    wrapper = bare(HyperConnectionHybridLayer)
    wrapper.inner_layer = SimpleNamespace()
    wrapper.forward = Mock(return_value=(x, None))
    assert wrapper.forward_layer(x, LayerInputs(mhc_recompute_manager=manager)) is x
    assert wrapper.forward.call_args.kwargs['mhc_recompute_manager'] is manager
    with pytest.raises(TypeError, match='mhc'):
        wrapper.forward_layer(x, inputs)


def test_graph_adapter_passes_tensors_to_legacy_graph_manager():
    module = bare(TransformerLayer)
    module.config = SimpleNamespace(cuda_graph_impl='local')
    x = torch.randn(2, 4)
    module.cudagraph_manager = Mock(return_value=(x, None))
    assert module(x, LayerInputs()) is x
    args = module.cudagraph_manager.call_args.args
    assert args[1] == ()
    assert args[2]['hidden_states'] is x
    assert 'inputs' not in args[2]


def test_shortcut_atomic_call_accepts_any_layer_with_shared_inputs():
    x = torch.randn(2, 4)
    layer = Mock(return_value=x)
    unit = SimpleNamespace(compute_layer=layer)
    assert (
        ShortcutMoEBlock._forward_compute_atomic(
            unit,
            x,
            attention_mask=None,
            inference_context=None,
            rotary_pos_emb=None,
            sequence_len_offset=None,
            packed_seq_params=None,
            padding_mask=None,
        )
        is x
    )
    assert isinstance(layer.call_args.args[1], LayerInputs)


@pytest.mark.parametrize("config_cls", [MambaLayerConfig, GDNLayerConfig, TransformerConfig])
@pytest.mark.parametrize("method", ["uniform", "block"])
def test_recompute_calls_shared_inputs_and_replays_gradients(config_cls, method):

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

        def forward(self, x, inputs):
            self.inputs.append(inputs)
            return x.sin()

    layer = Layer()
    stack = SimpleNamespace(
        _uses_layer_inputs=True,
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
    with patch(
        'megatron.core.recompute.tensor_parallel.checkpoint',
        side_effect=lambda fn, distribute, *args: checkpoint(fn, *args, use_reentrant=True),
    ):
        y = checkpointed_forward(
            stack,
            x,
            None,
            None,
            None,
            (None, None),
            None,
            None,
            False,
            input_ids=ids,
            packed_sequence_cp_metadata=metadata,
        )
        y.sum().backward()
    assert_bit_exact(y, {}, x.sin(), {})
    assert_bit_exact(x.grad, {}, x.detach().cos(), {})
    assert len(layer.inputs) == 2
    for inputs in layer.inputs:
        assert inputs.input_ids is ids
        assert inputs.cp_metadata is (metadata if config_cls is MambaLayerConfig else None)


def test_hybrid_stack_eager_and_full_recompute_use_the_same_call():

    class TensorLayer(nn.Module):

        def __init__(self, config, layer_number, **kwargs):
            super().__init__()
            self.config, self.layer_number = (config, layer_number)

        def forward(self, hidden_states, inputs):
            self.assert_inputs = isinstance(inputs, LayerInputs)
            return hidden_states.sin()

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
        group = SimpleNamespace(size=lambda: 1)
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
        x = torch.randn(3, 2, 4, requires_grad=True)
        with patch(
            'megatron.core.recompute.tensor_parallel.checkpoint',
            side_effect=lambda fn, distribute, *args: checkpoint(fn, *args, use_reentrant=True),
        ):
            result = stack(x, attention_mask=None)
            result.sum().backward()
        reference = x.detach().clone().requires_grad_(True)
        expected = reference.sin().sin()
        expected.sum().backward()
        assert_bit_exact(result, {}, expected, {})
        assert_bit_exact(x.grad, {}, reference.grad, {})
        assert all((layer.assert_inputs for layer in stack.layers))


def test_mtp_depth_invokes_the_shared_call_shape():
    seen = []

    def depth(hidden_states, inputs):
        seen.append(inputs)
        return hidden_states.sin()

    layer = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=False, fp8=False, fp4=False),
        mhc_enabled=False,
        mtp_model_layer=depth,
        _concat_embeddings=lambda h, embedding: h + embedding,
        _postprocess=lambda h: h,
    )
    x, embedding = (torch.randn(3, 2, 4), torch.randn(3, 2, 4))
    layout_plan = object()
    result = MultiTokenPredictionLayer._proj_and_transformer_layer(
        layer, x, embedding, cp_layout_plan=layout_plan
    )
    assert_bit_exact(result, {}, (x + embedding).sin(), {})
    assert isinstance(seen[0], LayerInputs)
    assert seen[0].cp_layout_plan is layout_plan


def test_mtp_stack_adapter_preserves_layout_inputs():
    stack = bare(HybridStack)
    x = torch.randn(2, 4)
    stack.forward = Mock(return_value=x)
    inputs = LayerInputs(packed_seq_params_by_layout=object(), cp_layout_plan=object())
    assert stack.forward_layer(x, inputs) is x
    assert stack.forward.call_args.kwargs['cp_layout_plan'] is inputs.cp_layout_plan
    assert (
        stack.forward.call_args.kwargs['packed_seq_params_by_layout']
        is inputs.packed_seq_params_by_layout
    )


def test_moe_forward_backward_shared_addends_and_pre_norm():
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
                module.pre_norm = nn.Identity()
                module.fwd_execution_map = {'route', 'expert_compute', 'postprocess'}
                module.moe_layer_recompute = recompute
                module.select_token_dispatcher = lambda: None
                module.shared_experts_compute = lambda x: x.sigmoid() if shared else None
                module.route = lambda x, *args, **kwargs: (None, None)
                module.preprocess = lambda x, *args: (x, None)
                module.dispatch = lambda x, probs: (x, probs)
                module.routed_experts_compute = lambda x, probs: (x.sin(), None)
                module.combine = lambda x: x
                module.token_dispatcher = SimpleNamespace(
                    combine_postprocess=lambda x, return_shared_expert_output=False: (
                        (x, None) if return_shared_expert_output else x
                    )
                )
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
                module.pre_norm = nn.LayerNorm(4)
                x = torch.randn(3, 2, 4)
                expected = module(module.pre_norm(x))[0]
                assert_bit_exact(add_updates(module(x, LayerInputs())), {}, expected, {})


def test_flex_dispatcher_defers_overlap_add_without_changing_execution():
    for deferred in (False, True):
        x = torch.randn(3, 4, requires_grad=True)
        shared = torch.randn(3, 4, requires_grad=True)
        expert = SimpleNamespace(
            linear_fc2_forward=Mock(),
            post_forward_comm=Mock(),
            get_output=Mock(return_value=shared),
        )
        dispatcher = SimpleNamespace(shared_experts=expert, hidden_shape=(3, 1, 4))
        result = MoEFlexTokenDispatcher.combine_postprocess(
            dispatcher, x.clone(), return_shared_expert_output=deferred
        )
        result = add_updates(result) if deferred else result
        assert_bit_exact(result, {}, (x + shared).view(3, 1, 4), {})
        result.sum().backward()
        assert_bit_exact(x.grad, {}, torch.ones_like(x), {})
        assert_bit_exact(shared.grad, {}, torch.ones_like(shared), {})
        expert.linear_fc2_forward.assert_called_once()
        expert.post_forward_comm.assert_called_once()
        expert.get_output.assert_called_once()


@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("wrapper", [False, True])
def test_input_ids_only_reach_hash_layers(hash_routing, wrapper):
    x = torch.randn(2, 4)
    inner = bare(TransformerLayer)
    inner.config = SimpleNamespace(cuda_graph_impl="none")
    inner.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=hash_routing))
    module = bare(HyperConnectionHybridLayer) if wrapper else inner
    if wrapper:
        module.inner_layer = inner
    module.forward = Mock(return_value=(x, None))
    ids = torch.ones(1, 2, dtype=torch.long)
    assert module.forward_layer(x, LayerInputs(input_ids=ids)) is x
    assert module.forward.call_args.kwargs.get("input_ids") is (ids if hash_routing else None)


def test_cross_attention_rejects_unsupported_shared_inputs():
    from megatron.core.transformer.attention import CrossAttention

    with pytest.raises(NotImplementedError, match="CrossAttention"):
        bare(CrossAttention).forward_layer(torch.randn(2, 4), LayerInputs(context=object()))


def test_mamba_mixer_rejects_chunkwise_metadata():
    with pytest.raises(NotImplementedError, match="packed-sequence chunkwise CP metadata"):
        bare(MambaMixer).forward_layer(torch.randn(2, 4), LayerInputs(cp_metadata=object()))


def test_moe_rejects_intermediate_tensors_with_addends_before_dispatch():
    module = bare(MoELayer)
    with pytest.raises(ValueError, match="intermediate_tensors"):
        module.forward(torch.randn(2, 4), intermediate_tensors=(), _return_addends=True)
