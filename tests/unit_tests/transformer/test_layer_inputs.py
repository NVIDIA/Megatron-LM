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
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.layer_interface import (
    LayerInputs,
    LayerInterface,
    bind_layer_inputs,
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
    hidden_states, bias = output
    assert isinstance(hidden_states, torch.Tensor)
    assert bias is None or isinstance(bias, torch.Tensor)
    return hidden_states if bias is None else hidden_states + bias


def test_frozen_inputs_and_interface():
    inputs = LayerInputs()
    with pytest.raises(FrozenInstanceError):
        inputs.input_ids = torch.ones(1)
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
    new = apply_output_bias(other(y, inputs))
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
    inputs = LayerInputs(
        attention_mask=mask,
        rotary_pos_emb=rope,
        packed_sequence_cp_metadata=metadata,
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
    module._call_with_bound_args = Mock(return_value=(x, bias))
    result = module(x, inputs)
    assert result[0] is x
    assert result[1] is bias
    kwargs = module._call_with_bound_args.call_args.kwargs
    assert kwargs['inference_context'] is inputs.inference_context
    assert kwargs['packed_seq_params'] is inputs.packed_seq_params


def test_new_core_implements_interface_without_legacy_adapter_or_layer_wrapper():

    class Core(GraphableMegatronModule, LayerInterface):

        def __init__(self):
            super().__init__(TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1))
            self.proj = nn.Linear(4, 4)

        def forward(self, x, inputs: LayerInputs):
            return self.proj(x), None

    core = Core()
    assert isinstance(core, LayerInterface)
    assert not hasattr(core, 'forward_layer')
    x = torch.randn(3, 2, 4, requires_grad=True)
    inputs = LayerInputs(input_ids=torch.ones(2, 3, dtype=torch.long))
    result = apply_output_bias(core(x, inputs))
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
        inputs = LayerInputs(packed_seq_params=packed, packed_sequence_cp_metadata=metadata)
        assert_equivalent(
            core,
            lambda m, x: apply_output_bias(
                m(x, packed_seq_params=packed, packed_sequence_cp_metadata=metadata)
            ),
            inputs,
        )
        core._gdp_chunk_forward = Mock(side_effect=lambda x, **kwargs: x.sin())
        x = torch.randn(3, 2, 4)
        updates = core(x, inputs)
        kwargs = core._gdp_chunk_forward.call_args.kwargs
        assert kwargs['packed_seq_params'] is packed
        assert kwargs['packed_sequence_cp_metadata'] is metadata
        assert len(updates) == 2
        assert updates[1] is (core.out_proj.bias if bias else None)


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
            assert len(updates) == 2
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
        packed_sequence_cp_metadata=metadata,
        input_ids=object(),
        residual_stream_recompute_context=replay,
    )
    for cls in (TransformerLayer, MambaLayer):
        module = bare(cls)
        module.config = SimpleNamespace(cuda_graph_impl='none', linear_cp_mode='chunkwise')
        module.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=True))
        module._call_with_bound_args = Mock(
            return_value=(x, None) if cls is TransformerLayer else x
        )
        assert module(x, inputs) == (x, None)
        kwargs = module._call_with_bound_args.call_args.kwargs
        assert kwargs['residual_stream_recompute_context'] is replay
        if cls is MambaLayer:
            assert kwargs['packed_sequence_cp_metadata'] is metadata
        else:
            assert kwargs['input_ids'] is inputs.input_ids
    wrapper = bare(HyperConnectionHybridLayer)
    wrapper.inner_layer = SimpleNamespace()
    wrapper._call_with_bound_args = Mock(return_value=(x, None))
    assert wrapper(x, LayerInputs(mhc_recompute_manager=manager)) == (x, None)
    assert wrapper._call_with_bound_args.call_args.kwargs['mhc_recompute_manager'] is manager
    with pytest.raises(TypeError, match='mhc'):
        wrapper(x, inputs)


def test_graph_adapter_passes_tensors_to_legacy_graph_manager():
    module = bare(TransformerLayer)
    module.config = SimpleNamespace(cuda_graph_impl='local')
    x = torch.randn(2, 4)
    module.cudagraph_manager = Mock(return_value=(x, None))
    assert module(x, LayerInputs()) == (x, None)
    args = module.cudagraph_manager.call_args.args
    assert args[1] == ()
    assert args[2]['hidden_states'] is x
    assert 'inputs' not in args[2]


@pytest.mark.parametrize("mode", ["headwise", "chunkwise"])
@pytest.mark.parametrize("has_metadata", [False, True])
def test_mamba_binding_owns_cp_metadata_selection(mode, has_metadata):
    module = bare(MambaLayer)
    module.config = SimpleNamespace(linear_cp_mode=mode)
    x = torch.randn(2, 4)
    metadata = object() if has_metadata else None
    module._call_with_bound_args = Mock(return_value=x)

    assert module(x, LayerInputs(packed_sequence_cp_metadata=metadata)) == (x, None)
    kwargs = module._call_with_bound_args.call_args.kwargs
    if mode == "chunkwise" and has_metadata:
        assert kwargs["packed_sequence_cp_metadata"] is metadata
    else:
        assert "packed_sequence_cp_metadata" not in kwargs

    # Explicit legacy calls retain their arguments and tensor return value.
    assert module(x, packed_sequence_cp_metadata=metadata) is x
    assert module._call_with_bound_args.call_args.kwargs == {
        "packed_sequence_cp_metadata": metadata
    }


def test_shortcut_atomic_call_accepts_any_layer_with_shared_inputs():
    x = torch.randn(2, 4)
    layer = Mock(return_value=(x, None))
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
            return x.sin(), None

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
        assert inputs.packed_sequence_cp_metadata is metadata


def test_hybrid_stack_eager_and_full_recompute_use_the_same_call():

    class TensorLayer(nn.Module):

        def __init__(self, config, layer_number, **kwargs):
            super().__init__()
            self.config, self.layer_number = (config, layer_number)

        def forward(self, hidden_states, inputs):
            self.received_inputs = inputs
            return hidden_states.sin(), None

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
            result = stack(x, attention_mask=None, packed_seq_params=packed)
            result.sum().backward()
        reference = x.detach().clone().requires_grad_(True)
        expected = reference.sin().sin()
        expected.sum().backward()
        assert_bit_exact(result, {}, expected, {})
        assert_bit_exact(x.grad, {}, reference.grad, {})
        for layer in stack.layers:
            assert isinstance(layer.received_inputs, LayerInputs)
            assert layer.received_inputs.packed_sequence_cp_metadata is metadata
            assert layer.received_inputs.packed_seq_params is packed


def test_mtp_depth_invokes_the_shared_call_shape():
    seen = []

    def depth(hidden_states, inputs):
        seen.append(inputs)
        return hidden_states.sin(), None

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
    stack._call_with_bound_args = Mock(return_value=x)
    inputs = LayerInputs(packed_seq_params_by_layout=object(), cp_layout_plan=object())
    assert stack(x, inputs) == (x, None)
    assert stack._call_with_bound_args.call_args.kwargs['cp_layout_plan'] is inputs.cp_layout_plan
    assert (
        stack._call_with_bound_args.call_args.kwargs['packed_seq_params_by_layout']
        is inputs.packed_seq_params_by_layout
    )


def test_moe_forward_backward_combined_output_and_pre_norm():
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
                module.pre_norm = nn.LayerNorm(4)
                x = torch.randn(3, 2, 4)
                expected = module(module.pre_norm(x))[0]
                assert_bit_exact(apply_output_bias(module(x, LayerInputs())), {}, expected, {})


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
def test_input_ids_only_reach_hash_layers(hash_routing, wrapper):
    x = torch.randn(2, 4)
    inner = bare(TransformerLayer)
    inner.config = SimpleNamespace(cuda_graph_impl="none")
    inner.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=hash_routing))
    inner._call_with_bound_args = Mock(return_value=(x, None))
    module = mock_mhc_boundary(inner, x) if wrapper else inner
    if wrapper:
        module._call_inner_transformer_layer_without_local_bda = Mock(return_value=None)
    ids = torch.ones(1, 2, dtype=torch.long)
    assert_bit_exact(apply_output_bias(module(x, LayerInputs(input_ids=ids))), {}, x, {})
    assert inner._call_with_bound_args.call_args.kwargs.get("input_ids") is (
        ids if hash_routing else None
    )


def mock_mhc_boundary(inner, x):
    wrapper = bare(HyperConnectionHybridLayer)
    wrapper.inner_layer = inner
    wrapper.config = SimpleNamespace(fp32_residual_connection=False)
    wrapper.hyper_connection = Mock(return_value=(x, None, None, x))
    wrapper.hyper_connection.fused_h_res_h_post_bda.side_effect = (
        lambda h_res, residual, h_post, output, **kwargs: residual + output[0]
    )
    return wrapper


@pytest.mark.parametrize("shared_call", [False, True])
def test_mhc_fallback_accepts_native_layer_inputs_and_preserves_gradients(shared_call):
    class NativeLayer(nn.Module):
        def forward(self, x, inputs):
            self.inputs = inputs
            return x + x.sin(), None

    inner = NativeLayer()
    x = torch.randn(3, 2, 4, requires_grad=True)
    wrapper = mock_mhc_boundary(inner, x)
    inputs = LayerInputs(
        attention_mask=object(),
        inference_context=object(),
        rotary_pos_emb=object(),
        sequence_len_offset=7,
        packed_seq_params=object(),
        packed_sequence_cp_metadata=object(),
        padding_mask=object(),
        input_ids=object(),
    )
    if shared_call:
        output, bias = wrapper(x, inputs)
        assert bias is None
    else:
        output, context = wrapper(
            x,
            attention_mask=inputs.attention_mask,
            inference_context=inputs.inference_context,
            rotary_pos_emb=inputs.rotary_pos_emb,
            sequence_len_offset=inputs.sequence_len_offset,
            packed_seq_params=inputs.packed_seq_params,
            padding_mask=inputs.padding_mask,
            packed_sequence_cp_metadata=inputs.packed_sequence_cp_metadata,
            input_ids=inputs.input_ids,
        )
        assert context is None
    assert inner.inputs == inputs
    reference = x.detach().clone().requires_grad_(True)
    expected = reference + ((reference + reference.sin()) - reference)
    output.square().sum().backward()
    expected.square().sum().backward()
    assert_bit_exact(output, {"input": x.grad}, expected, {"input": reference.grad})


@pytest.mark.parametrize("hash_routing", [False, True])
def test_mhc_fast_path_filters_ids_when_bypassing_the_inner_adapter(hash_routing):
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
    wrapper(x, LayerInputs(input_ids=ids))
    assert inner._forward_mlp_output_with_bias.call_args.kwargs["input_ids"] is (
        ids if hash_routing else None
    )


def test_cross_attention_rejects_unsupported_shared_inputs():
    from megatron.core.transformer.attention import CrossAttention

    with pytest.raises(NotImplementedError, match="CrossAttention"):
        bare(CrossAttention)(torch.randn(2, 4), LayerInputs(context=object()))


def test_mamba_mixer_rejects_chunkwise_metadata():
    with pytest.raises(NotImplementedError, match="packed-sequence chunkwise CP metadata"):
        bare(MambaMixer)(torch.randn(2, 4), LayerInputs(packed_sequence_cp_metadata=object()))


@pytest.mark.parametrize("cls", [GatedDeltaNet, GatedDeltaNet2])
def test_gdn_cores_preserve_chunkwise_metadata_rejection(cls):
    with pytest.raises(AssertionError, match="packed-sequence chunkwise CP metadata"):
        bare(cls)(torch.randn(2, 4), LayerInputs(packed_sequence_cp_metadata=object()))


class ExplicitSignatureLayer(MegatronModule):
    """Exercise binding without implementing any LayerInputs-specific method."""

    @bind_layer_inputs
    def forward(
        self, hidden_states, attention_mask=None, *, packed_sequence_cp_metadata=None, **kwargs
    ):
        self.seen = (attention_mask, packed_sequence_cp_metadata, kwargs)
        return hidden_states.sin()


@pytest.mark.parametrize("call_style", ["positional", "mixed", "x", "hidden_states", "forward"])
def test_signature_binding_preserves_hooks_gradients_and_legacy_calls(call_style):
    class CustomLayer(ExplicitSignatureLayer):
        def layer_inputs_forward(self, hidden_states, inputs, **kwargs):
            events.append("shared")
            return super().layer_inputs_forward(hidden_states, inputs, **kwargs)

    module = bare(CustomLayer)
    x = torch.randn(2, 4, requires_grad=True)
    mask, metadata, unrelated = object(), object(), object()
    inputs = LayerInputs(
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
            result = module(x, inputs)
        elif call_style == "mixed":
            result = module(x, inputs=inputs)
        elif call_style == "forward":
            result = module.forward(x, inputs)
        else:
            result = module(**{call_style: x, "inputs": inputs})
    assert len(result) == 2 and result[1] is None
    assert module.seen == (mask, metadata, {})
    expected = x.sin() if call_style == "forward" else (x * 2).sin() * 3
    assert_bit_exact(result[0], {}, expected, {})
    assert events == (["shared"] if call_style == "forward" else ["shared", "before", "after"])
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
        ((torch.ones(1), LayerInputs(), None), {}),
        ((torch.ones(1), LayerInputs()), {"attention_mask": None}),
        ((torch.ones(1), LayerInputs()), {"inputs": LayerInputs()}),
        ((torch.ones(1),), {"inputs": object()}),
        ((torch.ones(1),), {"inputs": None}),
        ((), {"inputs": LayerInputs()}),
        ((), {"x": torch.ones(1), "hidden_states": torch.ones(1), "inputs": LayerInputs()}),
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
        bind_layer_inputs(lambda self, *args, **kwargs: None)
    with pytest.raises(ValueError, match="alias"):
        bind_layer_inputs(lambda self, x: x, aliases={"typo": "attention_mask"})
    with pytest.raises(ValueError, match="Required parameter"):
        bind_layer_inputs(lambda self, x, required: x)
    with pytest.raises(ValueError, match="policies"):
        bind_layer_inputs(lambda self, x: x, omit_none=("attention_mask",))
    with pytest.raises(ValueError, match="alias"):
        bind_layer_inputs(lambda self, x, **kwargs: x, aliases={"x": "attention_mask"})


@pytest.mark.parametrize("mode", ["eager", "local", "te_capture", "te_replay"])
@pytest.mark.parametrize("output_kind", ["tensor", "pair", "bias"])
def test_shared_binding_surrounds_graph_dispatch_without_recursion(mode, output_kind):
    bias = torch.randn(4, requires_grad=True) if output_kind == "bias" else None

    class Layer(GraphableMegatronModule):
        def layer_inputs_forward(self, hidden_states, inputs, **kwargs):
            if inputs.context is not None:
                raise ValueError("Unsupported context")
            return super().layer_inputs_forward(hidden_states, inputs, **kwargs)

        @bind_layer_inputs
        def forward(self, hidden_states, attention_mask=None):
            output = hidden_states.sin()
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
        result = module(x, LayerInputs(attention_mask=mask))
        assert (
            len(result) == 2 and result[1] is bias
        )  # Both eager and graph output obey the pair contract.
        assert_bit_exact(result[0], {}, x.sin(), {})
        if output_kind == "tensor":
            assert result[0] is seen[-1]
        else:
            assert result is seen[-1]
    assert len(seen) == 2
    assert len(calls) == (0 if mode == "eager" else 2)
    with pytest.raises(ValueError, match="Unsupported context"):
        module(x, LayerInputs(attention_mask=mask, context=object()))
    assert len(seen) == 2
    assert len(calls) == (0 if mode == "eager" else 2)
    apply_output_bias(result).sum().backward()
    assert_bit_exact(x.grad, {}, x.detach().cos(), {})
    if bias is not None:
        assert_bit_exact(bias.grad, {}, torch.full_like(bias, x.shape[0]), {})
    if mode == "local":
        module._should_call_te_cudagraph.assert_not_called()
    else:
        module.cudagraph_manager.assert_not_called()


@pytest.mark.parametrize("cls", [TransformerLayer, MambaLayer])
def test_graph_argument_keys_match_legacy_skeleton_contract(cls):
    module = bare(cls)
    x = torch.randn(2, 4)
    module.cudagraph_manager = Mock(return_value=(x, None) if cls is TransformerLayer else x)
    module(x, LayerInputs(rotary_pos_emb=object(), rotary_pos_cos_sin=object()))
    _, args, kwargs = module.cudagraph_manager.call_args.args
    assert args == ()
    expected = {"hidden_states", "attention_mask", "inference_context", "packed_seq_params"}
    if cls is TransformerLayer:
        expected |= {"rotary_pos_emb", "sequence_len_offset", "padding_mask"}
    assert set(kwargs) == expected


def test_inherited_adapter_and_override_keep_distinct_signatures():
    class Inherited(ExplicitSignatureLayer):
        pass

    class Override(ExplicitSignatureLayer):
        @bind_layer_inputs
        def forward(self, hidden_states, input_ids=None):
            self.ids = input_ids
            return hidden_states, None

    x, ids = torch.ones(2, 4), object()
    assert len(bare(Inherited)(x, LayerInputs(input_ids=ids))) == 2
    module = bare(Override)
    assert module(x, LayerInputs(input_ids=ids)) == (x, None)
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
    output = compiled(x, LayerInputs())[0]
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
    module(x, LayerInputs())[0].sum().backward()
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
        return x, None

    module.cudagraph_manager = Mock(side_effect=graph)
    assert module(hidden_states=x, mhc_recompute_manager=manager) == (x, None)
    assert module(x, LayerInputs()) == (x, None)
    assert seen == [manager, None]


def test_te_capture_and_replay_use_legacy_tensors_and_manual_hooks():
    class Layer(GraphableMegatronModule):
        @bind_layer_inputs
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
    inputs = LayerInputs(attention_mask=torch.ones(2, 2))
    # Exercise the real TE dispatch and capture methods with an ordinary CPU forward.
    with patch("megatron.core.transformer.cuda_graphs.is_graph_capturing", return_value=True):
        assert_bit_exact(module(x, inputs)[0], {}, x.sin(), {})
    hook = Mock()
    module.cuda_graph_manual_hooks = [(hook, (module,))]
    graph = Mock(side_effect=lambda hidden_states, **kwargs: hidden_states.sin())
    module.cuda_graphs = [graph]
    for microbatch in (0, 1):
        module.current_microbatch = microbatch
        result = module(x, inputs)
        assert len(result) == 2 and result[1] is None
        assert_bit_exact(result[0], {}, x.sin(), {})
        assert graph.call_args.args == (x,)
        assert graph.call_args.kwargs == {
            "attention_mask": inputs.attention_mask,
            "is_first_microbatch": microbatch == 0,
        }
    assert module.forward_count == 1
    assert hook.call_count == 2


@pytest.mark.parametrize(
    "cls", [TransformerLayer, MambaLayer, HybridStack, HyperConnectionHybridLayer]
)
def test_complete_layer_output_never_exposes_context_as_bias(cls):
    module = bare(cls)
    x, context = torch.randn(2, 4), object()
    legacy_output = x if cls in (MambaLayer, HybridStack) else (x, context)
    module._call_with_bound_args = Mock(return_value=legacy_output)
    result = module(x, LayerInputs())
    assert isinstance(result, tuple) and len(result) == 2
    assert result[0] is x and result[1] is None
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
def test_core_output_preserves_pair_and_bias_identity(cls, with_bias):
    module = bare(cls)
    if cls is MoELayer:
        module.pre_norm = IdentityOp()
    x = torch.randn(2, 4)
    bias = torch.randn(4) if with_bias else None
    legacy_output = (x, bias)
    module._call_with_bound_args = Mock(return_value=legacy_output)
    assert module(x, LayerInputs()) is legacy_output


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


def test_stack_mhc_stream_is_not_a_shared_output_bias():
    module = bare(HybridStack)
    x, stream = torch.randn(2, 4), torch.randn(2, 16)
    legacy_output = (x, stream)
    module._call_with_bound_args = Mock(return_value=legacy_output)
    result = module(x, LayerInputs())
    assert result[0] is x and result[1] is None
    assert module(x) is legacy_output


@pytest.mark.parametrize("output_kind", ["tensor", "singleton", "bias", "none"])
def test_local_graph_record_and_replay_preserve_return_structure(output_kind):
    x = torch.randn(2, 4)
    bias = torch.randn(4)
    legacy_output = {"tensor": x, "singleton": (x,), "bias": (x, bias), "none": (x, None)}[
        output_kind
    ]
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
            else:
                assert_bit_exact(actual, {}, expected, {})
