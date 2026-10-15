# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Input adapter and arithmetic equivalence checks, also runnable with unittest on CPU."""

import copy
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.layer_interface import LayerInputs, LayerInterface, as_addends
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


class DeferredBiasLinear(nn.Linear):
    def forward(self, x):
        return nn.functional.linear(x, self.weight), self.bias


def bare(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def add_updates(updates):
    result = updates[0]
    for update in updates[1:]:
        result = result + update
    return result


class TestLayerInputs(unittest.TestCase):
    def test_frozen_inputs_and_bias_identity(self):
        inputs = LayerInputs()
        with self.assertRaises(FrozenInstanceError):
            inputs.input_ids = torch.ones(1)
        x, bias = torch.randn(2, 4), torch.randn(4)
        self.assertEqual(len(as_addends((x, None))), 1)
        self.assertIs(as_addends((x, bias))[0], x)
        self.assertIs(as_addends((x, bias))[1], bias)
        self.assertNotIsInstance(object(), LayerInterface)
        for cls in (
            SelfAttention,
            MLASelfAttention,
            MLP,
            MoELayer,
            MambaMixer,
            GatedDeltaNet,
            GatedDeltaNet2,
        ):
            self.assertIn(LayerInterface, cls.__mro__)

    def assert_equivalent(self, module, legacy, inputs=LayerInputs()):
        other = copy.deepcopy(module)
        x = torch.randn(3, 2, 4, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        old = legacy(module, x)
        new = add_updates(other(y, inputs))
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        old.square().sum().backward()
        new.square().sum().backward()
        torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)
        for (name, p), (other_name, q) in zip(module.named_parameters(), other.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(q.grad, p.grad, rtol=0, atol=0)

    def test_mlp_forward_backward_with_and_without_bias(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
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
                self.assert_equivalent(module, lambda m, x: add_updates(as_addends(m(x))))

    def test_attention_adapter_threads_every_execution_input(self):
        module = bare(SelfAttention)
        output, bias = torch.randn(2, 4), torch.randn(4)
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
        self.assertIs(result[0], output)
        self.assertIs(result[1], bias)
        kwargs = module.forward_pre_attn_and_core_attn.call_args.kwargs
        for key, value in dict(
            attention_mask=mask,
            rotary_pos_emb=rope,
            packed_sequence_cp_metadata=metadata,
            packed_seq_params=packed,
            inference_context=context,
            sequence_len_offset=3,
        ).items():
            self.assertIs(kwargs[key], value)
        module.forward_post_core_attn.assert_called_once_with(output)
        self.assertIs(module(hidden_states=output, inputs=inputs)[0], output)
        self.assertIs(module(x=output, inputs=inputs)[0], output)
        with self.assertRaises(TypeError):
            module(output, inputs, attention_mask=mask)

    def test_core_entry_points_delegate_to_legacy_inference_dispatch(self):
        x, bias = torch.randn(2, 4), torch.randn(4)
        inputs = LayerInputs(
            attention_mask=object(),
            packed_seq_params=object(),
            inference_context=object(),
            sequence_len_offset=7,
        )
        for cls in (MLASelfAttention, MambaMixer, GatedDeltaNet, GatedDeltaNet2):
            with self.subTest(core=cls.__name__):
                module = bare(cls)
                module.forward = Mock(return_value=(x, bias))
                result = module.forward_layer(x, inputs)
                self.assertIs(result[0], x)
                self.assertIs(result[1], bias)
                kwargs = module.forward.call_args.kwargs
                self.assertIs(kwargs['inference_context'], inputs.inference_context)
                self.assertIs(kwargs['packed_seq_params'], inputs.packed_seq_params)

    def test_skeleton_adapters_preserve_recompute_and_mhc_state(self):
        x = torch.randn(2, 4)
        replay, manager, metadata = object(), object(), object()
        inputs = LayerInputs(
            cp_metadata=metadata, input_ids=object(), residual_stream_recompute_context=replay
        )
        for cls in (TransformerLayer, MambaLayer):
            module = bare(cls)
            module.config = SimpleNamespace(cuda_graph_impl='none')
            module.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=True))
            module.forward = Mock(return_value=(x, None) if cls is TransformerLayer else x)
            self.assertIs(module(x, inputs), x)
            kwargs = module.forward.call_args.kwargs
            self.assertIs(kwargs['residual_stream_recompute_context'], replay)
            if cls is MambaLayer:
                self.assertIs(kwargs['packed_sequence_cp_metadata'], metadata)
            else:
                self.assertIs(kwargs['input_ids'], inputs.input_ids)
        wrapper = bare(HyperConnectionHybridLayer)
        wrapper.forward = Mock(return_value=(x, None))
        self.assertIs(wrapper.forward_layer(x, LayerInputs(mhc_recompute_manager=manager)), x)
        self.assertIs(wrapper.forward.call_args.kwargs['mhc_recompute_manager'], manager)
        with self.assertRaisesRegex(TypeError, 'mhc'):
            wrapper.forward_layer(x, inputs)

    def test_graph_adapter_passes_tensors_to_legacy_graph_manager(self):
        module = bare(TransformerLayer)
        module.config = SimpleNamespace(cuda_graph_impl='local')
        x = torch.randn(2, 4)
        module.cudagraph_manager = Mock(return_value=(x, None))
        self.assertIs(module(x, LayerInputs()), x)
        args = module.cudagraph_manager.call_args.args
        self.assertEqual(args[1], ())
        self.assertIs(args[2]['hidden_states'], x)
        self.assertNotIn('inputs', args[2])

    def test_shortcut_atomic_call_accepts_any_layer_with_shared_inputs(self):
        x = torch.randn(2, 4)
        layer = Mock(return_value=x)
        unit = SimpleNamespace(compute_layer=layer)
        self.assertIs(
            ShortcutMoEBlock._forward_compute_atomic(
                unit,
                x,
                attention_mask=None,
                inference_context=None,
                rotary_pos_emb=None,
                sequence_len_offset=None,
                packed_seq_params=None,
                padding_mask=None,
            ),
            x,
        )
        self.assertIsInstance(layer.call_args.args[1], LayerInputs)

    def test_recompute_calls_shared_inputs_and_replays_gradients(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_number = 1
                self.config = SimpleNamespace(linear_cp_mode='chunkwise')
                self.inputs = []

            def forward(self, x, inputs):
                self.inputs.append(inputs)
                return x.sin()

        for method in ('uniform', 'block'):
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
            ids, metadata = object(), object()
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
            torch.testing.assert_close(y, x.sin(), rtol=0, atol=0)
            torch.testing.assert_close(x.grad, x.detach().cos(), rtol=0, atol=0)
            self.assertEqual(len(layer.inputs), 2)
            for inputs in layer.inputs:
                self.assertIs(inputs.input_ids, ids)
                self.assertIs(inputs.cp_metadata, metadata)

    def test_hybrid_stack_eager_and_full_recompute_use_the_same_call(self):
        class TensorLayer(nn.Module):
            def __init__(self, config, layer_number, **kwargs):
                super().__init__()
                self.config, self.layer_number = config, layer_number

            def forward(self, hidden_states, inputs):
                self.assert_inputs = isinstance(inputs, LayerInputs)
                return hidden_states.sin()

        for method in (None, "uniform", "block"):
            with self.subTest(method=method):
                config = TransformerConfig(
                    num_layers=2,
                    hidden_size=4,
                    num_attention_heads=1,
                    use_cpu_initialization=True,
                    recompute_granularity="full" if method else None,
                    recompute_method=method,
                    recompute_num_layers=1 if method else None,
                )
                group = SimpleNamespace(size=lambda: 1)
                with patch(
                    "megatron.core.models.hybrid.hybrid_block.build_module",
                    side_effect=lambda spec, **kwargs: TensorLayer(**kwargs),
                ):
                    stack = HybridStack(
                        config,
                        hybrid_stack_spec.submodules,
                        layer_config_list=validate_segment_layers("*-", config),
                        post_process=False,
                        post_layer_norm=False,
                        pg_collection=SimpleNamespace(tp=group, cp=group, pp=group, tp_cp=group),
                    )
                x = torch.randn(3, 2, 4, requires_grad=True)
                with patch(
                    "megatron.core.recompute.tensor_parallel.checkpoint",
                    side_effect=lambda fn, distribute, *args: checkpoint(
                        fn, *args, use_reentrant=True
                    ),
                ):
                    result = stack(x, attention_mask=None)
                    result.sum().backward()
                reference = x.detach().clone().requires_grad_(True)
                expected = reference.sin().sin()
                expected.sum().backward()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                self.assertTrue(all(layer.assert_inputs for layer in stack.layers))

    def test_mtp_depth_invokes_the_shared_call_shape(self):
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
        x, embedding = torch.randn(3, 2, 4), torch.randn(3, 2, 4)
        layout_plan = object()
        result = MultiTokenPredictionLayer._proj_and_transformer_layer(
            layer, x, embedding, cp_layout_plan=layout_plan
        )
        torch.testing.assert_close(result, (x + embedding).sin(), rtol=0, atol=0)
        self.assertIsInstance(seen[0], LayerInputs)
        self.assertIs(seen[0].cp_layout_plan, layout_plan)

    def test_mtp_stack_adapter_preserves_layout_inputs(self):
        stack = bare(HybridStack)
        x = torch.randn(2, 4)
        stack.forward = Mock(return_value=x)
        inputs = LayerInputs(packed_seq_params_by_layout=object(), cp_layout_plan=object())
        self.assertIs(stack.forward_layer(x, inputs), x)
        self.assertIs(stack.forward.call_args.kwargs['cp_layout_plan'], inputs.cp_layout_plan)
        self.assertIs(
            stack.forward.call_args.kwargs['packed_seq_params_by_layout'],
            inputs.packed_seq_params_by_layout,
        )

    def test_moe_forward_backward_shared_addends_and_pre_norm(self):
        for shared in (False, True):
            for recompute in (False, True):
                for latent in (False, True):
                    with self.subTest(shared=shared, recompute=recompute, latent=latent):
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
                        module.fwd_execution_map = {"route", "expert_compute", "postprocess"}
                        module.moe_layer_recompute = recompute
                        module.select_token_dispatcher = lambda: None
                        module.shared_experts_compute = lambda x: x.sigmoid() if shared else None
                        module.route = lambda x, *args, **kwargs: (None, None)
                        module.preprocess = lambda x, *args: (x, None)
                        module.dispatch = lambda x, probs: (x, probs)
                        module.routed_experts_compute = lambda x, probs: (x.sin(), None)
                        module.combine = lambda x: x
                        module.token_dispatcher = SimpleNamespace(
                            combine_postprocess=(
                                lambda x, return_shared_expert_output=False: (
                                    (x, None) if return_shared_expert_output else x
                                )
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
                            self.assert_equivalent(module, lambda m, x: m(x)[0])
                        module.moe_layer_recompute = False
                        module.pre_norm = nn.LayerNorm(4)
                        x = torch.randn(3, 2, 4)
                        expected = module(module.pre_norm(x))[0]
                        torch.testing.assert_close(
                            add_updates(module(x, LayerInputs())), expected, rtol=0, atol=0
                        )

    def test_flex_dispatcher_defers_overlap_add_without_changing_execution(self):
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
            torch.testing.assert_close(result, (x + shared).view(3, 1, 4), rtol=0, atol=0)
            result.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
            torch.testing.assert_close(shared.grad, torch.ones_like(shared), rtol=0, atol=0)
            expert.linear_fc2_forward.assert_called_once()
            expert.post_forward_comm.assert_called_once()
            expert.get_output.assert_called_once()


if __name__ == '__main__':
    unittest.main()
