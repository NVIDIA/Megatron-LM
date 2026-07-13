# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_block import HybridStack, HyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.pipeline_parallel.schedules import set_current_microbatch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    _TE_CAPTURE_TIME_HOOKS_PROTOCOL,
    CudaGraphManager,
    TECudaGraphHelper,
    VisionTECudaGraphHelper,
    _CudagraphGlobalRecord,
    _get_model_with_decoder,
    _get_te_capture_time_hooks_contract,
    _layer_captures_attention,
    _layer_is_graphable,
    _merge_observed_rotary_kwargs,
    _validate_mrope_capture_inputs,
)
from megatron.core.transformer.cuda_graphs import (
    set_current_microbatch as set_cuda_graph_current_microbatch,
)
from megatron.core.transformer.cuda_graphs import validate_te_cuda_graph_topology
from megatron.core.transformer.enums import CudaGraphModule, CudaGraphScope, InferenceCudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import is_fa_min_version, is_te_min_version
from megatron.training import arguments as training_arguments
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils

fp8_available, _ = check_fp8_support()


class TestCudaGraphModelDiscovery:
    def test_direct_decoder_model(self):
        class Model:
            pass

        model = Model()
        model.decoder = object()

        assert _get_model_with_decoder(model) is model

    def test_nested_multimodal_language_model(self):
        class Model:
            pass

        language_model = Model()
        language_model.decoder = object()
        multimodal_model = Model()
        multimodal_model.language_model = language_model

        assert _get_model_with_decoder(multimodal_model) is language_model

    def test_wrapped_nested_multimodal_language_model(self):
        class Model:
            pass

        language_model = Model()
        language_model.decoder = object()
        multimodal_model = Model()
        multimodal_model.language_model = language_model
        wrapper = Model()
        wrapper.module = multimodal_model

        assert _get_model_with_decoder(wrapper) is language_model

    def test_missing_decoder_raises(self):
        class Model:
            pass

        with pytest.raises(RuntimeError, match="couldn't find attribute decoder"):
            _get_model_with_decoder(Model())

    def test_set_current_microbatch_finds_nested_language_decoder_and_mtp(self):
        class Model:
            pass

        decoder_layer = Model()
        decoder_layer.current_microbatch = None
        mtp_model_layer = Model()
        mtp_model_layer.current_microbatch = None
        language_model = Model()
        language_model.decoder = Model()
        language_model.decoder.layers = [decoder_layer]
        language_model.mtp = Model()
        language_model.mtp.layers = [SimpleNamespace(mtp_model_layer=mtp_model_layer)]
        multimodal_model = Model()
        multimodal_model.language_model = language_model
        wrapper = Model()
        wrapper.module = multimodal_model

        set_cuda_graph_current_microbatch(wrapper, 7)

        assert decoder_layer.current_microbatch == 7
        assert mtp_model_layer.current_microbatch == 7

    def test_set_current_microbatch_updates_vision_only_model_without_decoder(self):
        class Model:
            pass

        vision_layer = Model()
        vision_layer.current_microbatch = None
        vision_encoder = Model()
        vision_encoder.decoder = Model()
        vision_encoder.decoder.layers = [vision_layer]
        vision_only_model = Model()
        vision_only_model.vision_model = vision_encoder

        set_cuda_graph_current_microbatch(vision_only_model, 11)

        assert vision_layer.current_microbatch == 11


def _base_cuda_graph_config(**kwargs) -> TransformerConfig:
    return TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, **kwargs)


class _ObservedKwargLayer(GraphableMegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states * self.weight


class _AttentionProbe(torch.nn.Module):
    def __init__(self, supports_te_cuda_graph: bool):
        super().__init__()
        self.supports_te_cuda_graph = supports_te_cuda_graph


def _bare_transformer_layer(
    config: TransformerConfig, *, attention_supports_graph: bool = True
) -> TransformerLayer:
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.self_attention = _AttentionProbe(attention_supports_graph)
    layer.cross_attention = IdentityOp()
    layer.mlp = IdentityOp()
    return layer


def _bare_mhc_wrapper(
    config: TransformerConfig, inner_layer: TransformerLayer
) -> HyperConnectionHybridLayer:
    wrapper = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = config
    wrapper.inner_layer = inner_layer
    return wrapper


def _use_cpu_static_inputs(layer):
    """Override static-input creation so focused helper tests do not require CUDA."""

    def get_layer_static_inputs(seq_length, micro_batch_size):
        return {
            "hidden_states": torch.ones(
                seq_length,
                micro_batch_size,
                layer.config.hidden_size,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
        }

    layer.get_layer_static_inputs = get_layer_static_inputs


def _bare_vision_helper(config, layer, num_microbatches=1):
    helper = VisionTECudaGraphHelper.__new__(VisionTECudaGraphHelper)
    helper.config = config
    helper.seq_length = 2
    helper.num_microbatches = num_microbatches
    helper.flattened_callables = [layer]
    helper.vision_model = SimpleNamespace(_cuda_graph_requires_observed_rotary_inputs=True)
    return helper


def _force_cpu_zeros(monkeypatch):
    """Keep Vision helper unit tests independent of a local CUDA device."""
    original_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)


class TestGdnCudaGraphOptIn:
    def test_exact_env_opt_in_controls_direct_attention_scope(self):
        """GDN capture is enabled only by the exact environment value 1."""
        import importlib

        from megatron.core.ssm import gated_delta_net
        from megatron.core.transformer.moe.moe_layer import MoELayer

        env_name = "MEGATRON_GDN_TE_CUDA_GRAPH"
        missing = object()
        original_value = os.environ.get(env_name, missing)
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=[CudaGraphModule.attn],
            use_cpu_initialization=True,
        )

        try:
            for env_value, expected in (
                (None, False),
                ("0", False),
                ("false", False),
                ("1", True),
            ):
                if env_value is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = env_value
                gdn_module = importlib.reload(gated_delta_net)

                attention = gdn_module.GatedDeltaNet.__new__(gdn_module.GatedDeltaNet)
                torch.nn.Module.__init__(attention)
                layer = _bare_transformer_layer(config)
                layer.self_attention = attention

                assert gdn_module.GatedDeltaNet.supports_te_cuda_graph is expected
                assert layer._cuda_graph_captures_attention() is expected
                assert _layer_captures_attention(layer) is expected
                assert _layer_is_graphable(layer, config) is expected

                if env_value == "0":
                    # An unsupported attention stays eager without hiding an
                    # independently requested, graphable MoE-router region.
                    moe = MoELayer.__new__(MoELayer)
                    torch.nn.Module.__init__(moe)
                    layer.mlp = moe
                    config.cuda_graph_modules = [
                        CudaGraphModule.attn,
                        CudaGraphModule.moe_router,
                    ]
                    assert not layer._cuda_graph_captures_attention()
                    assert _layer_is_graphable(layer, config)
                    config.cuda_graph_modules = [CudaGraphModule.attn]
        finally:
            if original_value is missing:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = original_value
            importlib.reload(gated_delta_net)


class TestRotaryCudaGraphInputs:
    def test_eager_observation_keeps_only_detached_rotary_allowlist(self):
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine', use_cpu_initialization=True
        )
        layer = _ObservedKwargLayer(config)
        hidden_states = torch.randn(2, 1, 4, requires_grad=True)
        rotary_pos_emb = torch.randn(2, 1, 1, 4, requires_grad=True)

        layer(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            padding_mask=torch.ones(1, 2, dtype=torch.bool),
        )

        assert set(layer._cg_observed_tensor_kwargs) == {'rotary_pos_emb'}
        observed = layer._cg_observed_tensor_kwargs['rotary_pos_emb']
        assert observed.data_ptr() == rotary_pos_emb.data_ptr()
        assert not observed.requires_grad
        assert observed.grad_fn is None

    def test_direct_layer_merges_rotary_and_cleans_observation(self):
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=[CudaGraphModule.attn],
            use_cpu_initialization=True,
        )
        layer = _bare_transformer_layer(config)
        rotary_pos_emb = torch.randn(2, 1, 1, 4, requires_grad=True)
        layer._cg_observed_tensor_kwargs = {
            'rotary_pos_emb': rotary_pos_emb,
            'hidden_states': torch.randn(2, 1, 4, requires_grad=True),
        }
        static_inputs = {}

        _merge_observed_rotary_kwargs(layer, static_inputs, 'mrope', True, {})

        assert set(static_inputs) == {'rotary_pos_emb'}
        assert static_inputs['rotary_pos_emb'].data_ptr() != rotary_pos_emb.data_ptr()
        assert not static_inputs['rotary_pos_emb'].requires_grad
        assert not hasattr(layer, '_cg_observed_tensor_kwargs')
        assert _layer_captures_attention(layer)

    def test_mhc_uses_outer_observation_and_cleans_inner(self):
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=[CudaGraphModule.attn],
            use_cpu_initialization=True,
        )
        inner_layer = _bare_transformer_layer(config)
        wrapper = _bare_mhc_wrapper(config, inner_layer)
        outer_rotary = torch.randn(2, 1, 1, 4)
        wrapper._cg_observed_tensor_kwargs = {'rotary_pos_cos_sin': outer_rotary}
        inner_layer._cg_observed_tensor_kwargs = {'rotary_pos_emb': torch.randn(2, 1, 1, 4)}
        static_inputs = {}

        _merge_observed_rotary_kwargs(wrapper, static_inputs, 'mrope', True, {})

        assert set(static_inputs) == {'rotary_pos_cos_sin'}
        assert torch.equal(static_inputs['rotary_pos_cos_sin'], outer_rotary)
        assert not hasattr(wrapper, '_cg_observed_tensor_kwargs')
        assert not hasattr(inner_layer, '_cg_observed_tensor_kwargs')
        assert _layer_captures_attention(wrapper)

    def test_mrope_requires_observation_and_tensor_kwarg_support(self):
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=[CudaGraphModule.attn],
            use_cpu_initialization=True,
        )
        layer = _bare_transformer_layer(config)

        with pytest.raises(RuntimeError, match='no rotary tensor kwargs were observed'):
            _merge_observed_rotary_kwargs(layer, {}, 'mrope', True, {})

        observed = {'rotary_pos_emb': torch.randn(2, 1, 1, 4)}
        with pytest.raises(RuntimeError, match='TransformerEngine >= 1.10'):
            _validate_mrope_capture_inputs(layer, 'mrope', True, observed, False)

    def test_vision_helper_declares_rotary_for_each_microbatch(self, monkeypatch):
        """Vision capture clones an independent static rotary buffer per microbatch."""
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=[CudaGraphModule.attn],
            use_cpu_initialization=True,
        )
        layer = _bare_transformer_layer(config)
        _use_cpu_static_inputs(layer)
        rotary_pos_emb = torch.randn(2, 1, 1, 4)
        layer._cg_observed_tensor_kwargs = {'rotary_pos_emb': rotary_pos_emb}
        helper = _bare_vision_helper(config, layer, num_microbatches=2)
        _force_cpu_zeros(monkeypatch)
        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs.is_te_min_version", lambda version: True
        )

        sample_args, sample_kwargs = helper._get_sample_arguments(order=[1, -1])

        assert len(sample_args) == 2
        assert len(sample_kwargs) == 2
        assert all(set(kwargs) == {'rotary_pos_emb'} for kwargs in sample_kwargs)
        assert torch.equal(sample_kwargs[0]['rotary_pos_emb'], rotary_pos_emb)
        assert torch.equal(sample_kwargs[1]['rotary_pos_emb'], rotary_pos_emb)
        assert (
            sample_kwargs[0]['rotary_pos_emb'].data_ptr()
            != sample_kwargs[1]['rotary_pos_emb'].data_ptr()
        )
        assert not hasattr(layer, '_cg_observed_tensor_kwargs')

    def test_vision_helper_requires_eager_rotary_observation(self, monkeypatch):
        """Qwen-style vision attention fails loudly when capture has no warmup."""
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=[CudaGraphModule.attn],
            use_cpu_initialization=True,
        )
        layer = _bare_transformer_layer(config)
        _use_cpu_static_inputs(layer)
        helper = _bare_vision_helper(config, layer)
        _force_cpu_zeros(monkeypatch)
        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs.is_te_min_version", lambda version: True
        )

        with pytest.raises(RuntimeError, match='no rotary tensor kwargs were observed'):
            helper._get_sample_arguments(order=[1, -1])

    def test_vision_helper_requires_te_tensor_kwargs(self, monkeypatch):
        """Observed vision rotary inputs require TE's tensor-kwarg graph API."""
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=[CudaGraphModule.attn],
            use_cpu_initialization=True,
        )
        layer = _bare_transformer_layer(config)
        _use_cpu_static_inputs(layer)
        layer._cg_observed_tensor_kwargs = {'rotary_pos_emb': torch.randn(2, 1, 1, 4)}
        helper = _bare_vision_helper(config, layer)
        _force_cpu_zeros(monkeypatch)
        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs.is_te_min_version", lambda version: False
        )

        with pytest.raises(RuntimeError, match='TransformerEngine >= 1.10'):
            helper._get_sample_arguments(order=[1, -1])


class TestMhcCudaGraphSafety:
    def test_mhc_graphability_respects_attention_safety_and_mfsdp_gate(self):
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine', use_cpu_initialization=True
        )
        inner_layer = _bare_transformer_layer(config, attention_supports_graph=False)
        wrapper = _bare_mhc_wrapper(config, inner_layer)

        assert not wrapper._cuda_graph_captures_attention()
        assert not _layer_captures_attention(wrapper)
        assert not _layer_is_graphable(wrapper, config)

        inner_layer.self_attention.supports_te_cuda_graph = True
        assert wrapper._cuda_graph_captures_attention()
        assert _layer_captures_attention(wrapper)
        assert _layer_is_graphable(wrapper, config)
        assert not _layer_is_graphable(wrapper, config, use_megatron_fsdp=True)
        assert _layer_is_graphable(inner_layer, config, use_megatron_fsdp=True)

        config.cuda_graph_modules = [CudaGraphModule.attn]
        assert _layer_is_graphable(wrapper, config)
        assert not _layer_is_graphable(wrapper, config, use_megatron_fsdp=True)


    def test_mhc_discovery_stays_eager_with_megatron_fsdp(self, monkeypatch):
        """The helper must pass its Megatron-FSDP state into layer discovery."""
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine', use_cpu_initialization=True
        )
        wrapper = _bare_mhc_wrapper(config, _bare_transformer_layer(config))
        model = torch.nn.Module()
        model.decoder = torch.nn.Module()
        model.decoder.layers = torch.nn.ModuleList([wrapper])

        messages = []
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs.log_on_each_pipeline_stage",
            lambda **kwargs: messages.append(kwargs["msg"]),
        )

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.model = [model]
        helper.config = config
        helper.tp_group = None
        helper.dp_cp_group = None
        helper._uses_megatron_fsdp = True
        helper._rotary_observation_roots = []
        helper._discover_layers()

        assert helper.flattened_callables == []
        assert helper.callables_per_chunk == [[]]
        assert any("HyperConnectionHybridLayer" in message for message in messages)


class TestRotaryObservationLifecycle:
    def test_observation_is_disabled_for_non_graphable_layers(self):
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine', use_cpu_initialization=True
        )
        layer = _ObservedKwargLayer(config)
        layer._cg_observed_tensor_kwargs = {'rotary_pos_emb': torch.randn(2, 1, 1, 4)}
        sibling_layer = _ObservedKwargLayer(config)
        sibling_layer._cg_observed_tensor_kwargs = {'rotary_pos_emb': torch.randn(2, 1, 1, 4)}
        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper._rotary_observation_roots = [layer]

        helper._disable_rotary_kwarg_observation()

        assert not layer._cg_rotary_observation_enabled
        assert not hasattr(layer, '_cg_observed_tensor_kwargs')
        assert sibling_layer._cg_rotary_observation_enabled
        assert hasattr(sibling_layer, '_cg_observed_tensor_kwargs')
        layer(hidden_states=torch.ones(2, 1, 4), rotary_pos_emb=torch.randn(2, 1, 1, 4))
        assert not hasattr(layer, '_cg_observed_tensor_kwargs')


class TestCudaGraphAddressChecks:
    @pytest.mark.parametrize('kill_switch', ['0', 'false'])
    def test_address_check_kill_switch_only_accepts_one(self, monkeypatch, kill_switch):
        config = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine', use_cpu_initialization=True
        )
        layer = _ObservedKwargLayer(config)
        parameter = layer.weight
        layer._cg_param_ptr_snapshot = {
            'weight': (
                parameter.data_ptr() + parameter.element_size(),
                parameter.numel(),
                parameter.dtype,
            )
        }
        layer.cuda_graphs = [lambda *args, **kwargs: args[0]]
        monkeypatch.setenv('MEGATRON_CG_SKIP_BUFFER_ADDRESS_CHECK', kill_switch)

        with pytest.raises(RuntimeError, match='buffer address changed'):
            layer._te_cuda_graph_replay(torch.ones(1))

        monkeypatch.setenv('MEGATRON_CG_SKIP_BUFFER_ADDRESS_CHECK', '1')
        assert torch.equal(layer._te_cuda_graph_replay(torch.ones(1)), torch.ones(1))


def _bare_capture_lifecycle_helper(layer, helper_cls=TECudaGraphHelper):
    """Build a CPU-only helper for focused capture-state tests."""
    helper = helper_cls.__new__(helper_cls)
    helper.config = SimpleNamespace(
        sequence_parallel=False,
        overlap_moe_expert_parallel_comm=False,
        fine_grained_activation_offloading=False,
    )
    helper.model = []
    helper.optimizers = []
    helper.flattened_callables = [layer]
    helper.callables_per_chunk = [[layer]]
    helper._capture_finished = False
    helper._capture_failed = False
    helper._graphs_created = False
    helper._capture_flag_owned = False
    helper._capture_gc_frozen = False
    helper._capture_restore_hooks = []
    helper._fsdp_capture_param_states = []
    return helper


class TestTECaptureTimeHooksContract:
    def test_explicit_protocol_marker_is_authoritative(self):
        def marked_callable():
            return None

        marked_callable.__mcore_cuda_graph_protocols__ = {
            _TE_CAPTURE_TIME_HOOKS_PROTOCOL
        }
        assert (
            _get_te_capture_time_hooks_contract(marked_callable)
            == "explicit-protocol-marker"
        )

        marked_callable.__mcore_cuda_graph_protocols__ = {"different_protocol"}
        assert _get_te_capture_time_hooks_contract(marked_callable) is None

    def test_module_protocol_marker_is_supported(self, monkeypatch):
        def unmarked_callable():
            return None

        graph_module = SimpleNamespace(
            __mcore_cuda_graph_protocols__=(_TE_CAPTURE_TIME_HOOKS_PROTOCOL,)
        )
        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs.inspect.getmodule",
            lambda unused: graph_module,
        )

        assert (
            _get_te_capture_time_hooks_contract(unmarked_callable)
            == "explicit-protocol-marker"
        )

    def test_argument_name_alone_does_not_claim_protocol(self, monkeypatch):
        def public_callable(capture_time_hooks=None):
            return capture_time_hooks

        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs.inspect.getmodule",
            lambda unused: SimpleNamespace(),
        )

        assert _get_te_capture_time_hooks_contract(public_callable) is None


class TestTECudaGraphCaptureFailClean:
    @staticmethod
    def _planned_fsdp_module():
        return SimpleNamespace(
            param_and_grad_buffer=SimpleNamespace(_uses_planned_allocator=True)
        )

    def test_feature_contract_fails_before_capture_mutation(self, monkeypatch):
        helper = _bare_capture_lifecycle_helper(torch.nn.Linear(2, 2))
        helper._get_megatron_fsdp_instances = lambda: [self._planned_fsdp_module()]
        events = []
        helper._start_capturing = lambda: events.append("start-capturing")

        def unsupported_callable():
            return None

        unsupported_callable.__mcore_cuda_graph_protocols__ = set()
        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs.make_graphed_callables",
            unsupported_callable,
        )

        with pytest.raises(RuntimeError, match="capture-time-hooks protocol"):
            helper.create_cudagraphs()

        assert events == []
        assert not helper._capture_failed

    def test_no_graphable_callable_does_not_require_overlay(self, monkeypatch):
        helper = _bare_capture_lifecycle_helper(torch.nn.Linear(2, 2))
        helper.flattened_callables = []
        helper._get_megatron_fsdp_instances = lambda: [self._planned_fsdp_module()]

        def unsupported_callable():
            return None

        unsupported_callable.__mcore_cuda_graph_protocols__ = set()
        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs.make_graphed_callables",
            unsupported_callable,
        )

        helper.validate_capture_feature_contract()

    def test_abort_restores_owned_state_callable_hooks_and_fsdp_exposure(self, monkeypatch):
        layer = torch.nn.Linear(2, 2)

        def forward_hook(*unused):
            return None

        layer._forward_hooks.clear()
        helper = _bare_capture_lifecycle_helper(layer)
        helper._capture_flag_owned = True
        helper._capture_gc_frozen = True
        helper._capture_restore_hooks = [
            (layer, {'forward_hooks_restore': {7: forward_hook}})
        ]
        events = []
        fsdp_module = SimpleNamespace(
            _replace_param_with_distributed_if_needed=lambda: events.append('fsdp-distributed'),
            _replace_param_with_raw_if_needed=lambda: events.append('fsdp-raw'),
        )
        helper._fsdp_capture_param_states = [(fsdp_module, True)]
        helper._disable_rotary_kwarg_observation = lambda: None
        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs._set_capture_end",
            lambda: events.append('capture-end'),
        )
        monkeypatch.setattr(gc, "unfreeze", lambda: events.append('gc-unfreeze'))

        helper._abort_capturing()

        assert helper._capture_failed
        assert not helper._capture_flag_owned
        assert not helper._capture_gc_frozen
        assert helper._capture_restore_hooks == []
        assert helper._fsdp_capture_param_states == []
        assert layer._forward_hooks == {7: forward_hook}
        assert events == ['capture-end', 'fsdp-distributed', 'gc-unfreeze']

    def test_start_failure_preserves_caller_owned_gc_freeze(self, monkeypatch):
        layer = torch.nn.Linear(2, 2)
        layer._cg_rotary_observation_enabled = True
        layer._cg_observed_tensor_kwargs = {'rotary_pos_emb': torch.ones(1)}
        helper = _bare_capture_lifecycle_helper(layer)
        helper._rotary_observation_roots = [layer]
        helper._prepare_fsdp_params_for_capture = lambda: (_ for _ in ()).throw(
            RuntimeError("prepare failed")
        )
        helper._reset_after_capture = lambda: None

        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(gc, "collect", lambda: None)
        monkeypatch.setattr(gc, "get_freeze_count", lambda: 1)
        monkeypatch.setattr(
            gc,
            "freeze",
            lambda: (_ for _ in ()).throw(AssertionError("must not freeze caller state")),
        )
        monkeypatch.setattr(
            gc,
            "unfreeze",
            lambda: (_ for _ in ()).throw(AssertionError("must not unfreeze caller state")),
        )

        with pytest.raises(RuntimeError, match="prepare failed"):
            helper.create_cudagraphs()

        assert helper._capture_failed
        assert not helper._capture_gc_frozen
        assert not layer._cg_rotary_observation_enabled
        assert not hasattr(layer, '_cg_observed_tensor_kwargs')
        with pytest.raises(RuntimeError, match="cannot be retried"):
            helper.create_cudagraphs()

    def test_capture_state_guards_are_explicit(self, monkeypatch):
        finished = _bare_capture_lifecycle_helper(torch.nn.Linear(2, 2))
        finished._capture_finished = True
        with pytest.raises(RuntimeError, match="already been finished"):
            finished.create_cudagraphs()

        active = _bare_capture_lifecycle_helper(torch.nn.Linear(2, 2))
        monkeypatch.setattr(
            "megatron.core.transformer.cuda_graphs.is_graph_capturing", lambda: True
        )
        with pytest.raises(RuntimeError, match="Another CUDA Graph capture is already active"):
            active.create_cudagraphs()
        assert not active._capture_failed

    @pytest.mark.parametrize(
        "fsdp_unit_id",
        [pytest.param(-1, id="non-unit"), pytest.param(0, id="wrapper-around-inner-unit")],
    )
    def test_planned_allocator_requires_hook_at_callable_boundary(self, fsdp_unit_id):
        layer = torch.nn.Linear(2, 2)
        parameter_group = SimpleNamespace(
            fsdp_unit_id=fsdp_unit_id,
            model_weight_buffer=SimpleNamespace(is_data_distributed=True),
            transpose_weight_buffer=None,
            main_grad_buffer=None,
            hfsdp_helper_wbuf=None,
            hfsdp_helper_gbuf=None,
        )
        param_and_grad_buffer = SimpleNamespace(
            param_to_param_group={parameter: 0 for parameter in layer.parameters()},
            parameter_groups=[parameter_group],
        )
        fsdp_module = SimpleNamespace(
            all_gather_pipeline=None,
            param_and_grad_buffer=param_and_grad_buffer,
            _replace_param_with_raw_if_needed=lambda: None,
        )
        helper = _bare_capture_lifecycle_helper(layer)
        helper._get_megatron_fsdp_instances = lambda: [fsdp_module]

        with pytest.raises(RuntimeError, match="must own.*pre-backward hook"):
            helper._setup_fsdp_planned_allocators()

    def test_planned_allocator_accepts_exact_fsdp_unit_hook_boundary(self, monkeypatch):
        layer = torch.nn.Linear(2, 2)

        def tagged_pre_backward_hook(*unused):
            return None

        tagged_pre_backward_hook._cuda_graph_backward_pre_handler = lambda *unused: None
        layer.register_forward_hook(tagged_pre_backward_hook)

        parameter_group = SimpleNamespace(
            fsdp_unit_id=0,
            model_weight_buffer=SimpleNamespace(is_data_distributed=True),
            transpose_weight_buffer=None,
            main_grad_buffer=None,
            hfsdp_helper_wbuf=None,
            hfsdp_helper_gbuf=None,
        )
        freeze_calls = []

        def recording_allocator(name):
            return SimpleNamespace(
                freeze_plan=lambda bucket_ids: freeze_calls.append((name, set(bucket_ids)))
            )

        param_and_grad_buffer = SimpleNamespace(
            param_to_param_group={parameter: 0 for parameter in layer.parameters()},
            parameter_groups=[parameter_group],
            weight_alloc=recording_allocator("weight"),
            transpose_weight_alloc=recording_allocator("transpose"),
            main_grad_alloc=recording_allocator("main-grad"),
        )
        fsdp_module = SimpleNamespace(
            all_gather_pipeline=None,
            param_and_grad_buffer=param_and_grad_buffer,
            _replace_param_with_raw_if_needed=lambda: None,
        )
        helper = _bare_capture_lifecycle_helper(layer)
        helper._get_megatron_fsdp_instances = lambda: [fsdp_module]
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

        helper._setup_fsdp_planned_allocators()

        assert freeze_calls == [
            ("weight", {0}),
            ("transpose", {0}),
            ("main-grad", {0}),
        ]


class TestTECudaGraphTopologySignature:
    @staticmethod
    def _helper():
        helper = _bare_capture_lifecycle_helper(torch.nn.Linear(2, 2))
        helper.config = SimpleNamespace(
            microbatch_group_size_per_vp_stage=2,
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
            overlap_moe_expert_parallel_comm=False,
            delay_wgrad_compute=False,
            cuda_graph_dynamic_microbatches=False,
            variable_seq_lengths=False,
        )
        helper.model = [object(), object()]
        helper.callables_per_chunk = [[object()], [object(), object()]]
        helper.pp_group = SimpleNamespace(size=lambda: 4, rank=lambda: 1)
        helper.p2p_communicator = SimpleNamespace(
            virtual_pipeline_model_parallel_size=2
        )
        helper.seq_length = 4096
        helper.thd_sequence_length_upper_bound = None
        helper._graphs_created = True
        helper._fsdp_planned_topology_signature = (
            helper._current_fsdp_planned_topology_signature(
                num_microbatches=8,
                micro_batch_size=1,
            )
        )
        return helper

    def test_matching_signature_passes_and_predictable_drift_fails(self):
        helper = self._helper()

        helper.validate_runtime_topology(
            num_microbatches=8,
            micro_batch_size=1,
            phase="training",
        )
        with pytest.raises(RuntimeError, match="num_microbatches.*captured=8.*runtime=9"):
            helper.validate_runtime_topology(
                num_microbatches=9,
                micro_batch_size=1,
                phase="training",
            )

        helper.config.microbatch_group_size_per_vp_stage = 4
        with pytest.raises(RuntimeError, match="microbatch_group_size_per_vp_stage"):
            helper.validate_runtime_topology(
                num_microbatches=8,
                micro_batch_size=1,
                phase="evaluation",
            )

        helper.config.microbatch_group_size_per_vp_stage = 2
        helper.config.variable_seq_lengths = True
        with pytest.raises(RuntimeError, match="variable_seq_lengths"):
            helper.validate_runtime_topology(
                num_microbatches=8,
                micro_batch_size=1,
                phase="evaluation",
            )

    def test_validator_is_installed_and_removed_with_graph_ownership(self):
        helper = self._helper()
        helper._install_fsdp_planned_topology_validator()

        validate_te_cuda_graph_topology(
            helper.config,
            num_microbatches=8,
            micro_batch_size=1,
            phase="training",
        )
        helper._remove_fsdp_planned_topology_validator()
        validate_te_cuda_graph_topology(
            helper.config,
            num_microbatches=99,
            micro_batch_size=99,
            phase="training",
        )

    def test_stock_te_config_without_validator_is_unchanged(self):
        validate_te_cuda_graph_topology(
            SimpleNamespace(),
            num_microbatches=3,
            micro_batch_size=2,
            phase="evaluation",
        )


def _validated_cuda_graph_cli_args(monkeypatch, cli_args=None, **overrides):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    warning_messages = []
    print_messages = []

    monkeypatch.setattr(
        training_arguments, "warn_rank_0", lambda msg, *args, **kwargs: warning_messages.append(msg)
    )
    monkeypatch.setattr(
        training_arguments, "print_rank_0", lambda msg, *args, **kwargs: print_messages.append(msg)
    )
    monkeypatch.setattr(sys, "argv", ["test_cuda_graphs.py", *(cli_args or [])])

    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1

    for key, value in overrides.items():
        setattr(args, key, value)

    args = validate_args(args)
    return args, warning_messages, print_messages


class TestCudaGraphConfigAndArguments:
    def test_local_impl_defaults_to_layer_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_impl='local')
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.layer

    def test_full_iteration_impl_requires_empty_scope(self):
        with pytest.raises(
            AssertionError,
            match='cuda_graph_modules must be empty when cuda_graph_impl="full_iteration"',
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='full_iteration', cuda_graph_modules=[CudaGraphModule.attn]
            )

    def test_full_iteration_scope_string_in_config_migrated(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules='full_iteration'
            )
        assert cfg.cuda_graph_impl == 'full_iteration'
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_full_iteration_inference_scope_string_in_config_migrated(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules='full_iteration_inference'
            )
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_full_iteration_inference_scope_string_noops_without_local_impl(self):
        with pytest.warns(DeprecationWarning, match="has no effect"):
            cfg = _base_cuda_graph_config(cuda_graph_modules='full_iteration_inference')
        assert cfg.cuda_graph_impl == 'none'
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_full_iteration_scope_rejects_conflicting_new_scope(self):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration' cannot be combined with "
            "inference_cuda_graph_scope='block'",
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules='full_iteration',
                inference_cuda_graph_scope='block',
            )

    def test_deprecated_full_iteration_inference_scope_rejects_conflicting_new_scope(self):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration_inference' cannot be combined with "
            "inference_cuda_graph_scope='layer'",
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules='full_iteration_inference',
                inference_cuda_graph_scope='layer',
            )

    def test_enable_cuda_graph_flag_migrates_to_local_impl(self, monkeypatch):
        args, _, print_messages = _validated_cuda_graph_cli_args(
            monkeypatch, ['--enable-cuda-graph']
        )
        assert args.cuda_graph_impl == 'local'
        assert any("--enable-cuda-graph is deprecated" in msg for msg in print_messages)

    def test_full_iteration_inference_scope_cli_migrates_to_block_scope(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch,
            ['--cuda-graph-impl', 'local', '--cuda-graph-modules', 'full_iteration_inference'],
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert args.cuda_graph_modules == []
        assert any(
            "--cuda-graph-modules 'full_iteration_inference' is deprecated" in msg
            for msg in warning_messages
        )

    def test_full_iteration_inference_scope_cli_noops_without_local_impl(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-scope', 'full_iteration_inference']
        )
        assert args.cuda_graph_impl == 'none'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert args.cuda_graph_modules == []
        assert any("has no effect when --cuda-graph-impl=none" in msg for msg in warning_messages)

    def test_full_iteration_inference_scope_cli_rejects_conflicting_new_scope(self, monkeypatch):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration_inference' cannot be combined with "
            "inference_cuda_graph_scope='layer'",
        ):
            _validated_cuda_graph_cli_args(
                monkeypatch,
                [
                    '--cuda-graph-impl',
                    'local',
                    '--cuda-graph-modules',
                    'full_iteration_inference',
                    '--inference-cuda-graph-scope',
                    'layer',
                ],
            )

    def test_new_scope_cli_accepts_block(self, monkeypatch):
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--inference-cuda-graph-scope', 'block']
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.block

    def test_new_scope_cli_accepts_layer(self, monkeypatch):
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--inference-cuda-graph-scope', 'layer']
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.layer

    def test_removed_module_scoped_scope_name_is_not_accepted(self, monkeypatch):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                'test_cuda_graphs.py',
                '--cuda-graph-impl',
                'local',
                '--inference-cuda-graph-scope',
                'module_scoped',
            ],
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_removed_old_inference_bool_flag_is_not_accepted(self, monkeypatch):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        monkeypatch.setattr(
            sys, "argv", ['test_cuda_graphs.py', '--inference-use-full-iteration-cuda-graph']
        )
        with pytest.raises(SystemExit):
            parse_args()

    # --- Backward compat: cuda_graph_scope → cuda_graph_modules rename ---

    def test_deprecated_cuda_graph_scope_kwarg_migrates_to_modules(self):
        with pytest.warns(DeprecationWarning, match="cuda_graph_scope is deprecated"):
            cfg = _base_cuda_graph_config(cuda_graph_scope=['attn'])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn]
        assert cfg.cuda_graph_scope is None

    def test_new_cuda_graph_modules_does_not_populate_deprecated_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_modules=['attn', 'mlp'])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn, CudaGraphModule.mlp]
        assert cfg.cuda_graph_scope is None

    def test_new_full_iteration_impl_does_not_populate_deprecated_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_impl='full_iteration', cuda_graph_modules=[])
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_cli_migrates_to_modules(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--cuda-graph-scope', 'attn']
        )
        assert args.cuda_graph_modules == [CudaGraphModule.attn]
        assert any('--cuda-graph-scope is deprecated' in msg for msg in warning_messages)

    def test_cuda_graph_scope_is_standalone_class_for_pickle_compat(self):
        from megatron.core.transformer.enums import CudaGraphScope

        # CudaGraphScope is preserved as a standalone class (not an alias) so that
        # pre-refactor checkpoints can be deserialized without value-collision errors.
        assert CudaGraphScope is not CudaGraphModule
        assert CudaGraphScope.attn.value == 2  # original ordinals preserved
        assert CudaGraphScope.mamba.value == 7

    def test_cuda_graph_scope_and_inference_scope_in_safe_globals(self):
        from megatron.core.safe_globals import SAFE_GLOBALS
        from megatron.core.transformer.enums import CudaGraphScope

        assert CudaGraphScope in SAFE_GLOBALS
        assert InferenceCudaGraphScope in SAFE_GLOBALS

    def test_deprecated_cuda_graph_scope_enum_instance_migrates_to_modules(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning, match="cuda_graph_scope is deprecated"):
            cfg = _base_cuda_graph_config(cuda_graph_scope=[CudaGraphScope.attn])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn]
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_enum_migrates_to_impl(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning):
            cfg = _base_cuda_graph_config(cuda_graph_scope=[CudaGraphScope.full_iteration])
        assert cfg.cuda_graph_impl == "full_iteration"
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_inference_enum_migrates_to_scope(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl="local", cuda_graph_scope=[CudaGraphScope.full_iteration_inference]
            )
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_inference_noops_without_local_impl(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning, match="has no effect"):
            cfg = _base_cuda_graph_config(
                cuda_graph_scope=[CudaGraphScope.full_iteration_inference]
            )
        assert cfg.cuda_graph_impl == "none"
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None


class TestParallelTransformerBlockCudagraphs:
    def setup_method(self, method):
        # initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)

        # initialize transformer model
        num_layers = 8
        hidden_size = 64
        self.transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_layer_with_transformer_engine_spec()
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

    @pytest.mark.flaky_in_dev  # Issue #5474
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_gpu_cudagraph(self):
        parallel_transformer_block = self.parallel_transformer_block
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        sequence_length = 32
        micro_batch_size = 2
        transformer_config: TransformerConfig = parallel_transformer_block.config
        num_layers = transformer_config.num_layers
        hidden_size = transformer_config.hidden_size
        hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        for _ in range(num_layers):
            assert hasattr(parallel_transformer_block.layers[0], "cudagraph_manager")
            assert (
                len(parallel_transformer_block.layers[0].cudagraph_manager.cudagraph_runners) == 1
            )
            del (
                parallel_transformer_block.layers[_]
                .cudagraph_manager.cudagraph_runners[0]
                .fwd_graph
            )


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
@pytest.mark.parametrize(
    "total_num_layers, pp, vpp, account_for_embedding_in_pipeline_split, account_for_loss_in_pipeline_split, num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage, pp_layout, first_layer_numbers_golden, last_layer_numbers_golden",
    [
        (4, 1, None, False, False, None, None, None, [1], [4]),
        (8, 2, None, False, False, None, None, None, [1, 5], [4, 8]),
        (8, 2, 2, False, False, None, None, None, [1, 3, 5, 7], [2, 4, 6, 8]),
        (14, 4, None, True, True, None, None, None, [1, 4, 8, 12], [3, 7, 11, 14]),
        (
            14,
            4,
            2,
            True,
            True,
            None,
            None,
            None,
            [1, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 14],
        ),
        (12, 4, None, False, False, 2, 2, None, [1, 3, 7, 11], [2, 6, 10, 12]),
        (
            12,
            4,
            2,
            False,
            False,
            2,
            2,
            None,
            [1, 2, 4, 6, 7, 8, 10, 12],
            [1, 3, 5, 6, 7, 9, 11, 12],
        ),
        (
            14,
            4,
            2,
            False,
            False,
            None,
            None,
            [
                ["embedding", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "loss"],
            ],
            [1, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 14],
        ),
    ],
)
def test_cuda_graph_determine_first_last_layer_logic(
    total_num_layers,
    pp,
    vpp,
    account_for_embedding_in_pipeline_split,
    account_for_loss_in_pipeline_split,
    num_layers_in_first_pipeline_stage,
    num_layers_in_last_pipeline_stage,
    pp_layout,
    first_layer_numbers_golden,
    last_layer_numbers_golden,
):
    # Initialize RNG tracker
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    # Initialize parallel state
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pp, virtual_pipeline_model_parallel_size=vpp
    )

    # initialize model
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    hidden_size = 128
    transformer_config = TransformerConfig(
        num_layers=total_num_layers,
        hidden_size=hidden_size,
        num_attention_heads=1,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_size=pp,
        deallocate_pipeline_outputs=True,
        cuda_graph_impl="local",
        use_te_rng_tracker=True,
        account_for_embedding_in_pipeline_split=account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split=account_for_loss_in_pipeline_split,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        pipeline_model_parallel_layout=pp_layout,
    )
    model = []
    for i in range(vpp or 1):
        this_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=1024,
            position_embedding_type="rope",
            vp_stage=i,
        ).cuda()
        model.append(this_model)

    # create runner by running a fake forward pass
    sequence_length, micro_batch_size = 32, 1
    hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size)).cuda()
    attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
    for m in model:
        _ = m(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=hidden_states,
        )

    # Check if cuda graph is correctly setting is first/last layer
    for m in model:
        for l in m.decoder.layers:
            assert hasattr(l, "cudagraph_manager")
            assert (
                len(l.cudagraph_manager.cudagraph_runners) == 1
            ), "Cuda graph runner should be created"
            runner = l.cudagraph_manager.cudagraph_runners[0]
            assert runner.is_first_layer is not None and runner.is_last_layer is not None
            assert runner.is_first_layer == (l.layer_number in first_layer_numbers_golden)
            assert runner.is_last_layer == (l.layer_number in last_layer_numbers_golden)

            del l.cudagraph_manager.cudagraph_runners[0].fwd_graph

    # Destroy all captured graphs deterministically
    for m in model:
        for l in m.decoder.layers:
            for runner in getattr(l.cudagraph_manager, "cudagraph_runners", []):
                # Safely delete both graphs if present
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph

    # Ensure all pending work is complete and graph destruction runs now
    torch.cuda.synchronize()

    # Teardown
    Utils.destroy_model_parallel()
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    CudaGraphManager.global_mempool = None
    CudaGraphManager.fwd_mempools = None
    CudaGraphManager.bwd_mempools = None


class TestLLaVACudaGraph:
    """Test CUDA graphs with LLaVA model focusing on is_last_layer logic for encoder/decoder transitions."""

    def setup_method(self, method):
        # Initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
        )
        model_parallel_cuda_manual_seed(123)

        from copy import deepcopy

        from megatron.core.models.multimodal.llava_model import LLaVAModel
        from megatron.core.models.vision.vit_layer_specs import (
            get_vit_layer_with_transformer_engine_spec,
        )

        # Create language transformer config with CUDA graphs enabled
        self.language_hidden_size = 64
        self.language_num_attention_heads = 4
        language_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.language_hidden_size,
            num_attention_heads=self.language_num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="local",  # Enable CUDA graphs
        )

        # Create vision transformer config
        vision_config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=True,
            cuda_graph_impl="local",  # Enable CUDA graphs for vision model too
        )

        # Create vision projection config
        vision_projection_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.language_hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )

        # Get layer specs
        language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_projection_spec = deepcopy(get_submodules(language_layer_submodules.mlp))
        assert isinstance(vision_projection_spec, MLPSubmodules)

        # Set vision model type
        vision_config.vision_model_type = "clip"
        language_config.language_model_type = "dummy"

        # Create LLaVA model with both encoder and decoder
        self.llava_model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=language_layer_submodules
            ),
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_llava_cudagraph_is_last_layer_logic(self):
        """Test that is_last_layer logic correctly resets prev_bwd_hidden_state_inputgrad for LLaVA models."""

        # Move model to CUDA
        self.llava_model.cuda()

        set_current_microbatch(self.llava_model.vision_model, 1)
        set_current_microbatch(self.llava_model.language_model, 1)

        # Create test inputs
        batch_size = 2
        seq_length = 1024
        num_images = 1

        images = torch.ones((num_images, 3, 336, 336), dtype=torch.float32).cuda()

        # Create text input with image tokens
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).cuda()
        # Insert image token (using default image token index)
        input_ids[0, 5] = self.llava_model.image_token_index

        position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = None

        # Create labels and loss mask for training
        labels = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).cuda()
        loss_mask = torch.ones((batch_size, seq_length), dtype=torch.float32).cuda()

        # Create num_image_tiles
        num_image_tiles = torch.ones(num_images, dtype=torch.int).cuda()

        # First forward pass - this should record the CUDA graphs
        output1, loss_mask1 = self.llava_model(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )

        # Verify that CUDA graph managers were created
        if hasattr(self.llava_model.vision_model, 'decoder') and hasattr(
            self.llava_model.vision_model.decoder, 'layers'
        ):
            for layer in self.llava_model.vision_model.decoder.layers:
                if hasattr(layer, 'cudagraph_manager'):
                    assert (
                        layer.cudagraph_manager is not None
                    ), "Vision model layers should have CUDA graph managers"

        if hasattr(self.llava_model.language_model, 'decoder') and hasattr(
            self.llava_model.language_model.decoder, 'layers'
        ):
            for layer in self.llava_model.language_model.decoder.layers:
                if hasattr(layer, 'cudagraph_manager'):
                    assert (
                        layer.cudagraph_manager is not None
                    ), "Language model layers should have CUDA graph managers"

                    # Verify that CUDA graphs were created successfully
                    for runner in layer.cudagraph_manager.cudagraph_runners:
                        assert hasattr(runner, 'fwd_graph')
                        assert hasattr(runner, 'bwd_graph')

        # Perform backward pass to trigger backward graph recording
        if isinstance(output1, tuple):
            loss = output1[0].sum()
        else:
            loss = output1.sum()
        loss.backward()

        # Import the CUDA graph creation function
        from megatron.core.transformer.cuda_graphs import create_cudagraphs

        # Create the CUDA graphs - this is where the is_last_layer logic is tested
        create_cudagraphs()

        # Verify that CUDA graphs were created successfully
        assert _CudagraphGlobalRecord.cudagraph_created, "CUDA graphs should be created"

        if hasattr(self.llava_model.vision_model, 'decoder') and hasattr(
            self.llava_model.vision_model.decoder, 'layers'
        ):
            for layer in self.llava_model.vision_model.decoder.layers:
                del layer.cudagraph_manager.cudagraph_runners[0].fwd_graph
                del layer.cudagraph_manager.cudagraph_runners[0].bwd_graph

        if hasattr(self.llava_model.language_model, 'decoder') and hasattr(
            self.llava_model.language_model.decoder, 'layers'
        ):
            for layer in self.llava_model.language_model.decoder.layers:
                del layer.cudagraph_manager.cudagraph_runners[0].fwd_graph
                del layer.cudagraph_manager.cudagraph_runners[0].bwd_graph


class TestParallelHybridBlockCudagraphs:
    def setup_method(self, method):
        # initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

        # Ensure that this test is capturing to a fresh memory pool.
        CudaGraphManager.global_mempool = None

        def get_pg_collection():
            return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

        def get_mamba_block(hybrid_layer_pattern):
            layer_type_list = validate_segment_layers(hybrid_layer_pattern)
            transformer_config = TransformerConfig(
                hidden_size=256,  # The Mamba layer places several constraints on this
                # Need to specify num_attention_heads and num_layers or TransformerConfig
                # will generate errors.
                num_layers=len(layer_type_list),
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl="local",
            )
            modules = hybrid_stack_spec.submodules
            return HybridStack(
                transformer_config,
                modules,
                layer_type_list=layer_type_list,
                pp_layer_offset=0,
                pg_collection=get_pg_collection(),
            )

        self.mamba_block = get_mamba_block(hybrid_layer_pattern="M-M*-")
        self.transformer_config = self.mamba_block.config

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_gpu_cudagraph(self):
        parallel_mamba_block = self.mamba_block
        parallel_mamba_block.cuda()

        # [sequence length, batch size, hidden size]
        sequence_length = 32
        micro_batch_size = 2
        transformer_config: TransformerConfig = parallel_mamba_block.config
        num_layers = transformer_config.num_layers
        hidden_size = transformer_config.hidden_size
        hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_mamba_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        for _ in range(num_layers):
            assert hasattr(parallel_mamba_block.layers[0], "cudagraph_manager")
            assert len(parallel_mamba_block.layers[0].cudagraph_manager.cudagraph_runners) == 1

            del parallel_mamba_block.layers[_].cudagraph_manager.cudagraph_runners[0].fwd_graph

    def test_mhc_hybrid_layers_are_te_cudagraph_capturable(self):
        """Regression: a mHC-enabled HybridStack must expose graph-capturable layers.

        When ``enable_hyper_connections=True``, ``HybridStack`` wraps every layer in
        ``HyperConnectionHybridLayer``. That wrapper must subclass
        ``GraphableMegatronModule`` and be recognized by ``_layer_is_graphable`` so TE
        cuda-graph discovery finds the wrapped layers. Before the fix the wrapper
        subclassed plain ``MegatronModule``, so discovery rejected every layer (0
        graphable) and CUDA graph capture was silently skipped for the whole hybrid
        model -- making the mHC hybrid run fully eager (several times slower than the
        graphed GPT mHC path). This test fails on the pre-fix code via both assertions.
        """
        # The wrapper must be graph-capturable by construction.
        assert issubclass(HyperConnectionHybridLayer, GraphableMegatronModule)

        layer_type_list = validate_segment_layers("M-M*-")  # mamba / mlp / attention mix
        config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            enable_hyper_connections=True,
            num_residual_streams=4,
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.mamba, CudaGraphModule.mlp],
        )
        block = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "pp", "cp"]
            ),
        )

        # Every layer is wrapped, and the wrappers are discoverable as graphable.
        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in block.layers)
        graphable = [layer for layer in block.layers if _layer_is_graphable(layer, config)]
        assert len(graphable) > 0, (
            "mHC HybridStack produced 0 graphable layers -- TE cuda-graph capture would "
            "be silently skipped for the entire model (the pre-fix bug)."
        )


# Global storage for comparing unique buffer counts across different num_microbatches,
# keyed by (pp_size, vpp_size)
_unique_buffer_counts = {}


class TestTECudaGraphHelper:
    def setup_method(self, method):
        # Initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        # Note: _unique_buffer_counts is intentionally NOT cleared here so we can
        # compare values across parametrized test runs

    @pytest.mark.parametrize("num_microbatches", [16, 64, 256])
    @pytest.mark.parametrize("pp_size", [1, 2, 4])
    @pytest.mark.parametrize("vpp_size", [None, 2])
    def test_get_cuda_graph_input_data(self, num_microbatches, pp_size, vpp_size):
        """Test _get_cuda_graph_input_data function in TECudaGraphHelper."""

        if vpp_size and pp_size == 1:
            pytest.skip("vpp_size must be None when pp_size is 1")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
        )

        # Set up test configuration
        seq_length = 128
        micro_batch_size = 2
        num_layers = 8
        vocab_size = 1024
        hidden_size = 64
        num_attention_heads = 4

        # Initialize num_microbatches calculator
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=micro_batch_size * num_microbatches,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )

        # Create transformer config directly
        transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            use_te_rng_tracker=True,
            bf16=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
            pipeline_dtype=torch.bfloat16,
            context_parallel_size=1,
        )

        # Create model
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        model = []
        for i in range(vpp_size or 1):
            this_model = GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=vocab_size,
                max_sequence_length=seq_length,
                parallel_output=True,
                position_embedding_type="rope",
                vp_stage=i if vpp_size else None,
            ).cuda()
            model.append(this_model)

        # Initialize TECudaGraphHelper
        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=transformer_config,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            optimizers=[],
        )

        # Call _get_cuda_graph_input_data (which internally calls _get_sample_arguments)
        sample_args, make_graphed_callables_kwargs = cuda_graph_helper._get_cuda_graph_input_data()

        # Extract sample_kwargs from the kwargs dict
        # For TE >= 1.10.0, sample_kwargs should always be present
        assert (
            'sample_kwargs' in make_graphed_callables_kwargs
        ), "sample_kwargs should be present in make_graphed_callables_kwargs for TE >= 1.10.0"
        sample_kwargs = make_graphed_callables_kwargs['sample_kwargs']

        # Basic checks
        num_graphable_layers = len(cuda_graph_helper.flattened_callables)
        if pp_size > 1:
            expected_length = num_graphable_layers * num_microbatches
        else:
            expected_length = num_graphable_layers
        assert len(sample_args) == expected_length, (
            f"sample_args length mismatch: expected {expected_length}, " f"got {len(sample_args)}"
        )
        assert len(sample_kwargs) == expected_length, (
            f"sample_kwargs length mismatch: expected {expected_length}, "
            f"got {len(sample_kwargs)}"
        )

        # Check that all elements are not None
        for i, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            assert args_item is not None, f"sample_args[{i}] is None"
            assert kwargs_item is not None, f"sample_kwargs[{i}] is None"
            assert isinstance(args_item, tuple), f"sample_args[{i}] should be a tuple"
            assert isinstance(kwargs_item, dict), f"sample_kwargs[{i}] should be a dict"
            assert len(args_item) > 0, f"sample_args[{i}] should not be empty"
            # Check that hidden_states is present
            assert "hidden_states" in kwargs_item or (
                len(args_item) > 0 and torch.is_tensor(args_item[0])
            ), f"sample_args[{i}] or sample_kwargs[{i}] should contain hidden_states"

        # Check tensor properties
        for i, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Get hidden_states from args or kwargs
            if len(args_item) > 0 and torch.is_tensor(args_item[0]):
                hidden_states = args_item[0]
            elif "hidden_states" in kwargs_item:
                hidden_states = kwargs_item["hidden_states"]
            else:
                continue

            assert torch.is_tensor(hidden_states), f"hidden_states at index {i} should be a tensor"
            # Check shape matches expected (accounting for TP/CP)
            expected_seq_len = seq_length // transformer_config.context_parallel_size
            if transformer_config.sequence_parallel:
                expected_seq_len = expected_seq_len // transformer_config.tensor_model_parallel_size
            assert hidden_states.shape[0] == expected_seq_len, (
                f"hidden_states seq_len mismatch at index {i}: "
                f"expected {expected_seq_len}, got {hidden_states.shape[0]}"
            )
            assert hidden_states.shape[1] == micro_batch_size, (
                f"hidden_states batch_size mismatch at index {i}: "
                f"expected {micro_batch_size}, got {hidden_states.shape[1]}"
            )
            assert hidden_states.shape[2] == transformer_config.hidden_size, (
                f"hidden_states hidden_size mismatch at index {i}: "
                f"expected {transformer_config.hidden_size}, got {hidden_states.shape[2]}"
            )

        # Memory optimization check: verify that buffers with same signature are reused
        # Create a mapping of sample_keys to indices
        sample_keys_to_indices = {}
        for idx, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Create sample_keys similar to the function
            args_keys = tuple((t.shape, t.dtype, t.layout) for t in args_item if torch.is_tensor(t))
            kwargs_keys = tuple(
                (k, v.shape, v.dtype, v.layout)
                for k, v in sorted(kwargs_item.items())
                if torch.is_tensor(v)
            )
            sample_keys = args_keys + kwargs_keys

            if sample_keys not in sample_keys_to_indices:
                sample_keys_to_indices[sample_keys] = []
            sample_keys_to_indices[sample_keys].append(idx)

        # Check that buffers with same signature share references (memory optimization)
        # The optimization reuses buffers when:
        # 1. They have the same signature (shape, dtype, layout)
        # 2. The backward pass of the original buffer has completed
        # 3. A new forward pass with matching signature needs a buffer
        # Count how many times each tensor is reused
        unique_tensors = set()
        tensor_reuse_count = {}
        for idx, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Get the first tensor from args (hidden_states)
            if len(args_item) > 0 and torch.is_tensor(args_item[0]):
                tensor_ptr = args_item[0].data_ptr()
                unique_tensors.add(tensor_ptr)
                tensor_reuse_count[tensor_ptr] = tensor_reuse_count.get(tensor_ptr, 0) + 1

        # With memory optimization, we should see some buffers reused
        # (i.e., some tensors should appear multiple times)
        max_reuse = max(tensor_reuse_count.values()) if tensor_reuse_count else 0
        total_entries = len(sample_args)
        unique_buffer_count = len(unique_tensors)

        # Verify that memory optimization is working:
        # - The number of unique buffers should be <= total entries
        # - With the 1F1B schedule and multiple microbatches, we should see some buffer reuse
        # - The number of unique buffers should be bounded as num_microbatches grows.
        assert unique_buffer_count <= total_entries, (
            f"Memory optimization check: unique_buffer_count ({unique_buffer_count}) "
            f"should be <= total_entries ({total_entries})"
        )
        global _unique_buffer_counts
        # Use (pp_size, vpp_size) as key to track unique buffer counts per configuration
        config_key = (pp_size, vpp_size)
        if config_key not in _unique_buffer_counts:
            _unique_buffer_counts[config_key] = unique_buffer_count
        else:
            assert unique_buffer_count == _unique_buffer_counts[config_key], (
                f"Unique buffer count mismatch: expected {_unique_buffer_counts[config_key]}, "
                f"got {unique_buffer_count}"
            )

        # Verify that buffers with the same signature can potentially be reused
        # (the actual reuse depends on the schedule, but the mechanism should work)
        if expected_length > 1:
            # Check that we have multiple entries with the same signature
            has_duplicate_signatures = any(
                len(indices) > 1 for indices in sample_keys_to_indices.values()
            )
            assert has_duplicate_signatures, (
                "Memory optimization: expected duplicate signatures for buffer reuse, "
                "but all signatures are unique"
            )

            # We tested with a large number of microbatches, so we should see some buffer reuse.
            if pp_size > 1:
                assert max_reuse > 1, "Expected some buffer reuse"

        # Verify that make_graphed_callables_kwargs contains expected keys
        assert (
            '_order' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain '_order'"
        assert (
            'num_warmup_iters' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain 'num_warmup_iters'"
        assert (
            'allow_unused_input' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain 'allow_unused_input'"

        # Verify the order in kwargs matches expectations
        order = make_graphed_callables_kwargs['_order']
        num_model_chunks = cuda_graph_helper.num_model_chunks
        forward_count = sum(1 for chunk_id in order if chunk_id > 0)
        if pp_size > 1:
            # Verify that all forward passes in order have corresponding entries in sample_args
            assert forward_count == num_microbatches * num_model_chunks, (
                f"Forward count mismatch: expected {num_microbatches * num_model_chunks}, "
                f"got {forward_count}"
            )
            expected_order_length = num_microbatches * num_model_chunks * 2
        else:
            assert num_model_chunks == 1, "Expected only one model chunk for pp_size == 1"
            assert forward_count == 1, "Expected only one forward pass for pp_size == 1"
            expected_order_length = 2
        assert (
            len(order) == expected_order_length
        ), f"Order length mismatch: expected {expected_order_length}, got {len(order)}"


class TestRequiredNumMicrobatchSlots:
    """Pure-Python tests for ``_get_required_num_microbatch_slots_from_order``.

    The method derives the smallest cuda-graph slot count that guarantees no
    in-flight microbatch's static buffer is reused before its backward
    completes. ``order`` is a 1F1B / interleaved-1F1B schedule transcript
    where ``+chunk_id`` denotes a forward and ``-chunk_id`` a backward.
    Non-integer entries (e.g. ``0.5`` for wgrad sub-steps) are skipped.
    """

    @staticmethod
    def _slots(order, num_chunks):
        return TECudaGraphHelper._get_required_num_microbatch_slots_from_order(order, num_chunks)

    def test_single_chunk_single_microbatch(self):
        # F0 then B0: one slot is enough.
        assert self._slots([1, -1], 1) == 1

    def test_single_chunk_pp_pipeline_4_microbatches_pp2(self):
        # PP=2 1F1B with 4 microbatches: warmup F-F, then F-B-F-B-..., then cooldown B-B.
        # Max in-flight = 2.
        order = [1, 1, -1, 1, -1, 1, -1, -1]
        assert self._slots(order, 1) == 2

    def test_two_chunks_independent(self):
        # Two model chunks (VPP=2), each running a tiny PP=2-style 1F1B in turn.
        # Per chunk max in-flight = 2 -> 2 slots.
        order = [1, 1, -1, -1, 2, 2, -2, -2]
        assert self._slots(order, 2) == 2

    def test_two_chunks_interleaved(self):
        # Worst case: forwards stack up across chunks before any backward.
        # F0 F0 F1 F1 B1 B1 B0 B0 -> per-chunk max in-flight = 2.
        order = [1, 1, 2, 2, -2, -2, -1, -1]
        assert self._slots(order, 2) == 2

    def test_skips_non_integer_entries(self):
        # Float c_ids (e.g. 0.5 for wgrad sub-steps) must be ignored.
        order = [1, 0.5, -0.5, -1]
        assert self._slots(order, 1) == 1

    def test_minimum_slot_is_one(self):
        # Empty / no-op order still returns at least 1 (we always need a slot).
        assert self._slots([], 1) == 1

    def test_unbalanced_order_asserts(self):
        # Forward without matching backward -> outstanding != 0 at end -> assert.
        with pytest.raises(AssertionError):
            self._slots([1], 1)

    def test_negative_outstanding_asserts(self):
        # Backward before any forward for a chunk -> outstanding goes negative.
        with pytest.raises(AssertionError):
            self._slots([-1], 1)


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


class TestPartialCudaGraph:
    """Test that CUDA graph outputs match non-CUDA graph outputs for various scopes."""

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        self.tp_size = 2
        self.cp_size = 2
        self.cuda_graph_helper = None
        # Store original environment variable values
        self.original_env = {
            'CUDA_DEVICE_MAX_CONNECTIONS': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS'),
            'NVTE_ALLOW_NONDETERMINISTIC_ALGO': os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO'),
        }
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'

    def teardown_method(self, method):
        # Restore original environment variable values
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None
        gc.collect()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_decoder_block_spec,
        **config_kwargs,
    ):
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            config,
            use_transformer_engine=True,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
        )
        if args.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=True
            )
        else:
            mtp_block_spec = None
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            mtp_block_spec=mtp_block_spec,
        )

    def create_test_args(
        self, cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, ep_size, **kwargs
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_cuda_graphs.py']
        args = parse_args()
        args.num_layers = 4
        args.mtp_num_layers = 1
        args.vocab_size = 1024
        args.hidden_size = 512
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.global_batch_size = self.micro_batch_size * 8 // self.tp_size // self.cp_size
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = self.seq_length
        args.tensor_model_parallel_size = self.tp_size
        args.sequence_parallel = True if self.tp_size > 1 else False
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = self.cp_size
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.use_distributed_optimizer = True
        args.position_embedding_type = "rope"
        args.rotary_percent = 1.0
        args.hidden_dropout = 0.0
        args.attention_dropout = 0.0

        # MoE settings
        args.num_experts = 4
        args.expert_model_parallel_size = ep_size
        args.expert_tensor_parallel_size = 1 if ep_size > 1 else self.tp_size
        args.moe_shared_expert_intermediate_size = 1024
        args.moe_layer_freq = [0, 0, 1, 1]
        args.moe_permute_fusion = True
        args.moe_router_fusion = True
        args.moe_router_topk = 2
        args.moe_router_dtype = "fp32"

        # CUDA graph settings
        args.cuda_graph_impl = cuda_graph_impl
        args.cuda_graph_modules = cuda_graph_modules
        args.cuda_graph_warmup_steps = cuda_graph_warmup_steps

        # fp8 settings
        if fp8_available:
            args.fp8 = "e4m3"
            args.fp8_recipe = "tensorwise"
            args.first_last_layers_bf16 = True
            args.num_layers_at_start_in_bf16 = 1
            args.num_layers_at_end_in_bf16 = 1

        for key, value in kwargs.items():
            assert hasattr(args, key) or hasattr(TransformerConfig, key), f"Unknown argument: {key}"
            setattr(args, key, value)

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size, cp_size):
        data = list(range(seq_length // cp_size))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length // cp_size, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length // cp_size).repeat((micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def _run_test_helper(
        self, ep_size, cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, **kwargs
    ):
        """Test fp8_param with gpt_model."""
        args = self.create_test_args(
            cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, ep_size, **kwargs
        )

        set_args(args)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size, self.cp_size
        )

        gpt_model, optimizer, _ = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.

        if cuda_graph_impl == "transformer_engine":
            self.cuda_graph_helper = TECudaGraphHelper(
                model=gpt_model,
                config=gpt_model[0].config,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                optimizers=[optimizer],
            )

        loss_list = []

        for i in range(100):
            gpt_model[0].zero_grad_buffer()
            optimizer.zero_grad()

            # Capture CUDA graphs after warmup if helper is provided
            if self.cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                self.cuda_graph_helper.create_cudagraphs()

            gpt_model[0].set_is_first_microbatch()
            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

            # Check output shapes
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length // self.cp_size

            # Verify gradients
            loss = output.mean()
            loss.backward()

            for param in gpt_model[0].parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

            loss_list.append(loss.item())

        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None

        return torch.tensor(loss_list)

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    @pytest.mark.parametrize("ep_size", [1, 4])
    @pytest.mark.parametrize("moe_dropless_dispatcher", [False, True])
    @pytest.mark.parametrize("moe_dispatcher_type", ["alltoall", "deepep", "hybridep"])
    def test_moe_partial_cudagraph(self, ep_size, moe_dropless_dispatcher, moe_dispatcher_type):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1 if ep_size > 1 else self.tp_size,
            expert_model_parallel_size=ep_size,
        )

        extra_kwargs = {}
        if moe_dispatcher_type == "deepep":
            if not is_deep_ep_available():
                pytest.skip("Deep EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "deepep"
        elif moe_dispatcher_type == "hybridep":
            if not is_hybrid_ep_available():
                pytest.skip("Hybrid EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "hybridep"
        else:
            extra_kwargs["moe_token_dispatcher_type"] = moe_dispatcher_type
        if not moe_dropless_dispatcher:
            if moe_dispatcher_type == "deepep":
                pytest.skip("Deep EP doesn't support drop&pad MoE")
            extra_kwargs["moe_expert_capacity_factor"] = 1.0
            extra_kwargs["moe_pad_expert_input_to_capacity"] = True

        loss_list_ref = self._run_test_helper(ep_size, "none", None, 0, **extra_kwargs)
        for cuda_graph_modules in [
            None,
            [CudaGraphModule.attn],
            [CudaGraphModule.moe],
            [CudaGraphModule.mlp, CudaGraphModule.moe_router],
            [
                CudaGraphModule.attn,
                CudaGraphModule.mlp,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            ],
        ]:
            if (moe_dropless_dispatcher or moe_dispatcher_type == "hybridep") and (
                cuda_graph_modules is None or CudaGraphModule.moe in cuda_graph_modules
            ):
                # Dropless MoE or Hybrid EP doesn't work with "moe" scope cudagraph. Skip.
                continue
            cuda_graph_warmup_steps = 3
            loss_list = self._run_test_helper(
                ep_size,
                "transformer_engine",
                cuda_graph_modules,
                cuda_graph_warmup_steps,
                **extra_kwargs,
            )
            assert torch.equal(loss_list, loss_list_ref)

        if moe_dispatcher_type == "hybridep":
            reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    @pytest.mark.parametrize("ep_size", [1, 4])
    def test_mhc_moe_partial_cudagraph(self, ep_size):
        """Test that mHC (Hyper Connection) layers produce identical loss curves
        with and without TE partial CUDA graph capture.

        This validates the fix where HyperConnectionTransformerLayer overrides
        _te_cuda_graph_replay_impl (not _te_cuda_graph_replay) so that the parent's
        delay_offload_until_cuda_graph lifecycle and overlap_moe_expert_parallel_comm
        handling are preserved.
        """
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1 if ep_size > 1 else self.tp_size,
            expert_model_parallel_size=ep_size,
        )

        extra_kwargs = {
            "enable_hyper_connections": True,
            "num_residual_streams": 4,
            "mtp_num_layers": None,  # mHC is incompatible with MTP
        }

        loss_list_ref = self._run_test_helper(ep_size, "none", None, 0, **extra_kwargs)
        for cuda_graph_modules in [
            [CudaGraphModule.attn],
            [CudaGraphModule.mlp, CudaGraphModule.moe_router],
            [
                CudaGraphModule.attn,
                CudaGraphModule.mlp,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            ],
        ]:
            cuda_graph_warmup_steps = 3
            loss_list = self._run_test_helper(
                ep_size,
                "transformer_engine",
                cuda_graph_modules,
                cuda_graph_warmup_steps,
                **extra_kwargs,
            )
            assert torch.equal(loss_list, loss_list_ref), (
                f"mHC loss mismatch with cuda_graph_modules={cuda_graph_modules}, ep_size={ep_size}. "
                f"Max diff: {torch.max(torch.abs(loss_list - loss_list_ref))}"
            )

        Utils.destroy_model_parallel()


class _SimpleModule(MegatronModule):
    """Minimal MegatronModule for testing CudaGraphManager with function_name."""

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def my_op(self, x):
        return self.linear(x)


class _SimpleNonModule:
    """non-nn.Module base_module for testing the function_name= form of `CudaGraphManager`."""

    def __init__(self, config):
        self.weight = torch.randn(config.hidden_size, config.hidden_size, device="cuda")

    def my_op(self, x):
        return x @ self.weight


def _make_simple_module(config):
    return _SimpleModule(config).cuda().eval()


def _make_simple_non_module(config):
    return _SimpleNonModule(config)


class TestInlineCaptureManager:
    """Tests for CudaGraphManager with inline_capture, function_name, eager, and cache_key."""

    def _make_config(self):
        return TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            inference_rng_tracker=True,
        )

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def teardown_method(self, method):
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "make_module",
        [
            pytest.param(_make_simple_module, id="nn_module"),
            pytest.param(_make_simple_non_module, id="plain_class"),
        ],
    )
    @torch.inference_mode()
    def test_inline_capture_matches_eager(self, make_module):
        """Inline-captured graph output must match eager execution."""
        config = self._make_config()
        module = make_module(config)

        # Get eager reference before wrapping
        x = torch.randn(4, config.hidden_size, device="cuda")
        eager_out = module.my_op(x).clone()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        # First call captures, second replays
        graph_out_1 = module.my_op(x)
        graph_out_2 = module.my_op(x)
        assert torch.equal(eager_out, graph_out_1)
        assert torch.equal(eager_out, graph_out_2)
        assert len(mgr.cudagraph_runners) == 1
        assert mgr.cudagraph_runners[0].fwd_graph_recorded

    @torch.inference_mode()
    def test_eager_bypass(self):
        """eager=True must bypass graph capture entirely."""
        config = self._make_config()
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        _ = module.my_op(x, eager=True)
        _ = module.my_op(x, eager=True)
        assert len(mgr.cudagraph_runners) == 0, "eager=True should not create runners"

    @torch.inference_mode()
    def test_cache_key_routing(self):
        """Different cache_keys must create separate runners."""
        config = self._make_config()
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        module.my_op(x, cache_key="key_a")
        module.my_op(x, cache_key="key_b")

        assert len(mgr.cudagraph_runners) == 2
        assert mgr.custom_cudagraphs_lookup_table["key_a"] is not None
        assert mgr.custom_cudagraphs_lookup_table["key_b"] is not None
        assert (
            mgr.custom_cudagraphs_lookup_table["key_a"]
            is not mgr.custom_cudagraphs_lookup_table["key_b"]
        )

        # Same key reuses the runner
        module.my_op(x, cache_key="key_a")
        assert len(mgr.cudagraph_runners) == 2

    @torch.inference_mode()
    def test_num_warmup_steps_override(self):
        """num_warmup_steps on the manager must override the config value on runners."""
        config = self._make_config()
        config.cuda_graph_warmup_steps = 3
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        module.my_op(x, cache_key="test")

        runner = mgr.cudagraph_runners[0]
        assert (
            runner.num_warmup_steps == 0
        ), f"Expected 0 warmup steps (manager override), got {runner.num_warmup_steps}"


class TestSkipFp8WeightUpdateTensor:
    """Regression test for the TE 2.15 ``set_skip_fp8_weight_update_tensor`` removal."""

    @staticmethod
    def _read_skip_tensor():
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        getter = getattr(FP8GlobalStateManager, "get_skip_fp8_weight_update_tensor", None)
        if getter is not None:
            return getter()
        return FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor

    @staticmethod
    def _reset_skip_tensor():
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        if "skip_fp8_weight_update_tensor" in vars(FP8GlobalStateManager):
            FP8GlobalStateManager.skip_fp8_weight_update_tensor = None
        qstate = getattr(FP8GlobalStateManager, "quantization_state", None)
        if qstate is not None and hasattr(qstate, "skip_fp8_weight_update_tensor"):
            qstate.skip_fp8_weight_update_tensor = None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sets_value_in_place(self):
        """Helper writes the right value and reuses the same storage across calls."""
        from megatron.core.transformer.cuda_graphs import _set_skip_fp8_weight_update_tensor

        self._reset_skip_tensor()
        try:
            _set_skip_fp8_weight_update_tensor(True)
            t = self._read_skip_tensor()
            assert t.shape == (1,) and t.dtype == torch.float32 and t.is_cuda
            assert t.item() == 1.0

            # data_ptr must stay stable so captured cudagraphs read the same address.
            ptr = t.data_ptr()
            _set_skip_fp8_weight_update_tensor(False)
            assert self._read_skip_tensor().data_ptr() == ptr
            assert self._read_skip_tensor().item() == 0.0
        finally:
            self._reset_skip_tensor()


if __name__ == "__main__":

    test = TestParallelTransformerBlockCudagraphs()
    test.setup_method(method=None)
    test.test_gpu_cudagraph()
    test.teardown_method(method=None)

    llava_test = TestLLaVACudaGraph()
    llava_test.setup_method(method=None)
    llava_test.test_llava_cudagraph_is_last_layer_logic()
    llava_test.teardown_method(method=None)

    test = TestPartialCudaGraph()
    test.setup_method(method=None)
    test.test_moe_partial_cudagraph(4, True, "alltoall")
    test.teardown_method(method=None)

    test = TestPartialCudaGraph()
    test.setup_method(method=None)
    test.test_mhc_moe_partial_cudagraph(4)
    test.teardown_method(method=None)
