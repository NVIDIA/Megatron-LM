# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Selection/lifecycle identity only; CPU does not execute deployment kernels."""
import inspect
from types import SimpleNamespace
from unittest.mock import patch

from megatron.lite.model.deepseek_v41.lite import protocol as native
from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model as NativeModel
from megatron.lite.model.deepseek_v41.vllm import protocol
from megatron.lite.model.deepseek_v41.vllm.model import DeepseekV41Model
from megatron.lite.model.registry import get_train_runtime_module, resolve_runtime_model_name


def test_attention_factory_accepts_core_bootstrap_before_hybrid_rope():
    from megatron.lite.model.deepseek_v41.vllm import attention

    captured = {}

    def capture(config, submodules, layer, **kwargs):
        captured['spec'] = submodules.core_attention
        return SimpleNamespace()

    with patch.object(attention, 'SWAAttention', capture):
        attention.build_attention(
            SimpleNamespace(csa_compress_ratios=[0]), 0, SimpleNamespace(groups=None, release=None)
        )
    spec = captured['spec']
    # Core Attention constructs this before Hybrid adds ratio and rotary kwargs.
    inspect.signature(spec.module).bind(
        config=None,
        layer_number=1,
        pg_collection=None,
        attn_mask_type=None,
        attention_type='self',
        cp_comm_type=None,
        softmax_scale=None,
        **spec.params,
    )


def test_registry_reuses_native_lifecycle_and_dist_opt():
    name = resolve_runtime_model_name('deepseek_v41', 'vllm')
    assert name == 'deepseek_v41_vllm'
    assert get_train_runtime_module(name) is protocol
    for symbol in (
        'ImplConfig',
        'build_model_config',
        'forward_step',
        'load_hf_weights',
        'validate_impl_cfg_keys',
    ):
        assert getattr(protocol, symbol) is getattr(native, symbol)
    assert protocol.ImplConfig().optimizer == 'adam'
    assert inspect.signature(native.build_model).parameters['_model_class'].default is None


def test_attention_provider_keeps_compressed_gap_explicit_and_native_lifecycle():
    from megatron.lite.model.deepseek_v41.lite.components.attention import DS41SelfAttention
    from megatron.lite.model.deepseek_v41.vllm.attention import SWAAttention

    assert issubclass(SWAAttention, DS41SelfAttention)
    assert 'ratio2 parity pending' in DeepseekV41Model.deployment_providers['attention']
    assert DeepseekV41Model.set_input_tensor is NativeModel.set_input_tensor
    assert DeepseekV41Model.forward is NativeModel.forward
