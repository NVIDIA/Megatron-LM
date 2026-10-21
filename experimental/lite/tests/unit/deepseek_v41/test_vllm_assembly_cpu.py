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
    for symbol in ('build_model_config', 'forward_step', 'load_hf_weights'):
        assert getattr(protocol, symbol) is getattr(native, symbol)
    assert issubclass(protocol.ImplConfig, native.ImplConfig)
    assert protocol.ImplConfig().optimizer == 'adam'
    assert (protocol.ImplConfig().attention_core, protocol.ImplConfig().head_dtype) == (
        'flashmla',
        'float32',
    )
    assert inspect.signature(native.build_model).parameters['_model_class'].default is None


def test_deployment_keeps_native_pipeline_and_explicit_output_boundary():
    from megatron.lite.model.deepseek_v41.lite.components.attention import DS41SelfAttention
    from megatron.lite.model.deepseek_v41.vllm.attention import SWAAttention

    assert issubclass(SWAAttention, DS41SelfAttention)
    assert 'ratio2 parity pending' in DeepseekV41Model.deployment_providers['attention']
    assert DeepseekV41Model.set_input_tensor is NativeModel.set_input_tensor
    assert DeepseekV41Model.forward is not NativeModel.forward
    assert (
        inspect.signature(DeepseekV41Model.forward).parameters['labels'].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_native_hf_construction_recovers_release_token_map(tmp_path):
    from unittest.mock import Mock

    impl = protocol.ImplConfig(hf_path=str(tmp_path))
    model_cfg = SimpleNamespace(engram_layer_ids=[1])
    native_build = Mock(return_value=object())
    with patch.object(protocol, "release_token_map", return_value=[2, 0, 1]) as recover:
        with patch.object(native, "build_model", native_build):
            protocol.build_model(model_cfg, impl_cfg=impl)
            recovered = native_build.call_args.kwargs["impl_cfg"]
            assert recovered.token_map == [2, 0, 1]
            assert impl.token_map is None
            recover.assert_called_once_with(str(tmp_path))
            recover.reset_mock()
            (tmp_path / "token_map.pt").write_bytes(b"existence-only: native loader owns bytes")
            protocol.build_model(model_cfg, impl_cfg=impl)
            assert native_build.call_args.kwargs["impl_cfg"] is impl
            recover.assert_not_called()


def test_native_hf_construction_keeps_explicit_token_map(tmp_path):
    from unittest.mock import Mock

    impl = protocol.ImplConfig(hf_path=str(tmp_path), token_map=[1, 0])
    with patch.object(protocol, "release_token_map", side_effect=AssertionError("recomputed")):
        with patch.object(native, "build_model", Mock()) as build:
            protocol.build_model(SimpleNamespace(engram_layer_ids=[1]), impl_cfg=impl)
            assert build.call_args.kwargs["impl_cfg"] is impl


def test_recipe_options_are_explicit_and_future_modes_fail_before_build(monkeypatch):
    from unittest.mock import Mock

    import pytest

    monkeypatch.setenv("DSV41_ATTENTION_CORE", "mega")
    monkeypatch.setenv("DSV41_HEAD_DTYPE", "bfloat16")
    default = protocol.ImplConfig()
    assert (default.attention_core, default.head_dtype) == ("flashmla", "float32")
    protocol.validate_impl_cfg_keys({"attention_core": "mega", "head_dtype": "bfloat16"})
    with patch.object(native, "build_model", Mock()) as build:
        with pytest.raises(NotImplementedError):
            protocol.build_model(
                SimpleNamespace(), impl_cfg=protocol.ImplConfig(attention_core="mega")
            )
        build.assert_not_called()
        protocol.build_model(
            SimpleNamespace(engram_layer_ids=[]),
            impl_cfg=protocol.ImplConfig(head_dtype="bfloat16"),
        )
        assert build.call_args.kwargs["impl_cfg"].head_dtype == "bfloat16"
