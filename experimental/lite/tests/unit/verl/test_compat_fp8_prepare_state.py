# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

from verl_mlite import compat


def test_pure_fp8_ds4_prepare_state_is_promoted(monkeypatch) -> None:
    fp8_utils = ModuleType("verl.utils.vllm.vllm_fp8_utils")
    calls = []
    fp8_utils.prepare_quanted_weights_for_loading = (
        lambda model_runner: calls.append(model_runner) or False
    )
    dsv4_utils = ModuleType("verl.utils.vllm.vllm_dsv4_fp8_utils")
    dsv4_utils.is_deepseek_v4_model = lambda model: True

    routed_module = ModuleType(
        "vllm.model_executor.layers.fused_moe.routed_experts"
    )
    fp8_module = ModuleType("vllm.model_executor.layers.quantization.fp8")

    class RoutedExperts:
        pass

    class Fp8MoEMethod:
        pass

    routed_module.RoutedExperts = RoutedExperts
    fp8_module.Fp8MoEMethod = Fp8MoEMethod
    expert = RoutedExperts()
    expert.quant_method = Fp8MoEMethod()
    model = SimpleNamespace(modules=lambda: [expert])
    runner = SimpleNamespace(model=model)

    monkeypatch.setitem(sys.modules, fp8_utils.__name__, fp8_utils)
    monkeypatch.setitem(sys.modules, dsv4_utils.__name__, dsv4_utils)
    monkeypatch.setitem(sys.modules, routed_module.__name__, routed_module)
    monkeypatch.setitem(sys.modules, fp8_module.__name__, fp8_module)
    monkeypatch.setattr(compat, "_vllm_importable", lambda: True)

    assert compat._patch_verl_dsv4_fp8_prepare_state()
    assert fp8_utils.prepare_quanted_weights_for_loading(runner) is True
    assert calls == [runner]


def test_non_ds4_false_prepare_state_stays_false(monkeypatch) -> None:
    fp8_utils = ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fp8_utils.prepare_quanted_weights_for_loading = lambda model_runner: False
    dsv4_utils = ModuleType("verl.utils.vllm.vllm_dsv4_fp8_utils")
    dsv4_utils.is_deepseek_v4_model = lambda model: False
    routed_module = ModuleType(
        "vllm.model_executor.layers.fused_moe.routed_experts"
    )
    fp8_module = ModuleType("vllm.model_executor.layers.quantization.fp8")
    routed_module.RoutedExperts = type("RoutedExperts", (), {})
    fp8_module.Fp8MoEMethod = type("Fp8MoEMethod", (), {})

    monkeypatch.setitem(sys.modules, fp8_utils.__name__, fp8_utils)
    monkeypatch.setitem(sys.modules, dsv4_utils.__name__, dsv4_utils)
    monkeypatch.setitem(sys.modules, routed_module.__name__, routed_module)
    monkeypatch.setitem(sys.modules, fp8_module.__name__, fp8_module)
    monkeypatch.setattr(compat, "_vllm_importable", lambda: True)

    assert compat._patch_verl_dsv4_fp8_prepare_state()
    runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: []))
    assert fp8_utils.prepare_quanted_weights_for_loading(runner) is False
