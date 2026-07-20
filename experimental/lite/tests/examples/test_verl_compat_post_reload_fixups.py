# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

from verl_mlite import compat


def test_post_reload_restores_attention_and_only_finalizes_moe(monkeypatch) -> None:
    events = []

    class Extension:
        def __init__(self, model):
            self.model_runner = SimpleNamespace(model=model)

        def update_weights_from_ipc(self):
            events.append("load")
            return "loaded"

    utils = ModuleType("verl.workers.rollout.vllm_rollout.utils")
    utils.vLLMColocateWorkerExtension = Extension
    dsv4_utils = ModuleType("verl.utils.vllm.vllm_dsv4_fp8_utils")
    dsv4_utils.is_deepseek_v4_model = lambda model: True
    routed_experts = ModuleType(
        "vllm.model_executor.layers.fused_moe.routed_experts"
    )
    fp8 = ModuleType("vllm.model_executor.layers.quantization.fp8")

    class RoutedExperts:
        pass

    class Fp8MoEMethod:
        def process_weights_after_loading(self, module):
            events.append("finalize-moe")

    routed_experts.RoutedExperts = RoutedExperts
    fp8.Fp8MoEMethod = Fp8MoEMethod
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, dsv4_utils.__name__, dsv4_utils)
    monkeypatch.setitem(sys.modules, routed_experts.__name__, routed_experts)
    monkeypatch.setitem(sys.modules, fp8.__name__, fp8)
    monkeypatch.setattr(compat, "_vllm_importable", lambda: True)
    monkeypatch.setattr(
        compat,
        "_restore_dsv4_attn_sink_padding",
        lambda model: events.append("restore-attention") or 4,
    )

    class DenseQuantMethod:
        def process_weights_after_loading(self, module):
            raise AssertionError("online reload must not re-finalize dense FP8")

    moe = RoutedExperts()
    moe.quant_method = Fp8MoEMethod()
    dense = SimpleNamespace(quant_method=DenseQuantMethod())
    model = SimpleNamespace(
        modules=lambda: iter((moe, dense)),
    )
    assert compat._patch_verl_dsv4_fp8_process_weights()
    assert Extension(model).update_weights_from_ipc() == "loaded"
    assert events == ["load", "restore-attention", "finalize-moe"]
