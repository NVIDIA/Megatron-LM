"""Unit tests for the Megatron-Bridge runtime backend surface."""

from __future__ import annotations

import sys
import types

import pytest


def test_bridge_runtime_registered_and_lazy_constructible():
    from megatron.lite.runtime import RuntimeConfig, create_runtime
    from megatron.lite.runtime.backends import RUNTIME_REGISTRY
    from megatron.lite.runtime.backends.bridge.config import BridgeConfig
    from megatron.lite.runtime.backends.bridge.runtime import BridgeRuntime

    assert RUNTIME_REGISTRY["bridge"] == "megatron.lite.runtime.backends.bridge"

    runtime = create_runtime(
        RuntimeConfig(
            backend="bridge",
            hf_path="/tmp/hf-model",
            backend_cfg=BridgeConfig(model_name="qwen3_5"),
        )
    )

    assert isinstance(runtime, BridgeRuntime)
    assert runtime.tier == "rl_best"


def test_mbridge_runtime_registered_and_lazy_constructible():
    from megatron.lite.runtime import RuntimeConfig, create_runtime
    from megatron.lite.runtime.backends import RUNTIME_REGISTRY
    from megatron.lite.runtime.backends.bridge.config import BridgeConfig
    from megatron.lite.runtime.backends.mbridge.runtime import MBridgeRuntime

    assert RUNTIME_REGISTRY["mbridge"] == "megatron.lite.runtime.backends.mbridge"

    runtime = create_runtime(
        RuntimeConfig(
            backend="mbridge",
            hf_path="/tmp/hf-model",
            backend_cfg=BridgeConfig(model_name="qwen3_5"),
        )
    )

    assert isinstance(runtime, MBridgeRuntime)
    assert runtime.tier == "rl_best"


def test_bridge_config_from_dict_accepts_nested_and_flat_parallel_fields():
    from megatron.lite.runtime.backends.bridge.config import BridgeConfig

    cfg = BridgeConfig.from_dict(
        {
            "model_name": "qwen3_5",
            "parallel": {"tp": 2, "pp": 1},
            "ep": 4,
            "optimizer": {"lr": 2e-4, "weight_decay": 0.2},
            "lr_scheduler": {"total_training_steps": 16},
            "override_transformer_config": {"attention_backend": "unfused"},
        }
    )

    assert cfg.model_name == "qwen3_5"
    assert cfg.parallel.tp == 2
    assert cfg.parallel.ep == 4
    assert cfg.optimizer.lr == 2e-4
    assert cfg.optimizer.weight_decay == 0.2
    assert cfg.optimizer.total_training_steps == 16
    assert cfg.override_transformer_config == {"attention_backend": "unfused"}


def test_bridge_config_from_dict_rejects_num_microbatches():
    from megatron.lite.runtime.backends.bridge.config import BridgeConfig

    with pytest.raises(ValueError, match="num_microbatches"):
        BridgeConfig.from_dict({"num_microbatches": 2})


def test_bridge_builds_mcore_ddp_config_object(monkeypatch):
    from megatron.lite.runtime.backends.bridge.config import BridgeConfig
    from megatron.lite.runtime.backends.bridge.runtime import _build_ddp_config

    class _FakeDDPConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    monkeypatch.setitem(
        sys.modules,
        "megatron.core.distributed",
        types.SimpleNamespace(DistributedDataParallelConfig=_FakeDDPConfig),
    )

    ddp_config = _build_ddp_config(
        BridgeConfig(
            override_ddp_config={
                "overlap_grad_reduce": True,
                "bucket_size": 1024,
            }
        )
    )

    assert isinstance(ddp_config, _FakeDDPConfig)
    assert ddp_config.use_distributed_optimizer is True
    assert ddp_config.grad_reduce_in_fp32 is True
    assert ddp_config.overlap_grad_reduce is True
    assert ddp_config.bucket_size == 1024


def test_bridge_registers_qwen35_moe_compat_aliases(monkeypatch):
    from megatron.lite.runtime.backends.bridge.runtime import _register_bridge_compat_aliases

    registered = []

    class _FakeDispatcher:
        _exact_types = {}

    model_bridge = types.SimpleNamespace(
        get_model_bridge=_FakeDispatcher(),
        register_bridge_implementation=lambda **kwargs: registered.append(kwargs),
    )
    qwen_bridge_mod = types.SimpleNamespace(Qwen3MoEBridge=object)
    gpt_mod = types.SimpleNamespace(GPTModel=object)

    monkeypatch.setitem(sys.modules, "megatron.bridge.models.conversion", types.SimpleNamespace(model_bridge=model_bridge))
    monkeypatch.setitem(sys.modules, "megatron.bridge.models.qwen.qwen3_moe_bridge", qwen_bridge_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.models.gpt.gpt_model", gpt_mod)

    _register_bridge_compat_aliases()

    assert [item["source"] for item in registered] == [
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
    ]
    assert all(issubclass(item["bridge_class"], object) for item in registered)
    assert all(item["bridge_class"] is not object for item in registered)
    assert all("provider_bridge" in item["bridge_class"].__dict__ for item in registered)
