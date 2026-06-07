"""Unit tests for the Megatron-Bridge runtime backend surface."""

from __future__ import annotations

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
