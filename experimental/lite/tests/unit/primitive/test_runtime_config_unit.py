# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest

from megatron.lite.runtime import RuntimeConfig, create_runtime
from megatron.lite.runtime.backends.mlite.config import DebugConfig, MegatronLiteConfig
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

pytestmark = pytest.mark.mlite


def test_mlite_config_defaults_are_stable():
    cfg = MegatronLiteConfig(model_name="qwen3_moe")

    assert cfg.model_name == "qwen3_moe"
    assert cfg.impl == "lite"
    assert cfg.parallel.tp == 1
    assert cfg.parallel.ep == 1
    assert cfg.parallel.pp == 1
    assert cfg.parallel.cp == 1
    assert isinstance(cfg.optimizer, OptimizerConfig)
    assert isinstance(cfg.debug, DebugConfig)


def test_mlite_config_from_dict_preserves_parallel_optimizer_and_impl_cfg():
    cfg = MegatronLiteConfig.from_dict(
        "/models/qwen",
        {
            "model_name": "qwen3_moe",
            "impl": "lite",
            "tp": 2,
            "ep": 4,
            "pp": 2,
            "cp": 2,
            "optimizer": {
                "lr": 1.0e-4,
                "weight_decay": 0.1,
                "adam_beta1": 0.9,
                "offload_fraction": 1.0,
            },
            "impl_cfg": {"attn_impl": "mcore", "moe_impl": "ml"},
            "use_thd": True,
            "precision_aware_opt": True,
        },
    )

    assert cfg.hf_path == "/models/qwen"
    assert cfg.parallel == ParallelConfig(tp=2, etp=None, ep=4, pp=2, vpp=1, cp=2)
    assert cfg.optimizer.lr == 1.0e-4
    assert cfg.optimizer.weight_decay == 0.1
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.offload_fraction == 1.0
    assert cfg.impl_cfg["attn_impl"] == "mcore"
    assert cfg.impl_cfg["moe_impl"] == "ml"
    assert cfg.impl_cfg["use_thd"] is True
    assert cfg.impl_cfg["precision_aware_opt"] is True


def test_mlite_config_rejects_num_microbatches_in_backend_config():
    with pytest.raises(ValueError, match="num_microbatches"):
        MegatronLiteConfig.from_dict(
            "/models/qwen", {"model_name": "qwen3_moe", "num_microbatches": 2}
        )


def test_create_runtime_uses_mlite_backend_registry():
    runtime = create_runtime(
        RuntimeConfig(
            backend="mlite",
            hf_path="/models/qwen",
            backend_cfg={"model_name": "qwen3_moe", "load_hf_weights": False},
        )
    )

    assert type(runtime).__name__ == "MegatronLiteRuntime"
    assert runtime.tier == "rl_best"
