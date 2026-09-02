# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest

from megatron.lite.primitive.optimizers.megatron_wrap import (
    build_dist_opt_optimizer_config,
    validate_dist_opt_config,
    validate_dist_opt_session,
)
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.config import ParallelConfig


def _engine_cfg(*, model_name: str, pp: int = 1, vpp: int = 1) -> MegatronLiteConfig:
    return MegatronLiteConfig(model_name=model_name, parallel=ParallelConfig(pp=pp, vpp=vpp))


def test_dist_opt_validation_accepts_model_agnostic_config():
    validate_dist_opt_config(_engine_cfg(model_name="synthetic_custom_model", pp=1, vpp=1))


def test_dist_opt_validation_keeps_vpp_parallel_constraint():
    with pytest.raises(ValueError, match="dist_opt requires pp>1 when vpp>1"):
        validate_dist_opt_config(_engine_cfg(model_name="synthetic_custom_model", pp=1, vpp=2))


def test_validate_dist_opt_session_alias_matches_config_validator():
    assert validate_dist_opt_session is validate_dist_opt_config
    validate_dist_opt_session(_engine_cfg(model_name="another_synthetic_model", pp=2, vpp=2))


def test_dist_opt_offload_uses_shard_aware_state_offloader():
    config = build_dist_opt_optimizer_config(
        SimpleNamespace(
            optimizer="adam",
            lr=1.0e-6,
            min_lr=0.0,
            weight_decay=0.1,
            clip_grad=1.0,
            offload_fraction=1.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-8,
            use_precision_aware_optimizer=True,
            decoupled_weight_decay=True,
        )
    )

    assert config.chunked_optimizer_state_offload is True
    assert config.optimizer_state_offload_fraction == 1.0
    assert config.optimizer_state_offload_chunk_size_mb == 256
    assert config.optimizer_cpu_offload is False
    assert config.optimizer_offload_fraction == 0.0
