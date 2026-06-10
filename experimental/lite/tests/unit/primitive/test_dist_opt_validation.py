from __future__ import annotations

import pytest

from megatron.lite.primitive.optimizers.megatron_wrap import validate_mc_config, validate_mc_session
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.config import ParallelConfig


def _engine_cfg(*, model_name: str, pp: int = 1, vpp: int = 1) -> MegatronLiteConfig:
    return MegatronLiteConfig(model_name=model_name, parallel=ParallelConfig(pp=pp, vpp=vpp))


def test_dist_opt_validation_accepts_model_agnostic_config():
    validate_mc_config(_engine_cfg(model_name="synthetic_custom_model", pp=1, vpp=1))


def test_dist_opt_validation_keeps_vpp_parallel_constraint():
    with pytest.raises(ValueError, match="dist_opt requires pp>1 when vpp>1"):
        validate_mc_config(_engine_cfg(model_name="synthetic_custom_model", pp=1, vpp=2))


def test_validate_mc_session_alias_matches_config_validator():
    assert validate_mc_session is validate_mc_config
    validate_mc_session(_engine_cfg(model_name="another_synthetic_model", pp=2, vpp=2))
