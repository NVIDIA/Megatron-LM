# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dependency-free contract for the two QAT owners in MLite's verl config."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from megatron.lite.primitive.quantization.qat import (
    QATSpec,
    _DEFAULT_IGNORE_PATTERNS,
    normalize_qat_spec,
)
from omegaconf import OmegaConf

pytestmark = pytest.mark.optional


def _compose_engine(*overrides: str) -> dict:
    config_root = Path(__file__).parents[3] / "examples" / "verl"
    assert (config_root / "verl_mlite" / "config" / "engine" / "mlite.yaml").is_file()
    with initialize_config_module(
        config_module="verl_mlite.config",
        version_base=None,
    ):
        config = compose(
            config_name=None,
            overrides=["+engine@actor_rollout_ref.actor.engine=mlite", *overrides],
        )
    return OmegaConf.to_container(
        config.actor_rollout_ref.actor.engine,
        resolve=True,
    )


def test_default_yaml_keeps_export_and_training_qat_disabled() -> None:
    engine = _compose_engine()

    assert engine["qat"] == {}
    assert engine["impl_cfg"]["qat"] == {
        "enabled": False,
        "format": "mxfp4",
    }

    spec = normalize_qat_spec(engine["impl_cfg"]["qat"])
    assert spec == QATSpec(enabled=False, format="mxfp4")
    assert spec.group_size == 32
    assert spec.ignore_patterns == _DEFAULT_IGNORE_PATTERNS
    assert not spec.targets_module("layers.0.mlp.router.gate")
    assert spec.targets_module("layers.0.mlp.gate_up")


def test_colocated_ref_follows_actor_runtime_plugins() -> None:
    config_root = Path(__file__).parents[3] / "examples" / "verl"
    ref = OmegaConf.load(
        config_root / "verl_mlite" / "config" / "ref" / "mlite_ref.yaml"
    )
    actor_plugins = {
        "dynamic_context_parallel": {
            "enabled": True,
            "max_seqlen_per_dp_cp_rank": 4096,
        }
    }
    root = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"engine": {"impl_cfg": {"runtime_plugins": actor_plugins}}},
                "ref": {"engine": ref.engine},
            }
        }
    )

    resolved = OmegaConf.to_container(
        root.actor_rollout_ref.ref.engine.impl_cfg, resolve=True
    )

    assert resolved["runtime_plugins"] == actor_plugins
