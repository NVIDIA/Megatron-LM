# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``to_dict`` / ``to_yaml`` with section overrides: the detached dict a checkpoint save persists."""

import copy

import pytest
import yaml

from megatron.training.config import (
    CheckpointConfig,
    InferenceConfigContainer,
    InferenceSetupConfig,
)


def _container():
    return InferenceConfigContainer(
        model=None,
        checkpoint=CheckpointConfig(save="/run/save", ckpt_format="torch_dist"),
        inference=InferenceSetupConfig(),
    )


def test_no_override_is_the_plain_serialization():
    cfg = _container()
    assert cfg.to_dict() == cfg.to_dict(**{})
    assert cfg.to_dict()["checkpoint"]["save"] == "/run/save"
    assert cfg.to_dict()["_target_"].endswith(".InferenceConfigContainer")


def test_an_override_replaces_the_fields_of_its_section_only():
    cfg = _container()
    plain = cfg.to_dict()
    overrides = {"save": "/converted/torch", "ckpt_format": "torch", "verify_integrity": False}
    kept = copy.deepcopy(overrides)

    result = cfg.to_dict(checkpoint=overrides)

    assert result["checkpoint"]["save"] == "/converted/torch"
    assert result["checkpoint"]["ckpt_format"] == "torch"
    assert result["checkpoint"]["_target_"] == plain["checkpoint"]["_target_"]
    assert {k: v for k, v in result["checkpoint"].items() if k not in overrides} == {
        k: v for k, v in plain["checkpoint"].items() if k not in overrides
    }
    assert {k: v for k, v in result.items() if k != "checkpoint"} == {
        k: v for k, v in plain.items() if k != "checkpoint"
    }
    # Neither the live config nor the caller's mapping is written.
    assert cfg.checkpoint.save == "/run/save" and cfg.checkpoint.ckpt_format == "torch_dist"
    assert overrides == kept
    assert cfg.to_dict() == plain


def test_to_yaml_carries_the_overrides(tmp_path):
    cfg = _container()
    path = tmp_path / "run_config.yaml"
    cfg.to_yaml(str(path), checkpoint={"save": "/converted/torch", "ckpt_format": "torch"})
    with open(path) as f:
        loaded = yaml.safe_load(f)
    assert loaded["checkpoint"]["save"] == "/converted/torch"
    assert loaded["checkpoint"]["ckpt_format"] == "torch"
    assert loaded["checkpoint"]["_target_"] == cfg.to_dict()["checkpoint"]["_target_"]
    assert cfg.checkpoint.save == "/run/save"


def test_unknown_sections_fields_and_non_mappings_are_rejected():
    cfg = _container()
    with pytest.raises(ValueError, match="no section named"):
        cfg.to_dict(optimizer={"lr": 1.0})
    with pytest.raises(ValueError, match="no field named"):
        cfg.to_dict(checkpoint={"not_a_field": 1})
    with pytest.raises(TypeError, match="must map field names"):
        cfg.to_dict(checkpoint="torch")
    with pytest.raises(TypeError, match="not a dataclass"):
        cfg.to_dict(model={"vocab_size": 1})
