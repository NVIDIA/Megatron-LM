# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.elastification.memory_config."""

from argparse import Namespace

import pytest
import yaml

from megatron.elastification.memory_config import MemoryConfig, load_memory_config


@pytest.fixture
def profiles_file(tmp_path):
    """Write a minimal memory-profiles YAML and return its path."""
    data = {
        "presets": {
            "bf16": {
                "params": 2,
                "kv_cache": 2,
                "ssm_cache": 2,
                "max_buffer": 2,
                "param_budget_target": "active",
            },
            "fp8_kv": {
                "params": 2,
                "kv_cache": 1,
                "ssm_cache": 2,
                "max_buffer": 2,
                "param_budget_target": "active",
            },
            "total_target": {
                "params": 2,
                "kv_cache": 2,
                "ssm_cache": 2,
                "max_buffer": 2,
                "param_budget_target": "total",
            },
        }
    }
    path = tmp_path / "memory_profiles.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _make_args(profile="bf16", profiles_path=None, **overrides):
    defaults = dict(
        memory_profile=profile,
        memory_profile_path=profiles_path,
        bpe_params=None,
        bpe_kv_cache=None,
        bpe_ssm_cache=None,
        bpe_max_buffer=None,
        param_budget_target=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestMemoryConfigDataclass:
    def test_default_values(self):
        cfg = MemoryConfig()
        assert cfg.bpe_params == 2.0
        assert cfg.bpe_kv_cache == 2.0
        assert cfg.bpe_ssm_cache == 2.0
        assert cfg.bpe_max_buffer == 2.0
        assert cfg.param_budget_target == "active"

    def test_invalid_param_budget_target_rejected(self):
        with pytest.raises(ValueError, match="param_budget_target"):
            MemoryConfig(param_budget_target="bogus")

    def test_valid_param_budget_target_accepted(self):
        MemoryConfig(param_budget_target="active")
        MemoryConfig(param_budget_target="total")


class TestLoadMemoryConfig:
    def test_preset_applied(self, profiles_file):
        args = _make_args(profile="fp8_kv", profiles_path=profiles_file)
        cfg = load_memory_config(args)
        assert cfg.bpe_params == 2.0
        assert cfg.bpe_kv_cache == 1.0  # FP8
        assert cfg.bpe_ssm_cache == 2.0
        assert cfg.bpe_max_buffer == 2.0

    def test_preset_param_budget_target(self, profiles_file):
        args = _make_args(profile="total_target", profiles_path=profiles_file)
        cfg = load_memory_config(args)
        assert cfg.param_budget_target == "total"

    def test_cli_override_takes_priority_over_preset(self, profiles_file):
        args = _make_args(
            profile="bf16", profiles_path=profiles_file, bpe_kv_cache=0.5625  # override
        )
        cfg = load_memory_config(args)
        assert cfg.bpe_kv_cache == 0.5625  # override wins
        assert cfg.bpe_params == 2.0  # preset preserved

    def test_param_budget_target_override(self, profiles_file):
        args = _make_args(profile="bf16", profiles_path=profiles_file, param_budget_target="total")
        cfg = load_memory_config(args)
        assert cfg.param_budget_target == "total"

    def test_unknown_profile_raises(self, profiles_file):
        args = _make_args(profile="nonexistent", profiles_path=profiles_file)
        with pytest.raises(ValueError, match="not found"):
            load_memory_config(args)

    def test_missing_profile_file_raises(self, tmp_path):
        args = _make_args(profile="bf16", profiles_path=str(tmp_path / "missing.yaml"))
        with pytest.raises(FileNotFoundError):
            load_memory_config(args)

    def test_none_profile_name_defaults_to_bf16(self, profiles_file):
        args = _make_args(profile=None, profiles_path=profiles_file)
        cfg = load_memory_config(args)
        # bf16 defaults in the fixture.
        assert cfg.bpe_params == 2.0
        assert cfg.bpe_kv_cache == 2.0

    def test_default_profiles_path_loads_bundled_yaml(self):
        # When profiles_path is None, the loader falls back to the bundled
        # megatron/elastification/memory_profiles.yaml.
        args = _make_args(profile="bf16", profiles_path=None)
        cfg = load_memory_config(args)
        assert cfg.bpe_params == 2.0
        assert cfg.bpe_kv_cache == 2.0
