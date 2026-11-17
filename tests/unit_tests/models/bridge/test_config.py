# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/tests/unit_tests/models/test_config.py
# Source license: Apache-2.0.

"""Tests for megatron.bridge.models.config module."""

import json
from dataclasses import dataclass

import pytest
import yaml

from megatron.bridge.models.config import (
    ConfigProtocol,
    _contains_code_references,
    from_hf_pretrained,
    save_hf_pretrained,
)


@dataclass
class MockConfig:
    """Mock configuration class for testing."""

    model_type: str = "mock"
    hidden_size: int = 768
    num_layers: int = 12

    @classmethod
    def from_hf_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Mock from_hf_pretrained implementation."""
        return from_hf_pretrained(cls, pretrained_model_name_or_path, **kwargs)

    def save_hf_pretrained(self, save_directory, **kwargs):
        """Mock save_hf_pretrained implementation."""
        save_hf_pretrained(self, save_directory, **kwargs)


class TestConfigProtocol:
    """Tests for ConfigProtocol."""

    def test_protocol_implementation(self):
        """Test that MockConfig implements ConfigProtocol."""
        assert isinstance(MockConfig, ConfigProtocol)
        assert hasattr(MockConfig, "from_hf_pretrained")
        assert hasattr(MockConfig, "save_hf_pretrained")


class TestFromPretrained:
    """Tests for from_hf_pretrained function."""

    def test_load_from_yaml_file(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_data = {"model_type": "test", "hidden_size": 1024, "num_layers": 24}

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = from_hf_pretrained(MockConfig, str(config_file))
        assert config.model_type == "test"
        assert config.hidden_size == 1024
        assert config.num_layers == 24

    def test_load_from_json_file(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_data = {"model_type": "test_json", "hidden_size": 512, "num_layers": 6}

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = from_hf_pretrained(MockConfig, str(config_file))
        assert config.model_type == "test_json"
        assert config.hidden_size == 512
        assert config.num_layers == 6

    def test_load_from_directory(self, tmp_path):
        """Test loading configuration from directory containing config file."""
        config_data = {"model_type": "dir_test", "hidden_size": 256, "num_layers": 4}

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = from_hf_pretrained(MockConfig, str(tmp_path))
        assert config.model_type == "dir_test"
        assert config.hidden_size == 256
        assert config.num_layers == 4

    def test_kwargs_override(self, tmp_path):
        """Test that kwargs override loaded configuration."""
        config_data = {"model_type": "original", "hidden_size": 768, "num_layers": 12}

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = from_hf_pretrained(MockConfig, str(config_file), hidden_size=2048, num_layers=48)
        assert config.model_type == "original"  # Not overridden
        assert config.hidden_size == 2048  # Overridden
        assert config.num_layers == 48  # Overridden

    def test_custom_config_name(self, tmp_path):
        """Test loading with custom config name."""
        config_data = {"model_type": "custom", "hidden_size": 1024}

        config_file = tmp_path / "my_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = from_hf_pretrained(MockConfig, str(tmp_path), config_name="my_config")
        assert config.model_type == "custom"
        assert config.hidden_size == 1024

    def test_file_not_found(self, tmp_path):
        """Test error when config file not found."""
        with pytest.raises(FileNotFoundError):
            from_hf_pretrained(MockConfig, str(tmp_path / "nonexistent"))


class TestSavePretrained:
    """Tests for save_hf_pretrained function."""

    def test_save_to_yaml(self, tmp_path):
        """Test saving configuration to YAML format."""
        config = MockConfig(model_type="save_test", hidden_size=2048, num_layers=32)

        save_hf_pretrained(config, str(tmp_path), config_format="yaml")

        # Verify saved file
        saved_file = tmp_path / "config.yaml"
        assert saved_file.exists()

        with open(saved_file) as f:
            loaded_data = yaml.safe_load(f)

        assert loaded_data["model_type"] == "save_test"
        assert loaded_data["hidden_size"] == 2048
        assert loaded_data["num_layers"] == 32

    def test_save_to_json(self, tmp_path):
        """Test saving configuration to JSON format."""
        config = MockConfig(model_type="json_save", hidden_size=1024)

        save_hf_pretrained(config, str(tmp_path), config_format="json")

        saved_file = tmp_path / "config.json"
        assert saved_file.exists()

        with open(saved_file) as f:
            loaded_data = json.load(f)

        assert loaded_data["model_type"] == "json_save"
        assert loaded_data["hidden_size"] == 1024

    def test_save_with_custom_name(self, tmp_path):
        """Test saving with custom config name."""
        config = MockConfig()

        save_hf_pretrained(config, str(tmp_path), config_name="my_model_config")

        saved_file = tmp_path / "my_model_config.json"
        assert saved_file.exists()

    def test_create_directory(self, tmp_path):
        """Test that save_hf_pretrained creates directory if it doesn't exist."""
        config = MockConfig()
        save_dir = tmp_path / "new_dir"

        save_hf_pretrained(config, str(save_dir))

        assert save_dir.exists()
        assert (save_dir / "config.json").exists()


class TestContainsCodeReferences:
    """Tests for _contains_code_references security guard."""

    @pytest.mark.parametrize(
        "target",
        [
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "tuple",
            "builtins.str",
            "builtins.int",
            "builtins.float",
            "builtins.bool",
            "builtins.list",
            "builtins.dict",
            "builtins.tuple",
        ],
    )
    def test_safe_targets_allowed(self, target):
        """Safe built-in type targets should not be flagged."""
        assert _contains_code_references({"_target_": target}) is False

    @pytest.mark.parametrize(
        "target",
        [
            "builtins.exec",
            "builtins.eval",
            "builtins.__import__",
            "builtins.compile",
            "builtins.getattr",
            "builtins.setattr",
            "builtins.delattr",
            "builtins.globals",
            "builtins.open",
        ],
    )
    def test_dangerous_builtins_blocked(self, target):
        """Dangerous builtins must be flagged even though they share the builtins. prefix."""
        assert _contains_code_references({"_target_": target}) is True

    def test_arbitrary_module_blocked(self):
        """Arbitrary module paths must be flagged."""
        assert _contains_code_references({"_target_": "os.system"}) is True
        assert _contains_code_references({"_target_": "subprocess.call"}) is True

    def test_nested_dangerous_target_blocked(self):
        """Dangerous targets nested inside config dicts must be detected."""
        config = {"model": {"_target_": "builtins.exec", "_args_": ["print('pwned')"]}}
        assert _contains_code_references(config) is True

    def test_dangerous_target_in_list_blocked(self):
        """Dangerous targets inside lists must be detected."""
        config = [{"_target_": "builtins.eval", "_args_": ["1+1"]}]
        assert _contains_code_references(config) is True

    def test_call_false_flagged(self):
        """_call_=False is a code reference indicator and must be flagged."""
        assert _contains_code_references({"_call_": False}) is True

    def test_no_target_is_safe(self):
        """Config with no _target_ or _call_ should not be flagged."""
        assert _contains_code_references({"hidden_size": 768, "num_layers": 12}) is False

    def test_from_hf_pretrained_blocks_builtins_exec(self, tmp_path):
        """Loading a config with builtins.exec must raise even without trust_remote_code."""
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump({"_target_": "builtins.exec", "_args_": ["print('pwned')"]}, f)

        with pytest.raises(ValueError, match="trust_remote_code"):
            from_hf_pretrained(MockConfig, str(tmp_path), trust_remote_code=False)

    def test_from_hf_pretrained_blocks_code_references_in_kwargs(self, tmp_path):
        """Caller overrides must not inject code references after the trust check."""
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump({"model_type": "safe", "hidden_size": 128}, f)

        with pytest.raises(ValueError, match="trust_remote_code"):
            from_hf_pretrained(
                MockConfig,
                str(tmp_path),
                trust_remote_code=False,
                _target_="torch.ctypes.CDLL",
                _args_=["/proc/self/cwd/payload.so"],
            )
