# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/config.py
# Source license: Apache-2.0.

import json
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Protocol, Type, TypeVar, Union, runtime_checkable

import yaml
from omegaconf import OmegaConf

from megatron.bridge.utils.instantiate_utils import InstantiationMode, instantiate
from megatron.bridge.utils.yaml_utils import safe_yaml_representers

# For TOML support
try:
    import toml

    HAS_TOML = True
except ImportError:
    HAS_TOML = False


T = TypeVar("T")
ConfigFormat = Literal["yaml", "json", "toml"]


@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol defining the configuration interface for model providers."""

    @classmethod
    def from_hf_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        trust_remote_code: bool = False,
        mode: InstantiationMode = InstantiationMode.LENIENT,
        **kwargs,
    ) -> T:
        """Load a pretrained model configuration from a directory or file."""
        ...

    def save_hf_pretrained(
        self,
        save_directory: Union[str, Path],
        config_format: ConfigFormat | None = None,
        config_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Save the model configuration to a directory."""
        ...


def from_hf_pretrained(
    cls: Type[T],
    pretrained_model_name_or_path: Union[str, Path],
    trust_remote_code: bool = False,
    mode: InstantiationMode = InstantiationMode.LENIENT,
    config_name: str = "config",
    **kwargs,
) -> T:
    """
    Load a pretrained model configuration from a directory or file.

    Args:
        cls: The class to instantiate
        pretrained_model_name_or_path: Path to a directory containing a config file,
                                      or direct path to a config file (yaml/json/toml)
        trust_remote_code: Whether to trust and execute code references (classes/functions)
                          found in the configuration. Required to be True if the config
                          contains any class or function references. Default: False
        mode: Instantiation mode (STRICT or LENIENT) for the instantiate function
        config_name: Base name of the config file (without extension)
        **kwargs: Additional keyword arguments to override loaded configuration

    Returns:
        Instance of the class with loaded configuration

    Example:
        ```python
        # Load from directory (looks for config.yaml, config.json, or config.toml)
        model = from_hf_pretrained(MyModel, "./saved_model/")

        # Load from specific file
        model = from_hf_pretrained(MyModel, "./saved_model/config.yaml")

        # With code references
        model = from_pretrained(MyModel, "./saved_model/", trust_remote_code=True)

        # Override configuration values
        model = from_pretrained(MyModel, "./saved_model/", temperature=0.8)
        ```
    """
    path = Path(pretrained_model_name_or_path)

    # Determine the config file path
    if path.is_dir():
        # Look for config files in order of preference
        config_file = None
        for ext in [".yaml", ".yml", ".json", ".toml"]:
            candidate = path / f"{config_name}{ext}"
            if candidate.exists():
                config_file = candidate
                break

        if config_file is None:
            raise FileNotFoundError(
                f"No configuration file found in {path}. "
                f"Expected {config_name}.yaml, {config_name}.json, or {config_name}.toml"
            )
    else:
        config_file = path

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}")

    # Load the configuration based on file extension
    file_ext = config_file.suffix.lower()

    if file_ext in [".yaml", ".yml"]:
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    elif file_ext == ".json":
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    elif file_ext == ".toml":
        if not HAS_TOML:
            raise ImportError(
                "TOML support requires the 'toml' package. Install it with: pip install toml"
            )
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = toml.load(f)
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. Supported formats: .yaml, .yml, .json, .toml"
        )

    # Convert to OmegaConf for compatibility with instantiate
    omega_conf = OmegaConf.create(config_dict)

    # Merge with kwargs
    if kwargs:
        override_conf = OmegaConf.create(kwargs)
        omega_conf = OmegaConf.merge(omega_conf, override_conf)

    user_config = OmegaConf.to_container(omega_conf, resolve=True)

    # Check for trust_remote_code requirement after merging caller-supplied
    # overrides so kwargs cannot inject executable targets after validation.
    if not trust_remote_code and _contains_code_references(user_config):
        raise ValueError(
            "This configuration contains class or function references. "
            "Loading it requires trust_remote_code=True to prevent arbitrary code execution."
        )

    # Add _target_ if not present
    if "_target_" not in omega_conf:
        omega_conf["_target_"] = f"{cls.__module__}.{cls.__qualname__}"

    # Convert back to container for instantiate
    final_config = OmegaConf.to_container(omega_conf, resolve=True)

    # Use instantiate to create the object
    return instantiate(final_config, mode=mode)


def save_hf_pretrained(
    obj: Any,
    save_directory: Union[str, Path],
    config_format: ConfigFormat = "json",
    config_name: str = "config",
    **kwargs,
) -> None:
    """
    Save the model configuration to a directory.

    Args:
        obj: The object to save
        save_directory: Directory where to save the configuration
        config_format: Format to save in ("yaml", "json", or "toml"). Default: "json"
        config_name: Name for the config file (without extension)
        **kwargs: Additional metadata to save alongside the configuration

    Example:
        ```python
        # Save as JSON (default)
        save_hf_pretrained(model, "./saved_model/")

        # Save as YAML
        save_hf_pretrained(model, "./saved_model/", config_format="yaml")

        # Save with custom name
        save_hf_pretrained(model, "./saved_model/", config_name="my_config")
        ```
    """
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    # Determine file extension
    format_to_ext = {"yaml": ".yaml", "yml": ".yaml", "json": ".json", "toml": ".toml"}

    config_format = config_format.lower()
    if config_format not in format_to_ext:
        raise ValueError(
            f"Unsupported format: {config_format}. Supported formats: {list(format_to_ext.keys())}"
        )

    if config_format == "toml" and not HAS_TOML:
        raise ImportError(
            "TOML support requires the 'toml' package. Install it with: pip install toml"
        )

    config_file = save_path / f"{config_name}{format_to_ext[config_format]}"

    # Get the configuration dictionary
    config_dict = _to_dict(obj)

    # Add any additional metadata
    if kwargs:
        config_dict.update(kwargs)

    # Save based on format
    if config_format in ["yaml", "yml"]:
        with safe_yaml_representers():
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
    elif config_format == "json":
        # First convert to YAML string to use the custom representers
        with safe_yaml_representers():
            yaml_str = yaml.safe_dump(config_dict, default_flow_style=False)
        # Then parse and save as JSON
        yaml_dict = yaml.safe_load(yaml_str)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(yaml_dict, f, indent=2, ensure_ascii=False)
    elif config_format == "toml":
        # First convert to YAML string to use the custom representers
        with safe_yaml_representers():
            yaml_str = yaml.safe_dump(config_dict, default_flow_style=False)
        # Then parse and save as TOML
        yaml_dict = yaml.safe_load(yaml_str)
        with open(config_file, "w", encoding="utf-8") as f:
            toml.dump(yaml_dict, f)

    print(f"Configuration saved to {config_file}")


def _to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert an object to a dictionary representation.

    Args:
        obj: The object to convert

    Returns:
        Dictionary representation of the object
    """
    # Check if this is a ConfigContainer (has to_dict method)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    # Otherwise, build dict from dataclass fields or attributes
    result = {}
    result["_target_"] = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"

    if is_dataclass(obj):
        # Handle dataclass
        for field in dataclass_fields(obj):
            if field.name.startswith("_"):
                continue
            value = getattr(obj, field.name)
            result[field.name] = _convert_value_to_dict(value)
    else:
        # Handle regular class
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                result[key] = _convert_value_to_dict(value)

    return result


def _convert_value_to_dict(value: Any) -> Any:
    """
    Recursively convert a value to a dictionary representation.

    Args:
        value: The value to convert

    Returns:
        The converted value
    """
    if hasattr(value, "_to_dict"):
        return value._to_dict()
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    elif is_dataclass(value) and not isinstance(value, type):
        # Handle regular dataclasses
        result = {"_target_": f"{value.__class__.__module__}.{value.__class__.__qualname__}"}
        for field in dataclass_fields(value):
            if not field.name.startswith("_"):
                result[field.name] = _convert_value_to_dict(getattr(value, field.name))
        return result
    elif isinstance(value, (list, tuple)):
        return [_convert_value_to_dict(item) for item in value]
    elif isinstance(value, dict):
        return {k: _convert_value_to_dict(v) for k, v in value.items()}
    else:
        return value


# Exact set of _target_ values considered safe (no code execution risk).
# Using an exact-match allowlist prevents bypass via dangerous callables
# in allowed modules (e.g. builtins.exec, builtins.eval, builtins.__import__).
_SAFE_TARGETS = frozenset(
    {
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
    }
)


def _contains_code_references(config_dict: Any) -> bool:
    """
    Check if a configuration dictionary contains code references.

    Args:
        config_dict: The configuration dictionary to check

    Returns:
        True if code references are found, False otherwise
    """
    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            # Check for _target_ that's not a safe built-in type
            if key == "_target_" and isinstance(value, str):
                if value not in _SAFE_TARGETS:
                    return True
            # Check for _call_ = False which indicates a code reference
            if key == "_call_" and value is False:
                return True
            # Recursively check nested structures
            if _contains_code_references(value):
                return True
    elif isinstance(config_dict, (list, tuple)):
        for item in config_dict:
            if _contains_code_references(item):
                return True

    return False
