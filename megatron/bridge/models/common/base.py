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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/common/base.py
# Source license: Apache-2.0.
# Deep-copy handling also follows https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/7617cdaf44628873aa5f4f980a5cae88c26e664c/src/megatron/bridge/models/common/base.py (Apache-2.0).

import copy
from dataclasses import InitVar, dataclass
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, ClassVar

from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.training.config.instantiate_utils import _resolve_target, instantiate
from megatron.training.models.base import ModelConfig as _MegatronModelConfig


def _init_var_names(config_cls: type) -> list[str]:
    """Return the names of a dataclass's ``InitVar`` arguments in the order its constructor passes them."""
    return [
        name
        for name, f in getattr(config_cls, "__dataclass_fields__", {}).items()
        if isinstance(f.type, InitVar)
    ]


def deserialize_model_config(data: dict[str, Any]) -> _MegatronModelConfig:
    """Deserialize a builder config, including persisted enum and dtype values."""

    def _restore_value(value: Any, full_key: str) -> Any:
        if isinstance(value, list):
            return [_restore_value(item, f"{full_key}.{index}") for index, item in enumerate(value)]
        if not isinstance(value, dict):
            return value
        if "_target_" not in value:
            return {key: _restore_value(item, f"{full_key}.{key}") for key, item in value.items()}
        return _from_dict(value, full_key)

    def _from_dict(subdata: dict[str, Any], full_key: str) -> Any:
        target = subdata.get("_target_")
        if target is None:
            raise ValueError("Cannot deserialize: missing '_target_' field")
        if not isinstance(target, str):
            raise ValueError(
                f"Cannot deserialize: '_target_' must be a string, got {type(target).__name__}"
            )

        config_cls = _resolve_target(target, full_key=full_key, check_callable=False)
        if not isinstance(config_cls, type) or not is_dataclass(config_cls):
            return instantiate(subdata)

        # Constructor-only (InitVar) arguments are accepted too, so the constructor can check them.
        valid_fields = {f.name for f in dataclass_fields(config_cls) if f.init} | set(
            _init_var_names(config_cls)
        )
        filtered_data = {
            key: _restore_value(value, f"{full_key}.{key}")
            for key, value in subdata.items()
            if key in valid_fields and not key.startswith("_")
        }
        return config_cls(**filtered_data)

    builder = data.get("_builder_")
    if not isinstance(builder, str):
        raise ValueError("Cannot deserialize: missing '_builder_' field")
    _resolve_target(builder, full_key="_builder_")

    result = _from_dict(data, full_key="_target_")
    if not isinstance(result, _MegatronModelConfig):
        raise ValueError(
            f"Cannot deserialize: outer target produced {type(result).__name__}, not ModelConfig"
        )
    result.builder = builder
    return result


class ModelConfigOverrideMixin:
    """Route flat overrides to their declared owner and reject unknown fields.

    Builder-backed model configs store Megatron-Core transformer settings in a
    nested ``transformer`` dataclass while exposing flat assignment for recipe
    compatibility. This mixin keeps that convenience without allowing typos to
    create phantom configuration attributes. Subclasses declare the nested
    config type through ``transformer_config_class``.
    """

    transformer_config_class: ClassVar[type[TransformerConfig]]

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        """Copy the outer config instead of forwarding to the nested transformer's copier."""
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for name, value in vars(self).items():
            object.__setattr__(result, name, copy.deepcopy(value, memo))
        return result

    def __setattr__(self, name: str, value: Any, /) -> None:
        """Assign a declared outer or nested field and reject phantom fields."""
        try:
            transformer = object.__getattribute__(self, "transformer")
        except AttributeError:
            object.__setattr__(self, name, value)
            return

        model_fields = getattr(type(self), "__dataclass_fields__", {})
        transformer_fields = getattr(type(transformer), "__dataclass_fields__", {})
        descriptor = getattr(type(self), name, None)

        if name == "transformer" or name in model_fields:
            object.__setattr__(self, name, value)
        elif name in transformer_fields:
            setattr(transformer, name, value)
        elif name == "builder" or name.startswith("_") or hasattr(descriptor, "__set__"):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Neither {type(self).__name__} nor {type(transformer).__name__} declares a field named {name!r}."
            )


class _TransformerOwnedField:
    """Outer name of a setting that the nested transformer config stores.

    As the default of an ``InitVar``, it keeps the setting out of the outer
    config's fields, so the outer config neither stores nor serializes it. A
    value given to the constructor reaches ``__post_init__``; reads and later
    assignments use the transformer config.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            # Class access supplies the InitVar default: the constructor was not given a value.
            return None
        return getattr(instance.transformer, self._name)

    def __set__(self, instance: Any, value: Any) -> None:
        setattr(instance.transformer, self._name, value)


@dataclass(kw_only=True)
class MLAModelConfigMixin:
    """Nest an MLA transformer config, which owns the rotary settings that MLA attention applies.

    Megatron Core's MLA transformer config declares ``rotary_base`` and
    ``rotary_percent``, as do the GPT and Hybrid model configs. MLA attention
    reads the nested values, so only the transformer config stores and
    serializes them. On this config they are constructor arguments and views of
    the nested values. ``__post_init__`` checks a constructor value against the
    transformer config's own value before copying it there: they must match
    unless the transformer config kept its default. List this mixin before the
    model config class it extends.
    """

    transformer_config_class: ClassVar[type[TransformerConfig]] = MLATransformerConfig

    rotary_base: InitVar[float | None] = _TransformerOwnedField()
    rotary_percent: InitVar[float | None] = _TransformerOwnedField()

    def __post_init__(self, *constructor_values: Any) -> None:
        # Dataclasses pass InitVars in field order, which the extended model config sets.
        supplied = {
            name: value
            for name, value in zip(_init_var_names(type(self)), constructor_values, strict=True)
            if value is not None
        }
        defaults = {f.name: f.default for f in dataclass_fields(self.transformer)}
        for name, value in supplied.items():
            nested = getattr(self.transformer, name)
            if nested != value and nested != defaults[name]:
                raise ValueError(
                    f"{name}={value!r} conflicts with transformer.{name}={nested!r}. "
                    f"{type(self.transformer).__name__} stores {name}; give it one value."
                )
        for name, value in supplied.items():
            setattr(self.transformer, name, value)


class ModelConfig(_MegatronModelConfig):
    """Bridge compatibility wrapper for Megatron-LM model configs."""

    def get_builder_cls(self) -> type:
        """Get the appropriate builder type for this config."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _MegatronModelConfig:
        """Deserialize config from dictionary with Bridge target validation."""
        return deserialize_model_config(data)
