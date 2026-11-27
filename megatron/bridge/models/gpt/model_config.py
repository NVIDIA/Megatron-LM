# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/src/megatron/bridge/models/gpt/model_config.py
# Source license: Apache-2.0.

"""Serializable Bridge extension of Megatron-LM's GPT model config."""

from dataclasses import dataclass
from typing import Any, ClassVar

from megatron.bridge.models.common.base import (
    MLAModelConfigMixin,
    ModelConfig,
    ModelConfigOverrideMixin,
)
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.activation_map import callable_to_str, str_to_callable
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.training.config.instantiate_utils import _resolve_target
from megatron.training.models.gpt import GPTModelConfig


@dataclass(kw_only=True)
class BridgeGPTModelConfig(ModelConfigOverrideMixin, GPTModelConfig, ModelConfig):
    """GPT model config with strict overrides and callable serialization.

    Models build through Megatron-LM's ``GPTModelBuilder``. Outer GPT build
    fields and nested transformer fields keep one declared owner. Flat
    assignment remains convenient, but unknown names fail instead of silently
    creating phantom configuration.

    Hugging Face source provenance (model id and revision) is not declared as a
    dedicated field here. It is recorded in the inherited serializable
    ``extra_checkpoint_metadata`` mapping so it round-trips through
    ``run_config.yaml`` without adding model-specific config fields.
    ``hf_model_text_only`` records that the source checkpoint was projected to
    its language model, so a restored config reopens the same projection.
    """

    transformer_config_class: ClassVar[type[TransformerConfig]] = TransformerConfig

    hf_model_text_only: bool = False
    """Whether the source HF checkpoint was projected to its language model."""

    def get_builder_cls(self) -> type:
        """Resolve the configured builder through Megatron-LM's target allowlist."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config with a symbolic activation function and a restorable pipeline layout."""
        data = super().as_dict()
        transformer_data = data.get("transformer")
        if not isinstance(transformer_data, dict):
            raise TypeError("Serialized GPT model config must contain a transformer mapping.")

        activation_func = self.transformer.activation_func
        if isinstance(activation_func, str):
            str_to_callable(activation_func)
            activation_name = activation_func
        else:
            activation_name = callable_to_str(activation_func)
        if activation_name is None:
            raise ValueError(
                f"Cannot serialize unregistered activation callable: {activation_func!r}."
            )

        transformer_data["activation_func"] = activation_name

        # finalize() parses the layout into an object that a saved run config cannot restore.
        # Persist the layout as given.
        layout = transformer_data.get("pipeline_model_parallel_layout")
        if isinstance(layout, PipelineParallelLayerLayout):
            transformer_data["pipeline_model_parallel_layout"] = layout.input_data
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgeGPTModelConfig":
        """Deserialize a config while restoring its activation callable."""
        restored_data = dict(data)
        transformer_data = restored_data.get("transformer")
        if isinstance(transformer_data, dict):
            restored_transformer = dict(transformer_data)
            activation_name = restored_transformer.get("activation_func")
            if isinstance(activation_name, str):
                restored_transformer["activation_func"] = str_to_callable(activation_name)
            restored_data["transformer"] = restored_transformer

        result = ModelConfig.from_dict(restored_data)
        if not isinstance(result, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(result).__name__}.")
        return result


@dataclass(kw_only=True)
class BridgeGPTMLAModelConfig(MLAModelConfigMixin, BridgeGPTModelConfig):
    """GPT model config with Multi-Latent Attention (MLA).

    The nested transformer config is an ``MLATransformerConfig``. It alone
    stores and serializes ``rotary_base`` and ``rotary_percent``, because MLA
    attention applies RoPE from them; on this config they are constructor
    arguments and views of the nested values (see ``MLAModelConfigMixin``).
    """


__all__ = ["BridgeGPTMLAModelConfig", "BridgeGPTModelConfig"]
