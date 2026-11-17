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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/hybrid/model_config.py
# Source license: Apache-2.0.

"""Serializable Bridge extension of Megatron-LM's Hybrid model config."""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from megatron.bridge.models.common.base import (
    MLAModelConfigMixin,
    ModelConfig,
    ModelConfigOverrideMixin,
)
from megatron.bridge.models.hybrid.hybrid_builder import DEFAULT_MAMBA_CHUNK_SIZE, HybridModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.activation_map import callable_to_str, str_to_callable
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.instantiate_utils import _resolve_target
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_hybrid_total_layer_count,
    parse_hybrid_pattern,
)
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BridgeHybridModelConfig(ModelConfigOverrideMixin, HybridModelConfig, ModelConfig):
    """Hybrid model config with strict overrides and callable serialization.

    This is the ``HybridModel`` counterpart of ``BridgeGPTModelConfig``. Outer
    Hybrid build fields and nested transformer fields keep one declared owner.
    Flat assignment remains convenient, but unknown names fail instead of
    silently creating phantom configuration. Subclasses can select a
    specialized transformer config, such as an MLA config, through
    ``transformer_config_class``.

    It keeps the construction contract of ``HybridModelProvider``.
    ``validate()`` checks the settings without changing them, and
    ``finalize()`` runs it before appending ``mtp_hybrid_override_pattern`` to
    ``hybrid_layer_pattern`` once per MTP depth and deriving ``num_layers`` from
    the pattern. :class:`~megatron.bridge.models.hybrid.BridgeHybridModelBuilder`
    resolves stack-spec factories, the local attention backend and
    ``mamba_chunk_size``.

    Hugging Face source provenance is recorded in the inherited
    ``extra_checkpoint_metadata`` mapping, as for ``BridgeGPTModelConfig``.
    ``hf_model_text_only`` records that the source checkpoint was projected to
    its language model, so a restored config reopens the same projection.
    """

    builder: ClassVar[str] = "megatron.bridge.models.hybrid.BridgeHybridModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = TransformerConfig

    hybrid_stack_spec: (
        ModuleSpec
        | Callable[[], ModuleSpec]
        | Callable[["BridgeHybridModelConfig"], ModuleSpec]
        | None
    ) = None
    """Stack spec, or a factory that takes no arguments or this config. ``None`` selects Bridge's default."""

    mamba_chunk_size: int = DEFAULT_MAMBA_CHUNK_SIZE
    """Chunk size of the Mamba scan."""

    hf_model_text_only: bool = False
    """Whether the source HF checkpoint was projected to its language model."""

    def validate(self) -> None:
        """Check the Hybrid settings without changing the config.

        Raises:
            ValueError: If ``mamba_chunk_size`` is not positive or both pattern fields are set.
        """
        if self.mamba_chunk_size < 1:
            raise ValueError("mamba_chunk_size must be at least 1.")
        if self.hybrid_override_pattern is not None and self.hybrid_layer_pattern is not None:
            raise ValueError(
                "hybrid_override_pattern and hybrid_layer_pattern cannot both be specified. "
                "hybrid_override_pattern is deprecated; use hybrid_layer_pattern instead."
            )

    def finalize(self) -> None:
        """Validate, combine the hybrid layer patterns and derive ``num_layers``, then finalize the transformer config.

        Raises:
            ValueError: If :meth:`validate` rejects the config or ``num_layers`` disagrees with
                ``hybrid_layer_pattern``.
        """
        self.validate()

        if self.hybrid_override_pattern is not None:
            if get_rank_safe() == 0:
                warnings.warn(
                    "hybrid_override_pattern is deprecated. Use hybrid_layer_pattern instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.hybrid_layer_pattern = self.hybrid_override_pattern
            self.hybrid_override_pattern = None

        transformer = self.transformer
        mtp_pattern = transformer.mtp_hybrid_override_pattern
        if self.hybrid_layer_pattern is not None and mtp_pattern:
            separator = Symbols.MTP_SEPARATOR
            main_pattern = self.hybrid_layer_pattern.split(separator)[0]
            # A repeated MTP layer always exists; mtp_num_layers then counts its forward repetitions.
            num_depths = transformer.mtp_num_layers or 0
            if transformer.mtp_use_repeated_layer:
                num_depths = max(1, num_depths)
            self.hybrid_layer_pattern = (
                main_pattern + separator + separator.join([mtp_pattern] * num_depths)
            )

            parsed = parse_hybrid_pattern(self.hybrid_layer_pattern)
            if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
                if transformer.mtp_num_layers is None:
                    transformer.mtp_num_layers = parsed.mtp_num_depths
                elif (
                    not transformer.mtp_use_repeated_layer
                    and transformer.mtp_num_layers != parsed.mtp_num_depths
                ):
                    logger.warning(
                        "mtp_num_layers (%s) conflicts with MTP depth count (%s) in pattern '%s'. "
                        "Using the inferred value (%s).",
                        transformer.mtp_num_layers,
                        parsed.mtp_num_depths,
                        self.hybrid_layer_pattern,
                        parsed.mtp_num_depths,
                    )
                    transformer.mtp_num_layers = parsed.mtp_num_depths

        if self.hybrid_layer_pattern is not None:
            num_layers = get_hybrid_total_layer_count(self.hybrid_layer_pattern)
            if transformer.num_layers and transformer.num_layers != num_layers:
                raise ValueError(
                    f"num_layers ({transformer.num_layers}) does not match the {num_layers} layers in "
                    f"hybrid_layer_pattern ({self.hybrid_layer_pattern!r})."
                )
            transformer.num_layers = num_layers

        super().finalize()

    def get_builder_cls(self) -> type:
        """Resolve the configured builder through Bridge's target allowlist."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config with a symbolic activation function and a restorable pipeline layout."""
        data = super().as_dict()
        transformer_data = data.get("transformer")
        if not isinstance(transformer_data, dict):
            raise TypeError("Serialized Hybrid model config must contain a transformer mapping.")

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
        # Persist the layout as given, as the provider checkpoint path does.
        layout = transformer_data.get("pipeline_model_parallel_layout")
        if isinstance(layout, PipelineParallelLayerLayout):
            transformer_data["pipeline_model_parallel_layout"] = layout.input_data
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgeHybridModelConfig":
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
class BridgeHybridMLAModelConfig(MLAModelConfigMixin, BridgeHybridModelConfig):
    """Hybrid model config with Multi-Latent Attention (MLA) in its attention layers.

    The nested transformer config is an ``MLATransformerConfig``. It alone
    stores and serializes ``rotary_base`` and ``rotary_percent``, because MLA
    attention applies RoPE from them; on this config they are constructor
    arguments and views of the nested values (see ``MLAModelConfigMixin``).
    Hybrid MLA families such as DeepSeek-V4 and GLM-5 subclass it, as their
    providers combine ``HybridModelProvider`` with an MLA config.
    """


__all__ = ["BridgeHybridMLAModelConfig", "BridgeHybridModelConfig"]
