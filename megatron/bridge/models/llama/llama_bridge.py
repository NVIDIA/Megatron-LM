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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/src/megatron/bridge/models/llama/llama_bridge.py
# Source license: Apache-2.0.

import logging
from collections.abc import Mapping
from typing import Any

import torch
from transformers import LlamaForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.core.models.gpt.gpt_model import GPTModel

logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel, model_type="llama")
class LlamaBridge(MegatronModelBridge):
    """
    Megatron Bridge for Llama Causal LM.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = BridgeGPTModelConfig

    # Megatron Core applies llama3 RoPE scaling with these fixed values; only the factor is configurable.
    _MCORE_LLAMA3_ROPE_SCALING = {
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }

    # RoPE types this bridge converts. Megatron Core implements no others for Llama, such as
    # dynamic NTK scaling.
    _SUPPORTED_ROPE_TYPES = (None, "default", "linear", "llama3", "yarn")

    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Convert a Hugging Face Llama config to builder config kwargs.

        Raises:
            ValueError: If the checkpoint uses a RoPE type, or llama3 RoPE scaling values, that
                Megatron Core cannot apply.
        """
        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
        if rope_type not in self._SUPPORTED_ROPE_TYPES:
            raise ValueError(
                f"Unsupported RoPE type {rope_type!r}. Megatron Core applies default, linear, llama3 "
                "and yarn RoPE to Llama models, so converting this checkpoint would change its "
                "rotary frequencies."
            )

        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            hidden_dropout=0.0,
            bias_activation_fusion=True,
            masked_softmax_fusion=True,
            persist_layer_norm=True,
            bias_dropout_fusion=True,
            apply_rope_fusion=True,
            rotary_percent=1.0,
            rope_scaling=False,
            rope_scaling_factor=1.0,
        )
        # The shared mapping already selects "yarn" and its Hugging Face defaults for YaRN-scaled checkpoints.
        config_kwargs.setdefault("position_embedding_type", "rope")

        if rope_type == "llama3":
            unsupported = {
                key: rope_scaling[key]
                for key, value in self._MCORE_LLAMA3_ROPE_SCALING.items()
                if key in rope_scaling and rope_scaling[key] != value
            }
            if unsupported:
                raise ValueError(
                    f"Unsupported llama3 RoPE scaling values {unsupported}. Megatron Core applies llama3 RoPE "
                    f"scaling with the fixed values {self._MCORE_LLAMA3_ROPE_SCALING} and only the factor is "
                    "configurable, so converting this checkpoint would change its rotary frequencies."
                )
            config_kwargs["rope_scaling"] = True
            config_kwargs["rope_scaling_factor"] = rope_scaling.get("factor", 8.0)
        elif rope_type == "linear":
            config_kwargs["seq_len_interpolation_factor"] = rope_scaling["factor"]

        return config_kwargs

    @classmethod
    def megatron_to_hf_config(cls, model_config: BridgeGPTModelConfig) -> dict:
        """Convert a Megatron Llama model config to a Hugging Face Llama config dict.

        Uses the base class implementation, which exports YaRN, then adds llama3 or linear RoPE scaling, or
        ``{"rope_type": "default"}`` when the model has none.

        Args:
            model_config: Builder-backed GPT config with the Llama configuration.

        Returns:
            Dictionary of HuggingFace LlamaConfig parameters
        """
        hf_config = super(LlamaBridge, cls).megatron_to_hf_config(model_config)

        # Handle RoPE scaling for Llama 3.1/3.2 models
        if model_config.rope_scaling:
            hf_config["rope_scaling"] = {
                "rope_type": "llama3",
                "factor": model_config.rope_scaling_factor,
                **cls._MCORE_LLAMA3_ROPE_SCALING,
            }
        elif model_config.seq_len_interpolation_factor is not None:
            hf_config["rope_scaling"] = {
                "rope_type": "linear",
                "factor": model_config.seq_len_interpolation_factor,
            }
        elif "rope_scaling" not in hf_config:
            # State "no scaling" explicitly, so a config-only export does not keep a reference model's scaling.
            hf_config["rope_scaling"] = {"rope_type": "default"}

        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",  # te implementation
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",  # local implementation
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",  # te implementation
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",  # local implementation
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Preserve persisted Llama rotary inverse-frequency buffers on export."""
        input_layernorm_key = next(
            (
                name
                for name in converted_weights_dict
                if name.startswith("model.layers.") and name.endswith(".input_layernorm.weight")
            ),
            None,
        )
        if input_layernorm_key is None:
            return converted_weights_dict

        parts = input_layernorm_key.split(".")
        if len(parts) < 5 or not parts[2].isdigit():
            return converted_weights_dict

        layer_idx = int(parts[2])
        inv_freq_key = f"model.layers.{layer_idx}.self_attn.rotary_emb.inv_freq"
        if inv_freq_key not in hf_state_dict or inv_freq_key in converted_weights_dict:
            return converted_weights_dict

        inv_freq = hf_state_dict[inv_freq_key]
        reference_tensor = next(iter(converted_weights_dict.values()), None)
        if reference_tensor is not None:
            inv_freq = inv_freq.to(reference_tensor.device)

        converted_weights_dict[inv_freq_key] = inv_freq
        return converted_weights_dict
