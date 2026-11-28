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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/src/megatron/bridge/models/gpt_oss/gpt_oss_bridge.py
# Source license: Apache-2.0.

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from transformers import GptOssForCausalLM, PretrainedConfig

from megatron.bridge.models.conversion.gtp import _get_mapping_shape
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    QKVMapping,
    _align_expert_weight_to_shape,
    _LooseGatedMLPMapping,
)
from megatron.bridge.models.conversion.quantization_utils import dequantize_mxfp4
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.utils.common import extract_expert_number_from_param
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.models.gpt.gpt_model import GPTModel


@MegatronModelBridge.register_bridge(
    source=GptOssForCausalLM, target=GPTModel, model_type="gpt_oss"
)
class GPTOSSBridge(MegatronModelBridge):
    """
    Megatron Bridge for GPT-OSS models.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("openai/gpt-oss-20b")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = BridgeGPTModelConfig

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face GPT-OSS config to builder config kwargs."""
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=True,
            add_qkv_bias=getattr(hf_config, "attention_bias", True),
            share_embeddings_and_output_weights=False,
            position_embedding_type="yarn",
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
            moe_router_pre_softmax=False,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_router_load_balancing_type="none",
            bias_activation_fusion=True,
            bias_dropout_fusion=False,
            hidden_dropout=0.0,
            fp16=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            # GPT-OSS specific activation
            activation_func=quick_gelu,
            activation_func_clamp_value=7.0,
            glu_linear_offset=1.0,
            softmax_type="learnable",
            window_size=(hf_config.sliding_window - 1, 0),
            window_attn_skip_freq=2,
            # GPT-OSS uses intermediate_size for MoE FFN hidden size
            moe_ffn_hidden_size=hf_config.intermediate_size,
            # YaRN settings of the released checkpoints
            yarn_rotary_scaling_factor=32.0,
            yarn_original_max_position_embeddings=4096,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            yarn_correction_range_round_to_int=False,
            yarn_mscale=None,
            yarn_mscale_all_dim=None,
        )
        return config_kwargs

    @classmethod
    def megatron_to_hf_config(cls, model_config: BridgeGPTModelConfig) -> dict:
        """Convert a Megatron GPT-OSS model config to a Hugging Face GPT-OSS config dict."""
        hf_config = super().megatron_to_hf_config(model_config)
        hf_config["attention_bias"] = True
        return hf_config

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load weights from HuggingFace state dict with MXFP4 dequantization support.

        Per-expert ``down_proj`` is square for GPT-OSS-20B/120B (hidden == intermediate), so
        the bridge cannot auto-detect orientation from shape alone. BF16 checkpoints (e.g.
        ``unsloth/gpt-oss-20b-BF16``, and what ``transformers.GptOssForCausalLM`` produces at
        init) store it as ``[E, intermediate, hidden]``, matching ``gate_up_proj``'s
        ``[E, hidden, 2*intermediate]`` convention. MXFP4-dequantized weights come out as
        ``[E, hidden, intermediate]``. Megatron's TE ``RowParallelGroupedLinear`` expects
        per-expert ``(hidden, intermediate)``, so the BF16 path needs a transpose here while
        the MXFP4 path is already aligned. Without this, BF16 imports silently store down_proj
        in the wrong orientation and inference is broken.

        ``gate_up_proj`` gets the same treatment: BF16 checkpoints store it as
        ``[E, hidden, 2*intermediate]`` and MXFP4-dequantized weights come out as
        ``[E, 2*intermediate, hidden]``, Megatron's layout. The BF16 transpose happens here
        because the mapping can only tell the layouts apart by shape, which fails when
        ``hidden == 2*intermediate``.
        """
        if isinstance(hf_param, str):
            if hf_param in hf_state_dict:
                hf_weights = hf_state_dict[hf_param]
                if hf_param.endswith(".mlp.experts.down_proj") and hf_weights.ndim == 3:
                    cfg = self.hf_pretrained.config
                    hidden = cfg.hidden_size
                    intermediate = cfg.intermediate_size
                    last2 = tuple(hf_weights.shape[-2:])
                    if last2 == (intermediate, hidden) and intermediate != hidden:
                        # Unambiguous BF16 layout (E, intermediate, hidden); transpose to (E, hidden, intermediate).
                        hf_weights = hf_weights.transpose(-1, -2).contiguous()
                    elif last2 == (hidden, intermediate) and intermediate != hidden:
                        # Already aligned with Megatron — no-op.
                        pass
                    elif intermediate == hidden:
                        # Square: HF GptOssForCausalLM init produces (E, intermediate, hidden), so a plain BF16
                        # checkpoint is in that layout. Transpose to (E, hidden, intermediate) for Megatron.
                        hf_weights = hf_weights.transpose(-1, -2).contiguous()
                elif hf_param.endswith(".mlp.experts.gate_up_proj") and hf_weights.ndim == 3:
                    cfg = self.hf_pretrained.config
                    if tuple(hf_weights.shape[-2:]) == (cfg.hidden_size, 2 * cfg.intermediate_size):
                        # BF16 layout (E, hidden, 2*intermediate); transpose to Megatron's
                        # (E, 2*intermediate, hidden). When the two are equal the shape cannot tell
                        # the layouts apart, and BF16 checkpoints use HF's.
                        hf_weights = hf_weights.transpose(-1, -2).contiguous()
                return hf_weights
            blocks_key = hf_param + "_blocks"
            scales_key = hf_param + "_scales"
            if blocks_key in hf_state_dict and scales_key in hf_state_dict:
                hf_weights = dequantize_mxfp4(hf_state_dict[blocks_key], hf_state_dict[scales_key])
                # MXFP4 dequant already emits [E, hidden, intermediate] for down_proj — leave as-is.
                return hf_weights
            raise KeyError(
                f"Cannot locate weights for '{hf_param}'. Missing both de-quantized tensor and "
                f"quantized representation (blocks='{blocks_key}', scales='{scales_key}')."
            )
        return {k: hf_state_dict[v] for k, v in hf_param.items()}

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format.
        Based on the GPT-OSS importer code provided.
        """

        # Dictionary maps HF parameter names -> Megatron parameter names
        param_mappings = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.self_attn.o_proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.self_attn.sinks": "decoder.layers.*.self_attention.core_attention.softmax_offset",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.pre_mlp_layernorm.weight",
            "model.layers.*.mlp.router.bias": "decoder.layers.*.mlp.router.bias",
            "model.layers.*.mlp.router.weight": "decoder.layers.*.mlp.router.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        mapping_list.extend(
            [
                QKVMapping(
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                ),
                QKVMapping(
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                ),
                # Register the de-quantized weight names. If HF model is quantized,
                # the logic in `modify_loaded_hf_weight` will find the blocks and scales tensors.
                # Export is always de-quantized
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                ),
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.bias*",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.bias*",
                ),
                # SequentialMLP (moe_grouped_gemm=False): expert weights stored per local_expert
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj",
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                ),
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.bias",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj",
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.bias",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


class GPTOSSMLPDownProjMapping(AutoMapping):
    """MLPDownProj for expert weights in GPT-OSS models."""

    is_grouped_export = True

    def __init__(
        self, megatron_param: str, hf_param: str, permute_dims: tuple[int, ...] | None = None
    ):
        super().__init__(megatron_param, hf_param, permute_dims)
        self.allow_hf_name_mismatch = True

    @property
    def group_key(self) -> str:
        return self.hf_param

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        return super().hf_to_megatron(hf_weights[global_expert_number], megatron_module)

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor, megatron_module: nn.Module
    ) -> dict[str, torch.Tensor]:
        # Megatron stores per-expert weight as (hidden, intermediate); HF down_proj
        # weight is (E, intermediate, hidden). Transpose the last two dims so the
        # grouped-export stack assembles in HF's layout. Under EP the parent's gather
        # may have already cat'd across the EP group, producing a 3D (ep_size, out, in)
        # tensor — handle that too. The bias has no orientation to align (per-expert
        # 1-D, stacked to (E, hidden) on export), so leave bias mappings untouched.
        if megatron_weights is not None:
            megatron_weights = megatron_weights.contiguous()
        result = super().megatron_to_hf(megatron_weights, megatron_module)
        if self.hf_param.endswith("_bias"):
            return result
        return {
            k: v.transpose(-1, -2).contiguous() if v.ndim >= 2 else v for k, v in result.items()
        }


class GPTOSSMLPGateUpProjMapping(AutoMapping):
    """MLPGateUpProj for expert weights in GPT-OSS models.

    GPT-OSS uses alternating row interleaving for gate/up projections.
    """

    is_grouped_export = True

    def __init__(
        self, megatron_param: str, hf_param: str, permute_dims: tuple[int, ...] | None = None
    ):
        super().__init__(megatron_param, hf_param, permute_dims)
        self.allow_hf_name_mismatch = True
        self._gated_mapping = _LooseGatedMLPMapping(
            megatron_param=self.megatron_param,
            gate=f"{self.hf_param}.gate",
            up=f"{self.hf_param}.up",
        )

    @property
    def group_key(self) -> str:
        return self.hf_param

    @staticmethod
    def _interleave(gate_up_proj):
        return torch.cat((gate_up_proj[::2, ...], gate_up_proj[1::2, ...]), dim=0)

    @staticmethod
    def _uninterleave(elem: torch.Tensor) -> torch.Tensor:
        gate, up = torch.chunk(elem, 2, dim=0)
        output = torch.empty_like(elem)
        output[::2, ...] = gate
        output[1::2, ...] = up
        return output

    def hf_to_megatron(
        self, hf_weights: torch.Tensor | dict, megatron_module: nn.Module
    ) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        expert_weight = hf_weights[global_expert_number] if hf_weights.ndim >= 2 else hf_weights
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        target_shape = list(_get_mapping_shape(target_param))
        if not isinstance(target_param, DTensor):
            target_shape[0] *= self.tp_size
        expert_weight = _align_expert_weight_to_shape(expert_weight, target_shape, "gate_up_proj")
        if self.permute_dims is not None:
            return super().hf_to_megatron(self._interleave(expert_weight), megatron_module)
        return self._gated_mapping.hf_to_megatron(
            {"gate": expert_weight[::2, ...], "up": expert_weight[1::2, ...]}, megatron_module
        )

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor, megatron_module: nn.Module
    ) -> dict[str, torch.Tensor]:
        if self.permute_dims is not None:
            if megatron_weights is not None:
                megatron_weights = self._uninterleave(megatron_weights)
                if megatron_weights.ndim == 2:
                    megatron_weights = megatron_weights.t()
                megatron_weights = megatron_weights.contiguous()
            return super().megatron_to_hf(megatron_weights, megatron_module)
        converted = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        is_bias = self.hf_param.endswith("_bias")
        result = {}
        for name, gate in converted.items():
            if not name.endswith(".gate"):
                continue
            base_name = name.removesuffix(".gate")
            up = converted[f"{base_name}.up"]
            # Reconstruct alternating gate/up rows only after TP and EP gathering.
            # EP may add a leading expert axis; keep it intact for grouped export.
            if is_bias:
                interleaved = torch.stack((gate, up), dim=-1).flatten(-2, -1)
            else:
                interleaved = torch.stack((gate, up), dim=-2).flatten(-3, -2).transpose(-1, -2)
            result[base_name] = interleaved.contiguous()
        return result
