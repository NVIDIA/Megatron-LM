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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/src/megatron/bridge/models/deepseek/common.py
# Source license: Apache-2.0.

"""Weight mappings shared by the DeepSeek family."""

from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping


def get_common_mapping_list(hf_config=None) -> list:
    """
    Returns a list of common parameter mappings for the DeepSeek family of models.

    Args:
        hf_config: Optional HuggingFace config. If provided and contains MTP layers,
                   MTP mappings will be included.
    """
    param_mappings = {
        # Embed
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        # Attention
        "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
        "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
        #  In deepseek, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
        #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
        #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
        "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
        "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
        "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
        # Mcore local spec
        "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
        # Dense MLP
        "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        # MoE
        "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
        "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
        "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "model.layers.*.mlp.experts.*.down_proj.weight",
        "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
        # LM Head
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
        # MLA
        "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        # Mcore local spec
        "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        # For models without MLA
        "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
    }

    mapping_list = []
    # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
    for megatron_param, hf_param in param_mappings.items():
        mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

    mapping_list.extend(
        [
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                up="model.layers.*.mlp.shared_experts.up_proj.weight",
            ),
        ]
    )

    if hf_config is not None:
        # Add MTP mappings if config has MTP layers
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        if num_mtp_layers > 0:
            num_transformer_layers = hf_config.num_hidden_layers

            for mtp_layer in range(num_mtp_layers):
                # Add layer-specific mappings for MTP transformer layers
                for megatron_param, hf_param in param_mappings.items():
                    megatron_param_mtp = (
                        megatron_param.replace(".*", ".*.mtp_model_layer", 1)
                        .replace("decoder", "mtp")
                        .replace(".*", f".{mtp_layer}", 1)
                    )
                    hf_param_mtp = hf_param.replace(
                        "layers.*", f"layers.{mtp_layer + num_transformer_layers}"
                    )
                    mapping_list.append(
                        AutoMapping(megatron_param=megatron_param_mtp, hf_param=hf_param_mtp)
                    )

                # Add MTP-specific normalization and projection layers
                mapping_list.extend(
                    [
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                            hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.enorm.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                            hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.hnorm.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                            hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.eh_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                            hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.shared_head.norm.weight",
                        ),
                    ]
                )

                # Add MTP Gated MLP mappings
                mapping_list.extend(
                    [
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.up_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.shared_experts.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.up_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.linear_fc1.weight*",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.local_experts.*.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                        ),
                    ]
                )

    return mapping_list
