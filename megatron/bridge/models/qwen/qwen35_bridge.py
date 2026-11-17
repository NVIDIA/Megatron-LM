# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen/qwen35_bridge.py
# Source license: Apache-2.0.
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

from typing import Any

import torch
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (  # noqa: F401
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    GDNLinearMappingSeparate,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.conversion.transformers_compat import full_attention_interval_from_hf
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed
from megatron.bridge.models.qwen.modeling_qwen35.hybrid import (
    configure_qwen35_hybrid,
    qwen35_hf_layer_config,
    qwen35_hybrid_mappings,
)
from megatron.bridge.models.qwen.modeling_qwen35.model_config import Qwen35HybridModelConfig
from megatron.core.models.hybrid.hybrid_model import HybridModel


def _qwen35_common_config_kwargs(text_config) -> dict[str, Any]:
    """Return the Qwen3.5 LM configuration shared by the dense and MoE variants.

    Covers normalization and the GDN hybrid architecture. The text bridges add these
    fields to their builder config kwargs, which the VL bridges reuse for their
    language model.

    Args:
        text_config: HuggingFace config object (or text_config for VLMs)
            so that language-model fields are read from the correct level.

    Returns:
        Megatron config fields keyed by name.
    """
    return dict(
        # --- Common Qwen3 LLM settings ---
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_qkv_bias=getattr(text_config, "attention_bias", False),
        add_bias_linear=False,
        qk_layernorm=True,
        hidden_dropout=0.0,
        # --- Qwen3-Next hybrid architecture settings ---
        layernorm_zero_centered_gamma=True,
        attention_output_gate=True,
        experimental_attention_variant="gdn",
        # full_attention_interval defines how often standard attention appears:
        # e.g., 4 means every 4th layer is standard attention (3 GDN + 1 Attn)
        linear_attention_freq=full_attention_interval_from_hf(text_config),
        linear_num_value_heads=getattr(text_config, "linear_num_value_heads", 32),
        rotary_percent=getattr(text_config, "rope_parameters", {}).get(
            "partial_rotary_factor", 0.25
        ),
        # --- GDN (Gated DeltaNet) specific parameters ---
        linear_conv_kernel_dim=getattr(text_config, "linear_conv_kernel_dim", 4),
        linear_key_head_dim=getattr(text_config, "linear_key_head_dim", 128),
        linear_value_head_dim=getattr(text_config, "linear_value_head_dim", 128),
        linear_num_key_heads=getattr(text_config, "linear_num_key_heads", 16),
    )


def _qwen35_moe_config_kwargs(text_config) -> dict[str, Any]:
    """Return the Qwen3.5 MoE configuration added on top of the common LM configuration.

    Args:
        text_config: HuggingFace config object (or text_config for VLMs)
            so that language-model fields are read from the correct level.

    Returns:
        Megatron config fields keyed by name.
    """
    return dict(
        moe_ffn_hidden_size=getattr(text_config, "moe_intermediate_size", 1024),
        num_moe_experts=getattr(text_config, "num_experts", 512),
        moe_router_topk=getattr(text_config, "num_experts_per_tok", 10),
        moe_shared_expert_intermediate_size=getattr(
            text_config, "shared_expert_intermediate_size", None
        ),
        moe_shared_expert_gate=True,
        moe_grouped_gemm=True,
        moe_router_load_balancing_type="global_aux_loss",
        moe_router_pre_softmax=False,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        moe_router_dtype="fp32",
    )


def _moe_routed_expert_mappings(
    hf_prefix, megatron_prefix, experts_packed, transpose_on_export=False
):
    """Routed-expert mappings for a MoE decoder, for both the grouped-GEMM and SequentialMLP layouts.

    ``experts_packed`` selects fused HF tensors (``experts.gate_up_proj`` / ``experts.down_proj``)
    vs per-expert (``experts.<i>.gate_proj`` / ``up_proj`` / ``down_proj``), matching the checkpoint so
    HF->Megatron->HF round-trips. ``transpose_on_export`` applies to the fused mappings (e.g. Qwen3-VL).
    """
    grouped_fc1 = f"{megatron_prefix}decoder.layers.*.mlp.experts.linear_fc1.weight*"
    grouped_fc2 = f"{megatron_prefix}decoder.layers.*.mlp.experts.linear_fc2.weight*"
    seq_fc1 = f"{megatron_prefix}decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight"
    seq_fc2 = f"{megatron_prefix}decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight"
    if experts_packed:
        gate_up = f"{hf_prefix}layers.*.mlp.experts.gate_up_proj"
        down = f"{hf_prefix}layers.*.mlp.experts.down_proj"
        return [
            FusedGatedExpertMapping(
                megatron_param=grouped_fc1,
                hf_param=gate_up,
                transpose_on_export=transpose_on_export,
            ),
            FusedExpertMapping(
                megatron_param=grouped_fc2, hf_param=down, transpose_on_export=transpose_on_export
            ),
            FusedGatedExpertMapping(
                megatron_param=seq_fc1, hf_param=gate_up, transpose_on_export=transpose_on_export
            ),
            FusedExpertMapping(
                megatron_param=seq_fc2, hf_param=down, transpose_on_export=transpose_on_export
            ),
        ]
    gate = f"{hf_prefix}layers.*.mlp.experts.*.gate_proj.weight"
    up = f"{hf_prefix}layers.*.mlp.experts.*.up_proj.weight"
    down = f"{hf_prefix}layers.*.mlp.experts.*.down_proj.weight"
    return [
        GatedMLPMapping(megatron_param=grouped_fc1, gate=gate, up=up),
        AutoMapping(megatron_param=grouped_fc2, hf_param=down),
        GatedMLPMapping(megatron_param=seq_fc1, gate=gate, up=up),
        AutoMapping(megatron_param=seq_fc2, hf_param=down),
    ]


@MegatronModelBridge.register_bridge(
    source="Qwen3_5MoeForCausalLM", target=HybridModel, model_type="qwen3_5_moe_text"
)
class Qwen35MoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3.5 Language Model (MoE variant).

    This bridge handles the conversion between HuggingFace Qwen3.5 language
    model and Megatron-Core Qwen3.5 Model formats, including weight mappings and
    configuration translation for the hybrid GDN+Attention LM architecture.

    The weight mappings handle:
    - Language model hybrid layers (GDN + standard attention)
    - MoE layers with routed and shared experts
    - QK layernorm, zero-centered RMSNorm for GDN output norm

    Architecture: 15 × (3 × (GDN → MoE) + 1 × (Attention → MoE)) = 60 layers

    The VL variant (Qwen35VLMoEBridge) reuses this bridge's builder config kwargs
    for its language model, and the static mapping methods.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-397B-A17B")
        >>> model.save_pretrained("./Qwen3.5-397B-A17B-LM")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-397B-A17B")
        >>> tokenizer.save_pretrained("./Qwen3.5-397B-A17B")
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("./Qwen3.5-397B-A17B")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = Qwen35HybridModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    @staticmethod
    def _get_moe_lm_mappings(hf_prefix="model.", megatron_prefix="", experts_packed=False):
        """Get language model parameter mappings for MoE Qwen3.5.

        Args:
            hf_prefix: Prefix for HF param names in safetensors. Use "model.layers.*"
                for LM and "model.language_model.layers.*" for VL models.
            megatron_prefix: Prefix for Megatron param names. Use "" for LM
                (default) and "language_model." for VL models.
            experts_packed: Whether the routed experts are stored fused (``experts.gate_up_proj`` /
                ``down_proj``) rather than per-expert (``experts.<i>.gate_proj`` / ``up_proj`` /
                ``down_proj``). Selected from the checkpoint layout by ``mapping_registry``.

        Returns:
            List of mapping objects for the MoE LM portion.
        """

        # =====================================================================
        # Simple 1:1 parameter mappings
        # =====================================================================
        param_mappings = {
            # =================================================================
            # Language Model: Embeddings and output
            # =================================================================
            f"{megatron_prefix}embedding.word_embeddings.weight": f"{hf_prefix}embed_tokens.weight",
            f"{megatron_prefix}output_layer.weight": "lm_head.weight",
            f"{megatron_prefix}decoder.final_layernorm.weight": f"{hf_prefix}norm.weight",
            # =================================================================
            # Language Model: MoE router
            # =================================================================
            f"{megatron_prefix}decoder.layers.*.mlp.router.weight": f"{hf_prefix}layers.*.mlp.gate.weight",
            f"{megatron_prefix}decoder.layers.*.pre_mlp_layernorm.weight": f"{hf_prefix}layers.*.post_attention_layernorm.weight",
            # =================================================================
            # Language Model: Standard attention layers (Gated Attention)
            # These mappings apply to layers where standard attention is used
            # (every 4th layer in the 15 × (3 GDN + 1 Attn) pattern)
            # =================================================================
            f"{megatron_prefix}decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": f"{hf_prefix}layers.*.input_layernorm.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.q_layernorm.weight": f"{hf_prefix}layers.*.self_attn.q_norm.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.k_layernorm.weight": f"{hf_prefix}layers.*.self_attn.k_norm.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.linear_proj.weight": f"{hf_prefix}layers.*.self_attn.o_proj.weight",
            # =================================================================
            # Language Model: Linear attention (Gated DeltaNet) layers
            # These mappings apply to layers where GDN is used
            # (3 out of every 4 layers)
            # =================================================================
            f"{megatron_prefix}decoder.layers.*.self_attention.in_proj.layer_norm_weight": f"{hf_prefix}layers.*.input_layernorm.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.out_proj.weight": f"{hf_prefix}layers.*.linear_attn.out_proj.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.A_log": f"{hf_prefix}layers.*.linear_attn.A_log",
            f"{megatron_prefix}decoder.layers.*.self_attention.dt_bias": f"{hf_prefix}layers.*.linear_attn.dt_bias",
        }

        mapping_list = []

        # Convert simple 1:1 mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Register module types for GDN and shared expert (needed for AutoMapping detection)
        AutoMapping.register_module_type("SharedExpertMLP", "column")
        AutoMapping.register_module_type("GatedDeltaNet", "column")

        # =====================================================================
        # Special mappings requiring parameter transformation
        # =====================================================================
        mapping_list.extend(
            [
                # =============================================================
                # Language Model: Standard Attention QKV
                # Combines separate Q, K, V matrices into single QKV matrix
                # =============================================================
                QKVMapping(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.self_attention.linear_qkv.weight",
                    q=f"{hf_prefix}layers.*.self_attn.q_proj.weight",
                    k=f"{hf_prefix}layers.*.self_attn.k_proj.weight",
                    v=f"{hf_prefix}layers.*.self_attn.v_proj.weight",
                ),
                # =============================================================
                # Language Model: GDN (Gated DeltaNet) specific mappings
                # =============================================================
                # GDN Conv1d: depthwise causal convolution
                GDNConv1dMapping(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.self_attention.conv1d.weight",
                    hf_param=f"{hf_prefix}layers.*.linear_attn.conv1d.weight",
                ),
                # GDN Input Projection: Qwen3.5 stores 4 separate weight tensors
                # (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) instead of the
                # 2 fused tensors (in_proj_qkvz, in_proj_ba) used by Qwen3-Next.
                GDNLinearMappingSeparate(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.self_attention.in_proj.weight",
                    qkv=f"{hf_prefix}layers.*.linear_attn.in_proj_qkv.weight",
                    z=f"{hf_prefix}layers.*.linear_attn.in_proj_z.weight",
                    b=f"{hf_prefix}layers.*.linear_attn.in_proj_b.weight",
                    a=f"{hf_prefix}layers.*.linear_attn.in_proj_a.weight",
                ),
                # GDN Output Norm: zero-centered RMSNorm conversion
                # Qwen3-Next uses standard RMSNorm initialized to ones for output norm,
                # while Megatron uses zero-centered RMSNorm, so we subtract 1 during conversion.
                RMSNorm2ZeroCenteredRMSNormMapping(
                    f"{megatron_prefix}decoder.layers.*.self_attention.out_norm.weight",
                    f"{hf_prefix}layers.*.linear_attn.norm.weight",
                ),
                # =============================================================
                # Language Model: MoE routed-expert MLPs. Covers both the grouped-GEMM
                # (experts.linear_fc*.weight*) and SequentialMLP (experts.local_experts.*.linear_fc*)
                # Megatron layouts, selecting fused vs per-expert HF mappings from the checkpoint.
                # =============================================================
                *_moe_routed_expert_mappings(hf_prefix, megatron_prefix, experts_packed),
                # =============================================================
                # Language Model: Shared Expert MLPs
                # =============================================================
                GatedMLPMapping(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate=f"{hf_prefix}layers.*.mlp.shared_expert.gate_proj.weight",
                    up=f"{hf_prefix}layers.*.mlp.shared_expert.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                    hf_param=f"{hf_prefix}layers.*.mlp.shared_expert.down_proj.weight",
                ),
                # Shared expert gate weight (replicated across TP ranks)
                ReplicatedMapping(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.mlp.shared_experts.gate_weight",
                    hf_param=f"{hf_prefix}layers.*.mlp.shared_expert_gate.weight",
                ),
            ]
        )

        return mapping_list

    @staticmethod
    def _get_moe_mtp_mappings(megatron_prefix: str = "", mtp_experts_packed: bool = False):
        """Get MTP parameter mappings for MoE Qwen3.5.

        Args:
            megatron_prefix: Prefix for Megatron param names. Use "" for LM and
                "language_model." for VL models.
            mtp_experts_packed: Whether the MTP experts are packed.
                Qwen3.5 stores per-expert
                (mtp.layers.0.mlp.experts.{i}.gate_proj.weight),
                whereas Qwen3.6 stores packed
                (mtp.layers.0.mlp.experts.gate_up_proj).

        Returns:
            List of mapping objects for the MoE MTP portion.
        """
        mapping_list = []

        # =================================================================
        # MTP (Multi-Token Prediction) mappings
        # MTP uses standard attention (not GDN) and standard per-expert
        # MoE format (unlike the fused gate_up_proj in main decoder).
        # Megatron VL prefix: language_model.mtp.*
        # Megatron LM prefix: mtp.* (LM)
        # HF prefix: mtp.* (top-level, not under model.language_model.)
        # =================================================================
        mtp_param_mappings = {
            f"{megatron_prefix}mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            f"{megatron_prefix}mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            f"{megatron_prefix}mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            f"{megatron_prefix}mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
        }
        for megatron_param, hf_param in mtp_param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Megatron-Core has exposed the MTP transformer sub-layer as both
        # ``mtp_model_layer`` and ``transformer_layer`` depending on version, so
        # register every sub-layer mapping under both spellings (as glm45_bridge
        # and glm5_bridge do) and checkpoints from either version convert.
        for mtp_layer_attr in ("mtp_model_layer", "transformer_layer"):
            layer0 = f"{megatron_prefix}mtp.layers.0.{mtp_layer_attr}"
            layer = f"{megatron_prefix}mtp.layers.*.{mtp_layer_attr}"
            sublayer_param_mappings = {
                f"{layer0}.mlp.router.weight": "mtp.layers.0.mlp.gate.weight",
                f"{layer0}.pre_mlp_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
                f"{layer0}.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
                f"{layer0}.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
                f"{layer0}.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
                f"{layer0}.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
            }
            for megatron_param, hf_param in sublayer_param_mappings.items():
                mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

            mapping_list.extend(
                [
                    QKVMapping(
                        megatron_param=f"{layer}.self_attention.linear_qkv.weight",
                        q="mtp.layers.*.self_attn.q_proj.weight",
                        k="mtp.layers.*.self_attn.k_proj.weight",
                        v="mtp.layers.*.self_attn.v_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"{layer}.mlp.shared_experts.linear_fc1.weight",
                        gate="mtp.layers.*.mlp.shared_expert.gate_proj.weight",
                        up="mtp.layers.*.mlp.shared_expert.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{layer}.mlp.shared_experts.linear_fc2.weight",
                        hf_param="mtp.layers.*.mlp.shared_expert.down_proj.weight",
                    ),
                    ReplicatedMapping(
                        megatron_param=f"{layer0}.mlp.shared_experts.gate_weight",
                        hf_param="mtp.layers.0.mlp.shared_expert_gate.weight",
                    ),
                ]
            )

            if mtp_experts_packed:
                # Qwen3.6: packed format (same as main decoder)
                mapping_list.extend(
                    [
                        FusedGatedExpertMapping(
                            megatron_param=f"{layer}.mlp.experts.linear_fc1.weight*",
                            hf_param="mtp.layers.*.mlp.experts.gate_up_proj",
                        ),
                        FusedExpertMapping(
                            megatron_param=f"{layer}.mlp.experts.linear_fc2.weight*",
                            hf_param="mtp.layers.*.mlp.experts.down_proj",
                        ),
                    ]
                )
            else:
                # Qwen3.5: per-expert format (current behavior)
                mapping_list.extend(
                    [
                        GatedMLPMapping(
                            megatron_param=f"{layer}.mlp.experts.linear_fc1.weight*",
                            gate="mtp.layers.*.mlp.experts.*.gate_proj.weight",
                            up="mtp.layers.*.mlp.experts.*.up_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"{layer}.mlp.experts.linear_fc2.weight*",
                            hf_param="mtp.layers.*.mlp.experts.*.down_proj.weight",
                        ),
                    ]
                )

        return mapping_list

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Reject the removed provider path instead of constructing an inconsistent model."""
        raise NotImplementedError(
            "Qwen3.5 uses Hybrid builders; use AutoBridge.get_model_config() or get_model()."
        )

    @classmethod
    def megatron_to_hf_config(cls, model_config) -> dict:
        """Restore HF block coordinates and attention order on export."""
        config = super().megatron_to_hf_config(model_config)
        config.update(qwen35_hf_layer_config(model_config.hybrid_layer_pattern))
        return config

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face Qwen3.5 MoE text config to builder config kwargs."""
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)

        # --- LM-specific parameters ---
        config_kwargs.update(_qwen35_common_config_kwargs(hf_config))
        config_kwargs.update(_qwen35_moe_config_kwargs(hf_config))

        # --- LM-specific overrides ---
        config_kwargs.update(
            autocast_dtype=torch.bfloat16,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
            # Heterogeneous checkpointing for mixed attention layers
            hetereogenous_dist_checkpoint=True,
        )
        # The shared mapping already selects "yarn" for YaRN-scaled checkpoints.
        config_kwargs.setdefault("position_embedding_type", "rope")

        # --- MTP (Multi-Token Prediction) ---
        if config_kwargs.get("mtp_num_layers"):
            config_kwargs["mtp_loss_scaling_factor"] = 0.1
        configure_qwen35_hybrid(config_kwargs, hf_config)
        return config_kwargs

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings for Qwen3.5 LM.

        Combines:
        - Standard attention: QKV, output projection, QK layernorm
        - Linear attention (GDN): in_proj, out_proj, conv1d, A_log, dt_bias, out_norm
        - MoE: router, routed expert MLPs, shared expert MLPs, shared expert gate
        - Embeddings, output layer, final layernorm

        Naming Convention:
        - Megatron language model params are prefixed with "decoder."
        - HF language model params are prefixed with "model.layers.*"

        Returns:
            MegatronMappingRegistry with all parameter mappings
        """
        # Detect MTP MoE expert weight format: Qwen3.5 stores per-expert
        # (mtp.layers.0.mlp.experts.{i}.gate_proj.weight), Qwen3.6 stores packed
        # (mtp.layers.0.mlp.experts.gate_up_proj). Same architecture string,
        # different storage — must inspect HF keys.
        mtp_experts_packed = False
        hf_pretrained = getattr(self, "hf_pretrained", None)
        if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
            hf_keys = set(hf_pretrained.state.source.get_all_keys())
            if "mtp.layers.0.mlp.experts.gate_up_proj" in hf_keys:
                mtp_experts_packed = True
        experts_packed = moe_experts_stored_packed(hf_pretrained, "model.layers.")

        mapping_list = []

        mapping_list.extend(
            self._get_moe_lm_mappings(megatron_prefix="", experts_packed=experts_packed)
        )
        mapping_list.extend(
            self._get_moe_mtp_mappings(megatron_prefix="", mtp_experts_packed=mtp_experts_packed)
        )
        return MegatronMappingRegistry(
            *qwen35_hybrid_mappings(mapping_list, num_layers=self.hf_config.num_hidden_layers)
        )


@MegatronModelBridge.register_bridge(
    source="Qwen3_5ForCausalLM", target=HybridModel, model_type="qwen3_5_text"
)
class Qwen35Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3.5 Dense Language Model.

    This bridge handles the conversion between HuggingFace Qwen3.5 language
    model and Megatron-Core Qwen3.5 Model formats, including weight mappings and
    configuration translation for the hybrid GDN+Attention LM architecture.

    The weight mappings handle:
    - Language model hybrid layers (GDN + standard attention)
    - Dense MLP with gated SiLU activation (fused pre-MLP layernorm)
    - QK layernorm, zero-centered RMSNorm for GDN output norm

    Architecture (27B): 16 × (3 × GDN + 1 × Attention) = 64 layers

    The VL variant (Qwen35VLBridge) reuses this bridge's builder config kwargs
    for its language model, and the static mapping methods.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B")
        >>> model.save_pretrained("./Qwen3.5-27B-LM")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
        >>> tokenizer.save_pretrained("./Qwen3.5-27B-LM")
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("./Qwen3.5-27B-LM")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = Qwen35HybridModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    @staticmethod
    def _get_dense_lm_mappings(
        *,
        hf_prefix: str = "model.",
        megatron_prefix: str = "",
        output_layer_hf_param: str | None = "lm_head.weight",
    ) -> list[MegatronParamMapping]:
        """Get language model parameter mappings for dense (non-MoE) Qwen3.5.

        Args:
            hf_prefix: Prefix for HF param names in safetensors. Use "model.layers.*"
                for LM and "model.language_model.layers.*" for VL models.
            megatron_prefix: Prefix for Megatron param names. Use "" for LM
                (default) and "language_model." for VL models.
            output_layer_hf_param: HF output-head parameter name. If ``None``,
                omit the output-head mapping.

        Returns:
            List of mapping objects for the dense LM portion.
        """
        param_mappings = {
            # =================================================================
            # Language Model: Embeddings and output
            # =================================================================
            f"{megatron_prefix}embedding.word_embeddings.weight": f"{hf_prefix}embed_tokens.weight",
            f"{megatron_prefix}decoder.final_layernorm.weight": f"{hf_prefix}norm.weight",
            # =================================================================
            # Language Model: Dense MLP (pre-MLP layernorm fused into linear_fc1)
            # =================================================================
            f"{megatron_prefix}decoder.layers.*.mlp.linear_fc1.layer_norm_weight": f"{hf_prefix}layers.*.post_attention_layernorm.weight",
            f"{megatron_prefix}decoder.layers.*.mlp.linear_fc2.weight": f"{hf_prefix}layers.*.mlp.down_proj.weight",
            # =================================================================
            # Language Model: Standard attention layers (Gated Attention)
            # =================================================================
            f"{megatron_prefix}decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": f"{hf_prefix}layers.*.input_layernorm.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.q_layernorm.weight": f"{hf_prefix}layers.*.self_attn.q_norm.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.k_layernorm.weight": f"{hf_prefix}layers.*.self_attn.k_norm.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.linear_proj.weight": f"{hf_prefix}layers.*.self_attn.o_proj.weight",
            # =================================================================
            # Language Model: Linear attention (Gated DeltaNet) layers
            # =================================================================
            f"{megatron_prefix}decoder.layers.*.self_attention.in_proj.layer_norm_weight": f"{hf_prefix}layers.*.input_layernorm.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.out_proj.weight": f"{hf_prefix}layers.*.linear_attn.out_proj.weight",
            f"{megatron_prefix}decoder.layers.*.self_attention.A_log": f"{hf_prefix}layers.*.linear_attn.A_log",
            f"{megatron_prefix}decoder.layers.*.self_attention.dt_bias": f"{hf_prefix}layers.*.linear_attn.dt_bias",
        }
        if output_layer_hf_param is not None:
            param_mappings[f"{megatron_prefix}output_layer.weight"] = output_layer_hf_param

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        AutoMapping.register_module_type("GatedDeltaNet", "column")

        mapping_list.extend(
            [
                # =============================================================
                # Language Model: Standard Attention QKV
                # =============================================================
                QKVMapping(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.self_attention.linear_qkv.weight",
                    q=f"{hf_prefix}layers.*.self_attn.q_proj.weight",
                    k=f"{hf_prefix}layers.*.self_attn.k_proj.weight",
                    v=f"{hf_prefix}layers.*.self_attn.v_proj.weight",
                ),
                # =============================================================
                # Language Model: Dense MLP (gated: gate_proj + up_proj → linear_fc1)
                # =============================================================
                GatedMLPMapping(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.mlp.linear_fc1.weight",
                    gate=f"{hf_prefix}layers.*.mlp.gate_proj.weight",
                    up=f"{hf_prefix}layers.*.mlp.up_proj.weight",
                ),
                # =============================================================
                # Language Model: GDN (Gated DeltaNet) specific mappings
                # =============================================================
                GDNConv1dMapping(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.self_attention.conv1d.weight",
                    hf_param=f"{hf_prefix}layers.*.linear_attn.conv1d.weight",
                ),
                GDNLinearMappingSeparate(
                    megatron_param=f"{megatron_prefix}decoder.layers.*.self_attention.in_proj.weight",
                    qkv=f"{hf_prefix}layers.*.linear_attn.in_proj_qkv.weight",
                    z=f"{hf_prefix}layers.*.linear_attn.in_proj_z.weight",
                    b=f"{hf_prefix}layers.*.linear_attn.in_proj_b.weight",
                    a=f"{hf_prefix}layers.*.linear_attn.in_proj_a.weight",
                ),
                RMSNorm2ZeroCenteredRMSNormMapping(
                    f"{megatron_prefix}decoder.layers.*.self_attention.out_norm.weight",
                    f"{hf_prefix}layers.*.linear_attn.norm.weight",
                ),
            ]
        )

        return mapping_list

    @staticmethod
    def _get_dense_mtp_mappings(megatron_prefix=""):
        """Get MTP (Multi-Token Prediction) parameter mappings for dense Qwen3.5.

        Args:
            megatron_prefix: Prefix for Megatron param names. Use "" for LM and
                "language_model." for VL models.

        Returns:
            List of mapping objects for the MTP portion.
        """
        mapping_list = []

        # =================================================================
        # MTP (Multi-Token Prediction) mappings
        # MTP uses standard attention (not GDN) and dense MLP.
        # Megatron VL prefix: language_model.mtp.*
        # Megatron ML prefix: mtp.*
        # HF prefix: mtp.* (top-level, not under model.language_model.)
        # =================================================================
        mtp_param_mappings = {
            f"{megatron_prefix}mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            f"{megatron_prefix}mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            f"{megatron_prefix}mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            f"{megatron_prefix}mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
            f"{megatron_prefix}mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight": "mtp.layers.0.post_attention_layernorm.weight",
            f"{megatron_prefix}mtp.layers.0.mtp_model_layer.mlp.linear_fc2.weight": "mtp.layers.0.mlp.down_proj.weight",
            f"{megatron_prefix}mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
            f"{megatron_prefix}mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
            f"{megatron_prefix}mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
            f"{megatron_prefix}mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        }
        for megatron_param, hf_param in mtp_param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param=f"{megatron_prefix}mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight",
                    q="mtp.layers.*.self_attn.q_proj.weight",
                    k="mtp.layers.*.self_attn.k_proj.weight",
                    v="mtp.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param=f"{megatron_prefix}mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight",
                    gate="mtp.layers.0.mlp.gate_proj.weight",
                    up="mtp.layers.0.mlp.up_proj.weight",
                ),
            ]
        )

        return mapping_list

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Reject the removed provider path instead of constructing an inconsistent model."""
        raise NotImplementedError(
            "Qwen3.5 uses Hybrid builders; use AutoBridge.get_model_config() or get_model()."
        )

    @classmethod
    def megatron_to_hf_config(cls, model_config) -> dict:
        """Restore HF block coordinates and attention order on export."""
        config = super().megatron_to_hf_config(model_config)
        config.update(qwen35_hf_layer_config(model_config.hybrid_layer_pattern))
        return config

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face Qwen3.5 dense text config to builder config kwargs."""
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)

        # --- LM-specific parameters ---
        config_kwargs.update(_qwen35_common_config_kwargs(hf_config))

        # --- LM-specific overrides ---
        config_kwargs.update(
            autocast_dtype=torch.bfloat16,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
            # Heterogeneous checkpointing for mixed attention layers
            hetereogenous_dist_checkpoint=True,
        )
        # The shared mapping already selects "yarn" for YaRN-scaled checkpoints.
        config_kwargs.setdefault("position_embedding_type", "rope")

        # --- MTP (Multi-Token Prediction) ---
        if config_kwargs.get("mtp_num_layers"):
            config_kwargs["mtp_loss_scaling_factor"] = 0.1
        configure_qwen35_hybrid(config_kwargs, hf_config)
        return config_kwargs

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry for Qwen3.5 dense ML model.

        Key differences from the MoE variant:
        - Dense MLP: gate_proj + up_proj fused into linear_fc1, down_proj as linear_fc2
        - Pre-MLP layernorm fused into mlp.linear_fc1 (not a separate pre_mlp_layernorm)
        - No MoE router, routed expert MLPs, or shared expert mappings
        """
        mapping_list = []

        mapping_list.extend(self._get_dense_lm_mappings(megatron_prefix=""))
        mapping_list.extend(self._get_dense_mtp_mappings(megatron_prefix=""))
        return MegatronMappingRegistry(
            *qwen35_hybrid_mappings(mapping_list, num_layers=self.hf_config.num_hidden_layers)
        )
