# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/nemotronh/nemotron_h_bridge.py
# Source license: Apache-2.0.

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    MambaConv1dMapping,
    MambaInProjMapping,
    MegatronParamMapping,
    QKVMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.core.activations import squared_relu
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.enums import AttnBackend


def _replace_wildcards(pattern: str, captures: Tuple[str, ...]) -> str:
    """Replace ** then * sequentially with captures."""
    out = pattern
    capture_index = 0
    while "**" in out and capture_index < len(captures):
        out = out.replace("**", captures[capture_index], 1)
        capture_index += 1
    while "*" in out and capture_index < len(captures):
        out = out.replace("*", captures[capture_index], 1)
        capture_index += 1
    return out


class _MTPFlatteningMapping(MegatronParamMapping[torch.Tensor]):
    """
    Pattern mapping that flattens Megatron's two-level MTP indices:

      megatron: mtp.layers.{outer}.mtp_model_layer.layers.{inner}.<...>
      hf:       mtp.layers.{outer * L + inner}.<...>

    Also supports an optional `inner_override` for parameters that live outside
    `mtp_model_layer.layers.*` in Megatron but should attach to a specific inner
    layer in HF (e.g., eh_proj on inner=0, final_layernorm on inner=L-1).
    """

    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        *,
        mtp_layers_per_block: int,
        inner_override: Optional[int] = None,
    ):
        # NOTE: We intentionally bypass wildcard validation because Megatron has
        # 2+N wildcards while HF has 1+N in the flattened scheme.
        self.megatron_param = megatron_param
        self.hf_param = hf_param
        self._mtp_layers_per_block = int(mtp_layers_per_block)
        self._inner_override = inner_override

        if self._mtp_layers_per_block <= 0:
            raise ValueError(
                "mtp_layers_per_block must be > 0 for MTP flattening mappings. "
                "This should come from len(hf_config.mtp_hybrid_override_pattern)."
            )

        # minimal init for PP/TP groups (copied pattern from MegatronParamMapping.__init__)
        self._broadcast_obj_cache = {}
        self._tensor_spec_output_cache = {}
        if torch.distributed.is_available():
            # Groups are populated in MegatronParamMapping when mpu is initialized.
            # Here we don't need them because this class resolves into AutoMapping/QKVMapping
            # (which will handle TP/PP), and we never run conversions on this instance.
            self.pp_group = None
            self.ep_group = None
            self._tp_group = None
            self._etp_group = None

        # Precompute wildcard counts to disambiguate resolve() calls coming from
        # megatron_to_hf_lookup vs hf_to_megatron_lookup.
        self._megatron_wc = MegatronParamMapping._count_wildcard_groups(self.megatron_param)
        self._hf_wc = (
            MegatronParamMapping._count_wildcard_groups(self.hf_param)
            if isinstance(self.hf_param, str)
            else 0
        )

    def resolve(self, captures: Tuple[str, ...]) -> MegatronParamMapping:
        # We primarily expect captures from Megatron pattern matching.
        # If called from HF reverse lookup, captures length will typically match hf wildcards.
        treat_as_hf = (len(captures) == self._hf_wc) and (self._hf_wc != self._megatron_wc)

        if treat_as_hf:
            # captures: (flat_layer_idx, [maybe more like expert idx...])
            flat = int(captures[0])
            outer = flat // self._mtp_layers_per_block
            inner = flat % self._mtp_layers_per_block
            # For outer-only megatron params (eh_proj, etc.) we only need `outer`.
            # If the megatron pattern contains an inner wildcard, use computed `inner`.
            # Build a synthetic capture tuple matching the megatron pattern.
            # We replace the first two megatron wildcards with outer/inner unless inner_override is set.
            inner_eff = self._inner_override if self._inner_override is not None else inner
            # Preserve any remaining captures (e.g., expert id) after the flat idx.
            remaining = captures[1:]
            megatron_captures = (str(outer), str(inner_eff), *remaining)
            resolved_megatron = _replace_wildcards(self.megatron_param, megatron_captures)
            resolved_hf = _replace_wildcards(self.hf_param, captures)
            return AutoMapping(megatron_param=resolved_megatron, hf_param=resolved_hf)

        # Treat captures as coming from the Megatron pattern.
        # captures: (outer, inner, [maybe expert idx...]) for nested params
        # or (outer,) for outer-only params like eh_proj.
        if len(captures) < 1:
            raise ValueError(f"Expected at least 1 capture for MTP mapping, got {captures}")

        outer = int(captures[0])

        # Determine the effective inner index
        if self._inner_override is not None:
            inner_eff = self._inner_override
            remaining = captures[1:]
        else:
            if len(captures) < 2:
                raise ValueError(
                    f"Expected 2 captures (outer, inner) for MTP nested mapping, got {captures}"
                )
            inner_eff = int(captures[1])
            remaining = captures[2:]

        flat = outer * self._mtp_layers_per_block + inner_eff

        resolved_megatron = _replace_wildcards(self.megatron_param, captures)

        # Resolve HF: first wildcard is the flattened idx, remaining wildcards pass through.
        hf_captures = (str(flat), *remaining)
        resolved_hf = _replace_wildcards(self.hf_param, hf_captures)

        return AutoMapping(megatron_param=resolved_megatron, hf_param=resolved_hf)

    # Required by ABC but never called at runtime: resolve() always returns an
    # AutoMapping, so hf_to_megatron/megatron_to_hf are invoked on that resolved
    # instance, not on this class.
    def hf_to_megatron(
        self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module
    ) -> torch.Tensor:
        raise NotImplementedError

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[torch.nn.Module]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class _MTPFlatteningQKVMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """Resolve-time wrapper that flattens MTP indices then delegates to QKVMapping."""

    def __init__(self, megatron_param: str, *, q: str, k: str, v: str, mtp_layers_per_block: int):
        # Bypass wildcard validation (2 wildcards on Megatron side vs 1 on HF q/k/v).
        self.megatron_param = megatron_param
        self.hf_param = {"q": q, "k": k, "v": v}
        self._mtp_layers_per_block = int(mtp_layers_per_block)
        if self._mtp_layers_per_block <= 0:
            raise ValueError("mtp_layers_per_block must be > 0 for MTP flattening QKV mapping.")
        self._megatron_wc = MegatronParamMapping._count_wildcard_groups(self.megatron_param)
        self._hf_wc = MegatronParamMapping._count_wildcard_groups(
            q
        )  # q/k/v share pattern structure here

    def resolve(self, captures: Tuple[str, ...]) -> MegatronParamMapping:
        # Reverse lookup starts from one flattened HF layer index. Reconstruct
        # Megatron's outer MTP depth and inner hybrid-layer index before
        # returning the concrete QKV mapping.
        treat_as_hf = (len(captures) == self._hf_wc) and (self._hf_wc != self._megatron_wc)
        if treat_as_hf:
            flat = int(captures[0])
            outer = flat // self._mtp_layers_per_block
            inner = flat % self._mtp_layers_per_block
            resolved_megatron = _replace_wildcards(
                self.megatron_param, (str(outer), str(inner), *captures[1:])
            )
            resolved_q = _replace_wildcards(self.hf_param["q"], captures)
            resolved_k = _replace_wildcards(self.hf_param["k"], captures)
            resolved_v = _replace_wildcards(self.hf_param["v"], captures)
            return QKVMapping(
                megatron_param=resolved_megatron, q=resolved_q, k=resolved_k, v=resolved_v
            )

        # Forward lookup starts from Megatron's (outer, inner) indices.
        if len(captures) < 2:
            raise ValueError(
                f"Expected (outer, inner) captures for MTP QKV mapping, got {captures}"
            )
        outer = int(captures[0])
        inner = int(captures[1])
        flat = outer * self._mtp_layers_per_block + inner

        resolved_megatron = _replace_wildcards(self.megatron_param, captures)
        resolved_q = _replace_wildcards(self.hf_param["q"], (str(flat),))
        resolved_k = _replace_wildcards(self.hf_param["k"], (str(flat),))
        resolved_v = _replace_wildcards(self.hf_param["v"], (str(flat),))

        return QKVMapping(
            megatron_param=resolved_megatron, q=resolved_q, k=resolved_k, v=resolved_v
        )

    # Required by ABC but never called at runtime: resolve() always returns a
    # QKVMapping, so hf_to_megatron/megatron_to_hf are invoked on that resolved
    # instance, not on this class.
    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: torch.nn.Module
    ) -> torch.Tensor:
        raise NotImplementedError

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[torch.nn.Module]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@MegatronModelBridge.register_bridge(
    source="NemotronHForCausalLM",
    target=HybridModel,
    provider=HybridModelProvider,
    model_type="nemotron_h",
)
class NemotronHBridge(MegatronModelBridge):
    """
    Megatron Bridge for Nemotron-H Causal LM.

    This bridge handles the conversion between HuggingFace NemotronHForCausalLM
    and Megatron-Core HybridModel formats, including weight mappings and
    configuration translation.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True)
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = BridgeHybridModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def text_only_pretrained(self, hf_pretrained: PreTrainedCausalLM) -> PreTrainedCausalLM:
        """Return the already-text-only checkpoint without changing its config or weights."""
        return hf_pretrained

    # Extend CONFIG_MAPPING with Nemotron-H/Mamba-specific fields
    # Common bidirectional config field name mapping: (hf_name, megatron_name)
    CONFIG_MAPPING = MegatronModelBridge.CONFIG_MAPPING + [
        # Mamba-specific fields
        ("mamba_head_dim", "mamba_head_dim"),
        ("mamba_num_heads", "mamba_num_heads"),
        ("chunk_size", "mamba_chunk_size"),
        ("n_groups", "mamba_num_groups"),
        ("ssm_state_size", "mamba_state_dim"),
        ("hybrid_override_pattern", "hybrid_layer_pattern"),
        ("residual_in_fp32", "fp32_residual_connection"),
        ("use_bias", "add_bias_linear"),
        ("layer_norm_epsilon", "layernorm_epsilon"),
        # MoE-specific fields (already in base but with different HF names)
        ("moe_shared_expert_intermediate_size", "moe_shared_expert_intermediate_size"),
    ]

    # Additional files to copy during HF export (reasoning parser utilities)
    ADDITIONAL_FILE_PATTERNS = ["*reasoning_parser.py"]

    _HF_LAYER_TYPE_TO_PATTERN = {
        "mamba": "M",
        "linear_attention": "M",
        "attention": "*",
        "full_attention": "*",
        "moe": "E",
        "mlp": "-",
    }

    @classmethod
    def _layer_types_to_hybrid_pattern(cls, layer_types, *, field_name: str) -> str:
        """Convert Transformers' list-based Nemotron-H layer schema to MCore symbols."""
        pattern = []
        for index, layer_type in enumerate(layer_types):
            try:
                pattern.append(cls._HF_LAYER_TYPE_TO_PATTERN[layer_type])
            except KeyError as exc:
                expected = ", ".join(sorted(cls._HF_LAYER_TYPE_TO_PATTERN))
                raise ValueError(
                    f"Unknown {field_name} entry at index {index}: {layer_type!r}. Expected one of: {expected}."
                ) from exc
        return "".join(pattern)

    @classmethod
    def _hf_hybrid_pattern(cls, hf_config) -> Optional[str]:
        """Return the main hybrid pattern from either supported HF config schema."""
        pattern = getattr(hf_config, "hybrid_override_pattern", None)
        if pattern:
            return pattern

        layer_types = getattr(hf_config, "layers_block_type", None)
        if layer_types:
            return cls._layer_types_to_hybrid_pattern(layer_types, field_name="layers_block_type")
        return None

    @classmethod
    def _hf_mtp_config(cls, hf_config) -> tuple[int, Optional[str]]:
        """Return the normalized MTP depth and block pattern from an HF config."""
        mtp_num_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
        if mtp_num_layers < 0:
            raise ValueError("num_nextn_predict_layers must be non-negative.")
        if mtp_num_layers == 0:
            return 0, None

        mtp_pattern = getattr(hf_config, "mtp_hybrid_override_pattern", None)
        if not mtp_pattern:
            layer_types = getattr(hf_config, "mtp_layers_block_type", None)
            if layer_types:
                mtp_pattern = cls._layer_types_to_hybrid_pattern(
                    layer_types, field_name="mtp_layers_block_type"
                )
        if not mtp_pattern:
            raise ValueError(
                "An HF config with num_nextn_predict_layers > 0 must define "
                "mtp_hybrid_override_pattern or mtp_layers_block_type."
            )
        return mtp_num_layers, mtp_pattern

    @classmethod
    def _hf_mtp_settings(cls, hf_config) -> dict[str, Any]:
        """Return the MTP fields that the model config and provider paths share."""
        mtp_num_layers, mtp_pattern = cls._hf_mtp_config(hf_config)
        settings = {
            "mtp_num_layers": mtp_num_layers,
            "mtp_hybrid_override_pattern": mtp_pattern,
            "mtp_use_repeated_layer": bool(
                mtp_num_layers and getattr(hf_config, "mtp_use_repeated_layer", True)
            ),
        }
        if mtp_num_layers:
            settings["mtp_loss_scaling_factor"] = getattr(hf_config, "mtp_loss_scaling_factor", 0.3)
        return settings

    @staticmethod
    def _keep_mtp_spec_in_bf16(hf_config, mtp_num_layers: int) -> bool:
        """Return the provider-only ``keep_mtp_spec_in_bf16`` flag, which nothing reads."""
        return bool(mtp_num_layers and getattr(hf_config, "keep_mtp_spec_in_bf16", True))

    @classmethod
    def _configure_mtp_provider(cls, provider: HybridModelProvider, hf_config) -> None:
        """Apply shared Nemotron-H MTP settings to a text or multimodal provider."""
        for name, value in cls._hf_mtp_settings(hf_config).items():
            setattr(provider, name, value)
        provider.keep_mtp_spec_in_bf16 = cls._keep_mtp_spec_in_bf16(
            hf_config, provider.mtp_num_layers
        )

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face Nemotron-H config to builder config kwargs."""
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            # Nemotron-H attention layers use no position embedding.
            position_embedding_type="none",
            activation_func=squared_relu,
            masked_softmax_fusion=True,
            apply_query_key_layer_scaling=False,
            persist_layer_norm=True,
            attention_softmax_in_fp32=False,
            first_last_layers_bf16=True,
            is_hybrid_model=True,
            moe_aux_loss_coeff=0.0001,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_router_dtype="fp32",
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_shared_expert_overlap=getattr(hf_config, "moe_shared_expert_overlap", True),
            # HybridModelProvider defaults that differ from the builder config's.
            normalization="RMSNorm",
            apply_rope_fusion=True,
            attention_backend=AttnBackend.flash,
            bias_dropout_fusion=True,
            **self._hf_mtp_settings(hf_config),
        )
        # HybridModelProvider defaults for fields that some Hugging Face configs omit.
        for name, default in (
            ("add_bias_linear", False),
            ("hidden_dropout", 0.0),
            ("attention_dropout", 0.0),
            ("num_attention_heads", 1),
        ):
            config_kwargs.setdefault(name, default)

        hybrid_pattern = self._hf_hybrid_pattern(hf_config)
        if hybrid_pattern is not None:
            config_kwargs["hybrid_layer_pattern"] = hybrid_pattern

        kv_channels = getattr(hf_config, "head_dim", None) or getattr(
            hf_config, "attention_head_dim", None
        )
        if kv_channels is not None:
            config_kwargs["kv_channels"] = kv_channels

        if hasattr(hf_config, "moe_latent_size"):
            config_kwargs["moe_latent_size"] = hf_config.moe_latent_size
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path."""
        provider_kwargs = self.hf_config_to_model_config_kwargs(hf_config)
        provider_kwargs["keep_mtp_spec_in_bf16"] = self._keep_mtp_spec_in_bf16(
            hf_config, provider_kwargs["mtp_num_layers"]
        )
        return provider_kwargs

    @classmethod
    def get_hf_tokenizer_kwargs(cls) -> dict:
        """Return HuggingFace tokenizer kwargs for Nemotron-H models.

        Nemotron-H models only provide a fast tokenizer (tokenizer.json),
        so use_fast=True is required.
        """
        return {"use_fast": True}

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        hf_cfg = super().megatron_to_hf_config(provider)
        # Clean hybrid_override_pattern: strip pipeline-parallel delimiters, split out
        # unified MTP patterns, and validate the HF-facing layer symbols.
        pattern = hf_cfg.pop("hybrid_override_pattern", None)
        if pattern:
            pattern_parts = pattern.split("/")
            clean_pattern = pattern_parts[0].replace("|", "")
            mtp_patterns = [part for part in pattern_parts[1:] if part]
            valid_chars = {"M", "E", "*", "-"}

            for pattern_name, layer_pattern in [("hybrid_override_pattern", clean_pattern)] + [
                ("mtp_hybrid_override_pattern", mtp_pattern) for mtp_pattern in mtp_patterns
            ]:
                unknown = set(layer_pattern) - valid_chars
                if unknown:
                    raise ValueError(
                        f"Unknown layer type characters in {pattern_name}: {unknown}. "
                        f"Expected: M (mamba), * (attention), E (moe), - (mlp)."
                    )

            if mtp_patterns:
                mtp_pattern = mtp_patterns[0]
                if any(pattern_part != mtp_pattern for pattern_part in mtp_patterns[1:]):
                    raise ValueError(
                        f"All MTP patterns in hybrid_override_pattern must be identical. Got: {mtp_patterns}."
                    )
                hf_cfg["mtp_hybrid_override_pattern"] = mtp_pattern

            hf_cfg["hybrid_override_pattern"] = clean_pattern

        # Add auto_map for custom config/modeling classes
        hf_cfg["auto_map"] = {
            "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
            "AutoModel": "modeling_nemotron_h.NemotronHForCausalLM",
            "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
        }

        # We choose not to set HF-only defaults such as:
        # Mamba: conv_kernel, time_step_*, mamba_hidden_act, etc.
        # MoE: n_shared_experts, norm_topk_prob

        # Megatron uses None="not set/disabled", but HF modeling code expects integers
        # and will crash on None (e.g. n_routed_experts // n_group → TypeError)
        mtp_num_layers = int(hf_cfg.get("num_nextn_predict_layers") or 0)
        # HF serializes the physical shared block; recipes set its training repetitions.
        if mtp_num_layers > 0 and getattr(provider, "mtp_use_repeated_layer", False):
            mtp_num_layers = 1
        hf_cfg["num_nextn_predict_layers"] = mtp_num_layers
        hf_cfg["n_group"] = hf_cfg.get("n_group") or 1
        hf_cfg["topk_group"] = hf_cfg.get("topk_group") or 1

        return hf_cfg

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        _, mtp_pattern = self._hf_mtp_config(self.hf_config)
        mtp_layers_per_block = len(mtp_pattern) if mtp_pattern else 0

        # Transformers saves the embedding under a singular name, while older
        # remote-code checkpoints use the plural name. Preserve the source key.
        embedding_hf_param = "backbone.embeddings.weight"
        hf_state = getattr(getattr(self, "hf_pretrained", None), "state", None)
        state_source = getattr(hf_state, "source", None)
        if state_source is not None:
            hf_keys = set(state_source.get_all_keys())
            if embedding_hf_param not in hf_keys and "backbone.embedding.weight" in hf_keys:
                embedding_hf_param = "backbone.embedding.weight"

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "decoder.layers.*.mlp.linear_fc1.weight": "backbone.layers.*.mixer.up_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "backbone.layers.*.mixer.down_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "backbone.layers.*.mixer.o_proj.weight",
            "decoder.final_norm.weight": "backbone.norm_f.weight",
            "decoder.final_layernorm.weight": "backbone.norm_f.weight",
            # Fused TE layer norm weights (when using TELayerNormColumnParallelLinear)
            # if the megatron key does not exist for a given layer it will be ignored,
            # so only one of these will be used per layer
            "decoder.layers.*.mixer.in_proj.layer_norm_weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "backbone.layers.*.norm.weight",
            # Separate Norm layer weights (when using Norm for quantization)
            # These are used when quantization spec uses Norm instead of TENorm
            "decoder.layers.*.norm.weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.input_layernorm.weight": "backbone.layers.*.norm.weight",
            # TODO (@maanug): need to find a way to prune the vocab padding from the vocab dimension for these params
            "embedding.word_embeddings.weight": embedding_hf_param,
            "output_layer.weight": "lm_head.weight",
            # MoE layers
            "decoder.layers.*.mlp.router.weight": "backbone.layers.*.mixer.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "backbone.layers.*.mixer.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.fc1_latent_proj.weight": "backbone.layers.*.mixer.fc1_latent_proj.weight",
            "decoder.layers.*.mlp.fc2_latent_proj.weight": "backbone.layers.*.mixer.fc2_latent_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc1.weight": "backbone.layers.*.mixer.shared_experts.up_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "backbone.layers.*.mixer.shared_experts.down_proj.weight",
            # GroupedMLP (moe_grouped_gemm=True): expert weights are stored as weight0, weight1, ...
            "decoder.layers.*.mlp.experts.linear_fc1.weight*": "backbone.layers.*.mixer.experts.*.up_proj.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "backbone.layers.*.mixer.experts.*.down_proj.weight",
            # SequentialMLP (moe_grouped_gemm=False): expert weights are stored per local_expert
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight": "backbone.layers.*.mixer.experts.*.up_proj.weight",
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "backbone.layers.*.mixer.experts.*.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # -----------------------------
        # MTP (flattened HF structure)
        # -----------------------------
        #
        # Megatron stores MTP in a nested form:
        #   mtp.layers.{outer}.mtp_model_layer.layers.{inner}.<...>
        # while HF stores it as:
        #   mtp.layers.{outer * L + inner}.<...>
        #
        # Additionally, some MTP parameters live outside mtp_model_layer in Megatron
        # (eh_proj/enorm/hnorm/final_layernorm) but attach to the first/last inner layer
        # in HF.
        if mtp_layers_per_block > 0:
            # First-inner (inner=0) params
            for p in ["eh_proj.weight", "enorm.weight", "hnorm.weight"]:
                mapping_list.append(
                    _MTPFlatteningMapping(
                        megatron_param=f"mtp.layers.*.{p}",
                        hf_param=f"mtp.layers.*.{p}",
                        mtp_layers_per_block=mtp_layers_per_block,
                        inner_override=0,
                    )
                )

            # Last-inner (inner=L-1) params
            mapping_list.append(
                _MTPFlatteningMapping(
                    megatron_param="mtp.layers.*.final_layernorm.weight",
                    hf_param="mtp.layers.*.final_layernorm.weight",
                    mtp_layers_per_block=mtp_layers_per_block,
                    inner_override=mtp_layers_per_block - 1,
                )
            )

            # Nested MTP transformer block params
            mtp_nested = {
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.linear_fc1.weight": "mtp.layers.*.mixer.up_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.linear_fc2.weight": "mtp.layers.*.mixer.down_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_proj.weight": "mtp.layers.*.mixer.o_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mixer.in_proj.layer_norm_weight": "mtp.layers.*.norm.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.linear_fc1.layer_norm_weight": "mtp.layers.*.norm.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.*.norm.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.router.weight": "mtp.layers.*.mixer.gate.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.router.expert_bias": "mtp.layers.*.mixer.gate.e_score_correction_bias",
                # GroupedMLP (moe_grouped_gemm=True): expert weights are stored as weight0, weight1, ...
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.experts.linear_fc1.weight*": "mtp.layers.*.mixer.experts.*.up_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.experts.linear_fc2.weight*": "mtp.layers.*.mixer.experts.*.down_proj.weight",
                # SequentialMLP (moe_grouped_gemm=False): expert weights are stored per local_expert
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.experts.local_experts.*.linear_fc1.weight": "mtp.layers.*.mixer.experts.*.up_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "mtp.layers.*.mixer.experts.*.down_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.fc1_latent_proj.weight": "mtp.layers.*.mixer.fc1_latent_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.fc2_latent_proj.weight": "mtp.layers.*.mixer.fc2_latent_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.shared_experts.linear_fc1.weight": "mtp.layers.*.mixer.shared_experts.up_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.shared_experts.linear_fc2.weight": "mtp.layers.*.mixer.shared_experts.down_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.pre_mlp_layernorm.weight": "mtp.layers.*.norm.weight",
            }

            for megatron_param, hf_param in mtp_nested.items():
                mapping_list.append(
                    _MTPFlatteningMapping(
                        megatron_param=megatron_param,
                        hf_param=hf_param,
                        mtp_layers_per_block=mtp_layers_per_block,
                        inner_override=None,
                    )
                )
        # Handling Mamba Mixer submodules separately for more clarity
        # Special Handling for InProj and Conv1d due to specific TP logic
        for mixer_sub_module in ["A_log", "D", "dt_bias", "norm.weight"]:
            mapping_list.extend(
                [
                    ColumnParallelMapping(
                        megatron_param=rf"decoder.layers.*.mixer.{mixer_sub_module}",
                        hf_param=rf"backbone.layers.*.mixer.{mixer_sub_module}",
                    )
                ]
            )
        mapping_list.extend(
            [
                RowParallelMapping(
                    megatron_param="decoder.layers.*.mixer.out_proj.weight",
                    hf_param="backbone.layers.*.mixer.out_proj.weight",
                )
            ]
        )
        mapping_list.extend(
            [
                MambaInProjMapping(
                    megatron_param="decoder.layers.*.mixer.in_proj.weight",
                    hf_param="backbone.layers.*.mixer.in_proj.weight",
                )
            ]
        )
        for megatron_conv1d_param, hf_conv1d_param in [
            ("conv1d_weight", "conv1d.weight"),
            ("conv1d_bias", "conv1d.bias"),
            ("conv1d.weight", "conv1d.weight"),
            ("conv1d.bias", "conv1d.bias"),
        ]:
            mapping_list.extend(
                [
                    MambaConv1dMapping(
                        megatron_param=rf"decoder.layers.*.mixer.{megatron_conv1d_param}",
                        hf_param=rf"backbone.layers.*.mixer.{hf_conv1d_param}",
                    )
                ]
            )
        # Add special mappings that require parameter concatenation/transformation, pruning, etc.
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="backbone.layers.*.mixer.q_proj.weight",
                    k="backbone.layers.*.mixer.k_proj.weight",
                    v="backbone.layers.*.mixer.v_proj.weight",
                )
            ]
        )

        if mtp_layers_per_block > 0:
            mapping_list.extend(
                [
                    _MTPFlatteningQKVMapping(
                        megatron_param="mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_qkv.weight",
                        q="mtp.layers.*.mixer.q_proj.weight",
                        k="mtp.layers.*.mixer.k_proj.weight",
                        v="mtp.layers.*.mixer.v_proj.weight",
                        mtp_layers_per_block=mtp_layers_per_block,
                    )
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
