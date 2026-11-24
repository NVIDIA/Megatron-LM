# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen/modeling_qwen35/hybrid.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen3.5 composition of Core's separate attention and feed-forward layers."""

from copy import copy, deepcopy
from typing import Any

from transformers import PretrainedConfig

from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
from megatron.bridge.models.conversion.transformers_compat import full_attention_interval_from_hf
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


class Qwen35HybridAttentionLayer(TransformerLayer):
    """Keep TE's class-level graph wrappers separate from the MoE/MLP subclasses.

    Capturing TransformerLayer and its subclasses together can retain an
    inherited autocast wrapper on uncaptured MTP layers after graph creation.
    A sibling attention class preserves those layers' quantization contexts.
    """


def qwen35_pipeline_pattern(config: TransformerConfig, pattern: str) -> str:
    """Assign Hybrid positions to PP/VPP stages without splitting HF blocks."""
    pp_size = config.pipeline_model_parallel_size
    vp_size = config.virtual_pipeline_model_parallel_size
    if pp_size * (vp_size or 1) == 1:
        return pattern

    layout = config.pipeline_model_parallel_layout
    segments, offset = [], 0
    # Hybrid orders segments by vp_stage * pp_size + pp_rank, just like GPT.
    for vp_stage in range(vp_size or 1):
        for pp_rank in range(pp_size):
            if layout is not None:
                count = layout.layout[pp_rank][vp_stage].count(LayerType.decoder)
            else:
                count = get_num_layers_to_build(config, vp_stage if vp_size else None, pp_rank)
            width = count
            if width % 2:
                raise ValueError(
                    "Qwen3.5 pipeline stages must contain whole mixer/feed-forward pairs."
                )
            segments.append(pattern[offset : offset + width])
            offset += width
    if offset != len(pattern):
        raise ValueError(
            f"Pipeline stages hold {offset} Hybrid positions, but the model has {len(pattern)}."
        )
    return "|".join(segments)


def qwen35_hybrid_pattern(text_config: PretrainedConfig) -> str:
    """Expand each HF block into a GDN/attention position followed by its MLP."""
    count = text_config.num_hidden_layers
    schedule = full_attention_interval_from_hf(text_config)
    if isinstance(schedule, int):
        if schedule < 1:
            raise ValueError("full_attention_interval must be positive.")
        schedule = [int((index + 1) % schedule != 0) for index in range(count)]
    if len(schedule) != count:
        raise ValueError("The attention schedule must contain one entry per HF layer.")

    feed_forward = "E" if getattr(text_config, "num_experts", 0) else "-"
    return "".join(("G" if linear else "*") + feed_forward for linear in schedule)


def qwen35_hybrid_stack_spec() -> ModuleSpec:
    """Keep Core's GDN and MLP kernels while enabling Qwen's per-head Q/K norms."""
    spec = deepcopy(hybrid_stack_spec)
    spec.submodules.gdn_layer.module = Qwen35HybridAttentionLayer
    spec.submodules.attention_layer.module = Qwen35HybridAttentionLayer
    attention = spec.submodules.attention_layer.submodules.self_attention.submodules
    attention.q_layernorm = TESpecProvider().layer_norm(for_qk=True)
    attention.k_layernorm = TESpecProvider().layer_norm(for_qk=True)
    return spec


def configure_qwen35_hybrid(config_kwargs: dict[str, Any], text_config: PretrainedConfig) -> None:
    """Convert the language configuration from HF block coordinates to Hybrid positions."""
    pattern = qwen35_hybrid_pattern(text_config)
    config_kwargs.update(
        num_layers=len(pattern),
        hybrid_layer_pattern=pattern,
        hybrid_stack_spec=qwen35_hybrid_stack_spec,
        is_hybrid_model=True,
        linear_attention_freq=[int(symbol == "G") for symbol in pattern],
        moe_layer_freq=[int(symbol == "E") for symbol in pattern],
        mtp_hybrid_override_pattern="*E" if "E" in pattern else "*-",
    )
    config_kwargs.pop("rope_scaling", None)
    config_kwargs.pop("rope_scaling_factor", None)


def qwen35_hf_layer_config(pattern: str) -> dict[str, Any]:
    """Recover HF block counts and attention order from a finalized Hybrid pattern."""
    main_pattern = pattern.split("/")[0].replace("|", "")
    return {
        "num_hidden_layers": len(main_pattern) // 2,
        "layer_types": [
            "linear_attention" if symbol == "G" else "full_attention"
            for symbol in main_pattern[::2]
        ],
    }


def qwen35_hybrid_mappings(
    mappings: list[MegatronParamMapping], *, num_layers: int
) -> list[MegatronParamMapping]:
    """Translate GPT block templates to split Hybrid positions, retaining expert wildcards."""
    result = []
    for mapping in mappings:
        name = mapping.megatron_param
        if "decoder.layers.*." not in name:
            renamed = copy(mapping)
            renamed.megatron_param = name.replace("decoder.final_layernorm.", "decoder.final_norm.")
            for attribute in ("mtp_model_layer", "transformer_layer"):
                marker = f".{attribute}."
                if marker in renamed.megatron_param:
                    suffix = renamed.megatron_param.split(marker, 1)[1]
                    inner = int(suffix.startswith(("mlp.", "pre_mlp_layernorm.")))
                    renamed.megatron_param = renamed.megatron_param.replace(
                        marker, f".{attribute}.layers.{inner}."
                    )
            result.append(
                renamed.resolve(tuple("*" for _ in range(renamed.megatron_param.count("*"))))
            )
            continue

        suffix = name.split("decoder.layers.*.", 1)[1]
        inner = int(suffix.startswith(("mlp.", "pre_mlp_layernorm.")))
        for block in range(num_layers):
            renamed = copy(mapping)
            renamed.megatron_param = name.replace(
                "decoder.layers.*.", f"decoder.layers.{2 * block + inner}."
            )

            def replace_hf_layer(value: str) -> str:
                return value.replace("layers.*.", f"layers.{block}.", 1)

            if isinstance(mapping.hf_param, dict):
                renamed.hf_param = {
                    key: replace_hf_layer(value) for key, value in mapping.hf_param.items()
                }
            else:
                renamed.hf_param = replace_hf_layer(mapping.hf_param)
            # Reconstruct through the mapping's own factory to retain layout
            # options and rebuild nested TP mappings with the new parameter name.
            result.append(
                renamed.resolve(tuple("*" for _ in range(renamed.megatron_param.count("*"))))
            )
    return result
