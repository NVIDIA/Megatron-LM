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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/nemotron_omni/nemotron_omni_builder.py
# Source license: Apache-2.0.

"""Super VL builder: derive RADIO and projector configs from the language config."""

import copy
from typing import Any

from megatron.bridge.models.hybrid.hybrid_builder import BridgeHybridModelBuilder
from megatron.bridge.models.nemotron_super_vl.config import Nemotron35SuperVLModelConfig
from megatron.bridge.models.nemotron_super_vl.modeling_super_vl import Nemotron35SuperVLModel
from megatron.core.activations import fast_gelu, squared_relu
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig


def _get_transformer_engine_projection_submodules():
    """Build the standard dense TE MLP submodules used by media projectors."""
    return copy.deepcopy(get_submodules(get_mlp_module_spec(use_te=True)))


def _make_radio_vision_config(settings: Any, language_cfg: TransformerConfig) -> TransformerConfig:
    """Build the RADIO ViT-H vision encoder config from a copy of the language config.

    Args:
        settings: Nemotron Omni model config or provider carrying the vision settings.
        language_cfg: Finalized language transformer config; it is not modified.

    Returns:
        The vision encoder config.

    Raises:
        ValueError: If the vision recompute settings are inconsistent.
    """
    if settings.recompute_vision and settings.radio_force_eval_mode:
        raise ValueError("Vision recompute requires radio_force_eval_mode=False.")
    vision_cfg = copy.deepcopy(language_cfg)
    if not settings.use_vision_backbone_fp8_arch:
        vision_cfg.fp8 = None
        vision_cfg.fp8_param = False
    vision_cfg.sequence_parallel = False
    vision_cfg.context_parallel_size = 1
    vision_cfg.tp_comm_overlap = False
    vision_cfg.mtp_num_layers = None
    vision_cfg.num_layers = 32
    vision_cfg.num_attention_heads = 16
    vision_cfg.add_bias_linear = True
    vision_cfg.add_qkv_bias = True
    vision_cfg.hidden_size = 1280
    vision_cfg.ffn_hidden_size = 5120
    vision_cfg.gated_linear_unit = False
    vision_cfg.activation_func = fast_gelu
    vision_cfg.kv_channels = 80
    vision_cfg.num_query_groups = 16
    vision_cfg.layernorm_zero_centered_gamma = False
    vision_cfg.apply_query_key_layer_scaling = False
    vision_cfg.attention_softmax_in_fp32 = True
    vision_cfg.normalization = "LayerNorm"
    vision_cfg.qk_layernorm = False
    vision_cfg.layernorm_epsilon = 1e-6
    if settings.recompute_vision:
        granularity = (
            settings.vision_recompute_granularity
            if settings.vision_recompute_granularity is not None
            else vision_cfg.recompute_granularity
        )
        if granularity is None:
            raise ValueError("Vision recompute requires an effective recompute granularity.")
        if granularity not in {"full", "selective"}:
            raise ValueError("Vision recompute granularity must be 'full' or 'selective'.")
        vision_cfg.recompute_granularity = granularity

        if granularity == "selective":
            if (
                settings.vision_recompute_method is not None
                or settings.vision_recompute_num_layers is not None
            ):
                raise ValueError("Selective vision recompute does not use a method or layer count.")
            vision_cfg.recompute_method = None
            vision_cfg.recompute_num_layers = None
            if settings.vision_recompute_modules is not None:
                if not settings.vision_recompute_modules:
                    raise ValueError("Selective vision recompute modules must not be empty.")
                vision_cfg.recompute_modules = list(settings.vision_recompute_modules)
        else:
            if settings.vision_recompute_modules is not None:
                raise ValueError("Full vision recompute does not use selective recompute modules.")
            method = (
                settings.vision_recompute_method
                if settings.vision_recompute_method is not None
                else vision_cfg.recompute_method
            )
            num_layers = (
                settings.vision_recompute_num_layers
                if settings.vision_recompute_num_layers is not None
                else vision_cfg.recompute_num_layers
            )
            if method is None:
                raise ValueError("Full vision recompute requires a recompute method.")
            if method not in {"uniform", "block"}:
                raise ValueError("Full vision recompute method must be 'uniform' or 'block'.")
            if num_layers is None:
                raise ValueError("Full vision recompute requires a layer count.")
            if (
                not isinstance(num_layers, int)
                or isinstance(num_layers, bool)
                or not 1 <= num_layers <= vision_cfg.num_layers
            ):
                raise ValueError(
                    f"Full vision recompute layer count must be an integer between 1 and {vision_cfg.num_layers}."
                )
            vision_cfg.recompute_method = method
            vision_cfg.recompute_num_layers = num_layers
    else:
        vision_cfg.recompute_granularity = None
        vision_cfg.recompute_method = None
        vision_cfg.recompute_num_layers = None
    if settings.vision_class_token_len is not None:
        vision_cfg.class_token_len = settings.vision_class_token_len
    return vision_cfg


def _make_radio_projection_config(
    settings: Any, language_cfg: TransformerConfig
) -> TransformerConfig:
    """Build the vision projection MLP config from a copy of the language config.

    Args:
        settings: Nemotron Omni model config or provider carrying the projector width.
        language_cfg: Finalized language transformer config; it is not modified.

    Returns:
        The vision projection config.
    """
    vision_proj_cfg = copy.deepcopy(language_cfg)
    vision_proj_cfg.sequence_parallel = False
    vision_proj_cfg.context_parallel_size = 1
    vision_proj_cfg.tp_comm_overlap = False
    vision_proj_cfg.recompute_granularity = None
    vision_proj_cfg.recompute_method = None
    vision_proj_cfg.recompute_num_layers = None
    vision_proj_cfg.ffn_hidden_size = settings.vision_proj_ffn_hidden_size
    vision_proj_cfg.bias_activation_fusion = False
    return vision_proj_cfg


def _make_omni_vision_config(settings: Any, language_cfg: TransformerConfig) -> TransformerConfig:
    """Build the RADIO config, pinned to PP=1 (Omni training uses PP>1 on the LLM).

    The vision encoder is always co-located with the first pipeline stage.

    Args:
        settings: Nemotron Omni model config or provider carrying the vision settings.
        language_cfg: Finalized language transformer config; it is not modified.

    Returns:
        The vision encoder config.
    """
    vision_cfg = _make_radio_vision_config(settings, language_cfg)
    vision_cfg.pipeline_model_parallel_size = 1
    vision_cfg.virtual_pipeline_model_parallel_size = None
    vision_cfg.num_layers_in_first_pipeline_stage = None
    vision_cfg.num_layers_in_last_pipeline_stage = None
    vision_cfg.pipeline_model_parallel_layout = None
    vision_cfg.account_for_embedding_in_pipeline_split = False
    vision_cfg.account_for_loss_in_pipeline_split = False
    if settings.vision_final_layernorm:
        # MCore places a final norm in the vision TransformerBlock when
        # mtp_num_layers is non-None. Nemotron 3.5 Super VL was trained
        # with that layout and its HF projector applies the same norm.
        vision_cfg.mtp_num_layers = 1
    return vision_cfg


def _make_omni_vision_projection_config(
    settings: Any, language_cfg: TransformerConfig
) -> TransformerConfig:
    """Build the vision projection MLP config.

    Nemotron Omni applies squared ReLU in the vision projection MLP,
    matching the HF checkpoint and vLLM implementation. The projector is
    pinned to PP=1 (see :func:`_make_omni_vision_config`).

    Args:
        settings: Nemotron Omni model config or provider carrying the projector width.
        language_cfg: Finalized language transformer config; it is not modified.

    Returns:
        The vision projection config.
    """
    vision_proj_cfg = _make_radio_projection_config(settings, language_cfg)
    vision_proj_cfg.activation_func = squared_relu
    vision_proj_cfg.pipeline_model_parallel_size = 1
    vision_proj_cfg.virtual_pipeline_model_parallel_size = None
    vision_proj_cfg.num_layers_in_first_pipeline_stage = None
    vision_proj_cfg.num_layers_in_last_pipeline_stage = None
    vision_proj_cfg.pipeline_model_parallel_layout = None
    vision_proj_cfg.account_for_embedding_in_pipeline_split = False
    vision_proj_cfg.account_for_loss_in_pipeline_split = False
    return vision_proj_cfg


class Nemotron35SuperVLModelBuilder(BridgeHybridModelBuilder):
    """Combine the Nemotron-H hybrid language model with the RADIO vision encoder."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Nemotron35SuperVLModel:
        """Build one Nemotron Omni pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Optional virtual pipeline stage index.

        Returns:
            A Nemotron Omni model stage. The media encoders and projectors live on the first
            pipeline stage.

        Raises:
            TypeError: If the builder was given another model config.
        """
        model_config = self._model_config
        if not isinstance(model_config, Nemotron35SuperVLModelConfig):
            raise TypeError(
                f"Expected Nemotron35SuperVLModelConfig, got {type(model_config).__name__}."
            )

        vp_size = model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_last_stage(pg_collection.pp)

        language_cfg = model_config.transformer
        add_encoder = (
            is_pp_first_stage(pg_collection.pp)
            if model_config.pipeline_model_parallel_size > 1
            else True
        )
        model = Nemotron35SuperVLModel(
            language_transformer_config=language_cfg,
            language_transformer_layer_spec=self._resolve_hybrid_stack_spec(),
            language_vocab_size=model_config.vocab_size,
            language_max_sequence_length=model_config.seq_length,
            vision_transformer_config=_make_omni_vision_config(model_config, language_cfg),
            vision_transformer_layer_spec=get_vit_layer_with_transformer_engine_spec(),
            vision_projection_config=_make_omni_vision_projection_config(
                model_config, language_cfg
            ),
            vision_projection_layer_spec=_get_transformer_engine_projection_submodules(),
            image_token_index=model_config.image_token_index,
            parallel_output=model_config.parallel_output,
            share_embeddings_and_output_weights=model_config.share_embeddings_and_output_weights,
            language_position_embedding_type=model_config.position_embedding_type,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=True,
            hybrid_layer_pattern=model_config.hybrid_layer_pattern,
            dynamic_resolution=model_config.dynamic_resolution,
            vision_class_token_len=model_config.vision_class_token_len or 10,
            radio_force_eval_mode=model_config.radio_force_eval_mode,
            radio_force_cpe_eval_mode=model_config.radio_force_cpe_eval_mode,
            radio_interpolate_only_cpe=model_config.radio_interpolate_only_cpe,
            radio_cpe_aspect_ratio_select=model_config.radio_cpe_aspect_ratio_select,
            radio_disable_cpe=model_config.radio_disable_cpe,
            temporal_patch_dim=model_config.temporal_patch_dim,
            separate_video_embedder=model_config.separate_video_embedder,
            temporal_ckpt_compat=model_config.temporal_ckpt_compat,
            vision_dp_over_cp=model_config.vision_dp_over_cp,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            model_config=model_config,
        )
        # Conversion may build without Float16Module; keep the encoder in the model's dtype.
        for encoder in (model.vision_model,):
            if encoder is not None:
                encoder.to(dtype=model_config.params_dtype)

        if any(
            (
                model_config.freeze_language_model,
                model_config.freeze_vision_model,
                model_config.freeze_vision_projection,
            )
        ):
            model.freeze(
                freeze_language_model=model_config.freeze_language_model,
                freeze_vision_model=model_config.freeze_vision_model,
                freeze_vision_projection=model_config.freeze_vision_projection,
            )
        return model


__all__ = ["Nemotron35SuperVLModelBuilder"]
