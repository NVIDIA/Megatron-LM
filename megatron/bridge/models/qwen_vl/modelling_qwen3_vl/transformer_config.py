# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/transformer_config.py
# Source license: Apache-2.0.
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
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from megatron.bridge.utils.cuda_graph import clear_cuda_graph_modules, set_cuda_graph_modules
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Qwen3VLTransformerConfig(TransformerConfig):
    """Configuration for Qwen3-VL transformer with vision and language components."""

    vocab_size: int = 64000
    make_vocab_size_divisible_by: int = 128
    should_pad_vocab: bool = False
    language_max_sequence_length: int = 4096

    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 4096

    apply_rotary_pos_emb_in_fp32: bool = False
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    fp16_lm_cross_entropy: bool = False
    logit_dtype: torch.dtype | None = None
    share_embeddings_and_output_weights: bool = False
    rotary_percent: float = 1.0
    rotary_base: float = 10000

    # Multimodal rope section for [temporal, height, width] dimensions
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    apply_rope_fusion: bool = False

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    hf_text_config: Optional[Qwen3VLTextConfig] = None
    vision_dp_when_cp: bool = False
    use_hf_vision_model: bool = False
    # Maximum sequence length for vision encoder CUDA graphs.
    max_vision_cuda_graph_seq_length: Optional[int] = None


def _apply_vision_recompute_config(
    config: Qwen3VLTransformerConfig, megatron_config: TransformerConfig
) -> None:
    """Resolve an explicit vision policy without changing the decoder configuration."""
    granularity = getattr(megatron_config, "vision_recompute_granularity", "inherit")
    method = getattr(megatron_config, "vision_recompute_method", None)
    num_layers = getattr(megatron_config, "vision_recompute_num_layers", None)
    modules = getattr(megatron_config, "vision_recompute_modules", None)

    if granularity == "inherit":
        if any(value is not None for value in (method, num_layers, modules)):
            raise ValueError(
                "Set vision_recompute_granularity explicitly when specifying vision recompute options."
            )
        # Preserve the legacy policy, including the vision config's default selective
        # modules. Decoder-only modules (e.g. GDN and MoE) must not leak into vision.
        config.recompute_granularity = megatron_config.recompute_granularity
        config.recompute_method = megatron_config.recompute_method
        config.recompute_num_layers = megatron_config.recompute_num_layers
        return

    if granularity is None:
        if any(value is not None for value in (method, num_layers, modules)):
            raise ValueError(
                "Disabled vision recomputation cannot specify method, num_layers, or modules."
            )
    elif granularity == "full":
        if method not in ("uniform", "block"):
            raise ValueError(
                "Full vision recomputation requires vision_recompute_method='uniform' or 'block'."
            )
        if type(num_layers) is not int or not 1 <= num_layers <= config.num_layers:
            raise ValueError(
                "vision_recompute_num_layers must be an integer between 1 and the vision depth."
            )
        if modules is not None:
            raise ValueError(
                "vision_recompute_modules is only supported for selective vision recomputation."
            )
    elif granularity == "selective":
        if method is not None or num_layers is not None:
            raise ValueError("Selective vision recomputation cannot specify method or num_layers.")
        # The TE ViT spec fuses layernorm into QKV/FC1, and has no GDN or MoE.
        if not modules or any(module not in ("core_attn", "mlp") for module in modules):
            raise ValueError(
                "vision_recompute_modules must be a non-empty list containing only 'core_attn' and 'mlp'."
            )
    else:
        raise ValueError(
            "vision_recompute_granularity must be 'inherit', None, 'full', or 'selective'."
        )

    config.recompute_granularity = granularity
    config.recompute_method = method
    config.recompute_num_layers = num_layers
    config.recompute_modules = list(modules) if modules is not None else []


def get_vision_model_config(hf_config, megatron_config=None):
    """
    Get the vision model config for Qwen3VL vision model.
    """
    # init config from scratch to avoid deepcopy of parallel_state
    config = Qwen3VLTransformerConfig(
        num_layers=hf_config.depth,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_heads,
        ffn_hidden_size=hf_config.intermediate_size,
        add_bias_linear=True,
        add_qkv_bias=True,
    )

    # apply text model config to vision model config
    _apply_vision_recompute_config(config, megatron_config)
    config.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
    config.enable_cuda_graph = megatron_config.enable_cuda_graph
    config.cuda_graph_use_single_mempool = megatron_config.cuda_graph_use_single_mempool
    config.cuda_graph_retain_backward_graph = megatron_config.cuda_graph_retain_backward_graph
    config.cuda_graph_warmup_steps = megatron_config.cuda_graph_warmup_steps
    config.external_cuda_graph = megatron_config.external_cuda_graph

    config.num_moe_experts = None
    config.expert_model_parallel_size = 1
    config.moe_ffn_hidden_size = None

    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.layernorm_epsilon = 1e-6
    config.apply_rotary_pos_emb_in_fp32 = True

    config.patch_size = hf_config.patch_size
    config.temporal_patch_size = hf_config.temporal_patch_size
    config.in_channels = hf_config.in_channels
    config.spatial_merge_size = hf_config.spatial_merge_size
    config.num_position_embeddings = hf_config.num_position_embeddings
    config.out_hidden_size = hf_config.out_hidden_size
    # ``deepstack_visual_indexes`` is a Qwen3-VL-only field. Qwen3.5/3.6-VL HF
    # configs do not declare it in the schema, so the field is sometimes
    # absent — e.g. after a YAML round-trip through
    # ``PretrainedConfig.to_dict()`` / ``from_dict()`` which only persists
    # declared fields. ``Qwen3VLModel.__init__`` already reads it
    # defensively with the same default (``modelling_qwen3_vl/model.py:193``).
    config.deepstack_visual_indexes = deepcopy(getattr(hf_config, "deepstack_visual_indexes", []))

    config.apply_rope_fusion = False
    config.gated_linear_unit = False  # no gated
    config.activation_func = partial(F.gelu, approximate="tanh")  # hidden_act
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads  # no GQA
    config.layernorm_zero_centered_gamma = False  # False
    config.apply_query_key_layer_scaling = False  # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = False  # no swiglu, set false
    config.bias_dropout_fusion = False  # no dropout, set false
    config.attention_softmax_in_fp32 = True  # use True
    config.normalization = "LayerNorm"

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.context_parallel_size = 1
    config.pipeline_model_parallel_size = 1
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = None
    config.virtual_pipeline_model_parallel_size = 1
    config.pipeline_model_parallel_layout = None
    config.account_for_embedding_in_pipeline_split = None
    config.account_for_loss_in_pipeline_split = None

    # Vision encoder CUDA graph settings
    # Check megatron_config (the provider / language config) for vision-specific CUDA graph
    # settings. The provider stores these as vision_cuda_graph_impl / vision_cuda_graph_scope.
    # If present, use them; otherwise default to "none" for backward compatibility.
    if (
        megatron_config is not None
        and hasattr(megatron_config, "vision_cuda_graph_impl")
        and megatron_config.vision_cuda_graph_impl != "none"
    ):
        config.cuda_graph_impl = megatron_config.vision_cuda_graph_impl
        if (
            hasattr(megatron_config, "vision_cuda_graph_scope")
            and megatron_config.vision_cuda_graph_scope
        ):
            set_cuda_graph_modules(config, megatron_config.vision_cuda_graph_scope)
        else:
            clear_cuda_graph_modules(config)
    else:
        config.cuda_graph_impl = "none"
        clear_cuda_graph_modules(config)
    if config.recompute_granularity == "full" and config.cuda_graph_impl not in (
        "none",
        "full_iteration",
    ):
        raise ValueError(
            "Full vision recomputation is incompatible with per-layer vision CUDA graphs. "
            "Set vision_cuda_graph_impl='none' to disable vision graph capture."
        )
    # Propagate max vision CUDA graph sequence length from provider
    if megatron_config is not None and hasattr(megatron_config, "max_vision_cuda_graph_seq_length"):
        config.max_vision_cuda_graph_seq_length = megatron_config.max_vision_cuda_graph_seq_length

    if megatron_config is not None and hasattr(megatron_config, "use_cpu_initialization"):
        config.use_cpu_initialization = megatron_config.use_cpu_initialization

    return config
