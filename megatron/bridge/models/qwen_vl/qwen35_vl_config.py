# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/qwen35_vl_config.py
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

"""Builder-backed model config for Qwen3.5 and Qwen3.6 vision-language models.

The language half is the Qwen3.5 hybrid Gated DeltaNet + Gated Attention decoder, configured
with the same Hybrid positions as the text bridges' ``Qwen35HybridModelConfig``. The vision-language settings
live on the outer config; ``Qwen35VLModelBuilder`` combines the decoder with the Qwen3-VL vision
encoder.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from megatron.bridge.models.qwen.modeling_qwen35.model_config import Qwen35HybridModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig

_QWEN_VISUAL_ENCODER_KEY = "qwen_visual"
_IMAGES_MODALITY_KEY = "images"


@dataclass
class Qwen35VLTransformerConfig(TransformerConfig):
    """Qwen3.5-VL fields not represented by MCore's base config.

    ``Qwen3VLModel`` and its ``Qwen3VLSelfAttention`` layers read them from the language
    transformer config they are built with.
    """

    apply_rotary_pos_emb_in_fp32: bool = False
    spatial_merge_size: int = 2
    vision_dp_when_cp: bool = False


@dataclass(kw_only=True)
class Qwen35VLModelConfig(Qwen35HybridModelConfig):
    """Builder config for Qwen3.5 and Qwen3.6 vision-language models, dense and MoE.

    ``vision_config`` holds the Hugging Face vision config, as on the providers. The token IDs,
    vision geometry and vision-encoder execution settings are the provider fields of the same
    names. Language layer counts and pipeline layouts use Hybrid positions: two positions
    per HF block, with stage boundaries between complete mixer/feed-forward pairs.
    """

    builder: ClassVar[str] = "megatron.bridge.models.qwen_vl.qwen35_vl_builder.Qwen35VLModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = Qwen35VLTransformerConfig
    modality_keys: ClassVar[dict[str, str]] = {_IMAGES_MODALITY_KEY: _QWEN_VISUAL_ENCODER_KEY}

    position_embedding_type: Literal["learned_absolute", "rope", "mrope", "yarn", "none"] = "mrope"
    scatter_embedding_sequence_parallel: bool = False
    transformer_layer_spec: Any = None

    vision_config: Any = None
    """Hugging Face ``Qwen3_5VisionConfig`` or ``Qwen3_5MoeVisionConfig``."""

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    temporal_patch_size: int = 2
    patch_size: int = 16
    language_max_sequence_length: int = 2048

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    use_hf_vision_model: bool = False
    # "inherit" preserves legacy decoder-policy inheritance; None disables vision recompute.
    vision_recompute_granularity: Literal["inherit", "full", "selective"] | None = "inherit"
    vision_recompute_method: Literal["uniform", "block"] | None = None
    vision_recompute_num_layers: int | None = None
    vision_recompute_modules: list[str] | None = None
    vision_cuda_graph_impl: str = "none"
    vision_cuda_graph_scope: list[str] = field(default_factory=list)
    max_vision_cuda_graph_seq_length: int | None = None

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return modality token IDs."""
        return {_IMAGES_MODALITY_KEY: self.image_token_id}

    def finalize(self) -> None:
        """Compute the loss per token under context parallelism, as Qwen3-VL requires, then validate."""
        if (self.context_parallel_size or 1) > 1:
            self.calculate_per_token_loss = True
        super().finalize()


__all__ = ["Qwen35VLModelConfig", "Qwen35VLTransformerConfig"]
