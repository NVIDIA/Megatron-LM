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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/nemotron_omni/nemotron_omni_config.py
# Source license: Apache-2.0.

"""Builder-backed model config for Nemotron 3.5 Super VL.

The language half is the Nemotron-H hybrid decoder, configured with the same fields as
``BridgeHybridModelConfig``. The media settings live on the outer config;
``Nemotron35SuperVLModelBuilder`` combines the decoder with the RADIO vision encoder and the
vision projector. Audio models are outside this port's scope.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig


def _validate_super_vl_settings(settings: Any) -> None:
    """Validate the media settings that the checkpoint configuration must supply.

    Args:
        settings: Super VL model config.

    Raises:
        ValueError: If dynamic resolution is disabled or the image token ID is not positive.
    """
    if settings.dynamic_resolution is not True:
        raise ValueError("Nemotron 3.5 Super VL only supports dynamic_resolution=True.")
    if settings.image_token_index is None or settings.image_token_index <= 0:
        raise ValueError(
            "Nemotron 3.5 Super VL requires a positive image_token_index from the checkpoint configuration; "
            f"got {settings.image_token_index}. Construct the config through AutoBridge or set it explicitly."
        )


@dataclass(kw_only=True)
class Nemotron35SuperVLModelConfig(BridgeHybridModelConfig):
    """Super VL language decoder, RADIO encoder and projection configuration."""

    builder: ClassVar[str] = (
        "megatron.bridge.models.nemotron_super_vl.builder.Nemotron35SuperVLModelBuilder"
    )

    image_token_index: int = 0
    """Token ID that each projected image feature replaces."""

    vision_proj_ffn_hidden_size: int = 20480
    vision_class_token_len: int | None = None

    # The public processor emits variable-resolution images, which Bridge pre-patchifies for RADIO.
    dynamic_resolution: Literal[True] = True
    use_vision_backbone_fp8_arch: bool = True
    radio_force_eval_mode: bool = True
    radio_force_cpe_eval_mode: bool = True
    radio_interpolate_only_cpe: bool = False
    radio_cpe_aspect_ratio_select: bool = False
    radio_disable_cpe: bool = False

    recompute_vision: bool = False
    vision_recompute_granularity: Literal["full", "selective"] | None = None
    vision_recompute_modules: list[str] | None = None
    vision_recompute_method: Literal["uniform", "block"] | None = None
    vision_recompute_num_layers: int | None = None

    temporal_patch_dim: int = 1
    separate_video_embedder: bool = False
    temporal_ckpt_compat: bool = False
    vision_final_layernorm: bool = False

    vision_dp_over_cp: bool = True
    """Shard images across the context-parallel group instead of encoding them on every CP rank."""

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def validate(self) -> None:
        """Check media and hybrid settings without changing the config."""
        super().validate()
        _validate_super_vl_settings(self)


__all__ = ["Nemotron35SuperVLModelConfig"]
