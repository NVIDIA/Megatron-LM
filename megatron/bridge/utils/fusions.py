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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/utils/fusions.py
# Source license: Apache-2.0.

"""Fusion capability checks for Megatron models.

This module provides functions to check if various fusion optimizations
can be enabled based on the current environment and dependencies.
"""

import logging
import os

from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)

# Control whether to log warnings when fusions are disabled
# Set environment variable MEGATRON_SUPPRESS_FUSION_WARNINGS=1 to disable warnings
LOG_FUSION_DISABLE = os.environ.get("MEGATRON_SUPPRESS_FUSION_WARNINGS", "0") != "1"


def can_enable_gradient_accumulation_fusion() -> bool:
    """Check if gradient accumulation fusion can be enabled.

    There are two independent backends that support wgrad fusion:
    - TransformerEngine (TE): handles fuse_wgrad_accumulation internally,
      no extra CUDA extension needed.
    - APEX fused_weight_gradient_mlp_cuda: required for the non-TE
      (ColumnParallelLinear) path.

    In TE-based training containers (the common case for high-perf models),
    fused_weight_gradient_mlp_cuda is not installed, so the APEX check alone
    incorrectly returns False even though TE can perform the fusion.

    Returns:
        bool: True if gradient accumulation fusion is available via either backend.
    """
    try:
        import transformer_engine.pytorch  # noqa: F401

        return True
    except ImportError:
        pass

    try:
        import fused_weight_gradient_mlp_cuda  # noqa: F401

        return True
    except ImportError:
        if LOG_FUSION_DISABLE:
            logger.warning(
                "gradient_accumulation_fusion requires either TransformerEngine or the "
                "fused_weight_gradient_mlp_cuda APEX extension, but neither is available. "
                "Fusion disabled."
            )
        return False


def validate_rope_fusion_compatibility(config: TransformerConfig) -> bool:
    """Validate if RoPE fusion is compatible with the current model configuration.

    Args:
        config: The TransformerConfig instance to validate.

    Returns:
        bool: True if RoPE fusion is compatible, False otherwise.
    """
    if not config.apply_rope_fusion:
        return True

    # Check if position embedding type is RoPE (similar to arguments.py logic)
    position_embedding_type = getattr(config, "position_embedding_type", "learned_absolute")
    if position_embedding_type != "rope":
        if LOG_FUSION_DISABLE:
            logger.warning(
                f"apply_rope_fusion is only compatible with RoPE position embeddings. "
                f"Current position_embedding_type: {position_embedding_type}. "
                f"Consider disabling apply_rope_fusion."
            )
        return False

    # Check for multi_latent_attention incompatibility
    if getattr(config, "multi_latent_attention", False):
        if LOG_FUSION_DISABLE:
            logger.warning(
                "apply_rope_fusion for multi-latent attention only supports training. "
                "It is experimental and may change in future versions."
            )
        return True

    return True
