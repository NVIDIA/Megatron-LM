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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/mla_provider.py
# Source license: Apache-2.0.

"""MLA (Multi-Latent Attention) Model Provider.

This module provides a minimal provider for models using Multi-Latent Attention,
such as DeepSeek V3 and Kimi K2.
"""

from dataclasses import dataclass

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


@dataclass
class MLAModelProvider(MLATransformerConfig, GPTModelProvider):
    """Provider for models using Multi-Latent Attention (MLA).

    This class combines MLATransformerConfig (which provides MLA-specific fields
    like q_lora_rank, kv_lora_rank, qk_head_dim, v_head_dim) with GPTModelProvider
    (which provides the model instantiation logic).

    Model-specific defaults (normalization, activation, fusions, etc.) should be
    configured via MEGATRON_DEFAULTS in the respective bridge classes.

    Used by:
        - DeepSeek V3
        - Kimi K2
        - Other MLA-based models
    """

    pass
