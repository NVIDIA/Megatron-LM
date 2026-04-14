# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# isort: skip_file
# Import order matters: RMSNorm must be imported before gpt_model
# to avoid circular import.
from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from .rms_norm import RMSNorm

from .gpt_model import GPTModel
from .language_model import get_language_model
