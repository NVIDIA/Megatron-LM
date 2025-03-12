# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
Exports:

  - RetroConfig: configuration dataclass for RetroModel.
  - RetroModel: The Retro model.
  - get_retro_decoder_block_spec: Get spec for Retro decoder transformer block.
"""

from .config import RetroConfig
from .decoder_spec import get_retro_decoder_block_spec
from .model import RetroModel
