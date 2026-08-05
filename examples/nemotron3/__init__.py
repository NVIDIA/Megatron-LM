# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Direct HybridModel architecture examples for Nemotron 3 models."""

from .nemotron_3_5_nano_30b_a3b import make_model_config as make_nano_model_config
from .nemotron_labs_3_puzzle_75b_a9b import make_model_config as make_puzzle_model_config

__all__ = ["make_nano_model_config", "make_puzzle_model_config"]
