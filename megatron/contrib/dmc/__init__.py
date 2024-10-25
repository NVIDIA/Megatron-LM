# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""
Dynamic Memory Compression (DMC)

This package provides code for training and inference of DMC-enabled transformer models.
For more details, please see https://arxiv.org/abs/2403.09636 .
"""

from .dmc import add_dmc_layer, get_dmc_loss
