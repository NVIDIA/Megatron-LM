# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import enum


class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2
    retro_encoder = 3
    retro_decoder = 4
