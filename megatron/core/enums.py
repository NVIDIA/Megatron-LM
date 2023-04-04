# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import enum

class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2
