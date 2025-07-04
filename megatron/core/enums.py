# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import enum


class ModelType(enum.Enum):
    """Model type."""

    encoder_or_decoder = 1
    encoder_and_decoder = 2
    retro_encoder = 3
    retro_decoder = 4


class Fp8Recipe(str, enum.Enum):
    """FP8 recipe names: delayed, tensorwise, mxfp8, blockwise."""

    delayed = "delayed"
    tensorwise = "tensorwise"
    mxfp8 = "mxfp8"
    blockwise = "blockwise"
