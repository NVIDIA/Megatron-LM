# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import enum


# can we get rid of this?
# it's being used in pipeline schedules
class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2


# class LayerType(enum.Enum):
#     encoder = 1
#     decoder = 2


class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
    no_mask = 3  # only used for TE
