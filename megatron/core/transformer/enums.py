# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import enum


# can we get rid of this?
# it's being used in pipeline schedules
class ModelType(enum.Enum):
    """Model Type

    encoder_or_decoder for bert, gpt etc
    encoder_and_decoder for multimodal , T5 etc
    """

    encoder_or_decoder = 1
    encoder_and_decoder = 2


class LayerType(enum.Enum):
    """Layer type
    embedding: embedding layer
    loss: loss layer
    encoder: encoder layer, not implemented yet, expect to be used in MLLM models
    decoder: decoder layer
    mtp: multi-token prediction layer, not implemented yet
    """

    embedding = 1
    loss = 2
    encoder = 3
    decoder = 4
    mtp = 5


class AttnType(enum.Enum):
    """Attention type"""

    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    """Attention Mask Type"""

    padding = 1
    causal = 2
    no_mask = 3  # only used for TE
    padding_causal = 4  # only used for thd attention
    arbitrary = 5


class AttnBackend(enum.Enum):
    """Attention Backend"""

    flash = 1
    fused = 2
    unfused = 3
    local = 4
    auto = 5
