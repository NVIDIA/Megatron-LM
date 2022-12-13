# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import enum

class LayerType(enum.Enum):
    encoder = 1
    decoder = 2
    retro_encoder = 3
    retro_decoder = 4
    retro_decoder_with_retriever = 5
 
class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2

class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
    prefix = 3

class UL2ModelType(enum.Enum):
    encoder_decoder = 'ED'
    non_causal_decoder = 'ND'
    causal_decoder = 'CD'

# For backward compatibility with old model checkpoints
from megatron.core.enums import ModelType
