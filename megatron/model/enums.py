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
    causal = 2 # Overrides `attention_mask` to be a lower triangular matrix
    prefix = 3
    custom = 4 # Forces one to pass an `attention_mask` that's 1 if we need to mask. Tensor that can be broadcast to [micro_batch_size, n_head, seq_length, seq_length]

class PositionEmbeddingType(enum.Enum):
    rotary = 1 # NOTE: this one is not used so far, however for future compatibility the enum left as is
    absolute = 2
    alibi = 3

# For backward compatibility with old model checkpoints
from megatron.core.enums import ModelType
