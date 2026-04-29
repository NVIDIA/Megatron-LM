# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import enum


# can we get rid of this?
# it's being used in pipeline schedules
class ModelType(enum.Enum):
    """Model Type

    encoder_or_decoder for bert, gpt etc
    """

    encoder_or_decoder = 1


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
    causal_bottom_right = 6  # only used for TE


class AttnBackend(enum.Enum):
    """Attention Backend"""

    flash = 1
    fused = 2
    unfused = 3
    local = 4
    auto = 5


class CudaGraphModule(enum.Enum):
    """Named capture regions for per-layer CUDA graphs.

    Whole-layer capture is represented outside this enum by an empty scope. Current per-layer
    implementations that consume these values are `local` and `transformer_engine`.
    """

    attn = 1  # Captures attention layers
    mlp = 2  # Captures MLP layers (dense layers only)
    moe = 3  # Captures MoE layers (drop-and-pad MoE layers only)
    moe_router = 4  # Captures MoE router part
    moe_preprocess = 5  # Captures MoE preprocessing part (requires moe_router)
    mamba = 6  # Captures Mamba layers


# Deprecated: use CudaGraphModule instead. Retained only for checkpoint backward compat.
class CudaGraphScope(enum.Enum):
    """Deprecated predecessor of CudaGraphModule.

    Preserved as a standalone class (not an alias) so that pre-refactor checkpoints that
    stored CudaGraphScope enum instances can be deserialized correctly. The original ordinals
    differ from CudaGraphModule (full_iteration=1, attn=2, …), so a simple alias would
    silently reconstruct enum members with the wrong identity.

    Do NOT use in new code. Migration guide:
    - full_iteration → cuda_graph_impl="full_iteration"
    - full_iteration_inference → inference_cuda_graph_scope=InferenceCudaGraphScope.block
    - all other members → equivalent CudaGraphModule member
    """

    full_iteration = 1
    attn = 2
    mlp = 3
    moe = 4
    moe_router = 5
    moe_preprocess = 6
    mamba = 7
    full_iteration_inference = 8


class InferenceCudaGraphScope(enum.Enum):
    """Inference CUDA graph scope.

    This controls the ownership boundary for inference CUDA graphs:
    - none: inference runs without CUDA graphs.
    - layer: graphs are owned at the module/layer boundary. This name
      does not by itself imply any finer-grained replay contract within the layer.
    - block: graphs are owned by the enclosing block rather than
      individual modules.
    """

    none = 1
    layer = 2
    block = 3
