# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stack-wide recurrent-inference configuration."""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence


class SSMChunking(NamedTuple):
    """Chunk-related facts shared by every SSM layer in a stack."""

    chunk_size: int
    """The mixer's configured chunk size."""

    inference_chunk_size: int
    """The chunk length the dynamic-inference prefill kernels actually run at."""

    num_householder: int
    """Householder copies for Gated Delta Product layers; 0 for other mixers."""


def ssm_chunking(layer_type_list: List[str], layers: Sequence) -> Optional[SSMChunking]:
    """Returns the chunking every SSM layer in a stack shares, or None.

    None means the stack holds no recurrent layer, which happens on a pipeline
    stage made up entirely of attention and MLP layers.

    The stack is assumed homogeneous: one mixer type, one chunking. A mixed
    stack would need a per-mixer alignment quantum and per-mixer chunk
    descriptors, and nothing downstream models that, so it is rejected here
    rather than silently taking the first layer's answer for every layer.

    Args:
        layer_type_list: Per-layer symbols, positionally matching `layers`. See
            `megatron/core/models/hybrid/hybrid_layer_allocation.py`.
        layers: The stack's layers.

    Returns:
        The shared `SSMChunking`, or None if no layer is recurrent.
    """
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    chunking = None
    first_layer_idx = None
    for layer_idx, (layer_type, layer) in enumerate(zip(layer_type_list, layers)):
        # Mamba-family mixers (including Gated Delta Product) hang off `.mixer`;
        # Gated Delta Net registers its recurrent mixer in the attention slot.
        if layer_type == Symbols.MAMBA:
            mixer = getattr(layer, 'mixer', None)
        elif layer_type == Symbols.GDN:
            mixer = getattr(layer, 'self_attention', None)
        else:
            continue
        if mixer is None:
            continue

        layer_chunking = SSMChunking(
            chunk_size=mixer.chunk_size,
            # The chunk length the inference kernels actually run at, which is
            # not always `chunk_size`: the forked Gated Delta Product prefill
            # kernels chunk at a fixed 64. Falls back to chunk_size for any
            # mixer predating the property.
            inference_chunk_size=getattr(mixer, 'ssm_inference_chunk_size', mixer.chunk_size),
            # Gated Delta Product layers register as Mamba layers but carry a
            # Householder count, which sizes their (separate) chunk descriptors.
            num_householder=getattr(mixer, 'num_householder', 0) or 0,
        )
        if chunking is None:
            chunking, first_layer_idx = layer_chunking, layer_idx
        else:
            assert layer_chunking == chunking, (
                f"every SSM layer must share one chunking; layer {first_layer_idx} has "
                f"{chunking} but layer {layer_idx} has {layer_chunking}"
            )
    return chunking
