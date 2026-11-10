# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated -> canonical module table for the SSM / sparse-attention move.

Generated alongside the forwarder modules; keep the two in sync.
"""

FORWARDED = {
    "megatron.core.tensor_parallel.cross_entropy": ("megatron.core.ops.loss.cross_entropy",),
    "megatron.core.fusions.fused_cross_entropy": ("megatron.core.ops.loss.fused_cross_entropy",),
    "megatron.core.fusions.fused_layer_norm": ("megatron.core.ops.norm.fused_layer_norm",),
    "megatron.core.transformer.torch_norm": ("megatron.core.ops.norm.torch_norm",),
}

# Names each historical package re-exported, and the module that defines each. The
# canonical packages export nothing; these stay available on the deprecated paths only.
EXPORTS = {}
