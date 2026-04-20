# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""ETP (Extended Tensor Parallelism) utilities for Megatron Core."""

import torch

try:
    from transformer_engine.pytorch.module.extended_tensor_parallelism import wrap_module_params_etp
except ImportError:
    wrap_module_params_etp = None


class ETPEmbeddingWeight(torch.autograd.Function):
    """All-gather embedding weight across the ETP group in the forward pass,
    reduce-scatter the weight gradient back in the backward pass.

    The embedding weight is stored sharded along the vocab dimension (dim 0)
    across the ETP group.  This function makes the full weight available for
    the embedding lookup and correctly distributes the gradient.
    """

    @staticmethod
    def forward(ctx, weight):
        """Forward: all-gather the sharded weight via the ETP prefetch chain."""
        ctx.save_for_backward(weight)
        gathered_weight = weight.all_gather_and_prefetch(fwd=True)
        return gathered_weight

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: reduce-scatter grad to match the sharded weight shape."""
        (weight,) = ctx.saved_tensors
        # grad_output: [full_vocab/tp, hidden] → sharded: [full_vocab/tp/ps, hidden]
        return weight.wgrad_reduce_scatter(grad_output)
