# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch
from torch._utils import _flatten_dense_tensors

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


class PSEmbeddingWeightGathering(torch.autograd.Function):
    """Custom embedding weight processing function for PS:
    Gathering weight across PS group in the fwd pass, and
    reduce-scatter weight
    """

    @staticmethod
    def forward(ctx, sharded_weight):
        ctx.save_for_backward(sharded_weight)

        nvtx_range_push(msg="PSEmbeddingWeightGathering_fwd")
        weight = sharded_weight.all_gather_and_prefetch(fwd=True)
        nvtx_range_pop(msg="PSEmbeddingWeightGathering_bwd")

        return weight

    @staticmethod
    def backward(ctx, wgrad):
        """Backward."""
        (sharded_weight,) = ctx.saved_tensors

        nvtx_range_push(msg="PSEmbeddingWeightGathering_bwd")
        wgrad = sharded_weight.wgrad_reduce_scatter(wgrad)
        nvtx_range_pop(msg="PSEmbeddingWeightGathering_bwd")

        return wgrad
