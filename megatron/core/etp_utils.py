# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""ETP (Extended Tensor Parallelism) utilities for Megatron Core."""

import torch

try:
    from transformer_engine.pytorch.module.extended_tensor_parallelism import (
        ETP_CONFIG,
        ETPChain,
        ETPShardedParam,
        classify_etp_chains,
        get_ag_stream,
        get_ag_streams_for_chain,
        get_all_ag_streams,
        get_all_rs_streams,
        get_rs_stream,
        get_rs_streams_for_chain,
        reallocate_etp_cache_to_mempool,
        set_cuda_graph_scope,
        tag_etp_params_with_names,
        update_config as update_etp_config,
        wait_async_comms,
        wrap_module_params_etp,
    )

    HAVE_ETP = True
except ImportError:
    ETP_CONFIG = None
    ETPChain = None
    ETPShardedParam = None
    classify_etp_chains = None
    get_ag_stream = None
    get_ag_streams_for_chain = None
    get_all_ag_streams = None
    get_all_rs_streams = None
    get_rs_stream = None
    get_rs_streams_for_chain = None
    reallocate_etp_cache_to_mempool = None
    set_cuda_graph_scope = None
    tag_etp_params_with_names = None
    wait_async_comms = None
    update_etp_config = None
    wrap_module_params_etp = None
    HAVE_ETP = False


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
