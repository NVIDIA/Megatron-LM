# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generalized Tensor Parallelism (GTP) public API.

GTP shards weight tensors 1/N across a GTP process group along ``out_features``
and materializes them on-demand via async all-gather. The implementation lives
in ``megatron.core.tensor_parallel.generalized_tensor_parallelism`` and depends
on TransformerEngine's FP8 / MXFP8 / NVFP4 primitives.

If TransformerEngine is missing or too old, the inner module imports cleanly
but stubs its TE-backed symbols and reports ``HAVE_TE = False``; this module
mirrors that as ``HAVE_GTP = False``. Consumers gate every GTP code path behind
``if HAVE_GTP:``, so no core module uses GTP symbols without TE.
"""

try:
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        GTP_CONFIG,
        HAVE_TE,
        GTPChain,
        GTPEmbeddingWeight,
        GTPShardedParam,
        classify_gtp_chains,
        get_ag_stream,
        get_rs_stream,
        make_sharded_tensors_for_checkpoint_with_gtp_remat,
        reset_gtp_quantize_cache,
        reset_gtp_state,
        set_cuda_graph_mempool,
        set_cuda_graph_modules,
        tag_gtp_params_with_names,
        update_gtp_config,
        wait_async_comms,
        wait_for_gtp_grad_reduction_on_current_stream,
        wrap_module_params_gtp,
    )

    HAVE_GTP = HAVE_TE
except ImportError:
    # Defensive fallback for any unexpected inner-import failure; consumers import
    # the other symbols lazily under an ``if HAVE_GTP:`` guard, so no stubs needed.
    HAVE_GTP = False


__all__ = [
    "HAVE_GTP",
    "GTP_CONFIG",
    "GTPChain",
    "GTPEmbeddingWeight",
    "GTPShardedParam",
    "classify_gtp_chains",
    "get_ag_stream",
    "get_rs_stream",
    "make_sharded_tensors_for_checkpoint_with_gtp_remat",
    "reset_gtp_quantize_cache",
    "reset_gtp_state",
    "set_cuda_graph_mempool",
    "set_cuda_graph_modules",
    "tag_gtp_params_with_names",
    "update_gtp_config",
    "wait_async_comms",
    "wait_for_gtp_grad_reduction_on_current_stream",
    "wrap_module_params_gtp",
]
