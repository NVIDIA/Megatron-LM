# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generalized Tensor Parallelism (GTP) public API.

GTP shards weight tensors 1/N across a GTP process group along ``out_features``
and materializes them on-demand via async all-gather. The implementation lives
in ``megatron.experimental.gtp.generalized_tensor_parallelism`` and depends on
TransformerEngine's FP8 / MXFP8 / NVFP4 primitives.

If TransformerEngine is missing or too old, the inner import fails and the
package exposes only ``HAVE_GTP = False``. No core module imports GTP symbols
unconditionally at module load time.
"""

try:
    from megatron.experimental.gtp.generalized_tensor_parallelism import (
        GTP_CONFIG,
        GTPChain,
        GTPEmbeddingWeight,
        GTPShardedParam,
        classify_gtp_chains,
        get_ag_stream,
        get_rs_stream,
        make_sharded_tensors_for_checkpoint_with_gtp,
        reset_gtp_quantize_cache,
        set_cuda_graph_mempool,
        set_cuda_graph_modules,
        tag_gtp_params_with_names,
        update_gtp_config,
        wait_async_comms,
        wait_for_gtp_grad_reduction_on_current_stream,
        wrap_module_params_gtp,
    )

    HAVE_GTP = True
except ImportError:
    # GTP requires TransformerEngine with the GTP hook registry; when it's
    # unavailable only ``HAVE_GTP`` is exposed. Consumers import the other
    # symbols lazily under an ``if HAVE_GTP:`` guard, so no fallbacks are needed.
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
    "make_sharded_tensors_for_checkpoint_with_gtp",
    "reset_gtp_quantize_cache",
    "set_cuda_graph_mempool",
    "set_cuda_graph_modules",
    "tag_gtp_params_with_names",
    "update_gtp_config",
    "wait_async_comms",
    "wait_for_gtp_grad_reduction_on_current_stream",
    "wrap_module_params_gtp",
]
