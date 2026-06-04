# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

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
        classify_gtp_chains,
        get_ag_stream,
        get_all_ag_streams,
        get_all_rs_streams,
        get_rs_stream,
        set_cuda_graph_mempool,
        set_cuda_graph_modules,
        tag_gtp_params_with_names,
        update_gtp_config,
        wait_async_comms,
        wrap_module_params_gtp,
    )

    HAVE_GTP = True
except ImportError:
    HAVE_GTP = False


__all__ = [
    "HAVE_GTP",
    "GTP_CONFIG",
    "GTPChain",
    "GTPEmbeddingWeight",
    "classify_gtp_chains",
    "get_ag_stream",
    "get_all_ag_streams",
    "get_all_rs_streams",
    "get_rs_stream",
    "set_cuda_graph_mempool",
    "set_cuda_graph_modules",
    "tag_gtp_params_with_names",
    "update_gtp_config",
    "wait_async_comms",
    "wrap_module_params_gtp",
]
