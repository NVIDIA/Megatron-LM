# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Generalized Tensor Parallelism (GTP) for Megatron Core.

See ``README.md`` in this folder for the design overview. The whole
implementation lives in ``generalized_tensor_parallelism.py``; this
``__init__`` re-exports the public surface and owns the ``HAVE_GTP``
capability flag (False when the implementation module fails to import,
e.g. when TransformerEngine's low-precision tensor primitives are missing).
"""

try:
    from megatron.experimental.gtp.generalized_tensor_parallelism import (
        GTP_CONFIG,
        GTPChain,
        GTPEmbeddingWeight,
        GTPShardedParam,
        classify_gtp_chains,
        get_ag_stream,
        get_ag_streams_for_chain,
        get_all_ag_streams,
        get_all_rs_streams,
        get_rs_stream,
        get_rs_streams_for_chain,
        reallocate_gtp_cache_to_mempool,
        set_cuda_graph_modules,
        tag_gtp_params_with_names,
    )
    from megatron.experimental.gtp.generalized_tensor_parallelism import (
        update_config as update_gtp_config,
    )
    from megatron.experimental.gtp.generalized_tensor_parallelism import (
        wait_async_comms,
        wrap_module_params_gtp,
    )

    HAVE_GTP = True
except ImportError:
    GTP_CONFIG = None
    GTPChain = None
    GTPEmbeddingWeight = None
    GTPShardedParam = None
    classify_gtp_chains = None
    get_ag_stream = None
    get_ag_streams_for_chain = None
    get_all_ag_streams = None
    get_all_rs_streams = None
    get_rs_stream = None
    get_rs_streams_for_chain = None
    reallocate_gtp_cache_to_mempool = None
    set_cuda_graph_modules = None
    tag_gtp_params_with_names = None
    update_gtp_config = None
    wait_async_comms = None
    wrap_module_params_gtp = None
    HAVE_GTP = False


__all__ = [
    "GTP_CONFIG",
    "GTPChain",
    "GTPEmbeddingWeight",
    "GTPShardedParam",
    "HAVE_GTP",
    "classify_gtp_chains",
    "get_ag_stream",
    "get_ag_streams_for_chain",
    "get_all_ag_streams",
    "get_all_rs_streams",
    "get_rs_stream",
    "get_rs_streams_for_chain",
    "reallocate_gtp_cache_to_mempool",
    "set_cuda_graph_modules",
    "tag_gtp_params_with_names",
    "update_gtp_config",
    "wait_async_comms",
    "wrap_module_params_gtp",
]
