# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generalized Tensor Parallelism (GTP) public API.

Thin re-export of the implementation in
``megatron.core.tensor_parallel.generalized_tensor_parallelism`` (see that module
for the design). GTP depends on TransformerEngine: if TE is missing or too old the
inner module imports cleanly but reports ``HAVE_TE = False``, mirrored here as
``HAVE_GTP = False``. Consumers gate every GTP code path behind ``if HAVE_GTP:``,
so no core module uses GTP symbols without TE.
"""

try:
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        HAVE_TE,
        GTPChain,
        GTPEmbeddingWeight,
        attach_gtp_to_presharded_module,
        classify_gtp_remat_chains,
        configure_gtp_remat_from_recipe,
        dequantize_gtp_native_fp8,
        get_ag_stream,
        get_rs_stream,
        gtp_native_fp8_load_context,
        gtp_remat_shard_dim0,
        is_gtp_param,
        make_sharded_tensors_for_checkpoint_with_gtp_remat,
        set_cuda_graph_mempool,
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
    "GTPChain",
    "GTPEmbeddingWeight",
    "attach_gtp_to_presharded_module",
    "classify_gtp_remat_chains",
    "configure_gtp_remat_from_recipe",
    "dequantize_gtp_native_fp8",
    "get_ag_stream",
    "get_rs_stream",
    "gtp_native_fp8_load_context",
    "gtp_remat_shard_dim0",
    "is_gtp_param",
    "make_sharded_tensors_for_checkpoint_with_gtp_remat",
    "set_cuda_graph_mempool",
    "wait_async_comms",
    "wait_for_gtp_grad_reduction_on_current_stream",
    "wrap_module_params_gtp",
]
