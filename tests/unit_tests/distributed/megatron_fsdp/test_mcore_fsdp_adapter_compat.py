# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    _validate_ep_overlap_cuda_graph_compat,
)
from megatron.core.transformer.enums import CudaGraphModule


def _ep_overlap_config(cuda_graph_impl, cuda_graph_modules=None):
    return SimpleNamespace(
        overlap_moe_expert_parallel_comm=True,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=cuda_graph_modules,
    )


def test_ep_overlap_fsdp_allows_te_partial_cuda_graph_safe_modules():
    _validate_ep_overlap_cuda_graph_compat(
        _ep_overlap_config(
            "transformer_engine",
            [
                CudaGraphModule.attn,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            ],
        )
    )


@pytest.mark.parametrize("cuda_graph_impl", ["none", "full_iteration"])
def test_ep_overlap_fsdp_allows_non_partial_cuda_graph_modes(cuda_graph_impl):
    _validate_ep_overlap_cuda_graph_compat(_ep_overlap_config(cuda_graph_impl))


@pytest.mark.parametrize("cuda_graph_module", [CudaGraphModule.moe, CudaGraphModule.mlp])
def test_ep_overlap_fsdp_rejects_unsafe_te_partial_cuda_graph_modules(cuda_graph_module):
    with pytest.raises(AssertionError, match="does not support TE partial CUDA graph"):
        _validate_ep_overlap_cuda_graph_compat(
            _ep_overlap_config("transformer_engine", [cuda_graph_module])
        )


def test_ep_overlap_fsdp_rejects_local_cuda_graph_impl():
    with pytest.raises(AssertionError, match="supports cuda_graph_impl"):
        _validate_ep_overlap_cuda_graph_compat(_ep_overlap_config("local", [CudaGraphModule.attn]))
