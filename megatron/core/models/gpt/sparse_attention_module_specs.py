# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.sparse_attention import (
    Indexer,
    IndexerSubmodules,
    SparseAttention,
    SparseAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec


def get_indexer_spec_for_backend(
    backend: BackendSpecProvider, normalization: Optional[str] = None
) -> ModuleSpec:
    """Helper function to get Indexer module spec for a given backend.

    Args:
        backend: Backend specification provider (TE or Local).
        normalization: Normalization type ("RMSNorm" or None for LayerNorm).

    Returns:
        ModuleSpec for Indexer with appropriate submodules.
    """
    rms_norm = normalization == "RMSNorm"
    return ModuleSpec(
        module=Indexer,
        submodules=IndexerSubmodules(
            linear_wq_b=backend.linear(),
            linear_wk=backend.linear(),
            k_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=True),
            linear_weights_proj=backend.linear(),
        ),
    )


def get_sparse_attention_module_spec_for_backend(
    backend: BackendSpecProvider, sparse_attention_type: str, normalization: Optional[str] = None
) -> ModuleSpec:
    """Helper function to get module spec for Sparse Attention.

    Args:
        backend: Backend specification provider (TE or Local).
        sparse_attention_type: Type of sparse attention.
        normalization: Normalization type ("RMSNorm" or None for LayerNorm).

    Returns:
        ModuleSpec for the sparse attention implementation with appropriate submodules.
    """
    if sparse_attention_type == "dsa":
        # Because TransformerEngine does not support sparse attention yet, we use local
        # implementation whether the backend is TransformerEngine or not.
        return ModuleSpec(
            module=SparseAttention,
            submodules=SparseAttentionSubmodules(
                indexer=get_indexer_spec_for_backend(backend, normalization=normalization),
            ),
        )
    else:
        raise ValueError(f"Invalid sparse attention type: {sparse_attention_type}")
