# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Experimental MLA latent context parallelism.

The package keeps the original import path stable while isolating layout, transport, backend,
module, and model-spec responsibilities in focused implementation modules.
"""

from __future__ import annotations

# Compatibility aliases retained for existing feature tests and downstream diagnostics.
import importlib

import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.layers import ColumnParallelLinear

from . import backend as _backend
from . import transport as _transport
from .cudnn_backend import (
    CudnnFusedAttentionAdapter,
    _CudnnPlanKey,
    _CudnnUid,
    _shared_cudnn_adapter,
)
from .fa4_backend import FA4Adapter
from .layout import (
    AlreadyZigZagTHDAdapter,
    LatentCPLayoutAdapter,
    PhaseSpec,
    ZigZagLayout,
    build_zigzag_layout,
)
from .mla_with_latent_cp import (
    MLAWithLatentCP,
    _build_local_latent_norm,
    apply_rotary_pos_emb,
    checkpoint,
    mcore_tp,
    preprocess_mla_latent_cp,
    tp_mappings,
)
from .specs import (
    configure_mla_latent_cp_decoder,
    configure_mla_latent_cp_hybrid_stack,
    get_mla_with_latent_cp_spec,
    make_mla_with_latent_cp_spec,
)
from .utils import (
    CUDNN_FRONTEND_SOURCE_REV,
    QUALIFIED_BACKEND_CONFIGS,
    BackendNotQualifiedError,
    BackendPlanNotSupportedError,
    LatentCPError,
    QualifiedBackendTuple,
    cudnn_backward_proxy,
    merge_attention_partials,
    scatter_upper_phase,
)

DirectAttentionAdapter = _backend.DirectAttentionAdapter
_qualified_backend_adapter = _backend._qualified_backend_adapter
_runtime_backend_tuple = _backend._runtime_backend_tuple

LatentCPTransport = _transport.LatentCPTransport
P2PRingTransport = _transport.P2PRingTransport
PayloadLease = _transport.PayloadLease
_LatentRingExchange = _transport._LatentRingExchange

__all__ = [
    "AlreadyZigZagTHDAdapter",
    "BackendNotQualifiedError",
    "BackendPlanNotSupportedError",
    "CUDNN_FRONTEND_SOURCE_REV",
    "CudnnFusedAttentionAdapter",
    "DirectAttentionAdapter",
    "FA4Adapter",
    "LatentCPError",
    "LatentCPLayoutAdapter",
    "LatentCPTransport",
    "MLAWithLatentCP",
    "P2PRingTransport",
    "PayloadLease",
    "PhaseSpec",
    "QUALIFIED_BACKEND_CONFIGS",
    "QualifiedBackendTuple",
    "ZigZagLayout",
    "build_zigzag_layout",
    "configure_mla_latent_cp_decoder",
    "configure_mla_latent_cp_hybrid_stack",
    "cudnn_backward_proxy",
    "get_mla_with_latent_cp_spec",
    "make_mla_with_latent_cp_spec",
    "merge_attention_partials",
    "preprocess_mla_latent_cp",
    "scatter_upper_phase",
]
