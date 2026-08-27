# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime qualification and direct-backend dispatch for MLA latent CP."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Protocol

import torch
from torch import Tensor

from megatron.core.transformer.enums import AttnBackend

from . import utils as backend_utils
from .cudnn_backend import _resolve_cudnn_frontend_version, _shared_cudnn_adapter
from .fa4_backend import FA4Adapter
from .layout import PhaseSpec
from .utils import QUALIFIED_BACKEND_CONFIGS, BackendNotQualifiedError


class DirectAttentionAdapter(Protocol):
    """Direct packed-attention backend boundary used by each CP phase."""

    def prepare(
        self,
        *,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        phases: tuple[PhaseSpec, ...],
        scale: float,
    ) -> None:
        """Validate and prepare every phase before ring communication starts."""
        ...

    def forward_phase(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Return canonical FP32 output and LSE for one packed phase."""
        ...


def _runtime_backend_tuple(backend: AttnBackend) -> backend_utils.QualifiedBackendTuple:
    """Resolve the exact immutable qualification key without running numerical probes."""

    if not torch.cuda.is_available():
        raise BackendNotQualifiedError("latent CP attention requires CUDA")
    capability = torch.cuda.get_device_capability()
    if backend is AttnBackend.fused:
        try:
            cudnn = importlib.import_module("cudnn")
            frontend_version = _resolve_cudnn_frontend_version(cudnn)
            runtime_version = str(cudnn.backend_version_string())
        except (ImportError, AttributeError) as error:
            raise BackendNotQualifiedError(
                "unable to resolve cuDNN Frontend package/runtime versions"
            ) from error
        if not runtime_version:
            raise BackendNotQualifiedError("incomplete cuDNN qualification metadata")
        return backend, frontend_version, runtime_version, capability
    if backend is AttnBackend.flash:
        try:
            version = importlib.metadata.version("flash-attn-4")
            importlib.import_module("flash_attn.cute")
        except (importlib.metadata.PackageNotFoundError, ImportError) as error:
            raise BackendNotQualifiedError(
                "FA4 requires the exact flash-attn-4 distribution and flash_attn.cute"
            ) from error
        return backend, version, f"flash-attn-4=={version}", capability
    raise BackendNotQualifiedError(f"unsupported backend {backend!r}")


def _qualified_backend_adapter(
    backend: AttnBackend,
    runtime_tuple: backend_utils.QualifiedBackendTuple | None = None,
) -> tuple[DirectAttentionAdapter, backend_utils.QualifiedBackendTuple]:
    if runtime_tuple is None:
        runtime_tuple = _runtime_backend_tuple(backend)
    if runtime_tuple not in QUALIFIED_BACKEND_CONFIGS:
        raise BackendNotQualifiedError(
            "unqualified latent-CP backend tuple "
            f"{runtime_tuple!r}; qualified tuples are {QUALIFIED_BACKEND_CONFIGS!r}"
        )
    if backend is AttnBackend.fused:
        return _shared_cudnn_adapter(runtime_tuple), runtime_tuple
    if backend is AttnBackend.flash:
        return FA4Adapter(), runtime_tuple
    raise BackendNotQualifiedError(f"unsupported backend {backend!r}")
