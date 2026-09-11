# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time binding for the existing optional DSA hook interfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Callable

from megatron.core.ops.attention.dsa.kernel_metadata import (
    CUDNN_ATTENTION,
    CUDNN_FULL,
    CUDNN_LOSS,
    CUDNN_TOPK,
    DSA_INDEXER_REFERENCE,
    DSA_REFERENCE,
    TILELANG_ATTENTION,
    TILELANG_LOSS,
    TILELANG_TOPK,
)
from megatron.core.ops.kernel_metadata import DeterminismPolicy, KernelMetadata, validate_kernels

if TYPE_CHECKING:
    from torch import Tensor

    from megatron.core.transformer.transformer_config import TransformerConfig

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSAKernels:
    """Concrete backend hooks, with None for hooks the selected backend does not supply.

    Signatures are those in ``dsa_tilelang_kernels`` and ``dsa_cudnn_kernels``.
    A hook can also return None for unsupported runtime inputs. Callers then use
    the reference implementation. The object owns no tensors or model state.
    """

    backend: str = "none"
    run_fused_qk_topk: Callable[..., tuple[Tensor, Tensor | None] | None] | None = None
    run_fused_qk_topk_with_loss: (
        Callable[..., tuple[Tensor, Tensor | None, Tensor] | None] | None
    ) = None
    run_fused_absorbed_sparse_attention: Callable[..., Tensor | None] | None = None
    run_fused_dsa_attention: Callable[..., tuple[Tensor, Tensor] | None] | None = None
    metadata: tuple[KernelMetadata, ...] = ()

    def log_declined(self, hook_name: str) -> None:
        """Keep fallback diagnostics without wrapping or resolving a kernel call."""
        _LOGGER.debug(
            "DSA fused backend %s %s declined; falling back (backend returned None).",
            self.backend,
            hook_name,
        )


def select_dsa_kernels(config: TransformerConfig) -> DSAKernels:
    """Bind the configured backend once; do not resolve it from a model forward."""
    from megatron.core.ops.attention.dsa import dsa_kernels

    policy = (
        DeterminismPolicy.WARN
        if getattr(config, "deterministic_mode", False)
        else DeterminismPolicy.IGNORE
    )
    if not dsa_kernels.use_fused_dsa_kernels(config):
        validate_kernels((DSA_REFERENCE, DSA_INDEXER_REFERENCE), determinism=policy)
        return DSAKernels()
    module_name = dsa_kernels._get_backend_module_name(config)
    assert module_name is not None
    try:
        declarations = (
            (TILELANG_TOPK, TILELANG_LOSS, TILELANG_ATTENTION)
            if config.dsa_kernel_backend == "tilelang"
            else (CUDNN_TOPK, CUDNN_LOSS, CUDNN_ATTENTION, CUDNN_FULL)
        )
        validate_kernels(declarations, determinism=policy)
        # Validate native requirements before loading the selected adapter.
        backend = import_module(module_name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Failed to import DSA kernel backend {module_name}: {exc}") from exc
    return DSAKernels(
        backend=config.dsa_kernel_backend,
        run_fused_qk_topk=getattr(backend, "run_fused_qk_topk", None),
        run_fused_qk_topk_with_loss=getattr(backend, "run_fused_qk_topk_with_loss", None),
        run_fused_absorbed_sparse_attention=getattr(
            backend, "run_fused_absorbed_sparse_attention", None
        ),
        run_fused_dsa_attention=getattr(backend, "run_fused_dsa_attention", None),
        metadata=declarations,
    )
