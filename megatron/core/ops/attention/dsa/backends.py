# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time binding for the existing optional DSA hook interfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Callable

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

    if not dsa_kernels.use_fused_dsa_kernels(config):
        return DSAKernels()
    module_name = dsa_kernels._get_backend_module_name(config)
    assert module_name is not None
    try:
        # Python caches modules; the legacy mutable backend-selection cache is not needed.
        backend = import_module(module_name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Failed to import DSA kernel backend {module_name}.") from exc
    return DSAKernels(
        backend=config.dsa_kernel_backend,
        run_fused_qk_topk=getattr(backend, "run_fused_qk_topk", None),
        run_fused_qk_topk_with_loss=getattr(backend, "run_fused_qk_topk_with_loss", None),
        run_fused_absorbed_sparse_attention=getattr(
            backend, "run_fused_absorbed_sparse_attention", None
        ),
        run_fused_dsa_attention=getattr(backend, "run_fused_dsa_attention", None),
    )
