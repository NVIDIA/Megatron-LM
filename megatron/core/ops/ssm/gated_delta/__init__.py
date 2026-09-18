# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDN/GDN2 operation modules, recurrence contracts, references and FLA targets.

Inputs use BTHD layout. Kernels own the local recurrence and its backward, not
parameters, CP transforms, checkpointing or inference-cache allocation. The
``common``, ``gdn`` and ``gdn2`` modules own operation parameters, CP execution
and checkpoint mappings; global inference-cache allocation remains external. Optional
initial/final states are BHKV. GDN uses scalar gates; GDN2 uses channel-wise gates.
Their additional keywords differ, but their common call surface is below.

The Torch references preserve the existing deterministic-mode path and reject
packed ``cu_seqlens``. FLA supplies packed training and GDN recurrent inference;
GDN2 inference remains unsupported. No broader determinism guarantee is implied.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


_EXPORT_MODULES = {
    "HAVE_FLA": ".common",
    "HAVE_FLA_GDN2": ".gdn2",
    "GatedDeltaNet": ".gdn",
    "GatedDeltaNet2": ".gdn2",
    "GatedDeltaNetSubmodules": ".common",
    "causal_conv1d": ".common",
    "chunk_gated_delta_rule": ".common",
    "chunk_gdn2": ".gdn2",
    "get_parameter_local_cp": ".common",
    "l2norm": ".common",
    "tensor_a2a_cp2hp": ".common",
    "tensor_a2a_hp2cp": ".common",
    "torch_chunk_gated_delta_rule": ".gdn",
    "torch_chunk_gdn2": ".gdn2",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    """Resolve the historical GDN exports without loading kernels on package import."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_EXPORT_MODULES[name], __name__), name)


class GatedDeltaRuleInterface(Protocol):
    """Common keyword call surface for the GDN and GDN2 recurrence kernels."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        *,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return local recurrence output and, if requested, the final state."""
        ...
