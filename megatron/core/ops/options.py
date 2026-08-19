# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every input that influences backend selection, in one value."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = ["BackendOptions"]


@dataclass(frozen=True)
class BackendOptions:
    """The complete set of selectors read while a provider is built.

    Adding a selector means adding a field here and one line to :meth:`from_config`. That is
    deliberately the only place to look for "what can change which implementation I get".
    Names are left as strings; :mod:`megatron.core.ops.resolve` is what turns them into
    operations and backends, so this stays plain data.
    """

    transformer_impl: str = "transformer_engine"
    """Chooses the backend each family should prefer for every operation it declares."""

    use_kitchen: bool = False
    use_kitchen_attention: bool = False
    kitchen_attention_backend: str = "sdpa"

    cross_entropy_loss_fusion: bool = False
    cross_entropy_fusion_impl: str = "native"
    cuda_graph_impl: str | None = None

    operation_backends: Mapping[str, str] = field(default_factory=dict)
    """Explicit per-operation choices, applied last and never silently ignored."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_backends", dict(self.operation_backends))

    @classmethod
    def from_config(
        cls, config: object, *, transformer_impl: str | None = None
    ) -> "BackendOptions":
        """Read the selectors a TransformerConfig already carries.

        Args:
            config: a TransformerConfig, or anything exposing the same attributes.
            transformer_impl: overrides ``config.transformer_impl``, for the callers that build
                a Transformer Engine spec regardless of what the config says.
        """
        return cls(
            transformer_impl=transformer_impl or getattr(config, "transformer_impl"),
            use_kitchen=getattr(config, "use_kitchen", False),
            use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
            kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
            cross_entropy_loss_fusion=getattr(config, "cross_entropy_loss_fusion", False),
            cross_entropy_fusion_impl=getattr(config, "cross_entropy_fusion_impl", "native"),
            cuda_graph_impl=getattr(config, "cuda_graph_impl", None),
            operation_backends=getattr(config, "op_backend_overrides", None) or {},
        )
