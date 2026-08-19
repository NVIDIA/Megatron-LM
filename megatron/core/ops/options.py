# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every input that influences backend selection, in one value."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from megatron.core.ops.operations import Operation, parse_operation

__all__ = ["BackendOptions"]


@dataclass(frozen=True)
class BackendOptions:
    """The complete set of selectors read while a provider is built.

    Adding a new selector means adding a field here and one line to :meth:`from_config`. That
    is deliberately the only place to look for "what can change which implementation I get".
    """

    transformer_impl: str = "transformer_engine"
    """Chooses the base backend that supplies every operation not overridden below."""

    use_kitchen: bool = False
    use_kitchen_attention: bool = False
    kitchen_attention_backend: str = "sdpa"

    operation_backends: Mapping[Operation, str] = field(default_factory=dict)
    """Explicit per-operation choices, applied last and never silently ignored."""

    def __post_init__(self) -> None:
        normalized = {
            operation if isinstance(operation, Operation) else parse_operation(str(operation)): name
            for operation, name in self.operation_backends.items()
        }
        object.__setattr__(self, "operation_backends", normalized)

    @classmethod
    def from_config(
        cls, config: object, *, transformer_impl: str | None = None
    ) -> "BackendOptions":
        """Read the selectors a TransformerConfig already carries.

        Args:
            config: a TransformerConfig, or anything exposing the same attributes.
            transformer_impl: overrides ``config.transformer_impl``, for the callers that
                build a Transformer Engine spec regardless of what the config says.
        """
        return cls(
            transformer_impl=transformer_impl or getattr(config, "transformer_impl"),
            use_kitchen=getattr(config, "use_kitchen", False),
            use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
            kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
            operation_backends=getattr(config, "op_backend_overrides", None) or {},
        )
