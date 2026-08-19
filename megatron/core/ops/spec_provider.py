# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The construction-time API that model code uses to obtain operation implementations."""

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import TYPE_CHECKING, Mapping, Optional, Protocol

from megatron.core.ops.operations import Operation

if TYPE_CHECKING:
    from megatron.core.transformer.mlp import TEActivationFunctionBuilder
    from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
    from megatron.core.transformer.torch_norm import LayerNormBuilder


class BackendSpecProvider(Protocol):
    """Provides the implementations a backend contributes to spec building.

    A *backend* is any object implementing one or more of these methods. A backend that
    implements all of them can serve as a base; one that implements a subset can take over
    those operations on top of another base (see :func:`compose`). Methods are called while
    the model is being built and return the target itself, so there is no dispatch left in
    the forward path.
    """

    @abstractmethod
    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        ...

    @abstractmethod
    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        ...

    @abstractmethod
    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module fuses layernorm and column parallel linear, or None if unfused."""
        ...

    @abstractmethod
    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> "LayerNormBuilder":
        """Which module to use for layernorm."""
        ...

    @abstractmethod
    def core_attention(self) -> type:
        """Which module to use for attention."""
        ...

    @abstractmethod
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> "ExpertsBuilder":
        """Which module and submodules to use for grouped mlp."""
        ...

    @abstractmethod
    def activation_func(self) -> Optional["TEActivationFunctionBuilder"]:
        """Which module to use for the activation function, or None for config.activation_func."""
        ...

    # Methods below have a default every backend can inherit. Override only to change them.

    def linear(self) -> type:
        """Which non-parallel linear module the backend uses."""
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not provide a non-tensor-parallel linear. "
            f"Select a backend that does for the '{Operation.LINEAR}' operation."
        )

    def fuse_layernorm_and_linear(self) -> bool:
        """Whether the backend fuses layernorm into the column parallel linear.

        Derived from :meth:`column_parallel_layer_norm_linear` so the two cannot disagree.
        """
        return self.column_parallel_layer_norm_linear() is not None

    def moe_router(self) -> Optional[type]:
        """Which MoE router to use, or None to keep the MoESubmodules default."""
        return None


def compose(base: BackendSpecProvider, owners: Mapping[Operation, object]) -> BackendSpecProvider:
    """Return ``base`` with each listed operation taken over by another backend.

    This is the only way operations from different backends are combined. The owning
    backend's bound method is attached to the result, so a composed provider costs one
    attribute lookup while the model is built and nothing at all afterwards.

    Args:
        base: a backend implementing the whole protocol.
        owners: operations to take over, and the backend that owns each one.

    Raises:
        ValueError: if an owner does not implement the operation it was given.
    """
    if not owners:
        return base

    composed = copy.copy(base)
    for operation, owner in owners.items():
        method = getattr(owner, operation.value, None)
        if not callable(method):
            raise ValueError(
                f"Backend '{type(owner).__name__}' was selected for operation "
                f"'{operation}' but does not implement {operation.value}()."
            )
        # Shadow the base method with the owner's bound method. Methods the base keeps stay
        # bound to `composed`, so a base method that calls self.<other op> still sees overrides.
        setattr(composed, operation.value, method)  # type: ignore[method-assign]
    return composed
