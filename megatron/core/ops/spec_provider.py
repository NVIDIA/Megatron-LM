# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The one type model code holds, and how it is assembled."""

from __future__ import annotations

from typing import Mapping

from megatron.core.ops.attention.contract import AttentionSlots
from megatron.core.ops.linear.contract import LinearSlots
from megatron.core.ops.loss.contract import LossSlots
from megatron.core.ops.moe.contract import MoeSlots
from megatron.core.ops.norm.contract import NormSlots
from megatron.core.ops.operations import Operation

__all__ = ["BackendSpecProvider"]


def _backend_name(owner: object) -> str:
    """``module.Class`` for a backend, since the class name alone is deliberately generic.

    Every backend module names its class after the family, so ``Linear`` says nothing on its
    own; the module is what identifies the backend.
    """
    kind = type(owner)
    return f"{kind.__module__.rsplit('.', 1)[-1]}.{kind.__name__}"


class BackendSpecProvider(NormSlots, LinearSlots, AttentionSlots, MoeSlots, LossSlots):
    """Provides the implementation of every operation, for one configuration.

    This is the only type model code holds, so nothing downstream can branch on which backend
    it got. Its slots come from the per-family ``*Slots`` classes, each of which sits next to
    the contract it describes; inherited slots raise until a backend takes them over.

    A *backend* is any object implementing one or more of those slot methods. Assembling binds
    the owning backend's method onto this object, so a call costs one attribute lookup while
    the model is built and nothing at all afterwards.
    """

    def __init__(self, owners: Mapping[Operation, object] | None = None) -> None:
        # Defaulted so a subclass that does not call up, such as the out-of-tree Kitchen
        # provider, still gets a usable object.
        owners = dict(owners or {})
        self._owners = owners
        for operation, owner in owners.items():
            method = getattr(owner, operation.method, None)
            if not callable(method):
                raise ValueError(
                    f"Backend '{_backend_name(owner)}' was selected for operation "
                    f"'{operation.qualified_name}' but does not implement {operation.method}()."
                )
            # Shadow the inherited slot with the owner's bound method.
            setattr(self, operation.method, method)  # type: ignore[method-assign]

    def __repr__(self) -> str:
        owners = ", ".join(
            f"{operation.method}={_backend_name(owner)}"
            for operation, owner in sorted(
                getattr(self, "_owners", {}).items(), key=lambda item: str(item[0])
            )
        )
        return f"BackendSpecProvider({owners})"
