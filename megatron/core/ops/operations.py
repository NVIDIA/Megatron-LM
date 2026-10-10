# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""What an operation is. The operations themselves are declared by their family."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Operation", "unowned"]


@dataclass(frozen=True)
class Operation:
    """One :class:`~megatron.core.ops.BackendSpecProvider` method a backend can own.

    ``family`` names the package under ``megatron.core.ops`` that declares this operation and
    the backends able to fill it. ``method`` is the provider method name, so an operation and
    the method implementing it cannot drift apart.

    Operations are declared by their family, next to their contract, rather than in one central
    list. Declare one only when an existing class, callable, or builder already owns that
    boundary at construction time; an implementation that has to branch on shape, phase, or
    communication keeps its current owner and exposes a target through ``ops`` instead.
    """

    family: str
    method: str
    optional: bool = False
    """True when a backend may leave this slot alone, because not every backend has one.

    An optional slot that nobody owns raises :func:`unowned` when it is called, rather than
    silently returning something wrong.
    """

    def __str__(self) -> str:
        return self.method

    @property
    def qualified_name(self) -> str:
        """``family.method``, for error messages and for disambiguating on the command line."""
        return f"{self.family}.{self.method}"


def unowned(operation: Operation) -> NotImplementedError:
    """The error a slot raises when no selected backend fills it."""
    return NotImplementedError(
        f"No backend owns '{operation.qualified_name}' for this configuration. "
        f"Select one with --op-backend {operation.method}=<backend>."
    )
