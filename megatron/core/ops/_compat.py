# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated import-path forwarding for modules that moved into ``megatron.core.ops``.

A forwarder module is two lines::

    from megatron.core.ops._compat import deprecated_module
    __getattr__, __dir__ = deprecated_module(__name__, "megatron.core.ops.ssm.mamba2.mixer")

It resolves every attribute lazily through PEP 562 module ``__getattr__``, so importing the
old path does not import the new module (or its optional kernel dependencies) until a name
is actually used. ``from old import *``, private names, and pickles that recorded the old
``__module__`` all keep working, and every object is the canonical one -- no copies.

The forwarder emits a ``DeprecationWarning`` once per process at import. It never adds an
import hook or alias registry; it is a plain module that can be deleted on the removal date.
"""

from __future__ import annotations

import sys
import warnings
from importlib import import_module
from importlib.util import find_spec
from typing import Callable, Iterable

REMOVAL_VERSION = "0.22"  # two minor releases after the move landed in 0.20


def deprecated_module(
    old_name: str, *targets: str, removal_version: str = REMOVAL_VERSION
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Return ``(__getattr__, __dir__)`` for a forwarder at ``old_name``.

    ``targets`` are tried in order, so a module that was split during the move can list
    each new home. The first target is the documented replacement in the warning.
    """
    if not targets:
        raise ValueError(f"{old_name}: a deprecated module needs at least one target.")
    warnings.warn(
        f"{old_name} is deprecated and will be removed in Megatron Core {removal_version}; "
        f"import from {targets[0]} instead.",
        DeprecationWarning,
        stacklevel=3,
    )

    def _modules() -> Iterable[object]:
        for target in targets:
            yield import_module(target)

    def _submodule(name: str) -> object | None:
        """A forwarder submodule of a forwarder package, without touching the targets.

        ``from old_pkg import child`` asks the package for ``child`` before importing it as
        a submodule; answering here keeps that lookup from importing the implementation.
        """
        package = sys.modules.get(old_name)
        if package is None or not hasattr(package, "__path__"):
            return None
        try:
            spec = find_spec(f"{old_name}.{name}")
        except (ImportError, ValueError):
            return None
        return import_module(f"{old_name}.{name}") if spec is not None else None

    def __getattr__(name: str) -> object:
        # Dunder probes (``hasattr(mod, "__wrapped__")`` from inspect, pytest, pkgutil, ...)
        # must not drag in the implementation and its optional kernels; only ``__all__``
        # is meaningful on a forwarder.
        if name.startswith("__") and name.endswith("__") and name != "__all__":
            raise AttributeError(f"module {old_name!r} has no attribute {name!r}")
        submodule = _submodule(name)
        if submodule is not None:
            return submodule
        if name == "__all__":
            names: list[str] = []
            for module in _modules():
                explicit = getattr(module, "__all__", None)
                names.extend(
                    explicit
                    if explicit is not None
                    else [n for n in vars(module) if not n.startswith("_")]
                )
            return names
        for module in _modules():
            try:
                return getattr(module, name)
            except AttributeError:
                continue
        raise AttributeError(
            f"module {old_name!r} has no attribute {name!r} "
            f"(forwarded to {', '.join(targets)})"
        )

    def __dir__() -> list[str]:
        names: set[str] = set()
        for module in _modules():
            names.update(dir(module))
        return sorted(names)

    return __getattr__, __dir__
