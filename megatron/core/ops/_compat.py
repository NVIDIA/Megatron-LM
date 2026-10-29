# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated import-path forwarding for modules that moved into ``megatron.core.ops``.

A forwarder module is two lines::

    from megatron.core.ops._compat import deprecated_module
    __getattr__, __dir__ = deprecated_module(__name__, "megatron.core.ops.ssm.mamba2.mixer")

It resolves every attribute lazily through PEP 562 module ``__getattr__``, so importing the
old path does not import the new module (or its optional kernel dependencies) until a name
is actually used. ``from old import *``, private names, and pickles that recorded the old
``__module__`` all keep working, and every object is the canonical one -- no copies.

Writes go the same way as reads. Assigning or deleting a name on a forwarder -- including
``mock.patch`` and ``monkeypatch.setattr`` on the old path -- changes the module the name is
read from, so a patch through the old path still reaches the code that uses it.

The forwarder emits a ``DeprecationWarning`` once per process at import. It never adds an
import hook or alias registry; it is a plain module that can be deleted on the removal date.
"""

from __future__ import annotations

import sys
import warnings
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Callable, Iterable, Mapping

REMOVAL_VERSION = "0.22"  # two minor releases after the move landed in 0.20

# Keys, in a forwarder's ``__dict__``, of the function that names the module a write goes to
# and of what that module held before each write through the forwarder still in effect.
_OWNER = "__forwarder_owner__"
_ORIGINALS = "__forwarder_originals__"
_MISSING = object()


class _ForwarderModule(ModuleType):
    """Module type of a forwarder: names it does not hold itself are written where they are read.

    Without this, ``setattr`` on a forwarder would only shadow the name in the forwarder, and a
    patch applied through the old path would silently leave the implementation untouched.

    A written name also stays in the forwarder's own namespace until the write ends, as on any
    patched module, because patch tools choose how to undo from the target's ``__dict__``:
    ``mock.patch`` and pytest 9.2+ ``monkeypatch`` restore a name the target held by assigning
    the value they saved, so nested patches unwind by assignment, and remove a name it did not
    hold by deleting it. Deleting the name, or assigning the original back as ``monkeypatch``
    does before pytest 9.2, ends the write and restores what the implementation held before
    it. Reads of a written name still come from the implementation.
    """

    def __getattribute__(self, name: str) -> object:
        namespace = super().__getattribute__("__dict__")
        if name in namespace.get(_ORIGINALS, ()):
            return getattr(namespace[_OWNER](name), name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: object) -> None:
        namespace = self.__dict__
        originals = namespace[_ORIGINALS]
        local = name not in originals and (
            (name.startswith("__") and name.endswith("__"))
            or name in namespace
            # The import system binding a forwarder child, ``old_pkg.child``, on its parent.
            or (isinstance(value, ModuleType) and value.__name__ == f"{self.__name__}.{name}")
        )
        if local:
            super().__setattr__(name, value)
            return
        owner = namespace[_OWNER](name)
        original = originals.setdefault(name, vars(owner).get(name, _MISSING))
        if value is original:
            del originals[name], namespace[name]
        else:
            namespace[name] = value
        setattr(owner, name, value)

    def __delattr__(self, name: str) -> None:
        namespace = self.__dict__
        originals = namespace[_ORIGINALS]
        if name in namespace and name not in originals:
            super().__delattr__(name)
            return
        owner = namespace[_OWNER](name)
        if name not in originals:
            delattr(owner, name)
            return
        original = originals.pop(name)
        del namespace[name]
        if original is not _MISSING:
            setattr(owner, name, original)
        elif name in vars(owner):
            delattr(owner, name)


def deprecated_module(
    old_name: str,
    *targets: str,
    exports: Mapping[str, str] | None = None,
    removal_version: str = REMOVAL_VERSION,
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Return ``(__getattr__, __dir__)`` for a forwarder at ``old_name``.

    ``targets`` are tried in order, so a module that was split during the move can list
    each new home. The first target is the documented replacement in the warning.

    ``exports`` maps the names a historical package re-exported to the modules that define
    them. Those names resolve there and form the forwarder's ``__all__``, so the canonical
    package does not have to keep re-exporting them.
    """
    if not targets:
        raise ValueError(f"{old_name}: a deprecated module needs at least one target.")
    exports = dict(exports or {})
    warnings.warn(
        f"{old_name} is deprecated and will be removed in Megatron Core {removal_version}; "
        f"import from {targets[0]} instead.",
        DeprecationWarning,
        stacklevel=3,
    )

    def _modules() -> Iterable[object]:
        for target in targets:
            yield import_module(target)

    def _owner(name: str) -> object:
        """The module that defines ``name`` as this forwarder reads it, for attribute writes."""
        if name in exports:
            return import_module(exports[name])
        if len(targets) == 1:
            return import_module(targets[0])
        raise AttributeError(
            f"cannot assign {name!r} through {old_name!r}, which forwards to several modules; "
            f"assign it on the module that defines it ({', '.join(targets)})."
        )

    forwarder = sys.modules.get(old_name)
    if forwarder is not None:
        forwarder.__dict__[_OWNER] = _owner
        forwarder.__dict__.setdefault(_ORIGINALS, {})
        forwarder.__class__ = _ForwarderModule

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
        if name in exports:
            return getattr(import_module(exports[name]), name)
        submodule = _submodule(name)
        if submodule is not None:
            return submodule
        if name == "__all__":
            if exports:
                return list(exports)
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
            f"module {old_name!r} has no attribute {name!r} " f"(forwarded to {', '.join(targets)})"
        )

    def __dir__() -> list[str]:
        names: set[str] = set(exports)
        for module in _modules():
            names.update(dir(module))
        return sorted(names)

    return __getattr__, __dir__
