# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time checks for the optional libraries an operation family binds.

``require`` imports a module, verifies that the exports the caller is about to use exist
and are not ``None``, optionally checks a minimum version, and hands the module back. It
is called from ``__init__`` (or from a family selector that ``__init__`` calls), never from
``forward``: an operation binds its kernels once and then calls them directly.

This module keeps no state. There is no registry, inventory or cached availability table;
each family's ``backends.py`` says, in code next to the import, exactly what it needs.
"""

from __future__ import annotations

import logging
from importlib import import_module, metadata
from importlib.util import find_spec
from operator import attrgetter
from types import ModuleType

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)


def is_available(module: str) -> bool:
    """Whether ``module`` has an import spec.

    Use this to record a capability once at construction (``self._packed_cp = ...``) when
    the input that needs it -- packed sequences, say -- is only known at execution.
    For a dotted name, ``find_spec`` may import its parent package. Probe a top-level
    package when parent-import side effects must be avoided.
    """
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def installed_version(module: str, dist: str | None = None) -> Version | None:
    """The installed version of ``module``, or ``None`` when it cannot be determined.

    Prefers ``module.__version__`` so source checkouts without distribution metadata still
    report a version, then falls back to ``importlib.metadata`` under ``dist`` (defaults to
    the top-level module name). Every ``None`` is logged with its reason: callers treat an
    unknown version as "requirement not met", which is safe but should not be silent.
    """
    top = module.split(".")[0]
    try:
        package = import_module(top)
    except ImportError as exc:
        logger.warning("Cannot determine the version of %s: it failed to import (%s).", top, exc)
        return None
    candidate = getattr(package, "__version__", None)
    if candidate is None:
        try:
            candidate = metadata.version(dist or top)
        except metadata.PackageNotFoundError:
            logger.warning(
                "Cannot determine the version of %s: no __version__ and no distribution "
                "metadata for %r (a source checkout that was not pip-installed?).",
                top,
                dist or top,
            )
            return None
    try:
        return Version(str(candidate))
    except InvalidVersion:
        logger.warning("Cannot parse the version of %s: %r is not PEP 440.", top, candidate)
        return None


def has_min_version(module: str, min_version: str, dist: str | None = None) -> bool:
    """Whether ``module`` is installed at ``min_version`` or newer. Unknown versions are False."""
    installed = installed_version(module, dist)
    return installed is not None and installed >= Version(min_version)


def require(
    module: str,
    *symbols: str,
    min_version: str | None = None,
    dist: str | None = None,
    needed_by: str,
) -> ModuleType:
    """Import ``module`` and check it provides ``symbols`` (dotted names allowed).

    ``symbols`` is variadic so the common call reads as a sentence, ``require("fla.ops.foo",
    "chunk_foo", needed_by="Foo recurrence")``, and stays a one-liner when several exports
    are needed. Everything after it is keyword-only, so a version string or a distribution
    name can never be mistaken for a symbol.

    Raises ``ImportError`` -- the one exception type every family uses for a missing or
    broken optional library -- with a message that names the operation asking for it.
    A ``RuntimeError`` or ``OSError`` raised while the library loads (a native extension
    built against another CUDA or Torch, for instance) is reported the same way, with the
    original error chained.
    """
    what = f"{needed_by} requires {module}"
    try:
        loaded = import_module(module)
    except ModuleNotFoundError as exc:
        # Keep the subclass (and ``name``) so callers that distinguish "not installed" from
        # "installed but broken" still can.
        raise ModuleNotFoundError(f"{what}: {exc}", name=exc.name) from exc
    except ImportError as exc:
        raise ImportError(f"{what}: {exc}") from exc
    except (OSError, RuntimeError) as exc:
        raise ImportError(f"{what}, which is installed but failed to load: {exc}") from exc
    for symbol in symbols:
        try:
            value = attrgetter(symbol)(loaded)
        except AttributeError as exc:
            raise ImportError(f"{what} with {module}.{symbol}, which is missing.") from exc
        except (ImportError, OSError, RuntimeError) as exc:
            # Lazy namespaces (``cudnn.DSA``) load their extension on first attribute access.
            raise ImportError(
                f"{what} with {module}.{symbol}, which failed to load: {exc}"
            ) from exc
        if value is None:
            raise ImportError(
                f"{what} with {module}.{symbol}, which is unavailable in this installation."
            )
    if min_version is not None:
        installed = installed_version(module, dist)
        if installed is None:
            raise ImportError(f"{what}>={min_version}, but its version cannot be determined.")
        if installed < Version(min_version):
            raise ImportError(f"{what}>={min_version}; found {installed}.")
    return loaded
