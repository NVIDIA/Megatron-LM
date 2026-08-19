# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional-dependency checks for operation backends.

Backends state their requirements here instead of setting a module-level ``HAVE_*`` flag.
Two rules keep the failure modes distinct:

* An optional dependency that is *not installed* is a normal, silent absence. Only a backend
  that was explicitly selected turns that into an error, and it does so while the model is
  being built, not at import time.
* An optional dependency that is installed but *broken* (a missing transitive library, a
  failed native loader) is always reported. Treating it as "not installed" is how a broken
  install silently becomes a slow fallback.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType


@lru_cache(maxsize=None)
def _import_outcome(module_name: str) -> tuple[ModuleType | None, BaseException | None, bool]:
    """Import ``module_name`` once. Returns (module, error, is_absent)."""
    try:
        return importlib.import_module(module_name), None, False
    except ModuleNotFoundError as error:
        # `error.name` distinguishes "this package is absent" from "this package is present
        # but one of its own imports is not".
        absent = error.name in {module_name, module_name.split(".", maxsplit=1)[0]}
        return None, error, absent
    except Exception as error:  # pylint: disable=broad-except
        return None, error, False


def is_installed(module_name: str) -> bool:
    """Whether an optional dependency is importable, without raising if it is absent."""
    module, _, absent = _import_outcome(module_name)
    if module is not None:
        return True
    if absent:
        return False
    raise ImportError(f"Optional dependency '{module_name}' is installed but failed to import.")


def require(module_name: str, *, backend: str) -> ModuleType:
    """Return a selected backend's dependency, or explain what to install."""
    module, error, absent = _import_outcome(module_name)
    if module is not None:
        return module
    reason = "is not installed" if absent else "failed to import"
    raise ImportError(f"Backend '{backend}' requires '{module_name}', which {reason}.") from error


def reset_cache() -> None:
    """Clear cached import outcomes. Intended for tests only."""
    _import_outcome.cache_clear()
