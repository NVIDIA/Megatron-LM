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
import importlib.metadata
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.version import Version


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


def require(module_name: str) -> ModuleType:
    """Return a selected dependency, or explain what is wrong with it.

    Callers that know which backend asked should catch this and add that context;
    A provider does exactly that, from the ``REQUIRES`` its chosen backend declares.
    """
    module, error, absent = _import_outcome(module_name)
    if module is not None:
        return module
    reason = "is not installed" if absent else "failed to import"
    raise ImportError(f"'{module_name}' {reason}.") from error


def installed_version(module_name: str) -> "Version | None":
    """The installed version of an optional dependency, or None if it cannot be determined.

    Transformer Engine is asked through Megatron's own resolver, which already handles its
    module-versus-distribution naming and its dev suffixes; duplicating that here is how the
    two would drift. Anything else comes from package metadata.
    """
    from packaging.version import Version

    if module_name == "transformer_engine":
        from megatron.core.utils import get_te_version

        return get_te_version()
    try:
        return Version(importlib.metadata.version(module_name))
    except Exception:  # pylint: disable=broad-except
        return None


def reset_cache() -> None:
    """Clear cached import outcomes. Intended for tests only."""
    _import_outcome.cache_clear()
