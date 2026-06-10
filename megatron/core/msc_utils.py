# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import multistorageclient as msc

    logger.info('The multistorageclient package is available.')
except ModuleNotFoundError:
    msc = None


class _FeatureFlag:

    def __init__(self, default: bool = False):
        self._enabled = default

    def enable(self) -> None:
        """Enable the feature flag."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the feature flag."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if the feature flag is enabled."""
        return self._enabled and msc is not None

    def import_package(self) -> Any:
        """Import the package."""
        if msc is None:
            raise RuntimeError(
                "The multistorageclient package is not available. "
                "Please install it using `pip install multi-storage-client`."
            )
        if not self.is_enabled():
            raise RuntimeError(
                "The MSC feature is disabled. Please enable it by passing --enable-msc."
            )
        return msc

    def __getstate__(self):
        """Get the state for pickling."""
        return {'_enabled': self._enabled}

    def __setstate__(self, state):
        """Set the state during unpickling."""
        self._enabled = state['_enabled']


MultiStorageClientFeature = _FeatureFlag(default=False)


class MaybeMultiStorageClient:
    """
    Helper class to use MultiStorageClient
    """
    def path_isdir(self, path, strict:bool = True):
        if MultiStorageClientFeature.is_enabled():
            pkg = MultiStorageClientFeature.import_package()
            return pkg.os.path.isdir(path, strict=strict)
        else:
            import os
            return os.path.isdir(path)

    def __getattr__(self, name):
        if MultiStorageClientFeature.is_enabled():
            pkg = MultiStorageClientFeature.import_package()
            if hasattr(pkg, name):
                return getattr(pkg, name)

        if name == "open":
            return open
        if name == "os":
            import os

            return os
        if name == "Path":
            from pathlib import Path

            return Path
        if name == "torch":
            import torch

            return torch
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!s}")

    def __dir__(self):
        attrs = {"open", "os", "Path", "torch"}
        if MultiStorageClientFeature.is_enabled():
            try:
                pkg = MultiStorageClientFeature.import_package()
                attrs.update(dir(pkg))
            except RuntimeError:
                pass
        return sorted(attrs)


maybe_msc = MaybeMultiStorageClient()
__all__ = ['MultiStorageClientFeature', 'maybe_msc']
