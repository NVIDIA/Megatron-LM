# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import multistorageclient as msc

    _msc_available = True
    logger.info('The multistorageclient package is available.')
except ModuleNotFoundError:
    msc = None
    _msc_available = False


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
                "The MSC feature is disabled. Please enable by removing the --disable-msc argument."
            )
        return msc

    def __getstate__(self):
        """Get the state for pickling."""
        return {'_enabled': self._enabled}

    def __setstate__(self, state):
        """Set the state during unpickling."""
        self._enabled = state['_enabled']


MultiStorageClientFeature = _FeatureFlag(_msc_available)


def open_file(*args, **kwargs):
    """Open a file with the appropriate method based on whether MSC is enabled."""
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.open(*args, **kwargs)
    else:
        return open(*args, **kwargs)


__all__ = ['MultiStorageClientFeature', 'open_file']
