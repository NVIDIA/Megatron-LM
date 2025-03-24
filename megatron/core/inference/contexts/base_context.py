# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import abc


class BaseInferenceContext(abc.ABC):
    """Base class for inference contexts.

    Currently extended by `StaticInferenceContext` and `DynamicInferenceContext`.
    Extend this class for any future contexts types.
    """

    @abc.abstractmethod
    def is_static_batching(self) -> bool:
        """Return `True` if context uses static batching."""
        pass

    def is_dynamic_batching(self) -> bool:
        """Return `True` if context uses dynamic batching."""
        return not self.is_static_batching()
