# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import abc

from megatron.core.inference.config import InferenceConfig


class BaseInferenceContext(abc.ABC):
    """Base class for inference contexts.

    Currently extended by `DynamicInferenceContext`.
    Extend this class for any future context types.
    """

    def __init__(self, inference_config: InferenceConfig):
        """
        Args:
        """
        self.config = inference_config
