# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import abc

from megatron.core.inference.config import InferenceConfig


class BaseInferenceContext(abc.ABC):
    """Base class for inference contexts.

    Currently extended by `StaticInferenceContext` and `DynamicInferenceContext`.
    Extend this class for any future contexts types.
    """

    def __init__(self, inference_config: InferenceConfig):
        """
        Args:
        """
        self.config = inference_config

        # True while an inference engine is currently using the model with this context.
        # Modules that need to distinguish between inference and non-inference (e.g. training,
        # logprobs) paths should read `inference_context.is_active` rather than relying on
        # `self.training`, `torch.is_grad_enabled()`, or `inference_context is not None`.
        self.is_active: bool = False

    def set_active(self) -> None:
        """Mark the inference engine as active. Idempotent."""
        self.is_active = True

    def unset_active(self) -> None:
        """Mark the inference engine as inactive. Idempotent."""
        self.is_active = False

    @abc.abstractmethod
    def is_static_batching(self) -> bool:
        """Return `True` if context uses static batching."""
        pass

    def is_dynamic_batching(self) -> bool:
        """Return `True` if context uses dynamic batching."""
        return not self.is_static_batching()

    def increment_sequence_len_offset(self, increment: int) -> None:
        """Update sequence length offset. No-op for dynamic batching."""
        if self.is_static_batching():
            self.sequence_len_offset += increment

    def increment_batch_size_offset(self, increment: int) -> None:
        """Update batch size offset. No-op for dynamic batching."""
        if self.is_static_batching():
            self.batch_size_offset += increment

    def reset_batch_size_offset(self) -> None:
        """Reset batch size offset to 0. No-op for dynamic batching."""
        if self.is_static_batching():
            self.batch_size_offset = 0
