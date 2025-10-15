# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import abc


class BaseInferenceContext(abc.ABC):
    """Base class for inference contexts.

    Currently extended by `StaticInferenceContext` and `DynamicInferenceContext`.
    Extend this class for any future contexts types.
    """

    def __init__(self, materialize_only_last_token_logits: bool):
        """
        Args:
            materialize_only_last_token_logits (bool):
                If True, only the last-token logits will be extracted during decode
        """
        self.materialize_only_last_token_logits = materialize_only_last_token_logits

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
