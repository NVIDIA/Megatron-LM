# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)

from .base_context import BaseInferenceContext


class StaticInferenceContext(BaseInferenceContext):
    """Static inference context that is passed to the main model in order
    to efficiently manage the KV cache during inference.

    Args:
        max_batch_size (int): Max supported batch size.
        max_sequence_length (int): Max supported sequence length.
    """

    def __init__(self, max_batch_size: int, max_sequence_length: int):
        super().__init__(materialize_only_last_token_logits=False)
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}
        self.decode_mode = False

    @classmethod
    def from_config(cls, config: InferenceWrapperConfig) -> "StaticInferenceContext":
        """Initialize context from a config."""
        max_batch_size = config.inference_max_requests
        max_sequence_length = config.inference_max_seq_length
        return cls(max_batch_size, max_sequence_length)

    def swap_key_value_dict(self, batch_idx):
        "swap between batches"
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")

        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert (
                len(batch_idx) == inference_key_memory.shape[1]
            )  # make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (
                new_inference_key_memory,
                new_inference_value_memory,
            )

    def enable_prefill_mode(self):
        """
        Indicates the generation loop is in the prefill phase (still processing
        input prompt tokens). This should be enabled if the generation loop is
        encoding prompt tokens for *any* request in a batch.
        """
        self.decode_mode = False

    def enable_decode_mode(self):
        """
        Indicates the generation loop is in the decode phase (generating new output
        tokens). This should only be enabled if the generation loop has fully encoded
        the prompts for *all* requests in a batch.
        """
        self.decode_mode = True

    def is_decode_only(self):
        """Functional access to `.decode_mode`, to match dynamic context."""
        return self.decode_mode

    def reset(self):
        """Resets the inference state for a new batch."""
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.enable_prefill_mode()

    def __str__(self):
        return (
            f"StaticInferenceContext(max_seq_len = {self.max_sequence_length}, "
            f"max_batch_size = {self.max_batch_size}, "
            f"sequence_len_offset = {self.sequence_len_offset}, "
            f"batch_size_offset = {self.batch_size_offset}, "
            f"key_value_memory_dict = {self.key_value_memory_dict.keys()})"
            f"decode_mode = {self.decode_mode}"
            f"materialize_only_last_token_logits = {self.materialize_only_last_token_logits}"
        )

    def __eq__(self, other):

        if id(self) == id(other):
            return True

        if not isinstance(other, StaticInferenceContext):
            return False

        # Check all attributes match
        basic_attrs = [
            'max_sequence_length',
            'max_batch_size',
            'sequence_len_offset',
            'batch_size_offset',
            'decode_mode',
            'materialize_only_last_token_logits',
        ]

        if not all(hasattr(other, attr) for attr in basic_attrs):
            return False

        # Check dictionary keys match; i.e. the same number of layers are cached
        if self.key_value_memory_dict.keys() != other.key_value_memory_dict.keys():
            return False

        # Check each tensor tuple in the dictionary
        for key in self.key_value_memory_dict:
            self_tensors = self.key_value_memory_dict[key]
            other_tensors = other.key_value_memory_dict[key]

            # Compare each key, value tensor in the tuple
            for self_tensor, other_tensor in zip(self_tensors, other_tensors):
                if (
                    self_tensor.data_ptr() != other_tensor.data_ptr()
                    or self_tensor.shape != other_tensor.shape
                ):
                    return False

    def is_static_batching(self):
        return True
