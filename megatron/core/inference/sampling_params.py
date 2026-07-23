# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SamplingParams:
    """Inference parameters sent along with the prompts.
    This class contains request-level attributes that control the sampling techniques used when
    generating text. This is distinct from megatron.core.inference.contexts.BaseInferenceContext,
        which is sets model-level
    inference attributes such as the maximum sequence length, and contains the KV cache.

    For an explanation of these parameters refer to this blog
    https://ivibudh.medium.com/a-guide-to-controlling-llm-model-output-exploring-top-k-top-p-and-
    temperature-parameters-ed6a31313910
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    return_log_probs: bool = False
    skip_prompt_log_probs: bool = False
    return_segments: bool = False  # Whether to return individually detokenized tokens
    num_tokens_to_generate: int = None
    num_tokens_total: Optional[int] = None  # Cannot set both this and num_tokens_to_generate
    termination_id: Optional[int] = None
    top_n_logprobs: int = 0
    return_prompt_top_n_logprobs: bool = False  # Deprecated field for backwards compatibility
    add_BOS: bool = False
    stop_words: Optional[List[str]] = (
        None  # List of strings that will stop generation when produced
    )
    detokenize_stop_sequence: bool = False  # Keep stop words and EOD in generated text
    streaming: bool = False  # Emit incremental ENGINE_REPLY_PARTIAL frames.
    streaming_interval: int = 1  # Minimum unsent tokens per ENGINE_REPLY_PARTIAL.

    def __post_init__(self):
        """Validate parameters and maintain backward compatibility.

        Sets return_prompt_top_n_logprobs based on skip_prompt_log_probs and top_n_logprobs:
        - return_prompt_top_n_logprobs = not skip_prompt_log_probs and top_n_logprobs > 0
        """
        self._sync_prompt_logprobs_fields()
        self._validate_streaming_interval()

    def _sync_prompt_logprobs_fields(self):
        """Synchronize return_prompt_top_n_logprobs with skip_prompt_log_probs."""

        if self.return_prompt_top_n_logprobs:
            warnings.warn(
                "return_prompt_top_n_logprobs is deprecated, use skip_prompt_log_probs instead",
                DeprecationWarning,
            )
            assert (
                not self.skip_prompt_log_probs
            ), "return_prompt_top_n_logprobs requires skip_prompt_log_probs to be False"
        if self.top_n_logprobs > 0:
            self.return_prompt_top_n_logprobs = not self.skip_prompt_log_probs
        else:
            self.return_prompt_top_n_logprobs = False

    def _validate_streaming_interval(self):
        """Validate the minimum number of tokens emitted in a streaming delta."""
        if (
            isinstance(self.streaming_interval, bool)
            or not isinstance(self.streaming_interval, int)
            or self.streaming_interval < 1
        ):
            raise ValueError("streaming_interval must be an integer greater than or equal to 1")

    def add_attributes(self, attribute_value_pair: dict):
        """Utility to add more attributes to sampling params

        Use this method to pass in a custom dictionary to add more sampling parameter attributes.
        c = SamplingParams
        c.add_attributes({'min_length':4, 'eod_id':153})

        Args:
            attribute_value_pair (dict): A dictionary containing attributes as the key names and
            their values as the values.
        """
        for key, value in attribute_value_pair.items():
            setattr(self, key, value)

        # Synchronize fields after setting attributes
        self._sync_prompt_logprobs_fields()
        self._validate_streaming_interval()

    def serialize(self) -> dict:
        """Return a dictionary that is msgpack-serializable."""
        return self.__dict__.copy()

    @classmethod
    def deserialize(cls, data: dict) -> "SamplingParams":
        """Construct SamplingParams from a msgpack-compatible dictionary."""
        obj = cls()
        obj.add_attributes(data)
        return obj
