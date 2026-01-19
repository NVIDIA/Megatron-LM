# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SpeculativeMethod(Enum):
    """Enum for speculative decoding methods."""

    MTP = "mtp"  # Multi-Token Prediction


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Speculative decoding is a technique to accelerate inference by predicting multiple
    future tokens in parallel using a draft mechanism, then verifying them with the
    main model. This reduces the number of sequential forward passes needed.

    Currently, only MTP (Multi-Token Prediction) method is supported. MTP uses additional
    prediction heads trained alongside the main model to predict future tokens.

    Attributes:
        method (SpeculativeMethod): The speculative decoding method to use.
            Currently only "mtp" is supported.
        num_speculative_tokens (int): Number of speculative tokens to predict per step.
            Must be >= 1 and <= model's mtp_num_layers. Default is 1.
        use_greedy_verification (bool): If True, use greedy sampling for verification.
            If False, use the same sampling strategy as the main generation. Default is True.
    """

    method: SpeculativeMethod = SpeculativeMethod.MTP
    num_speculative_tokens: int = 1
    use_greedy_verification: bool = True

    def __post_init__(self):
        """Validate speculative config parameters."""
        if isinstance(self.method, str):
            self.method = SpeculativeMethod(self.method.lower())

        if self.num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {self.num_speculative_tokens}"
            )

        if self.method != SpeculativeMethod.MTP:
            raise ValueError(
                f"Only MTP method is currently supported, got {self.method}"
            )

    def serialize(self) -> dict:
        """Return a dictionary that is msgpack-serializable."""
        return {
            "method": self.method.value,
            "num_speculative_tokens": self.num_speculative_tokens,
            "use_greedy_verification": self.use_greedy_verification,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SpeculativeConfig":
        """Construct SpeculativeConfig from a msgpack-compatible dictionary."""
        return cls(
            method=SpeculativeMethod(data.get("method", "mtp")),
            num_speculative_tokens=data.get("num_speculative_tokens", 1),
            use_greedy_verification=data.get("use_greedy_verification", True),
        )


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
    num_tokens_to_generate: int = 30
    num_tokens_total: Optional[int] = None  # Cannot set both this and num_tokens_to_generate
    termination_id: Optional[int] = None
    top_n_logprobs: int = 0
    return_prompt_top_n_logprobs: bool = False  # Deprecated field for backwards compatibility
    add_BOS: bool = False
    stop_words: Optional[List[str]] = (
        None  # List of strings that will stop generation when produced
    )
    speculative_config: Optional[SpeculativeConfig] = None  # Config for speculative decoding

    def __post_init__(self):
        """Ensure backward compatibility for return_prompt_top_n_logprobs.

        Sets return_prompt_top_n_logprobs based on skip_prompt_log_probs and top_n_logprobs:
        - return_prompt_top_n_logprobs = not skip_prompt_log_probs and top_n_logprobs > 0
        """
        self._sync_prompt_logprobs_fields()

    def _sync_prompt_logprobs_fields(self):
        """Synchronize return_prompt_top_n_logprobs with skip_prompt_log_probs."""

        if self.return_prompt_top_n_logprobs:
            warnings.warn(
                "return_prompt_top_n_logprobs is deprecated, use skip_prompt_log_probs instead",
                DeprecationWarning,
            )
            assert (
                self.skip_prompt_log_probs
            ), "return_prompt_top_n_logprobs requires skip_prompt_log_probs to be False"
        if self.top_n_logprobs > 0:
            self.return_prompt_top_n_logprobs = not self.skip_prompt_log_probs
        else:
            self.return_prompt_top_n_logprobs = False

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

    def serialize(self) -> dict:
        """Return a dictionary that is msgpack-serializable."""
        result = self.__dict__.copy()
        # Handle speculative_config serialization
        if self.speculative_config is not None:
            result["speculative_config"] = self.speculative_config.serialize()
        return result

    @classmethod
    def deserialize(cls, data: dict) -> "SamplingParams":
        """Construct SamplingParams from a msgpack-compatible dictionary."""
        # Handle speculative_config deserialization
        spec_config_data = data.pop("speculative_config", None)
        if spec_config_data is not None:
            data["speculative_config"] = SpeculativeConfig.deserialize(spec_config_data)
        obj = cls()
        obj.add_attributes(data)
        return obj
