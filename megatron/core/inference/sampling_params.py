# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Optional


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
    return_segments: bool = False  # Whether to return individually detokenized tokens
    num_tokens_to_generate: int = 30
    num_tokens_total: Optional[int] = None  # Cannot set both this and num_tokens_to_generate
    termination_id: Optional[int] = None
    top_n_logprobs: int = 0
    return_prompt_top_n_logprobs: bool = False
    add_BOS: bool = False

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

    def serializable(self) -> dict:
        """Return a dictionary that is msgpack-serializable."""
        return self.__dict__.copy()

    @classmethod
    def deserialize(cls, data: dict) -> "SamplingParams":
        """Construct SamplingParams from a msgpack-compatible dictionary."""
        obj = cls()
        obj.add_attributes(data)
        return obj
