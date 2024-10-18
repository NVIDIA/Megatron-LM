# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from enum import Enum
from typing import List

import torch

from megatron.core.inference.common_inference_params import CommonInferenceParams


# class syntax
class Status(Enum):
    """Enum for status"""

    WAITING_IN_QUEUE = 1
    ACTIVE_AND_GENERATING_TOKENS = 2
    ACTIVE_BUT_NOT_GENERATING_TOKENS = 3
    COMPLETED = 4


@dataclass
class InferenceRequest:
    """Class for one inference request

    Containing relevant data for an inference request

    """

    request_id: str
    prompt: str
    inference_parameters: CommonInferenceParams
    prompt_tokens: List[int]
    arrival_time: float
    status: Status
    encoder_prompt: str = None
    generated_text: str = None
    generated_tokens: torch.Tensor = None
    generated_log_probs: torch.Tensor = None
    generated_length: int = 0
