# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch

from megatron.core.inference.sampling_params import SamplingParams


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
    inference_parameters: SamplingParams
    prompt_tokens: List[int]
    arrival_time: float
    status: Status
    prompt_log_probs: Optional[float] = None
    encoder_prompt: Optional[str] = None
    generated_text: Optional[str] = None
    generated_segments: Optional[List[List[str]]] = None
    generated_sequence_lengths: Optional[List[int]] = None
    generated_tokens: Optional[torch.Tensor] = None
    generated_log_probs: Optional[float] = None
    generated_length: int = 0
