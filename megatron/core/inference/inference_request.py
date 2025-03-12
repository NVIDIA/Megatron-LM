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


@dataclass(kw_only=True)
class InferenceRequest:
    """Class for one inference request

    Containing relevant data for an inference request

    """

    request_id: str
    prompt: str
    inference_parameters: Optional[SamplingParams] = None
    prompt_tokens: Optional[List[int]] = None
    arrival_time: Optional[float] = None
    status: Optional[Status] = None
    encoder_prompt: Optional[str] = None
    generated_text: Optional[str] = None
    segments: Optional[List[str]] = None
    generated_segments: Optional[List[str]] = None
    generated_sequence_lengths: Optional[List[int]] = None
    generated_tokens: Optional[torch.Tensor] = None
    generated_log_probs: Optional[torch.Tensor] = None
    generated_length: Optional[int] = None


@dataclass(kw_only=True)
class VLMInferenceRequest(InferenceRequest):
    """Class for a VLM inference request"""

    num_img_embeddings_per_tile: int
    imgs: torch.Tensor
    num_tiles: torch.Tensor
    decoder_seq_length: int
