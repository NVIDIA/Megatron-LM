# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference API."""


import torch

from .communication import broadcast_float_list
from .generation import generate_tokens_probs_and_return_on_first_stage
from .tokenization import tokenize_prompts


def generate(model,
             prompts=None,
             tokens_to_generate=0,
             return_output_log_probs=False,
             return_all_log_probs=False,
             temperature=1.0):
    """TO DO ..."""

    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate, return_output_log_probs,
              return_all_log_probs, temperature]
    values_float_tensor = broadcast_float_list(4, float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    return_all_log_probs = bool(values_float_tensor[2].item())
    temperature = values_float_tensor[2].item()

    # Tokenize prompts and get the batch.
    # Note that these tensors are broadcaseted to all ranks.
    if torch.distributed.get_rank() == 0:
        assert prompts is not None
    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate)

    # Main inference function.
    # Note that the outputs are available on the first stage.
    return generate_tokens_probs_and_return_on_first_stage(
        model, context_tokens_tensor, context_length_tensor,
        return_output_log_probs=return_output_log_probs,
        return_all_log_probs=return_all_log_probs,
        temperature=temperature)
