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

"""Forward step utilities."""



import torch

from megatron.p2p_communication import recv_forward, send_forward
from .sampling import sample
from megatron import mpu
import torch.nn.functional as F
from megatron import print_rank_0
from megatron import get_args, get_tokenizer
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from .communication import (
    broadcast_float_list,
    copy_from_last_to_first_pipeline_stage,
    broadcast_from_last_pipeline_stage)
from .tokenization import tokenize_prompts
# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module


def forward_step(model, tokens, position_ids, attention_mask,
                 set_inference_key_value_memory=False,
                 inference_max_sequence_len=None):

    # Hidden size changes when not using recompute, need to tell p2p_communicate
    # functions the correct size
    args = get_args()
    orig_seq_length = args.seq_length
    args.seq_length = tokens.shape[1]
    args.micro_batch_size = tokens.shape[0]

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor = model(
        tokens, position_ids, attention_mask,
        set_inference_key_value_memory=set_inference_key_value_memory,
        inference_max_sequence_len=inference_max_sequence_len)

    send_forward(output_tensor)

    args.seq_length = orig_seq_length

    return output_tensor




