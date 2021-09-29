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
from megatron import get_args


class InferenceParams:
    
    def __init__(self, micro_batch_size_list, max_sequence_len):

        assert isinstance(micro_batch_size_list, list)
        assert max_sequence_len > 0

        self.micro_batch_size_list = micro_batch_size_list
        self.max_sequence_len = max_sequence_len
        self.allocate_key_value_memory = False
        self.micro_batch_size_index = 0


def forward_step(model, tokens, position_ids, attention_mask, inference_params):

    # Hidden size changes when not using recompute, need to tell p2p_communicate
    # functions the correct size
    args = get_args()
    orig_seq_length = args.seq_length
    args.seq_length = tokens.shape[1]
    args.micro_batch_size = tokens.shape[0]

    input_tensor = recv_forward()

    # Forward pass through the model.
    model.set_input_tensor(input_tensor)
    output_tensor = model(tokens, position_ids, attention_mask,
                          inference_params=inference_params)

    send_forward(output_tensor)

    args.seq_length = orig_seq_length

    return output_tensor




