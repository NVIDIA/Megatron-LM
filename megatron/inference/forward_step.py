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

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable

import torch

from megatron import (
    get_args,
    mpu)
from megatron.p2p_communication import (
    recv_forward,
    send_forward)


def forward_step_provider(model,
                          batch_size,
                          micro_batch_size,
                          max_sequence_len):

    args = get_args()

    if args.pipeline_model_parallel_size == 1 or micro_batch_size >= batch_size:
        return NoPipeliningForwardStep(model, batch_size, max_sequence_len)

    return SimplePipeliningForwardStep(model, batch_size,
                                       micro_batch_size,
                                       max_sequence_len)



class InferenceParams:

    def __init__(self, micro_batch_size_list, max_sequence_len):

        assert isinstance(micro_batch_size_list, list)
        assert max_sequence_len > 0

        self.micro_batch_size_list = micro_batch_size_list
        self.max_sequence_len = max_sequence_len
        self.allocate_key_value_memory = True
        self.micro_batch_index = 0


class ForwardStepBase(ABC):

    def __init__(self, model):

        if isinstance(model, Iterable):
            for this_model in model:
                this_model.eval()
        else:
            model.eval()
        self.model = model

    @abstractmethod
    def __call__(self, tokens, position_ids, attention_mask):
        pass



class SimplePipeliningForwardStep(ForwardStepBase):

    def __init__(self, model, batch_size, micro_batch_size, max_sequence_len):
        super().__init__(model)

        self.batch_size = batch_size
        # Divide the batch dimension into micro batches.
        self.num_micro_batches, last_chunk = divmod(batch_size,
                                                    micro_batch_size)
        self.micro_batch_size_list = []
        self.batch_dim_start_index = [0]
        for i in range(self.num_micro_batches):
            self.micro_batch_size_list.append(micro_batch_size)
            self.batch_dim_start_index.append((i + 1) * micro_batch_size)
        if last_chunk > 0:
            self.num_micro_batches += 1
            self.micro_batch_size_list.append(last_chunk)
            self.batch_dim_start_index.append(batch_size)

        self.inference_params = InferenceParams(self.micro_batch_size_list,
                                                max_sequence_len)


    def __call__(self, tokens, position_ids, attention_mask):

        # Need to tell p2p_communicate functions the correct size.
        args = get_args()
        orig_seq_length = args.seq_length
        args.seq_length = tokens.size(1)
        assert args.seq_length <= self.inference_params.max_sequence_len

        # Preallocate memory for output logits.
        logits = None
        if mpu.is_pipeline_last_stage():
            logits = torch.empty(tokens.size(0),
                                 tokens.size(1),
                                 args.padded_vocab_size,
                                 dtype=torch.float32,
                                 device=torch.cuda.current_device())

        # Pileline using micro batches.
        for micro_batch_index in range(self.num_micro_batches):
            # Set micro-batch size and index.
            self.inference_params.micro_batch_index = micro_batch_index
            args.micro_batch_size = self.micro_batch_size_list[
                micro_batch_index]
            # Slice among the batch dimenion.
            start = self.batch_dim_start_index[micro_batch_index]
            end = self.batch_dim_start_index[micro_batch_index + 1]
            tokens2use = tokens[start:end, ...]
            position_ids2use = position_ids[start:end, ...]

            # Receive from previous stage.
            input_tensor = recv_forward()

            # Forward pass through the model.
            self.model.set_input_tensor(input_tensor)
            output_tensor = self.model(tokens2use, position_ids2use,
                                       attention_mask,
                                       inference_params=self.inference_params)

            # Send output to the next stage.
            send_forward(output_tensor)

            # Reset the sequence lenght to whatwever it was before.
            # Make sure we do not allocate context memory anymore.
            if self.inference_params.allocate_key_value_memory:
                self.inference_params.allocate_key_value_memory = False

            if mpu.is_pipeline_last_stage():
                logits[start:end, ...] = output_tensor

        # Adjust the sequence length back to whatever it was before.
        args.seq_length = orig_seq_length

        return logits



class NoPipeliningForwardStep(ForwardStepBase):

    def __init__(self, model, batch_size, max_sequence_len):
        super().__init__(model)

        self.inference_params = InferenceParams([batch_size], max_sequence_len)


    def __call__(self, tokens, position_ids, attention_mask):

        # Need to tell p2p_communicate functions the correct size.
        args = get_args()
        orig_seq_length = args.seq_length
        args.seq_length = tokens.shape[1]
        assert args.seq_length <= self.inference_params.max_sequence_len
        args.micro_batch_size = tokens.shape[0]
        assert self.inference_params.micro_batch_size_list[0] == tokens.shape[0]
        assert self.inference_params.micro_batch_index == 0

        # Receive from previous stage.
        input_tensor = recv_forward()

        # Forward pass through the model.
        self.model.set_input_tensor(input_tensor)
        output_tensor = self.model(tokens, position_ids, attention_mask,
                                   inference_params=self.inference_params)

        # Send output to the next stage.
        send_forward(output_tensor)

        # Reset the sequence lenght to whatwever it was before.
        args.seq_length = orig_seq_length
        # Make sure we do not allocate context memory anymore.
        if self.inference_params.allocate_key_value_memory:
            self.inference_params.allocate_key_value_memory = False

        return output_tensor
