# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Multiple choice model."""

import torch

from megatron.model.bert_model import bert_attention_mask_func
from megatron.model.bert_model import bert_extended_attention_mask
from megatron.model.bert_model import bert_position_ids
from megatron.model.language_model import get_language_model
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from megatron.module import MegatronModule
from megatron.utils import print_rank_0


class MultipleChoice(MegatronModule):

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 num_tokentypes=2,
                 apply_query_key_layer_scaling=False,
                 attention_softmax_in_fp32=False):

        super(MultipleChoice, self).__init__()

        init_method = init_method_normal(init_method_std)

        self.language_model, self._language_model_key = get_language_model(
            num_layers=num_layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            embedding_dropout_prob=embedding_dropout_prob,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
            max_sequence_length=max_sequence_length,
            num_tokentypes=num_tokentypes,
            add_pooler=True,
            attention_mask_func=bert_attention_mask_func,
            checkpoint_activations=checkpoint_activations,
            checkpoint_num_layers=checkpoint_num_layers,
            layernorm_epsilon=layernorm_epsilon,
            init_method=init_method,
            scaled_init_method=scaled_init_method_normal(init_method_std,
                                                         num_layers),
            residual_connection_post_layernorm=False,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            attention_softmax_in_fp32=attention_softmax_in_fp32)

        # Multi-choice head.
        self.multichoice_dropout = torch.nn.Dropout(output_dropout_prob)
        self.multichoice_head = get_linear_layer(hidden_size, 1, init_method)
        self._multichoice_head_key = 'multichoice_head'


    def forward(self, input_ids, attention_mask, tokentype_ids):

        # [batch, choices, sequence] --> [batch * choices, sequence] -->
        #    transformer --> [batch, choices] --> softmax

        # Ensure the shape is [batch-size, choices, sequence]
        assert len(input_ids.shape) == 3
        assert len(attention_mask.shape) == 3
        assert len(tokentype_ids.shape) == 3

        # Reshape and treat choice dimension the same as batch.
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        tokentype_ids = tokentype_ids.view(-1, tokentype_ids.size(-1))

        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype)
        position_ids = bert_position_ids(input_ids)

        _, pooled_output = self.language_model(input_ids,
                                               position_ids,
                                               extended_attention_mask,
                                               tokentype_ids=tokentype_ids)

        # Output.
        multichoice_output = self.multichoice_dropout(pooled_output)
        multichoice_logits = self.multichoice_head(multichoice_output)

        # Reshape back to separate choices.
        multichoice_logits = multichoice_logits.view(-1, num_choices)

        return multichoice_logits


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        state_dict_[self._multichoice_head_key] \
            = self.multichoice_head.state_dict(
                destination, prefix, keep_vars)
        return state_dict_


    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self._multichoice_head_key in state_dict:
            self.multichoice_head.load_state_dict(
                state_dict[self._multichoice_head_key], strict=strict)
        else:
            print_rank_0('***WARNING*** could not find {} in the checkpoint, '
                         'initializing to random'.format(
                             self._multichoice_head_key))
