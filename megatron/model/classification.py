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

"""Classification model."""

import torch

from megatron import get_args, print_rank_last
from megatron import mpu
from megatron.enums import AttnMaskType
from megatron.model.bert_model import bert_extended_attention_mask, bert_position_ids
from megatron.model.language_model import get_language_model
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule


class Classification(MegatronModule):

    def __init__(self,
                 num_classes,
                 num_tokentypes=2,
                 pre_process=True,
                 post_process=True):
        super(Classification, self).__init__(share_word_embeddings=False)
        args = get_args()

        self.num_classes = num_classes
        self.pre_process = pre_process
        self.post_process = post_process
        init_method = init_method_normal(args.init_method_std)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=True,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers),
            pre_process=self.pre_process,
            post_process=self.post_process)

        # Multi-choice head.
        if self.post_process:
            self.classification_dropout = torch.nn.Dropout(args.hidden_dropout)
            self.classification_head = get_linear_layer(args.hidden_size,
                                                        self.num_classes,
                                                        init_method)
            self._classification_head_key = 'classification_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, model_input, attention_mask, tokentype_ids=None):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)
        input_ids = model_input
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids
        )

        if self.post_process:
            _, pooled_output = lm_output
            classification_output = self.classification_dropout(pooled_output)
            classification_logits = self.classification_head(classification_output)

            # Reshape back to separate choices.
            classification_logits = classification_logits.view(-1, self.num_classes)

            return classification_logits
        return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.post_process:
            state_dict_[self._classification_head_key] \
                = self.classification_head.state_dict(
                    destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            if self._classification_head_key in state_dict:
                self.classification_head.load_state_dict(
                    state_dict[self._classification_head_key], strict=strict)
            else:
                print_rank_last('***WARNING*** could not find {} in the checkpoint, '
                                'initializing to random'.format(
                                    self._classification_head_key))
