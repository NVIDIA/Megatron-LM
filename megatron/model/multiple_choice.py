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

"""Multiple choice model."""

import torch

from megatron import get_args, print_rank_last
from megatron import mpu
from megatron.model.bert_model import bert_attention_mask_func, bert_extended_attention_mask, bert_position_ids
from megatron.model.language_model import get_language_model
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule


class MultipleChoiceBase(MegatronModule):

    def __init__(self, num_tokentypes=2):
        super(MultipleChoiceBase, self).__init__(share_word_embeddings=False)
        args = get_args()

        init_method = init_method_normal(args.init_method_std)

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=True,
            init_method=init_method,
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers))

        # Multi-choice head.
        if mpu.is_pipeline_last_stage():
            self.multichoice_dropout = torch.nn.Dropout(args.hidden_dropout)
            self.multichoice_head = get_linear_layer(args.hidden_size, 1,
                                                     init_method)
            self._multichoice_head_key = 'multichoice_head'

    def forward(self, model_input, attention_mask, tokentype_ids=None):

        # [batch, choices, sequence] --> [batch * choices, sequence] -->
        #    transformer --> [batch, choices] --> softmax

        # Ensure the shape is [batch-size, choices, sequence]
        assert len(attention_mask.shape) == 3
        num_choices = attention_mask.shape[1]

        # Reshape and treat choice dimension the same as batch.
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        kwargs = {}
        if mpu.is_pipeline_first_stage():
            input_ids = model_input
            # Do the same as attention_mask for input_ids, tokentype_ids
            assert len(input_ids.shape) == 3
            assert len(tokentype_ids.shape) == 3
            input_ids = input_ids.view(-1, input_ids.size(-1))
            tokentype_ids = tokentype_ids.view(-1, tokentype_ids.size(-1))

            position_ids = bert_position_ids(input_ids)
            args = [input_ids, position_ids, extended_attention_mask]
            kwargs['tokentype_ids'] = tokentype_ids
        else:
            args = [model_input, extended_attention_mask]
        lm_output = self.language_model(*args, **kwargs)
        if mpu.is_pipeline_last_stage():
            _, pooled_output = lm_output
            multichoice_output = self.multichoice_dropout(pooled_output)
            multichoice_logits = self.multichoice_head(multichoice_output)

            # Reshape back to separate choices.
            multichoice_logits = multichoice_logits.view(-1, num_choices)

            return multichoice_logits
        return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if mpu.is_pipeline_last_stage():
            state_dict_[self._multichoice_head_key] \
                = self.multichoice_head.state_dict(
                    destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if mpu.is_pipeline_last_stage():
            if self._multichoice_head_key in state_dict:
                self.multichoice_head.load_state_dict(
                    state_dict[self._multichoice_head_key], strict=strict)
            else:
                print_rank_last('***WARNING*** could not find {} in the checkpoint, '
                                'initializing to random'.format(
                                    self._multichoice_head_key))

class MultipleChoice(MultipleChoiceBase):

    def __init__(self, num_tokentypes=2):
        super(MultipleChoice, self).__init__(
            num_tokentypes=num_tokentypes)

    def forward(self, input_ids, attention_mask,
                tokentype_ids=None):
        return super(MultipleChoice, self).forward(
            input_ids,
            attention_mask,
            tokentype_ids=tokentype_ids)


class MultipleChoiceFirstStage(MultipleChoiceBase):

    def __init__(self, num_tokentypes=2):
        super(MultipleChoiceFirstStage, self).__init__(
            num_tokentypes=num_tokentypes)

    def forward(self, input_ids, attention_mask,
                tokentype_ids=None):
        return super(MultipleChoiceFirstStage, self).forward(
            input_ids,
            attention_mask,
            tokentype_ids=tokentype_ids)


class MultipleChoiceIntermediateStage(MultipleChoiceBase):

    def __init__(self, num_tokentypes=2):
        super(MultipleChoiceIntermediateStage, self).__init__(
            num_tokentypes=num_tokentypes)

    def forward(self, hidden_state, attention_mask):
        return super(MultipleChoiceIntermediateStage, self).forward(
            hidden_state,
            attention_mask)


class MultipleChoiceLastStage(MultipleChoiceBase):

    def __init__(self, num_tokentypes=2):
        super(MultipleChoiceLastStage, self).__init__(
            num_tokentypes=num_tokentypes)

    def forward(self, hidden_state, attention_mask):
        return super(MultipleChoiceLastStage, self).forward(
            hidden_state,
            attention_mask)
