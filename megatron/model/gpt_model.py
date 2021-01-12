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

"""GPT-2 model."""

import torch

from megatron import get_args
from megatron import mpu
from .module import MegatronModule

from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal


def gpt_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


def post_language_model_processing(lm_output, labels, logit_weights,
                                   get_key_value, parallel_output,
                                   forward_method_parallel_output,
                                   fp16_lm_cross_entropy):
    if get_key_value:
        lm_output, presents = lm_output

    # Output.
    if forward_method_parallel_output is not None:
        parallel_output = forward_method_parallel_output
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if get_key_value:
        output = [output, presents]

    if labels is None:
        return output
    else:
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = mpu.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss


class GPTModelBase(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self, num_tokentypes=0, parallel_output=True):
        super(GPTModelBase, self).__init__()
        args = get_args()

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=gpt_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers))

        self.initialize_word_embeddings(init_method_normal)

    def forward(self, gpt_model_input, attention_mask, labels=None,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                forward_method_parallel_output=None):

        kwargs = {'layer_past': layer_past, 'get_key_value': get_key_value}
        if mpu.is_pipeline_first_stage():
            (input_ids, position_ids) = gpt_model_input
            args = [input_ids, position_ids, attention_mask]
            kwargs['tokentype_ids'] = tokentype_ids
        else:
            args = [gpt_model_input, attention_mask]
        lm_output = self.language_model(*args, **kwargs)

        if mpu.is_pipeline_last_stage():
            return post_language_model_processing(
                lm_output, labels,
                self.word_embeddings_weight(),
                get_key_value,
                self.parallel_output,
                forward_method_parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        # Save word_embeddings.
        if mpu.is_pipeline_last_stage() and not mpu.is_pipeline_first_stage():
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if mpu.is_pipeline_last_stage() and not mpu.is_pipeline_first_stage():
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)


class GPTModel(GPTModelBase):

    def __init__(self, num_tokentypes=0, parallel_output=True):
        super(GPTModel, self).__init__(
            num_tokentypes=num_tokentypes,
            parallel_output=parallel_output)

    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                forward_method_parallel_output=None):
        return super(GPTModel, self).forward(
            (input_ids, position_ids),
            attention_mask,
            labels=labels,
            tokentype_ids=tokentype_ids,
            layer_past=layer_past,
            get_key_value=get_key_value,
            forward_method_parallel_output=forward_method_parallel_output)


class GPTModelFirstStage(GPTModelBase):

    def __init__(self, num_tokentypes=0):
        super(GPTModelFirstStage, self).__init__(
            num_tokentypes=num_tokentypes)

    def forward(self, input_ids, position_ids, attention_mask,
                tokentype_ids=None, layer_past=None, get_key_value=False):
        return super(GPTModelFirstStage, self).forward(
            (input_ids, position_ids),
            attention_mask,
            tokentype_ids=tokentype_ids,
            layer_past=layer_past,
            get_key_value=get_key_value)


class GPTModelIntermediateStage(GPTModelBase):

    def __init__(self, num_tokentypes=0):
        super(GPTModelIntermediateStage, self).__init__(
            num_tokentypes=num_tokentypes)

    def forward(self, hidden_state, attention_mask,
                layer_past=None, get_key_value=False):
        return super(GPTModelIntermediateStage, self).forward(
            hidden_state,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value)


class GPTModelLastStage(GPTModelBase):

    def __init__(self, num_tokentypes=0, parallel_output=True):
        super(GPTModelLastStage, self).__init__(
            num_tokentypes=num_tokentypes,
            parallel_output=parallel_output)

    def forward(self, hidden_state, attention_mask, labels=None,
                layer_past=None, get_key_value=False,
                forward_method_parallel_output=None):
        return super(GPTModelLastStage, self).forward(
            hidden_state,
            attention_mask,
            labels=labels,
            layer_past=layer_past,
            get_key_value=get_key_value,
            forward_method_parallel_output=forward_method_parallel_output)
