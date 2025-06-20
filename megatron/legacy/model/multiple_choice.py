# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Multiple choice model."""

import torch

from megatron.training import get_args, print_rank_last
from megatron.legacy.model.enums import AttnMaskType
from megatron.legacy.model.bert_model import bert_extended_attention_mask, bert_position_ids
from megatron.legacy.model.language_model import get_language_model
from megatron.legacy.model.utils import get_linear_layer
from megatron.legacy.model.utils import init_method_normal
from megatron.legacy.model.utils import scaled_init_method_normal
from .module import MegatronModule


class MultipleChoice(MegatronModule):

    def __init__(self,
                 config,
                 num_tokentypes=2,
                 pre_process=True,
                 post_process=True):
        super(MultipleChoice, self).__init__(share_embeddings_and_output_weights=False)
        args = get_args()

        self.pre_process = pre_process
        self.post_process = post_process

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=True,
            encoder_attn_mask_type=AttnMaskType.padding,
            pre_process=self.pre_process,
            post_process=self.post_process)

        # Multi-choice head.
        if self.post_process:
            self.multichoice_dropout = torch.nn.Dropout(args.hidden_dropout)
            self.multichoice_head = get_linear_layer(args.hidden_size, 1,
                                                     init_method)
            self._multichoice_head_key = 'multichoice_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.legacy.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, model_input, attention_mask, tokentype_ids=None):

        # [batch, choices, sequence] --> [batch * choices, sequence] -->
        #    transformer --> [batch, choices] --> softmax

        # Ensure the shape is [batch-size, choices, sequence]
        assert len(attention_mask.shape) == 3
        num_choices = attention_mask.shape[1]

        # Reshape and treat choice dimension the same as batch.
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        input_ids = model_input
        # Do the same as attention_mask for input_ids, tokentype_ids
        assert len(input_ids.shape) == 3
        assert len(tokentype_ids.shape) == 3
        input_ids = input_ids.view(-1, input_ids.size(-1))
        tokentype_ids = tokentype_ids.view(-1, tokentype_ids.size(-1))
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids
        )
        if self.post_process:
            _, pooled_output = lm_output
            multichoice_output = self.multichoice_dropout(pooled_output)
            multichoice_logits = self.multichoice_head(multichoice_output)

            # Reshape back to separate choices.
            multichoice_logits = multichoice_logits.view(-1, num_choices)

            return multichoice_logits
        return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        if self.post_process:
            state_dict_[self._multichoice_head_key] \
                = self.multichoice_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            if self._multichoice_head_key in state_dict:
                self.multichoice_head.load_state_dict(
                    state_dict[self._multichoice_head_key], strict=strict)
            else:
                print_rank_last('***WARNING*** could not find {} in the checkpoint, '
                                'initializing to random'.format(
                                    self._multichoice_head_key))
