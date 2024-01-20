# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Classification model."""

import torch

from megatron import get_args, print_rank_last
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import get_language_model
from megatron.model.bert_model import bert_extended_attention_mask
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from .module import MegatronModule


class Classification(MegatronModule):

    def __init__(self,
                 config,
                 num_classes,
                 num_tokentypes=0,
                 pre_process=True,
                 post_process=True):
        super().__init__(config=config, share_embeddings_and_output_weights=False)
        args = get_args()

        self.num_classes = num_classes
        self.pre_process = pre_process
        self.post_process = post_process
        if config.init_method is None:
            config.init_method = init_method_normal(config.init_method_std)
        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=0,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process)

        # Classifications head.
        if self.post_process:
            self.classification_dropout = torch.nn.Dropout(args.hidden_dropout)
            self.classification_head = get_linear_layer(args.hidden_size,
                                                        self.num_classes,
                                                        config.init_method)
            self._classification_head_key = 'classification_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, model_input, attention_mask, tokentype_ids=None):
        
        input_ids = model_input
        batch_size, seq_length = input_ids.shape
        
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # build and combine padding and causal masks
        padding_attention_mask = bert_extended_attention_mask(attention_mask)
        causal_attention_mask = torch.triu(torch.ones((seq_length, seq_length), 
                                                      dtype=torch.bool, device=input_ids.device),
                                                      diagonal=1)
        causal_attention_mask = causal_attention_mask.unsqueeze(0).unsqueeze(0)
        causal_attention_mask = causal_attention_mask.expand(batch_size,-1,-1,-1)
        attention_mask = causal_attention_mask & padding_attention_mask
        
        lm_output = self.language_model(
            input_ids,
            position_ids, 
            attention_mask,
            tokentype_ids=None
        )
        if self.post_process:
            # feed the final token representation to classification head
            pooled_output = lm_output[-1]
            classification_output = self.classification_dropout(pooled_output)
            classification_logits = self.classification_head(classification_output)

            # Reshape back to separate choices.
            classification_logits = classification_logits.view(-1, self.num_classes)

            return classification_logits
        return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        if self.post_process:
            state_dict_[self._classification_head_key] \
                = self.classification_head.state_dict(prefix=prefix, keep_vars=keep_vars)
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
