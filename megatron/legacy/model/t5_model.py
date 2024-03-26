# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""T5 model."""

import torch

from megatron.training import get_args
from megatron.core import tensor_parallel
from megatron.legacy.model.enums import AttnMaskType
from megatron.legacy.model.language_model import parallel_lm_logits, get_language_model
from megatron.legacy.model import LayerNorm
from megatron.legacy.model.utils import (
    openai_gelu,
    get_linear_layer
)
from .module import MegatronModule


def t5_extended_attention_mask(attention_mask_list):

    def attn_mask_postprocess(attn_mask):
        # [b, 1, s, s]
        extended_attention_mask = attn_mask.unsqueeze(1)
        return extended_attention_mask

    return [attn_mask_postprocess(attn_mask) for attn_mask in attention_mask_list]


def t5_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class T5LMHead(MegatronModule):
    """Masked LM head for T5

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, parallel_output):
        super(T5LMHead, self).__init__()

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

    def forward(self, hidden_states, word_embeddings_weight):
        output = parallel_lm_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output


class T5Model(MegatronModule):
    """T5 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 add_encoder=True,
                 add_decoder=True):
        super().__init__(config=config)
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            encoder_attn_mask_type=AttnMaskType.padding,
            pre_process=self.pre_process,
            post_process=self.post_process)

        self.initialize_word_embeddings()

        if self.post_process and self.add_decoder:
            self.lm_head = T5LMHead(
                self.shared_embedding_or_output_weight().size(0),
                parallel_output)
            self._lm_head_key = 'lm_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.legacy.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, encoder_input_ids, decoder_input_ids, encoder_attn_mask,
                decoder_attn_mask, encoder_decoder_attn_mask,
                tokentype_ids=None, lm_labels=None, enc_hidden_states=None):

        # Converting the attention masks to proper parameter settings
        encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask = t5_extended_attention_mask(
            [encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask])

        encoder_position_ids = t5_position_ids(encoder_input_ids)
        decoder_position_ids = t5_position_ids(decoder_input_ids)

        lm_output = self.language_model(encoder_input_ids,
                                        encoder_position_ids,
                                        encoder_attn_mask,
                                        decoder_input_ids,
                                        decoder_position_ids,
                                        decoder_attn_mask,
                                        encoder_decoder_attn_mask,
                                        tokentype_ids=tokentype_ids,
                                        enc_hidden_states=enc_hidden_states)

        if self.post_process and self.add_decoder:
            decoder_output, encoder_output = lm_output
            # Output. [s, b, h]
            lm_logits = self.lm_head(decoder_output,
                                     self.shared_embedding_or_output_weight())

            if lm_labels is None:
                # [s b h] => [b s h]
                return lm_logits.transpose(0,1).contiguous()
            else:
                # [b s] => [s b]
                lm_labels = lm_labels.transpose(0,1).contiguous()
                if self.fp16_lm_cross_entropy:
                    assert lm_logits.dtype == torch.half
                    lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits, lm_labels)
                else:
                    lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits.float(),
                                                                                lm_labels)
                # [s b] => [b s]
                lm_loss = lm_loss.transpose(0,1).contiguous()
            return lm_loss
        elif self.add_decoder and not self.add_encoder:
            decoder_output, encoder_output = lm_output
            return decoder_output
        else:
            encoder_output = lm_output
            return encoder_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        if self.post_process and self.add_decoder:
            state_dict_[self._lm_head_key] \
                = self.lm_head.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
         # Save word_embeddings.
        if self.post_process and not self.pre_process and self.add_decoder:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process and self.add_decoder:
            self.lm_head.load_state_dict(state_dict[self._lm_head_key],
                                         strict=strict)
        # Load word embeddings.
        if self.post_process and not self.pre_process and self.add_decoder:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
