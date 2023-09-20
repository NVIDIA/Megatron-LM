# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""BERT model."""

import torch

from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits
from megatron.model.language_model import get_language_model
from megatron.model.utils import get_norm
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule


def bert_extended_attention_mask(attention_mask):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)

    # Convert attention mask to binary:
    extended_attention_mask = (extended_attention_mask < 0.5)

    return extended_attention_mask

def bert_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class BertLMHead(MegatronModule):
    """Masked LM head for Bert

    Arguments:
        config: TransformerConfig object
        mpu_vocab_size: model parallel size of vocabulary.
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, config, parallel_output):
        super().__init__(config=config)

        args = get_args()
        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        tensor_parallel.set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
        self.parallel_output = parallel_output

        self.dense = get_linear_layer(config.hidden_size, config.hidden_size, config.init_method)
        setattr(self.dense.weight, 'sequence_parallel', config.sequence_parallel)
        setattr(self.dense.bias, 'sequence_parallel', config.sequence_parallel)

        self.norm = get_norm(config)
        self.gelu = torch.nn.functional.gelu
        if args.openai_gelu:
            self.gelu = openai_gelu
        elif args.onnx_safe:
            self.gelu = erf_gelu

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.norm(hidden_states)
        output = parallel_lm_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output

    def load_state_dict(self, state_dict, strict=True):
        """Customize load."""

        # Handle renaming layernorm -> norm in component names
        state_dict_ = {}
        for key in state_dict.keys():
            newkey = key.replace("layernorm", "norm")
            state_dict_[newkey] = state_dict[key]

        super().load_state_dict(state_dict_, strict)


def post_language_model_processing(lm_output, pooled_output,
                                   lm_head, binary_head,
                                   lm_labels,
                                   logit_weights,
                                   fp16_lm_cross_entropy):
    # Output.
    lm_logits = lm_head(
        lm_output, logit_weights)

    binary_logits = None
    if binary_head is not None:
        binary_logits = binary_head(pooled_output)

    if lm_labels is None:
        # [s b h] => [b s h]
        return lm_logits.transpose(0,1).contiguous(), binary_logits
    else:
        # [b s] => [s b]
        lm_labels = lm_labels.transpose(0,1).contiguous()
        # lm_logits : [s, b, h] and lm_labels: [s, b]
        if fp16_lm_cross_entropy:
            assert lm_logits.dtype == torch.half
            lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits, lm_labels)
        else:
            lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits.float(),
                                                        lm_labels)
        # [s, b] => [b s]
        lm_loss = lm_loss.transpose(0,1).contiguous()
        return lm_loss, binary_logits


class BertModel(MegatronModule):
    """Bert Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=2,
                 add_binary_head=True,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        super().__init__(config=config)
        args = get_args()

        # TODO this option is not yet implemented in BERT
        assert args.untie_embeddings_and_output_weights is False

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process

        self.return_embeddings = args.output_bert_embeddings
        if self.return_embeddings:
            assert self.post_process and self.add_binary_head

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=self.add_binary_head,
            encoder_attn_mask_type=AttnMaskType.padding,
            pre_process=self.pre_process,
            post_process=self.post_process)

        self.initialize_word_embeddings()
        if self.post_process:
            self.lm_head = BertLMHead(self.shared_embedding_or_output_weight().size(0), config, parallel_output)
            self._lm_head_key = 'lm_head'
            self.binary_head = None
            if self.add_binary_head:
                self.binary_head = get_linear_layer(config.hidden_size, 2,
                                                    config.init_method)
                self._binary_head_key = 'binary_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, bert_model_input, attention_mask,
                tokentype_ids=None, lm_labels=None):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)
        input_ids = bert_model_input
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids
        )

        if self.post_process and self.add_binary_head:
            lm_output, pooled_output = lm_output

            # Return pooled output (e.g., when computing Bert embeddings).
            if self.return_embeddings:

                # Sum attention mask.
                embeddings = torch.transpose(lm_output, 0, 1)
                masks = torch.sum(attention_mask, dim=1)

                # Collect masked embeddings.
                output = torch.zeros(
                    size=(embeddings.shape[0], embeddings.shape[2]),
                    dtype=torch.float32,
                    device=torch.cuda.current_device())
                for i, (embedding, mask) in enumerate(zip(embeddings, masks)):
                    output[i, :] = torch.mean(embedding[1: mask - 1], dim=0)

                return output

        else:
            pooled_output = None

        if self.post_process:
            return post_language_model_processing(lm_output, pooled_output,
                                                  self.lm_head, self.binary_head,
                                                  lm_labels,
                                                  self.shared_embedding_or_output_weight(),
                                                  self.fp16_lm_cross_entropy)
        else:
            return lm_output


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        if self.post_process:
            state_dict_[self._lm_head_key] \
                = self.lm_head.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
        if self.post_process and self.add_binary_head:
            state_dict_[self._binary_head_key] \
                = self.binary_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            self.lm_head.load_state_dict(
                state_dict[self._lm_head_key], strict=strict)
        if self.post_process and self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict)
        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
