# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.models.gpt.gpt_embedding import GPTEmbedding


class GPTModel(MegatronModule):
    """Transformer language model.

    Arguments:
        config (TransformerConfig): transformer config

        vocab_size (int): vocabulary size

        max_sequence_length (int): maximum size of sequence. This is used for positional embedding

        pre_process (bool): Include embedding layer (used with pipeline parallelism)
        post_process (bool): Include an output layer (used with pipeline parallelism)

        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks

        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are
            shared. Defaults to False.

    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ):
        super(GPTModel, self).__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        # Embeddings.
        if self.pre_process:
            self.embedding = GPTEmbedding(
                config=self.config, vocab_size=self.vocab_size, max_sequence_length=self.max_sequence_length,
            )

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            self_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights)

        if self.share_embeddings_and_output_weights and (self.pre_process or self.post_process):
            self.initialize_last_stage_with_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt'
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor = None,
        inference_params=None,
    ):

        # Encoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # encoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Run encoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input, attention_mask=attention_mask, inference_params=inference_params
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss

    def shared_embedding_or_output_weight(self):
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def initialize_last_stage_with_word_embeddings(self):

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism and sharing word
        # embeddings. Nothing to do if we aren't sharing weights or aren't using
        # pipeline parallelism.
        if not self.share_embeddings_and_output_weights or (self.pre_process and self.post_process):
            return

        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.output_layer.weight.data.fill_(0)
            self.output_layer.weight.shared = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if parallel_state.is_rank_in_embedding_group():
                weight = self.shared_embedding_or_output_weight()
                torch.distributed.all_reduce(weight.data, group=parallel_state.get_embedding_group())

        elif not getattr(GPTModel, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            GPTModel.embedding_warning_printed = True

    # TODO: add distributed checkpointing
    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        pass
        # """For easy load."""

        # state_dict_ = {}
        # if self.pre_process:
        #     state_dict_[self._embedding_key] = self.embedding.state_dict_for_save_checkpoint(
        #         prefix=prefix, keep_vars=keep_vars
        #     )
        # state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(
        #     prefix=prefix, keep_vars=keep_vars
        # )

        # return state_dict_

    # TODO: add distributed checkpointing
    def load_state_dict(self, state_dict, strict=True):
        pass
        # """Customized load."""

        # # Embedding.
        # if self.pre_process:
        #     if self._embedding_key in state_dict:
        #         state_dict_ = state_dict[self._embedding_key]
        #     else:
        #         # for backward compatibility.
        #         state_dict_ = {}
        #         for key in state_dict.keys():
        #             if '_embeddings' in key:
        #                 state_dict_[key] = state_dict[key]
        #     self.embedding.load_state_dict(state_dict_, strict=strict)

        # # Encoder.
        # if self._encoder_key in state_dict:
        #     state_dict_ = state_dict[self._encoder_key]
        # # For backward compatibility.
        # elif 'transformer' in state_dict:
        #     state_dict_ = state_dict['transformer']
        # else:
        #     # For backward compatibility.
        #     state_dict_ = {}
        #     for key in state_dict.keys():
        #         if 'transformer.' in key:
        #             state_dict_[key.split('transformer.')[1]] = state_dict[key]

        # # For backward compatibility.
        # state_dict_self_attention = {}
        # for key in state_dict_.keys():
        #     if '.attention.' in key:
        #         state_dict_self_attention[key.replace(".attention.", ".self_attention.")] = state_dict_[key]
        #     else:
        #         state_dict_self_attention[key] = state_dict_[key]
        # state_dict_ = state_dict_self_attention

        # self.encoder.load_state_dict(state_dict_, strict=strict)
