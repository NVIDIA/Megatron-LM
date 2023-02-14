# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.parallel_transformer_block import ParallelTransformerBlock
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_embedding import GPTEmbedding


class GPTModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp_16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
    ):
        super(GPTModel, self).__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp_16_lm_cross_entropy = fp_16_lm_cross_entropy
        self.parallel_output = parallel_output

        # Embeddings.
        if self.pre_process:
            self.embedding = GPTEmbedding(
                config=self.config, vocab_size=self.vocab_size, max_sequence_length=self.max_sequence_length,
            )
            self._embedding_key = 'embedding'

        # Transformer.
        self.transformer_block = ParallelTransformerBlock(
            config=self.config,
            self_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self._encoder_key = 'encoder'

        self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt'
        self.transformer_block.set_input_tensor(input_tensor[0])

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
            encoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # encoder will get hidden_states from encoder.input_tensor
            encoder_input = None

        # Run encoder.
        hidden_states = self.transformer_block(
            hidden_states=encoder_input, attention_mask=attention_mask, inference_params=inference_params
        )

        if self.post_process:
            logits = self.post_language_model_processing(
                hidden_states=hidden_states, labels=labels, logit_weights=self.word_embeddings_weight(),
            )
            return logits

        return hidden_states

    def parallel_lm_logits(
        self, input_: Tensor, word_embeddings_weight: Tensor, bias: Tensor = None,
    ):
        """LM logits using word embedding weights."""
        # Parallel logits.
        if self.config.async_tensor_model_parallel_allreduce or self.config.sequence_parallel_enabled:
            input_parallel = input_
            model_parallel = parallel_state.get_tensor_model_parallel_world_size() > 1
            async_grad_allreduce = (
                self.config.async_tensor_model_parallel_allreduce
                and model_parallel
                and not self.config.sequence_parallel_enabled
            )
        else:
            input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)
            async_grad_allreduce = False

        # Matrix multiply.
        logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=word_embeddings_weight,
            bias=bias,
            gradient_accumulation_fusion=self.config.gradient_accumulation_fusion,
            async_grad_allreduce=async_grad_allreduce,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )

        # Gather if needed.
        if self.parallel_output:
            return logits_parallel
        else:
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)

        return logits

    def post_language_model_processing(self, hidden_states: Tensor, labels: Tensor, logit_weights: Tensor):

        # Output. Format [s b h]
        output = self.parallel_lm_logits(hidden_states, logit_weights)

        if labels is None:
            # [s b h] => [b s h]
            return output.transpose(0, 1).contiguous()
        else:
            # [b s] => [s b]
            labels = labels.transpose(0, 1).contiguous()
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

            # [s b] => [b, s]
            loss = loss.transpose(0, 1).contiguous()
            return loss

    def initialize_word_embeddings(self):

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.
        if self.config.pipeline_model_parallel_size == 1:
            return

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
        if parallel_state.is_pipeline_last_stage() and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                self.vocab_size,
                self.config.hidden_size,
                init_method=self.config.init_method(self.config.init_method_std),
                params_dtype=self.config.params_dtype,
                use_cpu_initialization=self.config.use_cpu_initialization,
                perform_initialization=self.config.perform_initialization,
            )
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True) and self.pre_process:
            self.transformer_block.embedding.zero_parameters()

        if not torch.distributed.is_initialized():
            # TODO: this should be log not print
            if not getattr(MegatronModule, "embedding_warning_printed", False):
                print(
                    "WARNING! Distributed processes aren't initialized, so "
                    "word embeddings in the last layer are not initialized. "
                    "If you are just manipulating a model this is fine, but "
                    "this needs to be handled manually. If you are training "
                    "something is definitely wrong."
                )
                MegatronModule.embedding_warning_printed = True
            return

        # Ensure that first and last stages have the same initial parameter
        # values.
        if parallel_state.is_rank_in_embedding_group():
            torch.distributed.all_reduce(
                self.word_embeddings_weight().data, group=parallel_state.get_embedding_group()
            )

    def word_embeddings_weight(self):
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        else:
            if not self.share_word_embeddings:
                raise Exception(
                    'word_embeddings_weight() called for last ' 'stage, but share_word_embeddings is false'
                )
            return self.word_embeddings.weight

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
