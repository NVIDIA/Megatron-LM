# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Literal, Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.base_embedding import BaseEmbedding
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


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

        position_embedding_type (string): Position embedding type. Options ['learned_absolute', 'rope'].
            Defaults is 'learned_absolute'.

        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
            Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.

        seq_len_interpolation_factor (float): scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
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
        position_embedding_type: Literal['learned_absolute',
                                         'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
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
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        self.embedding = None
        if self.pre_process:
            self.embedding = BaseEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor
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
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
            )

        if self.share_embeddings_and_output_weights and (self.pre_process or self.post_process):
            self.initialize_last_stage_with_word_embeddings(GPTModel)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params=None,
    ):
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.embedding is not None and self.position_embedding_type == 'rope':
            rotary_pos_emb = self.embedding.get_rotary_pos_emb(
                inference_params, self.decoder, decoder_input, self.config)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
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

        loss = self.compute_loss(labels, logits)

        return loss

    def shared_embedding_or_output_weight(self):
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def sharded_state_dict(self, prefix=''):
        sharded_state_dict = {}

        if self.pre_process:
            embedding_prefix = f'{prefix}embedding.'
            embedding_sharded_state_dict = self.embedding.sharded_state_dict(
                prefix=embedding_prefix
            )
            sharded_state_dict.update(embedding_sharded_state_dict)

        decoder_prefix = f'{prefix}decoder.'
        decoder_sharded_state_dict = self.decoder.sharded_state_dict(
            prefix=decoder_prefix)
        sharded_state_dict.update(decoder_sharded_state_dict)

        if self.post_process:
            output_layer_prefix = f'{prefix}output_layer.'
            output_layer_key = f'{output_layer_prefix}weight'
            if self.share_embeddings_and_output_weights:
                if not self.pre_process:
                    # when sharing embeddings with last stage, we need to use the weights from the first stage
                    # on pipeline first rank, word embeddings are saved to {prefix}embedding.word_embeddings.weight
                    tensor = self.shared_embedding_or_output_weight()
                    first_stage_word_emb_key = f'{prefix}embedding.word_embeddings.weight'
                    dp_rank = parallel_state.get_data_parallel_rank()
                    dp_size = parallel_state.get_data_parallel_world_size()
                    last_stage_word_emb_replica_id = (
                        dp_rank + dp_size
                    )  # copy of first stage embedding

                    sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                        tensor=tensor,
                        key=first_stage_word_emb_key,
                        replica_id=last_stage_word_emb_replica_id,
                        allow_shape_mismatch=True,
                    )

                    sharded_state_dict[output_layer_key] = sharded_output_layer_tensor

            else:
                output_layer_state_dict = self.output_layer.state_dict(
                    prefix=output_layer_prefix, keep_vars=True
                )
                output_layer_tensor = output_layer_state_dict[output_layer_key]
                # independent output layer
                sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                    tensor=output_layer_tensor,
                    key=output_layer_key,
                    replica_id=parallel_state.get_data_parallel_rank(),
                    allow_shape_mismatch=True,
                )

                sharded_state_dict[output_layer_key] = sharded_output_layer_tensor

        return sharded_state_dict

