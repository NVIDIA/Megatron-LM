# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import List, Literal, Optional, Tuple

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


class T5LMHead(MegatronModule):
    """Masked LM head for T5

    Args:
        config (TransformerConfig): transformer config
        parallel_output (bool): wether output logits being distributed or not.
        vocab_size (int): vocabulary size
        pre_process (bool): Include embedding layer
        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are
            shared.
    """

    def __init__(
        self,
        config: TransformerConfig,
        parallel_output: bool,
        vocab_size: int,
        pre_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ):
        super(T5LMHead, self).__init__(config=config)

        self.parallel_output = parallel_output

        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            vocab_size,
            config=config,
            init_method=config.init_method,
            bias=share_embeddings_and_output_weights,
            skip_bias_add=not share_embeddings_and_output_weights,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights,
        )

    def forward(self, hidden_states: Tensor, word_embeddings_weight: Tensor) -> Tensor:
        """Forward pass.

        Args:
            hidden_states (Tensor): output hidden states from decoder
            word_embeddings_weight (Tensor): word embedding weight

        Returns:
            Tensor: logits tensor
        """

        logits, _ = self.output_layer(hidden_states, weight=word_embeddings_weight)
        return logits


class T5Model(LanguageModule):
    """T5 Language model.

    Args:
        config (TransformerConfig): transformer config

        encoder_config (TransformerConfig): encoder transformer config

        transformer_encoder_layer_spec (ModuleSpec): transformer layer customization specs for encoder

        transformer_decoder_layer_spec (ModuleSpec): transformer layer customization specs for decoder

        vocab_size (int): vocabulary size

        max_sequence_length (int): maximum size of sequence. This is used for positional embedding

        pre_process (bool): Include embedding layer (used with pipeline parallelism)

        post_process (bool): Include an output layer (used with pipeline parallelism)

        fp16_lm_cross_entropy (bool, optional): Defaults to False

        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks

        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are
            shared. Defaults to False.

        position_embedding_type (string): Position embedding type. Options ['learned_absolute', 'rope'].
            Defaults is 'learned_absolute'.

        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
            Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.

        seq_len_interpolation_factor (float): scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.

        add_encoder (bool): Create the encoder (used with pipeline parallelism). When using pipelining,
            the encoder will only be created on a subset of the pipeline ranks.

        add_decoder (bool): Include an output layer (used with pipeline parallelism). As with `add_encoder`, when
            using this model and pipelining, the decoder will only be created on a subset of the pipeline ranks.
    """

    def __init__(
        self,
        config: TransformerConfig,
        encoder_config: TransformerConfig,
        transformer_encoder_layer_spec: ModuleSpec,
        transformer_decoder_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        add_encoder: bool = True,
        add_decoder: bool = True,
    ):

        super(T5Model, self).__init__(config=config)

        self.config: TransformerConfig = config
        self.encoder_config: TransformerConfig = encoder_config
        self.transformer_encoder_layer_spec: ModuleSpec = transformer_encoder_layer_spec
        self.transformer_decoder_layer_spec: ModuleSpec = transformer_decoder_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.encoder_hidden_state = None

        # Tells schedules.py that this model has a skip connection between the encoder's output and the decoder
        # (and hence both the encoder and decoder's tensors are required for correct backprop).
        self.xattn_needed = True

        # specify the position embeddings as a member variable in the T5 class
        # so that they are easy to find for `finalize_model_grads._allreduce_position_embedding_grads`
        self.position_embeddings = None
        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
            )
            self.position_embeddings = self.embedding.position_embeddings

        # Rotary Position Embeddings
        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
            )

        # Transformer encoder
        encoder_spec, decoder_spec = (
            self.transformer_encoder_layer_spec,
            self.transformer_decoder_layer_spec,
        )
        if self.add_encoder:
            self.encoder = TransformerBlock(
                config=self.encoder_config,
                spec=encoder_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
        else:
            self.encoder = None

        if self.add_decoder:
            # Transformer decoder
            self.decoder = TransformerBlock(
                config=self.config,
                spec=decoder_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
        else:
            self.decoder = None

        # Output
        if post_process:
            self.lm_head = T5LMHead(
                config,
                parallel_output,
                self.vocab_size,
                self.pre_process,
                self.share_embeddings_and_output_weights,
            )
            self.output_layer = self.lm_head.output_layer

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def forward(
        self,
        encoder_input_ids: Tensor,
        decoder_input_ids: Tensor,
        encoder_attn_mask: Tensor,
        decoder_attn_mask: Tensor,
        encoder_decoder_attn_mask: Tensor,
        lm_labels: Tensor = None,
        encoder_hidden_states: Tensor = None,
        output_encoder_hidden_only: bool = False,
        inference_params: InferenceParams = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            encoder_input_ids (Tensor): input ids for encoder
            decoder_input_ids (Tensor): input ids for decoder
            encoder_attn_mask (Tensor): self-attention mask for encoder
            decoder_attn_mask (Tensor): self-attention mask for decoder
            encoder_decoder_attn_mask (Tensor): cross-attention mask between encoder and decoder
            lm_labels (Tensor): labels for decoder output
            inference_params (InferenceParams): relevant arguments for inferencing

        Returns:
            Tensor: loss tensor
        """

        (
            encoder_attn_mask,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
        ) = t5_extended_attention_mask(
            [encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask]
        )

        ## Encoder forward
        if encoder_hidden_states is None:

            # Encoder position ids
            encoder_position_ids = t5_position_ids(encoder_input_ids)

            # Encoder embedding.
            if self.pre_process:
                encoder_input = self.embedding(
                    input_ids=encoder_input_ids, position_ids=encoder_position_ids
                )
            else:
                # intermediate stage of pipeline
                encoder_input = None

            # Rotary positional embeddings
            rotary_pos_emb = None
            if self.position_embedding_type == 'rope':
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_params, self.encoder, encoder_input, self.config
                )
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

            # Run encoder.
            if self.add_encoder:
                encoder_hidden_states = self.encoder(
                    hidden_states=encoder_input,
                    attention_mask=encoder_attn_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                )
            else:
                encoder_hidden_states = self.encoder_hidden_state

        if not self.add_decoder or output_encoder_hidden_only:
            return encoder_hidden_states

        ## Decoder forward
        # Decoder position ids
        decoder_position_ids = t5_position_ids(decoder_input_ids)

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(
                input_ids=decoder_input_ids, position_ids=decoder_position_ids
            )
        else:
            # intermediate stage of pipeline
            decoder_input = None  ### should it take encoder_hidden_states

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        decoder_hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=decoder_attn_mask,
            context=encoder_hidden_states,
            context_mask=encoder_decoder_attn_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        if self.post_process:
            lm_logits = self.lm_head(
                decoder_hidden_states, self.shared_embedding_or_output_weight()
            )
            if lm_labels is None:
                # [s b h] => [b s h]
                return lm_logits.transpose(0, 1).contiguous()
            else:
                # [b s] => [s b]
                lm_loss = self.compute_language_model_loss(lm_labels, lm_logits)
                return lm_loss
        else:
            return decoder_hidden_states

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert (
                len(input_tensor) == 1
            ), 'input_tensor should only be length 1 for stage with both encoder and decoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert (
                len(input_tensor) == 1
            ), 'input_tensor should only be length 1 for stage with only encoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.decoder.set_input_tensor(input_tensor[0])
                self.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.decoder.set_input_tensor(None)
                self.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception('input_tensor must have either length 1 or 2')
        else:
            raise Exception('Stage must have at least either encoder or decoder')

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Function to share the input embeddings and output logit weights."""

        if self.pre_process:
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.lm_head.output_layer.weight
        return None

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        assert not sharded_offsets, "Unexpected sharded offsets"
        sharded_state_dict = {}

        if self.pre_process:
            embedding_prefix = f'{prefix}embedding.'
            embedding_sharded_state_dict = self.embedding.sharded_state_dict(
                prefix=embedding_prefix, metadata=metadata
            )
            sharded_state_dict.update(embedding_sharded_state_dict)

        encoder_prefix = f'{prefix}encoder.'
        encoder_sharded_state_dict = self.encoder.sharded_state_dict(
            prefix=encoder_prefix, metadata=metadata
        )
        sharded_state_dict.update(encoder_sharded_state_dict)

        decoder_prefix = f'{prefix}decoder.'
        decoder_sharded_state_dict = self.decoder.sharded_state_dict(
            prefix=decoder_prefix, metadata=metadata
        )
        sharded_state_dict.update(decoder_sharded_state_dict)

        if self.post_process:
            output_layer_prefix = f'{prefix}output_layer.'
            output_layer_weight_key = f'{output_layer_prefix}weight'
            output_layer_bias_key = f'{output_layer_prefix}bias'
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

                    sharded_state_dict[output_layer_weight_key] = sharded_output_layer_tensor
                # output_layer.weight is shared, but we still need to process output_layer.bias
                sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                    tensor=self.lm_head.output_layer.bias,
                    key=output_layer_bias_key,
                    allow_shape_mismatch=True,
                )
                sharded_state_dict[output_layer_bias_key] = sharded_output_layer_tensor
            else:
                output_layer_state_dict = self.output_layer.state_dict(
                    prefix=output_layer_prefix, keep_vars=True
                )
                output_layer_tensor = output_layer_state_dict[output_layer_weight_key]
                # independent output layer
                sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                    tensor=output_layer_tensor,
                    key=output_layer_weight_key,
                    replica_id=parallel_state.get_data_parallel_rank(),
                    allow_shape_mismatch=True,
                )

                sharded_state_dict[output_layer_weight_key] = sharded_output_layer_tensor

        return sharded_state_dict


def t5_extended_attention_mask(attention_mask_list: List[Tensor]) -> List[Tensor]:
    def attn_mask_postprocess(attn_mask):
        # [b, 1, s, s]
        extended_attention_mask = attn_mask.unsqueeze(1)
        return extended_attention_mask

    return [
        (attn_mask_postprocess(attn_mask) if attn_mask is not None else None)
        for attn_mask in attention_mask_list
    ]


def t5_position_ids(token_ids: Tensor) -> Tensor:
    """Calculate position ids from token ids
    Args:
        token_ids (Tensor): input tokens

    Returns:
        Tensor: position ids
    """
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids
