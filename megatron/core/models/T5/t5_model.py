# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Literal, Optional, List

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel, InferenceParams
from megatron.core.models.common.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.T5.t5_embedding import T5Embedding
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


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

    def __init__(self, mpu_vocab_size, config, parallel_output, vocab_size, pre_process, share_embeddings_and_output_weights):
        super(T5LMHead, self).__init__(config=config)

        # self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        # self.bias.model_parallel = True
        # self.bias.partition_dim = 0
        # self.bias.stride = 1
        # self.parallel_output = parallel_output

        self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                config=config,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights,
            )       

    def forward(self, hidden_states, word_embeddings_weight):
        logits, _ = self.output_layer(hidden_states, weight=word_embeddings_weight)
        return logits


class T5Model(MegatronModule):
    """T5 Language model.

    Arguments:
        config (TransformerConfig): transformer config

        spec (List[TransformerBlockSpec]): transformer layer customization specs for encoder and decoder
        
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
            spec: List[ModuleSpec],
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
            ):
        
        super(T5Model, self).__init__(config=config)   

        self.config: TransformerConfig = config
        self.spec: List[ModuleSpec] = spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = True
        self.add_decoder = True
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_and_decoder

        # Embeddings.
        if self.pre_process: # lOOK INTO transformer.py in nemo (GPT/ BERT model)
            self.embedding = T5Embedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                add_position_embedding=(self.position_embedding_type == 'learned_absolute'),
            ) 

        # Rotary Position Embeddings
        if self.position_embedding_type == 'rope':
            rotary_dim = self.config.kv_channels
            if rotary_percent < 1.0:
                rotary_dim = int(rotary_dim * rotary_percent)

            self.rotary_pos_emb = RotaryEmbedding(rotary_dim, seq_len_interpolation_factor)
        else:
            self.rotary_pos_emb = None

        # Transformer encoder
        encoder_spec, decoder_spec = self.spec
        self.encoder = TransformerBlock(
            config=self.config,
            submodules=encoder_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        # Transformer decoder
        self.decoder = TransformerBlock(
            config=self.config,
            submodules=decoder_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            self.lm_head = T5LMHead(
                self.shared_embedding_or_output_weight().size(0), 
                config, 
                parallel_output,
                self.vocab_size,
                self.pre_process,
                self.share_embeddings_and_output_weights)

        if self.share_embeddings_and_output_weights and (self.pre_process or self.post_process):
            self.initialize_last_stage_with_word_embeddings()

    def set_input_tensor(self, input_tensor): ### what does this do?
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt'
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        encoder_input_ids: Tensor,
        decoder_input_ids: Tensor,
        encoder_attn_mask: Tensor,
        decoder_attn_mask: Tensor,
        encoder_decoder_attn_mask: Tensor,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
    ):

        encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask = t5_extended_attention_mask(
            [encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask]
        )
        encoder_position_ids = t5_position_ids(encoder_input_ids)
        decoder_position_ids = t5_position_ids(decoder_input_ids)
        

        ## Encoder forward
        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(input_ids=encoder_input_ids, position_ids=encoder_position_ids)
        else:
            # intermediate stage of pipeline
            encoder_input = None

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.rotary_pos_emb is not None:
            rotary_seq_len = self.max_sequence_length
            if inference_params is not None:
                rotary_seq_len = inference_params.max_sequence_length
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run encoder.
        encoder_hidden_states = self.encoder(
            hidden_states=encoder_input,
            attention_mask=encoder_attn_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        ## Decoder forward
        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(input_ids=decoder_input_ids, position_ids=decoder_position_ids)
        else:
            # intermediate stage of pipeline
            decoder_input = None   ### should it take encoder_hidden_states

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.rotary_pos_emb is not None:
            if inference_params is not None:
                rotary_seq_len = inference_params.max_sequence_length
            else:
                if self.decoder.input_tensor is not None:
                    rotary_seq_len = self.decoder.input_tensor.size(0)
                else:
                    rotary_seq_len = decoder_input.size(0)
                # Decoder input is split along sequence dimension, but RoPE is applied in tensor parallel region
                if self.config.sequence_parallel:
                    rotary_seq_len *= self.config.tensor_model_parallel_size
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

        # Return if not post_process
        if not self.post_process:
            return decoder_hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits = self.lm_head(decoder_hidden_states, word_embeddings_weight=output_weight)

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
            return self.lm_head.output_layer.weight
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
            self.lm_head.output_layer.weight.data.fill_(0)
            self.lm_head.output_layer.weight.shared = True

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
                torch.distributed.all_reduce(
                    weight.data, group=parallel_state.get_embedding_group()
                )

        elif not getattr(T5Model, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            T5Model.embedding_warning_printed = True


    def sharded_state_dict(self, prefix=''):
        sharded_state_dict = {}

        if self.pre_process:
            embedding_prefix = f'{prefix}embedding.'
            embedding_sharded_state_dict = self.embedding.sharded_state_dict(
                prefix=embedding_prefix
            )
            sharded_state_dict.update(embedding_sharded_state_dict)

        encoder_prefix = f'{prefix}encoder.'
        encoder_sharded_state_dict = self.encoder.sharded_state_dict(prefix=encoder_prefix)
        sharded_state_dict.update(encoder_sharded_state_dict)

        decoder_prefix = f'{prefix}decoder.'
        decoder_sharded_state_dict = self.decoder.sharded_state_dict(prefix=decoder_prefix)
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


    # def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
    #     pass


    # def load_state_dict(self, state_dict, strict=True):
    #     pass


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_["embedding"] \
            = self.embedding.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        state_dict_["encoder"] \
            = self.encoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        state_dict_["decoder"] \
            = self.decoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)

        if self.post_process and self.add_decoder:
            state_dict_["lm_head"] \
                = self.lm_head.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
         # Save word_embeddings.
        if self.post_process and not self.pre_process and self.add_decoder:
            state_dict_["word_embeddings_for_head"] \
                = self.embedding.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_


    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        self.embedding.load_state_dict(
            state_dict["embedding"], strict=strict)

        self.encoder.load_state_dict(
            state_dict["encoder"], strict=strict)

        self.decoder.load_state_dict(
            state_dict["decoder"], strict=strict)
        
        if self.post_process and self.add_decoder:
            self.lm_head.load_state_dict(state_dict["lm_head"],
                                         strict=strict)
            
        # Load word embeddings
        if self.post_process and not self.pre_process and self.add_decoder:
            self.word_embeddings.load_state_dict(
                state_dict["word_embeddings_for_head"], strict=strict)

