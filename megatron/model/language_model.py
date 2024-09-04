# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Transformer based language model."""

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType

from .enums import AttnMaskType, LayerType
from .module import MegatronModule
from .rotary_pos_embedding import apply_rotary_pos_emb, RotaryEmbedding
from .transformer import ParallelTransformer
from .utils import get_linear_layer
from .utils import init_method_normal, scaled_init_method_normal, gather_and_init


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    args = get_args()
    # Parallel logits.
    if args.async_tensor_model_parallel_allreduce or\
            args.sequence_parallel:
        input_parallel = input_
        model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        async_grad_allreduce = args.async_tensor_model_parallel_allreduce and \
            model_parallel and not args.sequence_parallel
    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)
        async_grad_allreduce = False

    # Matrix multiply.
    logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=args.gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel=args.sequence_parallel)
    # Gather if needed.

    if parallel_output:
        return logits_parallel

    return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(config, num_tokentypes, add_pooler,
                       encoder_attn_mask_type,
                       add_encoder=True,
                       add_decoder=False,
                       decoder_attn_mask_type=AttnMaskType.causal,
                       pre_process=True, post_process=True, num_experts=[1]):
    """Build language model and return along with the key to save."""
    args = get_args()
    if config.init_method is None:
        config.init_method = init_method_normal(config.init_method_std)

    if config.output_layer_init_method is None:
        config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                    config.num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        config,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process,
        num_experts=num_experts)
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        args = get_args()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method, gather_params_on_init=args.zero_stage == 3)
        self.sequence_parallel = args.sequence_parallel


    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False)

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
        embedding_weights_in_fp32: casts word embedding weights to
                                   fp32 before sampling. Required to
                                   maintain reproducibility when
                                   training in bf16.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 config,
                 num_tokentypes=0,
                 embedding_weights_in_fp32=False):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = config.init_method
        self.num_tokentypes = num_tokentypes

        args = get_args()

        # Word embeddings (parallel).
        self.embedding_weights_in_fp32 = embedding_weights_in_fp32
        self.params_dtype = args.params_dtype
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size, config=config, init_method=config.init_method)
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        self.add_position_embedding = args.add_position_embedding
        if self.add_position_embedding:
            self._position_embeddings_key = 'position_embeddings'
            if args.sequence_parallel:
                self.position_embeddings = tensor_parallel.layers.SequenceParallelPositionEmbedding(
                    max_sequence_length, self.hidden_size)
                # Initialize the position embeddings.
                self.init_method(self.position_embeddings.local_embeddings.weight)
            else:
                self.position_embeddings = torch.nn.Embedding(
                    max_sequence_length, self.hidden_size)
                # Initialize the position embeddings.
                if args.perform_initialization:
                    if args.zero_stage == 3:
                        gather_and_init(self.position_embeddings.weight, self.init_method)
                    else:
                        self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                           self.hidden_size)
            # Initialize the token-type embeddings.
            if args.perform_initialization:
                if args.zero_stage == 3:
                    gather_and_init(self.tokentype_embeddings.weight, self.init_method)
                else:
                    self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = args.fp32_residual_connection
        self.sequence_parallel = args.sequence_parallel
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.add_position_embedding:
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        args = get_args()
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(torch.float32)
        words_embeddings = self.word_embeddings(input_ids)
        if self.embedding_weights_in_fp32:
            words_embeddings = words_embeddings.to(self.params_dtype)
            self.word_embeddings = self.word_embeddings.to(self.params_dtype)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            # already partition sequence, do not need scatter_to_sequence_parallel_region ?
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(prefix=prefix,
                                              keep_vars=keep_vars)
        if self.add_position_embedding:
            state_dict_[self._position_embeddings_key] \
                = self.position_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(prefix=prefix,
                                                       keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self.add_position_embedding:
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'position_embeddings' in key:
                        state_dict_[key.split('position_embeddings.')[1]] \
                            = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)


class EmbeddingPipe(Embedding):

    def forward(self, inputs, **kwargs):
        if not hasattr(self, '_args'):
            self._args = get_args()

        input_ids = inputs[0]
        position_ids = inputs[1]
        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[2]

        if len(inputs) == 4:
            tokentype_ids = inputs[3]
        else:
            tokentype_ids = None
        
        embeddings = super().forward(input_ids, position_ids, tokentype_ids=tokentype_ids)

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return embeddings
        else:
            assert False
            return embeddings, attention_mask


    @property
    def word_embeddings_weight(self):
        """Easy accessory for the DeepSpeed pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight


class TransformerLanguageModel(MegatronModule):
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

    def __init__(self,
                 config,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True,
                 num_experts=[1]):
        args = get_args()
        # TODO: passing share_embeddings_and_output_weights=False will not work correctly for T5 and embeddings will not be synced. Fix later for T5.
        if args.untie_embeddings_and_output_weights: assert not add_decoder
        super(TransformerLanguageModel, self).__init__(share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = config.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = config.init_method
        self.add_encoder = add_encoder
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        self.add_retriever = args.retro_add_retriever
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.num_experts = num_experts

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       config,
                                       self.num_tokentypes,
                                       args.embedding_weights_in_fp32)
            self._embedding_key = 'embedding'

        # Rotary positional embeddings
        self.use_rotary_position_embeddings = \
            args.use_rotary_position_embeddings
        if args.use_rotary_position_embeddings:
            self.seq_length = args.seq_length
            rotary_dim = args.hidden_size // args.num_attention_heads \
                if args.kv_channels is None else args.kv_channels

            if args.rotary_percent < 1.0:
                rotary_dim = int(rotary_dim * args.rotary_percent)

            # partial rotary embeddings, which is better than full rotary
            # Wang and Komatsuzaki et al
            # https://github.com/kingoflolz/mesh-transformer-jax/
            self.rotary_pos_emb = RotaryEmbedding(rotary_dim, theta=args.rope_theta)

        # Encoder (usually set to True, False if part of an encoder-decoder
        # architecture and in encoder-only stage).
        if self.add_encoder:
            self.encoder = ParallelTransformer(
                config,
                model_type=args.model_type if not args.retro_add_retriever \
                    else ModelType.retro_decoder,
                self_attn_mask_type=self.encoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
                num_experts=self.num_experts
            )
            self._encoder_key = 'encoder'
        else:
            self.encoder = None

        # Decoder (usually set to False, True if part of an encoder-decoder
        # architecture and in decoder-only stage).
        if self.add_decoder:
            self.decoder = ParallelTransformer(
                config,
                model_type=args.model_type,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
                num_experts=self.num_experts)
            self._decoder_key = 'decoder'
        else:
            self.decoder = None

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = 'pooler'

            if self.untie_embeddings_and_output_weights:
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    args.hidden_size,
                    args.padded_vocab_size,
                    config=config,
                    init_method=self.init_method,
                    bias=False) # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
                self._output_layer_key = 'output_layer'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with both encoder and decoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with only encoder'
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

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                enc_dec_attn_mask=None, tokentype_ids=None,
                inference_params=None,
                pooling_sequence_index=0,
                enc_hidden_states=None, output_enc_hidden=False):
        args = get_args()
        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids,
                                           tokentype_ids=tokentype_ids)
        else:
            encoder_input = None

        # Retriever embedding.
        if self.add_retriever and self.pre_process:
            retriever_input = self.embedding(retriever_input_ids,
                                             retriever_position_ids,
                                             tokentype_ids=tokentype_ids)
        else:
            retriever_input = None

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            if inference_params is not None:
                rotary_pos_emb = \
                    self.rotary_pos_emb(inference_params.max_sequence_len)
            else:
                if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
                    rotary_pos_emb = self.rotary_pos_emb(args.curriculum_seqlen)
                else:
                    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output, *encoder_moe_losses = self.encoder(
                    encoder_input,
                    enc_attn_mask,
                    retriever_input=retriever_input,
                    retriever_attn_mask=retriever_attn_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output, encoder_moe_losses = enc_hidden_states.to(encoder_input.dtype), []

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output,
                                            pooling_sequence_index)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output, encoder_moe_losses
            else:
                return encoder_output, encoder_moe_losses

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids,
                                           dec_position_ids)
        else:
            decoder_input = None

        # Run decoder.
        decoder_output, *decoder_moe_losses = self.decoder(
            decoder_input,
            dec_attn_mask,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb)

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output, decoder_moe_losses, encoder_moe_losses
        else:
            return decoder_output, encoder_output, decoder_moe_losses, encoder_moe_losses

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""
        args = get_args()
        state_dict_ = {}
        moe_state_dict = {}
        if self.pre_process:
            state_dict_[self._embedding_key] \
                = self.embedding.state_dict_for_save_checkpoint(prefix=prefix,
                                                                keep_vars=keep_vars)
        if self.add_encoder:
            encoder_state_dict = self.encoder.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
            if args.random_ltd:
                # When using random-LTD, it is required to call remove_random_ltd_state_dict
                # during model checkpoint saving to transfer the random-LTD-wrapped
                # layers back to original layers. This will help to remove the dependency
                # to random-LTD inside the checkpoint, so that during evaluation or
                # finetuning of the checkpoint there is no need to depend on random-LTD
                # again.
                from deepspeed.runtime.data_pipeline.data_routing.helper import remove_random_ltd_state_dict
                encoder_state_dict = remove_random_ltd_state_dict(encoder_state_dict)
            # MoE states need to be handled separately by DeepSpeed engine, thus
            # moving them to the top level dictionary
            # If components other than encoder may contain MoE states, need to add
            # the same logic
            for key in list(encoder_state_dict.keys()):
                if 'expert' in key and 'moe.gate.wg.weight' not in key:
                    moe_state_dict[self._encoder_key+key] = encoder_state_dict.pop(key)
            state_dict_[self._encoder_key] = encoder_state_dict

        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] \
                    = self.pooler.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
            if self.untie_embeddings_and_output_weights:
                state_dict_[self._output_layer_key] \
                    = self.output_layer.state_dict(prefix=prefix, keep_vars=keep_vars)

        if self.add_decoder:
            state_dict_[self._decoder_key] \
                = self.decoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)

        state_dict_["moe_state_dict"] = moe_state_dict
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self.add_encoder:
            if self._encoder_key in state_dict:
                state_dict_ = state_dict[self._encoder_key]
            # For backward compatibility.
            elif 'transformer' in state_dict:
                state_dict_ = state_dict['transformer']
            else:
                # For backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'transformer.' in key:
                        state_dict_[key.split('transformer.')[1]] = state_dict[key]

            # For backward compatibility.
            # Somehow this backward compatibility could be wrong: sometimes
            # '.attention.' is the actual key used so should not be replaced. Thus
            # added another logic to only replace if the key does not match
            state_dict_self_attention = {}
            encoder_state_dict_keys = list(self.encoder.state_dict().keys())
            for key in state_dict_.keys():
                if '.attention.' in key and key not in encoder_state_dict_keys:
                    state_dict_self_attention[key.replace(".attention.",
                        ".self_attention.")] = state_dict_[key]
                else:
                    state_dict_self_attention[key] = state_dict_[key]
            state_dict_ = state_dict_self_attention

            # Gather encoder MoE states
            if "moe_state_dict" in state_dict:
                for key in list(state_dict["moe_state_dict"].keys()):
                    if self._encoder_key in key:
                        key_list = key.split('.')
                        while key_list[0] != 'encoder':
                            key_list.pop(0)
                        key_list.pop(0)
                        actual_key = '.'.join(key_list)
                        state_dict_[actual_key] = state_dict["moe_state_dict"].pop(key)
                if len(state_dict["moe_state_dict"]) == 0:
                    del state_dict["moe_state_dict"]
            self.encoder.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.post_process:
            if self.add_pooler:
                assert 'pooler' in state_dict, \
                    'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key],
                                            strict=strict)
            if self.untie_embeddings_and_output_weights:
                assert 'output_layer' in state_dict, \
                    'could not find data for output_layer in the checkpoint'
                self.output_layer.load_state_dict(state_dict[self._output_layer_key],
                                                  strict=strict)
        # Decoder.
        if self.add_decoder:
            assert 'decoder' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key],
                                         strict=strict)
