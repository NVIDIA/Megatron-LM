# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer based language model."""

from json import encoder
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.model.enums import LayerType, AttnMaskType
from megatron.model.transformer import ParallelTransformer
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal, scaled_init_method_normal

def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)
    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(num_tokentypes, add_pooler,
                       encoder_attn_mask_type, init_method=None,
                       scaled_init_method=None, add_decoder=False,
                       decoder_attn_mask_type=AttnMaskType.causal,
                       pre_process=True, post_process=True, num_experts=1):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        init_method,
        scaled_init_method,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
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
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [b, s, h]
        # sequence_index: index of the token to pool.
        pooled = hidden_states[:, sequence_index, :]
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
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 init_method,
                 num_tokentypes=0):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes

        args = get_args()

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, self.hidden_size,
            init_method=self.init_method)
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(
            max_sequence_length, self.hidden_size)
        self._position_embeddings_key = 'position_embeddings'
        # Initialize the position embeddings.
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
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

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
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] \
            = self.position_embeddings.state_dict(
                destination, prefix, keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(
                    destination, prefix, keep_vars)

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
                 init_method,
                 output_layer_init_method,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True,
                 num_experts=1):
        super(TransformerLanguageModel, self).__init__()
        args = get_args()

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.num_experts = num_experts

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       self.init_method,
                                       self.num_tokentypes)
            self._embedding_key = 'embedding'

        # Transformer.
        self.encoder = ParallelTransformer(
            self.init_method,
            output_layer_init_method,
            self_attn_mask_type=self.encoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            num_experts=self.num_experts
        )
        self._encoder_key = 'encoder'

        # Decoder
        if self.add_decoder:
            assert args.pipeline_model_parallel_size == 1, \
                'pipeline parallelism is not supported in the presence of decoder'
            self.decoder = ParallelTransformer(
                self.init_method,
                output_layer_init_method,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                num_experts=self.num_experts)
            self._decoder_key = 'decoder'

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = 'pooler'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        self.encoder.set_input_tensor(input_tensor)

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                enc_dec_attn_mask=None, tokentype_ids=None, layer_past=None,
                get_key_value=False, pooling_sequence_index=0,
                enc_hidden_states=None, output_enc_hidden=False):

        # Embeddings.
        if self.pre_process:
            embedding_output = self.embedding(enc_input_ids, enc_position_ids,
                                              tokentype_ids=tokentype_ids)
            encoder_input = embedding_output
        else:
            encoder_input = None

        # encoder.
        if enc_hidden_states is None:
            encoder_output, *moe_losses = self.encoder(encoder_input,
                                          enc_attn_mask,
                                          layer_past=layer_past,
                                          get_key_value=get_key_value)
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)
            moe_losses = []

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output,
                                            pooling_sequence_index)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output, *moe_losses
            else:
                return encoder_output, *moe_losses

        # Decoder Embedding
        dec_embedding_output = self.embedding(dec_input_ids,
                                              dec_position_ids)
        # decoder
        decoder_output, *moe_losses = self.decoder(dec_embedding_output,
                                      dec_attn_mask,
                                      layer_past=layer_past,
                                      get_key_value=get_key_value,
                                      encoder_output=encoder_output,
                                      enc_dec_attn_mask=enc_dec_attn_mask)

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output, *moe_losses
        else:
            return decoder_output, encoder_output, *moe_losses

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        moe_state_dict = {}
        if self.pre_process:
            state_dict_[self._embedding_key] \
                = self.embedding.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)
        encoder_state_dict = self.encoder.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
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
                    = self.pooler.state_dict_for_save_checkpoint(
                        destination, prefix, keep_vars)
        if self.add_decoder:
            state_dict_[self._decoder_key] \
                = self.decoder.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)
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
        if self._encoder_key in state_dict:
            state_dict_ = state_dict[self._encoder_key]
        # for backward compatibility.
        elif 'transformer' in state_dict:
            state_dict_ = state_dict['transformer']
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]

        # for backward compatibility.
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
                    actual_key = key[len(self._encoder_key):]
                    state_dict_[actual_key] = state_dict["moe_state_dict"].pop(key)
            if len(state_dict["moe_state_dict"]) == 0:
                del state_dict["moe_state_dict"]
        self.encoder.load_state_dict(state_dict_, strict=strict)

        if self.post_process:
            # pooler
            if self.add_pooler:
                assert 'pooler' in state_dict, \
                    'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key],
                                            strict=strict)
        # decoder
        if self.add_decoder:
            assert 'decoder' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key],
                                         strict=strict)
