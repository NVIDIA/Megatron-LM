# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""BERT model."""

import pickle

import numpy as np
import torch

from megatron import get_args
from megatron.module import MegatronModule

from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .transformer import LayerNorm
from .utils import gelu
from .utils import get_linear_layer
from .utils import init_method_normal
from .utils import scaled_init_method_normal


def bert_attention_mask_func(attention_scores, attention_mask):
    attention_scores = attention_scores + attention_mask
    return attention_scores


def bert_extended_attention_mask(attention_mask, dtype):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)
    # Since attention_mask is 1.0 for positions we want to attend and 0.0
    # for masked positions, this operation will create a tensor which is
    # 0.0 for positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    # fp16 compatibility
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

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
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: whether output logits being distributed or not.
    """
    def __init__(self, mpu_vocab_size, hidden_size, init_method,
                 layernorm_epsilon, parallel_output):

        super(BertLMHead, self).__init__()

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)


    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        output = parallel_lm_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output



class BertModel(MegatronModule):
    """Bert Language model."""

    def __init__(self, num_tokentypes=2, add_binary_head=True,
                 ict_head_size=None, parallel_output=True):
        super(BertModel, self).__init__()
        args = get_args()

        self.add_binary_head = add_binary_head
        self.ict_head_size = ict_head_size
        self.add_ict_head = ict_head_size is not None
        assert not (self.add_binary_head and self.add_ict_head)

        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)
        add_pooler = self.add_binary_head or self.add_ict_head
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)
        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=add_pooler,
            init_method=init_method,
            scaled_init_method=scaled_init_method)

        if not self.add_ict_head:
            self.lm_head = BertLMHead(
                self.language_model.embedding.word_embeddings.weight.size(0),
                args.hidden_size, init_method, args.layernorm_epsilon, parallel_output)
            self._lm_head_key = 'lm_head'
        if self.add_binary_head:
            self.binary_head = get_linear_layer(args.hidden_size, 2,
                                                init_method)
            self._binary_head_key = 'binary_head'
        elif self.add_ict_head:
            self.ict_head = get_linear_layer(args.hidden_size, ict_head_size, init_method)
            self._ict_head_key = 'ict_head'

    def forward(self, input_ids, attention_mask, tokentype_ids=None):

        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype)
        position_ids = bert_position_ids(input_ids)

        if self.add_binary_head or self.add_ict_head:
            lm_output, pooled_output = self.language_model(
                input_ids,
                position_ids,
                extended_attention_mask,
                tokentype_ids=tokentype_ids)
        else:
            lm_output = self.language_model(
                input_ids,
                position_ids,
                extended_attention_mask,
                tokentype_ids=tokentype_ids)

        # Output.
        if self.add_ict_head:
            ict_logits = self.ict_head(pooled_output)
            return ict_logits, None

        lm_logits = self.lm_head(
            lm_output, self.language_model.embedding.word_embeddings.weight)
        if self.add_binary_head:
            binary_logits = self.binary_head(pooled_output)
            return lm_logits, binary_logits

        return lm_logits, None


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if not self.add_ict_head:
            state_dict_[self._lm_head_key] \
                = self.lm_head.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)
        if self.add_binary_head:
            state_dict_[self._binary_head_key] \
                = self.binary_head.state_dict(destination, prefix, keep_vars)
        elif self.add_ict_head:
            state_dict_[self._ict_head_key] \
                = self.ict_head.state_dict(destination, prefix, keep_vars)
        return state_dict_


    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if not self.add_ict_head:
            self.lm_head.load_state_dict(
                state_dict[self._lm_head_key], strict=strict)
        if self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict)
        elif self.add_ict_head:
            self.ict_head.load_state_dict(
                state_dict[self._ict_head_key], strict=strict)


class REALMBertModel(MegatronModule):
    def __init__(self, ict_model, block_hash_data_path):
        super(REALMBertModel, self).__init__()
        bert_args = dict(
            num_tokentypes=2,
            add_binary_head=False,
            parallel_output=True
        )
        self.lm_model = BertModel(**bert_args)
        self._lm_key = 'realm_lm'

        self.ict_model = ict_model
        with open(block_hash_data_path, 'rb') as data_file:
            data = pickle.load(data_file)
            # {block_idx: block_embed} - the main index
            self.block_data = data['block_data']
            # {hash_num: [start, end, doc, block]} - the hash table
            self.hash_data = data['hash_data']
            # [embed_size x num_buckets / 2] - the projection matrix used for hashing
            self.hash_matrix = self.hash_data['matrix']

    def forward(self, tokens, attention_mask, token_types):
        # [batch_size x embed_size]
        query_logits = self.ict_model.embed_query(tokens, attention_mask, token_types)

        # [batch_size x num_buckets / 2]
        query_hash_pos = torch.matmul(query_logits, self.hash_matrix)
        query_hash_full = torch.cat((query_hash_pos, -query_hash_pos), axis=1)

        # [batch_size]
        query_hashes = torch.argmax(query_hash_full, axis=1)

        batch_block_embeds = []
        for hash in query_hashes:
            # TODO: this should be made into a single np.array in preprocessing
            bucket_blocks = self.hash_data[hash]
            block_indices = bucket_blocks[:, 3]
            # [bucket_pop x embed_size]
            block_embeds = [self.block_data[idx] for idx in block_indices]
            # will become [batch_size x bucket_pop x embed_size]
            # will require padding to do tensor multiplication
            batch_block_embeds.append(block_embeds)

        # [batch_size x max bucket_pop x embed_size]
        batch_block_embeds = np.array(batch_block_embeds)
        # [batch_size x 1 x max bucket_pop]
        retrieval_scores = query_logits.matmul(torch.transpose(batch_block_embeds, 1, 2))
        # [batch_size x max bucket_pop]
        retrieval_scores = retrieval_scores.squeeze()
        top5_vals, top5_indices = torch.topk(retrieval_scores, k=5)



class ICTBertModel(MegatronModule):
    """Bert-based module for Inverse Cloze task."""
    def __init__(self,
                 ict_head_size,
                 num_tokentypes=1,
                 parallel_output=True,
                 only_query_model=False,
                 only_block_model=False):
        super(ICTBertModel, self).__init__()
        bert_args = dict(
            num_tokentypes=num_tokentypes,
            add_binary_head=False,
            ict_head_size=ict_head_size,
            parallel_output=parallel_output
        )
        assert not only_block_model and only_query_model
        self.use_block_model = not only_query_model
        self.use_query_model = not only_block_model

        if self.use_query_model:
            # this model embeds (pseudo-)queries - Embed_input in the paper
            self.query_model = BertModel(**bert_args)
            self._query_key = 'question_model'

        if self.use_block_model:
            # this model embeds evidence blocks - Embed_doc in the paper
            self.block_model = BertModel(**bert_args)
            self._block_key = 'context_model'

    def forward(self, query_tokens, query_attention_mask, block_tokens, block_attention_mask):
        """Run a forward pass for each of the models and compute the similarity scores."""
        query_logits = self.embed_query(query_tokens, query_attention_mask)
        block_logits = self.embed_block(block_tokens, block_attention_mask)

        # [batch x embed] * [embed x batch]
        retrieval_scores = query_logits.matmul(torch.transpose(block_logits, 0, 1))
        return retrieval_scores

    def embed_query(self, query_tokens, query_attention_mask):
        """Embed a batch of tokens using the query model"""
        if self.use_query_model:
            query_types = torch.zeros(query_tokens.shape).type(torch.float16).cuda()
            query_ict_logits, _ = self.query_model.forward(query_tokens, query_attention_mask, query_types)
            return query_ict_logits
        else:
            raise ValueError("Cannot embed query without query model.")

    def embed_block(self, block_tokens, block_attention_mask):
        """Embed a batch of tokens using the block model"""
        if self.use_block_model:
            block_types = torch.zeros(block_tokens.shape).type(torch.float16).cuda()
            block_ict_logits, _ = self.block_model.forward(block_tokens, block_attention_mask, block_types)
            return block_ict_logits
        else:
            raise ValueError("Cannot embed block without block model.")

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """Save dict with state dicts of each of the models."""
        state_dict_ = {}
        if self.use_query_model:
            state_dict_[self._query_key] \
                = self.query_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)

        if self.use_block_model:
            state_dict_[self._block_key] \
                = self.block_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        if self.use_query_model:
            self.query_model.load_state_dict(
                state_dict[self._query_key], strict=strict)

        if self.use_block_model:
            self.block_model.load_state_dict(
                state_dict[self._block_key], strict=strict)
