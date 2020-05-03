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

"""BERT model."""

import pickle

import numpy as np
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.model.language_model import parallel_lm_logits
from megatron.model.language_model import get_language_model
from megatron.model.transformer import LayerNorm
from megatron.model.utils import openai_gelu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from megatron.module import MegatronModule


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

        args = get_args()
        
        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.gelu = torch.nn.functional.gelu
        if args.openai_gelu:
            self.gelu = openai_gelu

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
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

        max_pos_embeds = None
        if not add_binary_head and ict_head_size is None:
            max_pos_embeds = 2 * args.seq_length

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=add_pooler,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            max_pos_embeds=max_pos_embeds)

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
    def __init__(self, retriever):
        super(REALMBertModel, self).__init__()
        bert_args = dict(
            num_tokentypes=1,
            add_binary_head=False,
            parallel_output=True
        )
        self.lm_model = BertModel(**bert_args)
        self._lm_key = 'realm_lm'

        self.retriever = retriever
        self._retriever_key = 'retriever'

    def forward(self, tokens, attention_mask):
        # [batch_size x 5 x seq_length]
        top5_block_tokens, top5_block_attention_mask = self.retriever.retrieve_evidence_blocks(tokens, attention_mask)
        batch_size = tokens.shape[0]

        seq_length = top5_block_tokens.shape[2]
        top5_block_tokens = torch.cuda.LongTensor(top5_block_tokens).reshape(-1, seq_length)
        top5_block_attention_mask = torch.cuda.LongTensor(top5_block_attention_mask).reshape(-1, seq_length)

        # [batch_size x 5 x embed_size]
        fresh_block_logits = self.retriever.ict_model(None, None, top5_block_tokens, top5_block_attention_mask, only_block=True).reshape(batch_size, 5, -1)
        # fresh_block_logits.register_hook(lambda x: print("fresh block: ", x.shape, flush=True))

        # [batch_size x embed_size x 1]
        query_logits = self.retriever.ict_model(tokens, attention_mask, None, None, only_query=True).unsqueeze(2)


        # [batch_size x 5]
        fresh_block_scores = torch.matmul(fresh_block_logits, query_logits).squeeze()
        block_probs = F.softmax(fresh_block_scores, dim=1)

        # [batch_size * 5 x seq_length]
        tokens = torch.stack([tokens.unsqueeze(1)] * 5, dim=1).reshape(-1, seq_length)
        attention_mask = torch.stack([attention_mask.unsqueeze(1)] * 5, dim=1).reshape(-1, seq_length)

        # [batch_size * 5 x 2 * seq_length]
        all_tokens = torch.cat((tokens, top5_block_tokens), axis=1)
        all_attention_mask = torch.cat((attention_mask, top5_block_attention_mask), axis=1)
        all_token_types = torch.zeros(all_tokens.shape).type(torch.int64).cuda()

        # [batch_size x 5 x 2 * seq_length x vocab_size]
        lm_logits, _ = self.lm_model.forward(all_tokens, all_attention_mask, all_token_types)
        lm_logits = lm_logits.reshape(batch_size, 5, 2 * seq_length, -1)
        return lm_logits, block_probs

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._lm_key] = self.lm_model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        return state_dict_


class REALMRetriever(MegatronModule):
    """Retriever which uses a pretrained ICTBertModel and a HashedIndex"""
    def __init__(self, ict_model, ict_dataset, block_data, hashed_index, top_k=5):
        super(REALMRetriever, self).__init__()
        self.ict_model = ict_model
        self.ict_dataset = ict_dataset
        self.block_data = block_data
        self.hashed_index = hashed_index
        self.top_k = top_k

    def retrieve_evidence_blocks_text(self, query_text):
        """Get the top k evidence blocks for query_text in text form"""
        print("-" * 100)
        print("Query: ", query_text)
        padless_max_len = self.ict_dataset.max_seq_length - 2
        query_tokens = self.ict_dataset.encode_text(query_text)[:padless_max_len]

        query_tokens, query_pad_mask = self.ict_dataset.concat_and_pad_tokens(query_tokens)
        query_tokens = torch.cuda.LongTensor(np.array(query_tokens).reshape(1, -1))
        query_pad_mask = torch.cuda.LongTensor(np.array(query_pad_mask).reshape(1, -1))

        top5_block_tokens, _ = self.retrieve_evidence_blocks(query_tokens, query_pad_mask)
        for i, block in enumerate(top5_block_tokens[0]):
            block_text = self.ict_dataset.decode_tokens(block)
            print('\n    > Block {}: {}'.format(i, block_text))

    def retrieve_evidence_blocks(self, query_tokens, query_pad_mask):
        """Embed blocks to be used in a forward pass"""
        query_embeds = self.ict_model(query_tokens, query_pad_mask, None, None, only_query=True)
        query_hashes = self.hashed_index.hash_embeds(query_embeds)

        block_buckets = [self.hashed_index.get_block_bucket(hash) for hash in query_hashes]
        for j, bucket in enumerate(block_buckets):
            if len(bucket) < 5:
                for i in range(len(block_buckets)):
                    if len(block_buckets[i]) > 5:
                        block_buckets[j] = block_buckets[i].copy()

        # [batch_size x max_bucket_population x embed_size]
        block_embeds = [torch.cuda.FloatTensor(np.array([self.block_data.embed_data[idx]
                                                         for idx in bucket])) for bucket in block_buckets]

        all_top5_tokens, all_top5_pad_masks = [], []
        for query_embed, embed_tensor, bucket in zip(query_embeds, block_embeds, block_buckets):
            retrieval_scores = query_embed.matmul(torch.transpose(embed_tensor.reshape(-1, query_embed.size()[0]), 0, 1))
            print(retrieval_scores.shape, flush=True)
            top5_vals, top5_indices = torch.topk(retrieval_scores, k=5, sorted=True)

            top5_start_end_doc = [bucket[idx][:3] for idx in top5_indices.squeeze()]
            # top_k tuples of (block_tokens, block_pad_mask)
            top5_block_data = [self.ict_dataset.get_block(*indices) for indices in top5_start_end_doc]

            top5_tokens, top5_pad_masks = zip(*top5_block_data)

            all_top5_tokens.append(np.array(top5_tokens))
            all_top5_pad_masks.append(np.array(top5_pad_masks))

        # [batch_size x 5 x seq_length]
        return np.array(all_top5_tokens), np.array(all_top5_pad_masks)


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
        assert not (only_block_model and only_query_model)
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

    def forward(self, query_tokens, query_attention_mask, block_tokens, block_attention_mask, only_query=False, only_block=False):
        """Run a forward pass for each of the models and compute the similarity scores."""

        if only_query:
            return self.embed_query(query_tokens, query_attention_mask)

        if only_block:
            return self.embed_block(block_tokens, block_attention_mask)


        query_logits = self.embed_query(query_tokens, query_attention_mask)
        block_logits = self.embed_block(block_tokens, block_attention_mask)

        # [batch x embed] * [embed x batch]
        retrieval_scores = query_logits.matmul(torch.transpose(block_logits, 0, 1))
        return retrieval_scores

    def embed_query(self, query_tokens, query_attention_mask):
        """Embed a batch of tokens using the query model"""
        if self.use_query_model:
            query_types = torch.zeros(query_tokens.shape).type(torch.int64).cuda()
            query_ict_logits, _ = self.query_model.forward(query_tokens, query_attention_mask, query_types)
            return query_ict_logits
        else:
            raise ValueError("Cannot embed query without query model.")

    def embed_block(self, block_tokens, block_attention_mask):
        """Embed a batch of tokens using the block model"""
        if self.use_block_model:
            block_types = torch.zeros(block_tokens.shape).type(torch.int64).cuda()
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
            print("Loading ICT query model", flush=True)
            self.query_model.load_state_dict(
                state_dict[self._query_key], strict=strict)

        if self.use_block_model:
            print("Loading ICT block model", flush=True)
            self.block_model.load_state_dict(
                state_dict[self._block_key], strict=strict)
