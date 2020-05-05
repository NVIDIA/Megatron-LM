import numpy as np
import torch
import torch.nn.functional as F

from megatron.checkpointing import load_checkpoint
from megatron.data.realm_index import detach
from megatron.model import BertModel
from megatron.module import MegatronModule


class REALMBertModel(MegatronModule):
    def __init__(self, retriever):
        super(REALMBertModel, self).__init__()
        bert_args = dict(
            num_tokentypes=1,
            add_binary_head=False,
            parallel_output=True
        )
        self.lm_model = BertModel(**bert_args)
        load_checkpoint(self.lm_model, optimizer=None, lr_scheduler=None)
        self._lm_key = 'realm_lm'

        self.retriever = retriever
        self.top_k = self.retriever.top_k
        self._retriever_key = 'retriever'

    def forward(self, tokens, attention_mask, query_block_indices):
        # [batch_size x k x seq_length]
        topk_block_tokens, topk_block_attention_mask = self.retriever.retrieve_evidence_blocks(
            tokens, attention_mask, query_block_indices=query_block_indices, include_null_doc=True)
        batch_size = tokens.shape[0]

        seq_length = topk_block_tokens.shape[2]
        topk_block_tokens = torch.cuda.LongTensor(topk_block_tokens).reshape(-1, seq_length)
        topk_block_attention_mask = torch.cuda.LongTensor(topk_block_attention_mask).reshape(-1, seq_length)

        # [batch_size x k x embed_size]
        true_model = self.retriever.ict_model.module.module
        fresh_block_logits = true_model.embed_block(topk_block_tokens, topk_block_attention_mask)
        fresh_block_logits = fresh_block_logits.reshape(batch_size, self.top_k, -1)

        # [batch_size x embed_size x 1]
        query_logits = true_model.embed_query(tokens, attention_mask).unsqueeze(2)

        # [batch_size x k]
        fresh_block_scores = torch.matmul(fresh_block_logits, query_logits).squeeze()
        block_probs = F.softmax(fresh_block_scores, dim=1)

        # [batch_size * k x seq_length]
        tokens = torch.stack([tokens.unsqueeze(1)] * self.top_k, dim=1).reshape(-1, seq_length)
        attention_mask = torch.stack([attention_mask.unsqueeze(1)] * self.top_k, dim=1).reshape(-1, seq_length)

        # [batch_size * k x 2 * seq_length]
        all_tokens = torch.cat((tokens, topk_block_tokens), axis=1)
        all_attention_mask = torch.cat((attention_mask, topk_block_attention_mask), axis=1)
        all_token_types = torch.zeros(all_tokens.shape).type(torch.int64).cuda()

        # [batch_size x k x 2 * seq_length x vocab_size]
        lm_logits, _ = self.lm_model.forward(all_tokens, all_attention_mask, all_token_types)
        lm_logits = lm_logits.reshape(batch_size, self.top_k, 2 * seq_length, -1)
        return lm_logits, block_probs

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._lm_key] = self.lm_model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._retriever_key] = self.retriever.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        self.lm_model.load_state_dict(state_dict[self._lm_key], strict)
        self.retriever.load_state_dict(state_dict[self._retriever_key], strict)


class REALMRetriever(MegatronModule):
    """Retriever which uses a pretrained ICTBertModel and a HashedIndex"""
    def __init__(self, ict_model, ict_dataset, block_data, hashed_index, top_k=5):
        super(REALMRetriever, self).__init__()
        self.ict_model = ict_model
        self.ict_dataset = ict_dataset
        self.block_data = block_data
        self.hashed_index = hashed_index
        self.top_k = top_k
        self._ict_key = 'ict_model'

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

    def retrieve_evidence_blocks(self, query_tokens, query_pad_mask, query_block_indices=None, include_null_doc=False):
        """Embed blocks to be used in a forward pass"""
        with torch.no_grad():
            true_model = self.ict_model.module.module
            query_embeds = detach(true_model.embed_query(query_tokens, query_pad_mask))
        _, block_indices = self.hashed_index.search_mips_index(query_embeds, top_k=self.top_k, reconstruct=False)
        all_topk_tokens, all_topk_pad_masks = [], []
        for query_idx, indices in enumerate(block_indices):
            # [k x meta_dim]
            # exclude trivial candidate if it appears, else just trim the weakest in the top-k
            topk_metas = [self.block_data.meta_data[idx] for idx in indices if idx != query_block_indices[query_idx]]
            topk_block_data = [self.ict_dataset.get_block(*block_meta) for block_meta in topk_metas[:self.top_k - 1]]
            if include_null_doc:
                topk_block_data.append(self.ict_dataset.get_null_block())
            topk_tokens, topk_pad_masks = zip(*topk_block_data)

            all_topk_tokens.append(np.array(topk_tokens))
            all_topk_pad_masks.append(np.array(topk_pad_masks))

        # [batch_size x k x seq_length]
        return np.array(all_topk_tokens), np.array(all_topk_pad_masks)

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._ict_key] = self.ict_model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        self.ict_model.load_state_dict(state_dict[self._ict_key], strict)


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
