import numpy as np
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.checkpointing import load_checkpoint
from megatron.data.realm_index import detach, BlockData, FaissMIPSIndex
from megatron.model import BertModel
from megatron.model.utils import get_linear_layer, init_method_normal
from megatron.module import MegatronModule


class REALMAnswerSpanModel(MegatronModule):
    def __init__(self, realm_model, mlp_hidden_size=64):
        super(REALMAnswerSpanModel, self).__init__()
        self.realm_model = realm_model
        self.mlp_hidden_size = mlp_hidden_size

        args = get_args()
        init_method = init_method_normal(args.init_method_std)
        self.fc1 = get_linear_layer(2 * args.hidden_size, self.mlp_hidden_size, init_method)
        self._fc1_key = 'fc1'
        self.fc2 = get_linear_layer(self.mlp_hidden_size, 1, init_method)
        self._fc2_key = 'fc2'

        max_length = 10
        self.start_ends = []
        for length in range(max_length):
            self.start_ends.extend([(i, i + length) for i in range(288 - length)])

    def forward(self, question_tokens, question_attention_mask, answer_tokens, answer_token_lengths):
        lm_logits, block_probs, topk_block_tokens = self.realm_model(
            question_tokens, question_attention_mask, query_block_indices=None, return_topk_block_tokens=True)

        batch_span_reps, batch_loss_masks = [], []
        # go through batch one-by-one
        for i in range(len(answer_token_lengths)):
            answer_length = answer_token_lengths[i]
            answer_span_tokens = answer_tokens[i][:answer_length]
            span_reps, loss_masks = [], []
            # go through the top k for the batch item
            for logits, block_tokens in zip(lm_logits[i], topk_block_tokens[i]):
                block_logits = logits[len(logits) / 2:]
                span_starts = range(len(block_tokens) - (answer_length - 1))

                # record the start, end indices of spans which match the answer
                matching_indices = set([
                    (idx, idx + answer_length - 1) for idx in span_starts
                    if np.array_equal(block_tokens[idx:idx + answer_length], answer_span_tokens)
                ])
                # create a mask for computing the loss on P(y | z, x)
                # [num_spans]
                loss_masks.append(torch.LongTensor([int(idx_pair in matching_indices) for idx_pair in self.start_ends]))

                # get all of the candidate spans that need to be fed to MLP
                # [num_spans x 2 * embed_size]
                span_reps.append([torch.cat((block_logits[s], block_logits[e])) for (s, e) in self.start_ends])

            # data for all k blocks for a single batch item
            # [k x num_spans]
            batch_loss_masks.append(torch.stack(loss_masks))
            # [k x num_spans x 2 * embed_size]
            batch_span_reps.append(torch.stack(span_reps))

        # data for all batch items
        # [batch_size x k x num_spans]
        batch_loss_masks = torch.stack(batch_loss_masks)
        batch_span_reps = torch.stack(batch_span_reps)
        # [batch_size x k x num_spans]
        batch_span_logits = self.fc2(self.fc1(batch_span_reps)).squeeze()

        return batch_span_logits, batch_loss_masks, block_probs

        # block_probs = block_probs.unsqueeze(2).unsqueeze(3).expand_as(lm_logits)
        # lm_logits = torch.sum(lm_logits * block_probs, dim=1)


class REALMBertModel(MegatronModule):
    def __init__(self, retriever):
        super(REALMBertModel, self).__init__()
        bert_args = dict(
            num_tokentypes=2,
            add_binary_head=False,
            parallel_output=True
        )
        self.lm_model = BertModel(**bert_args)
        load_checkpoint(self.lm_model, optimizer=None, lr_scheduler=None)
        self._lm_key = 'realm_lm'

        self.retriever = retriever
        self.top_k = self.retriever.top_k
        self._retriever_key = 'retriever'

    def forward(self, tokens, attention_mask, query_block_indices, return_topk_block_tokens=False):
        # [batch_size x k x seq_length]
        topk_block_tokens, topk_block_attention_mask = self.retriever.retrieve_evidence_blocks(
            tokens, attention_mask, query_block_indices=query_block_indices, include_null_doc=True)
        batch_size = tokens.shape[0]
        # create a copy in case it needs to be returned
        ret_topk_block_tokens = np.array(topk_block_tokens)

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

        if return_topk_block_tokens:
            return lm_logits, block_probs, ret_topk_block_tokens

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

    def reload_index(self):
        args = get_args()
        print("loading from file", flush=True)
        self.block_data = BlockData.load_from_file(args.block_data_path)
        print("resetting index", flush=True)
        self.hashed_index.reset_index()
        print("adding block data", flush=True)
        self.hashed_index.add_block_embed_data(self.block_data)

    def prep_query_text_for_retrieval(self, query_text):
        padless_max_len = self.ict_dataset.max_seq_length - 2
        query_tokens = self.ict_dataset.encode_text(query_text)[:padless_max_len]

        query_tokens, query_pad_mask = self.ict_dataset.concat_and_pad_tokens(query_tokens)
        query_tokens = torch.cuda.LongTensor(np.array(query_tokens).reshape(1, -1))
        query_pad_mask = torch.cuda.LongTensor(np.array(query_pad_mask).reshape(1, -1))

        return query_tokens, query_pad_mask

    def retrieve_evidence_blocks_text(self, query_text):
        """Get the top k evidence blocks for query_text in text form"""
        print("-" * 100)
        print("Query: ", query_text)
        query_tokens, query_pad_mask = self.prep_query_text_for_retrieval(query_text)
        topk_block_tokens, _ = self.retrieve_evidence_blocks(query_tokens, query_pad_mask)
        for i, block in enumerate(topk_block_tokens[0]):
            block_text = self.ict_dataset.decode_tokens(block)
            print('\n    > Block {}: {}'.format(i, block_text))

    def retrieve_evidence_blocks(self, query_tokens, query_pad_mask, query_block_indices=None, include_null_doc=False):
        """Embed blocks to be used in a forward pass"""
        with torch.no_grad():
            if hasattr(self.ict_model, 'module'):
                true_model = self.ict_model.module
                if hasattr(true_model, 'module'):
                    true_model = true_model.module
            else:
                true_model = self.ict_model
            # print("true model: ", true_model, flush=True)

            query_embeds = detach(self.ict_model(query_tokens, query_pad_mask, None, None, only_query=True))
        _, block_indices = self.hashed_index.search_mips_index(query_embeds, top_k=self.top_k, reconstruct=False)
        all_topk_tokens, all_topk_pad_masks = [], []

        # this will result in no candidate exclusion
        if query_block_indices is None:
            query_block_indices = [-1] * len(block_indices)

        top_k_offset = int(include_null_doc)
        for query_idx, indices in enumerate(block_indices):
            # [k x meta_dim]
            # exclude trivial candidate if it appears, else just trim the weakest in the top-k
            topk_metas = [self.block_data.meta_data[idx] for idx in indices if idx != query_block_indices[query_idx]]
            topk_block_data = [self.ict_dataset.get_block(*block_meta) for block_meta in topk_metas[:self.top_k - top_k_offset]]
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
