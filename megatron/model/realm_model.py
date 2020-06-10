import os
import torch

from megatron import get_args
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.model import BertModel
from megatron.module import MegatronModule
from megatron import mpu


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

    def forward(self, query_tokens, query_attention_mask, block_tokens, block_attention_mask):
        """Run a forward pass for each of the models and return the respective embeddings."""
        query_logits = self.embed_query(query_tokens, query_attention_mask)
        block_logits = self.embed_block(block_tokens, block_attention_mask)
        return query_logits, block_logits

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

    def init_state_dict_from_bert(self):
        """Initialize the state from a pretrained BERT model on iteration zero of ICT pretraining"""
        args = get_args()
        tracker_filename = get_checkpoint_tracker_filename(args.bert_load)
        if not os.path.isfile(tracker_filename):
            raise FileNotFoundError("Could not find BERT load for ICT")
        with open(tracker_filename, 'r') as f:
            iteration = int(f.read().strip())
            assert iteration > 0

        checkpoint_name = get_checkpoint_name(args.bert_load, iteration, False)
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        except BaseException:
            raise ValueError("Could not load checkpoint")

        # load the LM state dict into each model
        model_dict = state_dict['model']['language_model']
        self.query_model.language_model.load_state_dict(model_dict)
        self.block_model.language_model.load_state_dict(model_dict)

        # give each model the same ict_head to begin with as well
        query_ict_head_state_dict = self.state_dict_for_save_checkpoint()[self._query_key]['ict_head']
        self.block_model.ict_head.load_state_dict(query_ict_head_state_dict)
