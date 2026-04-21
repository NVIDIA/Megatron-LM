# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import torch

from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.legacy.model import BertModel
from .module import MegatronModule
from megatron.core import mpu
from megatron.legacy.model.enums import AttnMaskType
from megatron.legacy.model.utils import get_linear_layer
from megatron.legacy.model.utils import init_method_normal
from megatron.legacy.model.language_model import get_language_model
from megatron.legacy.model.utils import scaled_init_method_normal
from megatron.legacy.model.bert_model import bert_extended_attention_mask, bert_position_ids


def general_ict_model_provider(only_query_model=False, only_block_model=False):
    """Build the model."""
    args = get_args()
    assert args.ict_head_size is not None, \
        "Need to specify --ict-head-size to provide an ICTBertModel"
    assert mpu.get_tensor_model_parallel_world_size() == 1 and mpu.get_pipeline_model_parallel_world_size() == 1, \
        "Model parallel size > 1 not supported for ICT"

    print_rank_0('building ICTBertModel...')

    # simpler to just keep using 2 tokentypes since the LM we initialize with has 2 tokentypes
    model = ICTBertModel(
        ict_head_size=args.ict_head_size,
        num_tokentypes=2,
        parallel_output=True,
        only_query_model=only_query_model,
        only_block_model=only_block_model)

    return model


class ICTBertModel(MegatronModule):
    """Bert-based module for Inverse Cloze task."""
    def __init__(self,
                 ict_head_size,
                 num_tokentypes=1,
                 parallel_output=True,
                 only_query_model=False,
                 only_block_model=False):
        super(ICTBertModel, self).__init__()
        bert_kwargs = dict(
            ict_head_size=ict_head_size,
            num_tokentypes=num_tokentypes,
            parallel_output=parallel_output
        )
        assert not (only_block_model and only_query_model)
        self.use_block_model = not only_query_model
        self.use_query_model = not only_block_model

        if self.use_query_model:
            # this model embeds (pseudo-)queries - Embed_input in the paper
            self.query_model = IREncoderBertModel(**bert_kwargs)
            self._query_key = 'question_model'

        if self.use_block_model:
            # this model embeds evidence blocks - Embed_doc in the paper
            self.block_model = IREncoderBertModel(**bert_kwargs)
            self._block_key = 'context_model'

    def forward(self, query_tokens, query_attention_mask, block_tokens, block_attention_mask):
        """Run a forward pass for each of the models and return the respective embeddings."""
        query_logits = self.embed_query(query_tokens, query_attention_mask)
        block_logits = self.embed_block(block_tokens, block_attention_mask)
        return query_logits, block_logits

    def embed_query(self, query_tokens, query_attention_mask):
        """Embed a batch of tokens using the query model"""
        if self.use_query_model:
            query_types = torch.cuda.LongTensor(*query_tokens.shape).fill_(0)
            query_ict_logits, _ = self.query_model.forward(query_tokens, query_attention_mask, query_types)
            return query_ict_logits
        else:
            raise ValueError("Cannot embed query without query model.")

    def embed_block(self, block_tokens, block_attention_mask):
        """Embed a batch of tokens using the block model"""
        if self.use_block_model:
            block_types = torch.cuda.LongTensor(*block_tokens.shape).fill_(0)
            block_ict_logits, _ = self.block_model.forward(block_tokens, block_attention_mask, block_types)
            return block_ict_logits
        else:
            raise ValueError("Cannot embed block without block model.")

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Save dict with state dicts of each of the models."""
        state_dict_ = {}
        if self.use_query_model:
            state_dict_[self._query_key] \
                = self.query_model.state_dict_for_save_checkpoint(
                    prefix=prefix, keep_vars=keep_vars)

        if self.use_block_model:
            state_dict_[self._block_key] \
                = self.block_model.state_dict_for_save_checkpoint(
                    prefix=prefix, keep_vars=keep_vars)

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
        except Exception:
            raise ValueError("Could not load checkpoint")

        # load the LM state dict into each model
        model_dict = state_dict['model']['language_model']
        self.query_model.language_model.load_state_dict(model_dict)
        self.block_model.language_model.load_state_dict(model_dict)

        # give each model the same ict_head to begin with as well
        query_ict_head_state_dict = self.state_dict_for_save_checkpoint()[self._query_key]['ict_head']
        self.block_model.ict_head.load_state_dict(query_ict_head_state_dict)


class IREncoderBertModel(MegatronModule):
    """BERT-based encoder for queries or blocks used for learned information retrieval."""
    def __init__(self, ict_head_size, num_tokentypes=2, parallel_output=True):
        super(IREncoderBertModel, self).__init__()
        args = get_args()

        self.ict_head_size = ict_head_size
        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=True,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method)

        self.ict_head = get_linear_layer(args.hidden_size, ict_head_size, init_method)
        self._ict_head_key = 'ict_head'

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype)
        position_ids = bert_position_ids(input_ids)

        lm_output, pooled_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids)

        # Output.
        ict_logits = self.ict_head(pooled_output)
        return ict_logits, None

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        state_dict_[self._ict_head_key] \
            = self.ict_head.state_dict(prefix=prefix,
                                       keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        self.ict_head.load_state_dict(
            state_dict[self._ict_head_key], strict=strict)


