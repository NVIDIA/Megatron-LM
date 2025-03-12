# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import torch
import sys

from megatron.training import get_args, print_rank_0, get_tokenizer
from megatron.core import mpu
from megatron.training.checkpointing import fix_query_key_value_ordering
from megatron.training.checkpointing import get_checkpoint_tracker_filename
from megatron.training.checkpointing import get_checkpoint_name
from megatron.legacy.model.bert_model import bert_position_ids
from megatron.legacy.model.enums import AttnMaskType
from megatron.legacy.model.language_model import get_language_model
from megatron.legacy.model.utils import get_linear_layer
from megatron.legacy.model.utils import init_method_normal
from megatron.legacy.model.utils import scaled_init_method_normal
from .module import MegatronModule

def get_model_provider(only_query_model=False, only_context_model=False,
        biencoder_shared_query_context_model=False):

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building Bienoder model ...')
        model = biencoder_model_provider(only_query_model=only_query_model,
                only_context_model = only_context_model,
                biencoder_shared_query_context_model = \
                biencoder_shared_query_context_model,
                pre_process=pre_process, post_process=post_process)

        return model

    return model_provider


def biencoder_model_provider(only_query_model=False,
                             only_context_model=False,
                             biencoder_shared_query_context_model=False,
                             pre_process=True,
                             post_process=True):
    """Build the model."""

    assert mpu.get_tensor_model_parallel_world_size() == 1 and \
        mpu.get_pipeline_model_parallel_world_size() == 1, \
        "Model parallel size > 1 not supported for ICT"

    print_rank_0('building BiEncoderModel...')

    # simpler to just keep using 2 tokentypes since
    # the LM we initialize with has 2 tokentypes
    model = BiEncoderModel(
        num_tokentypes=2,
        parallel_output=False,
        only_query_model=only_query_model,
        only_context_model=only_context_model,
        biencoder_shared_query_context_model=\
        biencoder_shared_query_context_model,
        pre_process=pre_process,
        post_process=post_process)

    return model


class BiEncoderModel(MegatronModule):
    """Bert-based module for Biencoder model."""

    def __init__(self,
                 num_tokentypes=1,
                 parallel_output=True,
                 only_query_model=False,
                 only_context_model=False,
                 biencoder_shared_query_context_model=False,
                 pre_process=True,
                 post_process=True):
        super(BiEncoderModel, self).__init__()
        args = get_args()

        bert_kwargs = dict(
            num_tokentypes=num_tokentypes,
            parallel_output=parallel_output,
            pre_process=pre_process,
            post_process=post_process)

        self.biencoder_shared_query_context_model = \
            biencoder_shared_query_context_model
        assert not (only_context_model and only_query_model)
        self.use_context_model = not only_query_model
        self.use_query_model = not only_context_model
        self.biencoder_projection_dim = args.biencoder_projection_dim

        if self.biencoder_shared_query_context_model:
            self.model = PretrainedBertModel(**bert_kwargs)
            self._model_key = 'shared_model'
            self.query_model, self.context_model = self.model, self.model
        else:
            if self.use_query_model:
                # this model embeds (pseudo-)queries - Embed_input in the paper
                self.query_model = PretrainedBertModel(**bert_kwargs)
                self._query_key = 'query_model'

            if self.use_context_model:
                # this model embeds evidence blocks - Embed_doc in the paper
                self.context_model = PretrainedBertModel(**bert_kwargs)
                self._context_key = 'context_model'

    def set_input_tensor(self, input_tensor):
        """See megatron.legacy.model.transformer.set_input_tensor()"""
        # this is just a placeholder and will be needed when model
        # parallelism will be used
        # self.language_model.set_input_tensor(input_tensor)
        return

    def forward(self, query_tokens, query_attention_mask, query_types,
                context_tokens, context_attention_mask, context_types):
        """Run a forward pass for each of the models and
        return the respective embeddings."""

        if self.use_query_model:
            query_logits = self.embed_text(self.query_model,
                                           query_tokens,
                                           query_attention_mask,
                                           query_types)
        else:
            raise ValueError("Cannot embed query without the query model.")
        if self.use_context_model:
            context_logits = self.embed_text(self.context_model,
                                             context_tokens,
                                             context_attention_mask,
                                             context_types)
        else:
            raise ValueError("Cannot embed block without the block model.")
        return query_logits, context_logits

    @staticmethod
    def embed_text(model, tokens, attention_mask, token_types):
        """Embed a batch of tokens using the model"""
        logits = model(tokens,
                              attention_mask,
                              token_types)
        return logits

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Save dict with state dicts of each of the models."""
        state_dict_ = {}
        if self.biencoder_shared_query_context_model:
            state_dict_[self._model_key] = \
                self.model.state_dict_for_save_checkpoint(
                    prefix=prefix, keep_vars=keep_vars)
        else:
            if self.use_query_model:
                state_dict_[self._query_key] = \
                    self.query_model.state_dict_for_save_checkpoint(
                        prefix=prefix, keep_vars=keep_vars)

            if self.use_context_model:
                state_dict_[self._context_key] = \
                    self.context_model.state_dict_for_save_checkpoint(
                        prefix=prefix, keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        if self.biencoder_shared_query_context_model:
            print_rank_0("Loading shared query-context model")
            self.model.load_state_dict(state_dict[self._model_key], \
                strict=strict)
        else:
            if self.use_query_model:
                print_rank_0("Loading query model")
                self.query_model.load_state_dict( \
                    state_dict[self._query_key], strict=strict)

            if self.use_context_model:
                print_rank_0("Loading context model")
                self.context_model.load_state_dict( \
                    state_dict[self._context_key], strict=strict)

    def init_state_dict_from_bert(self):
        """Initialize the state from a pretrained BERT model
        on iteration zero of ICT pretraining"""
        args = get_args()

        if args.bert_load is None:
            print_rank_0("bert-load argument is None")
            return

        tracker_filename = get_checkpoint_tracker_filename(args.bert_load)
        if not os.path.isfile(tracker_filename):
            raise FileNotFoundError("Could not find BERT checkpoint")
        with open(tracker_filename, 'r') as f:
            iteration = int(f.read().strip())
            assert iteration > 0

        checkpoint_name = get_checkpoint_name(args.bert_load, iteration, False)
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading BERT checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        except ModuleNotFoundError:
            from megatron.legacy.fp16_deprecated import loss_scaler
            # For backward compatibility.
            print_rank_0(' > deserializing using the old code structure ...')
            sys.modules['fp16.loss_scaler'] = sys.modules[
                'megatron.fp16_deprecated.loss_scaler']
            sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
                'megatron.fp16_deprecated.loss_scaler']
            state_dict = torch.load(checkpoint_name, map_location='cpu')
            sys.modules.pop('fp16.loss_scaler', None)
            sys.modules.pop('megatron.fp16.loss_scaler', None)
        except Exception:
            print_rank_0('could not load the BERT checkpoint')
            sys.exit()

        checkpoint_version = state_dict.get('checkpoint_version', 0)

        # load the LM state dict into each model
        model_dict = state_dict['model']['language_model']

        if self.biencoder_shared_query_context_model:
            self.model.language_model.load_state_dict(model_dict)
            fix_query_key_value_ordering(self.model, checkpoint_version)
        else:
            if self.use_query_model:
                self.query_model.language_model.load_state_dict(model_dict)
                # give each model the same ict_head to begin with as well
                if self.biencoder_projection_dim > 0:
                    query_proj_state_dict = \
                        self.state_dict_for_save_checkpoint()\
                        [self._query_key]['projection_enc']
                fix_query_key_value_ordering(self.query_model, checkpoint_version)

            if self.use_context_model:
                self.context_model.language_model.load_state_dict(model_dict)
                if self.query_model is not None and \
                    self.biencoder_projection_dim > 0:
                    self.context_model.projection_enc.load_state_dict\
                        (query_proj_state_dict)
                fix_query_key_value_ordering(self.context_model, checkpoint_version)


class PretrainedBertModel(MegatronModule):
    """BERT-based encoder for queries or contexts used for
    learned information retrieval."""

    def __init__(self, num_tokentypes=2,
            parallel_output=True, pre_process=True, post_process=True):
        super(PretrainedBertModel, self).__init__()

        args = get_args()
        tokenizer = get_tokenizer()
        self.pad_id = tokenizer.pad
        self.biencoder_projection_dim = args.biencoder_projection_dim
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(
            args.init_method_std, args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process)

        if args.biencoder_projection_dim > 0:
            self.projection_enc = get_linear_layer(args.hidden_size,
                                                   args.biencoder_projection_dim,
                                                   init_method)
            self._projection_enc_key = 'projection_enc'

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        extended_attention_mask = attention_mask.unsqueeze(1)
        #extended_attention_mask = bert_extended_attention_mask(attention_mask)
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        extended_attention_mask,
                                        tokentype_ids=tokentype_ids)
        # This mask will be used in average-pooling and max-pooling
        pool_mask = (input_ids == self.pad_id).unsqueeze(2)

        # Taking the representation of the [CLS] token of BERT
        pooled_output = lm_output[0, :, :]

        # Converting to float16 dtype
        pooled_output = pooled_output.to(lm_output.dtype)

        # Output.
        if self.biencoder_projection_dim:
            pooled_output = self.projection_enc(pooled_output)

        return pooled_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)

        if self.biencoder_projection_dim > 0:
            state_dict_[self._projection_enc_key] = \
                self.projection_enc.state_dict(prefix=prefix,
                                               keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        print_rank_0("loading pretrained weights")
        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)

        if self.biencoder_projection_dim > 0:
            print_rank_0("loading projection head weights")
            self.projection_enc.load_state_dict(
                state_dict[self._projection_enc_key], strict=strict)
