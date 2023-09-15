# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron Module"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import logging

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig

_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    # def __init__(self, config: TransformerConfig, share_word_embeddings=True):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Use this function to override the state dict for
           saving checkpoints.
        """

        return self.state_dict(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix=''):
        """ Override sharded_state_dict when using distributed checkpointing.
            keep_vars must always be set to True so that optimizer states
            can be sharded.
        """
        return self.state_dict(prefix=prefix, keep_vars=True)

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(
            input_tensor) == 1, 'input_tensor should only be length 1 for gpt'
        self.decoder.set_input_tensor(input_tensor[0])

    def compute_loss(self, loss, logits):
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        loss = tensor_parallel.vocab_parallel_cross_entropy(
            logits.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss

    def initialize_last_stage_with_word_embeddings(self, llm_model):

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
            self.output_layer.weight.data.fill_(0)
            self.output_layer.weight.shared = True

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

        elif not getattr(llm_model, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            llm_model.embedding_warning_printed = True


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class Float16Module(MegatronModule):
    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(Float16Module, self).__init__(config)
        self.config = config
        self.fp16 = config.fp16
        self.bf16 = config.bf16

        if self.fp16:
            self.add_module('module', module.half())

            def float16_convertor(val):
                return val.half()

        elif self.bf16:
            self.add_module('module', module.bfloat16())

            def float16_convertor(val):
                return val.bfloat16()

        else:
            raise Exception(
                'Either config.fp16 or config.bf16 should be True.')

        self.float16_convertor = float16_convertor

    def set_input_tensor(self, input_tensor):
        return self.module.set_input_tensor(input_tensor)

    def forward(self, *inputs, **kwargs):
        if parallel_state.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if parallel_state.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """ Retrieve state_dict from the module being wrapped."""
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix=''):
        """ Retrieve state_dict from the module being wrapped.
            When using distributed checkpointing, keep_vars must always be set to True.
        """
        return self.module.sharded_state_dict(prefix=prefix, keep_vars=True)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
