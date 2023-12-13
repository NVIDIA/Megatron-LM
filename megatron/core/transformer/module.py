# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
"""Megatron Module."""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


class MegatronModule(torch.nn.Module):
    """Base Megatron module inhertied by all Models.

    Megatron specific extensions of torch Module with support
    for pipelining

    Args:
        config (TransformerConfig): Transformer config
    """

    # def __init__(self, config: TransformerConfig, share_word_embeddings=True):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    def state_dict_for_save_checkpoint(self, prefix: str = '', keep_vars: bool = False):
        """Override state dict for saving checkpoints Use this function to override the
        state dict for saving checkpoints.

        Args:
            prefix (str, optional): _description_. Defaults to ''.
            keep_vars (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        return self.state_dict(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix='', sharded_key_prefix=None, sharded_offsets=()):
        self._intermediate_sharded_state_dict(prefix, sharded_key_prefix, sharded_offsets)


    def _intermediate_sharded_state_dict(self, prefix='', sharded_key_prefix=None, sharded_offsets=()):
        """Sharded state dict with Distributed Checkpointing.

        General definition of sharded_state_dict tries to call `sharded_state_dict`
        of submodules when possible, otherwise assumes tensors are replicated
        across TP and DP.
        When overriding, keep_vars argument of plain `state_dict` method must
        always be set to True so that optimizer states can be sharded.

        Args:
            prefix (str): prefix for the state dict keys
            sharded_key_prefix (str, optional): prefix for the ShardedTensor keys.
                If None, the same prefix as for state dict keys is assumed.
            sharded_offsets (Iterable[Tuple[int, int, int]], optional): sharding already
                applied (e.g. PP related) by sup-modules. Passed along to ShardedTensor

        Returns:
            dict: dictionary of state dict keys mapped to ShardedTensors
        """
        sharded_key_prefix = prefix if sharded_key_prefix is None else sharded_key_prefix
        sharded_state_dict = {}

        for name, module in self._modules.items():
            if hasattr(module, 'sharded_state_dict'):
                module_sharded_sd = module.sharded_state_dict(
                    prefix=f'{prefix}{name}.',
                    sharded_key_prefix=f'{sharded_key_prefix}{name}.',
                    sharded_offsets=sharded_offsets,
                )
            else:
                module_sd = module.state_dict(prefix='', keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f'{prefix}{name}.',
                    f'{sharded_key_prefix}{name}.',
                    {},
                    sharded_offsets,
                )
            sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict


def conversion_helper(val, conversion):
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class Float16Module(MegatronModule):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

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
            raise Exception('Either config.fp16 or config.bf16 should be True.')

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
        """Retrieve state_dict from the module being wrapped."""
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix='', *args, **kwargs):
        """Retrieve state_dict from the module being wrapped.

        When using distributed checkpointing, keep_vars must always be set to True.
        """
        return self.module.sharded_state_dict(prefix, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
