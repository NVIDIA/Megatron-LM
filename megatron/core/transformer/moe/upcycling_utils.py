# Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
""" Helpers for converting a dense model to a MoE model in runtime """
import copy
from enum import Enum

import torch

from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import BaseMoELayer

ExpertsType = Enum('ExpertsType', ('SequentialMLP', 'TEGroupedMLP'))
ActivationFuncName = Enum('ActivationFuncName', ('gelu', 'silu', 'squared_relu'))


def _get_keys_endswith(model, suffix):
    """
    Retrieve keys from the model that end with a specified suffix.
    """
    return [k for k in model if k.endswith(suffix)]


def _find_submodule(model, submodule_name):
    """
    Find sub-module in model
    """
    for name, submodule in model.named_modules():
        if name.endswith("." + submodule_name) or name == submodule_name:
            return submodule
    return None


def _get_config(moe_model, dense_model):
    """
    Get various params from dense state dict and moe model.
    """
    # Find mlp sub-module in moe model and get relatived args
    mlp = _find_submodule(moe_model, "mlp")
    assert mlp is not None, f'can not find mlp layer in moe model: {moe_model}'
    assert isinstance(mlp, BaseMoELayer), (
        f'The mlp layer {type(mlp)} is not supported by upcycling.'
        f"Please use mlp layer inherited from {type(BaseMoELayer)}"
    )
    num_local_experts = mlp.num_local_experts
    num_experts = mlp.config.num_moe_experts
    gated_linear_unit = mlp.config.gated_linear_unit
    moe_router_topk = mlp.config.moe_router_topk
    moe_router_pre_softmax = mlp.config.moe_router_pre_softmax
    moe_ffn_hidden_size = mlp.config.moe_ffn_hidden_size
    func_name = mlp.config.activation_func.__name__

    if func_name == "gelu":
        activation_func_name = ActivationFuncName.gelu
    elif func_name == "silu":
        activation_func_name = ActivationFuncName.silu
    elif func_name == "squared_relu":
        activation_func_name = ActivationFuncName.squared_relu
    else:
        raise ValueError(
            f"The activation func is not supported by upcycling."
            + f"Valid options are: {list(ActivationFuncName.__members__.keys())}"
            + f"But got {func_name} "
        )
    ep_rank = mlp.ep_group.rank()

    # Find experts sub-module in moe model and get relatived args
    experts = _find_submodule(mlp, "experts")
    assert (
        experts is not None
    ), f'The model is not supported by upcycling. Can not find experts in {mlp}'
    if isinstance(experts, SequentialMLP):
        experts_type = ExpertsType.SequentialMLP
    elif isinstance(experts, TEGroupedMLP):
        experts_type = ExpertsType.TEGroupedMLP
    else:
        raise TypeError(
            f"The experts type {type(experts)} is not supported by upcycling."
            f" Valid options are: {list(ExpertsType.__members__.keys())}"
        )

    # Find mlp sub-module in dense model and get relatived args
    dense_mlp = _find_submodule(dense_model, "mlp")
    assert dense_mlp is not None, f'can not find mlp layer in moe model: {moe_model}'
    dense_ffn_hidden_size = dense_mlp.config.ffn_hidden_size

    # calc granularity and expansion_rate
    assert (
        dense_ffn_hidden_size % moe_ffn_hidden_size == 0
    ), "The ffn hidden size of dense model must be divisible by ffn hidden size of moe model."
    granularity = dense_ffn_hidden_size // moe_ffn_hidden_size
    assert (
        num_experts % granularity == 0
    ), "The number of experts must be divisible by granularity for upcycling"
    expansion_rate = num_experts // granularity

    return (
        num_local_experts,
        moe_router_topk,
        granularity,
        expansion_rate,
        experts_type,
        gated_linear_unit,
        activation_func_name,
        moe_router_pre_softmax,
        ep_rank,
    )


def _convert_to_moe_state_dict(moe_model, dense_model):
    """
    Convert a dense model's state_dict to a MoE model's state_dict.

    This function takes the state dictionary of a dense model and modifies it to fit the
    structure required by a Mixture of Experts model. It handles the necessary
    transformations for weights and biases specific to the MoE architecture.

    Args:
        state_dict (dict): The dense model's state_dict.
        moe_model (nn.Module): The MoE model instance from which to get the submodule
                               and state_dict, must be a model without FP16 and/or
                               DDP wrapper.

    Returns:
        dict: The converted MoE model state_dict, ready for use in the MoE architecture.
    """
    (
        num_local_experts,
        moe_router_topk,
        granularity,
        expansion_rate,
        experts_type,
        gated_linear_unit,
        activation_func_name,
        moe_router_pre_softmax,
        ep_rank,
    ) = _get_config(moe_model, dense_model)

    def _process_router_param(value):
        value = value.data.data.clone()
        value = torch.tensor_split(value, granularity, dim=0)[0]
        value = [t.repeat(granularity, 1) for t in value]
        value = torch.cat(value, dim=0)
        return value

    def _get_moe_activation_scale():
        """
        Calc moe activation scale factor relative to dense activation.
        For more detail please refer to https://arxiv.org/abs/2410.07524.
        """
        if moe_router_pre_softmax:
            moe_activation_scale = (expansion_rate * granularity * granularity) / moe_router_topk
        else:
            moe_activation_scale = granularity
        return moe_activation_scale

    def _get_weight_scale():
        moe_activation_scale = _get_moe_activation_scale()
        if gated_linear_unit == True:
            scale = moe_activation_scale ** (1 / 3)
        elif activation_func_name == ActivationFuncName.squared_relu:
            scale = moe_activation_scale ** (1 / 3)
        else:
            scale = moe_activation_scale ** (1 / 2)
        return scale

    def _process_fc1_weight_param(param):
        param = param.clone()
        scale = _get_weight_scale()
        param = param * scale
        if activation_func_name == ActivationFuncName.silu and gated_linear_unit == True:
            param_1, param_2 = torch.chunk(param, 2, dim=0)
            params_1 = torch.tensor_split(param_1, granularity, dim=0)
            params_2 = torch.tensor_split(param_2, granularity, dim=0)
            params = [torch.cat([params_1[i], params_2[i]], dim=0) for i in range(granularity)]
        else:
            params = torch.tensor_split(param, granularity, dim=0)
        params = params * expansion_rate
        return params

    def _process_fc1_bias_param(param):
        # need to add test case, and re-implement this func according the test result
        params = _process_fc1_weight_param(param)
        params = [tensor.squeeze(0) for tensor in params]
        return params

    def _process_fc2_weight_param(param):
        param = param.clone()
        scale = _get_weight_scale()
        param = param * scale
        params = torch.tensor_split(param, granularity, dim=1)
        params = params * expansion_rate
        return params

    def _process_fc2_bias_param(param):
        param = param.clone()
        params = param.repeat(granularity * expansion_rate, 1)
        return params

    # Step 1. Copy values from dense state dict to moe state dict as init.
    dense_state_dict = copy.deepcopy(dense_model.state_dict())
    moe_state_dict = copy.deepcopy(moe_model.state_dict())
    for key in dense_state_dict.keys() & moe_state_dict.keys():
        moe_state_dict[key] = dense_state_dict[key]

    # Step 2. Convert key for layer norm layer
    def _convert_key_value(
        dist_dict=moe_state_dict,
        src_dict=dense_state_dict,
        key_replace_old=None,
        key_replace_new=None,
        value_process_func=lambda x: x,
    ):
        """
        Get value from src_dict according to the key, and copy value to dist_dict with new key.
        The new key is generated by formatting old key with defined pattern.
        """
        keys = _get_keys_endswith(src_dict, key_replace_old)
        for key in keys:
            value = src_dict[key]
            new_value = value_process_func(value)
            new_key = key.replace(key_replace_old, key_replace_new)
            dist_dict[new_key] = new_value.clone() if hasattr(new_value, 'clone') else new_value
        return

    _convert_key_value(
        key_replace_old='mlp.linear_fc1.layer_norm_weight',
        key_replace_new='pre_mlp_layernorm.weight',
    )
    _convert_key_value(
        key_replace_old='mlp.linear_fc1.layer_norm_bias', key_replace_new='pre_mlp_layernorm.bias'
    )

    # Step 3. Convert key and value for router layer
    _convert_key_value(
        src_dict=moe_state_dict,
        key_replace_old='mlp.router.weight',
        key_replace_new='mlp.router.weight',
        value_process_func=_process_router_param,
    )

    # Step 4. Expand linear layer
    def _expand_key_value(
        dist_dict=moe_state_dict,
        src_dict=dense_state_dict,
        key_replace_old=None,
        key_replace_new=None,
        value_process_func=lambda x: x,
        num_local_experts=num_local_experts,
    ):
        """
        Get value from src_dict according to the key,
        Copy and expand value to dist_dict with new key.
        The new key is generated by formatting old key with defined pattern.
        """
        keys = _get_keys_endswith(src_dict, key_replace_old)
        for key in keys:
            param = src_dict[key]
            params = value_process_func(param)
            for idx in range(num_local_experts):
                new_key = key.replace(key_replace_old, key_replace_new).format(idx)
                dist_dict[new_key] = params[ep_rank * num_local_experts + idx]
        return

    if experts_type == ExpertsType.SequentialMLP:
        _expand_key_value(
            key_replace_old='mlp.linear_fc1.weight',
            key_replace_new='mlp.experts.local_experts.{}.linear_fc1.weight',
            value_process_func=_process_fc1_weight_param,
        )
        _expand_key_value(
            key_replace_old='mlp.linear_fc1.bias',
            key_replace_new='mlp.experts.local_experts.{}.linear_fc1.bias',
            value_process_func=_process_fc1_bias_param,
        )
        _expand_key_value(
            key_replace_old='mlp.linear_fc2.weight',
            key_replace_new='mlp.experts.local_experts.{}.linear_fc2.weight',
            value_process_func=_process_fc2_weight_param,
        )
        _expand_key_value(
            key_replace_old='mlp.linear_fc2.bias',
            key_replace_new='mlp.experts.local_experts.{}.linear_fc2.bias',
            value_process_func=_process_fc2_bias_param,
        )
    elif experts_type == ExpertsType.TEGroupedMLP:
        _expand_key_value(
            key_replace_old='mlp.linear_fc1.weight',
            key_replace_new='mlp.experts.linear_fc1.weight{}',
            value_process_func=_process_fc1_weight_param,
        )
        _expand_key_value(
            key_replace_old='mlp.linear_fc2.weight',
            key_replace_new='mlp.experts.linear_fc2.weight{}',
            value_process_func=_process_fc2_weight_param,
        )
    else:
        raise ValueError(f"unknown moe weight format {experts_type}")

    return moe_state_dict


def upcycle_state_dict(moe_model, dense_model):
    """
    Convert a dense model's state_dict to a MoE model's state_dict.

    This function facilitates the conversion of the state_dict from a dense model to
    a MoE model, ensuring that the parameters are correctly mapped for each model.

    Args:
        moe_model (nn.Module): The MoE model, must be a model without FP16 and/or DDP wrapper.
        dense_model (nn.Module): The dense model instance.

    Returns:
        dict: A dictionary containing the converted state_dict for the MoE model.
    """

    state_dict = {}
    if len(moe_model) == 1:
        assert len(dense_model) == 1
        state_dict['model'] = _convert_to_moe_state_dict(moe_model[0], dense_model[0])
    else:
        assert len(moe_model) == len(dense_model)
        for i in range(len(moe_model)):
            state_dict['model%d' % i] = _convert_to_moe_state_dict(
                dense_model[i].state_dict(), moe_model[i]
            )
    return state_dict


def load_and_upcycle_model(
    load_dense_ckpt_func, moe_model, dense_model, strict=True, load_args=(), load_kwargs={}
):
    """
    Load a dense model checkpoint and convert it to a MoE model.

    This function loads a checkpoint for a dense model and converts it to the MoE model format,
    allowing for the integration of the dense model's parameters into the MoE architecture.
    For more detail please refer to https://arxiv.org/abs/2410.07524.

    Args:
        load_dense_ckpt_func (callable): The function to load the dense model checkpoint.
        moe_model (nn.Module): The MoE model instance.
        dense_model (nn.Module): The dense model instance.
        strict (bool): Whether to strictly load the state dictionary (default is True).
        load_args (tuple): Positional arguments to pass to the loading function.
        load_kwargs (dict): Keyword arguments to pass to the loading function.
    """

    iteration, num_floating_point_operations_so_far = load_dense_ckpt_func(
        *load_args, **load_kwargs
    )
    state_dict = upcycle_state_dict(moe_model, dense_model)

    if len(moe_model) == 1:
        moe_model[0].load_state_dict(state_dict['model'], strict=strict)
    else:
        for i in range(len(moe_model)):
            moe_model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    return iteration, num_floating_point_operations_so_far
