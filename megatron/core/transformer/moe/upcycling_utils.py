# Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
""" Helpers for converting a dense model to a MoE model in runtime """
from megatron.core import mpu


def _get_keys_endswith(model, suffix):
    """
    Retrieve keys from the model that end with a specified suffix.
    """
    return [k for k in model if k.endswith(suffix)]


def _covert_to_moe_state_dict(state_dict, moe_model):
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

    mlp = moe_model.get_submodule('decoder.layers.0.mlp')

    moe_state_dict = moe_model.state_dict()
    new_state_dict = state_dict

    mlp_lm_weight_keys = _get_keys_endswith(new_state_dict, 'mlp.linear_fc1.layer_norm_weight')
    mlp_lm_bias_keys = _get_keys_endswith(new_state_dict, 'mlp.linear_fc1.layer_norm_bias')
    mlp_fc1_weight_keys = _get_keys_endswith(new_state_dict, 'mlp.linear_fc1.weight')
    mlp_fc2_weight_keys = _get_keys_endswith(new_state_dict, 'mlp.linear_fc2.weight')
    mlp_fc1_bias_keys = _get_keys_endswith(new_state_dict, 'mlp.linear_fc1.bias')
    mlp_fc2_bias_keys = _get_keys_endswith(new_state_dict, 'mlp.linear_fc2.bias')
    mlp_fc1_extra_state_keys = _get_keys_endswith(new_state_dict, 'mlp.linear_fc1._extra_state')
    mlp_fc2_extra_state_keys = _get_keys_endswith(new_state_dict, 'mlp.linear_fc2._extra_state')

    for key in mlp_lm_weight_keys:
        params = new_state_dict.pop(key)
        new_key = key.replace('mlp.linear_fc1.layer_norm_weight', 'pre_mlp_layernorm.weight')
        new_state_dict[new_key] = params

    for key in mlp_lm_bias_keys:
        params = new_state_dict.pop(key)
        new_key = key.replace('mlp.linear_fc1.layer_norm_bias', 'pre_mlp_layernorm.bias')
        new_state_dict[new_key] = params

    for mlp_weight_key in mlp_fc1_weight_keys:
        router_key = mlp_weight_key.replace('mlp.linear_fc1.weight', 'mlp.router.weight')
        new_state_dict[router_key] = moe_state_dict[router_key].data.data.clone()

    use_te_grouped_gemm = 'decoder.layers.0.mlp.experts.linear_fc1.weight0' in moe_state_dict

    if mlp.config.moe_grouped_gemm and use_te_grouped_gemm:
        for mlp_weight_key in mlp_fc1_weight_keys:
            weight_tensor = new_state_dict.pop(mlp_weight_key)
            for expert_i in range(mlp.num_local_experts):
                new_key = mlp_weight_key.replace(
                    'mlp.linear_fc1.weight', f'mlp.experts.linear_fc1.weight{expert_i}'
                )
                new_state_dict[new_key] = weight_tensor.clone()

        for mlp_weight_key in mlp_fc2_weight_keys:
            weight_tensor = new_state_dict.pop(mlp_weight_key)
            for expert_i in range(mlp.num_local_experts):
                new_key = mlp_weight_key.replace(
                    'mlp.linear_fc2.weight', f'mlp.experts.linear_fc2.weight{expert_i}'
                )
                new_state_dict[new_key] = weight_tensor.clone()

        for extra_state_key in mlp_fc1_extra_state_keys:
            new_state_dict.pop(extra_state_key)
            new_key = extra_state_key.replace(
                'mlp.linear_fc1._extra_state', 'mlp.experts.linear_fc1._extra_state'
            )
            new_state_dict[new_key] = None

        for extra_state_key in mlp_fc2_extra_state_keys:
            new_state_dict.pop(extra_state_key)
            new_key = extra_state_key.replace(
                'mlp.linear_fc2._extra_state', 'mlp.experts.linear_fc2._extra_state'
            )
            new_state_dict[new_key] = None

    elif mlp.config.moe_grouped_gemm:
        for mlp_weight_key in mlp_fc1_weight_keys:
            weight_tensor = new_state_dict.pop(mlp_weight_key)
            shape = weight_tensor.shape
            weight_tensor = weight_tensor.repeat(mlp.num_local_experts, 1, 1)
            weight_tensor = weight_tensor.permute(0, 2, 1).reshape(
                shape[1], mlp.num_local_experts * shape[0]
            )
            new_key = mlp_weight_key.replace('mlp.linear_fc1.weight', 'mlp.experts.weight1')
            new_state_dict[new_key] = weight_tensor

        for mlp_weight_key in mlp_fc2_weight_keys:
            weight_tensor = new_state_dict.pop(mlp_weight_key)
            shape = weight_tensor.shape
            weight_tensor = weight_tensor.repeat(mlp.num_local_experts, 1, 1)
            weight_tensor = weight_tensor.permute(0, 2, 1).reshape(
                mlp.num_local_experts * shape[1], shape[0]
            )
            new_key = mlp_weight_key.replace('mlp.linear_fc2.weight', 'mlp.experts.weight2')
            new_state_dict[new_key] = weight_tensor

    else:

        def covert_to_experts(keys):
            for key in keys:
                params = new_state_dict.pop(key)
                new_key_format_str = key.replace('mlp', 'mlp.experts.local_experts.{}')
                for expert_i in range(mlp.num_local_experts):
                    new_key = new_key_format_str.format(expert_i)
                    if hasattr(params, 'clone'):
                        new_state_dict[new_key] = params.clone()
                    else:
                        # set extra_state to None for now
                        new_state_dict[new_key] = None

        covert_to_experts(mlp_fc1_weight_keys)
        covert_to_experts(mlp_fc2_weight_keys)
        covert_to_experts(mlp_fc1_bias_keys)
        covert_to_experts(mlp_fc2_bias_keys)
        covert_to_experts(mlp_fc1_extra_state_keys)
        covert_to_experts(mlp_fc2_extra_state_keys)

    return new_state_dict


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
        state_dict['model'] = _covert_to_moe_state_dict(dense_model[0].state_dict(), moe_model[0])
    else:
        assert len(moe_model) == len(dense_model)
        for i in range(len(moe_model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict['model%d' % i] = _covert_to_moe_state_dict(
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
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            moe_model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    return iteration, num_floating_point_operations_so_far
