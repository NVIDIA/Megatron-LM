# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Helpers for defining sharding for optimizer states based on existing sharding for model parameters. """

import logging
from copy import deepcopy
from dataclasses import replace
from itertools import chain
from typing import Dict, Iterable, List, Tuple, Union

logger = logging.getLogger(__name__)

import torch

from .dict_utils import nested_values
from .mapping import (
    LocalNonpersistentObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
)
from .utils import extract_sharded_tensors_and_factories

HAVE_APEX_OR_TE = True
try:
    import transformer_engine
except ModuleNotFoundError:
    try:
        import apex
    except ModuleNotFoundError:
        HAVE_APEX_OR_TE = False


def get_optim_param_to_id_map(optim_params_iter: Iterable[torch.nn.Parameter]) -> Dict[int, int]:
    param_mappings = {}
    for i, param in enumerate(optim_params_iter):
        if id(param) not in param_mappings:
            param_mappings[id(param)] = i
    return param_mappings


def get_param_id_to_sharded_param_map(
    model_sharded_state_dict: ShardedStateDict, optim_params_iter: Iterable[torch.nn.Parameter]
) -> Dict[int, Union[ShardedTensor, ShardedTensorFactory]]:
    """Generate mapping from optimizer state ids to model sharded parameters.

    Args:
        model_sharded_state_dict: sharded state dict with all model sharded tensors (can have any structure)
        optim_params_iter: iterable which iterates over model parameters tracked by the optimizer.
            The iteration must be in the same order as in the optimizer parameters.

    Returns:
        Dict[int, Union[ShardedTensor, ShardedTensorFactory]]: mapping from optimizer state ids
            to model sharded parameters.
    """
    model_sharded_state_dict, _ = extract_sharded_tensors_and_factories(model_sharded_state_dict)
    id_to_sharded_param_map = {}
    param_to_id_map = get_optim_param_to_id_map(optim_params_iter)
    for ten in nested_values(model_sharded_state_dict):
        if id(ten.data) in param_to_id_map:
            id_to_sharded_param_map[param_to_id_map[id(ten.data)]] = ten
        else:
            logger.debug(f'{ten} is not tracked by the optimizer')

    if not id_to_sharded_param_map:
        logger.warning(
            "Sharded parameters mapping is empty. It means tensors in model state dict"
            " do not correspond to tensors in optimizer parameters map."
            " Make sure to call state_dict with `keep_vars=True`."
        )
    return id_to_sharded_param_map


def make_sharded_optimizer_tensor(
    model_param: Union[ShardedTensor, ShardedTensorFactory], optim_param: torch.Tensor, prefix: str
) -> Union[ShardedTensor, ShardedTensorFactory]:
    """Build a ShardedTensor or ShardedTensorFactory for optimizer param based on model param

    Args:
        model_param (Union[ShardedTensor, ShardedTensorFactory]): model param
        optim_param (torch.Tensor): corresponding optimizer param
        prefix (str): optimizer prefix for the ShardedTensor or ShardedTensorFactory

    Returns:
        Union[ShardedTensor, ShardedTensorFactory]: wrapped optimizer parameter
    """
    if isinstance(model_param, ShardedTensorFactory):
        return replace(model_param, key=f'{prefix}.{model_param.key}', data=optim_param)

    assert (
        tuple(optim_param.shape) == model_param.local_shape
    ), f'Optimizer shape ({tuple(optim_param.shape)} does not match model shape ({model_param.local_shape})'
    sh_ten = replace(
        model_param, key=f'{prefix}.{model_param.key}', data=optim_param, dtype=optim_param.dtype
    )
    sh_ten.validate_metadata_integrity()
    return sh_ten


def optim_state_to_sharding_state(
    optim_state_dict: StateDict,
    id_to_sharded_param_map: Dict[int, ShardedTensor],
    exclude_keys: Tuple[str] = (),
):
    """Turn optimizer state dict to sharded state dict based on model state dict *in-place*.

    Can be used to add sharding information to most common optimizer state dict.
    Creates separate ShardedTensors for each key in `optim_state_dict['state']`
    (e.g. for torch.optim.Adam there will be separate tensors for `exp_avg` and `exp_avg_sq`)

    Args:
        optim_state_dict (StateDict): optimizer state dict with
            state parameters under `state` key and group hyperparameters under `param_groups` -> `params` key.
        id_to_sharded_param_map (Dict[int, ShardedTensor]): mapping from optimizer param ids to model sharded tensors.
            Can be generated with `get_param_id_to_sharded_param_map` function
        exclude_keys (Tuple[str]): optimizer state keys to exclude from the final state dict.

    Returns:
        None: state dict is modified in place
    """
    sharded_state = {}
    for param_id, param_state in optim_state_dict['state'].items():
        sharded_state[param_id] = {}
        for state_key, param in param_state.items():
            if state_key in exclude_keys:
                continue
            if not HAVE_APEX_OR_TE and state_key == 'step':
                sharded_state[param_id][state_key] = param
            elif param_id in id_to_sharded_param_map:
                sharded_state[param_id][state_key] = make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id], param, prefix=f'optimizer.state.{state_key}'
                )
            else:
                raise ValueError(f'Param id {param_id} does not match any model sharded param')

    optim_state_dict['param_groups'] = deepcopy(optim_state_dict['param_groups'])
    for group in optim_state_dict['param_groups']:
        group['params'] = LocalNonpersistentObject(group['params'])
    optim_state_dict['state'] = sharded_state
