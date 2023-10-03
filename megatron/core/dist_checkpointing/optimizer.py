# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Optimizer related helpers. """

import logging
from copy import deepcopy
from dataclasses import replace
from itertools import chain
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

import torch

from .dict_utils import nested_values
from .mapping import LocalNonpersitentObject, ShardedStateDict, ShardedTensor, StateDict
from .utils import extract_sharded_tensors


def get_optim_param_to_id_map(optim_params_iter: Iterable[torch.nn.Parameter]) -> Dict[int, int]:
    param_mappings = {}
    for i, param in enumerate(optim_params_iter):
        if id(param) not in param_mappings:
            param_mappings[id(param)] = i
    return param_mappings


def get_param_id_to_sharded_param_map(
    model_sharded_state_dict: ShardedStateDict, optim_params_iter: Iterable[torch.nn.Parameter]
) -> Dict[int, ShardedTensor]:
    model_sharded_state_dict, _ = extract_sharded_tensors(model_sharded_state_dict)
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
    model_param: ShardedTensor, optim_param: torch.Tensor, prefix: str
) -> ShardedTensor:
    assert (
        tuple(optim_param.shape) == model_param.local_shape
    ), f'Optimizer shape ({tuple(optim_param.shape)} does not match model shape ({model_param.local_shape})'
    return replace(
        model_param, key=f'{prefix}.{model_param.key}', data=optim_param, dtype=optim_param.dtype
    )


def optim_state_to_sharding_state(
    optim_state_dict: StateDict,
    id_to_sharded_param_map: Dict[int, ShardedTensor],
    exclude_keys: Tuple[str] = (),
):
    sharded_state = {}
    for param_id, param_state in optim_state_dict['state'].items():
        sharded_state[param_id] = {}
        for state_key, param in param_state.items():
            if state_key in exclude_keys:
                continue
            if param_id in id_to_sharded_param_map:
                sharded_state[param_id][state_key] = make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id], param, prefix=f'optimizer.state.{state_key}'
                )
            else:
                raise ValueError(f'Param id {param_id} does not match any model sharded param')

    optim_state_dict['param_groups'] = deepcopy(optim_state_dict['param_groups'])
    for group in optim_state_dict['param_groups']:
        group['params'] = LocalNonpersitentObject(group['params'])
    optim_state_dict['state'] = sharded_state
