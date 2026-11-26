# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Helpers for defining sharding for optimizer states based on existing sharding
for model parameters.
"""

import logging
from copy import deepcopy
from dataclasses import replace
from typing import Dict, Iterable, Tuple, Union

logger = logging.getLogger(__name__)

import torch

from megatron.core.utils import log_single_rank, to_local_if_dtensor

from .dict_utils import nested_values
from .mapping import (
    LocalNonpersistentObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
    get_checkpoint_parameter_id,
)
from .utils import extract_sharded_tensors_and_factories

KEEP_VARS_HINT = (
    " Make sure state dict contains original torch.nn.Parameters (not pure torch.Tensors)"
    " by passing `keep_vars=True` to `.state_dict()`. If any transformation of the original"
    " parameter is needed, use a ShardedTensorFactory."
)


def get_optim_param_to_id_map(optim_params_iter: Iterable[torch.nn.Parameter]) -> Dict[int, int]:
    """Generate mapping from optimizer param to optimizer state id."""
    param_mappings = {}
    for i, param in enumerate(optim_params_iter):
        param = to_local_if_dtensor(param)
        if id(param) not in param_mappings:
            param_mappings[id(param)] = i
    return param_mappings


def get_param_id_to_sharded_param_map(
    model_sharded_state_dict: ShardedStateDict, optim_params_iter: Iterable[torch.nn.Parameter]
) -> Dict[int, Union[ShardedTensor, ShardedTensorFactory]]:
    """Generate mapping from optimizer state ids to model sharded parameters.

    Args:
        model_sharded_state_dict: sharded state dict with all model sharded tensors
            (can have any structure)
        optim_params_iter: iterable which iterates over model parameters tracked by the optimizer.
            The iteration must be in the same order as in the optimizer parameters.

    Returns:
        Dict[int, Union[ShardedTensor, ShardedTensorFactory]]: mapping from optimizer state ids
            to model sharded parameters.
    """
    model_sharded_state_dict, _ = extract_sharded_tensors_and_factories(model_sharded_state_dict)
    id_to_sharded_param_map = {}
    optim_params = list(optim_params_iter)
    param_to_id_map = get_optim_param_to_id_map(optim_params)
    grouped_members = {}
    # If using PyTorch FSDP2 the values in model_sharded_state_dict would
    # have been converted to local tensors during initialization.
    # See the make_(tp)_sharded_tensor_for_checkpoint functions.
    for ten in nested_values(model_sharded_state_dict):
        param_id = get_checkpoint_parameter_id(ten.data)
        if param_id in param_to_id_map:
            id_to_sharded_param_map[param_to_id_map[param_id]] = ten
        else:
            logger.debug('%s is not tracked by the optimizer', ten)
        parent_id = getattr(ten.data, '_checkpoint_grouped_parent_id', None)
        if parent_id in param_to_id_map:
            member_index = ten.data._checkpoint_grouped_member_index
            grouped_members.setdefault(parent_id, {})[member_index] = ten

    for parent in optim_params:
        members = grouped_members.get(id(parent))
        if members is not None:
            if sorted(members) != list(range(parent.shape[0])):
                raise ValueError('Grouped optimizer checkpoint requires every expert member.')
            id_to_sharded_param_map[param_to_id_map[id(parent)]] = (
                _make_grouped_optimizer_checkpoint_factory(
                    parent, [members[index] for index in range(parent.shape[0])]
                )
            )

    if not id_to_sharded_param_map:
        log_single_rank(
            logger,
            logging.WARNING,
            "Sharded parameters mapping is empty. It means tensors in model state dict"
            " do not correspond to tensors in optimizer parameters map."
            " Make sure to call state_dict with `keep_vars=True`.",
        )
    return id_to_sharded_param_map


def _make_grouped_optimizer_checkpoint_factory(parent, members):
    """Map a whole grouped optimizer state to the model's existing expert checkpoint shards.

    Member factories (for example SwiGLU) retain their build/merge transformations. Only
    serialization splits the parent; optimizer state and DP ownership remain whole-parameter.
    """
    factory_key = members[0].key

    @torch.no_grad()
    def build(key, data, replica_id, flattened_range):
        if flattened_range is not None:
            raise ValueError('Grouped optimizer checkpoint expects a whole parameter.')
        if tuple(data.shape) != tuple(parent.shape):
            raise ValueError('Grouped optimizer state must have the parent parameter shape.')
        # make_sharded_optimizer_tensor prefixes the factory key for each state (master,
        # momentum, ...). Apply that same prefix to every original expert key.
        if not key.endswith(factory_key):
            raise ValueError(f'Unexpected grouped optimizer checkpoint key: {key}')
        prefix = key[: -len(factory_key)]
        result = []
        for index, member in enumerate(members):
            value = data[index]
            if isinstance(member, ShardedTensorFactory):
                result.append(member.build_fn(prefix + member.key, value, replica_id, None))
            else:
                shard = replace(
                    member,
                    key=prefix + member.key,
                    data=value,
                    dtype=value.dtype,
                    replica_id=replica_id,
                )
                shard.validate_metadata_integrity()
                result.append(shard)
        return result

    @torch.no_grad()
    def merge(values):
        return torch.stack(
            [
                member.merge_fn(value) if isinstance(member, ShardedTensorFactory) else value
                for member, value in zip(members, values, strict=True)
            ]
        )

    return ShardedTensorFactory(factory_key, parent, build, merge, members[0].replica_id)


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
    optim_param = to_local_if_dtensor(optim_param)
    if isinstance(model_param, ShardedTensorFactory):
        return replace(model_param, key=f'{prefix}.{model_param.key}', data=optim_param)

    assert tuple(optim_param.shape) == model_param.local_shape, (
        f'Optimizer shape ({tuple(optim_param.shape)} does not match model shape '
        f'({model_param.local_shape})'
    )
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
            state parameters under `state` key and group hyperparameters under
            `param_groups` -> `params` key.
        id_to_sharded_param_map (Dict[int, ShardedTensor]): mapping from optimizer param ids
            to model sharded tensors. Can be generated with `get_param_id_to_sharded_param_map`
            function.
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
            if param_id in id_to_sharded_param_map:
                sharded_state[param_id][state_key] = make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id], param, prefix=f'optimizer.state.{state_key}'
                )
            else:
                raise ValueError(f'Param id {param_id} does not match any model sharded param')

    optim_state_dict['param_groups'] = deepcopy(optim_state_dict['param_groups'])
    for group in optim_state_dict['param_groups']:
        group['params'] = LocalNonpersistentObject(group['params'])
    optim_state_dict['state'] = sharded_state
