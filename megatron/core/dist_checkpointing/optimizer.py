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
    optim_params = list(optim_params_iter)
    id_to_sharded_param_map = {}
    param_to_id_map = get_optim_param_to_id_map(optim_params)
    # If using PyTorch FSDP2 the values in model_sharded_state_dict would
    # have been converted to local tensors during initialization.
    # See the make_(tp)_sharded_tensor_for_checkpoint functions.
    for ten in nested_values(model_sharded_state_dict):
        if id(ten.data) in param_to_id_map:
            id_to_sharded_param_map[param_to_id_map[id(ten.data)]] = ten
        else:
            logger.debug('%s is not tracked by the optimizer', ten)

    _backfill_grouped_param_factories(
        id_to_sharded_param_map, optim_params, model_sharded_state_dict
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


def _backfill_grouped_param_factories(
    id_to_sharded_param_map: dict,
    optim_params: list[torch.nn.Parameter],
    model_sharded_state_dict: ShardedStateDict,
) -> None:
    """Map one grouped optimizer parameter to its per-expert model shards.

    Transformer Engine's ``single_grouped_weight`` exposes one optimizer parameter while the
    model checkpoint contains one view per expert. The factory preserves that checkpoint layout
    when optimizer states are saved and reconstructs the grouped tensor when they are loaded.
    """
    sharded_entries = list(nested_values(model_sharded_state_dict))

    for param_id, param in enumerate(optim_params):
        if param_id in id_to_sharded_param_map:
            continue
        rowwise_data = getattr(param, "rowwise_data", None)
        num_members = getattr(param, "num_tensors", None)
        if not isinstance(rowwise_data, torch.Tensor) or not isinstance(num_members, int):
            continue
        if num_members <= 0 or param.ndim < 1 or int(param.shape[0]) != num_members:
            continue
        if not rowwise_data.is_contiguous() or rowwise_data.numel() != param.numel():
            continue

        start = rowwise_data.data_ptr()
        end = start + rowwise_data.numel() * rowwise_data.element_size()
        candidates = []
        for entry in sharded_entries:
            data = getattr(entry, "data", None)
            if not isinstance(data, torch.Tensor):
                continue
            data_start = data.data_ptr()
            data_end = data_start + data.numel() * data.element_size()
            if (
                data.dtype == rowwise_data.dtype
                and data.device == rowwise_data.device
                and data.is_contiguous()
                and start <= data_start < data_end <= end
            ):
                candidates.append((data_start - start, entry))

        candidates.sort(key=lambda item: item[0])
        member_numel = rowwise_data.numel() // num_members
        member_nbytes = member_numel * rowwise_data.element_size()
        if len(candidates) != num_members or [offset for offset, _ in candidates] != [
            idx * member_nbytes for idx in range(num_members)
        ]:
            continue
        templates = [entry.without_data() for _, entry in candidates]
        if len({template.key for template in templates}) != 1:
            continue

        def build_grouped_shards(key, tensor, replica_id, flattened_range, templates=templates):
            if flattened_range is not None:
                raise ValueError("Grouped optimizer factories do not support flattened ranges")
            if tensor.ndim < 1 or int(tensor.shape[0]) != len(templates):
                raise ValueError(
                    f"Grouped optimizer tensor shape {tuple(tensor.shape)} does not match "
                    f"{len(templates)} checkpoint members"
                )
            built = []
            for member, template in zip(tensor.unbind(dim=0), templates):
                if isinstance(template, ShardedTensorFactory):
                    built.append(template.build_fn(key, member, replica_id, None))
                else:
                    built.append(
                        replace(
                            template,
                            key=key,
                            data=member,
                            dtype=member.dtype,
                            replica_id=replica_id,
                            flattened_range=None,
                        )
                    )
            return built

        def merge_grouped_shards(loaded, templates=templates):
            members = []
            for value, template in zip(loaded, templates):
                if isinstance(template, ShardedTensorFactory):
                    value = template.merge_fn(value)
                members.append(value)
            return torch.stack(members, dim=0)

        first = templates[0]
        id_to_sharded_param_map[param_id] = ShardedTensorFactory(
            key=first.key,
            data=param,
            build_fn=build_grouped_shards,
            merge_fn=merge_grouped_shards,
            replica_id=first.replica_id,
        )


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
