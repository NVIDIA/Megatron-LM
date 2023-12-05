# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

from typing import Tuple

from .dict_utils import dict_list_map_inplace, extract_matching_values
from .mapping import (
    LocalNonpersitentObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
)


def extract_sharded_tensors(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, ShardedTensor))


def extract_sharded_tensors_and_factories(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    return extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, (ShardedTensor, ShardedTensorFactory))
    )


def extract_sharded_tensors_or_nonpersistent(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    return extract_matching_values(
        sharded_state_dict,
        lambda v: isinstance(v, (ShardedTensor, LocalNonpersitentObject, ShardedTensorFactory)),
    )


def add_prefix_for_sharding(sharded_state_dict: ShardedStateDict, prefix: str):
    def add_prefix(t):
        if isinstance(t, ShardedTensor):
            t.key = f'{prefix}.{t.key}'
        return t

    dict_list_map_inplace(add_prefix, sharded_state_dict)
