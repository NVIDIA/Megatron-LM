# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import torch
from torch.optim import Adam

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, save, load
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.optimizer import \
    get_param_id_to_sharded_param_map, optim_state_to_sharding_state
from megatron.core.dist_checkpointing.utils import extract_sharded_tensors

from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 16, 3)
        self.proj = torch.nn.Linear(32, 7)
    def sharded_state_dict(self):
        sharded_state_dict = self.state_dict(keep_vars=True)
        # conv
        sharded_state_dict['conv.weight'] = ShardedTensor.from_rank_offsets(
            'conv.weight', sharded_state_dict['conv.weight'],
            (1, parallel_state.get_tensor_model_parallel_rank(), parallel_state.get_tensor_model_parallel_world_size())
        )
        # bias is non-sharded
        sharded_state_dict['conv.bias'] = ShardedTensor.from_rank_offsets('conv.bias', sharded_state_dict['conv.bias'])

        # proj
        sharded_state_dict['proj.weight'] = ShardedTensor.from_rank_offsets(
            'proj.weight', sharded_state_dict['proj.weight'],
            (0, Utils.rank, Utils.world_size)
        )
        sharded_state_dict['proj.bias'] = ShardedTensor.from_rank_offsets(
            'proj.bias', sharded_state_dict['proj.bias'],
            (0, Utils.rank, Utils.world_size)
        )
        return sharded_state_dict


class TestOptimizer:
    def test_optimizer_params(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1,1)
        model = Model()
        # Force optimizer state initialization
        for p in model.parameters():
            p.grad = torch.ones_like(p.data)
        optim = Adam(model.parameters())
        optim.step()

        model_state_dict = model.sharded_state_dict()
        param_map = get_param_id_to_sharded_param_map(model_state_dict, optim.param_groups[0]['params'])
        optim_state_dict = optim.state_dict()
        optim_state_to_sharding_state(optim_state_dict, param_map, exclude_keys=('step',))

        optim_sharded_tensors = nested_values(extract_sharded_tensors(optim_state_dict)[0])
        optim_sharded_keys = {sh_ten.key for sh_ten in optim_sharded_tensors}
        assert len(optim_sharded_keys) == 2 * len(model_state_dict)
        assert optim_sharded_keys == set([
            f'optimizer.state.{state_key}.{layer_name}'
            for state_key in ['exp_avg', 'exp_avg_sq']
            for layer_name in model_state_dict
        ])
