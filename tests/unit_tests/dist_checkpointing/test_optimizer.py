# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from functools import partial
from time import sleep
from unittest import mock

import numpy as np
import pytest
import torch
from torch.optim import Adam

from megatron.core import parallel_state, DistributedDataParallel as DDP
from megatron.core.dist_checkpointing import ShardedTensor, save, load
from megatron.core.dist_checkpointing.dict_utils import nested_values, diff
from megatron.core.dist_checkpointing.optimizer import \
    get_param_id_to_sharded_param_map, optim_state_to_sharding_state
from megatron.core.dist_checkpointing.utils import extract_sharded_tensors
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import DistributedOptimizer, OptimizerConfig, \
    get_megatron_optimizer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_model_config
from megatron.training import get_model
from pretrain_gpt import model_provider

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


def initialize_gpt_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs=dict(num_layers=8, hidden_size=16, num_attention_heads=8, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    # pre_process = parallel_state.is_pipeline_first_stage()
    # post_process = parallel_state.is_pipeline_last_stage()
    model = GPTModel(config=transformer_config, transformer_layer_spec=get_gpt_layer_local_spec(), vocab_size=128, max_sequence_length=4,
                     pre_process=pre_process, post_process=post_process)

    model.bfloat16()
    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


def init_mock_args(args):
    args.data_parallel_random_init = False
    args.virtual_pipeline_model_parallel_size = None
    args.bf16 = True
    args.accumulate_allreduce_grads_in_fp32 = False
    args.overlap_grad_reduce = False
    args.use_distributed_optimizer = True
    return args


def setup_model_and_optimizer(seed):
    with mock.patch('megatron.training.get_args', data_parallel_random_init=False) as mock_args:
        init_mock_args(mock_args.return_value)
        model = get_model(partial(initialize_gpt_model, seed=seed))

    config = OptimizerConfig(bf16=True, params_dtype=torch.bfloat16, use_distributed_optimizer=True)
    optimizer = get_megatron_optimizer(config, model)

    torch.manual_seed(seed + 1)
    model_parallel_cuda_manual_seed(seed + 1)

    for group in optimizer.optimizer.param_groups:
        for p in group['params']:
            if len(optimizer.optimizer.state[p]) == 0:
                optimizer.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                optimizer.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)

    optimizer.reload_model_params()

    return model, optimizer


class TestDistributedOptimizer:
    @pytest.mark.parametrize("tp_pp,src_dp,dest_dp", [
        ((4, 1), 2, 2),
        # ((1, 1), 8, 1),  # TODO: changing DP doesn't work for now
        # ((1, 1), 1, 8),
        # ((2, 1), 2, 1),
        # ((2, 1), 2, 2),
    ])
    def test_full_dp_sharding(self, tmp_path_dist_ckpt, tp_pp, src_dp, dest_dp):
        src_world_size = tp_pp[0] * tp_pp[1] * src_dp
        dest_world_size = tp_pp[0] * tp_pp[1] * dest_dp
        assert src_world_size <= Utils.world_size, (tp_pp, src_dp)
        assert dest_world_size <= Utils.world_size, (tp_pp, dest_dp)

        with TempNamedDir(tmp_path_dist_ckpt / 'test_dp_sharding', sync=False) as ckpt_dir:
            try:
                Utils.set_world_size(src_world_size)
                if Utils.rank >= 0:
                    # Save checkpoint A
                    Utils.initialize_model_parallel(*tp_pp)
                    model, optimizer_A = setup_model_and_optimizer(seed=2)
                    save(optimizer_A.sharded_state_dict(model[0].sharded_state_dict()), ckpt_dir)
                    optim_param_state_A = optimizer_A.get_parameter_state()
                    Utils.destroy_model_parallel()
                else:
                    # this prevents NCCL errors when changing DP. TODO: fix it properly
                    sleep(20)

                # Load checkpoint A with different TP/PP and save as checkpoint B
                Utils.set_world_size(dest_world_size)
                if Utils.rank == 0:
                    print('_____________________')
                if Utils.rank >= 0:
                    Utils.initialize_model_parallel(*tp_pp)

                    model, optimizer_B = setup_model_and_optimizer(seed=3)
                    optim_param_state_B = optimizer_B.get_parameter_state()
                    diffs = diff(optim_param_state_A, optim_param_state_B)
                    # Expect a mismatch in values - diffs[2] nonempty
                    if parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0:
                        assert not diffs[0] and not diffs[1] and diffs[2], diffs

                    optim_state_dict = load(optimizer_B.sharded_state_dict(model[0].sharded_state_dict()), ckpt_dir)
                    optimizer_B.load_state_dict(optim_state_dict)
                    optim_param_state_B = optimizer_B.get_parameter_state()

                    # Test both param state dicts are equal
                    diffs = diff(optim_param_state_A, optim_param_state_B)
                    assert not any(map(bool, diffs)), diffs

                    Utils.destroy_model_parallel()
                else:
                    # this prevents NCCL errors when changing DP. TODO: fix it properly
                    sleep(20)
            finally:
                Utils.set_world_size()
