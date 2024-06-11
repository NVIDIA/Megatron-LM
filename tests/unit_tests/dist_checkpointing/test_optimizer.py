# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from copy import deepcopy
from functools import partial
from time import sleep
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch.optim import Adam

from megatron.core import parallel_state, DistributedDataParallel as DDP
from megatron.core.dist_checkpointing import ShardedTensor, save, load, \
    load_plain_tensors
from megatron.core.dist_checkpointing.dict_utils import nested_values, diff
from megatron.core.dist_checkpointing.optimizer import \
    get_param_id_to_sharded_param_map, optim_state_to_sharding_state
from megatron.core.dist_checkpointing.serialization import \
    get_default_save_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import \
    FullyParallelSaveStrategyWrapper
from megatron.core.dist_checkpointing.utils import extract_sharded_tensors
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import DistributedOptimizer, OptimizerConfig, \
    get_megatron_optimizer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_model_config
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
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


def init_basic_mock_args(args, bf16=True):
    args.data_parallel_random_init = False
    args.virtual_pipeline_model_parallel_size = None
    args.fp16 = False
    args.bf16 = bf16
    args.accumulate_allreduce_grads_in_fp32 = False
    args.overlap_grad_reduce = False
    args.use_distributed_optimizer = True
    args.ddp_bucket_size = None
    args.check_for_nan_in_loss_and_grad = False
    args.ddp_average_in_collective = False
    return args


def init_checkpointing_mock_args(args, ckpt_dir, fully_parallel=False):
    args.save = ckpt_dir
    args.load = ckpt_dir
    args.pretrained_checkpoint = None
    args.ckpt_fully_parallel_save = fully_parallel
    args.ckpt_fully_parallel_load = fully_parallel
    args.async_save = False
    args.use_dist_ckpt = True
    args.dist_ckpt_format = 'torch_dist'
    args.no_save_optim = False
    args.no_save_rng = False
    args.ckpt_assume_constant_structure = False
    args.log_progress = False
    args.auto_detect_ckpt_format = False
    args.exit_on_missing_checkpoint = False
    args.finetune = False
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.retro_add_retriever = False
    args.no_load_optim = False
    args.no_load_rng = False


def load_checkpoint_no_arg_checks(*args, **kwargs):
    with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
        with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
            return load_checkpoint(*args, **kwargs)


def setup_model_and_optimizer(seed, bf16=True):
    mock_args = SimpleNamespace()
    with mock.patch('megatron.training.training.get_args', new=lambda: mock_args):
        init_basic_mock_args(mock_args, bf16=bf16)
        model = get_model(partial(initialize_gpt_model, seed=seed))

    config = OptimizerConfig(bf16=bf16, params_dtype=torch.bfloat16 if bf16 else torch.float, use_distributed_optimizer=bf16)
    optimizer = get_megatron_optimizer(config, model)

    torch.manual_seed(seed + 1)
    model_parallel_cuda_manual_seed(seed + 1)

    for group in optimizer.optimizer.param_groups:
        for p in group['params']:
            if len(optimizer.optimizer.state[p]) == 0:
                optimizer.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                optimizer.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)

    optimizer.reload_model_params()

    return unwrap_model(model), optimizer


class TestDistributedOptimizer:
    @pytest.mark.parametrize("use_fpsl", [False, True])
    @pytest.mark.parametrize("tp_pp,src_dp,dest_dp", [
        ((4, 1), 2, 2),
        # ((1, 1), 8, 1),  # TODO: changing DP doesn't work for now
        # ((1, 1), 1, 8),
        # ((2, 1), 2, 1),
        # ((2, 1), 2, 2),
    ])
    def test_full_dp_sharding(self, tmp_path_dist_ckpt, tp_pp, src_dp, dest_dp, use_fpsl):
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

                    save_strategy = get_default_save_sharded_strategy()
                    if use_fpsl:
                        save_strategy = FullyParallelSaveStrategyWrapper(
                            save_strategy,
                            parallel_state.get_data_parallel_group(with_context_parallel=True),
                            True
                        )
                    save(optimizer_A.sharded_state_dict(model[0].sharded_state_dict()), ckpt_dir, save_strategy)
                    optim_param_state_A = optimizer_A.get_parameter_state_dp_zero()
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
                    optim_param_state_B = optimizer_B.get_parameter_state_dp_zero()
                    diffs = diff(optim_param_state_A, optim_param_state_B)
                    # Expect a mismatch in values - diffs[2] nonempty
                    if parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0:
                        assert not diffs[0] and not diffs[1] and diffs[2], diffs

                    optim_state_dict = load(optimizer_B.sharded_state_dict(model[0].sharded_state_dict()), ckpt_dir)
                    optimizer_B.load_state_dict(optim_state_dict)
                    optim_param_state_B = optimizer_B.get_parameter_state_dp_zero()

                    # Test both param state dicts are equal
                    diffs = diff(optim_param_state_A, optim_param_state_B)
                    assert not any(map(bool, diffs)), diffs

                    Utils.destroy_model_parallel()
                else:
                    # this prevents NCCL errors when changing DP. TODO: fix it properly
                    sleep(20)
            finally:
                Utils.set_world_size()

    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp',),
        [
            ((2, 2), (2, 4)),
            ((1, 8), (4, 1)),
            ((2, 4), (4, 2)),
        ]
    )
    def test_finetune_doesnt_load_optimizer(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp,):
        with TempNamedDir(tmp_path_dist_ckpt / 'test_finetune_doesnt_load_optimizer') as ckpt_dir:
            mock_args = SimpleNamespace()
            with mock.patch('megatron.training.checkpointing.get_args', new=lambda: mock_args):
                init_basic_mock_args(mock_args)
                init_checkpointing_mock_args(mock_args, ckpt_dir, False)

                Utils.initialize_model_parallel(*src_tp_pp)
                model, optimizer = setup_model_and_optimizer(seed=2)

                # We need to save the TPxPP of the source model
                mock_args.tensor_model_parallel_size = src_tp_pp[0]
                mock_args.pipeline_model_parallel_size = src_tp_pp[1]
                save_checkpoint(10, model, optimizer, None, 0)
                Utils.destroy_model_parallel()

                Utils.initialize_model_parallel(*dest_tp_pp)
                model, optimizer = setup_model_and_optimizer(seed=3)
                model_unloaded_state_dict = deepcopy(model[0].state_dict())
                optim_unloaded_state_dict = deepcopy(optimizer.state_dict())

                # Load with different TPxPP should raise DistributeOptimizer error
                with pytest.raises(RuntimeError) as exc_info:
                    load_checkpoint_no_arg_checks(model, optimizer, None)
                assert "(TP, PP) mismatch" in str(exc_info.value)

                ## Check that the state didn't change
                assert not any(diff(model[0].state_dict(), model_unloaded_state_dict))
                assert not any(diff(optimizer.state_dict(), optim_unloaded_state_dict))

                # Now test the same with a `finetune` flag
                mock_args.finetune = True
                load_checkpoint_no_arg_checks(model, optimizer, None)

                ## Model weights should be different, but optimizer state is unchanged
                diffs = diff(model[0].state_dict(), model_unloaded_state_dict)
                # diffs[0] and diffs[1] is structural diff, diffs[2] is values diff - we expect only values diff
                assert not diffs[0] and not diffs[1] and diffs[2]
                assert not any(diff(optimizer.state_dict(), optim_unloaded_state_dict))

                # ... or `no_load_optim` flag
                model, optimizer = setup_model_and_optimizer(seed=3)
                mock_args.finetune = False
                mock_args.no_load_optim = True
                mock_args.no_load_rng = True
                load_checkpoint_no_arg_checks(model, optimizer, None)

                ## Model weights should be different, but optimizer state is unchanged
                diffs = (diff(model[0].state_dict(), model_unloaded_state_dict))
                # diffs[0] and diffs[1] is structural diff, diffs[2] is values diff - we expect only values diff
                assert not diffs[0] and not diffs[1] and diffs[2]
                assert not any(diff(optimizer.state_dict(), optim_unloaded_state_dict))

                Utils.destroy_model_parallel()


class TestFP32Optimizer:
    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp'),
        [
            ((2, 4), (2, 4)),
            ((2, 4), (4, 2)),
            ((8, 1), (1, 2)),
        ]
    )
    def test_fp32_optimizer_resharding(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp):
        with TempNamedDir(tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_A', sync=False) as ckpt_dir_A:
            with TempNamedDir(tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_B', sync=False) as ckpt_dir_B:
                Utils.initialize_model_parallel(*src_tp_pp)
                model_A, optimizer_A = setup_model_and_optimizer(seed=2, bf16=False)

                save(optimizer_A.sharded_state_dict(model_A[0].sharded_state_dict()), ckpt_dir_A)
                Utils.destroy_model_parallel()

                # Load checkpoint A with different TP/PP and save as checkpoint B
                Utils.initialize_model_parallel(*dest_tp_pp)
                model_B, optimizer_B = setup_model_and_optimizer(seed=3, bf16=False)
                load_sharded_state_dict = optimizer_B.sharded_state_dict(model_B[0].sharded_state_dict())
                state_dict = load(load_sharded_state_dict, ckpt_dir_A)

                optimizer_B.load_state_dict(state_dict)
                save(optimizer_B.sharded_state_dict(model_B[0].sharded_state_dict()), ckpt_dir_B)
                Utils.destroy_model_parallel()

                # Test both checkpoints are equal
                Utils.initialize_model_parallel(1, 1)
                plain_state_dict_A = load_plain_tensors(ckpt_dir_A)
                plain_state_dict_B = load_plain_tensors(ckpt_dir_B)
                diffs = diff(plain_state_dict_A, plain_state_dict_B)
                assert not any(map(bool, diffs)), diffs
                Utils.destroy_model_parallel()
