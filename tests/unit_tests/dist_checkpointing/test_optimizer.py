# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from copy import deepcopy
from functools import partial
from time import sleep
import traceback
from unittest import mock

import pytest
import torch
import torch.distributed
from torch.optim import Adam

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import (
    ShardedTensor,
    load,
    load_plain_tensors,
    load_tensors_metadata,
    save,
)
from megatron.core.dist_checkpointing.dict_utils import diff, nested_values
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    optim_state_to_sharding_state,
)
from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.utils import extract_sharded_tensors
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    initialize_gpt_model,
    setup_model_and_optimizer,
    setup_moe_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils
from megatron.core.device_utils import get_current_device, get_xla_model

try:
    import transformer_engine # pylint: disable=unused-import
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

xm = get_xla_model()

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 16, 3)
        self.proj = torch.nn.Linear(8, 5)
        self.config = TransformerConfig(hidden_size=8, num_attention_heads=1, num_layers=1)

    def sharded_state_dict(self):
        sharded_state_dict = self.state_dict(keep_vars=True)
        # conv
        sharded_state_dict['conv.weight'] = ShardedTensor.from_rank_offsets(
            'conv.weight',
            sharded_state_dict['conv.weight'],
            (
                1,
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_tensor_model_parallel_world_size(),
            ),
        )
        # bias is non-sharded
        sharded_state_dict['conv.bias'] = ShardedTensor.from_rank_offsets(
            'conv.bias', sharded_state_dict['conv.bias']
        )

        # proj
        sharded_state_dict['proj.weight'] = ShardedTensor.from_rank_offsets(
            'proj.weight', sharded_state_dict['proj.weight'], (0, Utils.rank, Utils.world_size)
        )
        sharded_state_dict['proj.bias'] = ShardedTensor.from_rank_offsets(
            'proj.bias', sharded_state_dict['proj.bias'], (0, Utils.rank, Utils.world_size)
        )
        return sharded_state_dict


class SwigluFactoryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(
            5, 64 // parallel_state.get_tensor_model_parallel_world_size(), bias=False
        )
        self.config = TransformerConfig(hidden_size=8, num_attention_heads=1, num_layers=1)

    def sharded_state_dict(self):
        sharded_state_dict = self.state_dict(keep_vars=True)
        sharded_state_dict['linear.weight'] = ShardedTensor.from_rank_offsets(
            'linear.weight',
            sharded_state_dict['linear.weight'],
            (
                (
                    0,
                    parallel_state.get_tensor_model_parallel_rank(),
                    parallel_state.get_tensor_model_parallel_world_size(),
                )
            ),
            replica_id=(
                (
                    parallel_state.get_pipeline_model_parallel_rank(),
                    0,
                    parallel_state.get_data_parallel_rank(with_context_parallel=True),
                )
            ),
        )
        sharded_state_dict['linear.weight'] = apply_swiglu_sharded_factory(
            sharded_state_dict['linear.weight'], ()
        )
        return sharded_state_dict


class Model1dFlattenTensor(torch.nn.Module):
    """This model is used to test whether a 1d flatten tensor can be correctly
    transformed into torch dist-ckpt form
    """

    def __init__(self):
        super().__init__()
        self.config = TransformerConfig(hidden_size=128, num_attention_heads=1, num_layers=1)
        self.weight_1d = torch.nn.Parameter(torch.randn(self.config.hidden_size))

    def sharded_state_dict(self):
        sharded_state_dict = self.state_dict(keep_vars=True)
        sharded_state_dict['weight_1d'] = ShardedTensor.from_rank_offsets(
            'weight_1d',
            sharded_state_dict['weight_1d'],
            (
                (
                    0,
                    parallel_state.get_tensor_model_parallel_rank(),
                    parallel_state.get_tensor_model_parallel_world_size(),
                )
            ),
            replica_id=(
                (
                    parallel_state.get_pipeline_model_parallel_rank(),
                    0,
                    parallel_state.get_data_parallel_rank(with_context_parallel=True),
                )
            ),
        )
        return sharded_state_dict


class TestOptimizer:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_optimizer_params(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1, 1)
        model = Model()
        model.to(device=get_current_device())
        # Force optimizer state initialization
        for p in model.parameters():
            p.grad = torch.ones_like(p.data, device=get_current_device())
        optim = Adam(model.parameters())
        optim.step()

        model_state_dict = model.sharded_state_dict()
        param_map = get_param_id_to_sharded_param_map(
            model_state_dict, optim.param_groups[0]['params']
        )
        optim_state_dict = optim.state_dict()
        optim_state_to_sharding_state(optim_state_dict, param_map, exclude_keys=('step',))

        optim_sharded_tensors = nested_values(extract_sharded_tensors(optim_state_dict)[0])
        optim_sharded_keys = {sh_ten.key for sh_ten in optim_sharded_tensors}
        assert len(optim_sharded_keys) == 2 * len(model_state_dict)
        assert optim_sharded_keys == set(
            [
                f'optimizer.state.{state_key}.{layer_name}'
                for state_key in ['exp_avg', 'exp_avg_sq']
                for layer_name in model_state_dict
            ]
        )


def initialize_small_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_device_manual_seed(seed)

    return SwigluFactoryModel()


def initialize_1d_flatten_tensor_model(
    pre_process=True, post_process=True, seed=0, **config_kwargs
):
    # This model is used to test whether a 1d flatten tensor can be correctly
    # transformed into torch dist-ckpt form
    torch.manual_seed(seed)
    model_parallel_device_manual_seed(seed)

    return Model1dFlattenTensor()


def load_checkpoint_no_arg_checks(*args, **kwargs):
    with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
        with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
            return load_checkpoint(*args, **kwargs)


@pytest.mark.skipif(xm is not None, reason="Distributed Optimizer not supported on XLA")
class TestDistributedOptimizer:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "initialize_fn",
        [initialize_small_model, initialize_gpt_model, initialize_1d_flatten_tensor_model],
    )
    @pytest.mark.parametrize("use_fpsl", [False, True])
    # TODO: changing DP doesn't work in unit tests because of NCCL crashes
    @pytest.mark.parametrize(
        "tp_pp,src_dp,dest_dp",
        [
            ((4, 1), 2, 2),
            # ((1, 1), 8, 1),
            # ((1, 1), 1, 8),
            # ((2, 1), 2, 1),
            # ((2, 1), 2, 2),
        ],
    )
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    def test_dp_sharding(self, tmp_path_dist_ckpt, tp_pp, src_dp, dest_dp, use_fpsl, initialize_fn):
        src_world_size = tp_pp[0] * tp_pp[1] * src_dp
        dest_world_size = tp_pp[0] * tp_pp[1] * dest_dp
        assert src_world_size <= Utils.world_size, (tp_pp, src_dp)
        assert dest_world_size <= Utils.world_size, (tp_pp, dest_dp)

        sharding_type = 'fully_sharded_model_space' if use_fpsl else 'dp_zero_gather_scatter'

        Utils.initialize_model_parallel(*tp_pp)

        # sync=True to make sure other ranks wait for rank 0 to finish creating directory.
        with TempNamedDir(tmp_path_dist_ckpt / 'test_dp_sharding', sync=True,
                          process_group=parallel_state.get_default_process_group()) as ckpt_dir:
            try:
                Utils.set_world_size(src_world_size)
                if Utils.rank >= 0:
                    # Save checkpoint A
                    model, optimizer_A = setup_model_and_optimizer(
                        seed=2, tp=tp_pp[0], pp=tp_pp[1], initialize_fn=initialize_fn, dist_opt=True
                    )
                    save_strategy = get_default_save_sharded_strategy()
                    if use_fpsl:
                        save_strategy = FullyParallelSaveStrategyWrapper(
                            save_strategy,
                            parallel_state.get_data_parallel_group(with_context_parallel=True) if xm is None else \
                                parallel_state.get_data_parallel_group_gloo(with_context_parallel=True),
                            parallel_state.get_default_process_group(),
                            True,
                        )
                    sharded_state_dict = optimizer_A.sharded_state_dict(
                        model[0].sharded_state_dict(), sharding_type=sharding_type
                    )
                    save(
                        sharded_state_dict,
                        ckpt_dir,
                        save_strategy,
                        process_group=parallel_state.get_default_process_group()
                    )
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

                    model, optimizer_B = setup_model_and_optimizer(
                        seed=3, tp=tp_pp[0], pp=tp_pp[1], initialize_fn=initialize_fn, dist_opt=True
                    )
                    optim_param_state_B = optimizer_B.get_parameter_state_dp_zero()
                    diffs = diff(optim_param_state_A, optim_param_state_B)
                    # Expect a mismatch in values - diffs[2] nonempty
                    if parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0:
                        assert not diffs[0] and not diffs[1] and diffs[2], diffs

                    sharded_state_dict = optimizer_B.sharded_state_dict(
                        model[0].sharded_state_dict(), is_loading=True, sharding_type=sharding_type
                    )
                    optim_state_dict = load(sharded_state_dict, ckpt_dir,
                                            process_group=parallel_state.get_default_process_group())
                    optimizer_B.load_state_dict(optim_state_dict)
                    optim_param_state_B = optimizer_B.get_parameter_state_dp_zero()
                    # Test both param state dicts are equal
                    diffs = diff(optim_param_state_A, optim_param_state_B)
                    assert not any(map(bool, diffs)), diffs
                else:
                    # this prevents NCCL errors when changing DP. TODO: fix it properly
                    sleep(20)
            finally:
                Utils.set_world_size()

    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp', 'use_glu'),
        [((2, 2), (2, 4), False), ((1, 8), (4, 1), True), ((2, 4), (4, 2), False)],
    )
    def test_finetune_doesnt_load_optimizer(
        self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp, use_glu
    ):
        # sync=True to make sure other ranks wait for rank 0 to finish creating directory.
        Utils.initialize_model_parallel(*src_tp_pp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_finetune_doesnt_load_optimizer', sync=True,
            process_group=parallel_state.get_default_process_group()
        ) as ckpt_dir:
            mock_args = parse_args(ignore_unknown_args=True)
            with mock.patch('megatron.training.checkpointing.get_args', new=lambda: mock_args):
                init_basic_mock_args(mock_args, tp=src_tp_pp[0], pp=src_tp_pp[1])
                init_checkpointing_mock_args(mock_args, ckpt_dir, False)

                model, optimizer = setup_model_and_optimizer(
                    seed=2,
                    tp=src_tp_pp[0],
                    pp=src_tp_pp[1],
                    initialize_fn=partial(initialize_gpt_model, use_glu=use_glu),
                    dist_opt=True
                )

                save_checkpoint(10, model, optimizer, None, 0)
                Utils.destroy_model_parallel()

                Utils.initialize_model_parallel(*dest_tp_pp)
                mock_args.tensor_model_parallel_size = dest_tp_pp[0]
                mock_args.pipeline_model_parallel_size = dest_tp_pp[1]
                model, optimizer = setup_model_and_optimizer(
                    seed=3,
                    tp=dest_tp_pp[0],
                    pp=dest_tp_pp[1],
                    initialize_fn=partial(initialize_gpt_model, use_glu=use_glu),
                    dist_opt=True
                )
                model_unloaded_state_dict = deepcopy(model[0].state_dict())
                optim_unloaded_state_dict = deepcopy(optimizer.state_dict())
                optim_unloaded_state_dict.pop('fp32_from_fp16_params', None)
                
                # Load with different TPxPP should raise DistributeOptimizer error
                with pytest.raises(RuntimeError) as exc_info:
                    load_checkpoint_no_arg_checks(model, optimizer, None)
                # "(TP, PP) mismatch" check is for backwards compatibility tests
                assert "(TP, PP) mismatch" in str(
                    exc_info.value
                ) or "(TP, PP, encoder TP, encoder PP) mismatch" in str(exc_info.value)

                # Check that the state didn't change
                assert not any(diff(model[0].state_dict(), model_unloaded_state_dict))
                optim_state_dict = optimizer.state_dict()
                optim_state_dict.pop('fp32_from_fp16_params', None)
                assert not any(diff(optim_state_dict, optim_unloaded_state_dict))

                # Now test the same with a `finetune` flag
                mock_args.finetune = True
                load_checkpoint_no_arg_checks(model, optimizer, None)

                # Model weights should be different, but optimizer state is unchanged
                diffs = diff(model[0].state_dict(), model_unloaded_state_dict)
                # diffs[0] and diffs[1] is structural diff, diffs[2] is values diff -
                # we expect only values diff
                assert not diffs[0] and not diffs[1] and diffs[2]
                optim_state_dict = optimizer.state_dict()
                optim_state_dict.pop('fp32_from_fp16_params', None)
                assert not any(diff(optim_state_dict, optim_unloaded_state_dict))

                # ... or `no_load_optim` flag
                model, optimizer = setup_model_and_optimizer(
                    seed=3,
                    tp=dest_tp_pp[0],
                    pp=dest_tp_pp[1],
                    initialize_fn=partial(initialize_gpt_model, use_glu=use_glu),
                    dist_opt=True
                )
                mock_args.finetune = False
                mock_args.no_load_optim = True
                mock_args.no_load_rng = True
                load_checkpoint_no_arg_checks(model, optimizer, None)

                # Model weights should be different, but optimizer state is unchanged
                diffs = diff(model[0].state_dict(), model_unloaded_state_dict)
                # diffs[0] and diffs[1] is structural diff, diffs[2] is values diff -
                # we expect only values diff
                assert not diffs[0] and not diffs[1] and diffs[2]
                optim_state_dict = optimizer.state_dict()
                optim_state_dict.pop('fp32_from_fp16_params', None)
                assert not any(diff(optim_state_dict, optim_unloaded_state_dict))


class TestFP32Optimizer:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp'), [((2, 4), (2, 4)), ((2, 4), (4, 2)), ((8, 1), (1, 2))]
    )
    def test_fp32_optimizer_resharding(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp):
        # sync=True to make sure other ranks wait for rank 0 to finish creating directory.

        def preprocess_fn(optim_common_dict):
            import copy

            preprocessed_optimzier_common_dict = copy.deepcopy(optim_common_dict)
            try:
                list = preprocessed_optimzier_common_dict['optimizer']['param_groups']
                for dict_item in list:
                    del dict_item['wd_mult']
            except KeyError:
                pass
            
            return preprocessed_optimzier_common_dict

        Utils.initialize_model_parallel(*src_tp_pp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_A', sync=True,
            process_group=parallel_state.get_default_process_group()
        ) as ckpt_dir_A:
            with TempNamedDir(
                tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_B', sync=True,
                process_group=parallel_state.get_default_process_group()
            ) as ckpt_dir_B:

                dist_opt = xm is None
                model_A, optimizer_A = setup_model_and_optimizer(
                    seed=2,
                    tp=src_tp_pp[0],
                    pp=src_tp_pp[1],
                    initialize_fn=initialize_small_model,
                    bf16=False,
                    dist_opt=dist_opt
                )

                save(
                    optimizer_A.sharded_state_dict(model_A[0].sharded_state_dict()),
                    ckpt_dir_A,
                    process_group=parallel_state.get_default_process_group(),
                    preprocess_common_before_consistancy_check=preprocess_fn,
                )
                Utils.destroy_model_parallel()

                # Load checkpoint A with different TP/PP and save as checkpoint B
                Utils.initialize_model_parallel(*dest_tp_pp)
                model_B, optimizer_B = setup_model_and_optimizer(
                    seed=3,
                    tp=dest_tp_pp[0],
                    pp=dest_tp_pp[1],
                    initialize_fn=initialize_small_model,
                    bf16=False,
                    dist_opt=dist_opt
                )
                load_sharded_state_dict = optimizer_B.sharded_state_dict(
                    model_B[0].sharded_state_dict()
                )
                state_dict = load(load_sharded_state_dict, ckpt_dir_A,
                                  process_group=parallel_state.get_default_process_group())

                optimizer_B.load_state_dict(state_dict)
                save(optimizer_B.sharded_state_dict(model_B[0].sharded_state_dict()), ckpt_dir_B,
                     process_group=parallel_state.get_default_process_group())
                Utils.destroy_model_parallel()

                # Test both checkpoints are equal
                Utils.initialize_model_parallel(1, 1)
                plain_state_dict_A = load_plain_tensors(ckpt_dir_A, process_group=parallel_state.get_default_process_group())
                plain_state_dict_B = load_plain_tensors(ckpt_dir_B, process_group=parallel_state.get_default_process_group())
                diffs = diff(plain_state_dict_A, plain_state_dict_B)
                assert not any(map(bool, diffs)), diffs

class TestOptimizerResharding:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ('use_dist_opt', 'bf16'),
        (
            (False, True),  # regular BF16
            (True, True),  # DistOpt BF16
            # (False, False), # FP32
        ),
    )
    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp'),
        [((2, 4), (2, 4)), ((2, 4), (2, 2)), ((2, 4), (4, 2)), ((8, 1), (1, 2))],
    )
    def test_optimizer_resharding(
        self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp, use_dist_opt, bf16
    ):
        use_dist_opt = use_dist_opt and xm is None
        Utils.initialize_model_parallel(*src_tp_pp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_A', sync=True,
            process_group=parallel_state.get_default_process_group()
        ) as ckpt_dir_A:
            with TempNamedDir(
                tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_B', sync=True,
                process_group=parallel_state.get_default_process_group()
            ) as ckpt_dir_B:

                model_A, optimizer_A = setup_model_and_optimizer(
                    seed=2, tp=src_tp_pp[0], pp=src_tp_pp[1], bf16=bf16, dist_opt=use_dist_opt
                )

                save(optimizer_A.sharded_state_dict(model_A[0].sharded_state_dict()), ckpt_dir_A,
                     process_group=parallel_state.get_default_process_group())
                Utils.destroy_model_parallel()

                # Load checkpoint A with different TP/PP and save as checkpoint B
                Utils.initialize_model_parallel(*dest_tp_pp)
                model_B, optimizer_B = setup_model_and_optimizer(
                    seed=3, tp=dest_tp_pp[0], pp=dest_tp_pp[1], bf16=bf16, dist_opt=use_dist_opt
                )
                load_sharded_state_dict = optimizer_B.sharded_state_dict(
                    model_B[0].sharded_state_dict()
                )
                state_dict = load(load_sharded_state_dict, ckpt_dir_A,
                                  process_group=parallel_state.get_default_process_group())

                optimizer_B.load_state_dict(state_dict)
                save(optimizer_B.sharded_state_dict(model_B[0].sharded_state_dict()), ckpt_dir_B,
                    process_group=parallel_state.get_default_process_group())
                Utils.destroy_model_parallel()

                # Test both checkpoints are equal
                Utils.initialize_model_parallel(1, 1)
                plain_state_dict_A = load_plain_tensors(ckpt_dir_A, process_group=parallel_state.get_default_process_group())
                plain_state_dict_B = load_plain_tensors(ckpt_dir_B, process_group=parallel_state.get_default_process_group())
                diffs = diff(plain_state_dict_A, plain_state_dict_B)
                assert not any(map(bool, diffs)), diffs

    @pytest.mark.parametrize(('use_dist_opt', 'bf16'), ((True, True),))  # DistOpt BF16
    @pytest.mark.parametrize(('use_te', 'use_grouped_mlp'), ((False, False), (False, True)))
    @pytest.mark.parametrize('use_glu', [False, True])
    @pytest.mark.parametrize(
        ('src_tp_pp_exp', 'dest_tp_pp_exp'),
        [
            ((2, 2, 2), (2, 2, 2)),
            ((4, 1, 2), (1, 2, 2)),
            ((1, 1, 2), (1, 1, 4)),
            ((2, 1, 2), (1, 1, 8)),
        ],
    )
    def test_chained_optimizer_resharding(
        self,
        tmp_path_dist_ckpt,
        src_tp_pp_exp,
        dest_tp_pp_exp,
        use_dist_opt,
        bf16,
        use_te,
        use_grouped_mlp,
        use_glu,
    ):
        use_dist_opt = use_dist_opt and xm is None
        use_te = use_te and HAVE_TE
        src_tp, src_pp, src_exp = src_tp_pp_exp
        dest_tp, dest_pp, dest_exp = dest_tp_pp_exp
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_A', sync=False
        ) as ckpt_dir_A:
            with TempNamedDir(
                tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_B', sync=False
            ) as ckpt_dir_B:
                Utils.initialize_model_parallel(src_tp, src_pp, expert_model_parallel_size=src_exp)
                model_A, optimizer_A = setup_moe_model_and_optimizer(
                    seed=2,
                    tp=src_tp,
                    pp=src_pp,
                    ep=src_exp,
                    bf16=bf16,
                    dist_opt=use_dist_opt,
                    use_te=use_te,
                    use_grouped_mlp=use_grouped_mlp,
                    use_glu=use_glu,
                )

                save(optimizer_A.sharded_state_dict(model_A[0].sharded_state_dict()), ckpt_dir_A, 
                     process_group=parallel_state.get_default_process_group())
                Utils.destroy_model_parallel()

                # Load checkpoint A with different TP/PP and save as checkpoint B
                Utils.initialize_model_parallel(
                    dest_tp, dest_pp, expert_model_parallel_size=dest_exp
                )
                model_B, optimizer_B = setup_moe_model_and_optimizer(
                    seed=3,
                    tp=dest_tp,
                    pp=dest_pp,
                    ep=dest_exp,
                    bf16=bf16,
                    dist_opt=use_dist_opt,
                    use_te=use_te,
                    use_grouped_mlp=use_grouped_mlp,
                    use_glu=use_glu,
                )
                load_sharded_state_dict = optimizer_B.sharded_state_dict(
                    model_B[0].sharded_state_dict()
                )
                state_dict = load(load_sharded_state_dict, ckpt_dir_A, 
                                  process_group=parallel_state.get_default_process_group())

                optimizer_B.load_state_dict(state_dict)
                save(optimizer_B.sharded_state_dict(model_B[0].sharded_state_dict()), ckpt_dir_B, 
                     process_group=parallel_state.get_default_process_group())
                Utils.destroy_model_parallel()

                # Test both checkpoints are equal
                Utils.initialize_model_parallel(1, 1)
                plain_state_dict_A = load_plain_tensors(ckpt_dir_A, process_group=parallel_state.get_default_process_group())
                plain_state_dict_B = load_plain_tensors(ckpt_dir_B, process_group=parallel_state.get_default_process_group())
                diffs = diff(plain_state_dict_A, plain_state_dict_B)
                assert not any(map(bool, diffs)), diffs
                Utils.destroy_model_parallel()
