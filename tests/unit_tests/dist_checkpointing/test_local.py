# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import filecmp
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Tuple, Union
from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.mapping import ShardedBase, ShardedTensorFactory
from megatron.core.dist_checkpointing.state_dict_transformation import (
    prepare_state_dict_for_save,
    recreate_state_dict_after_load,
)
from megatron.core.dist_checkpointing.utils import extract_nonpersistent
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.checkpointing import generate_state_dict, load_checkpoint, save_checkpoint
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils


def find_matching_values(
    x: Union[dict, list], predicate: Callable[[Any], bool]
) -> Tuple[Union[dict, list], Union[dict, list]]:
    """Return matching values in a single list

    Args:
        x (Union[dict, list]) : state dict to process. Top-level argument must be a dict or list
        predicate (object -> bool): determines matching values
    """

    matching_vals = []
    if isinstance(x, dict):
        values = x.values()
    elif isinstance(x, list):
        values = x
    else:
        raise ValueError(f'Unexpected top-level object type: {type(x)}')
    for v in values:
        if isinstance(v, (list, dict)):
            matching_vals += find_matching_values(v, predicate)
        elif predicate(v):
            matching_vals.append(v)
    return matching_vals


class TestLocalCheckpointing:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(('tp,pp'), [(2, 4)])
    def test_sharded_tensors(self, tp, pp):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)
        opt_param_scheduler = None
        rng_state = None
        use_dist_ckpt = True
        iteration = None
        optim_sd_kwargs = dict(sharding_type='fully_sharded_model_space')
        mock_args = SimpleNamespace()
        mock_args.no_save_optim = False
        mock_args.no_save_rng = True
        # Test save_local
        state_dict = generate_state_dict(
            mock_args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            use_dist_ckpt,
            iteration,
            optim_sd_kwargs=optim_sd_kwargs,
        )
        sharded_tensor_factories = find_matching_values(
            state_dict, lambda x: isinstance(x, ShardedTensorFactory)
        )
        sharded_tensors = find_matching_values(state_dict, lambda x: isinstance(x, ShardedTensor))
        for ten in sharded_tensors:
            assert ten.data != None
        saved_state_dict = prepare_state_dict_for_save(state_dict)
        saved_sharded_tensors = find_matching_values(
            saved_state_dict, lambda x: isinstance(x, ShardedTensor)
        )
        for ten in saved_sharded_tensors:
            assert ten.data == None
        assert (
            len(saved_sharded_tensors)
            == len(sharded_tensors) + 2 * len(sharded_tensor_factories)
            == len(saved_state_dict['raw_tensors'])
        )
        common_sharded_tensors = find_matching_values(
            saved_state_dict["common"], lambda x: isinstance(x, ShardedTensor)
        )
        assert common_sharded_tensors == []
        # Test load_local
        state_dict = generate_state_dict(
            mock_args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            True,
            iteration,
            optim_sd_kwargs=optim_sd_kwargs,
        )
        nonpersistent_state_dict, _ = extract_nonpersistent(state_dict)
        # For a given use case
        assert not nonpersistent_state_dict
        loaded_state_dict = recreate_state_dict_after_load(state_dict, saved_state_dict)
        only_left, only_right, mismatch = diff(loaded_state_dict, state_dict)
        assert not only_left
        assert not only_right
        for i in mismatch:
            # ShardedObjects and ShardedTensors should be replaced
            assert issubclass(i[-1], ShardedBase)

    @pytest.mark.parametrize(('tp,pp'), [(2, 4), (1, 1)])
    @pytest.mark.parametrize(('use_ramdisk'), [True, False])
    @pytest.mark.parametrize(('async_save'), [True, False])
    @pytest.mark.parametrize(('algo'), ['atomic', 'fully_parallel'])
    @pytest.mark.skip(reason="BasicLocalCheckpointManager is not yet integrated")
    def test_basic_save_load_scenarios(
        self, tmp_path_dist_ckpt, tp, pp, use_ramdisk, async_save, algo
    ):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)
        opt_param_scheduler = None

        mock_args = SimpleNamespace()
        if use_ramdisk:
            tmp_path_dist_ckpt = Path("/dev/shm")
        with TempNamedDir(tmp_path_dist_ckpt / "test_local") as local_ckpt_dir, mock.patch(
            'megatron.training.checkpointing.get_args', new=lambda: mock_args
        ), mock.patch('megatron.training.async_utils.get_args', new=lambda: mock_args), mock.patch(
            "megatron.training.checkpointing.update_num_microbatches"
        ):
            local_ckpt_dir = local_ckpt_dir / "subdir"  # Test handling of non-existent directories
            init_basic_mock_args(mock_args, tp, pp)
            init_checkpointing_mock_args(mock_args, None)
            mock_args.non_persistent_ckpt_type = 'local'
            mock_args.non_persistent_local_ckpt_algo = algo
            mock_args.async_save = async_save
            checkpointing_context = {
                'local_checkpoint_manager': BasicLocalCheckpointManager(local_ckpt_dir)
            }

            save_checkpoint(
                1,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context=checkpointing_context,
                non_persistent_ckpt=True,
            )
            if async_save:
                maybe_finalize_async_save(True)
            iteration, _ = load_checkpoint(
                model, optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context
            )
            assert iteration == 1
            ckpt_path = checkpointing_context['local_checkpoint_manager'].local_ckpt_path
            backup_path = ckpt_path.with_name('backup_' + ckpt_path.name)
            checkpointing_context['local_checkpoint_manager'].latest_iteration = -1
            iteration, _ = load_checkpoint(
                model, optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context
            )
            assert iteration == 1
            shutil.move(ckpt_path, backup_path)
            checkpointing_context['local_checkpoint_manager'].latest_iteration = -1
            torch.distributed.barrier()
            iteration, _ = load_checkpoint(
                model, optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context
            )
            assert iteration == 0
            save_checkpoint(
                1,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context=checkpointing_context,
                non_persistent_ckpt=True,
            )
            if async_save:
                maybe_finalize_async_save(True)
            assert filecmp.cmp(ckpt_path, backup_path, shallow=False), [ckpt_path, backup_path]
            save_checkpoint(
                2,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context=checkpointing_context,
                non_persistent_ckpt=True,
            )
            if async_save:
                maybe_finalize_async_save(True)
            assert not ckpt_path.exists()
            ckpt_path = checkpointing_context['local_checkpoint_manager'].local_ckpt_path
            assert ckpt_path.exists()

        Utils.destroy_model_parallel()
