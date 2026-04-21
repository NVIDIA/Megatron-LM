# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import filecmp
import logging
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Tuple, Union
from unittest import mock

import pytest
import torch

from megatron.training.arguments import parse_args

nvidia_resiliency_ext = pytest.importorskip(
    "nvidia_resiliency_ext",
    reason="nvidia_resiliency_ext is required for local checkpointing tests",
)

from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.base_manager import (
    CheckpointingException,
)
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
    LocalCheckpointManager,
)

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.mapping import ShardedBase, ShardedTensorFactory
from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict
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

from .utils import find_matching_values


# TODO: Use mock local checkpointing?
class TestLocalCheckpointingReplication:

    def test_filename_to_id(self):
        iteration_string = "0000123"
        rank = "4"
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = LocalCheckpointManager(tmpdir)
            filename = ckpt_mgr._filename_from_template(iteration_string, rank)
            assert (123, 4) == ckpt_mgr._filename_to_id(filename)[:2]

    @pytest.mark.parametrize(('tp,pp'), [(2, 4)])
    def test_sharded_tensors(self, tp, pp):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)


class TestLocalCheckpointing:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(('tp,pp'), [(2, 4)])
    @pytest.mark.parametrize(('use_torch_fsdp2'), [True, False])
    def test_sharded_tensors(self, tp, pp, use_torch_fsdp2):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)
        opt_param_scheduler = None
        rng_state = None
        iteration = None
        optim_sd_kwargs = dict(sharding_type='fully_sharded_model_space')
        mock_args = parse_args(ignore_unknown_args=True)
        mock_args.no_save_optim = False
        mock_args.no_save_rng = True
        mock_args.use_torch_fsdp2 = use_torch_fsdp2
        # Test save_local
        state_dict = generate_state_dict(
            mock_args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            iteration=iteration,
            optim_sd_kwargs=optim_sd_kwargs,
        )
        sharded_tensor_factories = find_matching_values(
            state_dict, lambda x: isinstance(x, ShardedTensorFactory)
        )
        sharded_tensors = find_matching_values(state_dict, lambda x: isinstance(x, ShardedTensor))
        for ten in sharded_tensors:
            assert ten.data != None
        saved_state_dict, _ = MCoreTensorAwareStateDict.from_state_dict(state_dict, algo='atomic')
        saved_sharded_tensors = find_matching_values(
            saved_state_dict, lambda x: isinstance(x, ShardedTensor)
        )
        assert (
            len(saved_sharded_tensors)
            == len(sharded_tensors) + 2 * len(sharded_tensor_factories)
            == len(list(saved_state_dict.tensors))
        )
        tensors = saved_state_dict.pop_tensors()
        for ten in saved_sharded_tensors:
            assert ten.data is None
        assert saved_state_dict.is_hollow
        hollow_sharded_tensors = find_matching_values(
            saved_state_dict, lambda x: isinstance(x, torch.Tensor)
        )
        assert hollow_sharded_tensors == []
        saved_state_dict.insert_tensors(tensors)
        common_sharded_tensors = find_matching_values(
            saved_state_dict.common_state_dict, lambda x: isinstance(x, ShardedTensor)
        )
        assert common_sharded_tensors == []
        # Test load_local
        state_dict = generate_state_dict(
            mock_args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            iteration=iteration,
            optim_sd_kwargs=optim_sd_kwargs,
        )
        nonpersistent_state_dict, _ = extract_nonpersistent(state_dict)
        # For a given use case
        assert not nonpersistent_state_dict
        loaded_state_dict = saved_state_dict.to_state_dict(state_dict)
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
    def test_basic_save_load_scenarios(
        self, tmp_path_dist_ckpt, tp, pp, use_ramdisk, async_save, algo
    ):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)
        opt_param_scheduler = None

        mock_args = (
            SimpleNamespace()
        )  # FIXME: fails with additional arguments (e.g.,'weight_decay')
        if use_ramdisk:
            tmp_path_dist_ckpt = Path("/dev/shm")
        with TempNamedDir(
            tmp_path_dist_ckpt / "test_local", sync=True
        ) as local_ckpt_dir, mock.patch(
            'megatron.training.checkpointing.get_args', new=lambda: mock_args
        ), mock.patch(
            'megatron.training.async_utils.get_args', new=lambda: mock_args
        ), mock.patch(
            "megatron.training.checkpointing.update_num_microbatches"
        ):
            local_ckpt_dir = local_ckpt_dir / "subdir"  # Test handling of non-existent directories
            init_basic_mock_args(mock_args, tp, pp)
            init_checkpointing_mock_args(mock_args, None)
            mock_args.non_persistent_ckpt_type = 'local'
            mock_args.non_persistent_local_ckpt_algo = algo
            mock_args.async_save = async_save
            checkpointing_context = {
                'local_checkpoint_manager': LocalCheckpointManager(local_ckpt_dir)
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
            ckpt_id = checkpointing_context['local_checkpoint_manager']._ckpt_id(iteration)
            ckpt_path = checkpointing_context['local_checkpoint_manager']._local_ckpt_path_from_id(
                ckpt_id
            )
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
            if Utils.rank > 0:  # Skip assertion on rank 0 due to harmless nondeterminism
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
            time.sleep(0.01)  # Allow sufficient time for async cleanup to complete
            assert not ckpt_path.exists()
            ckpt_id = checkpointing_context['local_checkpoint_manager']._ckpt_id(2)
            ckpt_path = checkpointing_context['local_checkpoint_manager']._local_ckpt_path_from_id(
                ckpt_id
            )
            assert ckpt_path.exists()

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(('tp,pp'), [(1, 1), (2, 4)])
    @pytest.mark.parametrize(('use_ramdisk'), [True, False])
    @pytest.mark.parametrize(('async_save'), [True, False])
    @pytest.mark.parametrize(('algo'), ['atomic', 'fully_parallel'])
    @pytest.mark.flaky_in_dev
    def test_failed_save(self, caplog, tmp_path_dist_ckpt, tp, pp, use_ramdisk, async_save, algo):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)
        opt_param_scheduler = None

        mock_args = parse_args(ignore_unknown_args=True)
        if use_ramdisk:
            tmp_path_dist_ckpt = Path("/dev/shm")

        def test_save_wrapper(save_wrapper, subdir):
            with TempNamedDir(tmp_path_dist_ckpt / subdir, sync=True) as local_ckpt_dir, mock.patch(
                'megatron.training.checkpointing.get_args', new=lambda: mock_args
            ), mock.patch(
                'megatron.training.async_utils.get_args', new=lambda: mock_args
            ), mock.patch(
                "megatron.training.checkpointing.update_num_microbatches"
            ), mock.patch.object(
                LocalCheckpointManager, '_save', new=save_wrapper
            ), caplog.at_level(
                logging.INFO
            ):

                local_ckpt_dir = (
                    local_ckpt_dir / "subdir"
                )  # Test handling of non-existent directories
                init_basic_mock_args(mock_args, tp, pp)
                init_checkpointing_mock_args(mock_args, None)
                mock_args.non_persistent_ckpt_type = 'local'
                mock_args.non_persistent_local_ckpt_algo = algo
                mock_args.async_save = async_save
                checkpointing_context = {
                    'local_checkpoint_manager': LocalCheckpointManager(local_ckpt_dir)
                }

                with pytest.raises(CheckpointingException):
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
                    model,
                    optimizer,
                    opt_param_scheduler,
                    checkpointing_context=checkpointing_context,
                )
                assert iteration == 0
                assert not any((local_ckpt_dir / str(Utils.rank)).iterdir())

            if Utils.rank == 1:
                assert f"iter_0000001_{Utils.rank}_local.pt" not in caplog.text
            else:
                assert f"iter_0000001_{Utils.rank}_local.pt" in caplog.text

        original_save = LocalCheckpointManager._save

        def silent_error(self, *args, **kwargs):
            if self.rank == 1:
                return
            return original_save(self, *args, **kwargs)

        def exception(self, *args, **kwargs):
            if self.rank == 1:
                raise Exception("TEST")
            return original_save(self, *args, **kwargs)

        test_save_wrapper(silent_error, "test_sync")
        if async_save:
            test_save_wrapper(exception, "test_async")
        Utils.destroy_model_parallel()
