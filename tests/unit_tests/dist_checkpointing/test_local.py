# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import filecmp
import logging
import shutil
import tempfile
from pathlib import Path
import traceback
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.training.arguments import parse_args

from megatron.core.device_utils import get_xla_model
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
from megatron.core.parallel_state import get_data_parallel_group, get_data_parallel_group_gloo, get_default_process_group
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.checkpointing import generate_state_dict, load_checkpoint, save_checkpoint
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils

xm = get_xla_model()
from .utils import find_matching_values

class TestLocalCheckpointingReplication:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_filename_to_id(self):
        Utils.initialize_model_parallel()
        iteration_string = "0000123"
        rank = "4"
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = LocalCheckpointManager(tmpdir, group=get_default_process_group())
            filename = ckpt_mgr._filename_from_template(iteration_string, rank)
            assert (123, 4) == ckpt_mgr._filename_to_id(filename)[:2]

    @pytest.mark.parametrize(('tp,pp'), [(2, 4)])
    def test_sharded_tensors(self, tp, pp):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        dist_opt = xm is None
        model, optimizer = setup_model_and_optimizer(1, tp, pp, dist_opt=dist_opt)

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
        dist_opt = xm is None
        model, optimizer = setup_model_and_optimizer(1, tp, pp, dist_opt=dist_opt)
        opt_param_scheduler = None
        rng_state = None
        use_dist_ckpt = True
        iteration = None
        optim_sd_kwargs = dict(sharding_type='fully_sharded_model_space')
        mock_args = SimpleNamespace()
        mock_args.no_save_optim = False
        mock_args.no_save_rng = True
        mock_args.use_torch_fsdp2 = use_torch_fsdp2
        mock_args.ckpt_format = "torch"
        # Test save_local
        state_dict = generate_state_dict(
            mock_args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            use_dist_ckpt=use_dist_ckpt,
            iteration=iteration,
            optim_sd_kwargs=optim_sd_kwargs,
        )
        sharded_tensor_factories = find_matching_values(
            state_dict, lambda x: isinstance(x, ShardedTensorFactory)
        )
        sharded_tensors = find_matching_values(state_dict, lambda x: isinstance(x, ShardedTensor))
        for ten in sharded_tensors:
            assert ten.data != None
        parallelization_group = get_data_parallel_group() if not xm else get_data_parallel_group_gloo()
        saved_state_dict, _ = MCoreTensorAwareStateDict.from_state_dict(state_dict, algo='atomic', 
                                                                        parallelization_group=parallelization_group,
                                                                        process_group=get_default_process_group())
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
            saved_state_dict, lambda x: isinstance(x, ShardedTensor) and x.data is not None
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
            use_dist_ckpt=use_dist_ckpt,
            iteration=iteration,
            optim_sd_kwargs=optim_sd_kwargs,
        )
        nonpersistent_state_dict, _ = extract_nonpersistent(state_dict)
        # For a given use case
        if dist_opt:
            assert not nonpersistent_state_dict
        else:
            assert not nonpersistent_state_dict or nonpersistent_state_dict['optimizer']['optimizer']['param_groups'], f"nonpersistent_state_dict: {nonpersistent_state_dict}"
        loaded_state_dict = saved_state_dict.to_state_dict(state_dict)
        if not dist_opt:
            for group in state_dict['optimizer']['optimizer']['param_groups']:
                del group['params']

            for group in loaded_state_dict['optimizer']['optimizer']['param_groups']:
                del group['params']

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
        dist_opt = xm is None
        model, optimizer = setup_model_and_optimizer(1, tp, pp, dist_opt=dist_opt)
        opt_param_scheduler = None

        mock_args = (
            SimpleNamespace()
        )  # FIXME: fails with additional arguments (e.g.,'weight_decay')
        if use_ramdisk:
            tmp_path_dist_ckpt = Path("/dev/shm")
        with TempNamedDir(tmp_path_dist_ckpt / "test_local", process_group=get_default_process_group()) as local_ckpt_dir, mock.patch(
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
                'local_checkpoint_manager': LocalCheckpointManager(local_ckpt_dir,
                                                                    group=get_default_process_group())
            }
            torch.distributed.barrier(group=get_default_process_group())
            
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
                torch.distributed.barrier(group=get_default_process_group())
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
                torch.distributed.barrier(group=get_default_process_group())
            assert not ckpt_path.exists(), f"rank: {Utils.rank} path: {ckpt_path}"
            ckpt_id = checkpointing_context['local_checkpoint_manager']._ckpt_id(2)
            ckpt_path = checkpointing_context['local_checkpoint_manager']._local_ckpt_path_from_id(
                ckpt_id
            )
            assert ckpt_path.exists(), f"rank: {Utils.rank} path: {ckpt_path}"

    @pytest.mark.parametrize(('tp,pp'), [(1, 1), (2, 4)])
    @pytest.mark.parametrize(('use_ramdisk'), [True, False])
    @pytest.mark.parametrize(('async_save'), [True, False])
    @pytest.mark.parametrize(('algo'), ['atomic', 'fully_parallel'])
    @pytest.mark.flaky_in_dev
    def test_failed_save(self, caplog, tmp_path_dist_ckpt, tp, pp, use_ramdisk, async_save, algo):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        dist_opt = xm is None
        model, optimizer = setup_model_and_optimizer(1, tp, pp, dist_opt=dist_opt)
        opt_param_scheduler = None

        mock_args = parse_args(ignore_unknown_args=True)
        if use_ramdisk:
            tmp_path_dist_ckpt = Path("/dev/shm")

        def test_save_wrapper(save_wrapper, subdir):
            with TempNamedDir(
                tmp_path_dist_ckpt / subdir, sync=True,
                process_group=get_default_process_group()
            ) as local_ckpt_dir, mock.patch(
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
                    'local_checkpoint_manager': LocalCheckpointManager(local_ckpt_dir,
                                                                       group=get_default_process_group())
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
                        torch.distributed.barrier(group=get_default_process_group())
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
