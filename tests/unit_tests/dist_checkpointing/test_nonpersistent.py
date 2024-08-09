# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import filecmp
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.training.checkpointing import (
    _NON_PERSISTENT_CKPT_SUBDIR,
    load_checkpoint,
    save_checkpoint,
)
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils


class TestNonPersistentSaveAndLoad:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(('tp,pp'), [(2, 4)])
    def test_basic_save_load_scenarios(self, tmp_path_dist_ckpt, tp, pp):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)
        opt_param_scheduler = None

        mock_args = SimpleNamespace()
        with TempNamedDir(
            tmp_path_dist_ckpt / "test_non_persistent"
        ) as non_persistent_ckpt_dir, mock.patch(
            'megatron.training.checkpointing.get_args', new=lambda: mock_args
        ), mock.patch(
            "megatron.training.checkpointing.update_num_microbatches"
        ):
            init_basic_mock_args(mock_args, tp, pp)
            init_checkpointing_mock_args(mock_args, non_persistent_ckpt_dir)
            mock_args.non_persistent_ckpt_type = "global"

            save_checkpoint(
                2,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                {},
                non_persistent_ckpt=True,
            )
            save_checkpoint(
                3, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far, {}
            )
            save_checkpoint(
                4,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                {},
                non_persistent_ckpt=True,
            )
            iteration, _ = load_checkpoint(model, optimizer, opt_param_scheduler)
            assert iteration == 4
            save_checkpoint(
                6, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far, {}
            )
            iteration, _ = load_checkpoint(model, optimizer, opt_param_scheduler)
            assert iteration == 6
            save_checkpoint(
                8,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                {},
                non_persistent_ckpt=True,
            )
            iteration, _ = load_checkpoint(model, optimizer, opt_param_scheduler)
            assert iteration == 8
            assert "iter_0000003" in os.listdir(non_persistent_ckpt_dir)
            assert "iter_0000006" in os.listdir(non_persistent_ckpt_dir)
            assert "iter_0000002" not in os.listdir(
                os.path.join(non_persistent_ckpt_dir, _NON_PERSISTENT_CKPT_SUBDIR)
            )
            assert "iter_0000004" in os.listdir(
                os.path.join(non_persistent_ckpt_dir, _NON_PERSISTENT_CKPT_SUBDIR)
            )
            assert "iter_0000008" in os.listdir(
                os.path.join(non_persistent_ckpt_dir, _NON_PERSISTENT_CKPT_SUBDIR)
            )
            ckpt_dirs = [
                "iter_0000003",
                "iter_0000006",
                _NON_PERSISTENT_CKPT_SUBDIR + "/iter_0000004",
                _NON_PERSISTENT_CKPT_SUBDIR + "/iter_0000008",
            ]
            for ckpt_a in ckpt_dirs:
                for ckpt_b in ckpt_dirs:
                    for filename in os.listdir(os.path.join(non_persistent_ckpt_dir, ckpt_a)):
                        if filename != "common.pt":
                            assert filecmp.cmp(
                                os.path.join(non_persistent_ckpt_dir, ckpt_a, filename),
                                os.path.join(non_persistent_ckpt_dir, ckpt_b, filename),
                                shallow=False,
                            ), [filename, ckpt_a, ckpt_b]
        Utils.destroy_model_parallel()


class TestLegacySaveAndLoad:
    @pytest.mark.parametrize(('tp,pp'), [(2, 4)])
    @pytest.mark.skip(reason="Tests are flaky and need to be debugged")
    def test_basic_save_load_scenario(self, tmp_path_dist_ckpt, tp, pp):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)
        opt_param_scheduler = None

        mock_args = SimpleNamespace()
        with TempNamedDir(tmp_path_dist_ckpt / "test_legacy") as legacy_ckpt_dir, mock.patch(
            'megatron.training.checkpointing.get_args', new=lambda: mock_args
        ), mock.patch("megatron.training.checkpointing.update_num_microbatches"):
            init_basic_mock_args(mock_args, tp, pp)
            init_checkpointing_mock_args(mock_args, legacy_ckpt_dir)

            save_checkpoint(
                2, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far, {}
            )
            iteration, _ = load_checkpoint(model, optimizer, opt_param_scheduler)
            assert iteration == 2
            assert "iter_0000002" in os.listdir(legacy_ckpt_dir)

        Utils.destroy_model_parallel()
