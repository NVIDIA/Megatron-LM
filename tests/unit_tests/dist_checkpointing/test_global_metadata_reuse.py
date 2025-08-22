# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


from unittest import mock

import pytest

from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils


class TestGlobalMetadataReuse:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(('tp,pp'), [(2, 4)])
    def test_global_metadata_reuse(self, tmp_path_dist_ckpt, tp, pp):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)
        opt_param_scheduler = None

        mock_args = parse_args(ignore_unknown_args=True)
        with TempNamedDir(
            tmp_path_dist_ckpt / "test_global_metadata_reuse"
        ) as non_persistent_ckpt_dir, mock.patch(
            'megatron.training.checkpointing.get_args', new=lambda: mock_args
        ), mock.patch(
            "megatron.training.checkpointing.update_num_microbatches"
        ):
            init_basic_mock_args(mock_args, tp, pp)
            init_checkpointing_mock_args(mock_args, non_persistent_ckpt_dir)
            mock_args.non_persistent_ckpt_type = "global"
            mock_args.ckpt_assume_constant_structure = True
            save_ckpt_context = {}

            # Check we avoid reduce_scatter
            with mock.patch(
                'torch.distributed.checkpoint.utils._DistWrapper.reduce_scatter'
            ) as reduce_scatter_mock:
                save_checkpoint(
                    1,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    save_ckpt_context,
                )

                assert reduce_scatter_mock.call_count == 0

            assert save_ckpt_context['save_strategy'].cached_global_metadata is None

            resume_ckpt_context = {}
            _, _ = load_checkpoint(
                model, optimizer, opt_param_scheduler, checkpointing_context=resume_ckpt_context
            )

            load_strategy_cached_metadata = resume_ckpt_context[
                'load_strategy'
            ].cached_global_metadata
            assert load_strategy_cached_metadata is not None
            assert getattr(load_strategy_cached_metadata, "all_local_plans", None) is not None

            # Check we avoid reduce_scatter
            with mock.patch(
                'torch.distributed.checkpoint.utils._DistWrapper.reduce_scatter'
            ) as reduce_scatter_mock:
                save_checkpoint(
                    2,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    resume_ckpt_context,
                )
                assert reduce_scatter_mock.call_count == 0

            assert (
                load_strategy_cached_metadata
                is resume_ckpt_context['save_strategy'].cached_global_metadata
            )

            assert resume_ckpt_context['save_strategy'].validated_loaded_metadata_reuse

    @pytest.mark.parametrize(('tp,pp'), [(2, 4)])
    def test_no_global_metadata_reuse_on_different_parallelism(self, tmp_path_dist_ckpt, tp, pp):
        Utils.initialize_model_parallel(tp, pp)
        num_floating_point_operations_so_far = 0
        model, optimizer = setup_model_and_optimizer(1, tp, pp)
        opt_param_scheduler = None

        mock_args = parse_args(ignore_unknown_args=True)
        with TempNamedDir(
            tmp_path_dist_ckpt / "test_global_metadata_reuse"
        ) as non_persistent_ckpt_dir, mock.patch(
            'megatron.training.checkpointing.get_args', new=lambda: mock_args
        ), mock.patch(
            "megatron.training.checkpointing.update_num_microbatches"
        ):
            init_basic_mock_args(mock_args, tp, pp)
            init_checkpointing_mock_args(mock_args, non_persistent_ckpt_dir)
            mock_args.non_persistent_ckpt_type = "global"
            mock_args.ckpt_assume_constant_structure = True
            mock_args.ckpt_fully_parallel_save = True

            save_ckpt_context = {}

            # Check we avoid reduce_scatter
            with mock.patch(
                'torch.distributed.checkpoint.utils._DistWrapper.reduce_scatter'
            ) as reduce_scatter_mock:
                save_checkpoint(
                    1,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    save_ckpt_context,
                )

                assert reduce_scatter_mock.call_count == 0

            assert save_ckpt_context['save_strategy'].base_strategy.cached_global_metadata is None

            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(pp, tp)
            model, optimizer = setup_model_and_optimizer(1, pp, tp)
            init_basic_mock_args(mock_args, pp, tp)
            mock_args.no_load_rng = True

            resume_ckpt_context = {}
            _, _ = load_checkpoint(
                model, optimizer, opt_param_scheduler, checkpointing_context=resume_ckpt_context
            )

            load_strategy_cached_metadata = resume_ckpt_context[
                'load_strategy'
            ].cached_global_metadata

            assert load_strategy_cached_metadata is not None
            assert getattr(load_strategy_cached_metadata, "all_local_plans", None) is not None

            # Check we avoid reduce_scatter
            with mock.patch(
                'torch.distributed.checkpoint.utils._DistWrapper.reduce_scatter'
            ) as reduce_scatter_mock:
                save_checkpoint(
                    2,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    resume_ckpt_context,
                )
                assert reduce_scatter_mock.call_count == 0

            assert not resume_ckpt_context[
                'save_strategy'
            ].base_strategy.validated_loaded_metadata_reuse
