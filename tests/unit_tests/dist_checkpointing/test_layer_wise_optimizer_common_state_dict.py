# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from functools import partial
from unittest import mock

import pytest
import torch

from megatron.training.arguments import parse_args
from megatron.training.checkpointing import save_checkpoint
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_checkpointing_mock_args,
    initialize_gpt_model,
    setup_model_and_optimizer,
    setup_moe_model_and_optimizer,
)
from tests.unit_tests.dist_checkpointing.test_layer_wise_optimizer import (
    check_equal,
    initialize_real_model,
    load_checkpoint_no_arg_checks,
)
from tests.unit_tests.test_utilities import Utils


class TestLayerWiseOptimizerCommonStateDict:
    """Tests for LayerWiseDistributedOptimizer common state dictionaries."""

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    # TODO(@boxiangw): Add test for loading with different TP/PP sizes
    @pytest.mark.parametrize("fully_parallel", [True, False])
    @pytest.mark.parametrize('optimizer_type', ['dist_muon', 'muon'])
    @pytest.mark.parametrize('tp', [1, 2, 4])
    @pytest.mark.parametrize('pp', [1, 2])
    @pytest.mark.parametrize('ep', [1, 2, 4])
    @pytest.mark.parametrize('is_moe', [True, False])
    @pytest.mark.parametrize('is_mla', [True, False])
    def test_optimizer_common_state_dict(
        self, tmp_path_dist_ckpt, fully_parallel, tp, pp, ep, is_moe, is_mla, optimizer_type
    ):
        if tp * pp * ep > 8:
            pytest.skip(f"TP*PP*EP > 8 is larger than world size")

        if ep > 1 and not is_moe:
            pytest.skip(f"EP > 1 needs to be used with MoE")

        initialize_fn = partial(initialize_real_model, is_moe=is_moe, is_mla=is_mla)

        # Initialize parallel
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
        )
        rank = torch.distributed.get_rank()

        with TempNamedDir(tmp_path_dist_ckpt / 'test_dp_sharding', sync=True) as ckpt_dir:
            mock_args = parse_args(ignore_unknown_args=True)
            mock_args.use_distributed_optimizer = False
            mock_args.save_tokenizer_assets = False
            with mock.patch('megatron.training.checkpointing.get_args', new=lambda: mock_args):
                # Initialize model and optimizer A
                if is_moe:
                    model, optimizer_A = setup_moe_model_and_optimizer(
                        seed=2,
                        tp=tp,
                        pp=pp,
                        ep=ep,
                        initialize_fn=initialize_fn,
                        dist_opt=False,
                        optimizer=optimizer_type,
                    )
                else:
                    model, optimizer_A = setup_model_and_optimizer(
                        seed=2,
                        tp=tp,
                        pp=pp,
                        initialize_fn=initialize_fn,
                        dist_opt=False,
                        optimizer=optimizer_type,
                    )

                # Save checkpoint
                init_checkpointing_mock_args(mock_args, ckpt_dir, fully_parallel=fully_parallel)
                from megatron.training.training import preprocess_common_state_dict

                save_checkpoint(
                    10,
                    model,
                    optimizer_A,
                    None,
                    0,
                    preprocess_common_state_dict_fn=preprocess_common_state_dict,
                )

                # Get optimizer A param state
                optim_param_state_A = optimizer_A.state_dict()

                # Initialize model and optimizer B
                if is_moe:
                    model, optimizer_B = setup_moe_model_and_optimizer(
                        seed=3,
                        tp=tp,
                        pp=pp,
                        ep=ep,
                        initialize_fn=initialize_fn,
                        dist_opt=False,
                        optimizer=optimizer_type,
                    )
                else:
                    model, optimizer_B = setup_model_and_optimizer(
                        seed=3,
                        tp=tp,
                        pp=pp,
                        initialize_fn=initialize_fn,
                        dist_opt=False,
                        optimizer=optimizer_type,
                    )

                # Load optimizer B from checkpoint
                load_checkpoint_no_arg_checks(model, optimizer_B, None)

                # Get optimizer B param state
                optim_param_state_B = optimizer_B.state_dict()

                # Test both param state dicts are equal
                check_equal(optim_param_state_A, optim_param_state_B)

    @pytest.mark.parametrize('tp', [1, 2])
    @pytest.mark.parametrize('pp', [1, 2])
    def test_optimizer_common_state_dict_hybrid(self, tmp_path_dist_ckpt, tp, pp):
        """End-to-end ``save_checkpoint``/``load_checkpoint`` roundtrip on the
        hybrid LayerWise + DistributedOptimizer path.

        Muon matrix params live in :class:`LayerWiseDistributedOptimizer`
        while non-Muon params (embeddings, biases, layernorm) go through a
        real :class:`DistributedOptimizer` sub-optimizer. Catches the case
        where ``_build_sharded_state_dict_metadata`` skipped populating
        ``distrib_optim_sharding_type`` because the arg parser flips
        ``use_distributed_optimizer`` off in Muon mode -- the DistOpt
        sub-optimizer then defaulted to the deprecated
        ``fully_sharded_model_space`` save path which is incompatible with
        the post-5ab481cb45 ShardedTensor validation.
        """
        if tp * pp > 8:
            pytest.skip("TP*PP > 8 is larger than world size")

        Utils.initialize_model_parallel(tp, pp)

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_optimizer_common_state_dict_hybrid', sync=True
        ) as ckpt_dir:
            mock_args = parse_args(ignore_unknown_args=True)
            mock_args.save_tokenizer_assets = False
            # Mirror the arg-parser's Muon path: ``use_distributed_optimizer``
            # is flipped off and ``use_layer_wise_distributed_optimizer`` is
            # the surviving flag.
            mock_args.use_distributed_optimizer = False
            mock_args.use_layer_wise_distributed_optimizer = True
            with mock.patch('megatron.training.checkpointing.get_args', new=lambda: mock_args):
                model, optimizer_A = setup_model_and_optimizer(
                    seed=2,
                    tp=tp,
                    pp=pp,
                    initialize_fn=initialize_gpt_model,
                    dist_opt=True,
                    optimizer='dist_muon',
                    use_param_layout=True,
                )

                init_checkpointing_mock_args(mock_args, ckpt_dir, fully_parallel=True)
                from megatron.training.training import preprocess_common_state_dict

                save_checkpoint(
                    10,
                    model,
                    optimizer_A,
                    None,
                    0,
                    preprocess_common_state_dict_fn=preprocess_common_state_dict,
                )

                optim_param_state_A = optimizer_A.state_dict()

                model, optimizer_B = setup_model_and_optimizer(
                    seed=3,
                    tp=tp,
                    pp=pp,
                    initialize_fn=initialize_gpt_model,
                    dist_opt=True,
                    optimizer='dist_muon',
                    use_param_layout=True,
                )

                load_checkpoint_no_arg_checks(model, optimizer_B, None)

                optim_param_state_B = optimizer_B.state_dict()

                check_equal(optim_param_state_A, optim_param_state_B)

        Utils.destroy_model_parallel()
