# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from functools import partial
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load_plain_tensors, save
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
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
from tests.unit_tests.dist_checkpointing.utils import initialize_moe_model
from tests.unit_tests.test_utilities import Utils


def advance_multi_instance_optimizer(optimizer):
    """Advance Muon, scalar state, and model weights with valid replicated gradients."""
    # TP- and CP-replicated parameters must receive the same reduced gradient
    # across their replicas. Distinct DP replicas still contribute distinct values.
    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=False)
    model_chunks = optimizer.chained_optimizers[0].model_chunks
    before = [param.detach().clone() for chunk in model_chunks for param in chunk.parameters()]
    # The checkpoint fixture initializes Muon with LR=0. One large test step
    # makes the model update visible after BF16 rounding.
    for child in optimizer.chained_optimizers:
        for member in getattr(child, 'chained_optimizers', [child]):
            for group in member.optimizer.param_groups:
                group['lr'] = 1.0
    for model_chunk in model_chunks:
        model_chunk.zero_grad_buffer()
        for index, param in enumerate(model_chunk.parameters()):
            if param.requires_grad:
                param.main_grad.fill_((dp_rank + 1) * (index + 1) / 128)
        model_chunk.finish_grad_sync()
    success, _, _ = optimizer.step()
    assert success
    for model_chunk in model_chunks:
        model_chunk.start_param_sync(force_sync=True)
    assert any(
        not torch.equal(previous, param)
        for previous, param in zip(
            before, (param for chunk in model_chunks for param in chunk.parameters())
        )
    )


class TestLayerWiseOptimizerCommonStateDict:
    """Tests for LayerWiseDistributedOptimizer common state dictionaries."""

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('tp,cp,ep', [(1, 1, 1), (1, 1, 2), (2, 1, 1), (1, 2, 1)])
    @pytest.mark.parametrize('fully_reshardable', [False, True])
    def test_multi_instance_training_checkpoint_roundtrip(
        self, tmp_path_dist_ckpt, tp, cp, ep, fully_reshardable
    ):
        """Exercise the training save/load path with two Muon optimizer instances."""
        if Utils.world_size % (2 * tp * cp * ep):
            pytest.skip('World size must be divisible by two instances times TP, CP, and EP')

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            expert_model_parallel_size=ep,
            num_distributed_optimizer_instances=2,
        )
        if ep > 1:
            initialize_fn = partial(initialize_moe_model, add_bias_linear=True)
        elif cp > 1:
            initialize_fn = partial(initialize_real_model, is_moe=False, is_mla=False)
        else:
            initialize_fn = initialize_gpt_model
        with TempNamedDir(tmp_path_dist_ckpt / 'multi_instance_training', sync=True) as ckpt_dir:
            mock_args = parse_args(ignore_unknown_args=True)
            mock_args.save_tokenizer_assets = False
            mock_args.use_distributed_optimizer = False
            mock_args.use_layer_wise_distributed_optimizer = True
            mock_args.num_distributed_optimizer_instances = 2
            mock_args.tensor_model_parallel_size = tp
            mock_args.context_parallel_size = cp
            mock_args.expert_model_parallel_size = ep
            with mock.patch('megatron.training.checkpointing.get_args', new=lambda: mock_args):
                model_A, optimizer_A = setup_model_and_optimizer(
                    seed=2,
                    tp=tp,
                    pp=1,
                    cp=cp,
                    ep=ep,
                    initialize_fn=initialize_fn,
                    dist_opt=True,
                    optimizer='dist_muon',
                    use_param_layout=True,
                    num_distributed_optimizer_instances=2,
                )
                if ep > 1:
                    expert_dp = parallel_state.get_expert_data_parallel_group(
                        with_gtp_remat=False, partial_expert_data_parallel=True
                    )
                    assert any(
                        isinstance(child, DistributedOptimizer)
                        and child.data_parallel_group is expert_dp
                        for child in optimizer_A.chained_optimizers
                    ), 'Expected an expert scalar fallback optimizer'
                # Give each DP replica distinct gradients, then checkpoint after Muon and
                # scalar optimizer state has advanced beyond initialization.
                advance_multi_instance_optimizer(optimizer_A)
                init_checkpointing_mock_args(mock_args, ckpt_dir, fully_parallel=True)
                mock_args.dist_ckpt_optim_fully_reshardable = fully_reshardable
                from megatron.training.training import preprocess_common_state_dict

                save_checkpoint(
                    10,
                    model_A,
                    optimizer_A,
                    None,
                    0,
                    preprocess_common_state_dict_fn=preprocess_common_state_dict,
                )

                model_B, optimizer_B = setup_model_and_optimizer(
                    seed=3,
                    tp=tp,
                    pp=1,
                    cp=cp,
                    ep=ep,
                    initialize_fn=initialize_fn,
                    dist_opt=True,
                    optimizer='dist_muon',
                    use_param_layout=True,
                    num_distributed_optimizer_instances=2,
                )
                load_checkpoint_no_arg_checks(model_B, optimizer_B, None)

                check_equal(model_A[0].state_dict(), model_B[0].state_dict())
                check_equal(optimizer_A.state_dict(), optimizer_B.state_dict())
                for source, restored in zip(
                    optimizer_A.chained_optimizers, optimizer_B.chained_optimizers
                ):
                    if isinstance(source, DistributedOptimizer):
                        check_equal(
                            source.get_parameter_state_dp_reshardable(),
                            restored.get_parameter_state_dp_reshardable(),
                        )

    @pytest.mark.parametrize(
        'source_instances,target_instances,fully_reshardable',
        [
            (2, 1, False),
            (2, 1, True),
            (1, 2, False),
            (1, 2, True),
            (2, 4, False),
            (2, 4, True),
        ],
    )
    def test_multi_instance_training_checkpoint_reshard(
        self,
        tmp_path_dist_ckpt,
        source_instances,
        target_instances,
        fully_reshardable,
    ):
        """Restore a training checkpoint after changing optimizer-instance count."""
        if Utils.world_size % max(source_instances, target_instances):
            pytest.skip('World size must be divisible by both instance counts')

        Utils.initialize_model_parallel(
            num_distributed_optimizer_instances=source_instances,
        )
        with TempNamedDir(tmp_path_dist_ckpt / 'instance_reshard', sync=True) as ckpt_dir:
            with TempNamedDir(tmp_path_dist_ckpt / 'source_optim', sync=True) as source_dir:
                with TempNamedDir(tmp_path_dist_ckpt / 'restored_optim', sync=True) as target_dir:
                    mock_args = parse_args(ignore_unknown_args=True)
                    mock_args.save_tokenizer_assets = False
                    mock_args.use_distributed_optimizer = False
                    mock_args.use_layer_wise_distributed_optimizer = True
                    mock_args.num_distributed_optimizer_instances = source_instances
                    with mock.patch('megatron.training.checkpointing.get_args', new=lambda: mock_args):
                        model_A, optimizer_A = setup_model_and_optimizer(
                            seed=2,
                            tp=1,
                            pp=1,
                            dist_opt=True,
                            optimizer='dist_muon',
                            use_param_layout=True,
                            num_distributed_optimizer_instances=source_instances,
                        )
                        advance_multi_instance_optimizer(optimizer_A)
                        init_checkpointing_mock_args(mock_args, ckpt_dir, fully_parallel=True)
                        mock_args.dist_ckpt_optim_fully_reshardable = fully_reshardable
                        from megatron.training.training import preprocess_common_state_dict

                        save_checkpoint(
                            10,
                            model_A,
                            optimizer_A,
                            None,
                            0,
                            preprocess_common_state_dict_fn=preprocess_common_state_dict,
                        )
                        canonical_metadata = {
                            'distrib_optim_sharding_type': 'fully_reshardable',
                            'chained_optim_avoid_prefix': True,
                        }
                        save(
                            optimizer_A.sharded_state_dict(
                                model_A[0].sharded_state_dict(), metadata=canonical_metadata
                            ),
                            source_dir,
                        )

                        Utils.destroy_model_parallel()
                        Utils.initialize_model_parallel(
                            num_distributed_optimizer_instances=target_instances,
                        )
                        mock_args.num_distributed_optimizer_instances = target_instances
                        model_B, optimizer_B = setup_model_and_optimizer(
                            seed=3,
                            tp=1,
                            pp=1,
                            dist_opt=True,
                            optimizer='dist_muon',
                            use_param_layout=True,
                            num_distributed_optimizer_instances=target_instances,
                        )
                        load_checkpoint_no_arg_checks(model_B, optimizer_B, None)
                        check_equal(model_A[0].state_dict(), model_B[0].state_dict())
                        save(
                            optimizer_B.sharded_state_dict(
                                model_B[0].sharded_state_dict(), metadata=canonical_metadata
                            ),
                            target_dir,
                        )
                        check_equal(
                            load_plain_tensors(source_dir), load_plain_tensors(target_dir)
                        )

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
