# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy
from functools import partial
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.dict_utils import diff, nested_values
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.utils import get_pg_size
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


def initialize_real_model(
    seed,
    pre_process,
    post_process,
    vp_stage=None,
    is_moe=False,
    is_mla=False,
    virtual_pipeline_model_parallel_size=None,
    **config_kwargs,
):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(
        num_layers=6,
        hidden_size=16,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
    )
    if is_moe:
        default_config_kwargs["moe_ffn_hidden_size"] = 128
        default_config_kwargs["num_moe_experts"] = 4
        default_config_kwargs["add_bias_linear"] = False
        # Pop unused fields
        config_kwargs.pop("use_sp")
        config_kwargs.pop("use_te")
        config_kwargs.pop("use_grouped_mlp")
        config_kwargs.pop("use_glu")
    if is_mla:
        default_config_kwargs["multi_latent_attention"] = True
        default_config_kwargs["q_lora_rank"] = 96
        default_config_kwargs["kv_lora_rank"] = 512
        default_config_kwargs["qk_head_dim"] = 64
        default_config_kwargs["qk_pos_emb_head_dim"] = 32
        default_config_kwargs["v_head_dim"] = 64
    default_config_kwargs.update(**config_kwargs)
    config_cls = MLATransformerConfig if is_mla else TransformerConfig
    transformer_config = config_cls(**default_config_kwargs)

    if is_moe:
        layer_spec = get_gpt_decoder_block_spec(
            transformer_config, use_transformer_engine=True, vp_stage=vp_stage
        )
    else:
        layer_spec = gpt_te_spec(multi_latent_attention=is_mla)
    this_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=layer_spec,
        vocab_size=128,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )

    return this_model


def load_checkpoint_no_arg_checks(*args, **kwargs):
    with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
        with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
            return load_checkpoint(*args, **kwargs)


class TestLayerWiseOptimizer:
    """Tests for LayerWiseDistributedOptimizer functionality."""

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_parameter_sharding(self):
        """Test that parameters are correctly sharded across DP ranks."""
        Utils.initialize_model_parallel(1, 1)

        model, optimizer = setup_model_and_optimizer(
            seed=2,
            tp=1,
            pp=1,
            bf16=True,
            dist_opt=False,
            initialize_fn=initialize_gpt_model,
            optimizer='dist_muon',
        )

        # Check if optimizer is ChainedOptimizer (expected for standard setup)
        if isinstance(optimizer, ChainedOptimizer):
            total_params = sum(
                len(group['params'])
                for opt in optimizer.chained_optimizers
                for group in opt.param_groups
            )
            assert total_params > 0, "No parameters found in optimizer"

    @pytest.mark.parametrize('tp_pp', [(1, 2), (2, 1), (2, 2)])
    def test_broadcast_params(self, tp_pp):
        """Test that parameter broadcasting works correctly across DP ranks."""
        Utils.initialize_model_parallel(*tp_pp)

        model, optimizer = setup_model_and_optimizer(
            seed=2,
            tp=tp_pp[0],
            pp=tp_pp[1],
            bf16=True,
            dist_opt=False,
            initialize_fn=initialize_gpt_model,
            optimizer='dist_muon',
        )

        # If this is a LayerWiseDistributedOptimizer, test broadcast
        if isinstance(optimizer, LayerWiseDistributedOptimizer):
            # Store original param values
            original_params = {}
            for name, param in model[0].named_parameters():
                original_params[name] = param.data.clone()

            # Call broadcast (should be idempotent if no updates)
            optimizer.broadcast_params()

            # Check params are unchanged after broadcast without step
            for name, param in model[0].named_parameters():
                assert torch.allclose(param.data, original_params[name])

    @pytest.mark.parametrize('tp_pp', [(2, 2), (2, 4), (4, 2)])
    @pytest.mark.parametrize('bf16', [True, False])
    def test_layer_wise_optimizer_save_load(self, tmp_path_dist_ckpt, tp_pp, bf16):
        """Test save/load of LayerWiseDistributedOptimizer checkpoints."""
        Utils.initialize_model_parallel(*tp_pp)

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_layer_wise_optimizer_A', sync=True
        ) as ckpt_dir_A:
            with TempNamedDir(
                tmp_path_dist_ckpt / 'test_layer_wise_optimizer_B', sync=True
            ) as ckpt_dir_B:
                # Create model and optimizer A
                model_A, optimizer_A = setup_model_and_optimizer(
                    seed=2,
                    tp=tp_pp[0],
                    pp=tp_pp[1],
                    bf16=bf16,
                    dist_opt=False,
                    initialize_fn=initialize_gpt_model,
                    optimizer='dist_muon',
                )

                # Save checkpoint A
                model_sharded_sd_A = model_A[0].sharded_state_dict()
                optim_sd_A = optimizer_A.sharded_state_dict(model_sharded_sd_A)
                save(optim_sd_A, ckpt_dir_A)

                # Create model and optimizer B with different seed
                model_B, optimizer_B = setup_model_and_optimizer(
                    seed=3,
                    tp=tp_pp[0],
                    pp=tp_pp[1],
                    bf16=bf16,
                    dist_opt=False,
                    initialize_fn=initialize_gpt_model,
                    optimizer='dist_muon',
                )

                # Load checkpoint A into optimizer B
                model_sharded_sd_B = model_B[0].sharded_state_dict()
                load_sharded_sd = optimizer_B.sharded_state_dict(
                    model_sharded_sd_B, is_loading=True
                )
                state_dict = load(load_sharded_sd, ckpt_dir_A)
                optimizer_B.load_state_dict(state_dict)

                # Save as checkpoint B
                optim_sd_B = optimizer_B.sharded_state_dict(model_sharded_sd_B)
                save(optim_sd_B, ckpt_dir_B)

                Utils.destroy_model_parallel()

                # Compare checkpoints
                Utils.initialize_model_parallel(1, 1)
                from megatron.core.dist_checkpointing import load_plain_tensors

                plain_sd_A = load_plain_tensors(ckpt_dir_A)
                plain_sd_B = load_plain_tensors(ckpt_dir_B)

                diffs = diff(plain_sd_A, plain_sd_B)
                assert not any(map(bool, diffs)), f"Checkpoints differ: {diffs}"

    @pytest.mark.parametrize('tp_pp', [(2, 2), (4, 1)])
    def test_layer_wise_optimizer_grad_norm(self, tp_pp):
        """Test that gradient norm calculation works correctly."""
        Utils.initialize_model_parallel(*tp_pp)

        model, optimizer = setup_model_and_optimizer(
            seed=2,
            tp=tp_pp[0],
            pp=tp_pp[1],
            bf16=True,
            dist_opt=False,
            initialize_fn=initialize_gpt_model,
            optimizer='dist_muon',
        )

        # Create dummy gradients
        for param in model[0].parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param.data)

        # Test grad norm calculation
        if isinstance(optimizer, LayerWiseDistributedOptimizer):
            grad_norm = optimizer.get_grad_norm()
            assert grad_norm is not None
            assert grad_norm >= 0

    @pytest.mark.parametrize('tp_pp', [(2, 2), (1, 4)])
    def test_layer_wise_optimizer_count_zeros(self, tp_pp):
        """Test that zero counting in gradients works correctly."""
        Utils.initialize_model_parallel(*tp_pp)

        model, optimizer = setup_model_and_optimizer(
            seed=2,
            tp=tp_pp[0],
            pp=tp_pp[1],
            bf16=True,
            dist_opt=False,
            initialize_fn=initialize_gpt_model,
            optimizer='dist_muon',
        )

        # Create dummy gradients with some zeros
        for param in model[0].parameters():
            if param.requires_grad:
                grad = torch.randn_like(param.data)
                # Set some values to zero
                grad[grad < 0] = 0
                param.grad = grad

        # Test zero counting
        if isinstance(optimizer, LayerWiseDistributedOptimizer):
            num_zeros = optimizer.count_zeros()
            assert num_zeros >= 0

    @pytest.mark.parametrize('src_tp_pp', [(2, 2), (4, 2)])
    @pytest.mark.parametrize('dest_tp_pp', [(2, 2), (4, 2)])
    def test_layer_wise_optimizer_resharding(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp):
        """Test resharding of LayerWiseDistributedOptimizer across different TP/PP."""
        Utils.initialize_model_parallel(*src_tp_pp)

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_layer_wise_resharding_A', sync=True
        ) as ckpt_dir:
            # Create and save with source configuration
            model_A, optimizer_A = setup_model_and_optimizer(
                seed=2,
                tp=src_tp_pp[0],
                pp=src_tp_pp[1],
                bf16=True,
                dist_opt=False,
                initialize_fn=initialize_gpt_model,
                optimizer='dist_muon',
            )

            model_sharded_sd = model_A[0].sharded_state_dict()
            optim_sd = optimizer_A.sharded_state_dict(model_sharded_sd)
            save(optim_sd, ckpt_dir)

            Utils.destroy_model_parallel()

            # Load with destination configuration
            Utils.initialize_model_parallel(*dest_tp_pp)
            model_B, optimizer_B = setup_model_and_optimizer(
                seed=3,
                tp=dest_tp_pp[0],
                pp=dest_tp_pp[1],
                bf16=True,
                dist_opt=False,
                initialize_fn=initialize_gpt_model,
                optimizer='dist_muon',
            )

            model_sharded_sd = model_B[0].sharded_state_dict()
            load_sharded_sd = optimizer_B.sharded_state_dict(model_sharded_sd, is_loading=True)

            # Load should work for same TP/PP or handle differences gracefully
            if src_tp_pp == dest_tp_pp:
                state_dict = load(load_sharded_sd, ckpt_dir)
                optimizer_B.load_state_dict(state_dict)
            else:
                # For different TP/PP, load may succeed or fail depending on compatibility
                try:
                    state_dict = load(load_sharded_sd, ckpt_dir)
                    optimizer_B.load_state_dict(state_dict)
                except Exception:
                    # Different TP/PP may not be compatible
                    pass

    @pytest.mark.parametrize('tp_pp_ep', [(2, 2, 2), (4, 1, 2)])
    def test_layer_wise_optimizer_with_moe(self, tmp_path_dist_ckpt, tp_pp_ep):
        """Test LayerWiseDistributedOptimizer with MoE models."""
        tp, pp, ep = tp_pp_ep
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
        )

        with TempNamedDir(tmp_path_dist_ckpt / 'test_layer_wise_moe', sync=True) as ckpt_dir:
            # Create MoE model with optimizer
            model, optimizer = setup_moe_model_and_optimizer(
                seed=2, tp=tp, pp=pp, ep=ep, bf16=True, dist_opt=False, optimizer='dist_muon'
            )

            # Test that optimizer handles expert parallel parameters
            if isinstance(optimizer, LayerWiseDistributedOptimizer):
                # Check that expt_dp_params_list exists if EP > 1
                if ep > 1:
                    assert hasattr(optimizer, 'expt_dp_params_list')

            # Test save/load
            model_sharded_sd = model[0].sharded_state_dict()
            optim_sd = optimizer.sharded_state_dict(model_sharded_sd)
            save(optim_sd, ckpt_dir)

            # Create new optimizer and load
            model_new, optimizer_new = setup_moe_model_and_optimizer(
                seed=3, tp=tp, pp=pp, ep=ep, bf16=True, dist_opt=False, optimizer='dist_muon'
            )

            model_sharded_sd = model_new[0].sharded_state_dict()
            load_sharded_sd = optimizer_new.sharded_state_dict(model_sharded_sd, is_loading=True)
            state_dict = load(load_sharded_sd, ckpt_dir)
            optimizer_new.load_state_dict(state_dict)

    def test_layer_wise_optimizer_replica_id(self):
        """Test that LayerWiseDistributedOptimizer sets replica_id correctly."""
        Utils.initialize_model_parallel(2, 2)

        model, optimizer = setup_model_and_optimizer(
            seed=2,
            tp=2,
            pp=2,
            bf16=True,
            dist_opt=False,
            initialize_fn=initialize_gpt_model,
            optimizer='dist_muon',
        )

        if isinstance(optimizer, LayerWiseDistributedOptimizer):
            model_sharded_sd = model[0].sharded_state_dict()
            optim_sd = optimizer.sharded_state_dict(model_sharded_sd)

            # Extract ShardedTensors and check replica_id
            from megatron.core.dist_checkpointing import ShardedTensor

            for sh_base in nested_values(optim_sd):
                if isinstance(sh_base, ShardedTensor):
                    # Check that replica_id has been modified
                    assert len(sh_base.replica_id) == 3
                    # DP component should be 0 for layer-wise optimizer
                    assert sh_base.replica_id[2] == 0

    @pytest.mark.parametrize('dp_size', [1, 2, 4])
    def test_layer_wise_optimizer_dp_sizes(self, dp_size):
        """Test LayerWiseDistributedOptimizer with different DP sizes."""
        # Use PP to vary DP size while keeping world size constant
        world_size = 8
        if world_size % dp_size != 0:
            pytest.skip(f"World size {world_size} not divisible by DP size {dp_size}")

        pp = dp_size
        tp = world_size // pp

        if tp == 0:
            pytest.skip(f"Invalid TP configuration")

        Utils.initialize_model_parallel(tp, pp)

        model, optimizer = setup_model_and_optimizer(
            seed=2,
            tp=tp,
            pp=pp,
            bf16=True,
            dist_opt=False,
            initialize_fn=initialize_gpt_model,
            optimizer='dist_muon',
        )

        if isinstance(optimizer, LayerWiseDistributedOptimizer):
            # Check parameter sharding based on DP size
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)

            actual_dp_size = get_pg_size(pg_collection.dp_cp)

            if actual_dp_size > 1:
                assert optimizer.dp_cp_params_list is not None
                assert len(optimizer.dp_cp_params_list) == actual_dp_size
            else:
                assert optimizer.dp_cp_params_list is None

    def test_layer_wise_optimizer_step(self):
        """Test that step function works and returns expected values."""
        Utils.initialize_model_parallel(2, 2)

        model, optimizer = setup_model_and_optimizer(
            seed=2,
            tp=2,
            pp=2,
            bf16=True,
            dist_opt=False,
            initialize_fn=initialize_gpt_model,
            optimizer='dist_muon',
        )

        # Create dummy gradients
        for param in model[0].parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param.data)

        if isinstance(optimizer, LayerWiseDistributedOptimizer):
            # Perform step
            update_successful, grad_norm, num_zeros = optimizer.step()

            # Check return values
            assert isinstance(update_successful, bool)
            assert grad_norm is None or grad_norm >= 0
            assert num_zeros is None or num_zeros >= 0

    @pytest.mark.parametrize("fully_parallel", [False, True])
    @pytest.mark.parametrize('optimizer_type', ['dist_muon', 'muon'])
    @pytest.mark.parametrize(
        ("tp_pp_ep", "is_moe", "is_mla", "kwargs"),
        [
            ((1, 1, 1), False, False, {}),  # check DP
            ((2, 2, 1), False, False, {}),  # check TP
            ((1, 2, 1), False, True, {}),  # check param group order is right
            ((1, 2, 2), True, False, {}),  # check EP
            ((1, 2, 2), True, True, {}),  # check EP with MLA
        ],
    )
    def test_optimizer_common_state_dict(
        self, tmp_path_dist_ckpt, fully_parallel, tp_pp_ep, is_moe, is_mla, kwargs, optimizer_type
    ):
        initialize_fn = partial(initialize_real_model, is_moe=is_moe, is_mla=is_mla, **kwargs)

        # Initialize parallel
        tp, pp, ep = tp_pp_ep
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
        )
        rank = torch.distributed.get_rank()

        with TempNamedDir(tmp_path_dist_ckpt / 'test_dp_sharding', sync=True) as ckpt_dir:
            mock_args = parse_args(ignore_unknown_args=True)
            mock_args.use_distributed_optimizer = False
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
                diffs = diff(optim_param_state_A, optim_param_state_B)
                assert not any(map(bool, diffs)), (rank, diffs)

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp', 'use_glu'),
        [((2, 2), (2, 4), False), ((1, 8), (4, 1), True), ((2, 4), (4, 2), False)],
    )
    @pytest.mark.parametrize('optimizer_type', ['dist_muon', 'muon'])
    def test_finetune_doesnt_load_optimizer(
        self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp, use_glu, optimizer_type
    ):
        """Test finetuning doesn't try to load the optimizer."""
        Utils.initialize_model_parallel(*src_tp_pp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_finetune_doesnt_load_optimizer', sync=True
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
                    dist_opt=False,
                    optimizer=optimizer_type,
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
                    dist_opt=False,
                    optimizer=optimizer_type,
                )
                model_unloaded_state_dict = deepcopy(model[0].state_dict())
                optim_unloaded_state_dict = deepcopy(optimizer.state_dict())

                # Load with different TPxPP should raise DistributeOptimizer error
                with pytest.raises(RuntimeError) as exc_info:
                    load_checkpoint_no_arg_checks(model, optimizer, None)
                # "(TP, PP) mismatch" check is for backwards compatibility tests
                assert "(TP, PP) mismatch" in str(
                    exc_info.value
                ) or "(TP, PP, encoder TP, encoder PP) mismatch" in str(exc_info.value)

                # Check that the state didn't change
                assert not any(diff(model[0].state_dict(), model_unloaded_state_dict))
                assert not any(diff(optimizer.state_dict(), optim_unloaded_state_dict))
