# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
from copy import deepcopy
from functools import partial
from unittest import mock
from unittest.mock import patch

import pytest
import torch
from torch.optim import Adam

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff, nested_values
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    optim_state_to_sharding_state,
)
from megatron.core.dist_checkpointing.utils import add_prefix_for_sharding, extract_sharded_tensors
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.utils import is_torch_min_version
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


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 16, 3)
        self.proj = torch.nn.Linear(8, 5)
        self.config = TransformerConfig(
            hidden_size=8, num_attention_heads=1, num_layers=1, bf16=True
        )

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
    def __init__(self, pp_separate_model: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(
            5, 64 // parallel_state.get_tensor_model_parallel_world_size(), bias=False
        )
        self.config = TransformerConfig(
            hidden_size=8, num_attention_heads=1, num_layers=1, bf16=True
        )
        self.pp_separate_model = pp_separate_model

    def sharded_state_dict(self):
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if self.pp_separate_model:
            pp_replica_id = 0
        else:
            pp_replica_id = pp_rank
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
                    pp_replica_id,
                    0,
                    parallel_state.get_data_parallel_rank(with_context_parallel=True),
                )
            ),
        )
        sharded_state_dict['linear.weight'] = apply_swiglu_sharded_factory(
            sharded_state_dict['linear.weight'], ()
        )
        if self.pp_separate_model:
            add_prefix_for_sharding(sharded_state_dict, f'pp_rank_{pp_rank}.')
        return sharded_state_dict


class SwigluFactoryModel(torch.nn.Module):
    def __init__(self, pp_separate_model: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(5, 64, bias=False)
        self.config = TransformerConfig(
            hidden_size=8, num_attention_heads=1, num_layers=1, bf16=True
        )
        self.pp_separate_model = pp_separate_model

    def sharded_state_dict(self):
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if self.pp_separate_model:
            pp_replica_id = 0
        else:
            pp_replica_id = pp_rank
        sharded_state_dict = self.state_dict(keep_vars=True)
        sharded_state_dict['linear.weight'] = ShardedTensor.from_rank_offsets(
            'linear.weight',
            sharded_state_dict['linear.weight'],
            replica_id=(
                (
                    pp_replica_id,
                    parallel_state.get_tensor_model_parallel_rank(),
                    parallel_state.get_data_parallel_rank(with_context_parallel=True),
                )
            ),
        )
        if self.pp_separate_model:
            add_prefix_for_sharding(sharded_state_dict, f'pp_rank_{pp_rank}.')
        return sharded_state_dict


class Model1dFlattenTensor(torch.nn.Module):
    """This model is used to test whether a 1d flatten tensor can be correctly
    transformed into torch dist-ckpt form
    """

    def __init__(self):
        super().__init__()
        self.config = TransformerConfig(
            hidden_size=128, num_attention_heads=1, num_layers=1, bf16=True
        )
        weight_size_per_rank = (
            self.config.hidden_size // parallel_state.get_tensor_model_parallel_world_size()
        )
        self.weight_1d = torch.nn.Parameter(torch.randn(weight_size_per_rank))

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


def get_param_state_dp_zero(optimizer):
    if isinstance(optimizer, ChainedOptimizer):
        assert len(optimizer.chained_optimizers) == 1
        optim_param_state_A = optimizer.chained_optimizers[0].get_parameter_state_dp_zero(
            use_gloo_comm=False
        )
    else:
        optim_param_state_A = optimizer.get_parameter_state_dp_zero(use_gloo_comm=False)
    return optim_param_state_A


class TestOptimizer:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_optimizer_params(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1, 1)
        model = Model()
        # Force optimizer state initialization
        for p in model.parameters():
            p.grad = torch.ones_like(p.data)
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


def initialize_pp_agnostic_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    return SwigluFactoryModel(False)


def initialize_pp_agnostic_gpt_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    return initialize_gpt_model(False, False, seed=seed, **config_kwargs)


def initialize_small_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    return SwigluFactoryModel()


def initialize_1d_flatten_tensor_model(
    pre_process=True, post_process=True, seed=0, **config_kwargs
):
    # This model is used to test whether a 1d flatten tensor can be correctly
    # transformed into torch dist-ckpt form
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    return Model1dFlattenTensor()


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
    # These kwargs are passed through training.get_model for model construction,
    # but are not part of TransformerConfig; strip them before building config.
    config_kwargs.pop("pg_collection", None)
    config_kwargs.pop("config", None)

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


class TestDistributedOptimizer:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("fully_parallel", [False, True])
    @pytest.mark.parametrize(
        ("tp_pp_ep", "is_moe", "is_mla", "test_step", "kwargs"),
        [
            ((2, 2, 1), False, False, False, {}),  # check TP
            ((1, 2, 1), False, False, True, {}),  # check "step" is synced
            ((1, 2, 1), False, True, False, {}),  # check param group order is right
            (
                (1, 8, 1),
                False,
                False,
                False,
                {
                    "account_for_embedding_in_pipeline_split": True,
                    "account_for_loss_in_pipeline_split": True,
                },
            ),  # check embedding standalone
            (
                (1, 2, 2),
                True,
                False,
                True,
                {"moe_layer_freq": [0, 0, 0, 1, 1, 1]},
            ),  # check moe not on all ranks (case 1)
            (
                (1, 2, 2),
                True,
                False,
                True,
                {"moe_layer_freq": [1, 1, 1, 0, 0, 0]},
            ),  # check moe not on all ranks (case 2)
        ],
    )
    def test_optimizer_common_state_dict(
        self, tmp_path_dist_ckpt, fully_parallel, tp_pp_ep, is_moe, is_mla, test_step, kwargs
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
            mock_args.use_distributed_optimizer = True
            with mock.patch('megatron.training.checkpointing.get_args', new=lambda: mock_args):
                # Initialize model and optimizer A
                if is_moe:
                    model, optimizer_A = setup_moe_model_and_optimizer(
                        seed=2, tp=tp, pp=pp, ep=ep, initialize_fn=initialize_fn
                    )
                else:
                    model, optimizer_A = setup_model_and_optimizer(
                        seed=2, tp=tp, pp=pp, initialize_fn=initialize_fn
                    )
                if test_step:
                    # Simulate "step" not set in some of the param groups on rank 0.
                    # TE FusedAdam may have "step" not set in some of the param groups on some ranks.
                    for i, param_group in enumerate(
                        optimizer_A.chained_optimizers[0].optimizer.param_groups
                    ):
                        if rank > 0 or i == 0:
                            param_group['step'] = 1234

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
                        seed=3, tp=tp, pp=pp, ep=ep, initialize_fn=initialize_fn
                    )
                else:
                    model, optimizer_B = setup_model_and_optimizer(
                        seed=3, tp=tp, pp=pp, initialize_fn=initialize_fn
                    )
                # Load optimizer B from checkpoint
                load_checkpoint_no_arg_checks(model, optimizer_B, None)
                if test_step:
                    # Complete "step" for comparison
                    for i, param_group in enumerate(
                        optimizer_A.chained_optimizers[0].optimizer.param_groups
                    ):
                        if rank == 0 and i > 0:
                            param_group['step'] = 1234
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
    def test_finetune_doesnt_load_optimizer(
        self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp, use_glu
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

                # Now test the same with a `finetune` flag
                mock_args.finetune = True
                load_checkpoint_no_arg_checks(model, optimizer, None)

                # Model weights should be different, but optimizer state is unchanged
                diffs = diff(model[0].state_dict(), model_unloaded_state_dict)
                # diffs[0] and diffs[1] is structural diff, diffs[2] is values diff -
                # we expect only values diff
                assert not diffs[0] and not diffs[1] and diffs[2]
                assert not any(diff(optimizer.state_dict(), optim_unloaded_state_dict))

                # ... or `no_load_optim` flag
                model, optimizer = setup_model_and_optimizer(
                    seed=3,
                    tp=dest_tp_pp[0],
                    pp=dest_tp_pp[1],
                    initialize_fn=partial(initialize_gpt_model, use_glu=use_glu),
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
                assert not any(diff(optimizer.state_dict(), optim_unloaded_state_dict))

    @pytest.mark.skipif(
        not is_torch_min_version("2.6a0"), reason="dp_reshardable requires PyTorch 2.6a0 or later"
    )
    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp', 'src_bucket_pad_divisor', 'dest_bucket_pad_divisor'),
        [
            # PP must be decreasing
            # Note: PP must be > 1 if TP <= 2 because of empty buckets otherwise
            ((1, 2), (1, 2), 8 * 7, 8 * 5),
            ((2, 4), (2, 4), 128, 128),
            ((8, 1), (8, 1), 8, 4 * 11),
            # DP resharding:
            ((4, 2), (4, 1), 8 * 7, 8 * 5),
            ((2, 4), (2, 2), 128, 128),
            ((1, 4), (1, 2), 8, 4 * 11),
            ((1, 8), (1, 4), 8 * 7, 8 * 5),
            ((1, 8), (1, 2), 128, 128),
        ],
    )
    def test_bucket_space_optimizer_save_load(
        self,
        tmp_path_dist_ckpt,
        src_tp_pp,
        dest_tp_pp,
        src_bucket_pad_divisor,
        dest_bucket_pad_divisor,
    ):
        """Test DistOpt save/load with dp_reshardable format.

        Since unit test have a fixed world size and "bucket_space" format is
        only DP-reshardable, we can't simply change DP. The trick is to use PP rank
        agnostic model and decrease PP for load - some DP groups will be missing
        but the common subset is enough to test correctness.
        """
        Utils.initialize_model_parallel(*src_tp_pp)
        src_num_dp_groups = src_tp_pp[1] * src_tp_pp[0]
        dest_num_dp_groups = dest_tp_pp[1] * dest_tp_pp[0]
        assert (
            dest_num_dp_groups <= src_num_dp_groups
        ), 'This test cant be run with increasing number of DP groups'

        with (
            TempNamedDir(
                tmp_path_dist_ckpt / 'test_bucket_state_optimizer_save_load_A', sync=True
            ) as ckpt_dir_A,
            TempNamedDir(
                tmp_path_dist_ckpt / 'test_bucket_state_optimizer_save_load_B', sync=True
            ) as ckpt_dir_B,
        ):
            # Init model and optimizer with "src" bucket padding
            with patch('megatron.core.distributed.param_and_grad_buffer.math.lcm') as lcm_mock:
                lcm_mock.return_value = src_bucket_pad_divisor
                assert len(lcm_mock.mock_calls) == 0
                model_A, optimizer_A = setup_model_and_optimizer(
                    seed=2,
                    tp=src_tp_pp[0],
                    pp=src_tp_pp[1],
                    bf16=True,
                    dist_opt=True,
                    initialize_fn=initialize_pp_agnostic_model,
                )
                assert len(lcm_mock.mock_calls) > 1

            metadata = {'distrib_optim_sharding_type': 'dp_reshardable'}

            model_sharded_sd = model_A[0].sharded_state_dict()
            optim_sd = optimizer_A.sharded_state_dict(model_sharded_sd, metadata=metadata)
            per_bucket_numel_unpadded_A = optim_sd['param_state']['per_bucket_numel_unpadded'].data
            save(optim_sd, ckpt_dir_A)
            Utils.destroy_model_parallel()

            # Load checkpoint A with different PP (and therefore DP) and save as checkpoint B
            Utils.initialize_model_parallel(*dest_tp_pp)
            dest_dp_group_idx = torch.distributed.get_rank(
                parallel_state.get_model_parallel_group()
            )
            # Init model and optimizer with "dest" bucket padding
            with patch('megatron.core.distributed.param_and_grad_buffer.math.lcm') as lcm_mock:
                lcm_mock.return_value = dest_bucket_pad_divisor
                assert len(lcm_mock.mock_calls) == 0
                model_B, optimizer_B = setup_model_and_optimizer(
                    seed=3,
                    tp=dest_tp_pp[0],
                    pp=dest_tp_pp[1],
                    bf16=True,
                    dist_opt=True,
                    initialize_fn=initialize_pp_agnostic_model,
                )
                assert len(lcm_mock.mock_calls) > 1

            model_sharded_sd = model_B[0].sharded_state_dict()
            load_sharded_state_dict = optimizer_B.sharded_state_dict(
                model_sharded_sd, metadata=metadata, is_loading=True
            )
            state_dict, missing_keys, unexpected_keys = load(
                load_sharded_state_dict, ckpt_dir_A, strict=StrictHandling.RETURN_ALL
            )

            # Check that because of decreasing PP, some DP groups were not read.
            assert not unexpected_keys
            missing_dp_groups = set()
            for missing_key in missing_keys:
                match = re.search(r'dp_group_idx_(\d+)', missing_key)
                assert match is not None
                missing_dp_groups.add(int(match.group(1)))

            assert missing_dp_groups == set(range(dest_num_dp_groups, src_num_dp_groups))

            # Save optimizer B checkpoint to compare them
            optimizer_B.load_state_dict(state_dict)
            model_sharded_sd = model_B[0].sharded_state_dict()
            optim_sd = optimizer_B.sharded_state_dict(model_sharded_sd, metadata=metadata)
            per_bucket_numel_unpadded_B = optim_sd['param_state']['per_bucket_numel_unpadded'].data
            save(optim_sd, ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            # Ckpt A has more keys because of larger PP.
            # Each rank tests correctness within its DP group, and only unpadded tensor part.
            assert per_bucket_numel_unpadded_A == per_bucket_numel_unpadded_B
            assert len(per_bucket_numel_unpadded_A) == 1  # Assuming a simple case with one buffer
            per_bucket_numel_unpadded_A = per_bucket_numel_unpadded_A[0]
            assert len(per_bucket_numel_unpadded_A) == 1  # Assuming a simple case with one dtype
            per_bucket_numel_unpadded = next(iter(per_bucket_numel_unpadded_A.values()))
            Utils.initialize_model_parallel(1, 1)
            plain_state_dict_A = load_plain_tensors(ckpt_dir_A)
            plain_state_dict_B = load_plain_tensors(ckpt_dir_B)
            torch.distributed.barrier()

            # We test only the `plain_state_dict_B` keys because of decreasing PP
            for key in list(plain_state_dict_B.keys()):
                if 'per_bucket_numel' in key or 'param_state_sharding_type' in key:
                    del plain_state_dict_A[key]
                    del plain_state_dict_B[key]
                    continue
                match = re.search(r'dp_group_idx_(\d+).+bucket_idx_(\d+)', key)
                assert match is not None, key
                dp_group_idx = int(match.group(1))
                bucket_idx = int(match.group(2))
                if dp_group_idx != dest_dp_group_idx:
                    del plain_state_dict_A[key]
                    del plain_state_dict_B[key]
                else:
                    numel_unpadded = per_bucket_numel_unpadded[bucket_idx]
                    assert len(plain_state_dict_A[key]) == numel_unpadded
                    assert len(plain_state_dict_B[key]) == numel_unpadded

            only_left, only_right, mismatch = diff(plain_state_dict_A, plain_state_dict_B)

            missing_tensors = set(
                key
                for key in missing_keys
                if 'per_bucket_numel' not in key and not key.endswith('.optimizer')
            )
            assert set(key[0] for key in only_left) == missing_tensors
            assert not only_right
            assert not mismatch

    @pytest.mark.skipif(
        not is_torch_min_version("2.6a0"), reason="dp_reshardable requires PyTorch 2.6a0 or later"
    )
    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp', 'sharding_type', 'mem_efficient'),
        [
            # PP must be decreasing
            # Note: PP must be > 1 if TP <= 2 because of empty buckets otherwise
            ((2, 4), (2, 4), 'fully_reshardable', False),
            ((4, 2), (4, 2), 'dp_reshardable', None),
            # DP resharding:
            ((4, 2), (4, 1), 'dp_reshardable', None),
            ((2, 4), (2, 2), 'fully_reshardable', False),
            ((2, 4), (2, 2), 'fully_reshardable', True),
        ],
    )
    @pytest.mark.parametrize("initalize_fn", [initialize_pp_agnostic_model])
    def test_nonreshardable_optimizer_save_load(
        self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp, initalize_fn, sharding_type, mem_efficient
    ):
        """Generalization of the test above for different formats.

        This time we don't load "plain" tensors from the checkpoint to compare.
        Instead, we use `get_param_state_dp_zero` method to have common representation
        irrespective of DP size.

        This test requires src and dest optimizers to be on the same rank.
        The `test_model_parallel_dp_group_idx_preservation` test checks that
        there is at least one testing rank for each DP group, given the 'tp-pp-dp'
        parallel state order.
        """
        Utils.initialize_model_parallel(*src_tp_pp, order='tp-pp-dp')
        src_num_dp_groups = src_tp_pp[1] * src_tp_pp[0]
        dest_num_dp_groups = dest_tp_pp[1] * dest_tp_pp[0]
        src_dp_group_idx = torch.distributed.get_rank(parallel_state.get_model_parallel_group())
        assert (
            dest_num_dp_groups <= src_num_dp_groups
        ), 'This test cant be run with increasing number of DP groups'

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_nonreshardable_optimizer_save_load', sync=True
        ) as ckpt_dir_A:
            model_A, optimizer_A = setup_model_and_optimizer(
                seed=2,
                tp=src_tp_pp[0],
                pp=src_tp_pp[1],
                bf16=True,
                dist_opt=True,
                initialize_fn=initalize_fn,
            )

            metadata = {
                'distrib_optim_sharding_type': sharding_type,
                'distrib_optim_fully_reshardable_mem_efficient': mem_efficient,
            }

            model_sharded_sd = model_A[0].sharded_state_dict()
            optim_sd = optimizer_A.sharded_state_dict(model_sharded_sd, metadata=metadata)
            save(optim_sd, ckpt_dir_A)

            dp_zero_optim_A = get_param_state_dp_zero(optimizer_A)
            Utils.destroy_model_parallel()

            # Load checkpoint A with different TP/PP and save as checkpoint B
            Utils.initialize_model_parallel(*dest_tp_pp, order='tp-pp-dp')
            dest_dp_group_idx = torch.distributed.get_rank(
                parallel_state.get_model_parallel_group()
            )
            same_dp_group = src_dp_group_idx == dest_dp_group_idx
            model_B, optimizer_B = setup_model_and_optimizer(
                seed=3,
                tp=dest_tp_pp[0],
                pp=dest_tp_pp[1],
                bf16=True,
                dist_opt=True,
                initialize_fn=initalize_fn,
            )
            # Before checkpoint load the state is expected to differ
            dp_zero_optim_B = get_param_state_dp_zero(optimizer_B)
            assert not self.check_equal_dp_zero_state(
                dp_zero_optim_A, dp_zero_optim_B, same_dp_group
            )

            model_sharded_sd = model_B[0].sharded_state_dict()
            load_sharded_state_dict = optimizer_B.sharded_state_dict(
                model_sharded_sd, metadata=metadata, is_loading=True
            )

            state_dict, missing_keys, unexpected_keys = load(
                load_sharded_state_dict, ckpt_dir_A, strict=StrictHandling.RETURN_ALL
            )
            assert not unexpected_keys
            missing_dp_groups = set()
            for missing_key in missing_keys:
                match = re.search(r'dp_group_idx_(\d+)', missing_key)
                assert match is not None
                missing_dp_groups.add(int(match.group(1)))

            optimizer_B.load_state_dict(state_dict)
            dp_zero_optim_B = get_param_state_dp_zero(optimizer_B)

            assert self.check_equal_dp_zero_state(
                dp_zero_optim_A, dp_zero_optim_B, same_dp_group, raise_if_different=True
            )

    def check_equal_dp_zero_state(
        self, dp_zero_state_A, dp_zero_state_B, same_dp_group, raise_if_different=False
    ):
        if same_dp_group and parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0:
            diffs = diff(dp_zero_state_A, dp_zero_state_B)
            is_equal = not any(map(bool, diffs))
        else:
            diffs = None
            is_equal = True

        all_equal = torch.tensor(int(is_equal), device='cuda')
        torch.distributed.all_reduce(all_equal, op=torch.distributed.ReduceOp.MIN)
        if bool(all_equal.item()):
            return True
        else:
            if raise_if_different:
                raise RuntimeError(f'[{Utils.rank}] {diffs}')
            return False

    @pytest.mark.parametrize('tp_pp', [(2, 4), (4, 2), (1, 1), (2, 1), (1, 8)])
    def test_model_parallel_rank_order(self, tp_pp):
        """Verifies that DP group idx is `PP rank * TP size + TP rank`."""
        Utils.initialize_model_parallel(*tp_pp, order='tp-pp-dp')
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        dp_group_idx = torch.distributed.get_rank(parallel_state.get_model_parallel_group())

        assert pp_rank * tp_size + tp_rank == dp_group_idx

    @pytest.mark.parametrize(
        ('src_pp', 'dest_pp'),
        [
            # PP must be decreasing
            (8, 1),
            (4, 1),
            (2, 1),
            (1, 1),
            (8, 2),
            (4, 2),
            (2, 2),
            (8, 4),
            (4, 4),
            (8, 4),
        ],
    )
    @pytest.mark.parametrize('tp', [1, 2, 4, 8])
    def test_model_parallel_dp_group_idx_preservation(self, tp, src_pp, dest_pp):
        """For each dst DP group, test there is at least one DP 0 rank both in the src and dest group.

        For this condition to hold, `parallel_state` must be initialized with 'tp-pp-dp' order.
        """
        assert src_pp >= dest_pp, 'This test is only for decreasing PP'
        if tp * src_pp > Utils.world_size:
            pytest.skip(f'TP ({tp}) * PP ({src_pp}) > {Utils.world_size}')

        Utils.initialize_model_parallel(tp, src_pp, order='tp-pp-dp')
        src_dp_group_idx = torch.distributed.get_rank(parallel_state.get_model_parallel_group())
        Utils.initialize_model_parallel(tp, dest_pp, order='tp-pp-dp')
        dest_dp_group_idx = torch.distributed.get_rank(parallel_state.get_model_parallel_group())
        num_dest_dp_groups = tp * dest_pp

        is_dp_rank_zero = parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
        if src_dp_group_idx == dest_dp_group_idx and is_dp_rank_zero:
            same_dp_group_idx = src_dp_group_idx
        else:
            same_dp_group_idx = None

        same_groups = [None] * Utils.world_size
        torch.distributed.all_gather_object(same_groups, same_dp_group_idx)

        same_groups = set(g for g in same_groups if g is not None)
        # Check each dst group has at least 1 rank both in src and dest
        assert same_groups == set(range(num_dest_dp_groups))


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
            list = preprocessed_optimzier_common_dict['optimizer']['param_groups']
            for dict_item in list:
                del dict_item['wd_mult']
            return preprocessed_optimzier_common_dict

        Utils.initialize_model_parallel(*src_tp_pp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_A', sync=True
        ) as ckpt_dir_A:
            with TempNamedDir(
                tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_B', sync=True
            ) as ckpt_dir_B:

                model_A, optimizer_A = setup_model_and_optimizer(
                    seed=2,
                    tp=src_tp_pp[0],
                    pp=src_tp_pp[1],
                    initialize_fn=initialize_small_model,
                    bf16=False,
                )

                metadata = {'distrib_optim_sharding_type': 'fully_reshardable'}

                save(
                    optimizer_A.sharded_state_dict(
                        model_A[0].sharded_state_dict(), metadata=metadata
                    ),
                    ckpt_dir_A,
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
                )
                load_sharded_state_dict = optimizer_B.sharded_state_dict(
                    model_B[0].sharded_state_dict(), is_loading=True, metadata=metadata
                )
                state_dict = load(load_sharded_state_dict, ckpt_dir_A)

                optimizer_B.load_state_dict(state_dict)
                save(
                    optimizer_B.sharded_state_dict(
                        model_B[0].sharded_state_dict(), metadata=metadata
                    ),
                    ckpt_dir_B,
                )
                Utils.destroy_model_parallel()

                # Test both checkpoints are equal
                Utils.initialize_model_parallel(1, 1)
                plain_state_dict_A = load_plain_tensors(ckpt_dir_A)
                plain_state_dict_B = load_plain_tensors(ckpt_dir_B)
                diffs = diff(plain_state_dict_A, plain_state_dict_B)
                assert not any(map(bool, diffs)), diffs


class TestOptimizerResharding:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ('use_dist_opt', 'bf16', 'fully_parallel'),
        (
            (False, True, False),  # regular BF16
            (True, True, False),  # DistOpt BF16
            (True, True, True),  # DistOpt BF16
            (False, False, False),  # FP32
        ),
    )
    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp'),
        [((2, 4), (2, 4)), ((2, 4), (2, 2)), ((2, 4), (4, 2)), ((8, 1), (1, 2))],
    )
    @pytest.mark.parametrize(
        "initialize_fn", [initialize_gpt_model, initialize_1d_flatten_tensor_model]
    )
    def test_optimizer_resharding(
        self,
        tmp_path_dist_ckpt,
        src_tp_pp,
        dest_tp_pp,
        use_dist_opt,
        bf16,
        initialize_fn,
        fully_parallel,
    ):
        Utils.initialize_model_parallel(*src_tp_pp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_A', sync=False
        ) as ckpt_dir_A:
            with TempNamedDir(
                tmp_path_dist_ckpt / 'test_fp32_optimizer_state_dict_B', sync=False
            ) as ckpt_dir_B:
                model_A, optimizer_A = setup_model_and_optimizer(
                    seed=2,
                    tp=src_tp_pp[0],
                    pp=src_tp_pp[1],
                    bf16=bf16,
                    dist_opt=use_dist_opt,
                    initialize_fn=initialize_fn,
                )

                metadata = {'distrib_optim_sharding_type': 'fully_reshardable'}

                save(
                    optimizer_A.sharded_state_dict(
                        model_A[0].sharded_state_dict(), metadata=metadata
                    ),
                    ckpt_dir_A,
                )
                Utils.destroy_model_parallel()

                # Load checkpoint A with different TP/PP and save as checkpoint B
                Utils.initialize_model_parallel(*dest_tp_pp)
                model_B, optimizer_B = setup_model_and_optimizer(
                    seed=3,
                    tp=dest_tp_pp[0],
                    pp=dest_tp_pp[1],
                    bf16=bf16,
                    dist_opt=use_dist_opt,
                    initialize_fn=initialize_fn,
                )
                load_sharded_state_dict = optimizer_B.sharded_state_dict(
                    model_B[0].sharded_state_dict(), metadata=metadata, is_loading=True
                )
                state_dict = load(load_sharded_state_dict, ckpt_dir_A)

                optimizer_B.load_state_dict(state_dict)
                save(
                    optimizer_B.sharded_state_dict(
                        model_B[0].sharded_state_dict(), metadata=metadata
                    ),
                    ckpt_dir_B,
                )
                Utils.destroy_model_parallel()

                # Test both checkpoints are equal
                Utils.initialize_model_parallel(1, 1)
                plain_state_dict_A = load_plain_tensors(ckpt_dir_A)
                plain_state_dict_B = load_plain_tensors(ckpt_dir_B)
                diffs = diff(plain_state_dict_A, plain_state_dict_B)
                assert not any(map(bool, diffs)), diffs

    @pytest.mark.parametrize('fully_parallel', (False, True))
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
        use_te,
        use_grouped_mlp,
        use_glu,
        fully_parallel,
    ):
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
                    bf16=True,
                    dist_opt=True,
                    use_te=use_te,
                    use_grouped_mlp=use_grouped_mlp,
                    use_glu=use_glu,
                )

                metadata = {'distrib_optim_sharding_type': 'fully_reshardable'}

                save(
                    optimizer_A.sharded_state_dict(
                        model_A[0].sharded_state_dict(), metadata=metadata
                    ),
                    ckpt_dir_A,
                )
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
                    bf16=True,
                    dist_opt=True,
                    use_te=use_te,
                    use_grouped_mlp=use_grouped_mlp,
                    use_glu=use_glu,
                )
                load_sharded_state_dict = optimizer_B.sharded_state_dict(
                    model_B[0].sharded_state_dict(), metadata=metadata, is_loading=True
                )
                state_dict = load(load_sharded_state_dict, ckpt_dir_A)

                optimizer_B.load_state_dict(state_dict)
                save(
                    optimizer_B.sharded_state_dict(
                        model_B[0].sharded_state_dict(), metadata=metadata
                    ),
                    ckpt_dir_B,
                )
                Utils.destroy_model_parallel()

                # Test both checkpoints are equal
                Utils.initialize_model_parallel(1, 1)
                plain_state_dict_A = load_plain_tensors(ckpt_dir_A)
                plain_state_dict_B = load_plain_tensors(ckpt_dir_B)
                diffs = diff(plain_state_dict_A, plain_state_dict_B)
                assert not any(map(bool, diffs)), diffs
                Utils.destroy_model_parallel()
