# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedTensor
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.training.checkpointing import load_checkpoint
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    initialize_gpt_model,
    setup_model_and_optimizer,
    setup_moe_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils

GRAD_COLLECTIVES = ['reduce-scatter', 'all-reduce']


def check_equal(input_1, input_2, path="state_dict"):
    """Check if two inputs are equal, used for checking checkpointing."""
    if isinstance(input_1, dict) and isinstance(input_2, dict):
        assert input_1.keys() == input_2.keys(), path
        for key in input_1.keys():
            check_equal(input_1[key], input_2[key], f"{path}.{key}")
    elif isinstance(input_1, list) and isinstance(input_2, list):
        assert len(input_1) == len(input_2), path
        for i in range(len(input_1)):
            check_equal(input_1[i], input_2[i], f"{path}[{i}]")
    elif isinstance(input_1, torch.Tensor) and isinstance(input_2, torch.Tensor):
        assert torch.all(input_1 == input_2), f"{path}: Input 1: {input_1} != Input 2: {input_2}"
    elif type(input_1) != type(input_2):
        assert False, f"{path}: Input 1 type: {type(input_1)} != Input 2 type: {type(input_2)}"
    else:
        assert input_1 == input_2, f"{path}: Input 1: {input_1} != Input 2: {input_2}"


def zero_distopt_padding(sharded_state_dict):
    """Zero the intra-bucket padding entries of a DistOpt ``dp_reshardable`` state dict.

    The padding is saved from fresh ``torch.empty`` tensors, so two checkpoints of equal
    optimizer state differ there; zeroing it lets a bitwise comparison cover the real state.
    """
    if isinstance(sharded_state_dict, dict):
        padding = sharded_state_dict.get('padding')
        if isinstance(padding, LocalNonpersistentObject) and padding.obj:
            for value in sharded_state_dict.values():
                if isinstance(value, ShardedTensor):
                    value.data.zero_()
            return
        values = sharded_state_dict.values()
    elif isinstance(sharded_state_dict, list):
        values = sharded_state_dict
    else:
        return
    for value in values:
        zero_distopt_padding(value)


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
    config_kwargs.pop("pg_collection", None)
    config_kwargs.pop("config", None)
    default_config_kwargs.update(**config_kwargs)
    config_cls = MLATransformerConfig if is_mla else TransformerConfig
    transformer_config = config_cls(**default_config_kwargs)

    if is_moe:
        layer_spec = get_gpt_decoder_block_spec(
            transformer_config, use_transformer_engine=True, vp_stage=vp_stage
        )
    else:
        layer_spec = get_gpt_layer_with_transformer_engine_spec(multi_latent_attention=is_mla)
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


def _layer_wise_of(optimizer):
    """The LayerWiseDistributedOptimizer of an optimizer chain."""
    if isinstance(optimizer, LayerWiseDistributedOptimizer):
        return optimizer
    return next(
        sub_optimizer
        for sub_optimizer in optimizer.chained_optimizers
        if isinstance(sub_optimizer, LayerWiseDistributedOptimizer)
    )


def load_checkpoint_no_arg_checks(*args, **kwargs):
    with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
        return load_checkpoint(*args, restore_training_state=True, **kwargs)


class TestLayerWiseOptimizer:
    """Tests for LayerWiseDistributedOptimizer functionality."""

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(('tp', 'pp'), [(1, 1), (2, 2), (1, 4)])
    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_layer_wise_optimizer_save_load(self, tmp_path_dist_ckpt, tp, pp, grad_collective):
        """Test save/load of LayerWiseDistributedOptimizer checkpoints."""
        bf16 = True
        Utils.initialize_model_parallel(tp, pp)
        # The padded and compact layouts chain a DistOpt sibling, which needs the sharding
        # type that training passes (see _build_sharded_state_dict_metadata).
        metadata = {'distrib_optim_sharding_type': 'dp_reshardable'}

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_layer_wise_optimizer_A', sync=True
        ) as ckpt_dir_A:
            with TempNamedDir(
                tmp_path_dist_ckpt / 'test_layer_wise_optimizer_B', sync=True
            ) as ckpt_dir_B:
                # Create model and optimizer A
                model_A, optimizer_A = setup_model_and_optimizer(
                    seed=2,
                    tp=tp,
                    pp=pp,
                    bf16=bf16,
                    dist_opt=False,
                    initialize_fn=initialize_gpt_model,
                    optimizer='dist_muon',
                    grad_collective=grad_collective,
                )

                # Save checkpoint A
                model_sharded_sd_A = model_A[0].sharded_state_dict()
                optim_sd_A = optimizer_A.sharded_state_dict(model_sharded_sd_A, metadata=metadata)
                zero_distopt_padding(optim_sd_A)
                save(optim_sd_A, ckpt_dir_A)

                # Create model and optimizer B with different seed
                model_B, optimizer_B = setup_model_and_optimizer(
                    seed=3,
                    tp=tp,
                    pp=pp,
                    bf16=bf16,
                    dist_opt=False,
                    initialize_fn=initialize_gpt_model,
                    optimizer='dist_muon',
                    grad_collective=grad_collective,
                )

                # Load checkpoint A into optimizer B
                model_sharded_sd_B = model_B[0].sharded_state_dict()
                load_sharded_sd = optimizer_B.sharded_state_dict(
                    model_sharded_sd_B, is_loading=True, metadata=metadata
                )
                state_dict = load(load_sharded_sd, ckpt_dir_A)
                optimizer_B.load_state_dict(state_dict)

                # Save as checkpoint B
                optim_sd_B = optimizer_B.sharded_state_dict(model_sharded_sd_B, metadata=metadata)
                zero_distopt_padding(optim_sd_B)
                save(optim_sd_B, ckpt_dir_B)

                Utils.destroy_model_parallel()

                # Compare checkpoints
                Utils.initialize_model_parallel(1, 1)
                from megatron.core.dist_checkpointing import load_plain_tensors

                plain_sd_A = load_plain_tensors(ckpt_dir_A)
                plain_sd_B = load_plain_tensors(ckpt_dir_B)

                check_equal(plain_sd_A, plain_sd_B)

    @pytest.mark.parametrize(
        ('src_tp', 'src_pp', 'dest_tp', 'dest_pp'),
        [(1, 1, 2, 2), (2, 2, 1, 1), (4, 1, 1, 4), (1, 4, 4, 2)],
    )
    def test_layer_wise_optimizer_resharding(
        self, tmp_path_dist_ckpt, src_tp, src_pp, dest_tp, dest_pp
    ):
        """Test resharding of LayerWiseDistributedOptimizer across different TP/PP."""
        Utils.initialize_model_parallel(src_tp, src_pp)

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_layer_wise_resharding_A', sync=True
        ) as ckpt_dir:
            # Create and save with source configuration
            model_A, optimizer_A = setup_model_and_optimizer(
                seed=2,
                tp=src_tp,
                pp=src_pp,
                bf16=True,
                dist_opt=False,
                initialize_fn=initialize_gpt_model,
                optimizer='dist_muon',
            )

            # Resharding across TP/PP needs the model-space format of the sibling DistOpt.
            metadata = {'distrib_optim_sharding_type': 'fully_reshardable'}
            model_sharded_sd = model_A[0].sharded_state_dict()
            optim_sd = optimizer_A.sharded_state_dict(model_sharded_sd, metadata=metadata)
            save(optim_sd, ckpt_dir)

            Utils.destroy_model_parallel()

            # Load with destination configuration
            Utils.initialize_model_parallel(dest_tp, dest_pp)
            model_B, optimizer_B = setup_model_and_optimizer(
                seed=3,
                tp=dest_tp,
                pp=dest_pp,
                bf16=True,
                dist_opt=False,
                initialize_fn=initialize_gpt_model,
                optimizer='dist_muon',
            )

            model_sharded_sd = model_B[0].sharded_state_dict()
            load_sharded_sd = optimizer_B.sharded_state_dict(
                model_sharded_sd, is_loading=True, metadata=metadata
            )

            state_dict = load(load_sharded_sd, ckpt_dir)
            optimizer_B.load_state_dict(state_dict)

    @pytest.mark.parametrize(('tp', 'pp', 'ep'), [(1, 1, 2), (2, 1, 4), (1, 2, 4)])
    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_layer_wise_optimizer_with_moe(self, tmp_path_dist_ckpt, tp, pp, ep, grad_collective):
        """Test LayerWiseDistributedOptimizer with MoE models."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
        )

        metadata = {'distrib_optim_sharding_type': 'dp_reshardable'}
        with TempNamedDir(tmp_path_dist_ckpt / 'test_layer_wise_moe', sync=True) as ckpt_dir:
            # Create MoE model with optimizer
            model, optimizer = setup_moe_model_and_optimizer(
                seed=2,
                tp=tp,
                pp=pp,
                ep=ep,
                bf16=True,
                dist_opt=False,
                optimizer='dist_muon',
                grad_collective=grad_collective,
            )

            # Test that optimizer handles expert parallel parameters
            if ep > 1:
                # Check that expt_dp_params_list exists if EP > 1
                assert hasattr(_layer_wise_of(optimizer), 'expt_dp_params_list')

            # Test save/load
            model_sharded_sd = model[0].sharded_state_dict()
            optim_sd = optimizer.sharded_state_dict(model_sharded_sd, metadata=metadata)
            save(optim_sd, ckpt_dir)

            # Create new optimizer and load
            model_new, optimizer_new = setup_moe_model_and_optimizer(
                seed=3,
                tp=tp,
                pp=pp,
                ep=ep,
                bf16=True,
                dist_opt=False,
                optimizer='dist_muon',
                grad_collective=grad_collective,
            )

            model_sharded_sd = model_new[0].sharded_state_dict()
            load_sharded_sd = optimizer_new.sharded_state_dict(
                model_sharded_sd, is_loading=True, metadata=metadata
            )
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

        model_sharded_sd = model[0].sharded_state_dict()
        optim_sd = _layer_wise_of(optimizer).sharded_state_dict(model_sharded_sd)

        # Extract ShardedTensors and check replica_id
        from megatron.core.dist_checkpointing import ShardedTensor

        for sh_base in nested_values(optim_sd):
            if isinstance(sh_base, ShardedTensor):
                # Check that replica_id has been modified
                assert len(sh_base.replica_id) == 3
                # DP component should be 0 for layer-wise optimizer
                assert sh_base.replica_id[2] == 0
