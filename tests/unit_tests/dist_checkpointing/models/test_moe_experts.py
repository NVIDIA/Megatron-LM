# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def initialize_expert_layer(seed, glu=True, expert_type='sequential', **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    num_moe_experts = 8
    num_local_experts = num_moe_experts // parallel_state.get_expert_model_parallel_world_size()
    default_config_kwargs = dict(
        num_layers=pp_size,
        hidden_size=12,
        num_attention_heads=4,
        num_moe_experts=num_moe_experts,
        use_cpu_initialization=True,
        gated_linear_unit=glu,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=num_moe_experts, moe_grouped_gemm=(expert_type != 'sequential')
    )
    if expert_type == 'grouped':
        model = GroupedMLP(num_local_experts, transformer_config)
    elif expert_type == 'te_grouped':
        model = TEGroupedMLP(
            num_local_experts,
            transformer_config,
            transformer_layer_spec.submodules.mlp.submodules.experts,
        )
    elif expert_type == 'sequential':
        model = SequentialMLP(
            num_local_experts,
            transformer_config,
            transformer_layer_spec.submodules.mlp.submodules.experts,
        )
    else:
        raise ValueError('expert_type can only be one of ["sequential", "grouped", "te_grouped"]')
    return model


def get_pp_offsets():
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    return ((0, pp_rank, pp_size),)


expert_type = ['sequential', 'grouped']
src_dest_expert_type = [('sequential', 'grouped'), ('grouped', 'sequential')]
if is_te_min_version("1.9.0.dev0"):
    expert_type.append('te_grouped')
    src_dest_expert_type.append(('sequential', 'te_grouped'))
    src_dest_expert_type.append(('te_grouped', 'sequential'))


class TestExpertLayerReconfiguration:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "use_fpsl,src_tp_pp_exp,dest_tp_pp_exp,use_glu",
        [
            # changing PP is impossible because the number of layers must be the same
            (False, (2, 4, 1), (2, 4, 1), False),
            (True, (2, 4, 1), (2, 4, 1), False),
            (False, (1, 1, 1), (1, 1, 1), False),
            (True, (1, 1, 1), (1, 1, 4), False),
            (False, (1, 1, 8), (1, 1, 2), False),
            (False, (2, 2, 2), (4, 2, 1), False),
            (True, (1, 1, 4), (8, 1, 1), False),
            (False, (1, 8, 1), (1, 8, 1), False),
            (False, (1, 1, 4), (2, 1, 1), False),
            (False, (1, 1, 1), (1, 1, 1), True),
            (False, (1, 1, 1), (1, 1, 4), True),
            (True, (1, 1, 1), (2, 1, 1), True),
            (False, (1, 1, 4), (8, 1, 1), True),
        ],
    )
    @pytest.mark.failing_on_rocm
    @pytest.mark.parametrize("expert_type", expert_type)
    def test_parallel_reconfiguration_e2e(
        self, tmp_path_dist_ckpt, src_tp_pp_exp, dest_tp_pp_exp, use_glu, use_fpsl, expert_type
    ):
        """Test model saving and loading with different TP/PP/expert parallelism"""
        src_tp, src_pp, src_exp = src_tp_pp_exp
        dest_tp, dest_pp, dest_exp = dest_tp_pp_exp
        if expert_type == 'grouped':
            add_bias_linear = False
        else:
            add_bias_linear = True
        # Save checkpoint A
        Utils.initialize_model_parallel(src_tp, src_pp, expert_model_parallel_size=src_exp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_expert_layer_reconfiguration_model_A'
        ) as ckpt_dir_A, TempNamedDir(
            tmp_path_dist_ckpt / 'test_expert_layer_reconfiguration_model_B'
        ) as ckpt_dir_B:
            model_A = initialize_expert_layer(
                1, use_glu, expert_type, add_bias_linear=add_bias_linear
            )
            sharded_state_dict = model_A.sharded_state_dict(sharded_offsets=get_pp_offsets())

            save_strategy = get_default_save_sharded_strategy()
            if use_fpsl:
                save_strategy = FullyParallelSaveStrategyWrapper(
                    save_strategy,
                    parallel_state.get_data_parallel_group(with_context_parallel=True),
                    True,
                )
            save(sharded_state_dict, ckpt_dir_A, save_strategy)
            Utils.destroy_model_parallel()

            # Load checkpoint A with different TP/PP/expert and save as checkpoint B
            # No FPS this time, only FPL
            Utils.initialize_model_parallel(dest_tp, dest_pp, expert_model_parallel_size=dest_exp)
            model_B = initialize_expert_layer(
                1, use_glu, expert_type, add_bias_linear=add_bias_linear
            )
            if use_fpsl:
                load_strategy = get_default_load_sharded_strategy(ckpt_dir_A)
                load_strategy = FullyParallelLoadStrategyWrapper(
                    load_strategy,
                    parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
            else:
                load_strategy = None
            state_dict = load(
                model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()),
                ckpt_dir_A,
                load_strategy,
            )
            model_B.load_state_dict(state_dict)
            save(model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs

    @pytest.mark.parametrize(
        "src_tp_pp_exp,dest_tp_pp_exp,use_glu",
        [
            # changing PP is impossible because the number of layers must be the same
            ((2, 4, 1), (2, 4, 1), False),
            ((1, 1, 1), (1, 1, 4), False),
            ((2, 2, 2), (4, 2, 1), False),
            ((1, 1, 4), (8, 1, 1), False),
            ((2, 1, 4), (1, 1, 8), False),
            ((2, 4, 1), (2, 4, 1), True),
            ((1, 1, 1), (1, 1, 4), True),
            ((2, 2, 2), (4, 2, 1), True),
            ((1, 1, 4), (8, 1, 1), True),
            ((2, 1, 4), (1, 1, 8), True),
        ],
    )
    @pytest.mark.parametrize("src_module,dest_module", src_dest_expert_type)
    @pytest.mark.failing_on_rocm
    def test_sequential_grouped_mlp_interchangeable(
        self, tmp_path_dist_ckpt, src_tp_pp_exp, dest_tp_pp_exp, use_glu, src_module, dest_module
    ):
        """Test model saving and loading with different TP/PP/expert parallelism"""
        src_tp, src_pp, src_exp = src_tp_pp_exp
        dest_tp, dest_pp, dest_exp = dest_tp_pp_exp
        if src_module == 'grouped' or dest_module == 'grouped':
            add_bias_linear = False
        else:
            add_bias_linear = True
        # Save checkpoint A
        Utils.initialize_model_parallel(src_tp, src_pp, expert_model_parallel_size=src_exp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_sequential_grouped_mlp_interchangeable_model_A'
        ) as ckpt_dir_A, TempNamedDir(
            tmp_path_dist_ckpt / 'test_sequential_grouped_mlp_interchangeable_model_B'
        ) as ckpt_dir_B:

            model_A = initialize_expert_layer(
                1, use_glu, expert_type=src_module, add_bias_linear=add_bias_linear
            )
            sharded_state_dict = model_A.sharded_state_dict(sharded_offsets=get_pp_offsets())

            save_strategy = get_default_save_sharded_strategy()
            save(sharded_state_dict, ckpt_dir_A, save_strategy)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(dest_tp, dest_pp, expert_model_parallel_size=dest_exp)
            model_B = initialize_expert_layer(
                1, use_glu, expert_type=dest_module, add_bias_linear=add_bias_linear
            )
            load_strategy = None
            state_dict = load(
                model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()),
                ckpt_dir_A,
                load_strategy,
            )
            model_B.load_state_dict(state_dict)
            save(model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs
            Utils.destroy_model_parallel()
