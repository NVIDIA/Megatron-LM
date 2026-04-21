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
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def initialize_mamba(seed, glu=True, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    num_moe_experts = 8
    default_config_kwargs = dict(
        num_layers=pp_size,
        hidden_size=256,  # for Mamba: expand=2, headdim=64 -> nheads=8 (divisible by ngroups=8)
        num_attention_heads=8,  # must be divisible by tp_size (testing up to tp_size=8)
        num_moe_experts=num_moe_experts,
        use_cpu_initialization=True,
        gated_linear_unit=glu,
        add_bias_linear=False,
        pipeline_dtype=torch.bfloat16,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    submodules = MambaMixerSubmodules(
        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
    )
    model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    model = MambaMixer(
        transformer_config,
        submodules,
        transformer_config.hidden_size,
        rmsnorm=True,
        model_comm_pgs=model_comm_pgs,
    )
    return model


def get_pp_offsets():
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    return ((0, pp_rank, pp_size),)


class TestMambaReconfiguration:
    @pytest.mark.parametrize(
        "use_fpsl,src_tp_pp_exp_cp,dest_tp_pp_exp_cp,use_glu",
        [
            (False, (2, 4, 1, 1), (2, 4, 1, 1), False),
            (True, (2, 4, 1, 1), (2, 4, 1, 1), False),
            (False, (1, 1, 1, 1), (1, 1, 1, 1), False),
            (True, (1, 1, 1, 1), (1, 1, 4, 1), False),
            (False, (1, 1, 8, 1), (1, 1, 2, 1), False),
            (False, (2, 2, 2, 1), (4, 2, 1, 1), False),
            (True, (1, 1, 4, 1), (8, 1, 1, 1), False),
            (False, (1, 8, 1, 1), (1, 8, 1, 1), False),
            (False, (1, 1, 4, 1), (2, 1, 1, 1), False),
            (False, (1, 1, 1, 1), (1, 1, 1, 1), True),
            (False, (1, 1, 1, 1), (1, 1, 4, 1), True),
            (True, (1, 1, 1, 1), (2, 1, 1, 1), True),
            (False, (1, 1, 4, 1), (8, 1, 1, 1), True),
            # CP-focused cases:
            (False, (8, 1, 1, 1), (1, 1, 1, 8), False),
            (False, (4, 1, 1, 2), (2, 1, 1, 4), False),
            # TODO(duncan): investigate why changing pp_size (up or down) yields an unexpected shape
            #     mismatch error on dt_bias
        ],
    )
    def test_parallel_reconfiguration_e2e(
        self, tmp_path_dist_ckpt, src_tp_pp_exp_cp, dest_tp_pp_exp_cp, use_glu, use_fpsl
    ):
        """Test model saving and loading with different TP/PP/expert parallelism"""
        src_tp, src_pp, src_exp, src_cp = src_tp_pp_exp_cp
        Utils.initialize_model_parallel(
            src_tp, src_pp, expert_model_parallel_size=src_exp, context_parallel_size=src_cp
        )
        dest_tp, dest_pp, dest_exp, dest_cp = dest_tp_pp_exp_cp
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_sequential_mlp_reconfiguration_model_A'
        ) as ckpt_dir_A, TempNamedDir(
            tmp_path_dist_ckpt / 'test_sequential_mlp_reconfiguration_model_B'
        ) as ckpt_dir_B:
            # Save checkpoint A
            model_A = initialize_mamba(
                1,
                use_glu,
                tensor_model_parallel_size=src_tp,
                pipeline_model_parallel_size=src_pp,
                expert_model_parallel_size=src_exp,
                context_parallel_size=src_cp,
                # Sequence parallelism is required when using both expert and tensor parallelism
                sequence_parallel=(src_exp > 1 and src_pp > 1),
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

            # Load checkpoint A with different TP/PP/expert/CP and save as checkpoint B
            # No FPS this time, only FPL
            Utils.initialize_model_parallel(
                dest_tp, dest_pp, expert_model_parallel_size=dest_exp, context_parallel_size=dest_cp
            )
            model_B = initialize_mamba(
                2,
                use_glu,
                tensor_model_parallel_size=dest_tp,
                pipeline_model_parallel_size=dest_pp,
                expert_model_parallel_size=dest_exp,
                context_parallel_size=dest_cp,
                # Sequence parallelism is required when using both expert and tensor parallelism
                sequence_parallel=(dest_exp > 1 and dest_pp > 1),
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
        Utils.destroy_model_parallel()
