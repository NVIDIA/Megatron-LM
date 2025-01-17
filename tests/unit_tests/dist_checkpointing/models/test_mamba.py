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
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def initialize_mamba(seed, glu=True, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    num_moe_experts = 8
    default_config_kwargs = dict(
        num_layers=pp_size,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=num_moe_experts,
        use_cpu_initialization=True,
        gated_linear_unit=glu,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    submodules = MambaMixerSubmodules(
        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
    )
    model = MambaMixer(transformer_config, submodules, transformer_config.hidden_size, rmsnorm=True)
    return model


def get_pp_offsets():
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    return ((0, pp_rank, pp_size),)


class TestMambaReconfiguration:
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
            # (True,  (1, 1, 4), (8, 1, 1), False),
            (False, (1, 8, 1), (1, 8, 1), False),
            (False, (1, 1, 4), (2, 1, 1), False),
            (False, (1, 1, 1), (1, 1, 1), True),
            (False, (1, 1, 1), (1, 1, 4), True),
            (True, (1, 1, 1), (2, 1, 1), True),
            # (False, (1, 1, 4), (8, 1, 1), True),
        ],
    )
    def test_parallel_reconfiguration_e2e(
        self, tmp_path_dist_ckpt, src_tp_pp_exp, dest_tp_pp_exp, use_glu, use_fpsl
    ):
        """Test model saving and loading with different TP/PP/expert parallelism"""
        src_tp, src_pp, src_exp = src_tp_pp_exp
        Utils.initialize_model_parallel(src_tp, src_pp, expert_model_parallel_size=src_exp)
        dest_tp, dest_pp, dest_exp = dest_tp_pp_exp
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_sequential_mlp_reconfiguration_model_A'
        ) as ckpt_dir_A, TempNamedDir(
            tmp_path_dist_ckpt / 'test_sequential_mlp_reconfiguration_model_B'
        ) as ckpt_dir_B:
            # Save checkpoint A
            model_A = initialize_mamba(1, use_glu)
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
            model_B = initialize_mamba(2, use_glu)
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
