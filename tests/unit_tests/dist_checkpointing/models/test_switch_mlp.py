# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import save, load, load_plain_tensors
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.models.gpt.gpt_layer_specs import \
    get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def initialize_switch_mlp(seed, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    num_moe_experts = 8
    num_local_experts = num_moe_experts // parallel_state.get_expert_model_parallel_world_size()
    default_config_kwargs = dict(num_layers=pp_size, hidden_size=12, num_attention_heads=4, num_moe_experts=num_moe_experts, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(num_experts=num_moe_experts, moe_grouped_gemm=False)
    model = SequentialMLP(num_local_experts,
                          transformer_config,
                          transformer_layer_spec.submodules.mlp.submodules)
    return model


def get_pp_offsets():
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    return ((0, pp_rank, pp_size),)


class TestSwitchMLPReconfiguration:
    @pytest.mark.parametrize("src_tp_pp_exp,dest_tp_pp_exp,", [
        # changing PP is impossible because the number of layers must be the same
        ((2, 4, 1), (2, 4, 1)),
        ((1, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 4)),
        ((1, 1, 8), (1, 1, 2)),
        ((2, 2, 2), (4, 2, 1)),
        ((1, 1, 4), (8, 1, 1)),
        ((1, 8, 1), (1, 8, 1)),
        ((1, 1, 4), (2, 1, 1)),
    ])
    def test_parallel_reconfiguration_e2e(self, tmp_path_dist_ckpt, src_tp_pp_exp, dest_tp_pp_exp):
        """ Test model saving and loading with different TP/PP/expert parallelism """
        src_tp, src_pp, src_exp = src_tp_pp_exp
        dest_tp, dest_pp, dest_exp = dest_tp_pp_exp
        with TempNamedDir(tmp_path_dist_ckpt / 'test_switch_mlp_reconfiguration_model_A') as ckpt_dir_A, \
             TempNamedDir(tmp_path_dist_ckpt / 'test_switch_mlp_reconfiguration_model_B') as ckpt_dir_B:
            # Save checkpoint A
            Utils.initialize_model_parallel(src_tp, src_pp, expert_model_parallel_size=src_exp)
            model_A = initialize_switch_mlp(1)
            sharded_state_dict = model_A.sharded_state_dict(sharded_offsets=get_pp_offsets())
            save(sharded_state_dict, ckpt_dir_A)
            Utils.destroy_model_parallel()

            # Load checkpoint A with different TP/PP/expert and save as checkpoint B
            Utils.initialize_model_parallel(dest_tp, dest_pp, expert_model_parallel_size=dest_exp)
            model_B = initialize_switch_mlp(2)
            state_dict = load(model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_A)
            model_B.load_state_dict(state_dict)
            save(model_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs