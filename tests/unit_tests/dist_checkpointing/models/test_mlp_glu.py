# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

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
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def initialize_mlp(glu=True):
    model_parallel_cuda_manual_seed(123)
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    transformer_config = TransformerConfig(
        num_layers=pp_size,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        gated_linear_unit=glu,
    )
    return MLP(
        transformer_config, get_gpt_layer_with_transformer_engine_spec().submodules.mlp.submodules
    )


def get_pp_offsets():
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    return ((0, pp_rank, pp_size),)


class TestParallelMLPWithGLU:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "src_tp_pp,dest_tp_pp",
        [
            # changing PP is impossible because the number of layers must be the same
            ((2, 2), (4, 2)),
            ((1, 1), (8, 1)),
            ((1, 8), (1, 8)),
            ((1, 1), (2, 1)),
        ],
    )
    def test_parallel_reconfiguration_e2e(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp):
        """Test module saving and loading with different TP/PP"""
        Utils.initialize_model_parallel(*src_tp_pp)

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_mlp_glu_reconfiguration_model_A'
        ) as ckpt_dir_A, TempNamedDir(
            tmp_path_dist_ckpt / 'test_mlp_glu_reconfiguration_model_B'
        ) as ckpt_dir_B:
            # Save checkpoint A
            mlp_A = initialize_mlp()
            save(mlp_A.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_A)
            Utils.destroy_model_parallel()

            # Load checkpoint A with different TP/PP and save as checkpoint B
            Utils.initialize_model_parallel(*dest_tp_pp)
            mlp_B = initialize_mlp()
            state_dict = load(
                mlp_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_A
            )
            mlp_B.load_state_dict(state_dict)
            save(mlp_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs
