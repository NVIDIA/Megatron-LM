# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_local_spec
)
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

def initialize_mlp(glu=True):
    model_parallel_device_manual_seed(123)
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    transformer_config = TransformerConfig(
        num_layers=pp_size,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        gated_linear_unit=glu,
    )
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
    return MLP(
        transformer_config, transformer_layer_spec.submodules.mlp.submodules
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
            save(mlp_A.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_A,
                 process_group=parallel_state.get_default_process_group())
            Utils.destroy_model_parallel()

            # Load checkpoint A with different TP/PP and save as checkpoint B
            Utils.initialize_model_parallel(*dest_tp_pp)
            mlp_B = initialize_mlp()
            state_dict = load(
                mlp_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_A,
                process_group=parallel_state.get_default_process_group()
            )
            mlp_B.load_state_dict(state_dict)
            save(mlp_B.sharded_state_dict(sharded_offsets=get_pp_offsets()), ckpt_dir_B,
                 process_group=parallel_state.get_default_process_group())
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A, process_group=parallel_state.get_default_process_group())
            state_dict_B = load_plain_tensors(ckpt_dir_B, process_group=parallel_state.get_default_process_group())
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs
