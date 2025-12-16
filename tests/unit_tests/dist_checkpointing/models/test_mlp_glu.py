# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import inspect
import logging

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
from megatron.core.transformer.mlp import MLP, apply_swiglu_sharded_factory
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
    @pytest.mark.parametrize('singleton_local_shards', [True, False])
    def test_parallel_reconfiguration_e2e(
        self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp, singleton_local_shards
    ):
        """Test module saving and loading with different TP/PP"""
        Utils.initialize_model_parallel(*src_tp_pp)
        metadata = {'singleton_local_shards': singleton_local_shards}

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_mlp_glu_reconfiguration_model_A') as ckpt_dir_A,
            TempNamedDir(tmp_path_dist_ckpt / 'test_mlp_glu_reconfiguration_model_B') as ckpt_dir_B,
        ):
            # Save checkpoint A
            layer_prefix = f'{parallel_state.get_pipeline_model_parallel_rank()}.'
            mlp_A = initialize_mlp()
            save(mlp_A.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_dir_A)
            Utils.destroy_model_parallel()

            if "dp_cp_group" in metadata.keys():
                del metadata["dp_cp_group"]

            # Load checkpoint A with different TP/PP and save as checkpoint B
            Utils.initialize_model_parallel(*dest_tp_pp)
            mlp_B = initialize_mlp()
            state_dict = load(
                mlp_B.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_dir_A
            )
            mlp_B.load_state_dict({k.removeprefix(layer_prefix): v for k, v in state_dict.items()})
            save(mlp_B.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs

    def test_oom_is_handled(self, caplog):
        Utils.initialize_model_parallel(Utils.world_size, 1)
        dtype = torch.bfloat16

        # Compute free memory in bytes
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        free = total - allocated

        # We should create two tensor which take up between 50% and 100% of free memory,
        # so that the torch.cat tries to allocate twice as many and OOMs.
        expected_local_num_bytes = free * 0.6

        local_num_elems = expected_local_num_bytes // torch._utils._element_size(dtype)
        local_num_elems = int(local_num_elems // 1024 * 1024)
        assert local_num_elems % 1024 == 0

        local_w_plus_v_shape = (local_num_elems // 512, 512)
        local_w_or_v_shape = (local_num_elems // 1024, 512)

        fc1_weight_sh_ten = ShardedTensor.from_rank_offsets(
            'a',
            torch.ones(local_w_plus_v_shape, device='cuda', dtype=dtype),
            (0, Utils.rank, Utils.world_size),
        )
        fc1_factory = apply_swiglu_sharded_factory(fc1_weight_sh_ten, ())
        sharded_state_dict = fc1_factory.build()
        assert len(sharded_state_dict) == 2
        assert sharded_state_dict[0].data.shape == local_w_or_v_shape
        # NOTE: with singleton_local_shards=True this assert would fail - global shape is
        #  `(Utils.world_size * local_w_or_v_shape[0], local_w_or_v_shape[1])`
        assert sharded_state_dict[0].global_shape[-2:] == (
            Utils.world_size * local_w_plus_v_shape[0],
            local_w_or_v_shape[1],
        )

        # Checkpoint load replaces ShardedTensors with tensors.
        # Load happens in-place, so we can just use the same tensors
        loaded_state_dict = [sh_ten.data for sh_ten in sharded_state_dict]

        # The critical part that should OOM:
        with caplog.at_level(logging.WARNING):
            fc1_factory.merge_fn(loaded_state_dict)
            assert "CUDA OutOfMemoryError encountered during tensors merging" in caplog.text
