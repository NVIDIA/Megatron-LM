# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end checkpointing for MoE expert weights.

Expert weights are sharded with a prepended expert axis
(`ShardedTensor.prepend_axis_num > 0`), one shard per local expert. These tests use the
local (non-TransformerEngine) MoE spec so they run without TE installed.
"""

import pytest
import torch

from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff, nested_values
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.utils import initialize_moe_model
from tests.unit_tests.test_utilities import Utils


def _sharded_state_dict_and_prepended_shards(model):
    """Returns the model's sharded state dict and the shards with a prepended axis."""
    sharded_sd = model.sharded_state_dict()
    prepended = [
        sh_ten
        for sh_ten in nested_values(sharded_sd)
        if isinstance(sh_ten, ShardedTensor) and sh_ten.prepend_axis_num > 0
    ]
    return sharded_sd, prepended


class TestMoEExpertSharding:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('expert_model_parallel_size', [1, 2, 4])
    def test_moe_roundtrip_same_parallelism(self, tmp_path_dist_ckpt, expert_model_parallel_size):
        """Expert weights survive a save/load round trip."""
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=expert_model_parallel_size)
        model = initialize_moe_model(use_te=False, num_layers=2)
        sharded_sd, prepended = _sharded_state_dict_and_prepended_shards(model)

        # Homogeneous transformer layers are saved with a prepended layer axis, and (on top of
        # it) routed experts add their own prepended global-expert axis.
        num_layers, num_global_experts = 2, 8
        assert prepended, 'expected shards with a prepended axis'
        expert_shards = [sh_ten for sh_ten in prepended if sh_ten.prepend_axis_num == 2]
        assert expert_shards, 'expected routed experts to have a prepended expert axis'
        for sh_ten in expert_shards:
            # The prepended axes span all layers/experts, and this rank holds one of each.
            assert sh_ten.global_shape[:2] == (num_layers, num_global_experts)
            assert sh_ten.axis_fragmentations[:2] == (num_layers, num_global_experts)
            assert sh_ten.local_shape == sh_ten.data.shape

        if expert_model_parallel_size > 1:
            # With EP > 1 each rank holds a distinct subset of the global experts.
            local_expert_indices = sorted({sh_ten.global_offset[1] for sh_ten in expert_shards})
            assert len(local_expert_indices) == num_global_experts / expert_model_parallel_size

        with TempNamedDir(tmp_path_dist_ckpt / 'test_moe_roundtrip') as ckpt_dir:
            save(sharded_sd, ckpt_dir, async_sharded_save=False)

            loaded_model = initialize_moe_model(use_te=False, num_layers=2, seed=7)
            state_dict = load(loaded_model.sharded_state_dict(), ckpt_dir)
            loaded_model.load_state_dict(state_dict)

        for (name, ref_param), (other_name, param) in zip(
            model.state_dict().items(), loaded_model.state_dict().items()
        ):
            assert name == other_name
            if ref_param is None or param is None:
                assert ref_param is param
            elif isinstance(ref_param, torch.Tensor):
                assert torch.equal(ref_param, param), f'{name} differs after loading'
            else:
                assert ref_param == param, f'{name} differs after loading'

    def test_moe_resharding_across_expert_parallelism(self, tmp_path_dist_ckpt):
        """A checkpoint saved with EP=4 yields the same global weights when loaded with EP=2."""
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=4)
        src_model = initialize_moe_model(use_te=False, num_layers=2)
        _, prepended = _sharded_state_dict_and_prepended_shards(src_model)
        assert prepended

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_moe_resharding_src') as ckpt_dir,
            TempNamedDir(tmp_path_dist_ckpt / 'test_moe_resharding_dst') as dst_ckpt_dir,
        ):
            save(src_model.sharded_state_dict(), ckpt_dir, async_sharded_save=False)
            Utils.destroy_model_parallel()

            # Load the EP=4 checkpoint into an EP=2 model and save it back.
            Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
            dest_model = initialize_moe_model(use_te=False, num_layers=2, seed=7)
            state_dict = load(dest_model.sharded_state_dict(), ckpt_dir)
            dest_model.load_state_dict(state_dict)
            save(dest_model.sharded_state_dict(), dst_ckpt_dir, async_sharded_save=False)
            Utils.destroy_model_parallel()

            # Compare both checkpoints as plain (model-parallelism independent) tensors.
            Utils.initialize_model_parallel(1, 1)
            diffs = diff(load_plain_tensors(ckpt_dir), load_plain_tensors(dst_ckpt_dir))
            assert not any(map(bool, diffs)), diffs
