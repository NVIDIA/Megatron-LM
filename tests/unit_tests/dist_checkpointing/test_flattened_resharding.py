# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import io

import numpy as np
import pytest
import torch
from torch.distributed.checkpoint import CheckpointException

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, save, load
from megatron.core.dist_checkpointing.core import CheckpointingException, \
    maybe_load_config
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory, \
    ShardedObject
from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestFlattenedResharding:
    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp',),
        [
            ((2, 4), (2, 4)),
            # TODO: uncomment after implementing flattened resharding
            # ((2, 4), (2, 2)),
            # ((8, 1), (1, 2)),
        ]
    )
    def test_partition_change_save_load(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp):
        with TempNamedDir(tmp_path_dist_ckpt / 'test_flattened_partition_change_save_load') as ckpt_dir:
            Utils.initialize_model_parallel(*src_tp_pp)
            state_dict = self._build_state_dict()

            save(state_dict, ckpt_dir)

            # change TPxPP
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(*dest_tp_pp)
            loaded_state_dict = load(self._build_state_dict(random=True), ckpt_dir)
            expected_state_dict = {k: v.data for k, v in self._build_state_dict().items()}

            diffs = diff(expected_state_dict, loaded_state_dict)
            assert not any(diffs), diffs
            Utils.destroy_model_parallel()


    def _build_state_dict(self, random=False):
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()

        init_fn = torch.rand if random else torch.arange
        global_ten = init_fn(4 * 5 * 80).reshape(4, 5, 80)
        local_ten = global_ten
        local_ten = local_ten.chunk(tp_size, dim=0)[tp_rank]
        local_ten = local_ten.chunk(pp_size, dim=2)[pp_rank]
        assert local_ten.shape == (4 // tp_size, 5, 80 // pp_size)

        local_ten_size_by_dp = local_ten.numel()
        assert local_ten_size_by_dp % dp_size == 0, (local_ten_size_by_dp, dp_size)
        local_ten_size_by_dp = local_ten_size_by_dp // dp_size
        # make a bit shifted DP slices so that they are not equal
        start_jitter = dp_rank
        end_jitter = dp_rank + 1 if dp_rank + 1 < dp_size else 0
        local_dp_slice = slice(
            local_ten_size_by_dp * dp_rank + start_jitter,
            local_ten_size_by_dp * (dp_rank + 1) + end_jitter
        )
        local_flat_ten = local_ten.flatten()[local_dp_slice]
        if dp_rank == dp_size - 1:
            assert local_flat_ten.numel() == local_ten_size_by_dp - dp_rank
        else:
            assert local_flat_ten.numel() == local_ten_size_by_dp + 1

        state_dict = {
            'sd_key_unflat': ShardedTensor.from_rank_offsets(
                'unflat',
                local_ten,
                (0, tp_rank, tp_size),
                (2, pp_rank, pp_size),
                replica_id=dp_rank,
            ),
            'sd_key_flat': ShardedTensor.from_rank_offsets_flat(
                'flat',
                local_flat_ten,
                local_ten.shape,
                (0, tp_rank, tp_size),
                (2, pp_rank, pp_size),
                flattened_range=local_dp_slice
            ),
        }
        return state_dict
