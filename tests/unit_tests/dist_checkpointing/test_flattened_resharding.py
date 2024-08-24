# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import io

import numpy as np
import pytest
import torch
from torch.distributed.checkpoint import CheckpointException

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.core import CheckpointingException, maybe_load_config
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensorFactory
from megatron.core.dist_checkpointing.serialization import load_tensors_metadata
from megatron.core.dist_checkpointing.strategies.resharding import (
    apply_nd_flattened_tensors_reformulation,
    restore_nd_flattened_tensors_formulation,
)
from megatron.core.dist_checkpointing.strategies.torch import get_reformulation_metadata
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestFlattenedResharding:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp'),
        [((2, 4), (2, 4)), ((2, 4), (2, 2)), ((2, 4), (4, 2)), ((8, 1), (1, 2))],
    )
    def test_partition_change_save_load(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp):
        Utils.initialize_model_parallel(*src_tp_pp)
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_flattened_partition_change_save_load'
        ) as ckpt_dir:

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

    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp', 'expected_ckpt_offsets_by_rank'),
        [
            (
                (2, 4),
                (2, 2),
                {
                    0: [(0, 0, 0), (0, 0, 10)],  # TP 0, DP 0, PP 0
                    1: [(4, 0, 0), (4, 0, 10)],  # TP 1, DP 0, PP 0
                    2: [(0, 0, 0), (0, 0, 10)],  # TP 0, DP 1, PP 0
                    3: [(4, 0, 0), (4, 0, 10)],  # TP 1, DP 1, PP 0
                    4: [(0, 0, 20), (0, 0, 30)],  # TP 0, DP 0, PP 1
                    5: [(4, 0, 20), (4, 0, 30)],  # TP 1, DP 0, PP 1
                    6: [(0, 0, 20), (0, 0, 30)],  # TP 0, DP 1, PP 1
                    7: [(4, 0, 20), (4, 0, 30)],  # TP 1, DP 1, PP 1
                },
            ),
            ((8, 1), (1, 2), {rank: [(tp, 0, 0) for tp in range(8)] for rank in range(8)}),
        ],
    )
    def test_reformulate_nd_flattened_tensors(
        self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp, expected_ckpt_offsets_by_rank
    ):
        Utils.initialize_model_parallel(*src_tp_pp, order='tp-dp-pp')
        with TempNamedDir(tmp_path_dist_ckpt / 'test_reformulate_nd_flattened_tensors') as ckpt_dir:

            state_dict = self._build_state_dict()

            ckpt_local_shape = state_dict['sd_key_flat'].local_shape

            save(state_dict, ckpt_dir)

            # change TPxPP
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(*dest_tp_pp, order='tp-dp-pp')
            load_state_dict = self._build_state_dict(random=True)

            reformulation_metadata = get_reformulation_metadata(load_state_dict, ckpt_dir)
            reformulated_state_dict, formulation_restore_data = (
                apply_nd_flattened_tensors_reformulation(load_state_dict, reformulation_metadata)
            )
            assert isinstance(reformulated_state_dict['sd_key_unflat'], ShardedTensor)
            assert isinstance(reformulated_state_dict['sd_key_flat'], dict)

            assert reformulated_state_dict['sd_key_flat'].keys() == set(
                (offset, ckpt_local_shape) for offset in expected_ckpt_offsets_by_rank[Utils.rank]
            ), (
                reformulated_state_dict['sd_key_flat'].keys(),
                ckpt_local_shape,
                expected_ckpt_offsets_by_rank[Utils.rank],
            )

            # We can even load the reformulated state dict with a high-level API
            loaded_state_dict = load(
                reformulated_state_dict, ckpt_dir, validate_access_integrity=False
            )
            loaded_state_dict = restore_nd_flattened_tensors_formulation(
                loaded_state_dict, formulation_restore_data
            )
            expected_state_dict = {k: v.data for k, v in self._build_state_dict().items()}
            diffs = diff(expected_state_dict, loaded_state_dict)
            assert not any(diffs), diffs

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(('src_tp_pp',), [((2, 4),), ((8, 1),), ((1, 1),), ((1, 4),)])
    def test_load_tensor_metadata(self, tmp_path_dist_ckpt, src_tp_pp):
        Utils.initialize_model_parallel(*src_tp_pp, order='tp-dp-pp')
        with TempNamedDir(tmp_path_dist_ckpt / 'test_reformulate_nd_flattened_tensors') as ckpt_dir:

            state_dict = self._build_state_dict()

            save(state_dict, ckpt_dir)

            # change TPxPP
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(1, 1)

            sharded_metadata = load_tensors_metadata(ckpt_dir)

            for attr_name in ('local_shape', 'global_shape'):
                flat_val = getattr(sharded_metadata['flat'], attr_name)
                unflat_val = getattr(sharded_metadata['unflat'], attr_name)
                assert flat_val == unflat_val, (attr_name, flat_val, unflat_val)

            for sh_ten in sharded_metadata.values():
                sh_ten.replica_id = Utils.rank
            loaded_state_dict = load(sharded_metadata, ckpt_dir)
            assert torch.all(
                loaded_state_dict['unflat'] == torch.arange(8 * 5 * 40).reshape(8, 5, 40)
            )
            assert torch.all(loaded_state_dict['flat'] == torch.arange(8 * 5 * 40))

        Utils.destroy_model_parallel()

    def _build_state_dict(self, random=False):
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()

        init_fn = torch.rand if random else torch.arange
        global_ten = init_fn(8 * 5 * 40).reshape(8, 5, 40)
        local_ten = global_ten
        local_ten = local_ten.chunk(tp_size, dim=0)[tp_rank]
        local_ten = local_ten.chunk(pp_size, dim=2)[pp_rank]
        assert local_ten.shape == (8 // tp_size, 5, 40 // pp_size)

        local_ten_size_by_dp = local_ten.numel()
        assert local_ten_size_by_dp % dp_size == 0, (local_ten_size_by_dp, dp_size)
        local_ten_size_by_dp = local_ten_size_by_dp // dp_size
        # make a bit shifted DP slices so that they are not equal
        start_jitter = dp_rank
        end_jitter = dp_rank + 1 if dp_rank + 1 < dp_size else 0
        local_dp_slice = slice(
            local_ten_size_by_dp * dp_rank + start_jitter,
            local_ten_size_by_dp * (dp_rank + 1) + end_jitter,
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
                flattened_range=local_dp_slice,
            ),
        }
        return state_dict
