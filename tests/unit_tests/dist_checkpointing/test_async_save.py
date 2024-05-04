# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.dist_checkpointing import ShardedTensor, save, load
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.strategies.async_utils import \
    AsyncCallsQueue
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestAsyncSave:
    def test_async_is_equivalent_to_sync(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 4)

        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets('keyA', torch.ones(2, 4), replica_id=Utils.rank),
            'sd_keyB': ShardedTensor.from_rank_offsets('keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1),
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_async') as async_ckpt_dir, \
             TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_sync') as sync_ckpt_dir:
            # async
            async_calls = AsyncCallsQueue()
            async_request = save(sharded_state_dict, async_ckpt_dir, async_sharded_save=True)
            async_calls.schedule_async_request(async_request)

            # sync
            save(sharded_state_dict, sync_ckpt_dir, async_sharded_save=False)

            # finalize async
            async_calls.maybe_finalize_async_calls(blocking=True)

            # load and compare
            loaded_async_state_dict = load(sharded_state_dict, async_ckpt_dir)
            loaded_sync_state_dict = load(sharded_state_dict, sync_ckpt_dir)
            diffs = diff(loaded_async_state_dict, loaded_sync_state_dict)
            assert not any(map(bool, diffs)), diffs

        Utils.destroy_model_parallel()
