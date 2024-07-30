# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pickle
from copy import deepcopy

from dataclasses import fields

import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestCachedMetadata:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()   
        
    def test_cached_metadata(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 4)

        sharded_state_dict_non_cached = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
            ),
        }

        sharded_state_dict_cached = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
            ),
        }

        loaded_non_cached, loaded_cached = None, None
        md_non_cached, md_cached = None, None
        with TempNamedDir(tmp_path_dist_ckpt / 'ckpt_dir') as ckpt_dir:
            save(sharded_state_dict_non_cached, ckpt_dir, async_sharded_save=False)
            loaded_non_cached = load(sharded_state_dict_non_cached, ckpt_dir)
            md_path = ckpt_dir / '.metadata'
            with md_path.open('rb') as f:
                md_non_cached = pickle.load(f)

        save_strategy = deepcopy(get_default_save_sharded_strategy())
        save_strategy.use_cached_ckpt_structure = True
        # Run over 3 iterations with cached metadata enabled
        # The 3rd iteration will run with cached metadata
        # `ckpt_dir` at the 3rd iteration 2 will be maintained for comparison
        ckpt_dir = None
        for i in range(3):
            ckpt_dir = TempNamedDir(tmp_path_dist_ckpt / f'ckpt_dir_${i}_cached')
            save(
                sharded_state_dict_cached,
                ckpt_dir.__enter__(),
                save_strategy,
                async_sharded_save=False,
            )
            if i < 2:
                ckpt_dir.cleanup()
        loaded_cached = load(sharded_state_dict_cached, ckpt_dir.__enter__())
        md_path = ckpt_dir.__enter__() / '.metadata'

        with md_path.open('rb') as f:
            md_cached = pickle.load(f)

        # Check loaded state dict
        diffs = diff(loaded_non_cached, loaded_cached)

        assert not any(
            len(x) for x in diffs
        ), 'Cached metadata doesn\'t produce the same state_dict in loading'
        # Check metadata recorded in .metadata, torch.distributed.metadata.Metadata
        for field in fields(md_non_cached):
            if field.name not in ['storage_data', 'storage_meta']:
                diffs = diff(getattr(md_non_cached, field.name), getattr(md_cached, field.name))
                assert not any(
                    len(x) for x in diffs
                ), f'{field.name} is different in metadata from non-cached, cached metadata impls'
        ckpt_dir.cleanup()
        Utils.destroy_model_parallel()
