# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from pathlib import Path

import pytest

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.dict_utils import nested_values, \
    map_reduce, dict_list_map_outplace
from megatron.core.dist_checkpointing.mapping import is_main_replica
from megatron.core.dist_checkpointing.strategies.base import \
    SaveShardedStrategy, LoadShardedStrategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import \
    FullyParallelSaveStrategyWrapper, _sharded_tensor_chunk_id, \
    FullyParallelLoadStrategyWrapper
from tests.unit_tests.test_utilities import Utils


class MockSaveStrategy(SaveShardedStrategy):
    def __init__(self):
        super().__init__('mock', 1)
        self.save_keys = set()

    def save(self, sharded_state_dict, ckpt_dir):
        self.save_keys = {sh_ten.key for sh_ten in nested_values(sharded_state_dict)
                          if is_main_replica(sh_ten.replica_id)}


class MockLoadStrategy(LoadShardedStrategy):
    def __init__(self):
        super().__init__()
        self.load_keys = set()

    def load(self, sharded_state_dict, ckpt_dir):
        self.load_keys = {sh_ten.key for sh_ten in nested_values(sharded_state_dict)
                          if is_main_replica(sh_ten.replica_id)}

        def load_rand(x):
            assert isinstance(x, ShardedTensor)
            x.init_data('cpu')
            x.data.fill_(Utils.rank)
            return x.data

        return dict_list_map_outplace(load_rand, sharded_state_dict)

    def load_tensors_metadata(self, checkpoint_dir: Path):
        pass

    def check_backend_compatibility(self, loaded_version):
        pass

    def check_version_compatibility(self, loaded_version):
        pass


class TestFullyParallelSaveAndLoad:
    @staticmethod
    def get_sharded_state_dict():
        return {
            'sd_key_tp_repl1': ShardedTensor.from_rank_offsets('key_TP_repl1', torch.ones(10),
                                                               (0, parallel_state.get_tensor_model_parallel_rank(), parallel_state.get_tensor_model_parallel_world_size()),
                                                               replica_id=parallel_state.get_data_parallel_rank(with_context_parallel=True)),
            'sd_key_tp_repl2': ShardedTensor.from_rank_offsets('key_TP_repl2', torch.ones(10),
                                                               (0, parallel_state.get_tensor_model_parallel_rank(), parallel_state.get_tensor_model_parallel_world_size()),
                                                               replica_id=parallel_state.get_data_parallel_rank(with_context_parallel=True)),
            'sd_keyB': ShardedTensor.from_rank_offsets('keyB', torch.ones(20), (0, Utils.rank, Utils.world_size)),
            'sd_keyE_no_C': ShardedTensor.from_rank_offsets('keyC', torch.ones(100), replica_id=Utils.rank),
            'sd_keyX_no_D': ShardedTensor.from_rank_offsets('keyD', torch.ones(1000), replica_id=Utils.rank),
            'sd_keyC_no_E': ShardedTensor.from_rank_offsets('keyE', torch.ones(100), replica_id=Utils.rank),
        }

    @pytest.mark.parametrize("parallelization_along_dp", [False, True])
    def test_save_distribution(self, parallelization_along_dp):
        Utils.initialize_model_parallel(2, 1)
        state_dict = self.get_sharded_state_dict()

        # Ranks assignment:
        # 1. Lowest coverage
        # 2. Largest tensor
        # 3. Chunk id (key)
        if not parallelization_along_dp:
            expected_key_to_saving_ranks = {
                'keyB': list(range(Utils.world_size)), # everyone must save (disjoint shards, coverage == 1)
                'key_TP_repl1': [0, 1],  # lowest coverage (4), first TP domain
                'key_TP_repl2': [2, 3],  # lowest coverage (4), second TP domain
                'keyD': [4],  # largest tensor
                'keyC': [5],  # second largest tensor
                'keyE': [6],  # second largest tensor
            }
        else:
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                expected_key_to_saving_ranks = {
                    # everyone must save (disjoint shards, coverage == 1):
                    'keyB': list(range(parallel_state.get_data_parallel_world_size(with_context_parallel=True))),
                    # this time, TP sharded tensors have the same coverage as fully replicated!
                    'keyD': [0],  # largest tensor
                    'keyC': [1],  # second largest tensor
                    'keyE': [2],  # second largest tensor
                    'key_TP_repl1': [3],  # smallest tensor
                    'key_TP_repl2': [3],  # smallest tensor, last rank is the least occupied
                }
            else:
                expected_key_to_saving_ranks = {
                    # everyone must save (disjoint shards, coverage == 1):
                    'keyB': list(range(parallel_state.get_data_parallel_world_size(with_context_parallel=True))),
                    # tensors C, D, E are absent in this DP group
                    'key_TP_repl1': [0],  # smallest tensor
                    'key_TP_repl2': [1],  # smallest tensor, last rank is the least occupied
                }

        parallelization_group = parallel_state.get_data_parallel_group(with_context_parallel=True) if parallelization_along_dp else None
        dp_rank = torch.distributed.get_rank(parallelization_group)
        expected_keys_saved_by_current_rank = {k for k, v in expected_key_to_saving_ranks.items() if dp_rank in v}

        # Run save and tests
        mock_strategy = MockSaveStrategy()
        save_strategy = FullyParallelSaveStrategyWrapper(mock_strategy,
                                                         parallelization_group,
                                                         do_cache_distribution=True)
        save_strategy.save(state_dict, Path('mock_dir'))
        shard_to_rank, shards_saved_by_this_dp_group = save_strategy.cached_distribution
        key_to_saving_rank = dict(map_reduce(shard_to_rank.items(), lambda shard_rank: shard_rank[0][0], lambda shard_rank: shard_rank[1]))
        assert expected_key_to_saving_ranks == key_to_saving_rank

        for k, sh_ten in state_dict.items():
            if _sharded_tensor_chunk_id(sh_ten) in shards_saved_by_this_dp_group:
                is_expected_to_be_saved_by_this_rank = dp_rank in expected_key_to_saving_ranks.get(sh_ten.key, [])
                assert sh_ten.replica_id == int(not is_expected_to_be_saved_by_this_rank), expected_key_to_saving_ranks

        assert mock_strategy.save_keys == expected_keys_saved_by_current_rank, (Utils.rank, mock_strategy.save_keys, expected_keys_saved_by_current_rank)

    @pytest.mark.parametrize("parallelization_along_dp", [False, True])
    def test_load_distribution(self, parallelization_along_dp):
        Utils.initialize_model_parallel(2, 1)

        state_dict = self.get_sharded_state_dict()

        # Ranks assignment:
        # 1. Lowest coverage
        # 2. Largest tensor
        # 3. Chunk id (key)
        if not parallelization_along_dp:
            expected_key_to_saving_ranks = {
                'keyB': list(range(Utils.world_size)), # everyone must save (disjoint shards, coverage == 1)
                'key_TP_repl1': [0, 1],  # lowest coverage (4), first TP domain
                'key_TP_repl2': [2, 3],  # lowest coverage (4), second TP domain
                'keyD': [4],  # largest tensor
                'keyC': [5],  # second largest tensor
                'keyE': [6],  # second largest tensor
            }
        else:
            # When loading, expected key distribution is the same across TP, because every replica needs to be loaded
            expected_key_to_saving_ranks = {
                # everyone must load (disjoint shards, coverage == 1):
                'keyB': list(range(parallel_state.get_data_parallel_world_size(with_context_parallel=True))),
                # this time, TP sharded tensors have the same coverage as fully replicated!
                'keyD': [0],  # largest tensor
                'keyC': [1],  # second largest tensor
                'keyE': [2],  # second largest tensor
                'key_TP_repl1': [3],  # smallest tensor
                'key_TP_repl2': [3],  # smallest tensor, last rank is the least occupied
            }

        parallelization_group = parallel_state.get_data_parallel_group(with_context_parallel=True) if parallelization_along_dp else None
        dp_rank = torch.distributed.get_rank(parallelization_group)
        expected_keys_saved_by_current_rank = {k for k, v in expected_key_to_saving_ranks.items() if dp_rank in v}

        # Run save and tests
        mock_strategy = MockLoadStrategy()
        load_strategy = FullyParallelLoadStrategyWrapper(mock_strategy,
                                                         parallelization_group,
                                                         do_cache_distribution=True)
        loaded_state_dict = load_strategy.load(state_dict, Path('mock_dir'))
        shard_to_rank, shards_saved_by_this_dp_group = load_strategy.cached_distribution
        key_to_saving_rank = dict(map_reduce(shard_to_rank.items(), lambda shard_rank: shard_rank[0][0], lambda shard_rank: shard_rank[1]))
        assert expected_key_to_saving_ranks == key_to_saving_rank

        assert mock_strategy.load_keys == expected_keys_saved_by_current_rank, (Utils.rank, mock_strategy.load_keys, expected_keys_saved_by_current_rank)

        assert loaded_state_dict.keys() == state_dict.keys()
