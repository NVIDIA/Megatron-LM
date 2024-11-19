# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.dict_utils import (
    dict_list_map_outplace,
    map_reduce,
    nested_values,
)
from megatron.core.dist_checkpointing.exchange_utils import _get_empty_tensor_for_exchange
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, is_main_replica
from megatron.core.dist_checkpointing.strategies.base import (
    LoadShardedStrategy,
    SaveShardedStrategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
    _sharded_tensor_shard_id,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class MockSaveStrategy(SaveShardedStrategy):
    def __init__(self):
        super().__init__('mock', 1)
        self.save_keys = set()

    def save(self, sharded_state_dict, ckpt_dir):
        self.save_keys = {
            sh_ten.key
            for sh_ten in nested_values(sharded_state_dict)
            if is_main_replica(sh_ten.replica_id)
        }


class MockLoadStrategy(LoadShardedStrategy):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.load_keys = set()

    def load(self, sharded_state_dict, ckpt_dir):
        self.load_keys = {
            sh_ten.key
            for sh_ten in nested_values(sharded_state_dict)
            if is_main_replica(sh_ten.replica_id)
        }

        def load_rand(x):
            assert isinstance(x, ShardedTensor)
            x.init_data(self.device)
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
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def get_sharded_state_dict():
        return {
            'sd_key_tp_repl1': ShardedTensor.from_rank_offsets(
                'key_TP_repl1',
                torch.ones(10),
                (
                    0,
                    parallel_state.get_tensor_model_parallel_rank(),
                    parallel_state.get_tensor_model_parallel_world_size(),
                ),
                replica_id=parallel_state.get_data_parallel_rank(with_context_parallel=True),
            ),
            'sd_key_tp_repl2': ShardedTensor.from_rank_offsets(
                'key_TP_repl2',
                torch.ones(10),
                (
                    0,
                    parallel_state.get_tensor_model_parallel_rank(),
                    parallel_state.get_tensor_model_parallel_world_size(),
                ),
                replica_id=parallel_state.get_data_parallel_rank(with_context_parallel=True),
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(20), (0, Utils.rank, Utils.world_size)
            ),
            'sd_keyE_no_C': ShardedTensor.from_rank_offsets(
                'keyC', torch.ones(100), replica_id=Utils.rank
            ),
            'sd_keyX_no_D': ShardedTensor.from_rank_offsets(
                'keyD', torch.ones(1000), replica_id=Utils.rank
            ),
            'sd_keyC_no_E': ShardedTensor.from_rank_offsets(
                'keyE', torch.ones(100), replica_id=Utils.rank
            ),
        }

    @pytest.mark.parametrize("parallelization_along_dp", [False, True])
    def test_save_distribution(self, parallelization_along_dp, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 1)
        state_dict = self.get_sharded_state_dict()

        # Ranks assignment:
        # 1. Lowest coverage
        # 2. Largest tensor
        # 3. Shard id (key)
        if not parallelization_along_dp:
            expected_key_to_saving_ranks = {
                'keyB': list(
                    range(Utils.world_size)
                ),  # everyone must save (disjoint shards, coverage == 1)
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
                    'keyB': list(
                        range(
                            parallel_state.get_data_parallel_world_size(with_context_parallel=True)
                        )
                    ),
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
                    'keyB': list(
                        range(
                            parallel_state.get_data_parallel_world_size(with_context_parallel=True)
                        )
                    ),
                    # tensors C, D, E are absent in this DP group
                    'key_TP_repl1': [0],  # smallest tensor
                    'key_TP_repl2': [1],  # smallest tensor, last rank is the least occupied
                }

        parallelization_group = (
            parallel_state.get_data_parallel_group(with_context_parallel=True)
            if parallelization_along_dp
            else None
        )
        dp_rank = torch.distributed.get_rank(parallelization_group)
        expected_keys_saved_by_current_rank = {
            k for k, v in expected_key_to_saving_ranks.items() if dp_rank in v
        }

        # Run save and tests
        mock_strategy = MockSaveStrategy()
        save_strategy = FullyParallelSaveStrategyWrapper(
            mock_strategy, parallelization_group, do_cache_distribution=True
        )
        with TempNamedDir(tmp_path_dist_ckpt / 'mock_dir') as ckpt_dir_A:
            save_strategy.save(state_dict, ckpt_dir_A)
        key_to_saving_rank = dict(
            map_reduce(
                save_strategy.cached_distribution.main_rank_for_shard.items(),
                lambda shard_rank: shard_rank[0][0],
                lambda shard_rank: shard_rank[1],
            )
        )
        assert expected_key_to_saving_ranks == key_to_saving_rank

        for _, sh_ten in state_dict.items():
            if (
                _sharded_tensor_shard_id(sh_ten)
                in save_strategy.cached_distribution.shards_in_this_group
            ):
                is_expected_to_be_saved_by_this_rank = dp_rank in expected_key_to_saving_ranks.get(
                    sh_ten.key, []
                )
                assert sh_ten.replica_id == int(
                    not is_expected_to_be_saved_by_this_rank
                ), expected_key_to_saving_ranks

        assert mock_strategy.save_keys == expected_keys_saved_by_current_rank, (
            Utils.rank,
            mock_strategy.save_keys,
            expected_keys_saved_by_current_rank,
        )

    @pytest.mark.parametrize("parallelization_along_dp", [False, True])
    def test_load_distribution(self, parallelization_along_dp, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 1)

        state_dict = self.get_sharded_state_dict()

        # Ranks assignment:
        # 1. Lowest coverage
        # 2. Largest tensor
        # 3. Shard id (key)
        if not parallelization_along_dp:
            expected_key_to_saving_ranks = {
                'keyB': list(
                    range(Utils.world_size)
                ),  # everyone must save (disjoint shards, coverage == 1)
                'key_TP_repl1': [0, 1],  # lowest coverage (4), first TP domain
                'key_TP_repl2': [2, 3],  # lowest coverage (4), second TP domain
                'keyD': [4],  # largest tensor
                'keyC': [5],  # second largest tensor
                'keyE': [6],  # second largest tensor
            }
        else:
            # When loading, expected key distribution is the same across TP, because every replica
            # needs to be loaded
            expected_key_to_saving_ranks = {
                # everyone must load (disjoint shards, coverage == 1):
                'keyB': list(
                    range(parallel_state.get_data_parallel_world_size(with_context_parallel=True))
                ),
                # this time, TP sharded tensors have the same coverage as fully replicated!
                'keyD': [0],  # largest tensor
                'keyC': [1],  # second largest tensor
                'keyE': [2],  # second largest tensor
                'key_TP_repl1': [3],  # smallest tensor
                'key_TP_repl2': [3],  # smallest tensor, last rank is the least occupied
            }

        parallelization_group = (
            parallel_state.get_data_parallel_group(with_context_parallel=True)
            if parallelization_along_dp
            else None
        )
        dp_rank = torch.distributed.get_rank(parallelization_group)
        expected_keys_saved_by_current_rank = {
            k for k, v in expected_key_to_saving_ranks.items() if dp_rank in v
        }

        # Run save and tests
        mock_strategy = MockLoadStrategy()
        load_strategy = FullyParallelLoadStrategyWrapper(
            mock_strategy, parallelization_group, do_cache_distribution=True
        )
        with TempNamedDir(tmp_path_dist_ckpt / 'mock_dir') as ckpt_dir_A:
            loaded_state_dict = load_strategy.load(state_dict, ckpt_dir_A)
        key_to_saving_rank = dict(
            map_reduce(
                load_strategy.cached_distribution.main_rank_for_shard.items(),
                lambda shard_rank: shard_rank[0][0],
                lambda shard_rank: shard_rank[1],
            )
        )
        assert expected_key_to_saving_ranks == key_to_saving_rank

        assert mock_strategy.load_keys == expected_keys_saved_by_current_rank, (
            Utils.rank,
            mock_strategy.load_keys,
            expected_keys_saved_by_current_rank,
        )

        assert loaded_state_dict.keys() == state_dict.keys()

    @pytest.mark.parametrize('state_dict_device', ['cpu', 'cuda'])
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    def test_memory_usage(self, state_dict_device, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 1)

        megabytes = 1024 * 1024
        mock_strategy = MockLoadStrategy(state_dict_device)

        mem_alloc = []

        real_get_empty_tensor_for_exchange = _get_empty_tensor_for_exchange

        def mock_get_empty_tensor_for_exchange(*args, **kwargs) -> torch.Tensor:
            ret = real_get_empty_tensor_for_exchange(*args, **kwargs)
            mem_alloc.append(torch.cuda.memory_allocated())
            return ret

        load_strategy = FullyParallelLoadStrategyWrapper(mock_strategy)
        torch.distributed.barrier()

        # Each tensor is 4MB, 40MB in total.
        # We expect extra memory usage peak at ~32MB, not 1GB
        sharded_state_dict = {
            f'ten_{i}': ShardedTensor.from_rank_offsets(
                f'ten_{i}',
                torch.rand(megabytes, dtype=torch.float, device=state_dict_device),
                (0, Utils.rank, Utils.world_size),
            )
            for i in range(10)
        }

        mem_alloc_start = torch.cuda.memory_allocated()

        with mock.patch(
            'megatron.core.dist_checkpointing.exchange_utils._get_empty_tensor_for_exchange',
            new=mock_get_empty_tensor_for_exchange,
        ), TempNamedDir(tmp_path_dist_ckpt / 'mock_dir') as ckpt_dir_A:
            _ = load_strategy.load(sharded_state_dict, ckpt_dir_A)

        # Each rank is expected to do 7 * 10 empty allocations
        assert len(mem_alloc) == 7 * 10
        # Peak mem usage should be within 4MB (single tensor)
        assert max(mem_alloc) - mem_alloc_start < 4.01 * megabytes, (
            max(mem_alloc),
            mem_alloc_start,
        )

        Utils.destroy_model_parallel()

    def test_only_necessary_exchanges_performed_during_load(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 1)

        # State dict with 2 expected exchanges
        sharded_state_dict_baseline_two_exchanges = {
            'needed_by_all_A': ShardedTensor.from_rank_offsets(
                'needed_by_all_A',
                torch.ones(4, dtype=torch.float, device='cuda'),
                replica_id=Utils.rank,
            ),
            'needed_by_all_B': ShardedTensor.from_rank_offsets(
                'needed_by_all_B',
                torch.ones(4, dtype=torch.float, device='cuda'),
                replica_id=Utils.rank,
            ),
        }
        # State dict with 1 expected exchange
        sharded_state_dict_baseline_one_exchange = {
            'needed_by_all': sharded_state_dict_baseline_two_exchanges['needed_by_all_A']
        }
        # State dict with 1 expected exchanges even though there are 2 tensors to load (1 is unique for each rank)
        sharded_state_dict_test_one_exchange = sharded_state_dict_baseline_one_exchange.copy()
        sharded_state_dict_test_one_exchange['unique'] = ShardedTensor.from_rank_offsets(
            'unique',
            torch.ones(4, dtype=torch.float, device='cuda'),
            (0, Utils.rank, Utils.world_size),
        )

        expected_call_counts: List[Tuple[ShardedStateDict, int]] = [
            (sharded_state_dict_baseline_one_exchange, 1),
            (sharded_state_dict_baseline_two_exchanges, 2),
            (sharded_state_dict_test_one_exchange, 1),
        ]

        mock_strategy = MockLoadStrategy()
        with TempNamedDir(tmp_path_dist_ckpt / 'mock_dir') as ckpt_dir:
            for sharded_state_dict, expected_count in expected_call_counts:
                load_strategy = FullyParallelLoadStrategyWrapper(
                    mock_strategy, None, do_cache_distribution=True, exchange_algo='broadcast'
                )
                with mock.patch(
                    'megatron.core.dist_checkpointing.strategies.fully_parallel.torch.distributed.broadcast'
                ) as broadcast_mock:
                    _ = load_strategy.load(sharded_state_dict, ckpt_dir)
                    assert broadcast_mock.call_count == expected_count

        Utils.destroy_model_parallel()
