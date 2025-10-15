# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import inspect
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Dict, List, Tuple
from unittest import mock

import pytest
import torch
import torch.distributed

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.dict_utils import (
    dict_list_map_outplace,
    map_reduce,
    nested_values,
)
from megatron.core.dist_checkpointing.exchange_utils import (
    _get_empty_tensor_for_exchange,
    distribute_shards_to_ranks,
)
from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedStateDict,
    ShardedTensorFactory,
    is_main_replica,
)
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.base import (
    LoadShardedStrategy,
    SaveShardedStrategy,
    StrategyAction,
    get_default_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
    _sharded_tensor_shard_id,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreLoadPlanner,
    TorchDistSaveShardedStrategy,
)
from megatron.core.utils import get_pg_rank
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class MockSaveStrategy(SaveShardedStrategy):
    def __init__(self):
        super().__init__('mock', 1)
        self.save_keys = set()

    def save(self, sharded_state_dict, ckpt_dir):
        for sh_ten in nested_values(sharded_state_dict):
            if is_main_replica(sh_ten.replica_id):
                self.save_keys.add(sh_ten.key)


class MockLoadStrategy(LoadShardedStrategy):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.load_keys = set()

    def load(self, sharded_state_dict, ckpt_dir):
        for sh_ten in nested_values(sharded_state_dict):
            if is_main_replica(sh_ten.replica_id):
                self.load_keys.add(sh_ten.key)

        def load_rand(x):
            assert isinstance(x, ShardedTensor) or isinstance(x, ShardedObject)
            if isinstance(x, ShardedTensor):
                x.init_data(self.device)
                x.data.fill_(Utils.rank)
                return x.data
            else:
                x.data = [Utils.rank]
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
            else torch.distributed.group.WORLD
        )
        dp_rank = get_pg_rank(parallelization_group)
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

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelize_within_dp", [False, True])
    def test_load_distribution(self, parallelize_within_dp, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 1)

        state_dict = self.get_sharded_state_dict()

        # Ranks assignment:
        # 1. non-cross-DP read
        # 2. Lowest coverage
        # 3. Largest tensor
        # 4. Shard id (key)
        if not parallelize_within_dp:
            expected_key_to_loading_ranks = {
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
            # We must check if we should expect old load behavior (<= v0.10) or aligned one (v0.11)
            sig = inspect.signature(distribute_shards_to_ranks)
            aligned_load = 'cross_parallelization_group_loads' in sig.parameters
            if not aligned_load or parallel_state.get_tensor_model_parallel_rank() == 0:
                # All main ranks are in the first DP group (TP rank 0),
                # so the load distribution is the same as the saving one
                expected_key_to_loading_ranks = {
                    # everyone must load (disjoint shards, coverage == 1):
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
                # 'C', 'D', 'E' are cross-DP reads, so are assigned at the end.
                # First 'key_TP_repl*' are assigned to rank 0 and 1
                expected_key_to_loading_ranks = {
                    # everyone must load (disjoint shards, coverage == 1):
                    'keyB': list(
                        range(
                            parallel_state.get_data_parallel_world_size(with_context_parallel=True)
                        )
                    ),
                    # the only intra-DP reads
                    'key_TP_repl1': [0],
                    'key_TP_repl2': [1],
                    # cross-DP reads are assigned at the end
                    'keyD': [2],  # largest tensor
                    'keyC': [3],  # second largest tensor
                    'keyE': [0],  # second largest tensor, round-robin
                }

        parallelization_group = (
            parallel_state.get_data_parallel_group(with_context_parallel=True)
            if parallelize_within_dp
            else torch.distributed.group.WORLD
        )
        dp_rank = get_pg_rank(parallelization_group)
        expected_keys_loaded_by_current_rank = {
            k for k, v in expected_key_to_loading_ranks.items() if dp_rank in v
        }

        # Run save and tests
        mock_strategy = MockLoadStrategy()
        load_strategy = FullyParallelLoadStrategyWrapper(
            mock_strategy, parallelization_group, do_cache_distribution=True
        )
        with TempNamedDir(tmp_path_dist_ckpt / 'mock_dir') as ckpt_dir_A:
            loaded_state_dict = load_strategy.load(state_dict, ckpt_dir_A)
        key_to_loading_rank = dict(
            map_reduce(
                load_strategy.cached_distribution.main_rank_for_shard.items(),
                lambda shard_rank: shard_rank[0][0],
                lambda shard_rank: shard_rank[1],
            )
        )
        assert expected_key_to_loading_ranks == key_to_loading_rank

        assert mock_strategy.load_keys == expected_keys_loaded_by_current_rank, (
            Utils.rank,
            mock_strategy.load_keys,
            expected_keys_loaded_by_current_rank,
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
                replica_id=Utils.rank,
            )
            for i in range(10)
        }

        mem_alloc_start = torch.cuda.memory_allocated()

        with mock.patch(
            'megatron.core.dist_checkpointing.exchange_utils._get_empty_tensor_for_exchange',
            new=mock_get_empty_tensor_for_exchange,
        ), TempNamedDir(tmp_path_dist_ckpt / 'mock_dir') as ckpt_dir_A:
            _ = load_strategy.load(sharded_state_dict, ckpt_dir_A)

        # Each rank is expected to do 9 allocations for all shards loaded by some other rank.
        # There are 10 shards and 8 ranks so ranks <= 1 load 2 shards (and allocate 10 - 2 = 8)
        assert len(mem_alloc) == 8 if Utils.rank <= 1 else 9
        # Peak mem usage should be within 4MB (single tensor)
        assert max(mem_alloc) - mem_alloc_start < 4.01 * megabytes, (
            max(mem_alloc),
            mem_alloc_start,
        )

        Utils.destroy_model_parallel()

    @pytest.mark.internal
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

    def test_broadcast_sharded_objects(self, tmp_path_dist_ckpt):

        sharded_state_dict = {
            f'Obj_{i}': ShardedObject(f'Obj_{i}', None, (1,), (0,), replica_id=abs(Utils.rank - i))
            for i in range(Utils.world_size)
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_broadcast_sharded_objects') as ckpt_dir:
            load_strategy = MockLoadStrategy()
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy, None)

            loaded_state_dict = load_strategy.load(sharded_state_dict, ckpt_dir)

            # each rank is supposed to only load obj_rank because of how replica_id is set
            assert load_strategy.base_strategy.load_keys == set({f'Obj_{Utils.rank}'})

            # since each rank only loaded their Obj they were broadcasted
            assert set(sharded_state_dict.keys()) == set(loaded_state_dict.keys())


class TestCrossRanksReads:
    RanksPlacementT = Dict[str, List[Tuple[int, int]]]  # maps from name to (TP, DP)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def get_sharded_state_dict(self, ranks_placement: RanksPlacementT):
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()

        sharded_state_dict = {}
        for name, tps_dps in ranks_placement.items():
            if (tp_rank, dp_rank) in tps_dps:
                is_main = (tp_rank, dp_rank) == tps_dps[0]
                sharded_state_dict[name] = ShardedTensor.from_rank_offsets(
                    name, torch.ones(1), replica_id=int(not is_main)
                )

        return sharded_state_dict

    def test_full_dp_reads(self, tmp_path_dist_ckpt):
        """DP is the whole world."""
        ranks_placement = {'a': [(0, 0)], 'b': [(0, 1)], 'c': [(0, i) for i in range(8)]}
        cross_rank_reads, same_rank_reads = self.determine_cross_rank_reads(
            1, ranks_placement, tmp_path_dist_ckpt
        )

        # We expect no cross-DP reads
        assert not cross_rank_reads
        # `c` was assigned to rank 2 during saving because 0 and 1 already saved `a` and `b`
        if Utils.rank == 0:
            assert same_rank_reads == {'a': [0]}
        elif Utils.rank == 1:
            assert same_rank_reads == {'b': [1]}
        elif Utils.rank == 2:
            assert same_rank_reads == {'c': [2]}
        else:
            assert not same_rank_reads

    def test_singleton_dp_reads(self, tmp_path_dist_ckpt):
        """DP group has 1 rank (TP=8)."""
        ranks_placement = {'a': [(0, 0)], 'b': [(1, 0)], 'c': [(i, 0) for i in range(8)]}
        cross_rank_reads, same_rank_reads = self.determine_cross_rank_reads(
            8, ranks_placement, tmp_path_dist_ckpt
        )

        # We expect (unfortunately) a lot of cross-DP reads for `c` tensor.
        if Utils.rank != 0:
            assert cross_rank_reads == {'c': [0]}

        # `c` was assigned to rank 0 during saving because rank 0 belonged to the DP group
        # which held the main replica for `c`
        if Utils.rank == 0:
            assert same_rank_reads == {'a': [0], 'c': [0]}
        elif Utils.rank == 1:
            assert same_rank_reads == {'b': [1]}
        else:
            assert not same_rank_reads

    def test_out_of_order_load(self, tmp_path_dist_ckpt):
        """DP group has 8 rank (TP=1)."""
        ranks_placement = {'a': [(0, 2)]}
        cross_rank_reads, same_rank_reads = self.determine_cross_rank_reads(
            1, ranks_placement, tmp_path_dist_ckpt
        )
        assert not cross_rank_reads
        if Utils.rank == 2:
            assert same_rank_reads == {'a': [2]}

    def test_cross_dp_access_does_not_disturb_the_distribution(self, tmp_path_dist_ckpt):
        """Each DP group has 4 ranks (TP=2)."""

        # See `distribute_shards_to_ranks` algorithm for assignment explanation
        ranks_placement = {
            'a': [(0, 0)],  # saved by rank 0 obviously
            # main replica is in DP group with ranks [0, 2, 4, 6],
            # will be saved on rank 4 because 'c' is assigned first:
            'b': [(tp, dp) for tp in range(2) for dp in range(4)],
            # assigned before 'b' because of a smaller potential saving ranks count
            'c': [(0, dp) for dp in range(3)],
            # Here main replica is on rank 1 so will be saved by rank 1:
            'd': [(1, 0), (0, 0), (1, 3)],
            # Rank 1 saved 'd' so rank 5 saves
            'e': [(1, 0), (1, 2)],
            # Can be saved by DP group [1, 3, 5, 7],
            # round-robin back to rank 1
            'f': [(1, 0), (0, 0), (1, 2)],
            'g': [(1, 3)],  # saved by rank 7
        }
        # This dict encodes the comments above (who saves a given tensor)
        # Save order:
        # DP group 0: 'a', 'c', 'b'
        # DP group 1: 'g', 'd', 'e', 'f'
        expected_saving_ranks = {'a': 0, 'b': 4, 'c': 2, 'd': 1, 'e': 5, 'f': 1, 'g': 7}
        # Which tensors are cross-read (from a different rank) by each rank.
        # After assigning the intra-DP loads on the ranks according to the saving distribution,
        # the cross-DP reads are assigned. So first `a`, 'e' and `g` are assigned, then
        # rank 0 must cross-read 'd' and 'f'
        # and one of the ranks [1, 3, 5, 7] must cross-read 'b'. Rank 3 does that (first empty)
        expected_cross_load_ranks = {'d': 0, 'f': 0, 'b': 3}
        cross_rank_reads, same_rank_reads = self.determine_cross_rank_reads(
            2, ranks_placement, tmp_path_dist_ckpt
        )

        for key, saving_rank in expected_saving_ranks.items():
            # Check `Utils.rank == saving_rank` *iff* it's expected
            if Utils.rank == saving_rank:
                assert same_rank_reads[key] == [Utils.rank], saving_rank
            if same_rank_reads.get(key, []) == [Utils.rank]:
                assert Utils.rank == saving_rank, key

        torch.distributed.barrier()

        if Utils.rank == 0:
            assert cross_rank_reads == {
                'd': [expected_saving_ranks['d']],
                'f': [expected_saving_ranks['f']],
            }
        elif Utils.rank == 3:
            assert cross_rank_reads == {'b': [expected_saving_ranks['b']]}
        else:
            assert not cross_rank_reads

    def determine_cross_rank_reads(
        self,
        tp_size: int,
        ranks_placement: RanksPlacementT,
        tmp_path_dist_ckpt: Path,
        parallel_within_dp: bool = True,
    ):
        Utils.initialize_model_parallel(tp_size, 1)
        parallelization_group = (
            parallel_state.get_data_parallel_group()
            if parallel_within_dp
            else torch.distributed.group.WORLD
        )
        state_dict = self.get_sharded_state_dict(ranks_placement)
        with TempNamedDir(tmp_path_dist_ckpt / 'determine_cross_rank_reads') as ckpt_dir:
            save_strategy = FullyParallelSaveStrategyWrapper(
                get_default_save_sharded_strategy(), parallelization_group
            )
            save_strategy.save(state_dict, ckpt_dir)

            load_strategy = FullyParallelLoadStrategyWrapper(
                get_default_strategy(StrategyAction.LOAD_SHARDED, 'torch_dist', 1),
                parallelization_group,
                do_cache_distribution=True,
                exchange_algo='broadcast',
            )

            # Create a mock that will do what it's supposed to do,
            # but additionally collect info about cross-rank reads.
            cross_rank_reads = None
            same_rank_reads = None

            def mock_local_plan(self):
                self._validate_global_shapes(self.metadata, self.shapes_validation_sharded_tensors)
                local_plan = super(MCoreLoadPlanner, self).create_local_plan()

                nonlocal cross_rank_reads
                nonlocal same_rank_reads
                cross_rank_reads = defaultdict(list)
                same_rank_reads = defaultdict(list)

                # Debug cross-reads
                for read_item in local_plan.items:
                    item_md = self.metadata.storage_data[read_item.storage_index]

                    read_rank = int(item_md.relative_path.split('_')[2])
                    if read_rank == torch.distributed.get_rank():
                        same_rank_reads[read_item.dest_index.fqn].append(read_rank)
                    else:
                        cross_rank_reads[read_item.dest_index.fqn].append(read_rank)

                return local_plan

            with mock.patch.object(MCoreLoadPlanner, 'create_local_plan', mock_local_plan):
                _ = load_strategy.load(state_dict, ckpt_dir)

        Utils.destroy_model_parallel()

        return cross_rank_reads, same_rank_reads
