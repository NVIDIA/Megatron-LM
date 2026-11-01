# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the PG-distribution disk cache (``--ckpt-pg-tensors-cache-path``).

The FullyParallel save/load distribution is a deterministic function of the
sharded-state-dict structure and the parallel layout, computed via a single
``all_gather_object`` (``_gather_shards_metadata``). The cache persists that
result so later jobs can:
  * CREATE (``pg_cache_create=True``): gather once and write the distribution.
  * READ  (``pg_cache_create=False``): load it from disk and skip the gather.

These tests run at world_size=2 (TP=2) but make no world-size-specific
assertions, so they hold at any size.
"""

import os

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, exchange_utils
from megatron.core.dist_checkpointing.exchange_utils import (
    _pg_dist_cache_file_path,
    determine_main_replica_uniform_distribution,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _state_dict():
    # Fully-replicated tensors of different sizes; varying replica_id makes
    # different ranks the main replica, so the save distribution is non-trivial.
    return {
        f"sd_key{i}": ShardedTensor.from_rank_offsets(
            f"key{i}", torch.ones(10 * (i + 1)), replica_id=(Utils.rank + i) % Utils.world_size
        )
        for i in range(4)
    }


def _rank_marked_state_dict():
    """Replicated shards whose *content* identifies the rank holding them.

    Replicas normally carry identical data, which makes the elected main replica
    unobservable in the loaded values. Marking each rank's copy with its own rank
    makes the save distribution visible end-to-end: the loaded value tells you
    which rank the distribution elected to write that shard.
    """
    return {
        f"sd_key{i}": ShardedTensor.from_rank_offsets(
            f"key{i}",
            torch.full((10 * (i + 1),), float(Utils.rank)),
            replica_id=(Utils.rank + i) % Utils.world_size,
        )
        for i in range(4)
    }


class TestPgDistributionCache:
    def setup_method(self, method):
        Utils.destroy_model_parallel()
        exchange_utils._PG_DIST_CACHE.clear()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        exchange_utils._PG_DIST_CACHE.clear()

    def test_create_writes_cache_and_read_matches(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 1)
        group = torch.distributed.group.WORLD
        with TempNamedDir(tmp_path_dist_ckpt / "pg_cache") as cache_dir:
            cache_path = str(cache_dir)

            # CREATE: runs the gather once and writes the cache file.
            created = determine_main_replica_uniform_distribution(
                _state_dict(), group, pg_cache_path=cache_path, pg_cache_create=True
            )
            torch.distributed.barrier()
            assert os.path.exists(_pg_dist_cache_file_path(cache_path, group))

            # Drop the in-process memo so the next call truly reads from disk.
            exchange_utils._PG_DIST_CACHE.clear()

            # READ: loads the distribution from disk.
            read = determine_main_replica_uniform_distribution(
                _state_dict(), group, pg_cache_path=cache_path, pg_cache_create=False
            )

            # The deterministic distribution decisions round-trip exactly.
            assert read.main_rank_for_shard == created.main_rank_for_shard
            assert read.all_ranks_for_shard == created.all_ranks_for_shard

    def test_read_path_skips_the_gather_collective(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 1)
        group = torch.distributed.group.WORLD
        with TempNamedDir(tmp_path_dist_ckpt / "pg_cache_skip") as cache_dir:
            cache_path = str(cache_dir)

            created = determine_main_replica_uniform_distribution(
                _state_dict(), group, pg_cache_path=cache_path, pg_cache_create=True
            )
            torch.distributed.barrier()
            exchange_utils._PG_DIST_CACHE.clear()

            # Make the metadata-exchange collective explode; the READ path must
            # never reach it.
            orig_gather = exchange_utils._gather_shards_metadata

            def _boom(*args, **kwargs):
                raise AssertionError("READ path must not run the all_gather_object collective")

            exchange_utils._gather_shards_metadata = _boom
            try:
                read = determine_main_replica_uniform_distribution(
                    _state_dict(), group, pg_cache_path=cache_path, pg_cache_create=False
                )
            finally:
                exchange_utils._gather_shards_metadata = orig_gather

            assert read is not None
            assert read.main_rank_for_shard == created.main_rank_for_shard

    def test_checkpoint_roundtrip_matches_without_cache(self, tmp_path_dist_ckpt):
        """A real save+load with the cache must return exactly what it does without it.

        The cache only replaces *how* the save/load distribution is obtained, so a
        checkpoint written and read back through it must be indistinguishable from
        the default path. This exercises the full FullyParallel save -> load
        round-trip, which is what would break if a cached distribution were applied
        to the wrong shards.
        """
        Utils.initialize_model_parallel(2, 1)
        group = torch.distributed.group.WORLD

        # Reference (no cache): plain save + load.
        with TempNamedDir(tmp_path_dist_ckpt / "rt_plain", sync=True) as plain_dir:
            FullyParallelSaveStrategyWrapper(TorchDistSaveShardedStrategy(), group).save(
                _rank_marked_state_dict(), plain_dir
            )
            torch.distributed.barrier()
            plain_loaded = FullyParallelLoadStrategyWrapper(
                TorchDistLoadShardedStrategy(), group
            ).load(_rank_marked_state_dict(), plain_dir)

        exchange_utils._PG_DIST_CACHE.clear()

        # Same round-trip, but the distribution is written to and then read back
        # from the on-disk cache.
        with TempNamedDir(tmp_path_dist_ckpt / "rt_cache_dir", sync=True) as cache_dir:
            cache_path = str(cache_dir)
            with TempNamedDir(tmp_path_dist_ckpt / "rt_cached", sync=True) as cached_ckpt_dir:
                # CREATE: this save computes and persists the distribution.
                FullyParallelSaveStrategyWrapper(
                    TorchDistSaveShardedStrategy(),
                    group,
                    pg_cache_path=cache_path,
                    pg_cache_create=True,
                ).save(_rank_marked_state_dict(), cached_ckpt_dir)
                torch.distributed.barrier()

                # The create-mode save must have persisted the distribution, and
                # dropping the in-process memo forces the load to read that file.
                assert os.path.exists(_pg_dist_cache_file_path(cache_path, group))
                exchange_utils._PG_DIST_CACHE.clear()

                # READ: the load takes the distribution from disk. Fail loudly if it
                # falls back to the collective, which would make this test vacuous.
                orig_gather = exchange_utils._gather_shards_metadata

                def _boom(*args, **kwargs):
                    raise AssertionError("cached load must not run the all_gather_object")

                exchange_utils._gather_shards_metadata = _boom
                try:
                    cached_loaded = FullyParallelLoadStrategyWrapper(
                        TorchDistLoadShardedStrategy(), group, pg_cache_path=cache_path
                    ).load(_rank_marked_state_dict(), cached_ckpt_dir)
                finally:
                    exchange_utils._gather_shards_metadata = orig_gather

        assert cached_loaded.keys() == plain_loaded.keys()
        for key, expected in plain_loaded.items():
            actual = cached_loaded[key]
            assert torch.equal(actual, expected), f"mismatch for {key}"
        # Guard against a vacuous comparison: each loaded shard must be a
        # rank-marked tensor of the expected shape (a single rank's copy).
        for i, key in enumerate(sorted(plain_loaded)):
            shard = plain_loaded[key]
            assert shard.shape == (10 * (i + 1),)
            assert torch.equal(shard, torch.full_like(shard, shard[0].item()))
            assert 0 <= shard[0].item() < Utils.world_size

    def test_groups_sharing_min_rank_use_distinct_cache_entries(self, tmp_path_dist_ckpt):
        """Distinct groups that share a minimum rank must not share a cache entry.

        A data-parallel group over ranks [0, 1, 2, 3] and expert-data-parallel style
        subgroups [0, 2] and [1, 3] share minimum ranks 0 and 1. If the cache were
        keyed by the minimum rank, creating the subgroup cache would overwrite the
        full group's file, and a later read for the full group would return the
        subgroup's distribution.
        """
        if Utils.world_size < 4:
            pytest.skip("Requires at least 4 ranks.")
        Utils.initialize_model_parallel(1, 1)
        rank = torch.distributed.get_rank()

        # new_group is collective over the default group: every rank creates every
        # group in the same order, including ranks that are not members.
        full_ranks = [0, 1, 2, 3]
        full_group = torch.distributed.new_group(full_ranks)
        sub_ranks_list = [[0, 2], [1, 3]]
        sub_groups = [torch.distributed.new_group(ranks) for ranks in sub_ranks_list]
        my_sub_group = None
        for ranks, group in zip(sub_ranks_list, sub_groups):
            if rank in ranks:
                my_sub_group = group
        in_full_group = rank in full_ranks

        with TempNamedDir(tmp_path_dist_ckpt / "pg_cache_min_rank", sync=True) as cache_dir:
            cache_path = str(cache_dir)

            # CREATE for the full group, then (with a fresh memo, as in a separate
            # job) for the subgroups that share its minimum ranks.
            if in_full_group:
                determine_main_replica_uniform_distribution(
                    _state_dict(), full_group, pg_cache_path=cache_path, pg_cache_create=True
                )
            torch.distributed.barrier()
            exchange_utils._PG_DIST_CACHE.clear()
            if my_sub_group is not None:
                determine_main_replica_uniform_distribution(
                    _state_dict(), my_sub_group, pg_cache_path=cache_path, pg_cache_create=True
                )
            torch.distributed.barrier()
            exchange_utils._PG_DIST_CACHE.clear()

            # Each group has its own file, written once by its minimum-rank member.
            if rank == 0:
                cache_files = sorted(os.listdir(cache_path))
                assert len(cache_files) == 1 + len(sub_ranks_list), cache_files
            if in_full_group:
                full_file = _pg_dist_cache_file_path(cache_path, full_group)
                sub_file = _pg_dist_cache_file_path(cache_path, my_sub_group)
                assert full_file != sub_file
                assert os.path.exists(full_file)
                assert os.path.exists(sub_file)

            # READ each group's distribution from disk and compare it with the
            # distribution computed without the cache for the same group.
            if in_full_group:
                read_full = determine_main_replica_uniform_distribution(
                    _state_dict(), full_group, pg_cache_path=cache_path, pg_cache_create=False
                )
                read_sub = determine_main_replica_uniform_distribution(
                    _state_dict(), my_sub_group, pg_cache_path=cache_path, pg_cache_create=False
                )
                expected_full = determine_main_replica_uniform_distribution(
                    _state_dict(), full_group
                )
                expected_sub = determine_main_replica_uniform_distribution(
                    _state_dict(), my_sub_group
                )
                assert read_full.main_rank_for_shard == expected_full.main_rank_for_shard
                assert read_full.all_ranks_for_shard == expected_full.all_ranks_for_shard
                assert read_sub.main_rank_for_shard == expected_sub.main_rank_for_shard
                assert read_sub.all_ranks_for_shard == expected_sub.all_ranks_for_shard
                # Guard against a vacuous comparison: the two layouts differ.
                assert read_full.all_ranks_for_shard != read_sub.all_ranks_for_shard
            torch.distributed.barrier()
