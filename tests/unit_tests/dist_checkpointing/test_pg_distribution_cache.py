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

import torch

from megatron.core.dist_checkpointing import ShardedTensor, exchange_utils
from megatron.core.dist_checkpointing.exchange_utils import (
    _pg_dist_cache_file_path,
    determine_main_replica_uniform_distribution,
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
