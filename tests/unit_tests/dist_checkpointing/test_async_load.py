# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Tests for the async distributed checkpoint load path."""

import time

import pytest
import torch

from megatron.core.dist_checkpointing import (
    ShardedTensor,
    load,
    prepare_async_load,
    prepare_async_load_reusing_topology,
    save,
    start_async_load_from_plan,
)
from megatron.core.dist_checkpointing.async_load_manager import AsyncCheckpointLoader
from megatron.core.dist_checkpointing.cpu_shadow import ShadowBufferPool
from megatron.core.dist_checkpointing.dict_utils import diff
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def sharded_state_dict(fill_value: float):
    """Sharded state dict with rank-dependent shards and a common entry."""
    return {
        'sd_keyA': ShardedTensor.from_rank_offsets(
            'keyA', torch.full((2, 4), fill_value) + Utils.rank, (0, Utils.rank, Utils.world_size)
        ),
        'sd_keyB': ShardedTensor.from_rank_offsets(
            'keyB',
            torch.arange(3 * 5 * 7, dtype=torch.float).reshape(3, 5, 7) * fill_value,
            (2, Utils.rank, Utils.world_size),
        ),
        'lr': 0.01,
    }


def assert_state_dicts_equal(actual, expected):
    diffs = diff(actual, expected)
    assert not any(len(x) for x in diffs), diffs


class TestAsyncLoad:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_async_load_matches_sync_load(self, tmp_path_dist_ckpt):
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_async_load_matches_sync_load', sync=True
        ) as ckpt_dir:
            save(sharded_state_dict(1.0), ckpt_dir)
            torch.distributed.barrier()

            sync_loaded = load(sharded_state_dict(0.0), ckpt_dir)

            plan = prepare_async_load(sharded_state_dict(0.0), ckpt_dir)
            request = start_async_load_from_plan(plan)
            async_loaded = request.maybe_finalize(blocking=True)

            assert_state_dicts_equal(async_loaded, sync_loaded)

    def test_async_load_non_blocking_poll(self, tmp_path_dist_ckpt):
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_async_load_non_blocking_poll', sync=True
        ) as ckpt_dir:
            save(sharded_state_dict(2.0), ckpt_dir)
            torch.distributed.barrier()

            sync_loaded = load(sharded_state_dict(0.0), ckpt_dir)

            plan = prepare_async_load(sharded_state_dict(0.0), ckpt_dir)
            request = start_async_load_from_plan(plan)
            async_loaded = None
            for _ in range(6000):
                async_loaded = request.maybe_finalize(blocking=False)
                if async_loaded is not None:
                    break
                time.sleep(0.01)
            assert async_loaded is not None, 'async load did not complete in time'
            assert request.is_done()

            assert_state_dicts_equal(async_loaded, sync_loaded)

    def test_reuse_topology_across_checkpoints(self, tmp_path_dist_ckpt):
        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_reuse_topology_A', sync=True) as ckpt_dir_a,
            TempNamedDir(tmp_path_dist_ckpt / 'test_reuse_topology_B', sync=True) as ckpt_dir_b,
        ):
            save(sharded_state_dict(1.0), ckpt_dir_a)
            save(sharded_state_dict(3.0), ckpt_dir_b)
            torch.distributed.barrier()

            sync_loaded_a = load(sharded_state_dict(0.0), ckpt_dir_a)
            sync_loaded_b = load(sharded_state_dict(0.0), ckpt_dir_b)

            # Template plan pays the collective planning cost once...
            template_plan = prepare_async_load(sharded_state_dict(0.0), ckpt_dir_a)
            request_a = start_async_load_from_plan(template_plan)
            async_loaded_a = request_a.maybe_finalize(blocking=True)
            assert_state_dicts_equal(async_loaded_a, sync_loaded_a)

            # ...and the second checkpoint reuses it with local-only setup.
            plan_b = prepare_async_load_reusing_topology(
                sharded_state_dict(0.0), ckpt_dir_b, template_plan
            )
            request_b = start_async_load_from_plan(plan_b)
            async_loaded_b = request_b.maybe_finalize(blocking=True)
            assert_state_dicts_equal(async_loaded_b, sync_loaded_b)

    @pytest.mark.skipif(Utils.world_size < 2, reason='call_idx mismatch requires at least 2 ranks')
    def test_maybe_finalize_call_idx_mismatch_raises(self, tmp_path_dist_ckpt):
        with TempNamedDir(tmp_path_dist_ckpt / 'test_call_idx_mismatch', sync=True) as ckpt_dir:
            save(sharded_state_dict(1.0), ckpt_dir)
            torch.distributed.barrier()

            plan = prepare_async_load(sharded_state_dict(0.0), ckpt_dir)
            request = start_async_load_from_plan(plan, call_idx=int(Utils.rank == 0))
            with pytest.raises(RuntimeError, match='call_idx mismatch'):
                request.maybe_finalize(blocking=True)


class _DummyModelChunk(torch.nn.Module):
    """Minimal module exposing a ShardedTensor sharded_state_dict for the loader."""

    def __init__(self, fill_value: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((2, 4), fill_value) + Utils.rank)

    def sharded_state_dict(self):
        return {
            'weight': ShardedTensor.from_rank_offsets(
                'weight', self.weight.data, (0, Utils.rank, Utils.world_size)
            )
        }


def _save_ckpt_with_tracker(base_dir, state_dict, iteration=1):
    """Save ``state_dict`` under ``base_dir/iter_*`` and write the tracker file,
    mirroring Megatron's on-disk layout so the loader's resolver works."""
    iter_dir = base_dir / f'iter_{iteration:07d}'
    if Utils.rank == 0:
        iter_dir.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    save(state_dict, iter_dir)
    if Utils.rank == 0:
        (base_dir / 'latest_checkpointed_iteration.txt').write_text(str(iteration))
    torch.distributed.barrier()


class TestShadowBufferPool:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_acquire_release_and_exhaustion(self):
        pool = ShadowBufferPool([_DummyModelChunk(1.0)], num_buffers=1)
        assert pool.num_buffers() == 1 and pool.num_free() == 1

        shadow = pool.acquire()
        assert pool.num_free() == 0
        with pytest.raises(RuntimeError, match='exhausted'):
            pool.acquire()

        pool.release(shadow)
        assert pool.num_free() == 1
        with pytest.raises(RuntimeError, match='twice'):
            pool.release(shadow)

    def test_foreign_shadow_rejected(self):
        pool = ShadowBufferPool([_DummyModelChunk(1.0)], num_buffers=1)
        with pytest.raises(RuntimeError, match='foreign'):
            pool.release({'model': {}})


class TestAsyncCheckpointLoader:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_kick_finalize_release_into_model(self, tmp_path_dist_ckpt):
        saved_model = [_DummyModelChunk(5.0)]
        with TempNamedDir(tmp_path_dist_ckpt / 'test_loader_e2e', sync=True) as base_dir:
            _save_ckpt_with_tracker(base_dir, saved_model[0].sharded_state_dict())

            # Fresh model with different weights; the load must overwrite them.
            target_model = [_DummyModelChunk(0.0)]
            loader = AsyncCheckpointLoader(target_model)

            handle = loader.kick('ckpt', str(base_dir))
            finalized = loader.finalize(handle)
            loader.load_finalized_to_model(finalized, strict=True)
            loader.release(handle)

            expected = torch.full((2, 4), 5.0) + Utils.rank
            assert torch.equal(target_model[0].weight.data.cpu(), expected)

    def test_reusing_topology_plan_cache_hit(self, tmp_path_dist_ckpt):
        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_loader_cache_a', sync=True) as base_a,
            TempNamedDir(tmp_path_dist_ckpt / 'test_loader_cache_b', sync=True) as base_b,
        ):
            _save_ckpt_with_tracker(base_a, _DummyModelChunk(1.0).sharded_state_dict())
            _save_ckpt_with_tracker(base_b, _DummyModelChunk(2.0).sharded_state_dict())

            loader = AsyncCheckpointLoader([_DummyModelChunk(0.0)])

            for key, base, fill in (('a', base_a, 1.0), ('b', base_b, 2.0)):
                handle = loader.kick(key, str(base))
                finalized = loader.finalize(handle)
                loader.load_finalized_to_model(finalized, strict=True)
                loader.release(handle)
                expected = torch.full((2, 4), fill) + Utils.rank
                assert torch.equal(loader._model[0].weight.data.cpu(), expected)

            # Same topology -> one plan built, second load is a cache hit.
            stats = loader.plan_cache_stats()
            assert stats['distinct_plans'] == 1
            assert stats['hits'] == 1 and stats['misses'] == 1
