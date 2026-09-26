# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layout-aware global plan validation (coordinator side of torch_dist saves)."""

import itertools
import logging
import random
import time

import pytest
import torch
import torch.distributed.checkpoint.default_planner as default_planner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import SavePlan

from megatron.core.dist_checkpointing import ShardedObject, ShardedTensor, load, save
from megatron.core.dist_checkpointing.strategies import global_plan_validation
from megatron.core.dist_checkpointing.strategies import torch as torch_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.global_plan_validation import (
    find_overlapping_chunks,
    validate_global_plan,
)
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _chunk(offsets, sizes):
    return ChunkStorageMetadata(offsets=torch.Size(offsets), sizes=torch.Size(sizes))


def _grid(size, frags, ranges_per_cell=None, rng=None):
    """Regular grid of chunks; optionally split each cell's last dim into random ranges."""
    cell = [s // f for s, f in zip(size, frags)]
    chunks = []
    for idx in itertools.product(*[range(f) for f in frags]):
        offsets = [i * c for i, c in zip(idx, cell)]
        if ranges_per_cell is None:
            chunks.append(_chunk(offsets, cell))
            continue
        last = cell[-1]
        cuts = (
            sorted(rng.sample(range(1, last), ranges_per_cell - 1))
            if last > ranges_per_cell
            else []
        )
        bounds = [0, *cuts, last]
        for a, b in zip(bounds, bounds[1:]):
            chunks.append(_chunk(offsets[:-1] + [offsets[-1] + a], cell[:-1] + [b - a]))
    return chunks


def _metadata(chunks_by_key, sizes):
    md = {}
    for key, chunks in chunks_by_key.items():
        if chunks is None:
            md[key] = BytesStorageMetadata()
        else:
            md[key] = TensorStorageMetadata(
                properties=TensorProperties(dtype=torch.float32),
                size=torch.Size(sizes[key]),
                chunks=list(chunks),
            )
    return Metadata(state_dict_metadata=md)


def _torch_is_valid(global_plan, metadata):
    result = default_planner._validate_global_plan(global_plan, metadata)
    # torch < 2.13 returns a bool, newer versions return the list of errors
    return (len(result) == 0) if isinstance(result, list) else bool(result)


def _unordered_pairs(pairs):
    return {
        frozenset([(tuple(a.offsets), tuple(a.sizes)), (tuple(b.offsets), tuple(b.sizes))])
        for a, b in pairs
    }


class TestGlobalPlanValidation:
    """CPU-only checks against torch's reference implementation."""

    def setup_method(self, method):
        logging.getLogger('torch.distributed.checkpoint.default_planner').setLevel(logging.ERROR)
        logging.getLogger(global_plan_validation.__name__).setLevel(logging.ERROR)

    def test_matches_torch_on_random_layouts(self):
        rng = random.Random(0)
        for _ in range(400):
            ndim = rng.choice([1, 2, 3, 4])
            size = [rng.choice([2, 4, 6, 8, 12]) * rng.choice([1, 2, 5]) for _ in range(ndim)]
            frags = [rng.choice([f for f in (1, 2, 3, 4) if size[d] % f == 0]) for d in range(ndim)]
            flat = rng.random() < 0.3 and size[-1] // frags[-1] >= 3
            chunks = _grid(
                size, frags, ranges_per_cell=rng.choice([2, 3]) if flat else None, rng=rng
            )
            mutation = rng.choice(['none', 'dup', 'overlap', 'gap', 'oob', 'zero'])
            if mutation == 'dup':
                chunks.append(rng.choice(chunks))
            elif mutation == 'overlap':
                c = rng.choice(chunks)
                offs = list(c.offsets)
                d = rng.randrange(ndim)
                offs[d] = max(0, offs[d] - 1)
                chunks.append(_chunk(offs, c.sizes))
            elif mutation == 'gap' and len(chunks) > 1:
                chunks.pop(rng.randrange(len(chunks)))
            elif mutation == 'oob':
                c = chunks[-1]
                offs = list(c.offsets)
                offs[-1] += 1
                chunks[-1] = _chunk(offs, c.sizes)
            elif mutation == 'zero':
                c = rng.choice(chunks)
                sizes = list(c.sizes)
                sizes[rng.randrange(ndim)] = 0
                chunks.append(_chunk(c.offsets, sizes))
            rng.shuffle(chunks)
            md = _metadata(
                {'t': chunks, 'scalar': [_chunk([], [])], 'bytes': None}, {'t': size, 'scalar': []}
            )
            plans = [SavePlan(items=[]) for _ in range(rng.choice([1, 2, 4]))]

            assert (len(validate_global_plan(plans, md)) == 0) == _torch_is_valid(plans, md), (
                size,
                frags,
                mutation,
            )
            reference = {
                frozenset([(tuple(a.offsets), tuple(a.sizes)), (tuple(b.offsets), tuple(b.sizes))])
                for i, a in enumerate(chunks)
                for b in chunks[i + 1 :]
                if default_planner._check_box_overlap(a, b)
            }
            assert _unordered_pairs(find_overlapping_chunks(chunks)) == reference, (
                size,
                frags,
                mutation,
            )

    def test_detects_each_violation(self):
        size = [8, 6]
        chunks = _grid(size, [4, 2])
        plans = [SavePlan(items=[]), SavePlan(items=[])]
        assert validate_global_plan(plans, _metadata({'t': chunks}, {'t': size})) == []
        # duplicate chunk -> overlap + volume error
        errors = validate_global_plan(plans, _metadata({'t': chunks + [chunks[0]]}, {'t': size}))
        assert any('overlapping' in e for e in errors) and any('invalid fill' in e for e in errors)
        # missing chunk -> volume error only
        errors = validate_global_plan(plans, _metadata({'t': chunks[1:]}, {'t': size}))
        assert errors and all('invalid fill' in e for e in errors)
        # out-of-bounds chunk
        oob = chunks[:-1] + [_chunk([6, 3], [3, 3])]
        errors = validate_global_plan(plans, _metadata({'t': oob}, {'t': size}))
        assert any('out of bounds' in e for e in errors)
        # single plan: coverage is not required
        assert validate_global_plan([plans[0]], _metadata({'t': chunks[1:]}, {'t': size})) == []

    def test_moe_expert_layout_is_fast(self):
        # fc2 expert weight of a 512-expert MoE at EGTP 8 -> 4096 chunks of (1, 384, 10240) in a
        # (512, 3072, 10240) tensor. torch sweeps the largest (unsharded) dim and takes ~1 s;
        # the layout-aware check must stay well under that.
        size, frags = [512, 3072, 10240], [512, 8, 1]
        chunks = _grid(size, frags)
        md = _metadata({'experts.linear_fc2.weight': chunks}, {'experts.linear_fc2.weight': size})
        plans = [SavePlan(items=[]), SavePlan(items=[])]
        start = time.perf_counter()
        assert validate_global_plan(plans, md) == []
        assert time.perf_counter() - start < 0.5

    def test_flattened_ranges_with_rank_specific_boundaries(self):
        # Two fragments along dim 0, each split into ranges on the flattened axis at different
        # points: the "single irregular dimension" path.
        chunks = [
            _chunk([0, 0], [1, 700]),
            _chunk([0, 700], [1, 324]),
            _chunk([1, 0], [1, 300]),
            _chunk([1, 300], [1, 724]),
        ]
        size = [2, 1024]
        plans = [SavePlan(items=[]), SavePlan(items=[])]
        assert validate_global_plan(plans, _metadata({'t': chunks}, {'t': size})) == []
        overlapping = chunks + [_chunk([1, 250], [1, 100])]
        errors = validate_global_plan(plans, _metadata({'t': overlapping}, {'t': size}))
        assert sum('overlapping' in e for e in errors) == 2


class TestMCoreSavePlannerUsesFastValidation:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _state_dict(self, n_experts):
        """Grouped-expert-like layout: expert axis prepended, axis 0 split across all ranks."""
        sd = {}
        for layer in range(2):
            key = f'decoder.layers.{layer}.mlp.experts.linear_fc1.weight'
            for e in range(n_experts):
                data = torch.full((8, 4), float(Utils.rank * 100 + layer * 10 + e), device='cuda')
                sd[f'{key}.{e}'] = ShardedTensor.from_rank_offsets(
                    key,
                    data,
                    (0, e, n_experts),
                    (1, Utils.rank, Utils.world_size),
                    replica_id=0,
                    prepend_axis_num=1,
                )
        sd['ln'] = ShardedTensor.from_rank_offsets(
            'decoder.final_layernorm.weight', torch.ones(16, device='cuda'), replica_id=Utils.rank
        )
        sd['obj'] = ShardedObject(
            'rng_state', {'rank': Utils.rank}, (Utils.world_size,), (Utils.rank,), replica_id=0
        )
        return sd

    def test_coordinator_runs_fast_validation_once(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(1, 1)
        calls = []
        original = torch_strategy.validate_global_plan

        def counting(global_plan, metadata):
            calls.append(set(metadata.state_dict_metadata))
            return original(global_plan, metadata)

        torch_strategy.validate_global_plan = counting
        try:
            sd = self._state_dict(n_experts=4)
            strategy = FullyParallelSaveStrategyWrapper(
                TorchDistSaveShardedStrategy('torch_dist', 1),
                do_cache_distribution=True,
                validate_access_integrity=False,
            )
            with TempNamedDir(tmp_path_dist_ckpt / 'test_fast_global_plan') as ckpt_dir:
                save(sd, ckpt_dir, strategy, validate_access_integrity=False)
                # Only the coordinator (global rank 0) merges the plans.
                expected = 1 if torch.distributed.get_rank() == 0 else 0
                assert len(calls) == expected
                if calls:
                    # Both grouped-expert keys are single metadata entries with
                    # n_experts * world_size chunks each. ShardedObjects are stored under
                    # '<key>/shard_<offset>_<shape>' names.
                    keys = calls[0]
                    assert {
                        'decoder.layers.0.mlp.experts.linear_fc1.weight',
                        'decoder.layers.1.mlp.experts.linear_fc1.weight',
                        'decoder.final_layernorm.weight',
                    } <= keys, keys
                    assert any(k.startswith('rng_state/') for k in keys), keys
                    assert any(k.startswith('common_state/') for k in keys), keys
                template = self._state_dict(n_experts=4)
                for value in template.values():
                    if isinstance(value, ShardedTensor):
                        value.data = torch.empty_like(value.data)
                loaded = load(template, ckpt_dir, validate_access_integrity=False)
                for key, value in sd.items():
                    if isinstance(value, ShardedTensor):
                        # save() re-views the saved data with the prepended axis in place
                        assert torch.equal(loaded[key].flatten(), value.data.flatten()), key
                    else:
                        assert loaded[key] == value.data, key
        finally:
            torch_strategy.validate_global_plan = original
