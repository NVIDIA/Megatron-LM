# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for LayerWiseDistributedOptimizer parameter layout computation.

These tests verify the size-matching shard-aligned layout logic without
requiring GPU or distributed setup.
"""

import math
from collections import Counter
from unittest import mock

import pytest
import torch

from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.param_layout import BufferKey, pad_param_start, pad_to_divisor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LWO = LayerWiseDistributedOptimizer


def _make_param(shape, dtype=torch.bfloat16, **attrs):
    p = torch.nn.Parameter(torch.randn(shape, dtype=dtype))
    p.use_layerwise_distributed_optimizer = True
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


def _make_ddp_config(pad_for_high_busbw=False, grad_reduce_in_fp32=True):
    cfg = mock.Mock()
    cfg.pad_buckets_for_high_nccl_busbw = pad_for_high_busbw
    cfg.grad_reduce_in_fp32 = grad_reduce_in_fp32
    return cfg


# ---------------------------------------------------------------------------
# Tests for _shard_divisor
# ---------------------------------------------------------------------------


class TestShardDivisor:

    def _verify(self, dp_size, high_busbw=False):
        cfg = _make_ddp_config(pad_for_high_busbw=high_busbw)
        sd = _LWO._shard_divisor(dp_size, cfg)
        assert sd % 64 == 0, f"shard_divisor {sd} not 64-aligned"
        if high_busbw:
            bucket_div = math.lcm(dp_size, 128, 2**16)
        else:
            bucket_div = math.lcm(dp_size, 128)
        assert (dp_size * sd) % bucket_div == 0

    def test_dp2(self):
        self._verify(2)

    def test_dp4(self):
        self._verify(4)

    def test_dp8(self):
        self._verify(8)

    def test_dp8_high_busbw(self):
        self._verify(8, high_busbw=True)

    def test_dp1(self):
        self._verify(1)


# ---------------------------------------------------------------------------
# Helpers for layout verification
# ---------------------------------------------------------------------------


def _get_shard_for_param(layout, param, dp_size):
    """Return which shard index a param lands in."""
    start, end, bucket_id = layout.param_index_map[param]
    bstart, bend = layout.bucket_indices[bucket_id]
    shard_size = (bend - bstart) // dp_size
    shard_idx = (start - bstart) // shard_size
    return shard_idx


def _assert_param_within_shard(layout, param, dp_size):
    """Assert that a param lies entirely within one shard."""
    start, end, bucket_id = layout.param_index_map[param]
    bstart, bend = layout.bucket_indices[bucket_id]
    shard_size = (bend - bstart) // dp_size
    shard_idx = (start - bstart) // shard_size
    shard_start = bstart + shard_idx * shard_size
    shard_end = shard_start + shard_size
    assert shard_start <= start, f"param start {start} before shard start {shard_start}"
    assert end <= shard_end, f"param end {end} past shard end {shard_end}"


# ---------------------------------------------------------------------------
# Tests for _compute_per_buffer_param_layout (size-matching)
# ---------------------------------------------------------------------------


class TestSizeMatchingLayout:

    # -- uniform params: all same size, dp_size divides count --

    def test_uniform_params_exact_fit(self):
        """8 same-size params with dp_size=4 → each shard gets 2 params."""
        dp_size = 4
        params = [_make_param((256,)) for _ in range(8)]
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        for p in params:
            _assert_param_within_shard(layout, p, dp_size)

        # With 8 params and dp_size=4, 2 rounds of size-matching.
        # Each round fills all 4 shards.  No padding needed.
        shard_counts = Counter(_get_shard_for_param(layout, p, dp_size) for p in params)
        assert set(shard_counts.values()) == {2}

    def test_uniform_params_remainder_gets_padding(self):
        """5 same-size params with dp_size=4 → 1 round fills 4, 1 round fills 1 + 3 padding."""
        dp_size = 4
        params = [_make_param((256,)) for _ in range(5)]
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        for p in params:
            _assert_param_within_shard(layout, p, dp_size)

        # All 5 params assigned; shard 0 gets 2, shards 1-3 get 1 each.
        shard_counts = Counter(_get_shard_for_param(layout, p, dp_size) for p in params)
        assert shard_counts[0] == 2
        assert sum(shard_counts.values()) == 5

    # -- mixed sizes --

    def test_mixed_sizes_no_param_split(self):
        """Params with different sizes: each param stays within one shard."""
        dp_size = 2
        params = [
            _make_param((100,)),
            _make_param((200,)),
            _make_param((100,)),
            _make_param((200,)),
            _make_param((100,)),
        ]
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        for p in params:
            _assert_param_within_shard(layout, p, dp_size)

    def test_size_matching_prefers_same_size(self):
        """When shard 0 gets a 256-elem param, other shards should also get 256-elem params."""
        dp_size = 4
        big = [_make_param((256,)) for _ in range(4)]
        small = [_make_param((64,)) for _ in range(4)]
        # Interleave: big[0], small[0], big[1], small[1], ...
        params = []
        for b, s in zip(big, small):
            params.extend([b, s])
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        for p in params:
            _assert_param_within_shard(layout, p, dp_size)

        # All 4 big params should be in the same round (one per shard).
        big_buckets = set()
        for p in big:
            _, _, bid = layout.param_index_map[p]
            big_buckets.add(bid)
        # They should share a bucket (matched in one round).
        assert len(big_buckets) == 1

    # -- shared_embedding isolation --

    def test_shared_embedding_isolated(self):
        """shared_embedding params go in their own bucket."""
        dp_size = 2
        regular = [_make_param((128,)) for _ in range(4)]
        emb = _make_param((128,), shared_embedding=True)
        params = [emb] + regular
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        _, _, emb_bucket = layout.param_index_map[emb]
        for p in regular:
            _, _, bid = layout.param_index_map[p]
            assert bid != emb_bucket, "shared_embedding should be in its own bucket"

    # -- bucket size threshold --

    def test_bucket_size_creates_multiple_buckets(self):
        """When bucket_size is small, multiple buckets are created."""
        dp_size = 2
        params = [_make_param((256,)) for _ in range(8)]
        cfg = _make_ddp_config()

        # Each round: 256 elements per shard.  shard_pos after round = 256.
        # Padded shard size = pad_to_divisor(256, shard_div).
        # bucket_total = dp_size * padded_shard_size.
        # Set bucket_size small enough to force a split after 1 round.
        shard_div = _LWO._shard_divisor(dp_size, cfg)
        padded = pad_to_divisor(256, shard_div)
        small_bucket = dp_size * padded  # triggers after 1 round

        layout = _LWO._compute_per_buffer_param_layout(params, small_bucket, dp_size, cfg)

        assert len(layout.bucket_indices) == 4  # 8 params / (dp_size per round) = 4 rounds

    # -- bucket alignment --

    def test_bucket_dp_divisible(self):
        """Every bucket total must be divisible by dp_size."""
        for dp_size in [2, 4, 8]:
            params = [_make_param((333,)) for _ in range(dp_size * 2)]
            cfg = _make_ddp_config()
            layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)
            for bstart, bend in layout.bucket_indices:
                assert (bend - bstart) % dp_size == 0

    def test_bucket_global_alignment(self):
        """Bucket end must be a multiple of lcm(dp_size, 128)."""
        dp_size = 4
        params = [_make_param((333,)) for _ in range(8)]
        cfg = _make_ddp_config()
        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)
        divisor = math.lcm(dp_size, 128)
        for _, bend in layout.bucket_indices:
            assert bend % divisor == 0, f"bucket end {bend} not aligned to {divisor}"

    # -- backprop order --

    def test_backprop_order_in_shards(self):
        """Within each shard, params should appear in backprop (reverse model) order."""
        dp_size = 2
        params = [_make_param((128,)) for _ in range(6)]
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        # Group params by shard in backprop (reverse model) order.
        shard_params: dict[int, list] = {i: [] for i in range(dp_size)}
        for p in reversed(params):
            s = _get_shard_for_param(layout, p, dp_size)
            start, _, _ = layout.param_index_map[p]
            shard_params[s].append((start, p))

        # Within each shard, buffer positions should be increasing in backprop order.
        for s, items in shard_params.items():
            positions = [pos for pos, _ in items]
            assert positions == sorted(positions), f"shard {s} not in order"

    # -- dp_size=1 --

    def test_dp_size_1(self):
        """With dp_size=1, every param goes to shard 0 (trivially no splitting)."""
        params = [_make_param((100,)) for _ in range(5)]
        cfg = _make_ddp_config()
        layout = _LWO._compute_per_buffer_param_layout(params, None, 1, cfg)

        for p in params:
            _assert_param_within_shard(layout, p, 1)

    # -- single param --

    def test_single_param(self):
        """A single param should produce one bucket."""
        dp_size = 4
        params = [_make_param((512,))]
        cfg = _make_ddp_config()
        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        assert len(layout.bucket_indices) == 1
        _assert_param_within_shard(layout, params[0], dp_size)

    # -- all params assigned --

    def test_all_params_in_layout(self):
        """Every input param appears in param_index_map exactly once."""
        dp_size = 4
        params = [_make_param((s,)) for s in [64, 128, 256, 64, 128, 256, 64]]
        cfg = _make_ddp_config()
        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        assert set(id(p) for p in layout.param_index_map.keys()) == set(id(p) for p in params)


# ---------------------------------------------------------------------------
# Tests for compute_full_param_layout
# ---------------------------------------------------------------------------


class TestLayerwiseFullParamLayout:

    def test_basic_full_layout(self):
        """End-to-end: params grouped by dtype, then laid out with shard alignment."""
        dp_size = 2
        params = [_make_param((256,)) for _ in range(4)]
        cfg = _make_ddp_config()
        layout = _LWO.compute_full_param_layout(params, None, dp_size, cfg)
        assert len(layout.layouts) == 1
        key = list(layout.layouts.keys())[0]
        assert key == BufferKey(torch.bfloat16, torch.float, False, True)

    def test_expert_parallel_separate_buffer(self):
        """Expert-parallel params should be in a separate buffer group."""
        dp_size = 2
        dense = _make_param((256,))
        expert = _make_param((256,), allreduce=False)
        cfg = _make_ddp_config()
        layout = _LWO.compute_full_param_layout([dense, expert], None, dp_size, cfg)
        assert len(layout.layouts) == 2
