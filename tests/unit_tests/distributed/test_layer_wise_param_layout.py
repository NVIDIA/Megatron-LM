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
    param = torch.nn.Parameter(torch.randn(shape, dtype=dtype))
    for attr_name, attr_value in attrs.items():
        setattr(param, attr_name, attr_value)
    return param


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
    """Return which shard a param lands in."""
    param_start_index, param_end_index, bucket_id = layout.param_index_map[param]
    bucket_start_index, bucket_end_index = layout.bucket_indices[bucket_id]
    shard_size = (bucket_end_index - bucket_start_index) // dp_size
    shard_id = (param_start_index - bucket_start_index) // shard_size
    return shard_id


def _assert_param_within_shard(layout, param, dp_size):
    """Assert that a param lies entirely within one shard."""
    param_start_index, param_end_index, bucket_id = layout.param_index_map[param]
    bucket_start_index, bucket_end_index = layout.bucket_indices[bucket_id]
    shard_size = (bucket_end_index - bucket_start_index) // dp_size
    shard_id = (param_start_index - bucket_start_index) // shard_size
    shard_start_index = bucket_start_index + shard_id * shard_size
    shard_end_index = shard_start_index + shard_size
    assert (
        shard_start_index <= param_start_index
    ), f"param start {param_start_index} before shard start {shard_start_index}"
    assert (
        param_end_index <= shard_end_index
    ), f"param end {param_end_index} past shard end {shard_end_index}"


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

        for param in params:
            _assert_param_within_shard(layout, param, dp_size)

        # With 8 params and dp_size=4, 2 rounds of size-matching.
        # Each round fills all 4 shards.  No padding needed.
        shard_counts = Counter(_get_shard_for_param(layout, param, dp_size) for param in params)
        assert set(shard_counts.values()) == {2}

    def test_uniform_params_remainder_gets_padding(self):
        """5 same-size params with dp_size=4 → 1 round fills 4, 1 round fills 1 + 3 padding."""
        dp_size = 4
        params = [_make_param((256,)) for _ in range(5)]
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        for param in params:
            _assert_param_within_shard(layout, param, dp_size)

        # All 5 params assigned; shard 0 gets 2, shards 1-3 get 1 each.
        shard_counts = Counter(_get_shard_for_param(layout, param, dp_size) for param in params)
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

        for param in params:
            _assert_param_within_shard(layout, param, dp_size)

    def test_size_matching_prefers_same_size(self):
        """When shard 0 gets a 256-elem param, other shards should also get 256-elem params."""
        dp_size = 4
        big_params = [_make_param((256,)) for _ in range(4)]
        small_params = [_make_param((64,)) for _ in range(4)]
        # Interleave: big_params[0], small_params[0], big_params[1], small_params[1], ...
        params = []
        for big_param, small_param in zip(big_params, small_params):
            params.extend([big_param, small_param])
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        for param in params:
            _assert_param_within_shard(layout, param, dp_size)

        # All 4 big params should be in the same round (one per shard).
        big_param_bucket_ids = set()
        for param in big_params:
            _, _, bucket_id = layout.param_index_map[param]
            big_param_bucket_ids.add(bucket_id)
        # They should share a bucket (matched in one round).
        assert len(big_param_bucket_ids) == 1

    # -- fallback packing for unique-large seeds --

    def test_unique_large_seed_packs_smaller_params(self):
        """A unique-large seed's empty shard slots should absorb smaller params.

        Pool order is the *reverse* of input/forward order, so the
        ``unique_large_param`` is placed last in the input list to make it the
        first seed (top of pool). Without the packing fallback, the unique
        seed would emit a bucket with ``dp_size - 1`` shards of pure padding
        and the trailing smaller params would form their own bucket. With the
        fallback, the smaller params land in the large param's bucket,
        eliminating the second bucket entirely.
        """
        dp_size = 4
        unique_large_param = _make_param((1024,))
        filler_params = [_make_param((128,)) for _ in range(3)]
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(
            filler_params + [unique_large_param], None, dp_size, cfg
        )

        # Invariant still holds: each param lies entirely within one shard.
        for param in [unique_large_param] + filler_params:
            _assert_param_within_shard(layout, param, dp_size)

        # All four params share a single bucket (no second bucket for the
        # filler params).
        assert len(layout.bucket_indices) == 1
        _, _, large_bucket_id = layout.param_index_map[unique_large_param]
        for filler_param in filler_params:
            _, _, filler_bucket_id = layout.param_index_map[filler_param]
            assert filler_bucket_id == large_bucket_id

        # Filler params land in shard slots other than the unique-large's.
        large_shard = _get_shard_for_param(layout, unique_large_param, dp_size)
        for filler_param in filler_params:
            assert _get_shard_for_param(layout, filler_param, dp_size) != large_shard

    # -- shared_embedding isolation --

    def test_shared_embedding_isolated(self):
        """shared_embedding params go in their own bucket and fit within shard 0."""
        dp_size = 2
        regular_params = [_make_param((128,)) for _ in range(4)]
        embedding_param = _make_param((128,), shared_embedding=True)
        params = [embedding_param] + regular_params
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        # The shared embedding must fit entirely within one shard so reduce-scatter
        # delivers the full reduced gradient to its owner rank.
        _assert_param_within_shard(layout, embedding_param, dp_size)

        # And it must be the sole real param in its bucket.
        _, _, embedding_bucket_id = layout.param_index_map[embedding_param]
        for param in regular_params:
            _, _, bucket_id = layout.param_index_map[param]
            assert bucket_id != embedding_bucket_id, "shared_embedding should be in its own bucket"

        # Embedding lives in shard 0; shards 1..dp_size-1 are pure padding.
        assert _get_shard_for_param(layout, embedding_param, dp_size) == 0

    def test_shared_embedding_fits_in_shard_at_high_dp(self):
        """With dp_size > 2, a vocab-sized shared embedding still fits in one shard.

        Regression test for the case where the isolated bucket was sized to ~numel
        instead of dp_size * numel, causing the embedding to silently span multiple
        shards on dp_size > 1.
        """
        for dp_size in [2, 4, 8]:
            embedding_param = _make_param((1024,), shared_embedding=True)
            cfg = _make_ddp_config()
            layout = _LWO._compute_per_buffer_param_layout([embedding_param], None, dp_size, cfg)
            _assert_param_within_shard(layout, embedding_param, dp_size)
            assert _get_shard_for_param(layout, embedding_param, dp_size) == 0

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
            for bucket_start_index, bucket_end_index in layout.bucket_indices:
                assert (bucket_end_index - bucket_start_index) % dp_size == 0

    def test_bucket_global_alignment(self):
        """Bucket end must be a multiple of lcm(dp_size, 128)."""
        dp_size = 4
        params = [_make_param((333,)) for _ in range(8)]
        cfg = _make_ddp_config()
        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)
        divisor = math.lcm(dp_size, 128)
        for _, bucket_end_index in layout.bucket_indices:
            assert (
                bucket_end_index % divisor == 0
            ), f"bucket end {bucket_end_index} not aligned to {divisor}"

    # -- backprop order --

    def test_backprop_order_in_shards(self):
        """Within each shard, params should appear in backprop (reverse model) order."""
        dp_size = 2
        params = [_make_param((128,)) for _ in range(6)]
        cfg = _make_ddp_config()

        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        # Group params by shard in backprop (reverse model) order.
        shard_params: dict[int, list] = {i: [] for i in range(dp_size)}
        for param in reversed(params):
            shard_id = _get_shard_for_param(layout, param, dp_size)
            param_start_index, _, _ = layout.param_index_map[param]
            shard_params[shard_id].append((param_start_index, param))

        # Within each shard, buffer positions should be increasing in backprop order.
        for shard_id, items in shard_params.items():
            param_start_indices = [param_start_index for param_start_index, _ in items]
            assert param_start_indices == sorted(
                param_start_indices
            ), f"shard {shard_id} not in order"

    # -- dp_size=1 --

    def test_dp_size_1(self):
        """With dp_size=1, every param goes to shard 0 (trivially no splitting)."""
        params = [_make_param((100,)) for _ in range(5)]
        cfg = _make_ddp_config()
        layout = _LWO._compute_per_buffer_param_layout(params, None, 1, cfg)

        for param in params:
            _assert_param_within_shard(layout, param, 1)

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
        params = [_make_param((numel,)) for numel in [64, 128, 256, 64, 128, 256, 64]]
        cfg = _make_ddp_config()
        layout = _LWO._compute_per_buffer_param_layout(params, None, dp_size, cfg)

        assert set(id(param) for param in layout.param_index_map.keys()) == set(
            id(param) for param in params
        )


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
        assert key == BufferKey(torch.bfloat16, torch.float, False)

    def test_expert_parallel_separate_buffer(self):
        """Expert-parallel params should be in a separate buffer group."""
        dp_size = 2
        dense = _make_param((256,))
        expert = _make_param((256,), allreduce=False)
        cfg = _make_ddp_config()
        layout = _LWO.compute_full_param_layout([dense, expert], None, dp_size, cfg)
        assert len(layout.layouts) == 2
