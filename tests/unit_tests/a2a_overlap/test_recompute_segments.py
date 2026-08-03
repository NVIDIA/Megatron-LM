# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Segmentation rules for layer-level full activation recompute.

These tests exercise ``TransformerModelChunkSchedulePlan._build_recompute_segments``
in isolation (no CUDA, no model build) to pin down that the segment boundaries match
the non-overlap recompute semantics:

* decoder block -> ``megatron.core.recompute.checkpointed_forward``
* MTP -> ``MultiTokenPredictionLayer._checkpointed_forward``
"""

from types import SimpleNamespace

import pytest

from megatron.core.models.common.model_chunk_schedule_plan import (
    ModelChunkState,
    TransformerModelChunkSchedulePlan,
)


class _StubLayer:
    """Minimal stand-in for TransformerLayerSchedulePlan."""

    def __init__(self, index):
        self.index = index
        self.recompute_segment = None
        self.is_segment_head = False
        self.is_segment_tail = False
        self.forward_no_grad = False

    def set_forward_no_grad(self, no_grad):
        self.forward_no_grad = no_grad


def _build(num_decoder_layers, num_mtp_layers, method, num_layers, granularity="full"):
    plan = object.__new__(TransformerModelChunkSchedulePlan)
    plan._model_chunk_state = ModelChunkState()
    plan._transformer_layers = [
        _StubLayer(i) for i in range(num_decoder_layers + num_mtp_layers)
    ]
    plan._num_decoder_layers = num_decoder_layers
    plan._recompute_segments = []
    plan.recompute_full = granularity == "full"
    config = SimpleNamespace(recompute_method=method, recompute_num_layers=num_layers)
    plan._build_recompute_segments(config)
    return plan


def _segment_indices(plan):
    return [[layer.index for layer in seg.layers] for seg in plan._recompute_segments]


def test_disabled_when_granularity_is_not_full():
    plan = _build(4, 0, "uniform", 1, granularity="selective")
    assert plan._recompute_segments == []
    assert all(layer.recompute_segment is None for layer in plan._transformer_layers)
    assert all(not layer.forward_no_grad for layer in plan._transformer_layers)


@pytest.mark.parametrize(
    "num_layers,expected",
    [
        (1, [[0], [1], [2], [3], [4]]),
        (2, [[0, 1], [2, 3], [4]]),
        (3, [[0, 1, 2], [3, 4]]),
        # A group larger than the layer count collapses to a single segment, matching
        # the chunk_end = min(...) clamp in checkpointed_forward's uniform branch.
        (8, [[0, 1, 2, 3, 4]]),
    ],
)
def test_uniform_groups_decoder_layers(num_layers, expected):
    plan = _build(5, 0, "uniform", num_layers)
    assert _segment_indices(plan) == expected
    # Every decoder layer is recomputed under 'uniform'.
    assert all(layer.recompute_segment is not None for layer in plan._transformer_layers)
    assert all(layer.forward_no_grad for layer in plan._transformer_layers)


def test_uniform_puts_each_mtp_layer_in_its_own_segment():
    plan = _build(4, 2, "uniform", 2)
    # Decoder grouped by 2; MTP one depth per segment regardless of the group size.
    assert _segment_indices(plan) == [[0, 1], [2, 3], [4], [5]]


def test_block_recomputes_only_the_first_n_decoder_layers():
    plan = _build(5, 1, "block", 2)
    # One segment per layer for the first two decoder layers; the remaining decoder
    # layers and every MTP layer keep their forward graph (the non-overlap path warns
    # and skips MTP recompute under 'block').
    assert _segment_indices(plan) == [[0], [1]]
    recomputed = [layer.index for layer in plan._transformer_layers if layer.forward_no_grad]
    assert recomputed == [0, 1]


def test_block_wider_than_the_decoder_recomputes_every_decoder_layer():
    plan = _build(3, 1, "block", 9)
    assert _segment_indices(plan) == [[0], [1], [2]]


def test_segment_head_and_tail_flags():
    plan = _build(4, 0, "uniform", 2)
    heads = [layer.index for layer in plan._transformer_layers if layer.is_segment_head]
    tails = [layer.index for layer in plan._transformer_layers if layer.is_segment_tail]
    assert heads == [0, 2]
    assert tails == [1, 3]


def test_empty_decoder_with_mtp_only():
    plan = _build(0, 2, "uniform", 1)
    assert _segment_indices(plan) == [[0], [1]]


@pytest.mark.parametrize("method", [None, "invalid"])
def test_rejects_unsupported_recompute_method(method):
    with pytest.raises(AssertionError, match="recompute_method"):
        _build(2, 0, method, 1)


@pytest.mark.parametrize("num_layers", [None, 0, -1])
def test_rejects_invalid_recompute_num_layers(num_layers):
    with pytest.raises(AssertionError, match="recompute_num_layers"):
        _build(2, 0, "uniform", num_layers)
