# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for chunked vision-encoder execution (partition + lockstep).

Real-encoder GPU equality (outputs / param grads / optimizer step) and the
2-rank MFSDP lockstep run live in ``test_vision_chunking_gpu.py``.
"""

import pytest
import torch

from examples.multimodal_dev.models.base import MultimodalModel, _vision_chunk_slices


def _grids(*sizes_hw):
    return torch.tensor([[1, h, w] for h, w in sizes_hw], dtype=torch.long)


class TestChunkSlices:
    def test_greedy_partition_at_image_boundaries(self):
        grids = _grids((2, 2), (2, 2), (2, 2))  # 4 raw patches each
        assert _vision_chunk_slices(grids, 8) == [(0, 2, 0, 8), (2, 3, 8, 12)]

    def test_exact_fit_keeps_one_chunk(self):
        grids = _grids((2, 2), (2, 2))
        assert _vision_chunk_slices(grids, 8) == [(0, 2, 0, 8)]

    def test_oversized_image_gets_its_own_chunk(self):
        grids = _grids((2, 2), (10, 2), (2, 2))  # 4, 20, 4
        assert _vision_chunk_slices(grids, 8) == [(0, 1, 0, 4), (1, 2, 4, 24), (2, 3, 24, 28)]

    def test_empty_payload(self):
        assert _vision_chunk_slices(torch.empty((0, 3), dtype=torch.long), 8) == []

    def test_rows_partition_the_payload_exactly(self):
        grids = _grids((2, 4), (4, 4), (2, 2), (6, 4), (2, 2))
        slices = _vision_chunk_slices(grids, 16)
        total = int((grids[:, 0] * grids[:, 1] * grids[:, 2]).sum().item())
        assert slices[0][2] == 0 and slices[-1][3] == total
        for (_, hi_a, _, row_hi_a), (lo_b, _, row_lo_b, _) in zip(slices, slices[1:]):
            assert hi_a == lo_b and row_hi_a == row_lo_b


class _PerImageFakeVision:
    """Deterministic per-image encoder: output depends only on that image.

    For each image emits one row per raw patch: the patch row scaled by the
    image's grid area. Chunked-vs-whole equality holds iff the caller never
    mixes rows across images.
    """

    spatial_merge_size = 2

    def __init__(self):
        self.calls = []
        self.pixel_tensors = []

    def __call__(self, pixel_values, image_grid_thw):
        self.calls.append((tuple(pixel_values.shape), image_grid_thw.tolist()))
        self.pixel_tensors.append(pixel_values)
        outputs = []
        row = 0
        for t, h, w in image_grid_thw.tolist():
            n = t * h * w
            outputs.append(pixel_values[row : row + n] * float(n))
            row += n
        return torch.cat(outputs) if outputs else pixel_values[:0]


def _make_model(chunk_patches, *, training=True, group=None, pool=None):
    model = MultimodalModel.__new__(MultimodalModel)
    model.vision_model = _PerImageFakeVision()
    model.vision_encoder_chunk_patches = chunk_patches
    model.vision_lockstep_group = group
    model.training = training
    if pool is not None:
        model.vision_noise_pool = pool
    return model


class TestChunkedForwardEquality:
    def test_chunked_equals_unchunked(self):
        grids = _grids((2, 2), (4, 2), (2, 2), (2, 4))  # 4, 8, 4, 8 raw patches
        pixels = torch.randn(24, 6)

        whole, anchor_whole = _make_model(0)._vision_forward(pixels, grids)
        chunked, anchor_chunked = _make_model(8)._vision_forward(pixels, grids)

        assert anchor_whole is None and anchor_chunked is None
        assert torch.equal(whole, chunked)

    def test_chunk_calls_respect_image_boundaries(self):
        grids = _grids((2, 2), (2, 2), (2, 2))
        model = _make_model(8)
        model._vision_forward(torch.randn(12, 6), grids)
        assert model.vision_model.calls == [((8, 6), [[1, 2, 2], [1, 2, 2]]), ((4, 6), [[1, 2, 2]])]


class _MaxThreeDist:
    """Fake distributed backend: group max chunk count is always 3."""

    @staticmethod
    def install(monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        def fake_all_reduce(tensor, op=None, group=None):
            tensor.fill_(max(int(tensor.item()), 3))

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)


class TestLockstep:
    def test_short_rank_pads_with_dummy_passes(self, monkeypatch):
        _MaxThreeDist.install(monkeypatch)
        grids = _grids((2, 2))  # one real chunk
        model = _make_model(8, group=object())
        embeddings, anchor = model._vision_forward(torch.randn(4, 6), grids)

        assert embeddings is not None
        assert anchor is not None and float(anchor) == 0.0
        # 1 real chunk + 2 dummy 4-patch passes to reach the group max of 3.
        shapes = [shape for shape, _ in model.vision_model.calls]
        assert shapes == [(4, 6), (4, 6), (4, 6)]
        assert model.vision_model.calls[1][1] == [[1, 2, 2]]

    def test_training_image_free_rank_runs_one_dummy_without_group(self):
        model = _make_model(8, training=True, group=None)
        embeddings, anchor = model._vision_forward(torch.empty(0, 6), _grids())
        assert embeddings is None
        assert anchor is not None and float(anchor) == 0.0
        assert len(model.vision_model.calls) == 1

    def test_eval_group_wide_image_free_skips_the_tower(self, monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        def fake_all_reduce(tensor, op=None, group=None):
            pass  # group max stays 0

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        model = _make_model(8, training=False, group=object())
        embeddings, anchor = model._vision_forward(torch.empty(0, 6), _grids())
        assert embeddings is None and anchor is None
        assert model.vision_model.calls == []

    def test_eval_lockstep_still_pads_when_group_has_images(self, monkeypatch):
        _MaxThreeDist.install(monkeypatch)
        model = _make_model(8, training=False, group=object())
        embeddings, anchor = model._vision_forward(torch.empty(0, 6), _grids())
        assert embeddings is None
        assert anchor is not None
        assert len(model.vision_model.calls) == 3

    def test_legacy_path_unchanged_when_disabled(self):
        model = _make_model(0, training=True)
        embeddings, anchor = model._vision_forward(torch.empty(0, 6), _grids())
        assert embeddings is None and anchor is not None
        assert len(model.vision_model.calls) == 1


class TestStreamingNoisePool:
    def test_every_chunk_input_aliases_the_pool_storage(self):
        pool = torch.randn(8, 6)
        grids = _grids((2, 2), (2, 2), (2, 2))  # 3 x 4 patches, chunk 8 -> 2 chunks
        model = _make_model(8, pool=pool)
        embeddings, anchor = model._vision_forward(None, grids)

        assert embeddings is not None and anchor is None
        pool_ptr = pool.untyped_storage().data_ptr()
        for shape, _ in model.vision_model.calls:
            assert shape[1] == 6
        for pixels in model.vision_model.pixel_tensors:
            assert pixels.untyped_storage().data_ptr() == pool_ptr

    def test_streaming_without_pool_or_chunking_fails(self):
        grids = _grids((2, 2))
        with pytest.raises(RuntimeError, match="vision_noise_pool"):
            _make_model(8, pool=None)._vision_forward(None, grids)
        with pytest.raises(RuntimeError, match="chunk-patches"):
            _make_model(0, pool=torch.randn(8, 6))._vision_forward(None, grids)

    def test_chunk_larger_than_pool_fails_loudly(self):
        grids = _grids((10, 2))  # 20-patch image > 8-row pool
        model = _make_model(8, pool=torch.randn(8, 6))
        with pytest.raises(RuntimeError, match="noise\npool|noise pool"):
            model._vision_forward(None, grids)

    def test_streaming_image_free_training_runs_one_dummy(self):
        model = _make_model(8, pool=torch.randn(8, 6), training=True)
        embeddings, anchor = model._vision_forward(None, _grids())
        assert embeddings is None
        assert anchor is not None and float(anchor) == 0.0
        assert len(model.vision_model.calls) == 1
