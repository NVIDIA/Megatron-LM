# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import math
import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.multimodal.context_parallel import (
    GatherFromContextParallelRanks,
    _compute_tubelet_aware_split_points,
    _split_num_frames,
    gather_from_context_parallel_ranks,
    gather_from_context_parallel_ranks_dynamic_res,
    split_to_context_parallel_ranks,
    split_to_context_parallel_ranks_dynamic_res,
)
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.test_utilities import Utils


def _assert_split_point_invariants(split_points, num_frames, temporal_patch_size, cp_size):
    """Check the contract of :func:`_compute_tubelet_aware_split_points`."""
    total_frames = sum(num_frames)

    assert len(split_points) == cp_size + 1
    assert split_points[0] == 0
    assert split_points[-1] == total_frames
    assert all(split_points[i] <= split_points[i + 1] for i in range(cp_size))

    if temporal_patch_size <= 1:
        return

    # Valid tubelet boundaries are: 0, total_frames, media_start + k*T (0 < k*T <= nf),
    # and media_end for each media.
    valid_boundaries = {0, total_frames}
    media_start = 0
    for nf in num_frames:
        media_end = media_start + nf
        valid_boundaries.add(media_start)
        valid_boundaries.add(media_end)
        for k in range(1, math.ceil(nf / temporal_patch_size)):
            valid_boundaries.add(media_start + k * temporal_patch_size)
        media_start = media_end

    for sp in split_points:
        assert (
            sp in valid_boundaries
        ), f"split point {sp} is not tubelet-aligned; valid={sorted(valid_boundaries)}"


class TestComputeTubeletAwareSplitPoints:
    """Unit tests for the pure tubelet-aware split-point helper."""

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "num_frames,temporal_patch_size,cp_size",
        [
            # Single video, tubelets align evenly with cp ranks.
            ([16], 4, 4),
            # Single video, tubelets > cp_size.
            ([24], 4, 3),
            # Two equal videos, split lands on a media boundary.
            ([8, 8], 8, 2),
            # Mixed-length videos across many cp ranks.
            ([8, 4, 12], 4, 4),
            # Single short video (1 tubelet) with cp_size=2 (triggers the
            # num_tubelets<=1 branch that snaps to media_start or media_end).
            ([4], 4, 2),
            # Temporal patch size 1 degenerates to frame-granular splits.
            ([10], 1, 5),
        ],
    )
    def test_invariants(self, num_frames, temporal_patch_size, cp_size):
        total = sum(num_frames)
        split_points = _compute_tubelet_aware_split_points(
            num_frames, temporal_patch_size, cp_size, total
        )
        _assert_split_point_invariants(split_points, num_frames, temporal_patch_size, cp_size)

    @pytest.mark.internal
    def test_single_video_even_split(self):
        # 16 frames, T=4 → 4 tubelets over 4 ranks → one tubelet per rank.
        split_points = _compute_tubelet_aware_split_points([16], 4, 4, 16)
        assert split_points == [0, 4, 8, 12, 16]

    @pytest.mark.internal
    def test_tubelet_size_1_matches_contiguous_split(self):
        # With T=1 every frame is its own tubelet; splits should land on
        # near-equal chunks and cover [0, total].
        split_points = _compute_tubelet_aware_split_points([10], 1, 5, 10)
        assert split_points == [0, 2, 4, 6, 8, 10]

    @pytest.mark.internal
    def test_single_tubelet_media_snaps_to_nearest_boundary(self):
        # num_tubelets == 1 ⇒ the split snaps to whichever of media_start /
        # media_end is closer to the target. [4, 2] frames, T=4, cp=2: the
        # target lands in the first media (1 tubelet, [0,4]); target=3 is
        # closer to media_end than media_start so we snap to 4.
        split_points = _compute_tubelet_aware_split_points([4, 2], 4, 2, 6)
        assert split_points == [0, 4, 6]

    @pytest.mark.internal
    def test_monotonicity_is_enforced(self):
        # A pathological case where the raw target-based split would go
        # backwards; the implementation must clamp to the previous split.
        # 3 frames total, T=2, cp=3, num_tubelets=2 on the only media.
        # Every interior rank must still produce a monotone sequence.
        split_points = _compute_tubelet_aware_split_points([3], 2, 3, 3)
        assert split_points[0] == 0
        assert split_points[-1] == 3
        assert all(split_points[i] <= split_points[i + 1] for i in range(3))


class TestSplitNumFrames:
    """Unit tests for the ``[lb, ub)`` per-media clipper."""

    @pytest.mark.internal
    def test_full_range_returns_all(self):
        assert _split_num_frames([8, 4, 12], 0, 24) == [8, 4, 12]

    @pytest.mark.internal
    def test_first_media_only(self):
        assert _split_num_frames([8, 4, 12], 0, 8) == [8]

    @pytest.mark.internal
    def test_spans_boundary_partial_tail(self):
        # [8,16) covers all of the second media (4 frames) plus the first 4
        # of the third (which starts at 12).
        assert _split_num_frames([8, 4, 12], 8, 16) == [4, 4]

    @pytest.mark.internal
    def test_partial_all_three(self):
        # [5,15) clips the first (5..8), all of the second (8..12), and the
        # first 3 frames of the third (12..15).
        assert _split_num_frames([8, 4, 12], 5, 15) == [3, 4, 3]

    @pytest.mark.internal
    def test_empty_range_drops_all(self):
        assert _split_num_frames([8, 4, 12], 10, 10) == []

    @pytest.mark.internal
    def test_range_past_end_drops_all(self):
        assert _split_num_frames([8, 4, 12], 100, 200) == []

    @pytest.mark.internal
    def test_empty_num_frames(self):
        assert _split_num_frames([], 0, 10) == []


class TestSplitToContextParallelRanks:
    """Single-rank tests for ``split_to_context_parallel_ranks``.

    With ``cp_size=1`` the helper is a no-op pass-through that just returns
    the original tensor and ``global_pad=0``. We use a single-rank
    ``Utils.initialize_model_parallel`` setup so we can exercise the public
    API without spinning up multi-rank CP.
    """

    def setup_method(self, method):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_cp_size_1_is_passthrough(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        global_t = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3)

        local_t, global_pad = split_to_context_parallel_ranks(global_t)

        assert torch.equal(local_t, global_t)
        assert global_pad == 0


@pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) < 2,
    reason="Dynamic-res CP split/gather require WORLD_SIZE >= 2",
)
class TestDynamicResCPDistributed:
    """Tests for the distributed split / gather helpers.

    These exercise the all_to_all code path, so they require ``cp_size >= 2``.
    """

    def setup_method(self, method):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_gather_dynamic_res_roundtrip(self):
        """Each rank contributes a rank-shaped slice; gather concatenates them in rank order."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        from megatron.core.parallel_state import (
            get_context_parallel_rank,
            get_context_parallel_world_size,
        )

        cp_rank = get_context_parallel_rank()
        cp_size = get_context_parallel_world_size()

        # Rank 0 contributes 1 tubelet, rank 1 contributes 2 — variable-size gather.
        local_num_tubelets = cp_rank + 1
        patches, hidden = 4, 8
        local_t = torch.full(
            (local_num_tubelets, patches, hidden),
            float(cp_rank),
            dtype=torch.float32,
            device="cuda",
        )

        gathered = gather_from_context_parallel_ranks_dynamic_res(local_t)

        total_tubelets = sum(r + 1 for r in range(cp_size))
        assert gathered.shape == (total_tubelets, patches, hidden)

        # Rank-0 rows come first (value 0.0), then rank-1 rows (value 1.0), etc.
        offset = 0
        for r in range(cp_size):
            n = r + 1
            assert torch.all(gathered[offset : offset + n] == float(r))
            offset += n

    @pytest.mark.internal
    def test_gather_dynamic_res_drops_padded_ranks(self):
        """``num_padded_imgs`` drops the trailing outputs before concatenation."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        from megatron.core.parallel_state import get_context_parallel_rank

        cp_rank = get_context_parallel_rank()
        local_t = torch.full((1, 2, 4), float(cp_rank), dtype=torch.float32, device="cuda")

        gathered = gather_from_context_parallel_ranks_dynamic_res(local_t, num_padded_imgs=1)

        # With cp_size=2 and one padded rank, only rank 0's tensor survives.
        assert gathered.shape == (1, 2, 4)
        assert torch.all(gathered == 0.0)

    @pytest.mark.internal
    def test_split_dynamic_res_distributes_images_across_ranks(self):
        """Split without temporal compression and without dummy padding.

        With cp_size=2 and 2 equal-sized images, each rank should receive
        exactly one image worth of tokens and an image-size tensor of length 1.
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        from megatron.core.parallel_state import get_context_parallel_rank

        cp_rank = get_context_parallel_rank()
        patch_dim = 16
        # Each "image" is 16x16 patches, flattened into a length-256 sequence.
        per_img_seq = patch_dim * patch_dim
        hidden = 3 * patch_dim * patch_dim  # so dummy_seqlen works out to 1 if triggered
        num_imgs = 2

        # Token-valued so we can trace which image each rank receives.
        chunks = [
            torch.full((per_img_seq, hidden), float(i), dtype=torch.float32, device="cuda")
            for i in range(num_imgs)
        ]
        global_t = torch.cat(chunks, dim=0).unsqueeze(0)  # [1, num_imgs*per_img_seq, hidden]

        global_imgs_sizes = torch.tensor(
            [[patch_dim, patch_dim]] * num_imgs, dtype=torch.int32, device="cuda"
        )
        cu_seqlens = torch.tensor(
            [i * per_img_seq for i in range(num_imgs + 1)], dtype=torch.int32, device="cuda"
        )
        global_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=per_img_seq,
            max_seqlen_kv=per_img_seq,
        )

        (
            local_t,
            local_imgs_sizes,
            local_packed_seq_params,
            has_padding,
            num_padded_ranks,
            local_num_frames,
        ) = split_to_context_parallel_ranks_dynamic_res(
            global_t,
            global_imgs_sizes,
            global_packed_seq_params,
            fp8_enabled=False,
            patch_dim=patch_dim,
            temporal_patch_size=1,
        )

        # Non-temporal, no FP8, and num_imgs==cp_size ⇒ no padding of any kind.
        assert has_padding is False
        assert num_padded_ranks == 0
        assert local_num_frames is None

        # Each rank owns exactly one image.
        assert local_t.shape == (1, per_img_seq, hidden)
        assert local_imgs_sizes.shape == (1, 2)
        # The value in local_t identifies which image this rank owns.
        assert torch.all(local_t == float(cp_rank))

    @pytest.mark.internal
    def test_split_dynamic_res_temporal_aware_tubelets(self):
        """``temporal_patch_size > 1`` triggers the tubelet-aware split.

        Two videos of 4 and 4 frames each, T=2 ⇒ 4 tubelets total over 2 CP
        ranks ⇒ each rank owns 2 tubelets. Verify ``local_num_frames`` reflects
        the per-rank frame counts (not tubelet counts).
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        from megatron.core.parallel_state import get_context_parallel_rank

        cp_rank = get_context_parallel_rank()
        patch_dim = 16
        per_img_seq = patch_dim * patch_dim
        hidden = 3 * patch_dim * patch_dim
        num_videos = 2
        frames_per_video = 4
        temporal_patch_size = 2
        # After temporal grouping each video has frames_per_video//T tubelets.
        # split_to_context_parallel_ranks_dynamic_res receives post-grouping
        # tubelet data: one entry per tubelet in imgs_sizes / cu_seqlens.
        total_tubelets = num_videos * (frames_per_video // temporal_patch_size)  # = 4

        chunks = [
            torch.full((per_img_seq, hidden), float(i), dtype=torch.float32, device="cuda")
            for i in range(total_tubelets)
        ]
        global_t = torch.cat(chunks, dim=0).unsqueeze(0)
        global_imgs_sizes = torch.tensor(
            [[patch_dim, patch_dim]] * total_tubelets, dtype=torch.int32, device="cuda"
        )
        cu_seqlens = torch.tensor(
            [i * per_img_seq for i in range(total_tubelets + 1)], dtype=torch.int32, device="cuda"
        )
        global_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=per_img_seq,
            max_seqlen_kv=per_img_seq,
        )
        num_frames = torch.tensor([frames_per_video] * num_videos, dtype=torch.int32, device="cuda")

        (local_t, local_imgs_sizes, _packed, has_padding, num_padded_ranks, local_num_frames) = (
            split_to_context_parallel_ranks_dynamic_res(
                global_t,
                global_imgs_sizes,
                global_packed_seq_params,
                patch_dim=patch_dim,
                num_frames=num_frames,
                temporal_patch_size=temporal_patch_size,
            )
        )

        assert has_padding is False
        assert num_padded_ranks == 0
        # Per-rank frame counts: 4 tubelets ÷ 2 ranks = 2 tubelets/rank ⇒ 4 frames/rank.
        assert local_num_frames is not None
        assert int(local_num_frames.sum().item()) == 4
        # Each rank owns 2 tubelets worth of patches (one tubelet = per_img_seq patches).
        assert local_t.shape == (1, 2 * per_img_seq, hidden)
        assert local_imgs_sizes.shape == (2, 2)

    @pytest.mark.internal
    def test_split_dynamic_res_pads_when_too_few_images(self):
        """``num_padded_imgs`` > 0 path: 1 image with cp_size=2 must pad with a dummy."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        patch_dim = 16
        per_img_seq = patch_dim * patch_dim
        hidden = 3 * patch_dim * patch_dim
        global_t = torch.zeros((1, per_img_seq, hidden), dtype=torch.float32, device="cuda")
        global_imgs_sizes = torch.tensor([[patch_dim, patch_dim]], dtype=torch.int32, device="cuda")
        cu_seqlens = torch.tensor([0, per_img_seq], dtype=torch.int32, device="cuda")
        global_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=per_img_seq,
            max_seqlen_kv=per_img_seq,
        )

        _local_t, _local_sizes, _packed, _has_pad, num_padded_ranks, _ = (
            split_to_context_parallel_ranks_dynamic_res(
                global_t,
                global_imgs_sizes,
                global_packed_seq_params,
                patch_dim=patch_dim,
                temporal_patch_size=1,
            )
        )
        # cp_size=2, num_imgs=1 ⇒ one dummy added ⇒ one rank is padded.
        assert num_padded_ranks == 1

    @pytest.mark.internal
    def test_split_dynamic_res_asserts_on_wrong_hidden_dim(self):
        """B10 regression: hidden dim must equal 3*patch_dim*patch_dim."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        patch_dim = 16
        per_img_seq = patch_dim * patch_dim
        wrong_hidden = 3 * patch_dim * patch_dim + 1  # off by one
        global_t = torch.zeros(
            (1, 2 * per_img_seq, wrong_hidden), dtype=torch.float32, device="cuda"
        )
        global_imgs_sizes = torch.tensor(
            [[patch_dim, patch_dim]] * 2, dtype=torch.int32, device="cuda"
        )
        cu_seqlens = torch.tensor(
            [0, per_img_seq, 2 * per_img_seq], dtype=torch.int32, device="cuda"
        )
        global_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=per_img_seq,
            max_seqlen_kv=per_img_seq,
        )

        with pytest.raises(AssertionError, match="3\\*patch_dim\\*patch_dim"):
            split_to_context_parallel_ranks_dynamic_res(
                global_t,
                global_imgs_sizes,
                global_packed_seq_params,
                patch_dim=patch_dim,
                temporal_patch_size=1,
            )

    @pytest.mark.internal
    def test_gather_from_context_parallel_ranks_drops_global_pad(self):
        """``gather_from_context_parallel_ranks`` removes the trailing pad columns."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        from megatron.core.parallel_state import get_context_parallel_rank

        cp_rank = get_context_parallel_rank()
        # Each rank contributes a [batch=1, seq=3, h=4] tensor of value cp_rank.
        local_t = torch.full((1, 3, 4), float(cp_rank), dtype=torch.float32, device="cuda")

        # Simulate global_pad=2 ⇒ trailing 2 columns of the gathered tensor get dropped.
        out = gather_from_context_parallel_ranks(local_t, global_pad=2)

        # cp_size * seq - global_pad = 2*3 - 2 = 4 effective columns.
        assert out.shape == (1, 4, 4)

    @pytest.mark.internal
    def test_gather_from_context_parallel_ranks_autograd_backward(self):
        """``GatherFromContextParallelRanks.backward`` must reduce-scatter gradients.

        We feed a ``[1, seq, h]`` tensor on each rank, all-gather along seq,
        sum the result, and backprop. The autograd backward path should run the
        reduce-scatter and yield a finite per-rank gradient.
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        from megatron.core.parallel_state import get_context_parallel_rank

        cp_rank = get_context_parallel_rank()
        local_t = torch.full(
            (1, 4, 8), float(cp_rank + 1), dtype=torch.float32, device="cuda", requires_grad=True
        )

        gathered = GatherFromContextParallelRanks.apply(local_t)
        loss = gathered.sum()
        loss.backward()

        assert local_t.grad is not None
        assert local_t.grad.shape == local_t.shape
        # Reduce-scatter of an all-ones grad gives a positive gradient on every rank.
        assert torch.all(local_t.grad > 0)
