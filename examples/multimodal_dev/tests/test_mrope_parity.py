# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Parity tests for ``get_rope_index`` (MRoPE position-ID computation).

Two properties are verified:

1. **BSHD backwards compatibility** — the refactored ``get_rope_index``
   returns bit-identical ``(position_ids, mrope_position_deltas)`` to
   the pre-refactor implementation on padded ``[B, S]`` batches.
2. **THD == BSHD on the valid region** — when the same variable-length
   samples are fed through both layouts (BSHD with right-padding; THD
   packed with ``cu_seqlens_q`` / ``cu_seqlens_q_padded``), positions at
   every valid slot agree.

The pre-refactor function is pinned inline as ``_old_get_rope_index``
so this test stays self-contained. Run with::

    python -m pytest examples/multimodal_dev/tests/test_mrope_parity.py -v

or directly::

    python examples/multimodal_dev/tests/test_mrope_parity.py
"""

import math
import os
import sys
from itertools import accumulate

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../.."),
)
# Insert at position 0 unconditionally — other entries on sys.path
# (e.g. a sibling Megatron-LM checkout) have their own ``examples``
# package that would otherwise shadow ours.
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from megatron.core.packed_seq_params import PackedSeqParams

from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index

# -----------------------------------------------------------------------------
# Token-ID constants (match Qwen3.5-VL, but values are arbitrary for this test)
# -----------------------------------------------------------------------------

IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057
VISION_START_TOKEN_ID = 248053
SPATIAL_MERGE_SIZE = 2


# -----------------------------------------------------------------------------
# Pinned reference implementation (pre-refactor BSHD path)
# -----------------------------------------------------------------------------

def _old_get_rope_index(
    spatial_merge_size,
    image_token_id,
    video_token_id,
    vision_start_token_id,
    input_ids=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
):
    """Pre-refactor BSHD implementation of ``get_rope_index``.

    Copied verbatim (modulo the broken cu_seqlens branch, which this
    parity test does not exercise) so we can diff against the new
    implementation on BSHD inputs.
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0,
        )
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []

    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, sample_input_ids in enumerate(total_input_ids):
            sample_input_ids = sample_input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(
                sample_input_ids == vision_start_token_id,
            ).squeeze(1)
            vision_tokens = sample_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1
                    if llm_pos_ids_list
                    else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1)
                    + st_idx
                )

                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index])
                    + text_len
                    + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1
                    if llm_pos_ids_list
                    else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1)
                    + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[
                ..., i, attention_mask[i] == 1
            ] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i]),
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=total_input_ids.device,
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # Text-only fallback.
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = (
            position_ids.unsqueeze(0)
            .expand(3, -1, -1)
            .to(attention_mask.device)
        )
        max_position_ids = (
            position_ids.max(0, keepdim=False)[0]
            .max(-1, keepdim=True)[0]
        )
        mrope_position_deltas = (
            max_position_ids + 1 - attention_mask.shape[-1]
        )
    else:
        position_ids = (
            torch.arange(
                input_ids.shape[1], device=input_ids.device,
            )
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
    return position_ids, mrope_position_deltas


# -----------------------------------------------------------------------------
# Synthetic-sample builder
# -----------------------------------------------------------------------------

def _build_sample(
    prefix_text_len,
    grids,
    suffix_text_len,
    text_token=100,
):
    """Build one variable-length sample with ``len(grids)`` images.

    Layout per image: ``vision_start_id`` then
    ``llm_grid_t * llm_grid_h * llm_grid_w`` ``image_token_id`` slots
    (where ``llm_grid_* = grid_* // spatial_merge_size`` for h/w).
    Grids use ``t=1``.

    Returns ``(input_ids [L], image_grid_thw [N, 3])``.
    """
    tokens = [text_token] * prefix_text_len
    grid_rows = []
    for t, h, w in grids:
        n_image_tokens = (
            t * (h // SPATIAL_MERGE_SIZE) * (w // SPATIAL_MERGE_SIZE)
        )
        tokens.append(VISION_START_TOKEN_ID)
        tokens.extend([IMAGE_TOKEN_ID] * n_image_tokens)
        grid_rows.append([t, h, w])
    tokens.extend([text_token + 1] * suffix_text_len)
    input_ids = torch.tensor(tokens, dtype=torch.int64)
    image_grid_thw = torch.tensor(grid_rows, dtype=torch.int64)
    return input_ids, image_grid_thw


def _sample_bank():
    """A small bank of samples covering text-only, single-image, multi-image."""
    return [
        _build_sample(
            prefix_text_len=5,
            grids=[(1, 4, 4)],
            suffix_text_len=7,
        ),
        _build_sample(
            prefix_text_len=3,
            grids=[(1, 2, 2), (1, 4, 6)],
            suffix_text_len=4,
        ),
        _build_sample(
            prefix_text_len=10,
            grids=[],
            suffix_text_len=0,
        ),
        _build_sample(
            prefix_text_len=0,
            grids=[(1, 6, 4)],
            suffix_text_len=2,
        ),
    ]


# -----------------------------------------------------------------------------
# Test 1: BSHD backwards compatibility
# -----------------------------------------------------------------------------

def test_bshd_matches_old_reference():
    """New ``get_rope_index`` equals the pinned reference on BSHD inputs."""
    samples = _sample_bank()
    max_len = max(s.numel() for s, _ in samples)

    input_ids_rows = []
    mask_rows = []
    grid_rows = []
    for tokens, grids in samples:
        L = tokens.numel()
        padded = F.pad(tokens, (0, max_len - L), value=0)
        mask = torch.zeros(max_len, dtype=torch.int64)
        mask[:L] = 1
        input_ids_rows.append(padded)
        mask_rows.append(mask)
        if grids.numel() > 0:
            grid_rows.append(grids)

    input_ids = torch.stack(input_ids_rows)              # [B, S]
    attention_mask = torch.stack(mask_rows)              # [B, S]
    image_grid_thw = (
        torch.cat(grid_rows, dim=0) if grid_rows else None
    )

    old_pos, old_delta = _old_get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    new_pos, new_delta = get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        packed_seq_params=None,
    )

    assert torch.equal(old_pos, new_pos), (
        f"BSHD position_ids differ.\nold:\n{old_pos}\nnew:\n{new_pos}"
    )
    assert torch.equal(old_delta, new_delta), (
        f"BSHD mrope_position_deltas differ.\n"
        f"old: {old_delta}\nnew: {new_delta}"
    )


# -----------------------------------------------------------------------------
# Test 2: THD positions match BSHD positions on the valid region
# -----------------------------------------------------------------------------

def _pack_samples(samples, divisible_by=1):
    """Pack ``samples`` into a single ``[1, T]`` tensor the same way
    ``pack_or_pad_batch`` does, and build ``PackedSeqParams``.

    Each per-sample tensor is right-padded to a multiple of
    ``divisible_by`` before concatenation. ``cu_seqlens_q`` tracks
    unpadded lengths; ``cu_seqlens_q_padded`` tracks the packed layout.
    """
    padded_chunks = []
    seqlens = []
    seqlens_padded = []
    grid_rows = []
    for tokens, grids in samples:
        L = tokens.numel()
        target_L = math.ceil(L / divisible_by) * divisible_by
        padded_chunks.append(F.pad(tokens, (0, target_L - L), value=0))
        seqlens.append(L)
        seqlens_padded.append(target_L)
        if grids.numel() > 0:
            grid_rows.append(grids)

    packed = torch.cat(padded_chunks, dim=0).unsqueeze(0)    # [1, T]
    cu_seqlens = torch.tensor(
        list(accumulate(seqlens, initial=0)), dtype=torch.int32,
    )
    cu_seqlens_padded = torch.tensor(
        list(accumulate(seqlens_padded, initial=0)), dtype=torch.int32,
    )
    psp = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max(seqlens_padded),
        max_seqlen_kv=max(seqlens_padded),
    )
    image_grid_thw = (
        torch.cat(grid_rows, dim=0) if grid_rows else None
    )
    return packed, psp, image_grid_thw, seqlens, seqlens_padded


def test_thd_matches_bshd_padded():
    """THD positions at every valid slot equal BSHD positions on the
    equivalent right-padded batch.
    """
    samples = _sample_bank()

    # BSHD side: right-pad to common max_len.
    max_len = max(s.numel() for s, _ in samples)
    input_ids_rows = []
    mask_rows = []
    grid_rows = []
    for tokens, grids in samples:
        L = tokens.numel()
        input_ids_rows.append(F.pad(tokens, (0, max_len - L), value=0))
        m = torch.zeros(max_len, dtype=torch.int64)
        m[:L] = 1
        mask_rows.append(m)
        if grids.numel() > 0:
            grid_rows.append(grids)
    bshd_input_ids = torch.stack(input_ids_rows)
    bshd_mask = torch.stack(mask_rows)
    bshd_grid = torch.cat(grid_rows, dim=0) if grid_rows else None

    bshd_pos, _ = get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        input_ids=bshd_input_ids,
        image_grid_thw=bshd_grid,
        attention_mask=bshd_mask,
    )
    # bshd_pos: [3, B, S_pad]

    # THD side: pack with a non-trivial divisor so the padded and
    # unpadded cu_seqlens diverge — this exercises the distinction.
    for divisible_by in (1, 4):
        packed_input_ids, psp, thd_grid, seqlens, seqlens_padded = (
            _pack_samples(samples, divisible_by=divisible_by)
        )
        thd_pos, _ = get_rope_index(
            spatial_merge_size=SPATIAL_MERGE_SIZE,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            input_ids=packed_input_ids,
            image_grid_thw=thd_grid,
            packed_seq_params=psp,
        )
        # thd_pos: [3, 1, T]
        assert thd_pos.shape == (
            3, 1, packed_input_ids.shape[1],
        ), f"bad THD shape {thd_pos.shape}"

        seg_starts = list(accumulate(seqlens_padded, initial=0))
        for k, (valid_len, seg_start) in enumerate(
            zip(seqlens, seg_starts)
        ):
            thd_slice = thd_pos[:, 0, seg_start:seg_start + valid_len]
            bshd_slice = bshd_pos[:, k, :valid_len]
            assert torch.equal(thd_slice, bshd_slice), (
                f"[divisible_by={divisible_by}] segment {k} "
                f"(valid_len={valid_len}, seg_start={seg_start}) "
                f"disagrees:\nTHD:\n{thd_slice}\nBSHD:\n{bshd_slice}"
            )


# -----------------------------------------------------------------------------
# Test 3: THD with no images (text-only packed)
# -----------------------------------------------------------------------------

def test_thd_text_only_restarts_per_segment():
    """Text-only THD: each segment gets a fresh ``[0..valid_len-1]`` range."""
    samples = [
        _build_sample(prefix_text_len=6, grids=[], suffix_text_len=0),
        _build_sample(prefix_text_len=11, grids=[], suffix_text_len=0),
        _build_sample(prefix_text_len=3, grids=[], suffix_text_len=0),
    ]
    packed_input_ids, psp, _, seqlens, seqlens_padded = _pack_samples(
        samples, divisible_by=4,
    )
    thd_pos, _ = get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        input_ids=packed_input_ids,
        image_grid_thw=None,
        packed_seq_params=psp,
    )

    seg_starts = list(accumulate(seqlens_padded, initial=0))
    for valid_len, seg_start in zip(seqlens, seg_starts):
        expected = (
            torch.arange(valid_len, dtype=thd_pos.dtype)
            .view(1, -1)
            .expand(3, -1)
        )
        got = thd_pos[:, 0, seg_start:seg_start + valid_len]
        assert torch.equal(got, expected), (
            f"text-only segment mismatch at seg_start={seg_start}, "
            f"valid_len={valid_len}:\n{got}\nexpected:\n{expected}"
        )


# -----------------------------------------------------------------------------
# Test 4: Explicit two-sequence batch with vision, both in BSHD and THD
# -----------------------------------------------------------------------------

def _two_image_samples():
    """Two samples, each with one image — the smallest case that can
    expose a bug where segment k > 0 positions leak state from segment
    k - 1 (e.g. non-restarted ``st_idx`` or a stale ``image_index``).
    """
    return [
        _build_sample(
            prefix_text_len=5,
            grids=[(1, 4, 4)],  # 4 image tokens after spatial merge
            suffix_text_len=3,
        ),
        _build_sample(
            prefix_text_len=4,
            grids=[(1, 4, 4)],
            suffix_text_len=6,
        ),
    ]


def test_bshd_batch_size_2_with_vision():
    """BSHD with ``B == 2``: both rows' positions restart at 0 and match
    the pinned reference.
    """
    samples = _two_image_samples()
    max_len = max(s.numel() for s, _ in samples)

    input_ids_rows, mask_rows, grid_rows = [], [], []
    for tokens, grids in samples:
        L = tokens.numel()
        input_ids_rows.append(F.pad(tokens, (0, max_len - L), value=0))
        m = torch.zeros(max_len, dtype=torch.int64)
        m[:L] = 1
        mask_rows.append(m)
        grid_rows.append(grids)

    input_ids = torch.stack(input_ids_rows)  # [2, S]
    attention_mask = torch.stack(mask_rows)
    image_grid_thw = torch.cat(grid_rows, dim=0)
    assert input_ids.shape[0] == 2

    old_pos, old_delta = _old_get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    new_pos, new_delta = get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    assert torch.equal(old_pos, new_pos), (
        f"BSHD B=2 position mismatch vs reference.\n"
        f"old:\n{old_pos}\nnew:\n{new_pos}"
    )
    assert torch.equal(old_delta, new_delta)

    # Both rows must start at position 0.
    for i in range(2):
        valid_len = int(attention_mask[i].sum().item())
        assert torch.all(new_pos[:, i, 0] == 0), (
            f"row {i} does not start at 0: {new_pos[:, i, 0]}"
        )
        # Sanity: positions within the valid region are strictly < valid_len
        # would be wrong (MRoPE can skip positions), so just check max.
        assert new_pos[:, i, :valid_len].max() < valid_len


def test_thd_batch_size_2_with_vision():
    """THD with 2 packed sequences: seg 1 positions restart at 0 and
    equal BSHD row 1 on the valid region (bit-identical).
    """
    samples = _two_image_samples()

    # BSHD reference.
    max_len = max(s.numel() for s, _ in samples)
    rows, masks, grids_bshd = [], [], []
    for tokens, grids in samples:
        L = tokens.numel()
        rows.append(F.pad(tokens, (0, max_len - L), value=0))
        m = torch.zeros(max_len, dtype=torch.int64)
        m[:L] = 1
        masks.append(m)
        grids_bshd.append(grids)
    bshd_input_ids = torch.stack(rows)
    bshd_mask = torch.stack(masks)
    bshd_grid = torch.cat(grids_bshd, dim=0)
    bshd_pos, _ = get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        input_ids=bshd_input_ids,
        image_grid_thw=bshd_grid,
        attention_mask=bshd_mask,
    )

    # THD packed version with a non-trivial divisor so padded and
    # unpadded cu_seqlens disagree.
    packed_input_ids, psp, thd_grid, seqlens, seqlens_padded = (
        _pack_samples(samples, divisible_by=4)
    )
    assert len(seqlens) == 2
    thd_pos, _ = get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        input_ids=packed_input_ids,
        image_grid_thw=thd_grid,
        packed_seq_params=psp,
    )

    seg_starts = list(accumulate(seqlens_padded, initial=0))
    for k, (valid_len, seg_start) in enumerate(
        zip(seqlens, seg_starts)
    ):
        thd_slice = thd_pos[:, 0, seg_start:seg_start + valid_len]
        bshd_slice = bshd_pos[:, k, :valid_len]
        assert torch.equal(thd_slice, bshd_slice), (
            f"seg {k} THD vs BSHD row {k} mismatch.\n"
            f"THD:\n{thd_slice}\nBSHD:\n{bshd_slice}"
        )
        # Critical: seg k must start at position 0 (bug 2 check).
        assert torch.all(thd_slice[:, 0] == 0), (
            f"seg {k} does not start at 0 — positions leaked from "
            f"previous segment: first col = {thd_slice[:, 0]}"
        )


if __name__ == "__main__":
    test_bshd_matches_old_reference()
    print("[ok] test_bshd_matches_old_reference")
    test_thd_matches_bshd_padded()
    print("[ok] test_thd_matches_bshd_padded")
    test_thd_text_only_restarts_per_segment()
    print("[ok] test_thd_text_only_restarts_per_segment")
    test_bshd_batch_size_2_with_vision()
    print("[ok] test_bshd_batch_size_2_with_vision")
    test_thd_batch_size_2_with_vision()
    print("[ok] test_thd_batch_size_2_with_vision")
    print("All parity tests passed.")
