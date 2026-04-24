# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MRoPE (Multimodal Rotary Position Embedding) position ID computation.

Computes 3D position IDs for Qwen3.5-VL: for text tokens all three
dimensions share sequential positions; for image/video tokens the three
dimensions encode (temporal, height, width) in the merged spatial grid.

Supports two input layouts:

* **BSHD** — ``input_ids`` is ``[B, S]``; each row is an independent
  sample (possibly padded) and ``attention_mask`` marks valid tokens.
* **THD** — ``input_ids`` is ``[1, T]``, a concatenation of ``N``
  sub-sequences. ``packed_seq_params.cu_seqlens_q_padded`` gives the
  physical segment boundaries in the packed tensor and
  ``cu_seqlens_q`` gives the valid (unpadded) token count inside each
  segment. Position IDs restart at 0 at every segment boundary; image
  / video grid rows are consumed in packed order across segments.

Ported from Megatron-Bridge ``get_rope_index`` (which itself is adapted
from HF ``Qwen3VLForConditionalGeneration.get_rope_index``).  The inner
loop iterates over vision occurrences, not individual tokens.
"""

from typing import Optional

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams


def _build_sample_mrope_positions(
    sample_input_ids: Tensor,
    image_grid_thw: Optional[Tensor],
    video_grid_thw: Optional[Tensor],
    image_index: int,
    video_index: int,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> tuple[Tensor, int, int]:
    """Compute MRoPE position IDs for a single sub-sequence.

    Walks vision occurrences in ``sample_input_ids`` and produces a
    ``[3, L]`` position tensor whose values start at 0. Advances
    ``image_index`` / ``video_index`` through ``image_grid_thw`` /
    ``video_grid_thw`` so callers can keep a running cursor across
    multiple sub-sequences.
    """
    vision_start_indices = torch.argwhere(
        sample_input_ids == vision_start_token_id,
    ).squeeze(1)
    vision_tokens = sample_input_ids[vision_start_indices + 1]
    image_nums = int((vision_tokens == image_token_id).sum())
    video_nums = int((vision_tokens == video_token_id).sum())
    input_tokens = sample_input_ids.tolist()
    llm_pos_ids_list: list = []
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

    if llm_pos_ids_list:
        positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    else:
        positions = torch.zeros(
            3, 0,
            dtype=sample_input_ids.dtype,
            device=sample_input_ids.device,
        )
    return positions, image_index, video_index


def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: Optional[Tensor] = None,
    image_grid_thw: Optional[Tensor] = None,
    video_grid_thw: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> tuple[Tensor, Tensor]:
    """Compute 3D MRoPE position IDs for Qwen3-VL / Qwen3.5-VL.

    Qwen3-VL uses timestamps rather than absolute time position IDs.

    For text tokens all three dimensions share sequential positions.
    For vision tokens the three dimensions encode (temporal, height,
    width) in the merged spatial grid.

    Args:
        spatial_merge_size: Merge factor for spatial dimensions.
        image_token_id: Token ID for image placeholders.
        video_token_id: Token ID for video placeholders.
        vision_start_token_id: Token ID marking start of a vision region.
        input_ids: ``[B, S]`` in BSHD or ``[1, T]`` in THD.
        image_grid_thw: ``[num_images, 3]`` per-image
            ``(temporal, height, width)`` in patch-grid units. Rows are
            consumed in the order their image tokens appear in
            ``input_ids`` (packed order across segments in THD).
        video_grid_thw: ``[num_videos, 3]`` per-video grid dimensions.
        attention_mask: ``[B, S]`` mask (1 = keep, 0 = pad). BSHD only.
        packed_seq_params: When provided, selects the THD branch and
            supplies segment boundaries via ``cu_seqlens_q`` (valid
            lengths) and ``cu_seqlens_q_padded`` (packed layout).

    Returns:
        ``(position_ids, mrope_position_deltas)`` where *position_ids*
        has shape ``[3, B, S]`` (``[3, 1, T]`` in THD).
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0,
        )
        video_grid_thw[:, 0] = 1

    # -----------------------------------------------------------------
    # THD (packed) branch
    # -----------------------------------------------------------------
    if packed_seq_params is not None and input_ids is not None:
        cu_seqlens = packed_seq_params.cu_seqlens_q
        cu_seqlens_padded = getattr(
            packed_seq_params, "cu_seqlens_q_padded", None,
        )
        if cu_seqlens_padded is None:
            cu_seqlens_padded = cu_seqlens

        assert (
            input_ids.dim() == 2 and input_ids.shape[0] == 1
        ), "THD get_rope_index expects input_ids shape [1, T]"

        total_tokens = input_ids.shape[1]
        device = input_ids.device

        # Padding slots default to 1 (matches BSHD convention where
        # masked positions get filled with 1).
        position_ids = torch.ones(
            3, 1, total_tokens,
            dtype=input_ids.dtype, device=device,
        )
        deltas: list = []
        image_index = 0
        video_index = 0
        num_segs = cu_seqlens.numel() - 1

        for k in range(num_segs):
            seg_start = int(cu_seqlens_padded[k].item())
            valid_len = int(
                cu_seqlens[k + 1].item() - cu_seqlens[k].item()
            )
            valid_end = seg_start + valid_len

            if valid_len == 0:
                deltas.append(0)
                continue

            sample_input_ids = input_ids[0, seg_start:valid_end]

            if (
                image_grid_thw is not None
                or video_grid_thw is not None
            ):
                (
                    positions,
                    image_index,
                    video_index,
                ) = _build_sample_mrope_positions(
                    sample_input_ids=sample_input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    image_index=image_index,
                    video_index=video_index,
                    spatial_merge_size=spatial_merge_size,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    vision_start_token_id=vision_start_token_id,
                )
            else:
                positions = (
                    torch.arange(valid_len, device=device)
                    .view(1, -1)
                    .expand(3, -1)
                )

            position_ids[:, 0, seg_start:valid_end] = positions.to(
                device=device, dtype=position_ids.dtype,
            )

            if positions.numel() > 0:
                deltas.append(
                    int(positions.max().item()) + 1 - valid_len
                )
            else:
                deltas.append(0)

        mrope_position_deltas = torch.tensor(
            deltas, device=device,
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # -----------------------------------------------------------------
    # BSHD branch with vision
    # -----------------------------------------------------------------
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        elif attention_mask.dim() > 2:
            attention_mask = attention_mask.any(dim=-1)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            attention_mask = attention_mask.to(dtype=total_input_ids.dtype)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        mrope_position_deltas = []
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, sample_input_ids in enumerate(total_input_ids):
            sample_input_ids = sample_input_ids[attention_mask[i] == 1]
            (
                llm_positions,
                image_index,
                video_index,
            ) = _build_sample_mrope_positions(
                sample_input_ids=sample_input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_index=image_index,
                video_index=video_index,
                spatial_merge_size=spatial_merge_size,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                vision_start_token_id=vision_start_token_id,
            )
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

    # -----------------------------------------------------------------
    # Text-only fallback
    # -----------------------------------------------------------------
    if attention_mask is not None:
        if attention_mask.dim() > 2:
            attention_mask = attention_mask.any(dim=-1)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            attention_mask = attention_mask.to(dtype=torch.long)
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
