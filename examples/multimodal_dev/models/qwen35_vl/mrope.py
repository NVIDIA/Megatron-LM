# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MRoPE (Multimodal Rotary Position Embedding) position ID computation.

Computes 3D position IDs for Qwen3.5-VL: for text tokens all three
dimensions share sequential positions; for image/video tokens the three
dimensions encode (temporal, height, width) in the merged spatial grid.

Ported from Megatron-Bridge ``get_rope_index`` (which itself is adapted
from HF ``Qwen3VLForConditionalGeneration.get_rope_index``).  The inner
loop iterates over vision occurrences, not individual tokens.
"""

from typing import Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from torch import Tensor


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
        input_ids: ``[B, S]`` token IDs.
        image_grid_thw: ``[num_images, 3]`` per-image
            ``(temporal, height, width)`` in patch-grid units.
        video_grid_thw: ``[num_videos, 3]`` per-video grid dimensions.
        attention_mask: ``[B, S]`` mask (1 = keep, 0 = pad).
        packed_seq_params: Packed-sequence metadata (``cu_seqlens_q``).

    Returns:
        ``(position_ids, mrope_position_deltas)`` where *position_ids*
        has shape ``[3, B, S]``.
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0,
        )
        video_grid_thw[:, 0] = 1

    if (
        packed_seq_params is not None
        and attention_mask is None
        and input_ids is not None
    ):
        cu_seqlens = packed_seq_params.cu_seqlens_q
        if cu_seqlens is not None and cu_seqlens.numel() >= 2:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            attention_mask = torch.zeros_like(
                input_ids, dtype=input_ids.dtype,
            )
            max_len = attention_mask.shape[1]
            for i, seq_len in enumerate(seq_lens.tolist()):
                valid = min(int(seq_len), max_len)
                attention_mask[i, :valid] = 1
        else:
            attention_mask = torch.ones_like(input_ids)

    mrope_position_deltas = []

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

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(
                3, -1,
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

    else:
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
