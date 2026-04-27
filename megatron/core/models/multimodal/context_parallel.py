# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Multimodal Sequence Parallel (SP) and Context Parallel (CP) functionality."""

import math

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
)


def get_padding(
    seq_len,
    cp_size,
    tp_size,
    has_sp,
    decoder_tp_comm_overlap=False,
    decoder_seq_len=None,
    fp8_enabled=False,
    fp8_recipe=None,
):
    """Calculate padding needed for SP, CP, TP comm overlap, and FP8.

    Args:
        seq_len (int): Model sequence length.
        cp_size (int): Context parallel size.
        tp_size (int): Tensor parallel size.
        has_sp (bool): Model uses sequence parallelism.
        decoder_tp_comm_overlap (bool): Decoder (LLM) uses tensor parallel communication overlap.
        decoder_seq_len (int): Decoder (LLM) maximum sequence length.
        fp8_enabled (bool): FP8 is enabled.
        fp8_recipe (str): FP8 recipe. Affects required padding.

    Returns:
        padding (int): Padding needed given model configuration.
    """

    padding = 0
    # TP Comm overlap is performed with combined text+image embeddings.
    if has_sp and decoder_tp_comm_overlap:
        # If TP Comm Overlap is enabled for combined text+image embedding in LM backbone,
        # user needs to provide decoder_seq_len with any potential padding needed for SP+CP
        assert (
            decoder_seq_len is not None
        ), "Please provide decoder seq length when using TP comm overlap for LM backbone"
        padding = decoder_seq_len - seq_len
        return padding

    padding_factor = 1
    if has_sp and cp_size > 1:
        # Padding to multiple of tp_size * cp_size * 2 when using CP + SP.
        padding_factor = tp_size * cp_size * 2
    elif cp_size > 1:
        padding_factor = cp_size * 2
    elif has_sp:
        padding_factor = tp_size
    elif fp8_enabled:
        padding_factor = 32 if fp8_recipe == "mxfp8" else 16

    padding = int((seq_len + padding_factor - 1) // padding_factor * padding_factor) - seq_len

    return padding


def get_packed_seq_params(tokens, img_seq_len, padding_needed, cp_size, use_packed_sequence=False):
    """Get PackedSeqParams for CP.

    Args:
        tokens (torch.Tensor): [batch, seq_len] input tokens.
        img_seq_len (int): Image sequence length.
        padding_needed (int): Padding to add.
        cp_size (int): Context parallel size.
        use_packed_sequence (bool): Uses sequence packing.

    Returns:
        packed_seq_params (PackedSeqParams): Parameters to be sent to Transformer Engine.
    """
    batch_size = tokens.shape[0]
    # Calculate the valid token seq len that LM backbone should compute on
    combined_valid_seqlen = tokens.shape[1] + img_seq_len - padding_needed
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * (combined_valid_seqlen),
        step=(combined_valid_seqlen),
        dtype=torch.int32,
        device=tokens.device,
    )
    # Calculate the total padded token seq len
    combined_padded_seqlen = tokens.shape[1] + img_seq_len
    cu_seqlens_padded = None
    qkv_format = 'sbhd'
    if cp_size > 1 and (padding_needed > 0 or use_packed_sequence):
        # Provide cu_seqlens_<q/kv>_padded for CP support
        cu_seqlens_padded = torch.arange(
            0,
            (batch_size + 1) * (combined_padded_seqlen),
            step=(combined_padded_seqlen),
            dtype=torch.int32,
            device=tokens.device,
        )
        # CP with padding mask type requires THD format
        qkv_format = 'thd'

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=combined_padded_seqlen,
        max_seqlen_kv=combined_padded_seqlen,
        qkv_format=qkv_format,
    )

    return packed_seq_params


def split_to_context_parallel_ranks(global_t, pad_value=0):
    """Split the tensor global_t into context parallel world size parts.

    Args:
        global_t: [batch, ...]
        pad_value: Value to pad the last rank with.

    Returns:
        local_t: [samples_per_rank, ...]. samples_per_rank is the # of samples per CP rank.
        global_pad: Total padding to have equal samples_per_rank across context parallel ranks.
    """
    cp_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()

    samples_per_rank = (global_t.shape[0] + cp_size - 1) // cp_size
    local_t = global_t[cp_rank * samples_per_rank : (cp_rank + 1) * samples_per_rank]
    global_pad = samples_per_rank * cp_size - global_t.shape[0]

    if local_t.shape[0] < samples_per_rank:
        local_pad = samples_per_rank - local_t.shape[0]
        zeros = torch.full(
            (local_pad, *local_t.shape[1:]), pad_value, device=local_t.device, dtype=local_t.dtype
        )
        local_t = torch.cat([local_t, zeros], dim=0)

    return local_t, global_pad


def _gather_along_second_dim(local_t):
    group = get_context_parallel_group()
    cp_size = get_context_parallel_world_size()
    if cp_size == 1:
        return local_t

    tensor_list = [
        torch.empty(local_t.shape, device=local_t.device, dtype=local_t.dtype)
        for _ in range(cp_size)
    ]
    torch.distributed.all_gather(tensor_list, local_t, group=group)
    return torch.cat(tensor_list, dim=1)


def _reduce_scatter_along_second_dim(global_t):
    cp_size = get_context_parallel_world_size()
    if cp_size == 1:
        return global_t

    assert global_t.shape[1] % cp_size == 0
    samples_per_rank = global_t.shape[1] // cp_size

    tensor_list = [
        global_t[:, cp_rank * samples_per_rank : (cp_rank + 1) * samples_per_rank]
        for cp_rank in range(cp_size)
    ]

    local_t = torch.zeros(
        global_t.shape[0],
        samples_per_rank,
        *global_t.shape[2:],
        device=global_t.device,
        dtype=global_t.dtype,
    )

    torch.distributed.reduce_scatter(local_t, tensor_list, group=get_context_parallel_group())
    return local_t


class GatherFromContextParallelRanks(torch.autograd.Function):
    """Gather the input from context parallel ranks."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_second_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_second_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_second_dim(grad_output)


def gather_from_context_parallel_ranks(local_t, global_pad):
    global_t = GatherFromContextParallelRanks.apply(local_t)
    if global_pad > 0:
        global_t = global_t[:, :-global_pad]
    return global_t


def gather_from_context_parallel_ranks_dynamic_res(local_t, num_padded_imgs=0):
    """Gather dynamic-resolution tensors (variable seq per rank) from CP ranks."""
    cp_size = get_context_parallel_world_size()
    shape = torch.as_tensor(local_t.shape, device=local_t.device)
    shapes = [torch.empty_like(shape) for _ in range(cp_size)]

    torch.distributed.all_gather(shapes, shape, group=get_context_parallel_group())

    inputs = [local_t] * cp_size
    outputs = [torch.empty(*s, dtype=local_t.dtype, device=local_t.device) for s in shapes]
    torch.distributed.nn.functional.all_to_all(outputs, inputs, group=get_context_parallel_group())

    if num_padded_imgs > 0:
        outputs = outputs[:-num_padded_imgs]

    return torch.cat(outputs, dim=0)


def _compute_tubelet_aware_split_points(num_frames, temporal_patch_size, cp_size, total_frames):
    """Compute split points that respect tubelet boundaries within videos."""
    T = temporal_patch_size
    target_per_rank = total_frames / cp_size

    media_boundaries = [0]
    for nf in num_frames:
        media_boundaries.append(media_boundaries[-1] + nf)

    split_points = [0]
    for rank in range(1, cp_size):
        target_split = int(rank * target_per_rank)

        media_idx = 0
        for i, boundary in enumerate(media_boundaries[1:], 1):
            if boundary > target_split:
                media_idx = i - 1
                break
        else:
            media_idx = len(num_frames) - 1

        media_start = media_boundaries[media_idx]
        media_end = media_boundaries[media_idx + 1]
        nf = num_frames[media_idx]
        num_tubelets = math.ceil(nf / T)

        if num_tubelets <= 1:
            if target_split - media_start < media_end - target_split:
                split_point = media_start
            else:
                split_point = media_end
        else:
            offset_in_media = target_split - media_start
            tubelet_idx = round(offset_in_media / T)
            tubelet_idx = max(1, min(tubelet_idx, num_tubelets - 1))
            split_point = media_start + tubelet_idx * T
            split_point = min(split_point, media_end)

        split_point = max(split_point, split_points[-1])
        split_points.append(split_point)

    split_points.append(total_frames)
    return split_points


def _split_num_frames(num_frames, lb, ub):
    """Return per-media frame counts clipped to the range ``[lb, ub)``."""
    new_num_frames = []
    frame_idx = 0
    for nf in num_frames:
        media_start = frame_idx
        media_end = frame_idx + nf
        overlap_start = max(media_start, lb)
        overlap_end = min(media_end, ub)
        if overlap_start < overlap_end:
            new_num_frames.append(overlap_end - overlap_start)
        frame_idx = media_end
    return new_num_frames


def split_to_context_parallel_ranks_dynamic_res(
    global_t,
    global_imgs_sizes,
    global_packed_seq_params,
    fp8_enabled=False,
    patch_dim=16,
    num_frames=None,
    temporal_patch_size=1,
):
    """Split patched vision input across CP ranks.

    ``global_packed_seq_params`` provides per-image seqlens; the split respects them
    so each rank owns an integer number of images. When ``temporal_patch_size > 1``,
    splits also respect tubelet boundaries and ``num_frames`` is required.

    Returns:
        (local_t, local_imgs_sizes, local_packed_seq_params, has_padding,
         num_padded_ranks, local_num_frames)
    """
    cp_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()

    use_tubelet_aware_split = temporal_patch_size > 1
    if use_tubelet_aware_split:
        assert num_frames is not None, (
            f"num_frames must be provided when using temporal compression "
            f"(temporal_patch_size={temporal_patch_size})"
        )
        num_frames_list = num_frames.tolist() if hasattr(num_frames, "tolist") else list(num_frames)

    cu_seqlens = global_packed_seq_params.cu_seqlens_q

    num_imgs = len(global_imgs_sizes)
    if use_tubelet_aware_split:
        T = temporal_patch_size
        total_tubelets = sum(math.ceil(nf / T) for nf in num_frames_list)
        num_padded_imgs = max(0, cp_size - total_tubelets)
    else:
        num_padded_imgs = max(0, cp_size - num_imgs)

    dummy_img_size = torch.tensor(
        [[patch_dim, patch_dim]],
        device=global_imgs_sizes.device,
        dtype=global_imgs_sizes.dtype,
    )
    hidden_dim = int(global_t.shape[2])
    dummy_seqlen = int(patch_dim * patch_dim * 3 / hidden_dim)
    dummy_img = torch.zeros(
        [1, dummy_seqlen, hidden_dim], device=global_t.device, dtype=global_t.dtype
    )

    def _add_dummies(n, global_t, global_imgs_sizes, cu_seqlens, num_frames_list):
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        for _ in range(n):
            global_imgs_sizes = torch.cat([global_imgs_sizes, dummy_img_size], dim=0)
            global_t = torch.cat([global_t, dummy_img], dim=1)
            seqlens = torch.cat(
                [
                    seqlens,
                    torch.tensor([dummy_seqlen], device=seqlens.device, dtype=seqlens.dtype),
                ]
            )
        if use_tubelet_aware_split:
            num_frames_list = num_frames_list + [1] * n
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=cu_seqlens.device, dtype=cu_seqlens.dtype),
                torch.cumsum(seqlens, dim=0),
            ]
        )
        return global_t, global_imgs_sizes, cu_seqlens, num_frames_list

    if num_padded_imgs > 0:
        global_t, global_imgs_sizes, cu_seqlens, num_frames_list = _add_dummies(
            num_padded_imgs,
            global_t,
            global_imgs_sizes,
            cu_seqlens,
            num_frames_list if use_tubelet_aware_split else None,
        )

    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    total_frames = len(global_imgs_sizes)
    num_padded_ranks = num_padded_imgs

    if use_tubelet_aware_split:
        for _retry in range(cp_size):
            total_frames = len(global_imgs_sizes)
            split_points = _compute_tubelet_aware_split_points(
                num_frames_list, temporal_patch_size, cp_size, total_frames
            )
            num_empty = sum(1 for k in range(cp_size) if split_points[k] == split_points[k + 1])
            if num_empty == 0:
                break
            global_t, global_imgs_sizes, cu_seqlens, num_frames_list = _add_dummies(
                num_empty, global_t, global_imgs_sizes, cu_seqlens, num_frames_list
            )
            num_padded_imgs += num_empty
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        original_total_frames = total_frames - num_padded_imgs
        if num_padded_imgs > 0 and original_total_frames not in split_points:
            for k in range(cp_size):
                if split_points[k] < original_total_frames < split_points[k + 1]:
                    split_points[k + 1] = original_total_frames
                    break

        num_padded_ranks = 0
        if num_padded_imgs > 0:
            for i in range(cp_size - 1, -1, -1):
                if split_points[i] >= original_total_frames:
                    num_padded_ranks += 1
                else:
                    break

        lb = split_points[cp_rank]
        ub = split_points[cp_rank + 1]
        local_num_frames = _split_num_frames(num_frames_list, lb, ub)
    else:
        seq_per_rank = total_frames // cp_size
        lb = cp_rank * seq_per_rank
        ub = (cp_rank + 1) * seq_per_rank if cp_rank < cp_size - 1 else len(cu_seqlens)
        local_num_frames = None

    seqlens_local = torch.cat([torch.tensor([0], device=seqlens.device), seqlens[lb:ub]])
    cu_seqlens_local = torch.cumsum(seqlens_local, dim=0).to(torch.int32)

    final_seqlen = cu_seqlens_local[-1]

    pad_img = None
    if fp8_enabled:
        padding_needed = get_padding(final_seqlen, 1, 1, False, fp8_enabled=True)
        patch_dim = 16
        if padding_needed > 0:
            pad_img = torch.zeros(
                [1, padding_needed, patch_dim * patch_dim * 3],
                device=global_t.device,
                dtype=global_t.dtype,
            )
            cu_seqlens_local = torch.cat(
                [
                    cu_seqlens_local,
                    torch.tensor(
                        [final_seqlen + padding_needed],
                        device=cu_seqlens_local.device,
                        dtype=cu_seqlens_local.dtype,
                    ),
                ]
            )

    has_padding = pad_img is not None

    local_packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_local,
        cu_seqlens_kv=cu_seqlens_local,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
    )
    max_seqlen_local = max(seqlens_local).to(torch.int32)
    local_packed_seq_params.max_seqlen_q = max_seqlen_local
    local_packed_seq_params.max_seqlen_kv = max_seqlen_local

    local_imgs_sizes = global_imgs_sizes[lb:ub]
    if has_padding:
        local_imgs_sizes = torch.cat(
            [
                local_imgs_sizes,
                torch.tensor(
                    [[patch_dim, patch_dim * padding_needed]],
                    device=local_imgs_sizes.device,
                    dtype=local_imgs_sizes.dtype,
                ),
            ]
        )

    offset = torch.cumsum(seqlens[:lb], dim=0)[-1] if lb > 0 else 0

    if not has_padding:
        local_t = global_t[:, offset + cu_seqlens_local[0] : offset + cu_seqlens_local[-1]]
    else:
        local_t = torch.cat(
            [global_t[:, offset + cu_seqlens_local[0] : offset + cu_seqlens_local[-2]], pad_img],
            dim=1,
        )

    if local_num_frames is not None:
        local_num_frames = torch.tensor(
            local_num_frames, dtype=torch.int32, device=global_imgs_sizes.device
        )

    return (
        local_t,
        local_imgs_sizes,
        local_packed_seq_params,
        has_padding,
        num_padded_ranks,
        local_num_frames,
    )
