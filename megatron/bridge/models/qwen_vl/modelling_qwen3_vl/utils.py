# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/utils.py
# Source license: Apache-2.0.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import nn

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
    Qwen3VLTransformerConfig,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import get_tensor_model_parallel_group_if_none


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
class Qwen3VLVisionPatchEmbed(nn.Module):
    """
    Vision Patch Embed for Qwen3VL vision model.
    """

    def __init__(self, config: Qwen3VLTransformerConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """
    Vision Rotary Embedding for Qwen3VL vision model.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int) -> torch.Tensor:
        if not hasattr(self, "inv_freq"):
            inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(
                        0, self.dim, 2, dtype=torch.float, device=torch.cuda.current_device()
                    )
                    / self.dim
                )
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


@dataclass
class PatchMergerSubmodules:
    """
    Patch Merger Submodules for Qwen3VL vision model.
    """

    patch_norm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class Qwen3VLVisionPatchMerger(MegatronModule):
    """
    Vision Patch Merger for Qwen3VL vision model.
    """

    def __init__(
        self,
        config: Qwen3VLTransformerConfig,
        submodules: PatchMergerSubmodules,
        use_postshuffle_norm=False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.input_size = config.hidden_size
        if self.use_postshuffle_norm:
            self.input_size = self.hidden_size
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=False)

        self.patch_norm = build_module(
            submodules.patch_norm,
            config=self.config,
            hidden_size=self.input_size,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="patch_fc1",
            tp_group=self.tp_group,
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.hidden_size,
            self.config.out_hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="patch_fc2",
            tp_group=self.tp_group,
        )

    def forward(self, hidden_states):
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.patch_norm(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)

        hidden_states, _ = self.linear_fc1(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        output, _ = self.linear_fc2(hidden_states)

        return output


def split_part_by_cp_tp(cp_size, cp_rank, tp_size, tp_rank, split_size):
    """
    Get the split part by CP and TP for Qwen3VL vision model using zigzag pattern.
    """
    part_list = list(range(split_size))

    cp_rank2 = 2 * cp_size - cp_rank - 1
    cp_part_list = (
        part_list[cp_rank * tp_size : (cp_rank + 1) * tp_size]
        + part_list[cp_rank2 * tp_size : (cp_rank2 + 1) * tp_size]
    )

    assert len(cp_part_list) % tp_size == 0
    echo_tp_len = len(cp_part_list) // tp_size
    cp_tp_part_list = cp_part_list[tp_rank * echo_tp_len : (tp_rank + 1) * echo_tp_len]
    return cp_tp_part_list


def split_deepstack_embs(
    visual_pos_masks: torch.Tensor,
    deepstack_visual_embeds: list[torch.Tensor],
    tp_size: int = 1,
    tp_rank: int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
    sequence_parallel: bool = False,
):
    """
    Split the deepstack visual embeddings by CP and TP for Qwen3VL vision model.
    NOTE:
        first split by cp(zigzag), then split by sp
        for example cp=2/tp=4
        visual_pos_masks will split in 16 part:
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        first split by cp(zigzag) is:
        cp_rank0: [0, 1, 2, 3, 12, 13, 14, 15]
        cp_rank1: [4, 5, 6, 7, 8, 9, 10, 11]
        then split by sp:
        cp_rank0/tp_rank0 = [0, 1]
        cp_rank0/tp_rank1 = [2, 3]
        ...
        cp_rank1/tp_rank2 = [8, 9]
        cp_rank1/tp_rank3 = [10, 11]
    """
    if not sequence_parallel:
        tp_size = 1
        tp_rank = 0
    split_size = tp_size
    if cp_size > 1:
        split_size *= cp_size * 2
    if split_size == 1 or visual_pos_masks is None:
        return visual_pos_masks, deepstack_visual_embeds

    assert visual_pos_masks.dim() == 2
    assert visual_pos_masks.shape[-1] % split_size == 0
    batch_size = visual_pos_masks.size(0)

    cp_tp_part_list = split_part_by_cp_tp(cp_size, cp_rank, tp_size, tp_rank, split_size)
    visual_pos_masks_list = visual_pos_masks.chunk(split_size, dim=-1)
    embed_lens = [ele.sum(-1) for ele in visual_pos_masks_list]

    embed_lens = torch.stack(embed_lens, dim=-1)
    embed_cu_lens = embed_lens.view(-1).cumsum(dim=-1).tolist()
    assert len(embed_cu_lens) == split_size * batch_size
    embed_cu_lens = [0] + embed_cu_lens

    cp_tp_slices = []
    for i in range(batch_size):
        for idx in cp_tp_part_list:
            idx += i * split_size
            cp_tp_slices.append(slice(embed_cu_lens[idx], embed_cu_lens[idx + 1]))

    deepstack_visual_embeds_ret = []
    for deepstack_visual_embed in deepstack_visual_embeds:
        tmp_slice_tensor = []
        for tp_slice in cp_tp_slices:
            tmp_slice_tensor.append(deepstack_visual_embed[tp_slice])
        deepstack_visual_embeds_ret.append(torch.cat(tmp_slice_tensor, dim=0))

    visual_pos_masks_ret = torch.cat([visual_pos_masks_list[i] for i in cp_tp_part_list], dim=-1)

    return visual_pos_masks_ret, deepstack_visual_embeds_ret


def find_vision_id_index(input_ids: torch.Tensor, image_token_id: int, video_token_id: int):
    """
    Find the vision id index for Qwen3VL vision model.
    """
    assert input_ids.dim() == 1, "input_ids should be flaaten"
    if input_ids.numel() == 0:
        return []

    device = input_ids.device
    dtype = input_ids.dtype
    assert dtype in [torch.int, torch.int64]

    # keep the value of image_token_id/video_token_id value, others are -1
    code = torch.where(
        (input_ids == image_token_id) | (input_ids == video_token_id),
        input_ids,
        torch.tensor(-1, device=device, dtype=dtype),
    )

    # find the change idx
    first = torch.tensor([True], device=device, dtype=torch.bool)
    change = torch.cat([first, code[1:] != code[:-1]])
    change_idx = torch.nonzero(change, as_tuple=False).flatten()

    # only keep the change of image_token_id/video_token_id
    keep = code[change_idx] > 0
    starts = change_idx[keep]

    # last change position is input_ids.numel()
    next_change = torch.cat(
        [change_idx[1:], torch.tensor([input_ids.numel()], device=device, dtype=change_idx.dtype)]
    )
    ends = next_change[keep]

    vals = code[starts]
    starts_cpu = starts.tolist()
    ends_cpu = ends.tolist()
    vals_cpu = vals.tolist()
    return [(int(s), int(e), int(v)) for s, e, v in zip(starts_cpu, ends_cpu, vals_cpu)]


def reorganize_inputs(
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor = None,
    pixel_values_videos: torch.Tensor = None,
    image_grid_thw: torch.Tensor = None,
    video_grid_thw: torch.Tensor = None,
    image_input_mask: torch.Tensor = None,
    video_input_mask: torch.Tensor = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    square_merge_size: int = 4,
):
    """
    Reorganize the inputs for Qwen3VL vision model.
    """
    if pixel_values is None:
        if video_input_mask is None and pixel_values_videos is not None:
            video_input_mask = (input_ids == video_token_id).contiguous()
        return pixel_values_videos, video_grid_thw, video_input_mask

    if pixel_values_videos is None:
        if image_input_mask is None and pixel_values is not None:
            image_input_mask = (input_ids == image_token_id).contiguous()
        return pixel_values, image_grid_thw, image_input_mask

    image_thw_cpu = image_grid_thw.tolist()
    video_thw_cpu = video_grid_thw.tolist()
    vision_indexs = find_vision_id_index(input_ids.view(-1), image_token_id, video_token_id)
    len_split = sum([thw[0] for thw in image_thw_cpu])
    len_split += sum([thw[0] for thw in video_thw_cpu])
    assert len_split == len(vision_indexs)

    vision_values = []
    vision_grid_thw = []
    idx = 0
    video_idx = 0
    image_idx = 0
    video_seqlen = 0
    image_seqlen = 0
    while idx < len(vision_indexs):
        start, end, token_id = vision_indexs[idx]
        if token_id == image_token_id:
            seqlen = 0
            thw = image_thw_cpu[image_idx]
            for i in range(thw[0]):
                start, end, token_id = vision_indexs[idx + i]
                assert token_id == image_token_id
                seqlen += (end - start) * square_merge_size
            assert seqlen == thw[0] * thw[1] * thw[2]
            vision_values.append(pixel_values[image_seqlen : (image_seqlen + seqlen)])
            vision_grid_thw.append(thw)

            image_idx += 1
            idx += thw[0]
            image_seqlen += seqlen
        elif token_id == video_token_id:
            seqlen = 0
            thw = video_thw_cpu[video_idx]
            for i in range(thw[0]):
                start, end, token_id = vision_indexs[idx + i]
                assert token_id == video_token_id
                seqlen += (end - start) * square_merge_size
            assert seqlen == thw[0] * thw[1] * thw[2]
            vision_values.append(pixel_values_videos[video_seqlen : (video_seqlen + seqlen)])
            vision_grid_thw.append(thw)

            video_idx += 1
            idx += thw[0]
            video_seqlen += seqlen
        else:
            raise AssertionError(f"should not have {token_id=}")

    if video_input_mask is None:
        video_input_mask = input_ids == video_token_id

    if image_input_mask is None:
        image_input_mask = input_ids == image_token_id

    vision_values = torch.cat(vision_values)
    vision_grid_thw = torch.tensor(
        vision_grid_thw, device=image_grid_thw.device, dtype=image_grid_thw.dtype
    )
    vision_input_mask = video_input_mask | image_input_mask

    return vision_values, vision_grid_thw, vision_input_mask


# reference: megatron/training/utils.py get_batch_on_this_cp_rank
def split_data_cp_rank(val: torch.Tensor, cp_size: int, seq_dim: int, cp_rank: int = None):
    """
    Split the data by CP rank for Qwen3VL vision model, using zigzag pattern.
    """
    assert cp_size > 1
    assert 0 == val.shape[seq_dim] % (2 * cp_size), f"{val.shape=} {cp_size=}"
    assert cp_rank is not None
    if val is None:
        return val

    val = val.view(
        *val.shape[0:seq_dim],
        2 * cp_size,
        val.shape[seq_dim] // (2 * cp_size),
        *val.shape[(seq_dim + 1) :],
    )

    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
    val = val.index_select(seq_dim, index)
    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])

    return val


def expand_thw(thw: torch.Tensor) -> torch.Tensor:
    """
    Expand the THW for Qwen3VL vision model.
    """
    assert thw.dim() == 2
    repeats = thw[:, 0].to(torch.long)
    assert torch.all(repeats > 0), "thw[:,0] must be > 0"

    idx = torch.arange(thw.size(0), device=thw.device).repeat_interleave(repeats)
    out = thw[idx].clone()
    out[:, 0] = 1
    return out


def collapse_thw(expanded: torch.Tensor) -> torch.Tensor:
    """
    Collapse the THW for Qwen3VL vision model.
    """
    assert expanded.dim() == 2
    assert expanded.size(1) >= 2
    if expanded.shape[0] < 2:
        return expanded

    # find the diff
    other = expanded[:, 1:]
    prev = torch.cat([other[:1], other[:-1]], dim=0)
    change = (other != prev).any(dim=1)
    # the index0 must be now row
    change[0] = True

    # find the diff
    starts = torch.nonzero(change, as_tuple=False).squeeze(1)
    ends = torch.cat([starts[1:], torch.tensor([other.size(0)], device=other.device)]) - 1
    counts = ends - starts + 1

    rows_other = other[starts]
    result_first_col = counts.to(expanded.dtype).unsqueeze(1)
    result = torch.cat([result_first_col, rows_other], dim=1)
    return result


# also can use in qwen2vl/qwen2.5vl
def qwen2vl_pad_and_split(
    cp_size: int,
    hw_factor: int,
    pixel_values: list[torch.Tensor],
    image_grid_thws: list[torch.Tensor],
):
    """
    Split the pixel values and image grid thws for Qwen3VL vision model.
    """
    assert len(pixel_values) == len(image_grid_thws)
    # split the pixel_values
    split_pixel_values = []
    split_image_grid_thws = []
    for pixel_value, image_grid_thw in zip(pixel_values, image_grid_thws):
        split_image_grid_thw = list(torch.split(image_grid_thw, 1, dim=0))
        split_image_grid_thws.extend(split_image_grid_thw)
        slice_begin = 0
        for ele in split_image_grid_thw:
            slice_end = slice_begin + ele.prod().item()
            split_pixel_values.append(pixel_value[slice_begin:slice_end].clone())
            slice_begin = slice_end

    pixel_values = split_pixel_values
    image_grid_thws = split_image_grid_thws
    img_num = len(image_grid_thws)

    img_num_per_rank = img_num // cp_size
    img_num_remain = img_num % cp_size
    cp_img_num = []
    for i in range(cp_size):
        cp_img_num.append(img_num_per_rank)
        if i < img_num_remain:
            cp_img_num[i] += 1

    img_idx = 0
    new_pixel_values = []
    new_image_grid_thws = []
    images_padded = []
    for i in range(cp_size):
        seq_len = 0
        img_begin_idx = img_idx
        img_end_idx = img_begin_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        for j in range(img_begin_idx, img_end_idx):
            seq_len += pixel_values[j].size(0)
            new_pixel_values.append(pixel_values[j])
            new_image_grid_thws.append(image_grid_thws[j])

        image_padded = 0 != seq_len % hw_factor
        if image_padded:
            padded_seqlen = (seq_len + hw_factor - 1) // hw_factor * hw_factor - seq_len
            assert padded_seqlen > 0 and padded_seqlen % 4 == 0
            new_pixel_values.append(
                torch.zeros(
                    [padded_seqlen, pixel_values[0].size(-1)],
                    dtype=pixel_values[0].dtype,
                    device=pixel_values[0].device,
                )
            )
            new_image_grid_thws.append(
                torch.tensor(
                    [[1, 2, padded_seqlen // 2]],
                    dtype=image_grid_thws[0].dtype,
                    device=image_grid_thws[0].device,
                )
            )
            cp_img_num[i] += 1
        images_padded.append(int(image_padded))

    return new_pixel_values, new_image_grid_thws, cp_img_num, images_padded


@torch.no_grad
def qwen3vl_cp_split(cp_size: int, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
    """
    Split the pixel values and image grid thws for Qwen3VL vision model.
    """
    assert cp_size > 1
    if pixel_values is None:
        assert image_grid_thw is None
        return None, None, None, None

    assert not pixel_values.requires_grad
    assert not image_grid_thw.requires_grad
    # expand video thw
    image_grid_thw = expand_thw(image_grid_thw)

    hw_factor = 4
    new_pixel_values, new_image_grid_thws, cp_img_num, images_padded = qwen2vl_pad_and_split(
        cp_size, hw_factor, [pixel_values], [image_grid_thw]
    )
    for image_padded in images_padded:
        assert not image_padded, "qwen3vl vit not support sp now, no need to paded"

    pixel_values = torch.cat(new_pixel_values, dim=0)
    image_grid_thw = torch.cat(new_image_grid_thws, dim=0)
    return pixel_values, image_grid_thw, cp_img_num, images_padded


def get_vision_cp_data(
    vision_data: torch.Tensor,
    vision_grid_thw: torch.Tensor,
    square_merge_size: int,
    cp_img_num: list[int],
    images_padded: list[bool],
    cp_rank: int,
    cp_size: int,
):
    """Get vision data and grid_thw for context parallelism.
    Returns:
        vision_data (torch.Tensor): Vision data of shape [total_thw_size, n_features].
        vision_grid_thw (torch.Tensor): Vision grid_thw of shape [total_thw_size, 3].
        seqlens_list (list of torch.Tensor): List of seqlens of the vision data in each context parallel rank,
                                             for the all gather after vision encoder.
    """
    # we use the context parallelism size and context parallel group of LLM for vision model.
    # we only divide the number of images in each context parallel rank.
    assert cp_size == len(cp_img_num)

    seqlens = torch.repeat_interleave(
        vision_grid_thw[:, 1] * vision_grid_thw[:, 2], vision_grid_thw[:, 0]
    )
    vision_grid_thw_list = []
    vision_data_list = []
    seqlens_list = []
    img_idx = 0
    for i in range(cp_size):
        start_idx = img_idx
        end_idx = start_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        vision_grid_thw_list.append(vision_grid_thw[start_idx:end_idx])
        if images_padded[i]:
            seqlens_list.append(seqlens[start_idx : end_idx - 1])
        else:
            seqlens_list.append(seqlens[start_idx:end_idx])
        data_start_idx = seqlens[:start_idx].sum()
        data_end_idx = seqlens[:end_idx].sum()
        vision_data_list.append(vision_data[data_start_idx:data_end_idx])
    new_vision_grid_thw = vision_grid_thw_list[cp_rank]
    new_vision_data = vision_data_list[cp_rank]
    new_seqlens_list = [t // square_merge_size for t in seqlens_list]
    return new_vision_data, new_vision_grid_thw, new_seqlens_list


def get_dist_train_vision_dp_data(
    vision_data: torch.Tensor, vision_grid_thw: torch.Tensor, *, num_chunks: int, dp_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard vision batch on dim 0 by ``num_chunks``; return this DP rank's slice."""
    chunk_idx = dp_rank % num_chunks
    vision_data_chunks = torch.chunk(vision_data, chunks=num_chunks, dim=0)
    vision_data_out = vision_data_chunks[chunk_idx]
    vision_grid_thw_chunks = torch.chunk(vision_grid_thw, chunks=num_chunks, dim=0)
    vision_grid_thw_out = vision_grid_thw_chunks[chunk_idx]
    return vision_data_out, vision_grid_thw_out


def pack_dist_train_vision_module_output(
    vision_embeds: torch.Tensor, deepstack_feature_lists: Sequence[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Concat deepstack features and final vision embeddings; shape as 3D for bridge communicator."""
    vision_module_output_tensor = torch.cat([vision_embeds, *deepstack_feature_lists], dim=0)
    # 2D [batch*seq, hidden] -> 3D [1, batch*seq, hidden]
    vision_module_output_tensor = vision_module_output_tensor.unsqueeze(0)
    return {"vision_module": vision_module_output_tensor}


class AllGatherVisionEmbeddings(torch.autograd.Function):
    """
    AllGatherVisionEmbeddings for Qwen3VL vision model.
    """

    @staticmethod
    def forward(ctx, input, seqlens_on_cp_ranks, cp_group: torch.distributed.ProcessGroup):
        """
        Forward pass for AllGatherVisionEmbeddings.
        """
        outputs = []
        for i in range(len(seqlens_on_cp_ranks)):
            o = torch.zeros(
                (seqlens_on_cp_ranks[i].sum(), *input.shape[1:]),
                device=input.device,
                dtype=input.dtype,
                layout=input.layout,
            )
            outputs.append(o)
        torch.distributed.all_gather(outputs, input, group=cp_group)
        ctx.cp_rank = torch.distributed.get_rank(group=cp_group)
        ctx.cp_group = cp_group
        ctx.save_for_backward(*seqlens_on_cp_ranks)

        output = torch.cat(outputs, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for AllGatherVisionEmbeddings.
        """
        cp_rank = ctx.cp_rank
        seqlens_on_cp_ranks = ctx.saved_tensors
        start_idx = torch.cat(seqlens_on_cp_ranks[:cp_rank]).sum() if cp_rank != 0 else 0
        end_idx = start_idx + seqlens_on_cp_ranks[cp_rank].sum()
        grad_output = grad_output.contiguous()
        torch.distributed.all_reduce(grad_output, group=ctx.cp_group)
        grad_output = grad_output[start_idx:end_idx]
        return grad_output, None, None


def ensure_requires_grad_for_cp_collective(tensors: Sequence[torch.Tensor]) -> None:
    """Make every tensor entering a CP collective participate in autograd.

    ``AllGatherVisionEmbeddings.backward`` issues a cp_group all_reduce, and autograd
    only invokes it for tensors that require grad. Tensors that arrive without grad
    (0-image placeholders, or the outputs of a fully frozen vision tower) would skip
    the collective on their rank and deadlock the remaining ranks, so force
    ``requires_grad`` here. No-op under ``torch.no_grad`` (no rank records backward).
    """
    if not torch.is_grad_enabled():
        return
    for tensor in tensors:
        if not tensor.requires_grad:
            tensor.requires_grad_(True)
