# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/vision_model.py
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

from collections.abc import Callable
from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import (
    Qwen3VLVisionTransformerBlock,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
    Qwen3VLTransformerConfig,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_layer import (
    Qwen3VLRotaryTransformerLayer,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
)
from megatron.core import InferenceParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer


def _maybe_pad_vision_sequence_for_cuda_graph(
    hidden_states: torch.Tensor, rotary_pos_emb: torch.Tensor, seq_len: int, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad vision token tensors to ``max_seq_len`` for fixed-shape CUDA graphs.

    Args:
        hidden_states: ``[seq_len, hidden_size]``.
        rotary_pos_emb: ``[seq_len, 1, 1, dim]`` (same layout as after ``reshape``/``repeat`` in :meth:`Qwen3VLVisionModel.forward`).
        seq_len: Current sequence length (must match tensor leading size).
        max_seq_len: Target length for CUDA graph capture.

    Returns:
        Tuple of (padded hidden_states, padded rotary_pos_emb, new seq_len).

    Raises:
        ValueError: If ``seq_len`` exceeds ``max_seq_len``.
    """
    if seq_len > max_seq_len:
        raise ValueError(
            f"Vision input sequence length ({seq_len}) exceeds max_vision_cuda_graph_seq_length ({max_seq_len}). "
            f"Increase max_vision_cuda_graph_seq_length in config or disable vision CUDA graphs."
        )
    if seq_len < max_seq_len:
        pad_len = max_seq_len - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len), value=0.0)
        rotary_pos_emb = F.pad(rotary_pos_emb, (0, 0, 0, 0, 0, 0, 0, pad_len), value=0.0)
        seq_len = max_seq_len
    return hidden_states, rotary_pos_emb, seq_len


def _vision_forward_packed_attention_setup(
    use_cuda_graph_padding: bool,
    hidden_states: torch.Tensor,
    original_seq_len: int,
    seq_len: int,
    grid_thw: torch.Tensor,
    build_packed_seq_params: Callable[[torch.Tensor], PackedSeqParams],
) -> tuple[Optional[PackedSeqParams], Optional[torch.Tensor]]:
    """Return ``(packed_seq_params, attention_mask)`` for vision encoder forward.

    When using CUDA graphs, packed sequence metadata (non-tensors) cannot be passed; use full
    attention on a fixed-length padded sequence and optionally an additive mask to ignore padding.

    Args:
        use_cuda_graph_padding: Whether vision CUDA graph padding path is active.
        hidden_states: Vision hidden states after adding the batch dimension, shape ``[S, 1, H]``.
        original_seq_len: Sequence length before padding.
        seq_len: Sequence length after optional padding (equals ``hidden_states`` leading size).
        grid_thw: Grid sizes per image/frame (used only when not using CUDA graph padding).
        build_packed_seq_params: Callback to build :class:`PackedSeqParams` from ``grid_thw``.

    Returns:
        ``packed_seq_params`` (``None`` when using CUDA graph padding) and ``attention_mask``
        (additive mask for padded CUDA graph runs, else ``None``).
    """
    if use_cuda_graph_padding:
        packed_seq_params = None
        if original_seq_len < seq_len:
            attention_mask = torch.ones(
                (1, 1, seq_len, seq_len), dtype=hidden_states.dtype, device=hidden_states.device
            )
            attention_mask[:, :, :, original_seq_len:] = 0
            attention_mask[:, :, original_seq_len:, :] = 0
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        else:
            attention_mask = None
        return packed_seq_params, attention_mask

    packed_seq_params = build_packed_seq_params(grid_thw)
    return packed_seq_params, None


class Qwen3VLVisionTransformerLayer(Qwen3VLRotaryTransformerLayer):
    """Keep vision graph wrappers separate from the language model's layers."""


class Qwen3VLVisionModel(VisionModule):
    """Qwen3 ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        patch_merger_spec (ModuleSpec): Specifies module to use for transformer layers.
    """

    def __init__(
        self,
        transformer_config: Qwen3VLTransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_merger_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        assert post_process and pre_process, "not support pp for deepstack_merger_list"
        super().__init__(config=transformer_config)
        self.spatial_merge_size = transformer_config.spatial_merge_size
        self.patch_size = transformer_config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.pg_collection = pg_collection
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.tp_group = self.pg_collection.tp

        assert (
            transformer_config.context_parallel_size == 1
        ), f"context_parallel_size should be 1 in vision model but got {transformer_config.context_parallel_size}"

        self.patch_embed = Qwen3VLVisionPatchEmbed(transformer_config)
        self.pos_embed = nn.Embedding(
            transformer_config.num_position_embeddings, transformer_config.hidden_size
        )
        self.num_grid_per_side = int(transformer_config.num_position_embeddings**0.5)

        head_dim = transformer_config.hidden_size // transformer_config.num_attention_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process

        # Transformer layers.
        if transformer_layer_spec.module is TransformerLayer:
            transformer_layer_spec = deepcopy(transformer_layer_spec)
            transformer_layer_spec.module = Qwen3VLVisionTransformerLayer
            transformer_layer_spec.params["rotary_dim"] = head_dim
        self.decoder = Qwen3VLVisionTransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
            patch_merger_spec=patch_merger_spec,
            pg_collection=self.pg_collection,
        )

        self.merger = None
        if self.post_process:
            self.merger = Qwen3VLVisionPatchMerger(
                transformer_config,
                patch_merger_spec,
                use_postshuffle_norm=False,
                tp_group=self.tp_group,
            )

        self.input_tensor = None

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        if self.pre_process:  # always True
            self.input_tensor = input_tensor
        else:
            raise NotImplementedError()

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def _get_max_vision_seq_length(self) -> int:
        """Get the maximum sequence length for vision encoder CUDA graphs."""
        if (
            hasattr(self.config, "max_vision_cuda_graph_seq_length")
            and self.config.max_vision_cuda_graph_seq_length
        ):
            return self.config.max_vision_cuda_graph_seq_length
        # Default: calculate from num_position_embeddings
        return self.config.num_position_embeddings // (self.config.spatial_merge_size**2)

    def _uses_vision_cuda_graph(self) -> bool:
        """Check if vision encoder CUDA graphs are enabled."""
        return (
            hasattr(self.config, "cuda_graph_impl")
            and self.config.cuda_graph_impl == "transformer_engine"
            and self.training
        )

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        grid_thw: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen3 Vision Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input image/video data of shape [n_tokens, n_dims]
            grid_thw (torch.Tensor): the size tensor indicates grid size of each image/frame
            packed_seq_params (PackedSeqParams): parameters to build attention mask in the backend

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        assert grid_thw is not None
        assert self.input_tensor is None
        assert inference_params is None

        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, 1, 1, -1).repeat(1, 1, 1, 2)

        # Check if we need to pad for CUDA graphs
        use_cuda_graph_padding = self._uses_vision_cuda_graph()
        original_seq_len = seq_len
        if use_cuda_graph_padding:
            max_seq_len = self._get_max_vision_seq_length()
            hidden_states, rotary_pos_emb, seq_len = _maybe_pad_vision_sequence_for_cuda_graph(
                hidden_states, rotary_pos_emb, seq_len, max_seq_len
            )
        hidden_states = hidden_states[:, None]
        packed_seq_params, attention_mask = _vision_forward_packed_attention_setup(
            use_cuda_graph_padding=use_cuda_graph_padding,
            hidden_states=hidden_states,
            original_seq_len=original_seq_len,
            seq_len=seq_len,
            grid_thw=grid_thw,
            build_packed_seq_params=self.build_packed_seq_params,
        )
        hidden_states, deepstack_feature_lists = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
        # Remove padding if we added it
        if use_cuda_graph_padding and original_seq_len < seq_len:
            # Merge the padded rows (cut to whole merged tokens) and drop the padded merged tokens afterwards, so the
            # merger always sees the same row count. On the real patch count, which changes with every micro batch,
            # Transformer Engine builds a new patch_norm plan for each new count when the cuDNN norm flags
            # (NVTE_NORM_FWD/BWD_USE_CUDNN=1) are set, while every other rank waits. The result is the same:
            # patch_norm is per row and every image has a multiple of spatial_merge_size**2 patches, so the padding
            # starts on a merged token boundary. The padded rows are zeroed first: their merged tokens get zero
            # gradient, but a non-finite padded row would still turn the merger's weight gradients into NaN (0 * NaN).
            merge_unit = self.spatial_merge_size**2
            merger_input = hidden_states[: seq_len - seq_len % merge_unit]
            is_pad = (
                torch.arange(merger_input.shape[0], device=merger_input.device) >= original_seq_len
            )
            hidden_states = self.merger(merger_input.masked_fill(is_pad[:, None, None], 0.0))
            original_merged_seq_len = original_seq_len // merge_unit
            hidden_states = hidden_states[:original_merged_seq_len]
            # Deepstack features went through a merger in the decoder: their length is seq_len // spatial_merge_size^2
            deepstack_feature_lists = [
                feat[:original_merged_seq_len] for feat in deepstack_feature_lists
            ]
        else:
            hidden_states = self.merger(hidden_states)

        # Encodes images into continuous embeddings that can be forwarded to the language model.
        split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        hidden_states = torch.split(hidden_states, split_sizes)
        hidden_states = torch.cat(hidden_states, dim=0)
        return hidden_states, deepstack_feature_lists

    def build_packed_seq_params(self, grid_thw: Optional[torch.Tensor]) -> PackedSeqParams:
        # NOTE: each frame is a sequence (rather than each grid)
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = seqlens.cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).int()

        max_seqlen_q = seqlens.max()
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            qkv_format="thd",
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
        )
