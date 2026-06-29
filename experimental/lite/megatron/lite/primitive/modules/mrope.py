# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multimodal rotary embedding primitive."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.utils.rope import get_pos_emb_on_this_cp_rank

__all__ = ["MultimodalRotaryEmbedding"]


class MultimodalRotaryEmbedding(nn.Module):
    """Qwen-style multimodal RoPE embedding with optional CP slicing."""

    def __init__(
        self,
        *,
        kv_channels: int,
        rotary_percent: float,
        rotary_base: float,
        cp_group: dist.ProcessGroup | None,
    ):
        super().__init__()
        dim = int(kv_channels * rotary_percent)
        inv_freq = 1.0 / (rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cp_group = cp_group

    @staticmethod
    def _apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]
        return freqs_t

    def forward(
        self, position_ids: torch.Tensor, mrope_section: list[int], *, packed_seq: bool = False
    ) -> torch.Tensor:
        seq = position_ids.to(device=self.inv_freq.device)
        inv_freq = self.inv_freq.float()
        inv = inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        seq_expanded = seq[:, :, None, :].float()
        freqs = (inv @ seq_expanded).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs, mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if not packed_seq and self.cp_group is not None and dist.get_world_size(self.cp_group) > 1:
            emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)
        return emb
