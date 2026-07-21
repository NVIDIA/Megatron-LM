# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

__all__ = ['BagelRotaryEmbedding', 'apply_qwen2_rotary_pos_emb']


def apply_qwen2_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE with native Qwen2's exact batch-1 tensor layout and op order."""

    if query.ndim != 4 or key.ndim != 4 or query.shape[1] != 1 or key.shape[1] != 1:
        raise ValueError(
            "BAGEL native Qwen RoPE alignment expects [seq, 1, heads, head_dim] "
            f"query/key tensors, got {tuple(query.shape)} and {tuple(key.shape)}"
        )
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError(
            "BAGEL native Qwen RoPE alignment expects [seq, head_dim] cos/sin, "
            f"got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )

    # Native PackedAttentionMoT removes the singleton batch dimension before
    # applying Hugging Face Qwen2 RoPE.  Preserve that TensorIterator layout,
    # not just the algebra, because BF16 kernel boundaries are alignment-critical.
    query_3d = query.squeeze(1)
    key_3d = key.squeeze(1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    query_embed = (query_3d * cos) + (rotate_half(query_3d) * sin)
    key_embed = (key_3d * cos) + (rotate_half(key_3d) * sin)
    return query_embed.unsqueeze(1), key_embed.unsqueeze(1)


class BagelRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
        rope_scaling (bool, optional): Apply rope scaling as used in llama 3.x.
        rope_scaling_factor (float, optional): rope scaling factor in llama 3.x. Defaults to 8.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly
            on the GPU. Defaults to False
        cp_group (torch.distributed.ProcessGroup, optional): Process group for context parallel.
            Defaults to None.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        use_cpu_initialization: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        native_qwen_for_alignment: bool = False,
    ) -> None:
        self.native_qwen_for_alignment = native_qwen_for_alignment
        super().__init__(
            kv_channels,
            rotary_percent,
            rotary_interleaved,
            seq_len_interpolation_factor,
            rotary_base,
            rope_scaling,
            rope_scaling_factor,
            use_cpu_initialization or native_qwen_for_alignment,
            cp_group,
        )

    @torch.no_grad()
    def forward_qwen_with_position_ids(
        self, x: Tensor, position_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Create BF16 cos/sin with native HF Qwen2's exact operation sequence."""

        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.ndim != 2:
            raise ValueError(
                "BAGEL native Qwen RoPE expects [batch, seq] position_ids, "
                f"got {tuple(position_ids.shape)}"
            )
        if self.inv_freq.device != position_ids.device or self.inv_freq.dtype != x.dtype:
            # Native Qwen constructs this non-persistent buffer on CPU, then its
            # model-wide ``to(bfloat16)`` converts the buffer before the first
            # forward.  The subsequent explicit ``.float()`` matmul therefore
            # uses FP32 values expanded from BF16-rounded frequencies.
            self.inv_freq = self.inv_freq.to(device=position_ids.device, dtype=x.dtype)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Qwen's default attention_scaling is 1.0, but retain the explicit
        # multiply because the reference executes these two operations.
        cos = cos * 1.0
        sin = sin * 1.0
        return cos.to(dtype=x.dtype).squeeze(0), sin.to(dtype=x.dtype).squeeze(0)

    def get_freqs_from_position_ids(self, position_ids: Tensor) -> Tensor:
        """Generates matrix of frequencies based on actual position_ids.

        Args:
            position_ids: Tensor of shape [seq_len] containing actual positions

        Returns:
            freqs: Tensor of shape [seq_len, dim/2]
        """
        if self.inv_freq.device.type == 'cpu':
            self.inv_freq = self.inv_freq.to(device=position_ids.device)

        # position_ids: [seq_len]
        # inv_freq: [dim/2]
        # freqs = position_ids @ inv_freq -> [seq_len, dim/2]
        position_ids = position_ids.to(dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            position_ids = position_ids * (1 / self.seq_len_interpolation_factor)

        freqs = torch.outer(position_ids, self.inv_freq)  # [seq_len, dim/2]
        return freqs

    def forward_with_position_ids(self, position_ids: Tensor) -> Tensor:
        """Forward pass with actual position_ids.

        Args:
            position_ids: Tensor of shape [seq_len] containing actual positions
            packed_seq: Whether using packed sequence

        Returns:
            Tensor: Rotary embeddings of shape [seq_len, 1, 1, dim]
        """
        if self.inv_freq.device.type == 'cpu':
            self.inv_freq = self.inv_freq.to(device=position_ids.device)

        freqs = self.get_freqs_from_position_ids(position_ids)

        # first part even vector components, second part odd vector components
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )

        # emb [seq_length, dim] -> [seq_length, 1, 1, dim]
        emb = emb[:, None, None, :]

        return emb

    def get_cos_sin_from_position_ids(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Get cos and sin based on actual position_ids (qwen2 style).

        Args:
            position_ids: Tensor of shape [seq_len] containing actual positions

        Returns:
            cos: Tensor of shape [seq_len, dim]
            sin: Tensor of shape [seq_len, dim]
        """
        if self.inv_freq.device.type == 'cpu':
            self.inv_freq = self.inv_freq.to(device=position_ids.device)

        freqs = self.get_freqs_from_position_ids(position_ids)

        # Duplicate freqs like qwen2: emb = cat((freqs, freqs), dim=-1)
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin
