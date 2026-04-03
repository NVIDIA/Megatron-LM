# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dual Chunk Attention (DCA) for efficient long-context modeling.

Reference:
    "Training-Free Long-Context Scaling of Large Language Models"
    An et al., 2024. https://arxiv.org/abs/2402.17463

    "Qwen2 Technical Report"
    Yang et al., 2024. https://arxiv.org/abs/2407.10671
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from flash_attn import flash_attn_func

    HAVE_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAVE_FLASH_ATTN = False


@dataclass
class DCASubmodules:
    """Configuration class for specifying the submodules of DualChunkAttention.

    Currently no submodules are required. This dataclass is provided for
    consistency with other attention variants (e.g., DSAttentionSubmodules)
    and for future extensibility.
    """

    pass


def _merge_chunk_attention_outputs(outputs: list[Tensor], logsumexps: list[Tensor]) -> Tensor:
    """Merge attention outputs from multiple chunk computations using log-sum-exp renormalization.

    When attention is computed separately over disjoint key sets (intra-chunk, successive-chunk,
    inter-chunk), the softmax normalization constants differ. This function correctly rescales
    and merges the partial outputs using the log-sum-exp trick for numerical stability.

    Args:
        outputs: List of attention output tensors, each [batch, heads, q_len, head_dim].
        logsumexps: List of corresponding log-sum-exp tensors, each [batch, heads, q_len, 1].

    Returns:
        Merged output tensor [batch, heads, q_len, head_dim].
    """
    if len(outputs) == 1:
        return outputs[0]

    stacked_out = torch.stack(outputs, dim=0)
    stacked_lse = torch.stack(logsumexps, dim=0)

    max_lse = stacked_lse.max(dim=0).values
    exp_diff = torch.exp(stacked_lse - max_lse).detach()
    weights = exp_diff / exp_diff.sum(dim=0)

    merged = (stacked_out * weights).sum(dim=0)
    return merged


def _get_yarn_mscale(config: TransformerConfig) -> float:
    """Get the YARN concentration factor (mscale) from config.

    Uses the same logic as attention.py to ensure consistency between DCA's
    RoPE application and the standard attention path.
    """
    from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
        _yarn_get_concentration_factor_from_config,
    )

    return _yarn_get_concentration_factor_from_config(config)


class DualChunkAttention(MegatronModule):
    """Dual Chunk Attention for efficient long-context modeling.

    Segments long sequences into chunks and applies three types of attention:
        1. Intra-chunk: standard causal attention within each chunk.
        2. Successive-chunk: non-causal attention to the immediately preceding chunk,
           preserving locality.
        3. Inter-chunk: non-causal attention to all earlier chunks (beyond the
           immediately preceding one), with a fixed relative position encoding.

    For sequences shorter than chunk_len, this falls back to standard attention
    and produces numerically identical results.

    When FlashAttention is available and inputs are on CUDA, uses FlashAttention
    for memory-efficient chunk attention with native GQA support. Otherwise falls
    back to explicit matrix operations.

    Reference:
        An et al., "Training-Free Long-Context Scaling of Large Language Models", 2024.
        https://arxiv.org/abs/2402.17463

    Args:
        config (TransformerConfig): Transformer configuration.
        submodules (DCASubmodules): Submodule specifications (reserved for future use).
        layer_number (int): Layer index (1-indexed).
        attn_mask_type (AttnMaskType): Attention mask type.
        attention_type (str): "self" or "cross".
        softmax_scale (float, optional): Softmax scaling factor.
        k_channels (int, optional): Key channels (defaults to config.kv_channels).
        v_channels (int, optional): Value channels (defaults to config.kv_channels).
        cp_comm_type (str, optional): Context parallelism communication type.
        pg_collection (ProcessGroupCollection, optional): Process group collection.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DCASubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number
        self.chunk_size = config.dca_chunk_size
        self.local_size = config.dca_local_size
        self.chunk_len = self.chunk_size - self.local_size

        k_ch = k_channels if k_channels is not None else config.kv_channels
        if softmax_scale is None:
            self.softmax_scale = k_ch**-0.5
        else:
            self.softmax_scale = softmax_scale

        self.mscale = _get_yarn_mscale(config)

    def _apply_rope(self, t: Tensor, freqs: Tensor) -> Tensor:
        """Apply rotary position embedding with YARN mscale."""
        return apply_rotary_pos_emb(t, freqs, config=self.config, mscale=self.mscale)

    def _compute_dca_freqs(
        self, rotary_pos_emb: Tuple[Tensor, Tensor], seq_len: int, device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute DCA-modified rotary position embedding frequencies.

        DCA uses three different query position encodings and one key encoding:
        - Keys: positions wrapped modulo chunk_len (reset per chunk)
        - Intra-chunk queries: same wrapped positions as keys
        - Successive-chunk queries: positions shifted by chunk_len (preserves locality)
        - Inter-chunk queries: fixed position at max chunk distance

        Args:
            rotary_pos_emb: Tuple (q_freqs, k_freqs), each [seq_len_emb, 1, 1, dim].
            seq_len: Current sequence length.
            device: Target device.

        Returns:
            Tuple of (key_freqs, q_intra_freqs, q_succ_freqs, q_inter_freqs),
            each of shape [seq_len, 1, 1, dim].
        """
        q_pos_emb, _ = rotary_pos_emb

        positions = torch.arange(seq_len, device=device)
        local_positions = positions % self.chunk_len

        key_freqs = q_pos_emb[local_positions]
        q_intra_freqs = q_pos_emb[local_positions]

        succ_positions = (local_positions + self.chunk_len).clamp(max=self.chunk_size)
        q_succ_freqs = q_pos_emb[succ_positions]

        inter_pos = min(2 * self.chunk_len - 1, self.chunk_size)
        q_inter_freqs = q_pos_emb[inter_pos : inter_pos + 1].expand(seq_len, -1, -1, -1)

        return key_freqs, q_intra_freqs, q_succ_freqs, q_inter_freqs

    def _flash_attention_with_lse(
        self, query: Tensor, key: Tensor, value: Tensor, causal: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention using FlashAttention with LSE output.

        FlashAttention handles GQA natively, so key/value should NOT be expanded
        before calling this method.

        Args:
            query: [q_len, batch, num_heads, head_dim]
            key: [kv_len, batch, num_kv_heads, head_dim]
            value: [kv_len, batch, num_kv_heads, head_dim]
            causal: Whether to apply causal masking.

        Returns:
            output: [batch, heads, q_len, head_dim]
            lse: [batch, heads, q_len, 1] log-sum-exp values.
        """
        q = query.permute(1, 0, 2, 3)
        k = key.permute(1, 0, 2, 3)
        v = value.permute(1, 0, 2, 3)

        output, softmax_lse, _ = flash_attn_func(
            q, k, v, softmax_scale=self.softmax_scale, causal=causal, return_attn_probs=True
        )

        output = output.permute(0, 2, 1, 3)
        softmax_lse = softmax_lse.unsqueeze(-1)

        return output, softmax_lse

    def _unfused_attention_with_lse(
        self, query: Tensor, key: Tensor, value: Tensor, causal: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention using explicit matrix operations with LSE output.

        This is the reference implementation for correctness verification and
        for environments without FlashAttention (e.g., CPU testing).

        Key and value must already be expanded for GQA before calling this method.

        Args:
            query: [q_len, batch, num_heads, head_dim]
            key: [kv_len, batch, num_heads, head_dim] (already GQA-expanded)
            value: [kv_len, batch, num_heads, head_dim] (already GQA-expanded)
            causal: Whether to apply causal masking.

        Returns:
            output: [batch, heads, q_len, head_dim]
            lse: [batch, heads, q_len, 1] log-sum-exp values.
        """
        q_len = query.size(0)
        kv_len = key.size(0)

        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        if causal and q_len > 1:
            kv_offset = kv_len - q_len
            causal_mask = torch.triu(
                torch.full(
                    (q_len, kv_len), float('-inf'), device=scores.device, dtype=scores.dtype
                ),
                diagonal=1 + kv_offset,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        lse = torch.logsumexp(scores, dim=-1, keepdim=True)

        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(attn_weights, v)

        return output, lse

    def _use_flash_attn(self, query: Tensor) -> bool:
        """Determine whether to use FlashAttention based on availability and device."""
        return HAVE_FLASH_ATTN and query.is_cuda

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        rotary_pos_emb: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Forward pass for Dual Chunk Attention.

        Expects query, key, value BEFORE rotary position embedding has been applied.
        DCA applies its own modified RoPE internally.

        Args:
            query: [seq_len, batch, num_heads, head_dim] (pre-RoPE).
            key: [seq_len, batch, num_kv_heads, head_dim] (pre-RoPE).
            value: [seq_len, batch, num_kv_heads, head_dim].
            attention_mask: Optional attention mask (unused in current implementation).
            attn_mask_type: Attention mask type.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters (not yet supported).
            rotary_pos_emb: Tuple of (q_freqs, k_freqs), each [emb_len, 1, 1, dim].

        Returns:
            output: [seq_len, batch, num_heads * head_dim].
        """
        assert packed_seq_params is None, "Packed sequences are not yet supported for DCA"

        seq_len, batch_size, num_heads, head_dim = query.shape
        num_kv_heads = key.shape[2]
        use_fa = self._use_flash_attn(query)

        if seq_len <= self.chunk_len:
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                query = self._apply_rope(query, q_pos_emb)
                key = self._apply_rope(key, k_pos_emb)
            return self._standard_attention_forward(
                query, key, value, num_heads, num_kv_heads, head_dim, seq_len, batch_size, use_fa
            )

        assert (
            rotary_pos_emb is not None
        ), "rotary_pos_emb is required for DCA with seq_len > chunk_len"

        key_freqs, q_intra_freqs, q_succ_freqs, q_inter_freqs = self._compute_dca_freqs(
            rotary_pos_emb, seq_len, query.device
        )

        key_rope = self._apply_rope(key, key_freqs)
        query_intra = self._apply_rope(query, q_intra_freqs)
        query_succ = self._apply_rope(query, q_succ_freqs)
        query_inter = self._apply_rope(query, q_inter_freqs)

        if not use_fa and num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            key_rope = key_rope.repeat_interleave(repeat_factor, dim=2)
            value = value.repeat_interleave(repeat_factor, dim=2)

        attn_fn = self._flash_attention_with_lse if use_fa else self._unfused_attention_with_lse

        num_chunks = (seq_len + self.chunk_len - 1) // self.chunk_len
        output_chunks = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_len
            chunk_end = min(chunk_start + self.chunk_len, seq_len)

            chunk_outputs = []
            chunk_lses = []

            q_intra = query_intra[chunk_start:chunk_end]
            k_intra = key_rope[chunk_start:chunk_end]
            v_intra = value[chunk_start:chunk_end]
            out_intra, lse_intra = attn_fn(q_intra, k_intra, v_intra, causal=True)
            chunk_outputs.append(out_intra)
            chunk_lses.append(lse_intra)

            if chunk_idx >= 1:
                prev_start = (chunk_idx - 1) * self.chunk_len
                prev_end = chunk_start

                q_succ_chunk = query_succ[chunk_start:chunk_end]
                k_succ = key_rope[prev_start:prev_end]
                v_succ = value[prev_start:prev_end]
                out_succ, lse_succ = attn_fn(q_succ_chunk, k_succ, v_succ, causal=False)
                chunk_outputs.append(out_succ)
                chunk_lses.append(lse_succ)

            if chunk_idx >= 2:
                inter_end = (chunk_idx - 1) * self.chunk_len

                q_inter_chunk = query_inter[chunk_start:chunk_end]
                k_inter = key_rope[:inter_end]
                v_inter = value[:inter_end]
                out_inter, lse_inter = attn_fn(q_inter_chunk, k_inter, v_inter, causal=False)
                chunk_outputs.append(out_inter)
                chunk_lses.append(lse_inter)

            merged = _merge_chunk_attention_outputs(chunk_outputs, chunk_lses)
            merged = merged.permute(2, 0, 1, 3)
            output_chunks.append(merged)

        output = torch.cat(output_chunks, dim=0)
        output = output.reshape(seq_len, batch_size, num_heads * head_dim)

        return output

    def _standard_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        use_flash_attn: bool = False,
    ) -> Tensor:
        """Standard causal attention for short sequences (seq_len <= chunk_len).

        When the sequence fits within a single chunk, DCA is equivalent to
        standard attention. This avoids unnecessary overhead.
        """
        if use_flash_attn:
            q = query.permute(1, 0, 2, 3)
            k = key.permute(1, 0, 2, 3)
            v = value.permute(1, 0, 2, 3)
            output = flash_attn_func(q, k, v, softmax_scale=self.softmax_scale, causal=True)
            output = output.permute(1, 0, 2, 3).contiguous()
            return output.reshape(seq_len, batch_size, num_heads * head_dim)

        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=2)
            value = value.repeat_interleave(repeat_factor, dim=2)

        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), float('-inf'), device=scores.device, dtype=scores.dtype
                ),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(attn_weights, v)

        output = output.permute(2, 0, 1, 3).contiguous()
        output = output.reshape(seq_len, batch_size, num_heads * head_dim)
        return output
