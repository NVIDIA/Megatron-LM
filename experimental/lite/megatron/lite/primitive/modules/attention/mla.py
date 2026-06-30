# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared Multi-Latent Attention primitive."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from megatron.lite.primitive.utils.rope import (
    _apply_rotary_pos_emb_bshd,
    _apply_rotary_pos_emb_thd,
)
from megatron.lite.primitive.utils.rotary import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    _yarn_get_mscale,
)

from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    gather_from_sequence_parallel,
)
from megatron.lite.primitive.parallel.cp import (
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
)
from megatron.lite.primitive.parallel.thd import (
    reconstruct_packed_from_cp_parts,
    split_packed_to_cp_local,
)

_KEPT_PSP_FIELDS = (
    "qkv_format",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
    "max_seqlen_q",
    "max_seqlen_kv",
)


def _apply_mla_rope_bshd(t: torch.Tensor, freqs: torch.Tensor, *, mscale: float) -> torch.Tensor:
    return _apply_rotary_pos_emb_bshd(
        t, freqs, rotary_interleaved=False, mscale=mscale, mla_rotary_interleaved=True
    )


def _apply_mla_rope_thd(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    *,
    mscale: float,
    cp_group,
) -> torch.Tensor:
    return _apply_rotary_pos_emb_thd(
        t,
        cu_seqlens,
        freqs,
        rotary_interleaved=False,
        mscale=mscale,
        cp_group=cp_group,
        mla_rotary_interleaved=True,
    )


class MultiLatentAttention(nn.Module):
    """Native MLA composition using lite parallel linears and TE core attention."""

    _cp_stream: torch.cuda.Stream | None = None

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        ps: ParallelState,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
        rope_scaling: dict | None = None,
        use_thd: bool = False,
    ):
        super().__init__()
        if num_attention_heads % ps.tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size")
        self.ps = ps
        self.num_heads_local = num_attention_heads // ps.tp_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.linear_proj = RowParallelLinear(
            num_attention_heads * v_head_dim,
            hidden_size,
            ps,
            bias=False,
        )
        self.linear_q_down_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.linear_q_up_proj = ColumnParallelLinear(
            q_lora_rank,
            num_attention_heads * self.q_head_dim,
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=rms_norm_eps,
        )
        self.linear_kv_down_proj = nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
        )
        self.linear_kv_up_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=rms_norm_eps,
        )

        rope_scaling = dict(rope_scaling or {})
        rope_type = rope_scaling.get("type", "rope")
        if rope_type == "yarn":
            factor = float(rope_scaling.get("factor", 1.0))
            self.rotary = YarnRotaryEmbedding(
                qk_rope_head_dim,
                rotary_base=rope_theta,
                scaling_factor=factor,
                original_max_position_embeddings=int(
                    rope_scaling.get("original_max_position_embeddings", 4096)
                ),
                beta_fast=float(rope_scaling.get("beta_fast", 32.0)),
                beta_slow=float(rope_scaling.get("beta_slow", 1.0)),
                mscale=float(rope_scaling.get("mscale", 1.0)),
                mscale_all_dim=float(rope_scaling.get("mscale_all_dim", 1.0)),
                cp_group=ps.cp_group if ps.cp_size > 1 else None,
            )
            attn_mscale = _yarn_get_mscale(factor, float(rope_scaling.get("mscale_all_dim", 1.0)))
        elif rope_type == "rope":
            self.rotary = RotaryEmbedding(
                kv_channels=qk_rope_head_dim,
                rotary_base=rope_theta,
                use_cpu_initialization=False,
                cp_group=ps.cp_group if ps.cp_size > 1 else None,
            )
            attn_mscale = 1.0
        else:
            raise ValueError(f"Unsupported MLA rope type: {rope_type!r}")
        self._softmax_scale = attn_mscale * attn_mscale / math.sqrt(self.q_head_dim)
        self._query_scale = 1.0

        cp_kwargs = {}
        if ps.cp_size > 1:
            if MultiLatentAttention._cp_stream is None:
                MultiLatentAttention._cp_stream = torch.cuda.Stream()
            cp_kwargs = dict(
                cp_group=ps.cp_group,
                cp_global_ranks=ps.cp_global_ranks,
                cp_stream=MultiLatentAttention._cp_stream,
            )
        self._use_torch_core = ps.cp_size > 1 and v_head_dim != self.q_head_dim
        self.core_attn = None
        if not self._use_torch_core:
            dpa_kwargs = dict(cp_kwargs, softmax_scale=self._softmax_scale)
            kv_channels = (
                (self.q_head_dim, v_head_dim) if v_head_dim != self.q_head_dim else self.q_head_dim
            )
            self.core_attn = te.DotProductAttention(
                num_attention_heads=self.num_heads_local,
                kv_channels=kv_channels,
                attention_dropout=0.0,
                attn_mask_type="causal",
                qkv_format="thd" if use_thd else "sbhd",
                **dpa_kwargs,
            )

    def forward(self, x: torch.Tensor, packed_seq_params=None) -> torch.Tensor:
        q_compressed = self.linear_q_down_proj(x)
        kv_combined = self.linear_kv_down_proj(x)
        kv_compressed, k_pos_emb = kv_combined.split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.ps.tp_size > 1:
            k_pos_emb = gather_from_sequence_parallel(k_pos_emb, self.ps)

        q_proj = self.linear_q_up_proj(q_compressed)
        q = q_proj.view(
            *q_proj.shape[:-1],
            self.num_heads_local,
            self.q_head_dim,
        )
        kv_proj = self.linear_kv_up_proj(kv_compressed)
        kv = kv_proj.view(
            *kv_proj.shape[:-1],
            self.num_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        q_nope, q_pos = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pos = k_pos_emb.unsqueeze(-2)

        is_thd = packed_seq_params is not None
        if is_thd:
            q_nope = q_nope.squeeze(1)
            q_pos = q_pos.squeeze(1)
            k_nope = k_nope.squeeze(1)
            value = value.squeeze(1)
            k_pos = k_pos.squeeze(1)

        q_pos, k_pos = self._apply_rope(q_pos, k_pos, packed_seq_params)
        if k_pos.dim() == q_nope.dim():
            k_pos = k_pos.expand(*q_nope.shape[:-1], self.qk_rope_head_dim)
        else:
            k_pos = k_pos.expand(-1, -1, self.num_heads_local, -1)
        query = torch.cat([q_nope, q_pos], dim=-1).contiguous()
        key = torch.cat([k_nope, k_pos], dim=-1).contiguous()
        value = value.contiguous()
        if self._query_scale != 1.0:
            query = query * self._query_scale

        if is_thd:
            if self._use_torch_core:
                out = self._torch_core_attention_thd(
                    query,
                    key,
                    value,
                    packed_seq_params=packed_seq_params,
                ).reshape(query.size(0), 1, -1)
            else:
                psp_kwargs = {
                    k: getattr(packed_seq_params, k)
                    for k in _KEPT_PSP_FIELDS
                    if getattr(packed_seq_params, k, None) is not None
                }
                assert self.core_attn is not None
                out = self.core_attn(
                    query,
                    key,
                    value,
                    core_attention_bias_type="no_bias",
                    attn_mask_type="padding_causal",
                    **psp_kwargs,
                ).reshape(query.size(0), 1, -1)
        else:
            if self._use_torch_core:
                out = self._torch_core_attention(query, key, value)
            else:
                assert self.core_attn is not None
                out = self.core_attn(query, key, value, core_attention_bias_type="no_bias")
            if out.dim() > x.dim():
                out = out.reshape(*out.shape[:-2], self.num_heads_local * self.v_head_dim)
        return self.linear_proj(out)

    def _torch_core_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        local_seq = query.size(0)
        if self.ps.cp_size > 1:
            from torch.distributed.nn.functional import all_gather

            query_parts = all_gather(query.contiguous(), group=self.ps.cp_group)
            key_parts = all_gather(key.contiguous(), group=self.ps.cp_group)
            value_parts = all_gather(value.contiguous(), group=self.ps.cp_group)
            query = zigzag_reconstruct_from_cp_parts(query_parts, seq_dim=0)
            key = zigzag_reconstruct_from_cp_parts(key_parts, seq_dim=0)
            value = zigzag_reconstruct_from_cp_parts(value_parts, seq_dim=0)

        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        scale = self._softmax_scale if self._query_scale == 1.0 else None
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )
        out = out.permute(2, 0, 1, 3).contiguous()
        if self.ps.cp_size > 1:
            out = zigzag_slice_for_cp(out, self.ps.cp_rank, self.ps.cp_size, seq_dim=0)
            if out.size(0) != local_seq:
                raise RuntimeError("CP MLA output shard has unexpected sequence length.")
        return out

    def _torch_core_attention_thd(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        packed_seq_params,
    ) -> torch.Tensor:
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
        if cu_seqlens is None:
            cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
        if cu_seqlens is None:
            raise ValueError("Packed THD MLA fallback requires cu_seqlens.")

        local_tokens = query.size(0)
        if self.ps.cp_size > 1:
            from torch.distributed.nn.functional import all_gather

            query = reconstruct_packed_from_cp_parts(
                list(all_gather(query.contiguous(), group=self.ps.cp_group)),
                cu_seqlens_padded=cu_seqlens,
                cp_size=self.ps.cp_size,
                dim=0,
            )
            key = reconstruct_packed_from_cp_parts(
                list(all_gather(key.contiguous(), group=self.ps.cp_group)),
                cu_seqlens_padded=cu_seqlens,
                cp_size=self.ps.cp_size,
                dim=0,
            )
            value = reconstruct_packed_from_cp_parts(
                list(all_gather(value.contiguous(), group=self.ps.cp_group)),
                cu_seqlens_padded=cu_seqlens,
                cp_size=self.ps.cp_size,
                dim=0,
            )

        outputs = []
        scale = self._softmax_scale if self._query_scale == 1.0 else None
        for idx in range(int(cu_seqlens.numel()) - 1):
            start = int(cu_seqlens[idx].item())
            end = int(cu_seqlens[idx + 1].item())
            if end <= start:
                continue
            q = query[start:end].permute(1, 0, 2).unsqueeze(0)
            k = key[start:end].permute(1, 0, 2).unsqueeze(0)
            v = value[start:end].permute(1, 0, 2).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )
            outputs.append(out.squeeze(0).permute(1, 0, 2).contiguous())
        full_out = torch.cat(outputs, dim=0) if outputs else value.new_empty(value.shape)
        if self.ps.cp_size <= 1:
            return full_out
        local_out = split_packed_to_cp_local(
            full_out,
            cu_seqlens_padded=cu_seqlens,
            cp_size=self.ps.cp_size,
            cp_rank=self.ps.cp_rank,
            dim=0,
        )
        if local_out.size(0) != local_tokens:
            raise RuntimeError("CP THD MLA output shard has unexpected token count.")
        return local_out

    def _apply_rope(self, q_pos: torch.Tensor, k_pos: torch.Tensor, packed_seq_params):
        is_thd = packed_seq_params is not None
        if is_thd:
            max_q = getattr(packed_seq_params, "max_seqlen_q", None)
            max_kv = getattr(packed_seq_params, "max_seqlen_kv", None)
            seq_len = (
                int(max(max_q, max_kv))
                if max_q is not None and max_kv is not None
                else int(packed_seq_params.cu_seqlens_q[-1])
            )
            freqs = self.rotary(seq_len, packed_seq=True)
            if isinstance(freqs, tuple):
                freqs, mscale = freqs
            else:
                mscale = 1.0
            q_pos = _apply_mla_rope_thd(
                q_pos,
                packed_seq_params.cu_seqlens_q,
                freqs,
                mscale=mscale,
                cp_group=self.ps.cp_group,
            )
            k_pos = _apply_mla_rope_thd(
                k_pos,
                packed_seq_params.cu_seqlens_kv,
                freqs,
                mscale=mscale,
                cp_group=self.ps.cp_group,
            )
            return q_pos, k_pos

        seq_len = q_pos.size(0) * self.ps.cp_size
        freqs = self.rotary(seq_len)
        if isinstance(freqs, tuple):
            freqs, mscale = freqs
        else:
            mscale = 1.0
        return (
            _apply_mla_rope_bshd(q_pos, freqs, mscale=mscale),
            _apply_mla_rope_bshd(k_pos, freqs, mscale=mscale),
        )


__all__ = ["MultiLatentAttention"]
