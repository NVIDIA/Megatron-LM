"""Grouped Query Attention + Rotary Embedding (MC-atoms).

Model-agnostic: takes explicit params instead of model-specific config.
Supports sequence parallel, context parallel, and THD (packed sequences).

RoPE internals call Megatron-Core's atomic
`megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding`
and `rope_utils._apply_rotary_pos_emb_bshd / _apply_rotary_pos_emb_thd` — this
matches the mbridge Qwen3MoE reference path (both run the unfused rotate-half
kernel because ``config.apply_rope_fusion`` defaults to ``False``). See
`docs/gqa_mc_atoms_plan.md` §4 (Option A).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from megatron.core.models.common.embeddings.rope_utils import (  # pyright: ignore[reportMissingImports]
    _apply_rotary_pos_emb_bshd,
    _apply_rotary_pos_emb_thd,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import (  # pyright: ignore[reportMissingImports]
    RotaryEmbedding as MCoreRotaryEmbedding,
)

from megatron.lite.primitive.parallel import ColumnParallelLinear, ParallelState, RowParallelLinear
from megatron.lite.primitive.utils import ensure_divisible

# Whitelist of MC PackedSeqParams fields accepted by TE DotProductAttention.forward().
# MC-only fields (local_cp_size, cp_group, total_tokens, seq_idx) are excluded.
# Mirror MC TEDotProductAttention.kept_packed_seq_params pattern
# (Megatron-LM/megatron/core/extensions/transformer_engine.py:1501-1593).
_KEPT_PSP_FIELDS = (
    "qkv_format",
    "cu_seqlens_q", "cu_seqlens_kv",
    "cu_seqlens_q_padded", "cu_seqlens_kv_padded",
    "max_seqlen_q", "max_seqlen_kv",
)


class GQAttention(nn.Module):
    """Grouped Query Attention with TE DotProductAttention.

    Model-agnostic: all architecture params passed explicitly.
    """

    _cp_stream: torch.cuda.Stream | None = None

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        ps: ParallelState,
        *,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        rotary_percent: float = 1.0,
        use_thd: bool = False,
        output_gate: bool = False,
        use_fp32_rope: bool = False,
        zero_centered_gamma: bool = False,
        qkv_layout: str = "flat",
    ):
        super().__init__()
        self.num_heads_local = ensure_divisible(num_attention_heads, ps.tp_size)
        self.num_kv_heads_local = ensure_divisible(num_key_value_heads, ps.tp_size)
        self.head_dim = head_dim
        self.ps = ps
        self._output_gate = output_gate
        self._use_fp32_rope = use_fp32_rope
        if qkv_layout not in {"flat", "mcore"}:
            raise ValueError(f"Unsupported qkv_layout={qkv_layout!r}")
        self._qkv_layout = qkv_layout

        # Declaration order follows MC's `SelfAttention` submodule order
        # (linear_proj → linear_qkv → q_layernorm → k_layernorm). `named_
        # parameters()` iterates in registration order, and MC's
        # `DistributedDataParallel` partitions gradient buckets by that order.
        # Mismatched order would put bucket boundaries in different places
        # between lite and MC reference paths, producing different per-rank fp32
        # master shard layouts and non-bitwise step-1 divergence.
        self.proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size, ps, bias=False,
        )
        q_cols = num_attention_heads * (2 if output_gate else 1)
        qkv_size = (q_cols + 2 * num_key_value_heads) * head_dim
        self.qkv = ColumnParallelLinear(
            hidden_size, qkv_size, ps,
            bias=False, normalization="RMSNorm", eps=rms_norm_eps,
            zero_centered_gamma=zero_centered_gamma,
        )
        self.q_norm = te.RMSNorm(head_dim, eps=rms_norm_eps, zero_centered_gamma=zero_centered_gamma)
        self.k_norm = te.RMSNorm(head_dim, eps=rms_norm_eps, zero_centered_gamma=zero_centered_gamma)

        # MC's RotaryEmbedding is atomic (flat kwargs, no TransformerConfig).
        # cp_group is read from self.cp_group inside forward() when not passed
        # — no manual CP shard needed on our side.
        self.rotary = MCoreRotaryEmbedding(
            kv_channels=head_dim,
            rotary_percent=rotary_percent,
            rotary_interleaved=False,
            rotary_base=int(rope_theta),
            use_cpu_initialization=False,
            cp_group=ps.cp_group if ps.cp_size > 1 else None,
        )

        cp_kwargs = {}
        if ps.cp_size > 1:
            if GQAttention._cp_stream is None:
                GQAttention._cp_stream = torch.cuda.Stream()
            cp_kwargs = dict(
                cp_group=ps.cp_group,
                cp_global_ranks=ps.cp_global_ranks,
                cp_stream=GQAttention._cp_stream,
            )

        self.core_attn = te.DotProductAttention(
            num_attention_heads=self.num_heads_local,
            kv_channels=head_dim,
            num_gqa_groups=self.num_kv_heads_local,
            attention_dropout=0.0,
            attn_mask_type="causal",
            qkv_format="thd" if use_thd else "sbhd",
            **cp_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        q, gate, k, v = self._split_qkv(self.qkv(x))

        is_thd = packed_seq_params is not None
        if is_thd:
            q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE — unfused bshd/thd to match MC's default apply_rope_fusion=False.
        # MC's RotaryEmbedding.forward takes only `max_seq_len` + optional
        # `offset`; position_ids is NOT consumed here (MC handles position via
        # offset for inference / mRoPE via a separate class).
        if self._use_fp32_rope:
            orig_dtype = q.dtype
            q, k = q.float(), k.float()
        if is_thd:
            max_q = getattr(packed_seq_params, "max_seqlen_q", None)
            max_kv = getattr(packed_seq_params, "max_seqlen_kv", None)
            if max_q is None or max_kv is None:
                seq_len_for_rope = int(packed_seq_params.cu_seqlens_q[-1])
            else:
                seq_len_for_rope = int(max(max_q, max_kv))
            # Match MC RotaryEmbedding.get_rotary_seq_len for packed THD: the
            # rotary length is the max per-sequence padded length, not total
            # packed tokens. Using total tokens makes rope_utils switch to
            # offset mapping, so later packed sequences do not restart at pos 0.
            #
            # MC contract (gpt_model.py:380-381): THD path passes packed_seq=True so the
            # rotary skips its internal cp-slice; _apply_rotary_pos_emb_thd does the
            # cp-zigzag slice itself via _get_thd_freqs_on_this_cp_rank.
            freqs = self.rotary(seq_len_for_rope, packed_seq=True)
            q = _apply_rotary_pos_emb_thd(
                q, packed_seq_params.cu_seqlens_q, freqs,
                rotary_interleaved=False, mscale=1.0,
                cp_group=self.ps.cp_group,
            )
            k = _apply_rotary_pos_emb_thd(
                k, packed_seq_params.cu_seqlens_kv, freqs,
                rotary_interleaved=False, mscale=1.0,
                cp_group=self.ps.cp_group,
            )
        else:
            # q is CP-zigzag pre-sliced; rotary needs FULL seq len,
            # its internal get_pos_emb_on_this_cp_rank re-slices to local len.
            local_seq_len = q.size(0)
            seq_len_for_rope = local_seq_len * self.ps.cp_size
            freqs = self.rotary(seq_len_for_rope)
            q = _apply_rotary_pos_emb_bshd(q, freqs, rotary_interleaved=False, mscale=1.0)
            k = _apply_rotary_pos_emb_bshd(k, freqs, rotary_interleaved=False, mscale=1.0)
        if self._use_fp32_rope:
            q, k = q.to(orig_dtype), k.to(orig_dtype)

        if is_thd:
            psp_kwargs = {
                k: getattr(packed_seq_params, k)
                for k in _KEPT_PSP_FIELDS
                if getattr(packed_seq_params, k, None) is not None
            }
            attn_out = self.core_attn(
                q, k, v,
                core_attention_bias_type="no_bias",
                attn_mask_type="padding_causal",
                **psp_kwargs,
            )
            attn_out = attn_out.reshape(attn_out.size(0), 1, -1)
        else:
            attn_out = self.core_attn(q, k, v, core_attention_bias_type="no_bias")
            if attn_out.dim() > x.dim():
                shape = attn_out.shape
                attn_out = attn_out.reshape(*shape[:-2], self.num_heads_local * self.head_dim)

        if gate is not None:
            gate_fp32 = gate.reshape(attn_out.shape).float().sigmoid()
            attn_out = (attn_out.float() * gate_fp32).to(attn_out.dtype)
        output = self.proj(attn_out)
        return output

    def _split_qkv(self, qkv: torch.Tensor):
        nq, nkv, hd = self.num_heads_local, self.num_kv_heads_local, self.head_dim
        lead = qkv.shape[:-1]
        if self._qkv_layout == "mcore":
            q_per_group = ensure_divisible(nq, nkv)
            if self._output_gate:
                qkv = qkv.view(*lead, nkv, (2 * q_per_group + 2) * hd)
                q = qkv[..., : q_per_group * hd].reshape(*lead, nq, hd)
                gate = qkv[..., q_per_group * hd : 2 * q_per_group * hd].reshape(
                    *lead, nq, hd,
                )
                k_start = 2 * q_per_group * hd
                k = qkv[..., k_start : k_start + hd]
                v = qkv[..., k_start + hd : k_start + 2 * hd]
                return q, gate, k, v

            qkv = qkv.view(*lead, nkv, (q_per_group + 2) * hd)
            q = qkv[..., : q_per_group * hd].reshape(*lead, nq, hd)
            k = qkv[..., q_per_group * hd : (q_per_group + 1) * hd]
            v = qkv[..., (q_per_group + 1) * hd : (q_per_group + 2) * hd]
            return q, None, k, v

        if self._output_gate:
            q_block = qkv[..., : nq * 2 * hd].reshape(*lead, nq, 2 * hd)
            kv_block = qkv[..., nq * 2 * hd :].reshape(*lead, 2 * nkv, hd)
            q = q_block[..., :hd]
            gate = q_block[..., hd:]
            k = kv_block[..., :nkv, :]
            v = kv_block[..., nkv:, :]
            return q, gate, k, v
        qkv = qkv.view(*lead, nq + 2 * nkv, hd)
        # Match MCore SelfAttention's split path: keep q/k/v as views instead
        # of inserting copy nodes, since those copy nodes alter backward
        # accumulation at the qkv boundary under TP/CP.
        q = qkv[..., :nq, :]
        k = qkv[..., nq : nq + nkv, :]
        v = qkv[..., nq + nkv :, :]
        return q, None, k, v
