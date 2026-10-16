# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax Sparse Attention (MSA): lightning indexer + block-sparse core attention on MagiAttention.

Model-agnostic primitive for the MiniMax-M3 family. Composes with
:class:`~megatron.lite.primitive.modules.gqa.GQAttention` (projections, per-head
QK-norm, partial RoPE) and replaces the core attention with Magi's ``calc_msa``
(MagiAttention MSA extension + msa_v1 kernels, :mod:`megatron.lite.primitive.kernels.magi_msa`).

Reference: HF ``transformers.models.minimax_m3_vl`` (``MiniMaxM3VLIndexer`` /
``MiniMaxM3VLAttention``).

Indexer semantics
* index-Q ``[H_idx x D_idx]`` and a single shared index-K ``[1 x D_idx]`` projected
  from the *normalised* attention input; both Gemma-RMSNormed then partially
  RoPE'd with the main branch's frequencies.
* Magi scores ``iQ . iK^T`` (no ``1/sqrt(d)``), masks keys ``j > i``, block max-pools,
  forces the query's own block in and keeps the top-k blocks per query token.
* Default: the indexer is a frozen selector: its projections run under ``torch.no_grad()`` and its
  parameters carry ``requires_grad=False`` (``kl_loss_coeff=0`` is not a freeze).
* ``indexer_trainable=True`` (MiniMax training): the indexer is trained only by Magi's KL loss;
  its input is detached so no indexer gradient reaches the backbone (mcore ``detach_gradient="qk"``).

Ownership split
* mlite: projections, QK-norm, RoPE by doc-local position, indexer projections, output linear.
* Magi: indexer scoring + top-k, sparse attention and all CP communication on its own
  load-balanced token layout (``MagiMsaContext`` carries the runtime key and positions).

Packed multi-document batches are supported (``cu_seqlens`` from the protocol). Attention
TP must be 1 and shapes are fixed to the msa_v1 kernels (64/4/4 heads, D=128, block 128,
top-16). Not supported: MRoPE, output gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.kernels import magi_msa
from megatron.lite.primitive.modules.gqa import GQAttention
from megatron.lite.primitive.parallel import ColumnParallelLinear, ParallelState
from megatron.lite.primitive.utils.rope import apply_rope_sbhd


class MSAIndexer(nn.Module):
    """Index-Q / index-K projections of the MSA lightning indexer (scoring and top-k run inside Magi)."""

    def __init__(
        self,
        hidden_size: int,
        *,
        index_n_heads: int,
        index_head_dim: int,
        ps: ParallelState,
        rms_norm_eps: float = 1e-6,
        zero_centered_gamma: bool = True,
        trainable: bool = False,
        rope_fusion: bool = False,
    ):
        super().__init__()
        self.trainable = trainable
        self.rope_fusion = rope_fusion
        self.num_heads = index_n_heads
        self.head_dim = index_head_dim
        self.q_proj = ColumnParallelLinear(hidden_size, index_n_heads * index_head_dim, ps, bias=False)
        self.k_proj = te.Linear(hidden_size, index_head_dim, bias=False, params_dtype=torch.bfloat16)
        self.q_norm = te.RMSNorm(index_head_dim, eps=rms_norm_eps, zero_centered_gamma=zero_centered_gamma)
        self.k_norm = te.RMSNorm(index_head_dim, eps=rms_norm_eps, zero_centered_gamma=zero_centered_gamma)
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def index_qk(self, x_normed: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x_normed`` ``[S, B, hidden]``, ``freqs`` ``[S, 1, 1, rot]`` -> ``iq`` ``[S, B, H_idx, D]``, ``ik`` ``[S, B, 1, D]``."""
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            x_normed = x_normed.detach()
            S, B = x_normed.shape[:2]
            iq = self.q_proj(x_normed).view(S, B, -1, self.head_dim)
            ik = self.k_proj(x_normed).view(S, B, 1, self.head_dim)
            iq = apply_rope_sbhd(self.q_norm(iq), freqs, fused=self.rope_fusion)
            ik = apply_rope_sbhd(self.k_norm(ik), freqs, fused=self.rope_fusion)
        return iq, ik


class MagiCoreAttention(nn.Module):
    """Parameter-free MSA core (indexer scoring + sparse attention + CP comm via Magi ``calc_msa``).

    A module so recompute/offload selectors (``MODULE_MAP["core_attn"]``) have a real target.
    Inputs are flat ``[T_local, heads, 128]`` bf16 tensors.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        *,
        magi_ctx: magi_msa.MagiMsaContext,
    ) -> torch.Tensor:
        return magi_msa.calc_msa_v1(q, k, v, index_q, index_k, magi_ctx)


class MSAttention(GQAttention):
    """GQAttention whose core attention is MiniMax block-sparse attention (Magi msa_v1)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        ps: ParallelState,
        *,
        index_n_heads: int,
        index_head_dim: int,
        block_size: int = 128,
        topk_blocks: int = 16,
        local_blocks: int = 1,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5_000_000.0,
        rotary_percent: float = 0.5,
        zero_centered_gamma: bool = True,
        qkv_layout: str = "flat",
        rope_fusion: bool = False,
        index_input_norm: str = "recompute",
        indexer_trainable: bool = False,
    ):
        if index_input_norm not in ("recompute", "fused"):
            raise ValueError(f"index_input_norm must be 'recompute' or 'fused', got {index_input_norm!r}")
        if index_n_heads != num_key_value_heads:
            raise ValueError("MSA requires index_n_heads == num_key_value_heads (selection is per GQA group)")
        if ps.tp_size > 1:
            raise NotImplementedError("MSA requires attention TP=1 (msa_v1 kernels are fixed to 64/4/4 heads)")
        magi_msa.validate_kernel_shapes(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            block_size=block_size,
            topk_blocks=topk_blocks,
            local_blocks=local_blocks,
        )
        super().__init__(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            ps,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rotary_percent=rotary_percent,
            use_thd=False,
            output_gate=False,
            zero_centered_gamma=zero_centered_gamma,
            qkv_layout=qkv_layout,
            attention_backend="magi",
            rope_fusion=rope_fusion,
            qkv_return_norm_output=index_input_norm == "fused",
        )
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.indexer = MSAIndexer(
            hidden_size,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            ps=ps,
            rms_norm_eps=rms_norm_eps,
            zero_centered_gamma=zero_centered_gamma,
            trainable=indexer_trainable,
            rope_fusion=rope_fusion,
        )

    def _build_core_attn(self, attention_backend: str) -> nn.Module:
        return MagiCoreAttention()

    def forward(self, x: torch.Tensor, magi_ctx: magi_msa.MagiMsaContext | None = None) -> torch.Tensor:
        """``x`` ``[T_local, 1, hidden]`` on Magi's dispatched layout -> ``[T_local, 1, hidden]``."""
        if magi_ctx is None:
            raise ValueError("MSAttention requires magi_ctx (built by the protocol's forward step)")
        qkv = self.qkv(x)
        if self.qkv.return_layernorm_output:
            qkv, index_input = qkv
        else:
            index_input = self._qkv_lora_input(x)
        q, _gate, k, v = self._split_qkv(qkv)  # [T, 1, H, D]
        T, B = q.shape[:2]
        if B != 1 or T != magi_ctx.local_tokens:
            raise ValueError(f"MSAttention expects [T_local={magi_ctx.local_tokens}, 1, hidden] input, got {tuple(x.shape)}")
        freqs = magi_ctx.rope_freqs_for(self.rotary)  # [T, 1, 1, rot] by doc-local position
        q = apply_rope_sbhd(self.q_norm(q), freqs, fused=self._rope_fusion)
        k = apply_rope_sbhd(self.k_norm(k), freqs, fused=self._rope_fusion)
        iq, ik = self.indexer.index_qk(index_input, freqs)
        out = self.core_attn(
            q.squeeze(1).contiguous(),
            k.squeeze(1).contiguous(),
            v.squeeze(1).contiguous(),
            iq.squeeze(1).contiguous(),
            ik.squeeze(1).contiguous(),
            magi_ctx=magi_ctx,
        )  # [T, H, D]
        return self.proj(out.reshape(T, 1, self.num_heads_local * self.head_dim))
