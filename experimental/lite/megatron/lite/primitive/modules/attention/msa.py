# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax Sparse Attention (MSA): lightning indexer + block-sparse core attention.

Model-agnostic primitive for the MiniMax-M3 family. Composes with
:class:`~megatron.lite.primitive.modules.gqa.GQAttention` (projections, per-head
QK-norm, partial RoPE, TP sharding) and swaps the dense TE core attention for
the per-token block-sparse attention in
:mod:`megatron.lite.primitive.kernels.msa_kernels`.

Reference: HF ``transformers.models.minimax_m3_vl`` (``MiniMaxM3VLIndexer`` /
``MiniMaxM3VLAttention``) and ``experimental/lite/ref/minimax_m3/msa_ref.py``.

Indexer semantics
* index-Q ``[H_idx x D_idx]`` and a single shared index-K ``[1 x D_idx]`` projected
  from the *normalised* attention input; both Gemma-RMSNormed then partially
  RoPE'd with the main branch's frequencies.
* token scores ``iQ . iK^T`` in fp32 (no ``1/sqrt(d)``), keys ``j > i`` masked, block
  max-pool (computed chunk-wise over KV blocks, never a dense S x S_k tensor), the
  query's own block(s) forced in, top-k per query token, ``-1`` pad.
* The indexer is a pure selector: top-k is non-differentiable, and (P0 decision)
  no auxiliary loss is attached, so it runs under ``torch.no_grad()``.

TP contract
* ``index_n_heads`` must equal ``num_key_value_heads``; index heads are sharded
  exactly like KV heads (``tp <= H_idx``) or replicated by all-gather (``tp > H_idx``).
* index-K is a single head: its weight is replicated on every TP rank; under
  sequence parallel the input is all-gathered first.

CP contract (all-gather CP, zigzag layout like the dense TE path)
* every rank holds a zigzag shard of the sequence; ``position_ids`` are the *global*
  positions of the local tokens (``zigzag_position_ids_for_cp`` when not given).
* K/V and index-K are projected and RoPE'd locally, all-gathered over the CP group and
  reordered to global sequence order; queries stay local. Block indices, the block
  max-pool and the causal mask are all computed in global positions, so the shard
  boundaries need no 128-alignment. dK/dV flow back through the gather.

``backend="magi"`` (MagiAttention MSA extension, ``kernels/magi_msa.py``)
* bypasses everything below the projections: no ``self.indexer(...)`` scoring, no
  ``gather_cp_sequence``, no ``msa_core_attention``. The indexer projections
  (``MSAIndexer.index_qk``) run under ``torch.no_grad()`` and Magi's ``calc_msa`` does
  indexer + sparse attention + all CP communication on its own load-balanced token
  layout (``MagiMsaContext`` carries the runtime key and the doc-local positions for RoPE).
* packed multi-document batches are supported (``cu_seqlens`` from the protocol);
  attention TP must be 1 (kernel shapes are fixed to 64/4/4 heads); indexer parameters
  carry ``requires_grad=False`` (``kl_loss_coeff=0`` is not a freeze).

Not supported in this drop: THD / packed sequences on the dense/flex backends, MRoPE,
output gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.kernels import magi_msa, msa_kernels
from megatron.lite.primitive.modules.gqa import GQAttention
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    all_gather_last_dim_with_grad_reduce,
    zigzag_position_ids_for_cp,
)
from megatron.lite.primitive.parallel.sp import gather_from_sequence_parallel
from megatron.lite.primitive.utils import ensure_divisible
from megatron.lite.primitive.utils.rope import _apply_rotary_pos_emb_bshd


def _cp_global_order(seq_local: int, ps: ParallelState) -> torch.Tensor:
    """Permutation taking rank-major concatenation of zigzag shards to global sequence order."""
    seq_full = seq_local * ps.cp_size
    rank_major = torch.cat(
        [zigzag_position_ids_for_cp(seq_full, r, ps.cp_size, torch.device("cuda")).flatten() for r in range(ps.cp_size)]
    )
    return torch.argsort(rank_major)


def gather_cp_sequence(x: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    """``[S_local, ...]`` zigzag shard -> ``[S_full, ...]`` in global order (autograd all-gather)."""
    if ps.cp_size <= 1:
        return x
    from torch.distributed.nn.functional import all_gather

    parts = all_gather(x.contiguous(), group=ps.cp_group)
    return torch.cat(list(parts), dim=0).index_select(0, _cp_global_order(x.shape[0], ps))


_SCORE_CHUNK_ELEMS = 1 << 28  # ~1 GiB of fp32 scores per chunk


def _blockwise_max_scores(iq: torch.Tensor, ik: torch.Tensor, position_ids: torch.Tensor, block_size: int) -> torch.Tensor:
    """``iq`` ``[B, H, S_q, D]``, ``ik`` ``[B, 1, S_k, D]`` (fp32) -> causal block-max scores ``[B, H, S_q, n_kv_blocks]``.

    Keys ``j > position_ids[q]`` (and the zero padding of the last block) score ``-inf``; a block whose keys are
    all masked therefore stays ``-inf``, exactly like HF's dense scores followed by ``amax``.
    """
    B, H, S_q, D = iq.shape
    S_k = ik.shape[2]
    n_kv = -(-S_k // block_size)
    pad = n_kv * block_size - S_k
    if pad:
        ik = torch.nn.functional.pad(ik, (0, 0, 0, pad))
    k_pos = torch.arange(n_kv * block_size, device=iq.device)  # padded keys have positions >= S_k > any query
    blocks_per_chunk = max(1, _SCORE_CHUNK_ELEMS // (B * H * S_q * block_size))
    out = iq.new_empty(B, H, S_q, n_kv)
    q_pos = position_ids[:, None, :, None]
    for j0 in range(0, n_kv, blocks_per_chunk):
        j1 = min(n_kv, j0 + blocks_per_chunk)
        keys = ik[:, :, j0 * block_size : j1 * block_size]
        scores = torch.matmul(iq, keys.transpose(-1, -2))  # [B, H, S_q, (j1-j0)*block]
        scores = scores.masked_fill(k_pos[None, None, None, j0 * block_size : j1 * block_size] > q_pos, float("-inf"))
        out[..., j0:j1] = scores.view(B, H, S_q, j1 - j0, block_size).amax(dim=-1)
    return out


class MSAIndexer(nn.Module):
    """Per-token top-k KV-block selection for MSA. Inputs/outputs use ``[S, B, ...]`` (sbhd) like GQAttention."""

    def __init__(
        self,
        hidden_size: int,
        *,
        index_n_heads: int,
        index_head_dim: int,
        block_size: int,
        topk_blocks: int,
        ps: ParallelState,
        local_blocks: int = 1,
        rms_norm_eps: float = 1e-6,
        zero_centered_gamma: bool = True,
    ):
        super().__init__()
        self.num_heads = index_n_heads
        self.head_dim = index_head_dim
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.local_blocks = local_blocks
        self.ps = ps
        self._replicate_heads = index_n_heads < ps.tp_size
        if self._replicate_heads:
            ensure_divisible(ps.tp_size, index_n_heads)
            self.num_heads_local = 1
        else:
            self.num_heads_local = ensure_divisible(index_n_heads, ps.tp_size)
        # index-Q: sharded like KV heads. index-K: single head, replicated (plain TE linear, no TP).
        self.q_proj = ColumnParallelLinear(hidden_size, index_n_heads * index_head_dim, ps, bias=False)
        self.k_proj = te.Linear(hidden_size, index_head_dim, bias=False, params_dtype=torch.bfloat16)
        self.q_norm = te.RMSNorm(index_head_dim, eps=rms_norm_eps, zero_centered_gamma=zero_centered_gamma)
        self.k_norm = te.RMSNorm(index_head_dim, eps=rms_norm_eps, zero_centered_gamma=zero_centered_gamma)

    def _local_head_slice(self, iq_full: torch.Tensor) -> torch.Tensor:
        """When heads are replicated (tp > H_idx) keep the head serving this rank's q heads."""
        if not self._replicate_heads:
            return iq_full
        ranks_per_head = self.ps.tp_size // self.num_heads
        h = self.ps.tp_rank // ranks_per_head
        return iq_full[:, :, h : h + 1]

    @torch.no_grad()
    def index_qk(self, x_normed: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Index-Q / index-K for the local tokens: projections + Gemma norm + partial RoPE (no scoring).

        ``x_normed``: ``[S_local, B, hidden]``; ``freqs``: ``[S_local, 1, 1, rot]``. Returns
        ``iq`` ``[S_local, B, H_idx_local, D]`` and ``ik`` ``[S_local, B, 1, D]``. Always under
        ``torch.no_grad()``: the indexer is a frozen selector.
        """
        iq = self.q_proj(x_normed)  # [S, B, H_local*D] (SP gathered inside TE)
        if self._replicate_heads:
            iq = all_gather_last_dim_with_grad_reduce(iq, self.ps.tp_group)
        S, B = iq.shape[:2]
        iq = self._local_head_slice(iq.view(S, B, -1, self.head_dim))
        x_full = gather_from_sequence_parallel(x_normed, self.ps) if self.q_proj.use_sp else x_normed
        ik = self.k_proj(x_full).view(S, B, 1, self.head_dim)
        iq = _apply_rotary_pos_emb_bshd(self.q_norm(iq), freqs)
        ik = _apply_rotary_pos_emb_bshd(self.k_norm(ik), freqs)
        return iq, ik

    @torch.no_grad()
    def forward(self, x_normed: torch.Tensor, freqs: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """``x_normed``: ``[S_local, B, hidden]`` (SP-sharded if SP; CP zigzag shard if CP); ``freqs``: ``[S_local, 1, 1, rot]``
        for the local tokens; ``position_ids``: ``[B, S_local]`` global positions.

        Returns HF-format block indices ``[B, H_idx_local, S_local, K]`` over the *global* KV blocks (``-1`` right padding).
        """
        iq, ik = self.index_qk(x_normed, freqs)
        ik = gather_cp_sequence(ik, self.ps)  # [S_k, B, 1, D] global order
        S_k = ik.shape[0]
        # -> [B, H, S, D] fp32 scores like HF, but pooled to block maxima chunk by chunk over the KV blocks so
        # nothing of size S x S_k is materialised (HF's dense [B, H, S, S_k] is O(S^2) memory).
        iq = iq.permute(1, 2, 0, 3).float()
        ik = ik.permute(1, 2, 0, 3).float()
        block_scores = _blockwise_max_scores(iq, ik, position_ids, self.block_size)
        n_kv_blocks = block_scores.shape[-1]
        q_block = position_ids // self.block_size
        if self.local_blocks > 0:
            local = torch.arange(self.local_blocks, device=block_scores.device)
            local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
            local_idx = local_idx.unsqueeze(1).expand(-1, iq.shape[1], -1, -1)
            block_scores.scatter_(-1, local_idx, float("inf"))
        k = min(self.topk_blocks, n_kv_blocks)
        top_scores, top_idx = block_scores.topk(k, dim=-1)
        return top_idx.masked_fill(top_scores == float("-inf"), -1).to(torch.int32)


class MagiCoreAttention(nn.Module):
    """Parameter-free core of the ``magi`` backend (indexer + sparse attention + CP comm via Magi ``calc_msa``).

    Exists as a module so recompute/offload selectors (``MODULE_MAP["core_attn"]``) have a real target.
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
    """GQAttention whose core attention is MiniMax block-sparse attention."""

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
        backend: str = "flex",
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5_000_000.0,
        rotary_percent: float = 0.5,
        zero_centered_gamma: bool = True,
        qkv_layout: str = "flat",
    ):
        if index_n_heads != num_key_value_heads:
            raise ValueError("MSA requires index_n_heads == num_key_value_heads (selection is per GQA group)")
        if backend == "magi" and ps.tp_size > 1:
            raise NotImplementedError(
                "msa_backend='magi' requires attention TP=1 in this drop (msa_v1 kernels are fixed to 64/4/4 heads); "
                "use CP/EP/PP for parallelism or msa_backend='flex'"
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
        )
        if backend not in msa_kernels.BACKENDS:
            raise ValueError(f"unknown MSA backend {backend!r}; expected one of {msa_kernels.BACKENDS}")
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.backend = backend
        if backend == "magi":
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
            # self.core_attn (param-free TE module) is kept so state_dict keys match the flex/dense backends,
            # but it is never called: Magi owns indexer + sparse attention + CP comm via magi_core.
            self.magi_core = MagiCoreAttention()
        self.indexer = MSAIndexer(
            hidden_size,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            block_size=block_size,
            topk_blocks=topk_blocks,
            local_blocks=local_blocks,
            ps=ps,
            rms_norm_eps=rms_norm_eps,
            zero_centered_gamma=zero_centered_gamma,
        )
        self.last_block_indices: torch.Tensor | None = None  # for alignment tests (None under backend="magi")
        # Frozen selector (constitution minimax_m3): the indexer forward runs under no_grad on every backend, so its
        # parameters never receive gradients. Mark them requires_grad=False explicitly as well: an optimizer that
        # treats a missing gradient as zero (e.g. dist_opt) would otherwise still apply decoupled weight decay and
        # silently drift the HF indexer weights.
        for p in self.indexer.parameters():
            p.requires_grad_(False)

    def _forward_magi(self, x: torch.Tensor, magi_ctx: magi_msa.MagiMsaContext) -> torch.Tensor:
        """``x`` ``[T_local, 1, hidden]`` on Magi's dispatched layout -> ``[T_local, 1, hidden]``."""
        qkv = self.qkv(x)
        q, _gate, k, v = self._split_qkv(qkv)  # [T, 1, H, D]
        T, B = q.shape[:2]
        if B != 1 or T != magi_ctx.local_tokens:
            raise ValueError(f"magi backend expects [T_local={magi_ctx.local_tokens}, 1, hidden] input, got {tuple(x.shape)}")
        freqs = magi_ctx.rope_freqs_for(self.rotary)  # [T, 1, 1, rot] by doc-local position
        q = _apply_rotary_pos_emb_bshd(self.q_norm(q), freqs)
        k = _apply_rotary_pos_emb_bshd(self.k_norm(k), freqs)
        x_normed = self._qkv_lora_input(x)
        with torch.no_grad():
            iq, ik = self.indexer.index_qk(x_normed, freqs)
        self.last_block_indices = None
        out = self.magi_core(
            q.squeeze(1).contiguous(),
            k.squeeze(1).contiguous(),
            v.squeeze(1).contiguous(),
            iq.squeeze(1).detach().contiguous(),
            ik.squeeze(1).detach().contiguous(),
            magi_ctx=magi_ctx,
        )  # [T, H, D]
        return self.proj(out.reshape(T, 1, self.num_heads_local * self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        magi_ctx: magi_msa.MagiMsaContext | None = None,
    ):
        if self.backend == "magi":
            if magi_ctx is None:
                raise ValueError("MSAttention(backend='magi') requires magi_ctx (built by the protocol's forward step)")
            return self._forward_magi(x, magi_ctx)
        if magi_ctx is not None:
            raise ValueError(f"magi_ctx was given but the MSA backend is {self.backend!r}")
        if packed_seq_params is not None:
            raise NotImplementedError("MSA does not support THD/packed sequences in this drop")
        qkv = self.qkv(x)
        if self._replicate_kv:
            qkv = all_gather_last_dim_with_grad_reduce(qkv, self.ps.tp_group)
        q, _gate, k, v = self._split_qkv(qkv)  # [S, B, H, D]
        S, B = q.shape[:2]
        if position_ids is None:
            position_ids = zigzag_position_ids_for_cp(S * self.ps.cp_size, self.ps.cp_rank, self.ps.cp_size, x.device)
            position_ids = position_ids.expand(B, -1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        freqs = self.rotary(S * self.ps.cp_size)  # with cp_group the rotary returns this rank's zigzag slice
        q = _apply_rotary_pos_emb_bshd(q, freqs)
        k = _apply_rotary_pos_emb_bshd(k, freqs)

        x_normed = self._qkv_lora_input(x)  # recompute the fused pre-attention RMSNorm output
        block_idx = self.indexer(x_normed, freqs, position_ids)
        self.last_block_indices = block_idx

        k = gather_cp_sequence(k, self.ps)
        v = gather_cp_sequence(v, self.ps)

        out = msa_kernels.msa_core_attention(
            q.permute(1, 2, 0, 3),
            k.permute(1, 2, 0, 3),
            v.permute(1, 2, 0, 3),
            block_idx,
            position_ids,
            block_size=self.block_size,
            scale=self.scaling if hasattr(self, "scaling") else None,
            backend=self.backend,
        )  # [B, H, S, D]
        out = out.permute(2, 0, 1, 3).reshape(S, B, self.num_heads_local * self.head_dim)
        return self.proj(out)
