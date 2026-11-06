# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Block-sparse attention with a learned block indexer (Magi-MSA ``MSAV1`` backend).

Each query head attends to ``topk_blocks`` key blocks of its KV group, chosen per
token by a small indexer (block score = max of token scores, the query's own
block always kept). The indexer is trained only by a KL distillation gradient
that Magi-MSA injects in backward; there is no KL term in the loss scalar.

Context parallelism, the dispatch permutation and per-document RoPE positions
come from one Magi-MSA runtime key per packed microbatch
(:func:`build_msa_runtime_key`). The key is built from host ``cu_seqlens`` before
the embedding and reused by every layer, pipeline stage and recompute; CP=1
uses a real world-size-one process group.

The projections run as Magi-MSA schedule callbacks (``calc_msa_scheduled``) so
the runtime can overlap them with CP communication. Tensors are sequence-major
``[S, 1, H]`` (micro-batch 1 after dispatch).

Models that mix dense (full) attention layers with MSA layers use a hybrid key
(:class:`MsaHybridKey`): the MSA key owns the dispatch, and a dense MagiAttention key
built from the same ranges, mask types and dispatch config computes the dense layers
on the same token layout (:class:`MsaDenseAttention`).
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive import transformer_engine as te

logger = logging.getLogger(__name__)

# MSAV1 kernels hard-code these shapes (Magi-MSA ``MSAv1Property``).
MSA_V1_NUM_HEADS = 64
MSA_V1_NUM_KV_HEADS = 4
MSA_V1_HEAD_DIM = 128
MSA_V1_NUM_INDEX_HEADS = 4
MSA_V1_INDEX_DIM = 128


@lru_cache(maxsize=1)
def _msa_api():
    try:
        from magi_attention.config import DispatchConfig, MinHeapDispatchAlg
        from magi_attn_extensions import MSA as msa
        from magi_attn_extensions.MSA.utils import resolve_chunk_size
    except (ImportError, OSError) as exc:
        raise ImportError(
            "MSA attention requires Magi-MSA (magi_attn_extensions) and MagiAttention."
        ) from exc
    return msa, DispatchConfig, MinHeapDispatchAlg, resolve_chunk_size


@dataclass(frozen=True)
class MsaV1Numerics:
    """Numerical settings of the MSA v1 indexer, sparse attention and KL gradient.

    The KL gradient is ``kl_loss_coeff * T**2`` times the mean over (tokens, index
    heads) of KL(teacher || indexer) at temperature ``T = kl_temperature``.
    """

    head_dim: int = MSA_V1_HEAD_DIM
    index_dim: int = MSA_V1_INDEX_DIM
    block_size: int = 128
    topk_blocks: int = 16
    init_blocks: int = 0
    local_blocks: int = 1
    kl_loss_coeff: float = 0.0
    kl_temperature: float = 1.0
    use_fp16_score: bool = False


def msa_v1_rank_kl_coeff(
    numerics: MsaV1Numerics, *, tp_size: int, cp_size: int, total_tokens: int, per_token_loss: bool
) -> float:
    """KL coefficient handed to Magi-MSA for one packed microbatch.

    Magi-MSA rescales it by local / global valid query tokens, so the per-rank
    contributions sum to the single-rank value. In per-token-loss mode the
    gradient is pre-multiplied by the average local token count so the global
    1/num_tokens finalisation yields a token mean. The expression order is kept
    fixed: the value reaches the KL kernel's gradient scale bit for bit.
    """
    rank_coeff = numerics.kl_loss_coeff * (numerics.kl_temperature**2) / tp_size
    if per_token_loss:
        rank_coeff *= total_tokens / cp_size
    return rank_coeff * cp_size


@dataclass(frozen=True)
class MsaHybridKey:
    """Runtime key of a model with dense and MSA attention layers: ``sparse`` (the Magi-MSA
    key) owns dispatch, positions and the MSA layers; ``dense`` is a MagiAttention key with
    the same dispatch solution for the dense layers."""

    sparse: Any
    dense: Any


def _sparse_key(key: Any) -> Any:
    return key.sparse if isinstance(key, MsaHybridKey) else key


@lru_cache(maxsize=1)
def _dense_api():
    try:
        from magi_attention.api import calc_attn, dispatch, magi_attn_flex_key
        from magi_attention.config import DistAttnConfig
    except (ImportError, OSError) as exc:
        raise ImportError("Dense attention layers next to MSA need MagiAttention.") from exc
    return calc_attn, magi_attn_flex_key, DistAttnConfig, dispatch


# MagiAttention reads its kernel switches from the environment. Its Blackwell kernel (FA4) has no
# deterministic mode, so deterministic runs use its online SDPA backend, in FP32 (in BF16 that
# backend is less accurate than a fused BF16 kernel). These settings are process-wide.
_DETERMINISTIC_DENSE_ENV = {
    "MAGI_ATTENTION_DETERMINISTIC_MODE": "1",
    "MAGI_ATTENTION_KERNEL_BACKEND": "sdpa_ol",
    "MAGI_ATTENTION_PRECISION": "fp32",
}
_LEGACY_BACKEND_ENV = ("MAGI_ATTENTION_FA4_BACKEND", "MAGI_ATTENTION_SDPA_BACKEND")


def _use_deterministic_dense_kernels() -> None:
    changed = {k: v for k, v in _DETERMINISTIC_DENSE_ENV.items() if os.environ.get(k) != v}
    changed.update({k: None for k in _LEGACY_BACKEND_ENV if k in os.environ})
    if not changed:
        return
    logger.warning(
        "deterministic dense attention: setting MagiAttention environment %s for this process",
        changed,
    )
    for k in _LEGACY_BACKEND_ENV:
        os.environ.pop(k, None)
    os.environ.update(_DETERMINISTIC_DENSE_ENV)


def _dense_key_for_msa_layout(
    msa_key: Any, *, num_heads: int, num_kv_heads: int, head_dim: int, deterministic: bool
) -> Any:
    """MagiAttention key for the dense layers on the MSA key's dispatch (same Q/K ranges, mask
    types and dispatch config, hence the same token layout)."""
    _calc_attn, magi_attn_flex_key, DistAttnConfig, magi_dispatch = _dense_api()
    if deterministic:
        _use_deterministic_dense_kernels()
    total = int(msa_key.cu_seqlens[-1])
    dense = magi_attn_flex_key(
        q_ranges=msa_key.q_ranges,
        k_ranges=msa_key.k_ranges,
        attn_mask_type=list(msa_key.attn_mask_type),
        total_seqlen_q=total,
        total_seqlen_k=total,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
        pad_size=0,
        cp_group_or_mesh=msa_key.cp_group,
        dist_attn_config=DistAttnConfig(dispatch_config=msa_key.dispatch_config),
    )
    # Both layer kinds must see the same tokens in the same order on every CP rank.
    token = torch.arange(total, device=torch.cuda.current_device())
    if not torch.equal(magi_dispatch(token, dense), _msa_api()[0].dispatch(token, msa_key)):
        raise RuntimeError("MagiAttention dense key dispatches differently from the MSA key")
    return dense


def build_msa_runtime_key(
    cu_seqlens: torch.Tensor,
    cp_group: dist.ProcessGroup,
    *,
    numerics: MsaV1Numerics,
    chunk_size: int,
    tp_size: int = 1,
    per_token_loss: bool = True,
    deterministic: bool = False,
    dense_heads: tuple[int, int] | None = None,
) -> Any:
    """Create (or fetch from Magi-MSA's cache) the runtime key of one packed microbatch.

    ``cu_seqlens`` must be host int32 (Magi-MSA rejects device metadata). Dispatch
    uses equal CP shards (MinHeap, no uneven shard) so pipeline shapes stay static,
    which requires ``total_tokens % (chunk_size * cp_size) == 0``. With
    ``dense_heads=(num_heads, num_kv_heads)`` the model also has dense attention layers
    and an :class:`MsaHybridKey` is returned.
    """
    if cu_seqlens.device.type != "cpu" or cu_seqlens.dtype != torch.int32:
        raise ValueError("MSA cu_seqlens must be a host int32 tensor")
    msa, DispatchConfig, MinHeapDispatchAlg, resolve_chunk_size = _msa_api()
    total_tokens = int(cu_seqlens[-1])
    cp_size = dist.get_world_size(cp_group)
    requested = DispatchConfig(chunk_size=chunk_size, uneven_shard=False, alg=MinHeapDispatchAlg())
    chunk = resolve_chunk_size(total_tokens, cp_size, requested)
    if total_tokens % (chunk * cp_size) != 0:
        raise ValueError(
            f"MSA needs equal CP shards: total_tokens={total_tokens} is not divisible by "
            f"chunk_size({chunk}) * cp_size({cp_size})"
        )
    config = msa.MsaConfig(
        softmax_scale=numerics.head_dim**-0.5,
        indexer_softmax_scale=numerics.index_dim**-0.5,
        init_blocks=numerics.init_blocks,
        local_blocks=numerics.local_blocks,
        kl_loss_coeff=msa_v1_rank_kl_coeff(
            numerics,
            tp_size=tp_size,
            cp_size=cp_size,
            total_tokens=total_tokens,
            per_token_loss=per_token_loss,
        ),
        kernel_backend=msa.MsaKernelBackend.MSAV1,
        deterministic=deterministic,
        topk_blocks=numerics.topk_blocks,
        kl_temperature=numerics.kl_temperature,
        use_fp16_score=numerics.use_fp16_score,
    )
    dispatch_config = DispatchConfig(chunk_size=chunk, uneven_shard=False, alg=MinHeapDispatchAlg())
    key = msa.make_magi_msa_key(
        cu_seqlens, cp_group, config=config, dispatch_config=dispatch_config
    )
    if dense_heads is None:
        return key
    num_heads, num_kv_heads = dense_heads
    dense = _dense_key_for_msa_layout(
        key,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=numerics.head_dim,
        deterministic=deterministic,
    )
    return MsaHybridKey(sparse=key, dense=dense)


def msa_dispatch(tensor: torch.Tensor, key: Any) -> torch.Tensor:
    """Permute a global packed token-major tensor into this rank's dispatched tokens."""
    return _msa_api()[0].dispatch(tensor, _sparse_key(key))


def msa_undispatch(tensor: torch.Tensor, key: Any) -> torch.Tensor:
    """Restore this rank's dispatched tokens to global packed order (all-gather over CP)."""
    return _msa_api()[0].undispatch(tensor, _sparse_key(key))


def msa_position_ids(key: Any) -> torch.Tensor:
    """Per-document RoPE positions of this rank's dispatched tokens."""
    return _msa_api()[0].get_position_ids(_sparse_key(key))


def rotary_freqs_for_positions(
    position_ids: torch.Tensor, *, rotary_dim: int, base: float
) -> torch.Tensor:
    """Non-interleaved RoPE angles ``[S, 1, 1, rotary_dim]`` (fp32) for ``position_ids``.

    ``inv_freq`` comes from a CUDA pow in the standard order and each angle is the
    fp32 product ``position * inv_freq``, so the result is bit-identical to gathering
    rows of a position table built with ``outer(arange(n), inv_freq)``.
    """
    device = position_ids.device
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    freqs = torch.outer(position_ids.reshape(-1).to(torch.float32), inv_freq)
    return torch.cat((freqs, freqs), dim=-1)[:, None, None, :]


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

    return apply_rotary_pos_emb(x, freqs, tensor_format="sbhd", interleaved=False, fused=True)


def _no_recompute_contexts():
    return nullcontext(), nullcontext()


class _GroupedQKVAttention(nn.Module):
    """Fused QKV projection with rows grouped per KV head ``[q x heads_per_group | k | v]``,
    output projection, per-head (1 + w) RMSNorms on q and k (one weight shared over heads)
    and partial RoPE; subclasses supply the attention core."""

    def __init__(
        self,
        hidden_size: int,
        *,
        rotary_dim: int,
        norm_eps: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be a multiple of num_kv_heads={num_kv_heads}"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.norm_eps = norm_eps
        q_per_group = num_heads // num_kv_heads
        self._group_split = [q_per_group * head_dim, head_dim, head_dim]
        # Declaration order fixes the parameter order (and so the DDP bucket layout).
        self.linear_qkv = self._linear(hidden_size, (num_heads + 2 * num_kv_heads) * head_dim)
        self.linear_proj = self._linear(num_heads * head_dim, hidden_size)
        self.q_norm = self._head_norm(head_dim)
        self.k_norm = self._head_norm(head_dim)

    @staticmethod
    def _linear(in_features, out_features):
        return te.Linear(in_features, out_features, bias=False, params_dtype=torch.bfloat16)

    def _head_norm(self, dim):
        return te.RMSNorm(
            dim, eps=self.norm_eps, zero_centered_gamma=True, params_dtype=torch.bfloat16
        )

    def _sparse_qkv(self, hidden: torch.Tensor, freqs: torch.Tensor):
        mixed = self.linear_qkv(hidden)
        seq, batch = mixed.shape[:2]
        mixed = mixed.view(seq, batch, self.num_kv_heads, -1)
        q, k, v = torch.split(mixed, self._group_split, dim=3)
        # q is not viewable as heads (group stride), so this reshape copies.
        q = q.reshape(seq, batch, self.num_heads, self.head_dim)
        k = k.reshape(seq, batch, self.num_kv_heads, self.head_dim).clone()
        v = v.reshape(seq, batch, self.num_kv_heads, self.head_dim).clone()
        q = _apply_rope(self.q_norm(q), freqs)
        k = _apply_rope(self.k_norm(k), freqs)
        return q[:, 0].contiguous(), k[:, 0].contiguous(), v[:, 0].contiguous()

    def _output(self, core: torch.Tensor) -> torch.Tensor:
        return self.linear_proj(core.view(core.shape[0], 1, -1))

    @staticmethod
    def _check_input(hidden: torch.Tensor) -> None:
        if hidden.dim() != 3 or hidden.shape[1] != 1:
            raise ValueError(f"expected [S, 1, H] dispatched tokens, got {tuple(hidden.shape)}")


class MsaV1Attention(_GroupedQKVAttention):
    """Magi-MSA ``MSAV1`` sparse attention with its block indexer.

    On top of the grouped QKV projections: the indexer ``index_q_proj``/``index_k_proj`` with
    ``index_q_norm``/``index_k_norm``. The indexer reads the detached input and is trained only
    by the KL gradient. Tensor parallelism is not supported (the kernels fix the head counts).
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        rotary_dim: int,
        norm_eps: float,
        num_heads: int = MSA_V1_NUM_HEADS,
        num_kv_heads: int = MSA_V1_NUM_KV_HEADS,
        head_dim: int = MSA_V1_HEAD_DIM,
        num_index_heads: int = MSA_V1_NUM_INDEX_HEADS,
        index_dim: int = MSA_V1_INDEX_DIM,
    ):
        shapes = (num_heads, num_kv_heads, head_dim, num_index_heads, index_dim)
        expected = (
            MSA_V1_NUM_HEADS,
            MSA_V1_NUM_KV_HEADS,
            MSA_V1_HEAD_DIM,
            MSA_V1_NUM_INDEX_HEADS,
            MSA_V1_INDEX_DIM,
        )
        if shapes != expected:
            raise ValueError(f"MSAV1 kernels require head shapes {expected}, got {shapes}")
        super().__init__(
            hidden_size,
            rotary_dim=rotary_dim,
            norm_eps=norm_eps,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.num_index_heads = num_index_heads
        self.index_dim = index_dim
        self.index_q_proj = self._linear(hidden_size, num_index_heads * index_dim)
        self.index_k_proj = self._linear(hidden_size, index_dim)
        self.index_q_norm = self._head_norm(index_dim)
        self.index_k_norm = self._head_norm(index_dim)

    def _index_qk(self, hidden: torch.Tensor, freqs: torch.Tensor):
        x = hidden.detach()
        index_q = self.index_q_proj(x).view(-1, 1, self.num_index_heads, self.index_dim)
        index_k = self.index_k_proj(x).view(-1, 1, 1, self.index_dim)
        index_q = _apply_rope(self.index_q_norm(index_q), freqs)
        index_k = _apply_rope(self.index_k_norm(index_k), freqs)
        return index_q[:, 0].contiguous(), index_k[:, 0].contiguous()

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        runtime_key: Any,
        rotary_freqs: torch.Tensor,
        precision_adapter: Any = None,
    ) -> torch.Tensor:
        """``hidden``: normalized input ``[S, 1, H]`` (dispatched tokens); returns ``[S, 1, H]``."""
        self._check_input(hidden)
        msa = _msa_api()[0]
        ops = msa.MsaV1ScheduleOps(
            index_qk=lambda h, _pos: self._index_qk(h, rotary_freqs),
            sparse_qkv=lambda h, _pos: self._sparse_qkv(h, rotary_freqs),
            output=self._output,
            recompute_qkv_rope=False,
            recompute_sparse_attn=False,
            qkv_context_fn=_no_recompute_contexts,
            precision_adapter=(
                msa.DEFAULT_MSA_PRECISION_ADAPTER
                if precision_adapter is None
                else precision_adapter
            ),
        )
        return msa.calc_msa_scheduled(hidden, _sparse_key(runtime_key), schedule_ops=ops).output


class MsaDenseAttention(_GroupedQKVAttention):
    """Dense causal (per-document) attention for the full-attention layers of a model that also
    has MSA layers: the same projections, norms and RoPE, computed with MagiAttention on the
    MSA dispatch (``runtime_key`` must be an :class:`MsaHybridKey`)."""

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        runtime_key: Any,
        rotary_freqs: torch.Tensor,
        precision_adapter: Any = None,
    ) -> torch.Tensor:
        """``hidden``: normalized input ``[S, 1, H]`` (dispatched tokens); returns ``[S, 1, H]``."""
        self._check_input(hidden)
        if precision_adapter is not None:
            raise NotImplementedError(
                "MSA precision adapters do not apply to dense attention layers"
            )
        if not isinstance(runtime_key, MsaHybridKey):
            raise ValueError("dense attention layers need build_msa_runtime_key(dense_heads=...)")
        q, k, v = self._sparse_qkv(hidden, rotary_freqs)
        calc_attn = _dense_api()[0]
        core, _meta = calc_attn(q, k, v, runtime_key.dense, softmax_scale=self.head_dim**-0.5)
        return self._output(core)


__all__ = [
    "MSA_V1_HEAD_DIM",
    "MSA_V1_INDEX_DIM",
    "MSA_V1_NUM_HEADS",
    "MSA_V1_NUM_INDEX_HEADS",
    "MSA_V1_NUM_KV_HEADS",
    "MsaDenseAttention",
    "MsaHybridKey",
    "MsaV1Attention",
    "MsaV1Numerics",
    "build_msa_runtime_key",
    "msa_dispatch",
    "msa_position_ids",
    "msa_undispatch",
    "msa_v1_rank_kl_coeff",
    "rotary_freqs_for_positions",
]
