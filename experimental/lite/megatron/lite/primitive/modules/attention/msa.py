# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax Sparse Attention (MSA v1) with all-gather context parallelism.

Each query head attends to ``topk_blocks`` 128-token key blocks of its KV group, chosen per token
by a small indexer (block score = max of token scores, the query's own block always kept). The
indexer is trained only by a KL distillation gradient that the backward adds to the indexer
projections; there is no KL term in the loss scalar. The kernels come from the ``msa_v1`` package
of MiniMax MSA (https://github.com/MiniMax-AI/MSA).

Context parallelism (:class:`MsaCpLayout`): the packed row (documents back to back, no padding) is
cut into ``2 * cp_size`` equal chunks and CP rank ``r`` keeps chunks ``r`` and
``2 * cp_size - 1 - r``, so the causal work balances when one document fills the row. Every layer
all-gathers keys, values and indexer keys over CP. This rank's queries form *fragments*, maximal
runs of local tokens inside one document; a fragment attends to the prefix of its document up to
its own last token, so it is the causal suffix of that key range, which always starts at the
document start. Keys are laid out over the rank-major gathered rows in two ways:

* shared prefix (``cu_seqlens_kv``, ``fragment_indices``, ``kv_index``), for the MSA kernels: one
  key range per document, from its start to the end of this rank's last fragment in it. Fragment
  ``i`` ends at ``cu_seqlens_kv[i + 1]`` and starts at ``cu_seqlens_kv[fragment_indices[i]]``, the
  start of its document's first fragment, so the fragments of a document share one copy of the
  keys.
* per fragment (:meth:`MsaCpLayout.fragment_keys`), for the dense (full) attention layers of the
  same model: every fragment gets its own copy of its key prefix and TransformerEngine computes
  bottom-right aligned causal attention (:class:`MsaDenseAttention`).

The backward gathers the keys again rather than keeping them, and reduce-scatters their gradients
in FP32. One :class:`MsaContext` per packed microbatch, built on the host from the document lengths
before the embedding, is shared by every layer, pipeline stage and recompute. Tensors are
sequence-major ``[S, 1, H]``.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive import transformer_engine as te

# The MSA v1 training kernels hard-code these shapes.
MSA_V1_NUM_HEADS = 64
MSA_V1_NUM_KV_HEADS = 4
MSA_V1_HEAD_DIM = 128
MSA_V1_NUM_INDEX_HEADS = 4
MSA_V1_INDEX_DIM = 128
MSA_V1_BLOCK_SIZE = 128


@lru_cache(maxsize=1)
def _msa_v1():
    try:
        import msa_v1
    except (ImportError, OSError) as exc:
        raise ImportError(
            "MSA attention requires the MiniMax MSA v1 kernels (package `minimax-msa`, "
            "https://github.com/MiniMax-AI/MSA)."
        ) from exc
    return msa_v1


@dataclass(frozen=True, eq=False)
class FragmentKeys:
    """Per-fragment key ranges: fragment ``i`` attends to rows ``cu_seqlens_kv[i]:[i + 1]``."""

    cu_seqlens_kv: torch.Tensor
    max_seqlen_kv: int
    # Rows of the rank-major gathered buffer; None: the local rows in their own order.
    index: torch.Tensor | None


@dataclass(frozen=True, eq=False)
class MsaCpLayout:
    """Token placement and attention metadata of one packed microbatch on one CP rank."""

    cp_size: int
    cp_rank: int
    total_tokens: int
    local_tokens: int
    # Global row position and in-document position of each local token, in local order.
    token_index: torch.Tensor
    position_ids: torch.Tensor
    # Query fragments over the local tokens (int32).
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    # Shared-prefix keys (int32 metadata); ``fragment_indices`` is None when no document has two
    # fragments, and ``kv_index`` is None when the keys are the local rows as they are (CP = 1).
    cu_seqlens_kv: torch.Tensor
    fragment_indices: torch.Tensor | None
    max_seqlen_kv: int
    total_kv: int
    kv_index: torch.Tensor | None
    # Host copy of each fragment's key length (from its document start).
    fragment_kv_lens: tuple[int, ...]
    # Host (document start, fragment start, fragment end) of each fragment, global positions.
    _fragments: tuple[tuple[int, int, int], ...] = field(repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def fragment_keys(self) -> FragmentKeys:
        """Key ranges with one copy of the document prefix per fragment."""
        keys = self._cache.get("fragment_keys")
        if keys is not None:
            return keys
        if self.fragment_indices is None:
            keys = FragmentKeys(self.cu_seqlens_kv, self.max_seqlen_kv, self.kv_index)
        else:
            device = self.cu_seqlens_q.device
            positions = torch.cat(
                [torch.arange(doc_start, end) for doc_start, _start, end in self._fragments]
            )
            keys = FragmentKeys(
                cu_seqlens_kv=_cumulative(self.fragment_kv_lens).to(device),
                max_seqlen_kv=max(self.fragment_kv_lens),
                index=_rank_major_rows(positions, self.local_tokens // 2, self.cp_size).to(device),
            )
        self._cache["fragment_keys"] = keys
        return keys


def _cumulative(lengths) -> torch.Tensor:
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int64)
    cu[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int64), 0)
    return cu.to(torch.int32)


def _rank_major_rows(positions: torch.Tensor, chunk: int, cp_size: int) -> torch.Tensor:
    chunk_id = torch.div(positions, chunk, rounding_mode="floor")
    offset = positions - chunk_id * chunk
    back = chunk_id >= cp_size
    owner = torch.where(back, 2 * cp_size - 1 - chunk_id, chunk_id)
    return owner * (2 * chunk) + back.to(positions.dtype) * chunk + offset


def build_msa_cp_layout(
    seq_lens: torch.Tensor, *, cp_size: int, cp_rank: int, device: torch.device | str
) -> MsaCpLayout:
    """Layout of the packed row with document lengths ``seq_lens`` (host tensor) on ``cp_rank``.

    The row length must be a multiple of ``2 * cp_size``; documents may have any length.
    """
    if seq_lens.device.type != "cpu":
        raise ValueError("seq_lens must be a host tensor")
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"invalid CP rank {cp_rank} of {cp_size}")
    lens = [int(n) for n in seq_lens.to(torch.int64).tolist() if int(n) > 0]
    total = sum(lens)
    if total == 0 or total % (2 * cp_size):
        raise ValueError(
            f"all-gather CP needs the packed row length ({total}) to be a positive multiple of "
            f"2 * cp_size ({2 * cp_size})"
        )
    chunk = total // (2 * cp_size)
    doc_ends = list(_cumulative(lens)[1:].tolist())
    owned = (
        (cp_rank * chunk, (cp_rank + 1) * chunk),
        ((2 * cp_size - 1 - cp_rank) * chunk, (2 * cp_size - cp_rank) * chunk),
    )

    # Fragments in local order: owned ranges cut at document ends, joined where they touch.
    docs: list[int] = []
    fragments: list[list[int]] = []  # [document start, start, end]
    for start, end in owned:
        doc = bisect.bisect_right(doc_ends, start)
        while start < end:
            stop = min(end, doc_ends[doc])
            if docs and docs[-1] == doc and fragments[-1][2] == start:
                fragments[-1][2] = stop
            else:
                docs.append(doc)
                fragments.append([doc_ends[doc] - lens[doc], start, stop])
            start = stop
            doc += 1

    # Shared-prefix keys: a document's key range ends where its last local fragment ends.
    kv_lens = [end - doc_start for doc_start, _start, end in fragments]
    cu_kv = [0]
    first_fragment: dict[int, int] = {}
    doc_base: dict[int, int] = {}
    fragment_indices = []
    for i, doc in enumerate(docs):
        if doc not in first_fragment:
            first_fragment[doc] = i
            doc_base[doc] = cu_kv[-1]
        fragment_indices.append(first_fragment[doc])
        cu_kv.append(doc_base[doc] + kv_lens[i])
    last_fragment = {doc: i for i, doc in enumerate(docs)}
    kv_positions = torch.cat(
        [torch.arange(fragments[i][0], fragments[i][2]) for i in sorted(last_fragment.values())]
    )
    shared = any(index != i for i, index in enumerate(fragment_indices))

    token_index = torch.cat([torch.arange(start, end) for start, end in owned])
    position_ids = torch.cat(
        [torch.arange(start - doc_start, end - doc_start) for doc_start, start, end in fragments]
    )
    if cp_size == 1:
        assert torch.equal(kv_positions, torch.arange(total)) and not shared
        kv_index = None
    else:
        kv_index = _rank_major_rows(kv_positions, chunk, cp_size).to(device)
    q_lens = [end - start for _doc_start, start, end in fragments]
    return MsaCpLayout(
        cp_size=cp_size,
        cp_rank=cp_rank,
        total_tokens=total,
        local_tokens=2 * chunk,
        token_index=token_index.to(device),
        position_ids=position_ids.to(device),
        cu_seqlens_q=_cumulative(q_lens).to(device),
        max_seqlen_q=max(q_lens),
        cu_seqlens_kv=torch.tensor(cu_kv, dtype=torch.int32).to(device),
        fragment_indices=(
            torch.tensor(fragment_indices, dtype=torch.int32).to(device) if shared else None
        ),
        max_seqlen_kv=max(kv_lens),
        total_kv=cu_kv[-1],
        kv_index=kv_index,
        fragment_kv_lens=tuple(kv_lens),
        _fragments=tuple(tuple(f) for f in fragments),
    )


def _gather_rows(
    local: torch.Tensor, index: torch.Tensor, group: dist.ProcessGroup | None
) -> torch.Tensor:
    """Rows ``index`` of the rank-major all-gather of ``local`` over ``group`` (no autograd)."""
    size = dist.get_world_size(group)
    gathered = local.new_empty((size * local.shape[0], *local.shape[1:]))
    dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    return gathered.index_select(0, index)


def _reduce_scatter_rows(
    grads: list[torch.Tensor], index: torch.Tensor, group: dist.ProcessGroup | None, local_rows: int
) -> list[torch.Tensor]:
    """Adjoint of :func:`_gather_rows` for several tensors gathered with the same ``index``.

    Each gradient is added into its rows of a zero FP32 rank-major buffer (rows may repeat), the
    buffers are reduce-scattered over ``group`` in one call and every rank gets the FP32 sum for
    its own rows, cast back to the gradient dtype.
    """
    size = dist.get_world_size(group)
    widths = [math.prod(g.shape[1:]) for g in grads]
    full = torch.zeros(
        (size * local_rows, sum(widths)), dtype=torch.float32, device=grads[0].device
    )
    column = 0
    for grad, width in zip(grads, widths):
        full[:, column : column + width].index_add_(0, index, grad.reshape(-1, width).float())
        column += width
    local = full.new_empty((local_rows, sum(widths)))
    dist.reduce_scatter_tensor(local, full, group=group)
    out = []
    column = 0
    for grad, width in zip(grads, widths):
        part = local[:, column : column + width].reshape(local_rows, *grad.shape[1:])
        out.append(part.to(grad.dtype))
        column += width
    return out


class _AllGatherRows(torch.autograd.Function):
    """:func:`_gather_rows` with autograd: the backward is :func:`_reduce_scatter_rows`."""

    @staticmethod
    def forward(ctx, local, index, group):
        ctx.save_for_backward(index)
        ctx.group = group
        ctx.local_rows = local.shape[0]
        return _gather_rows(local, index, group)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        (local_grad,) = _reduce_scatter_rows([grad], index, ctx.group, ctx.local_rows)
        return local_grad, None, None


@dataclass(frozen=True)
class MsaV1Numerics:
    """Numerical settings of the MSA v1 indexer, sparse attention and KL gradient.

    The KL gradient is ``kl_loss_coeff * T**2`` times the mean over (tokens, index
    heads) of KL(teacher || indexer) at temperature ``T = kl_temperature``.
    """

    head_dim: int = MSA_V1_HEAD_DIM
    index_dim: int = MSA_V1_INDEX_DIM
    block_size: int = MSA_V1_BLOCK_SIZE
    topk_blocks: int = 16
    init_blocks: int = 0
    local_blocks: int = 1
    kl_loss_coeff: float = 0.0
    kl_temperature: float = 1.0
    use_fp16_score: bool = False


def msa_v1_rank_kl_coeff(
    numerics: MsaV1Numerics,
    *,
    tp_size: int,
    cp_size: int,
    local_tokens: int,
    total_tokens: int,
    grad_scale: float = 1.0,
) -> float:
    """KL coefficient handed to the KL kernel on one CP rank for one packed microbatch.

    The kernel averages over its own query rows and the index heads; ``local_tokens /
    total_tokens`` turns that into this rank's share of the microbatch mean, ``cp_size`` undoes
    the gradient average over the context-parallel ranks, and ``grad_scale`` carries the runtime's
    auxiliary-loss scale (``1 / num_microbatches``), since the KL gradient is added in backward
    outside the loss scalar.
    """
    rank_coeff = numerics.kl_loss_coeff * (numerics.kl_temperature**2) / tp_size
    return rank_coeff * cp_size * grad_scale * local_tokens / total_tokens


@dataclass(frozen=True, eq=False)
class MsaContext:
    """Per-microbatch MSA state shared by all attention layers: token placement, CP group,
    numerics and this rank's KL coefficient."""

    layout: MsaCpLayout
    cp_group: dist.ProcessGroup | None
    numerics: MsaV1Numerics
    kl_coeff: float
    deterministic: bool
    # Sum over fragments of their key blocks (the kernels' logical K-block rows).
    total_rows: int = field(init=False)

    def __post_init__(self):
        block = self.numerics.block_size
        rows = sum((n + block - 1) // block for n in self.layout.fragment_kv_lens)
        object.__setattr__(self, "total_rows", rows)


def build_msa_context(
    seq_lens: torch.Tensor,
    cp_group: dist.ProcessGroup | None,
    *,
    numerics: MsaV1Numerics,
    tp_size: int = 1,
    grad_scale: float = 1.0,
    deterministic: bool = False,
    device: torch.device | str | None = None,
) -> MsaContext:
    """Build the :class:`MsaContext` of one packed microbatch from host document lengths.

    ``cp_group`` None means no context parallelism. The packed row length must be a multiple of
    ``2 * cp_size``.
    """
    cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else dist.get_rank(cp_group)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    layout = build_msa_cp_layout(seq_lens, cp_size=cp_size, cp_rank=cp_rank, device=device)
    kl_coeff = msa_v1_rank_kl_coeff(
        numerics,
        tp_size=tp_size,
        cp_size=cp_size,
        local_tokens=layout.local_tokens,
        total_tokens=layout.total_tokens,
        grad_scale=grad_scale,
    )
    return MsaContext(
        layout=layout,
        cp_group=cp_group if cp_size > 1 else None,
        numerics=numerics,
        kl_coeff=kl_coeff,
        deterministic=deterministic,
    )


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


def _gather_keys(context: MsaContext, tensors):
    layout = context.layout
    if layout.kv_index is None:
        return list(tensors)
    return [_gather_rows(t, layout.kv_index, context.cp_group) for t in tensors]


def _reduce_key_grads(context: MsaContext, grads):
    layout = context.layout
    if layout.kv_index is None:
        return list(grads)
    return _reduce_scatter_rows(grads, layout.kv_index, context.cp_group, layout.local_tokens)


class _MsaV1Function(torch.autograd.Function):
    """MSA v1 core on local queries and all-gathered keys.

    Inputs are this rank's ``q [S, 64, 128]``, ``k``/``v [S, 4, 128]``, ``index_q [S, 4, 128]``
    and ``index_k [S, 1, 128]``. The indexer and the KL teacher see the inputs detached; the
    indexer inputs get gradients only from the KL term.
    """

    @staticmethod
    def forward(ctx, q, k, v, index_q, index_k, context: MsaContext):
        ops = _msa_v1()
        layout, numerics = context.layout, context.numerics
        need_kl = context.kl_coeff != 0
        q, k, v, index_q, index_k = (t.detach() for t in (q, k, v, index_q, index_k))
        keys, values, index_keys = _gather_keys(context, (k, v, index_k))
        topk, index_lse = ops.indexer.forward(
            index_q,
            index_keys,
            cu_seqlens_q=layout.cu_seqlens_q,
            cu_seqlens_kv=layout.cu_seqlens_kv,
            max_seqlen_q=layout.max_seqlen_q,
            max_seqlen_kv=layout.max_seqlen_kv,
            fragment_indices=layout.fragment_indices,
            lse_temperature=numerics.kl_temperature,
            deterministic=context.deterministic,
            use_fp16_score=numerics.use_fp16_score,
        )
        meta = ops.attention.prepare(
            topk,
            layout.cu_seqlens_q,
            layout.cu_seqlens_kv,
            total_k=layout.total_kv,
            total_rows=context.total_rows,
            max_seqlen_q=layout.max_seqlen_q,
            max_seqlen_k=layout.max_seqlen_kv,
            fragment_indices=layout.fragment_indices,
            prepare_kl_schedule=need_kl,
            deterministic=context.deterministic,
        )
        result = ops.attention.forward(
            q,
            keys,
            values,
            meta,
            softmax_scale=numerics.head_dim**-0.5,
            lse_temperature_scale=numerics.kl_temperature,
            return_softmax_lse=True,
            return_temperature_lse=need_kl,
            partial_dtype=torch.bfloat16,
            deterministic=context.deterministic,
        )
        if need_kl:
            out, lse, temperature_lse = result
        else:
            (out, lse), temperature_lse = result, None
        del keys, values, index_keys
        ctx.save_for_backward(q, k, v, index_q, index_k, out, lse, temperature_lse, index_lse)
        ctx.meta = meta
        ctx.context = context
        return out

    @staticmethod
    def backward(ctx, dout):
        ops = _msa_v1()
        q, k, v, index_q, index_k, out, lse, temperature_lse, index_lse = ctx.saved_tensors
        context: MsaContext = ctx.context
        numerics = context.numerics
        need_kl = context.kl_coeff != 0
        if need_kl:
            keys, values, index_keys = _gather_keys(context, (k, v, index_k))
            temperature = numerics.kl_temperature
            dindex_q, dindex_keys = ops.kl.backward(
                q,
                keys,
                temperature_lse,
                index_q,
                index_keys,
                index_lse,
                ctx.meta,
                softmax_scale=numerics.head_dim**-0.5 / temperature,
                indexer_softmax_scale=numerics.index_dim**-0.5 / temperature,
                loss_coeff=context.kl_coeff,
                deterministic=context.deterministic,
            )
            del index_keys
        else:
            keys, values = _gather_keys(context, (k, v))
        dq, dkeys, dvalues = ops.attention.backward(
            q,
            keys,
            values,
            dout.contiguous(),
            out,
            lse,
            ctx.meta,
            softmax_scale=numerics.head_dim**-0.5,
            deterministic=context.deterministic,
        )
        del keys, values
        if need_kl:
            dk, dv, dindex_k = _reduce_key_grads(context, (dkeys, dvalues, dindex_keys))
        else:
            dk, dv = _reduce_key_grads(context, (dkeys, dvalues))
            dindex_q, dindex_k = torch.zeros_like(index_q), torch.zeros_like(index_k)
        return dq, dk, dv, dindex_q, dindex_k, None


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

    def _qkv(self, hidden: torch.Tensor, freqs: torch.Tensor):
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
        return self.linear_proj(core.reshape(core.shape[0], 1, -1))

    @staticmethod
    def _check_input(hidden: torch.Tensor, context: MsaContext) -> None:
        if hidden.dim() != 3 or hidden.shape[1] != 1:
            raise ValueError(f"expected [S, 1, H] local tokens, got {tuple(hidden.shape)}")
        if hidden.shape[0] != context.layout.local_tokens:
            raise ValueError(
                f"hidden has {hidden.shape[0]} tokens, the MSA context places "
                f"{context.layout.local_tokens} on this rank"
            )


class MsaV1Attention(_GroupedQKVAttention):
    """MSA v1 sparse attention with its block indexer.

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
            raise ValueError(f"MSA v1 kernels require head shapes {expected}, got {shapes}")
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
        self, hidden: torch.Tensor, *, msa_context: MsaContext, rotary_freqs: torch.Tensor
    ) -> torch.Tensor:
        """``hidden``: normalized input ``[S, 1, H]`` (local tokens); returns ``[S, 1, H]``."""
        self._check_input(hidden, msa_context)
        index_q, index_k = self._index_qk(hidden, rotary_freqs)
        q, k, v = self._qkv(hidden, rotary_freqs)
        core = _MsaV1Function.apply(q, k, v, index_q, index_k, msa_context)
        return self._output(core)


class MsaDenseAttention(_GroupedQKVAttention):
    """Dense causal (per-document) attention for the full-attention layers of a model that also
    has MSA layers: the same projections, norms and RoPE and the same token placement, computed
    by TransformerEngine with one copy of the key prefix per query fragment."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, **kwargs)
        self.core_attention = te.DotProductAttention(
            num_attention_heads=self.num_heads,
            kv_channels=self.head_dim,
            num_gqa_groups=self.num_kv_heads,
            attention_dropout=0.0,
            qkv_format="thd",
            attn_mask_type="padding_causal_bottom_right",
            softmax_scale=self.head_dim**-0.5,
        )

    def forward(
        self, hidden: torch.Tensor, *, msa_context: MsaContext, rotary_freqs: torch.Tensor
    ) -> torch.Tensor:
        """``hidden``: normalized input ``[S, 1, H]`` (local tokens); returns ``[S, 1, H]``."""
        self._check_input(hidden, msa_context)
        layout = msa_context.layout
        q, k, v = self._qkv(hidden, rotary_freqs)
        keys = layout.fragment_keys()
        if keys.index is not None:
            k = _AllGatherRows.apply(k, keys.index, msa_context.cp_group)
            v = _AllGatherRows.apply(v, keys.index, msa_context.cp_group)
        core = self.core_attention(
            q,
            k,
            v,
            qkv_format="thd",
            attn_mask_type="padding_causal_bottom_right",
            cu_seqlens_q=layout.cu_seqlens_q,
            cu_seqlens_kv=keys.cu_seqlens_kv,
            max_seqlen_q=layout.max_seqlen_q,
            max_seqlen_kv=keys.max_seqlen_kv,
        )
        return self._output(core)


__all__ = [
    "MSA_V1_BLOCK_SIZE",
    "MSA_V1_HEAD_DIM",
    "MSA_V1_INDEX_DIM",
    "MSA_V1_NUM_HEADS",
    "MSA_V1_NUM_INDEX_HEADS",
    "MSA_V1_NUM_KV_HEADS",
    "MsaContext",
    "MsaCpLayout",
    "MsaDenseAttention",
    "MsaV1Attention",
    "MsaV1Numerics",
    "build_msa_context",
    "msa_v1_rank_kl_coeff",
    "rotary_freqs_for_positions",
]
