# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix forest attention composed from FlashAttention varlen passes.

A forest packs stars ``[(token_offset, prompt_len, completion_lens), ...]`` contiguously from
token 0, each as ``[prompt, completion_1, ..., completion_G]``. A prompt token attends the
prompt causally; a completion token attends the whole prompt and its own completion causally.

:func:`flash_composed_forest_attention` runs at most two FlashAttention varlen passes per
forest and merges them by online softmax (log-sum-exp):

- a causal self pass with one sequence per prompt joined with its first completion, and one
  sequence per other completion;
- a non-causal cross pass in which completions 2..G of every star attend their prompt.

Forward and backward are exact (see :class:`_ComposedForestAttn`). Under
``torch.use_deterministic_algorithms(True)``, which ``--deterministic-mode`` enables, the
backward uses FlashAttention's deterministic kernel. :func:`flash_composed_forest_attention_cp`
runs the same primitive on zigzag context-parallel shards.
"""

from itertools import accumulate

import torch

from megatron.core.tensor_parallel.mappings import all_to_all_hp2sp, all_to_all_sp2hp

# Cache pass plans because the same packed layout recurs across model layers.
_PLAN_CACHE: dict = {}
_PLAN_CACHE_MAX = 128


def _gather_kv(k, v, k_idx):
    """Gather strided interleaved K/V projections into independent contiguous tensors."""
    return k.index_select(0, k_idx), v.index_select(0, k_idx)


def _forest_key(forest):
    """Normalize a forest to a hashable ``((offset, prompt_len, completion_lens), ...)`` tuple."""
    return tuple(
        (int(offset), int(prompt_len), tuple(int(length) for length in completion_lens))
        for offset, prompt_len, completion_lens in forest
    )


def _star_forest_plan(forest, device):
    """Decompose a normalized star forest into the FlashAttention passes it runs.

    Returns ``(total, passes)``. Each pass is ``(rows, k_idx, cu_q, cu_k, max_q, max_k, causal)``:
    query rows ``q[rows]`` attend the keys ``k`` (``k_idx is None``) or ``k[k_idx]`` as varlen
    sequences.

    - Self pass (causal, every row): the prompt and its first completion are adjacent, so they
      form one causal sequence; every other completion is its own causal sequence.
    - Cross pass (non-causal; only when a star has two or more completions): completions 2..G
      attend their prompt. Its query rows are the zero-copy view ``q[lo:hi]``; rows in that range
      outside these completions form zero-length-K padding sequences, for which FlashAttention
      returns zero output and gradient and LSE ``+inf``.
    """
    if not forest:
        raise ValueError("shared-prefix attention requires at least one star")
    self_lens, cross = [], []  # cross: (q0, q1, k0, k1) = completion rows and their prompt rows
    end = 0
    for index, (offset, prompt_len, completion_lens) in enumerate(forest):
        if offset != end:
            raise ValueError(
                f"shared-prefix star {index} starts at token {offset}; stars must be stored "
                f"contiguously from token 0 (expected {end})"
            )
        if prompt_len < 1 or any(length < 1 for length in completion_lens):
            raise ValueError(f"shared-prefix star {index} has an empty prompt or completion")
        head = prompt_len + sum(completion_lens[:1])
        self_lens.append(head)
        start = offset + head
        for length in completion_lens[1:]:
            self_lens.append(length)
            cross.append((start, start + length, offset, offset + prompt_len))
            start += length
        end = start

    cu_self = [0, *accumulate(self_lens)]
    cu_q, cu_k, max_q = [0], [0], 0
    cursor = cross[0][0] if cross else 0
    for q0, q1, k0, k1 in cross:
        if q0 > cursor:  # rows inside the view that attend nothing in this pass
            cu_q.append(cu_q[-1] + q0 - cursor)
            cu_k.append(cu_k[-1])
            max_q = max(max_q, q0 - cursor)
        cu_q.append(cu_q[-1] + q1 - q0)
        cu_k.append(cu_k[-1] + k1 - k0)
        max_q = max(max_q, q1 - q0)
        cursor = q1

    cu = torch.tensor(cu_self + cu_q + cu_k, dtype=torch.int32, device=device)
    self_cu = cu[: len(cu_self)]
    max_self = max(self_lens)
    passes = [(slice(0, end), None, self_cu, self_cu, max_self, max_self, True)]
    if cross:
        k_idx = torch.cat([torch.arange(k0, k1, device=device) for _, _, k0, k1 in cross])
        cross_cu_q = cu[len(cu_self) : len(cu_self) + len(cu_q)]
        cross_cu_k = cu[len(cu_self) + len(cu_q) :]
        max_k = max(k1 - k0 for _, _, k0, k1 in cross)
        rows = slice(cross[0][0], cross[-1][1])
        passes.append((rows, k_idx, cross_cu_q, cross_cu_k, max_q, max_k, False))
    return end, passes


def _star_forest_plan_cached(forest, device):
    """Return ``(total, passes)`` for ``forest``, reusing plans across layers."""
    key = (_forest_key(forest), str(device))
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _star_forest_plan(key[0], torch.device(device))
        if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
        _PLAN_CACHE[key] = plan
    return plan


class _ComposedForestAttn(torch.autograd.Function):
    """Composed forest attention with an EXACT backward.

    The forward runs the plan's passes with flash and merges them by online softmax (LSE), the
    union-softmax identity, so the forward is exact. In the backward, each pass's flash output is
    a sub-attention, and naive autograd through flash drops the inter-pass normalizer-coupling
    term (flash exposes no gradient through its LSE), giving ~15% wrong q/k grads. Instead the
    low-level flash backward runs per pass with ``dout = w_pass * do`` and the MERGED output
    ``o`` in place of the pass's own output: flash uses ``out`` only to form the row delta
    ``D = rowsum(dout * out)``, which is exactly the softmax ``G`` term, so the merged ``o``
    injects the missing global normalizer. The result is the exact union-softmax score gradient
    ``P_ij (v_j . do - o . do)`` for every pass, so dq, dk and dv are all exact."""

    @staticmethod
    def forward(ctx, q, k, v, passes, scale):
        """Evaluate the causal self pass and the prompt cross pass, then merge their outputs."""
        # q: [total, np, hn]; k, v: [total, ng, hn]; scale already resolved to a float.
        from flash_attn import flash_attn_varlen_func

        total, np_, hn = q.shape
        outs, lses = [], []
        for rows, k_idx, cu_q, cu_k, max_q, max_k, causal in passes:
            kx, vx = (k, v) if k_idx is None else _gather_kv(k, v, k_idx)
            o, lse, _ = flash_attn_varlen_func(
                q[rows],
                kx,
                vx,
                cu_q,
                cu_k,
                max_q,
                max_k,
                softmax_scale=scale,
                causal=causal,
                return_attn_probs=True,
            )  # o [rows, np, hn], lse [np, rows]
            outs.append(o)
            lses.append(lse)

        # Zero-length-K padding rows report LSE=+inf: give them weight 0 in the merge.
        merge_lses = [lse.float().masked_fill(torch.isinf(lse), float("-inf")) for lse in lses]
        lse_final = torch.full((np_, total), float("-inf"), device=q.device, dtype=torch.float32)
        for (rows, *_), lse in zip(passes, merge_lses):
            lse_final[:, rows] = torch.logaddexp(lse_final[:, rows], lse)
        # merged output: sum_pass w_pass * o_pass, w_pass = exp(lse_pass - lse_final).
        o_merged = torch.zeros(total, np_, hn, device=q.device, dtype=torch.float32)
        for (rows, *_), o, lse in zip(passes, outs, merge_lses):
            contrib = torch.exp(lse - lse_final[:, rows]).transpose(0, 1).unsqueeze(-1) * o.float()
            o_merged[rows] += contrib
        o_merged = o_merged.to(q.dtype)

        ctx.save_for_backward(q, k, v, o_merged)
        ctx.passes = passes
        ctx.lses = lses
        ctx.lse_final = lse_final
        ctx.scale = scale
        return o_merged

    @staticmethod
    def backward(ctx, do):
        """Accumulate pass gradients using the merged output for the global softmax correction."""
        from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

        q, k, v, o_merged = ctx.saved_tensors
        lse_final = ctx.lse_final
        do = do.contiguous()
        deterministic = torch.are_deterministic_algorithms_enabled()
        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
        for (rows, k_idx, cu_q, cu_k, max_q, max_k, causal), lse in zip(ctx.passes, ctx.lses):
            qx = q[rows]
            kx, vx = (k, v) if k_idx is None else _gather_kv(k, v, k_idx)
            weight = torch.exp(
                lse.float().masked_fill(torch.isinf(lse), float("-inf")) - lse_final[:, rows]
            )  # 0 on zero-K padding rows
            dox = (weight.transpose(0, 1).unsqueeze(-1) * do[rows]).to(q.dtype)
            dqx, dkx, dvx = torch.empty_like(qx), torch.empty_like(kx), torch.empty_like(vx)
            _flash_attn_varlen_backward(
                dox,
                qx,
                kx,
                vx,
                o_merged[rows],  # MERGED output -> exact
                lse,
                dqx,
                dkx,
                dvx,
                cu_q,
                cu_k,
                max_q,
                max_k,
                0.0,
                ctx.scale,
                causal,
                -1,
                -1,
                0.0,
                None,
                deterministic,
                None,
                False,
            )
            dq[rows] += dqx.float()
            if k_idx is None:
                dk += dkx.float()
                dv += dvx.float()
            else:
                dk.index_add_(0, k_idx, dkx.float())
                dv.index_add_(0, k_idx, dvx.float())
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def flash_composed_forest_attention(query, key, value, forest, scale=None):
    """Exact shared-prefix attention over a packed forest of stars.

    ``forest`` is a list of ``(token_offset, prompt_len, completion_lens)``, one per star, stored
    contiguously from token 0. All stars run in the same (at most two) flash passes regardless of
    their count. ``query`` is ``[sq, 1, np, hn]`` and ``key``/``value`` are ``[sq, 1, ng, hn]``;
    trailing rows past the forest get zero outputs. Returns ``[sq, 1, np * hn]``. ``scale``
    defaults to ``1 / sqrt(hn)``.
    """
    sq, b, np_, hn = query.shape
    assert b == 1, "shared-prefix packing uses a single packed sequence (b == 1)"
    total, passes = _star_forest_plan_cached(forest, query.device)
    scale = scale if scale is not None else hn**-0.5
    out = _ComposedForestAttn.apply(
        query[:total, 0], key[:total, 0], value[:total, 0], passes, scale
    )  # [total, np, hn]
    if sq > total:
        out = torch.cat([out, out.new_zeros(sq - total, np_, hn)], dim=0)
    return out.reshape(sq, 1, np_ * hn).contiguous()


def _undo_cp_zigzag(input_: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Convert rank-major CP zigzag chunks into canonical global token order.

    Standard context parallelism gives rank ``r`` chunks ``(r, 2*C-r-1)``.  An
    all-to-all from sequence to head parallelism concatenates those rank-local pairs in
    rank order, i.e. ``0,last,1,last-1,...``.  Mamba uses the same permutation before
    its sequential scan; forest attention needs it for its global node offsets too.
    """
    if cp_size < 1 or input_.shape[0] % (2 * cp_size):
        raise ValueError("CP zigzag sequence length must be divisible by 2 * CP size")
    chunks = torch.chunk(input_, chunks=2 * cp_size, dim=0)
    order = [2 * index for index in range(cp_size)] + [
        2 * cp_size - 2 * index - 1 for index in range(cp_size)
    ]
    return torch.cat([chunks[index] for index in order], dim=0)


def _redo_cp_zigzag(input_: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Convert canonical global token order back to rank-major CP zigzag chunks."""
    if cp_size < 1 or input_.shape[0] % (2 * cp_size):
        raise ValueError("CP zigzag sequence length must be divisible by 2 * CP size")
    chunks = torch.chunk(input_, chunks=2 * cp_size, dim=0)
    order = [None] * (2 * cp_size)
    order[::2] = range(cp_size)
    order[1::2] = reversed(range(cp_size, 2 * cp_size))
    return torch.cat([chunks[index] for index in order], dim=0)


def _cp_local_kv_head_slice(query_heads: int, kv_heads: int, cp_size: int, cp_rank: int) -> slice:
    """Return the global KV-head span serving one CP rank's contiguous Q-head span.

    The fused flash primitive supports GQA, but its local Q-to-KV grouping must exactly
    match the global grouping after Q heads are sharded over CP.  Some valid layouts
    (including 32 Q heads / 2 KV heads / CP4) replicate a single KV head on multiple CP
    ranks.  Layouts whose CP boundary cuts unequal pieces from several KV groups fail
    closed rather than silently changing attention semantics.
    """
    if min(query_heads, kv_heads, cp_size) < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(
            f"invalid shared-prefix CP head geometry: {query_heads=}, {kv_heads=}, "
            f"{cp_size=}, {cp_rank=}"
        )
    if query_heads % cp_size:
        raise NotImplementedError(
            f"shared-prefix CP attention requires {query_heads=} divisible by {cp_size=}"
        )
    if query_heads % kv_heads:
        raise NotImplementedError(
            "shared-prefix CP attention requires an integral global Q/KV head ratio"
        )

    local_query_heads = query_heads // cp_size
    queries_per_kv = query_heads // kv_heads
    query_start = cp_rank * local_query_heads
    global_mapping = [
        query_index // queries_per_kv
        for query_index in range(query_start, query_start + local_query_heads)
    ]
    kv_start = global_mapping[0]
    kv_stop = global_mapping[-1] + 1
    local_kv_heads = kv_stop - kv_start
    if local_query_heads % local_kv_heads:
        raise NotImplementedError(
            "shared-prefix CP attention cannot express this Q/KV grouping with local flash GQA"
        )
    local_queries_per_kv = local_query_heads // local_kv_heads
    expected_mapping = [
        kv_start + local_index // local_queries_per_kv for local_index in range(local_query_heads)
    ]
    if global_mapping != expected_mapping:
        raise NotImplementedError(
            "shared-prefix CP attention boundary cuts unequal portions of multiple KV groups"
        )
    return slice(kv_start, kv_stop)


def _cp_kv_head_slices_for_destinations(
    query_heads: int, kv_heads: int, cp_size: int
) -> tuple[slice, ...]:
    """Return equal-width KV-head blocks in all-to-all destination order.

    Each destination owns one contiguous Q-head block.  Its KV block can overlap
    another destination's block for GQA, but the equal-split all-to-all requires
    every destination block to contain the same number of heads.
    """
    destination_slices = tuple(
        _cp_local_kv_head_slice(query_heads, kv_heads, cp_size, cp_rank)
        for cp_rank in range(cp_size)
    )
    destination_widths = tuple(
        head_slice.stop - head_slice.start for head_slice in destination_slices
    )
    if len(set(destination_widths)) != 1:
        raise NotImplementedError(
            "shared-prefix CP attention requires equal KV-head widths for every "
            f"all-to-all destination; got {destination_widths}"
        )
    return destination_slices


def _cp_pack_destination_head_slices(
    input_: torch.Tensor, destination_slices: tuple[slice, ...]
) -> torch.Tensor:
    """Pack destination-specific head blocks before an equal-split all-to-all.

    Overlapping slices are intentionally concatenated more than once.  Autograd
    accumulates their backward contributions into the shared source heads, which
    preserves GQA KV-gradient multiplicity without materializing every KV head on
    every destination.
    """
    if not destination_slices:
        raise ValueError("shared-prefix CP attention requires at least one destination")
    head_count = input_.shape[2]
    widths = []
    for head_slice in destination_slices:
        if (
            head_slice.step not in (None, 1)
            or head_slice.start is None
            or head_slice.stop is None
            or not 0 <= head_slice.start < head_slice.stop <= head_count
        ):
            raise ValueError(
                "shared-prefix CP attention destination slices must be non-empty, "
                "unit-stride, and within the KV-head dimension"
            )
        widths.append(head_slice.stop - head_slice.start)
    if len(set(widths)) != 1:
        raise NotImplementedError(
            "shared-prefix CP attention requires equal KV-head widths for every "
            f"all-to-all destination; got {tuple(widths)}"
        )
    return torch.cat([input_[:, :, head_slice, :] for head_slice in destination_slices], dim=2)


def _cp_sequence_to_head_parallel(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    *,
    destination_head_slices: tuple[slice, ...] | None = None,
) -> torch.Tensor:
    """Change ``[S/C,1,H,D]`` zigzag sequence shards to canonical head shards."""
    if input_.ndim != 4 or input_.shape[1] != 1:
        raise ValueError("shared-prefix CP attention requires Q/K/V shape [S/C,1,H,D]")
    cp_size = cp_group.size()
    if input_.shape[0] % 2:
        raise ValueError(
            "shared-prefix CP attention requires an even local sequence length for zigzag CP"
        )
    if destination_head_slices is not None:
        if len(destination_head_slices) != cp_size:
            raise ValueError(
                "shared-prefix CP attention requires one head slice per all-to-all destination"
            )
        input_ = _cp_pack_destination_head_slices(input_, destination_head_slices)
    elif input_.shape[2] % cp_size:
        raise NotImplementedError(
            "shared-prefix CP attention requires Q heads divisible by context parallel size"
        )

    local_sequence, batch, heads, head_dim = input_.shape
    exchanged = all_to_all_sp2hp(
        input_.reshape(local_sequence, batch, heads * head_dim), group=cp_group
    )
    global_sequence = local_sequence * cp_size
    local_heads = heads // cp_size
    exchanged = exchanged.reshape(global_sequence, batch, local_heads, head_dim)
    return _undo_cp_zigzag(exchanged, cp_size)


def _cp_head_to_sequence_parallel(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Invert :func:`_cp_sequence_to_head_parallel` for attention output."""
    if input_.ndim != 4 or input_.shape[1] != 1:
        raise ValueError("shared-prefix CP attention output must have shape [S,1,H/C,D]")
    cp_size = cp_group.size()
    global_sequence, batch, local_heads, head_dim = input_.shape
    if global_sequence % (2 * cp_size):
        raise ValueError("shared-prefix CP global sequence length must be divisible by 2 * CP size")
    rank_major = _redo_cp_zigzag(input_, cp_size)
    exchanged = all_to_all_hp2sp(
        rank_major.reshape(global_sequence, batch, local_heads * head_dim), group=cp_group
    )
    return exchanged.reshape(global_sequence // cp_size, batch, local_heads * cp_size, head_dim)


def flash_composed_forest_attention_cp(
    query, key, value, forest, *, cp_group: torch.distributed.ProcessGroup, scale=None
):
    """Exact forest attention for standard zigzag context-parallel sequence shards.

    Q heads are sharded across CP while each destination receives only its required
    K/V heads. Overlapping GQA slices remain differentiably replicated. All tensors
    are converted to canonical global sequence order, then the existing optimized
    exact-forward/exact-backward forest primitive runs on the local head shard. Its
    output is transformed back to the caller's CP-local zigzag token order. No global
    logits, hidden states, or full-KV activation bases are materialized.
    """
    cp_size = cp_group.size()
    if cp_size == 1:
        return flash_composed_forest_attention(query, key, value, forest, scale=scale)
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("shared-prefix CP attention requires Q/K/V rank-4 tensors")
    if any(tensor.shape[1] != 1 for tensor in (query, key, value)):
        raise ValueError("shared-prefix CP attention requires Q/K/V batch size 1")
    if not query.shape[0] == key.shape[0] == value.shape[0]:
        raise ValueError("shared-prefix CP attention requires aligned local Q/K/V sequences")
    if key.shape[2] != value.shape[2]:
        raise ValueError("shared-prefix CP attention requires equal K/V head counts")
    if not query.shape[3] == key.shape[3] == value.shape[3]:
        raise ValueError("shared-prefix CP attention requires equal Q/K/V head dimensions")
    if not query.dtype == key.dtype == value.dtype:
        raise ValueError("shared-prefix CP attention requires equal Q/K/V dtypes")
    if not query.device == key.device == value.device:
        raise ValueError("shared-prefix CP attention requires Q/K/V on the same device")

    query_heads = query.shape[2]
    kv_heads = key.shape[2]
    destination_kv_slices = _cp_kv_head_slices_for_destinations(query_heads, kv_heads, cp_size)
    query_global = _cp_sequence_to_head_parallel(query, cp_group)
    key_global = _cp_sequence_to_head_parallel(
        key, cp_group, destination_head_slices=destination_kv_slices
    )
    value_global = _cp_sequence_to_head_parallel(
        value, cp_group, destination_head_slices=destination_kv_slices
    )

    output = flash_composed_forest_attention(
        query_global, key_global, value_global, forest, scale=scale
    )
    output = output.reshape(query_global.shape[0], 1, query_global.shape[2], query_global.shape[3])
    output = _cp_head_to_sequence_parallel(output, cp_group)
    return output.reshape(output.shape[0], 1, -1).contiguous()
