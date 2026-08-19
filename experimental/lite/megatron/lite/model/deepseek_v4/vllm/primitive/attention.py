from __future__ import annotations

from typing import Any

import torch

from ._contract import own_visible_tensor


def _rope_and_qnorm(value, positions, cache, rope_dim, eps, *, normalize):
    x = value.float()
    if normalize:
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    prefix, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    selected = cache.index_select(0, positions.long()).float()
    cos = selected[..., : rope_dim // 2]
    sin = selected[..., rope_dim // 2 : rope_dim]
    while cos.ndim < rope.ndim:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    even, odd = rope[..., 0::2], rope[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return torch.cat((prefix, rotated.flatten(-2)), dim=-1).to(value.dtype)


def _default_sparse_backward(q, kv, out, grad_out, lse, sink, indices, scale, length):
    from megatron.lite.primitive.kernels import dsa_kernels

    # The pinned DSA SM90 overlay still uses the pre-4.6 type annotation
    # ``cute.core.ThrMma``.  CUTLASS 4.6 keeps the same class at
    # ``cute.ThrMma`` but removed only that compatibility re-export.  Restore
    # the alias locally before the overlay lazily imports its kernel modules;
    # no runtime object or kernel behavior is changed.
    import cutlass.cute as cute
    import cutlass.cute.core as cute_core

    if not hasattr(cute_core, "ThrMma"):
        cute_core.ThrMma = cute.ThrMma
    dsa_kernels._ensure_dsa_namespace()
    # vLLM keeps the prefill workspace request-major even for batch=1
    # ([S_kv, 1, D]), while the cuDNN DSA backward API consumes the flattened
    # logical KV sequence ([total_S_kv, D]).  The query/output side is already
    # token-major.  Flatten only the singleton/request dimension here so the
    # visible forward layout remains untouched.
    if kv.ndim > 2:
        kv = kv.reshape(-1, kv.shape[-1])
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0]
    result = dsa_kernels._DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        grad_out,
        lse,
        sink,
        indices,
        softmax_scale=scale,
        topk_length=length,
    )
    return result["dq"], result["dkv"]


def _compressed_sequence_graph(
    kv_score,
    ape,
    norm_weight,
    positions,
    cache,
    *,
    ratio,
    head_dim,
    rope_dim,
    eps,
):
    blocks = kv_score.shape[0] // ratio
    if blocks == 0:
        return kv_score.new_empty((0, head_dim))
    cutoff = blocks * ratio
    coff = 2 if ratio == 4 else 1
    width = coff * head_dim
    content, gate = kv_score[:cutoff].split((width, width), dim=-1)
    content = content.view(blocks, ratio, coff, head_dim)
    gate = gate.view_as(content) + ape.view(1, ratio, coff, head_dim)
    if ratio == 4:
        expanded_content = content.new_zeros((blocks, 2 * ratio, head_dim))
        expanded_gate = gate.new_full((blocks, 2 * ratio, head_dim), float("-inf"))
        expanded_content[:, ratio:] = content[:, :, 1]
        expanded_gate[:, ratio:] = gate[:, :, 1]
        if blocks > 1:
            expanded_content[1:, :ratio] = content[:-1, :, 0]
            expanded_gate[1:, :ratio] = gate[:-1, :, 0]
        content, gate = expanded_content, expanded_gate
    else:
        content, gate = content.squeeze(2), gate.squeeze(2)
    weights = torch.softmax(gate.float(), dim=1).to(content.dtype)
    compressed = (content * weights).sum(dim=1).float()
    compressed = compressed * torch.rsqrt(
        compressed.square().mean(dim=-1, keepdim=True) + eps
    )
    compressed = compressed * norm_weight.float()
    return _rope_and_qnorm(
        compressed.to(kv_score.dtype),
        positions[:cutoff:ratio],
        cache,
        min(rope_dim, head_dim),
        eps,
        normalize=False,
    )


def _compressed_sequence_graph_packed(
    kv_score,
    ape,
    norm_weight,
    positions,
    cache,
    query_start_loc,
    **kwargs,
):
    starts = query_start_loc.detach().cpu().tolist()
    parts = [
        _compressed_sequence_graph(
            kv_score[start:end],
            ape,
            norm_weight,
            positions[start:end],
            cache,
            **kwargs,
        )
        for start, end in zip(starts[:-1], starts[1:], strict=True)
    ]
    return torch.cat(parts) if parts else kv_score.new_empty((0, kwargs["head_dim"]))


def compressed_compact_graph(
    kv_score,
    ape,
    norm_weight,
    compressed_group_ids,
    cache,
    *,
    ratio,
    head_dim,
    rope_dim,
    eps,
):
    groups = compressed_group_ids.numel()
    if groups == 0:
        return kv_score.new_empty((0, head_dim))
    coff = 2 if ratio == 4 else 1
    width = coff * head_dim
    content, gate = kv_score.split((width, width), dim=-1)
    content = content.view(groups, ratio, coff, head_dim)
    gate = gate.view_as(content) + ape.view(1, ratio, coff, head_dim)
    if ratio == 4:
        expanded_content = content.new_zeros((groups, 2 * ratio, head_dim))
        expanded_gate = gate.new_full((groups, 2 * ratio, head_dim), float("-inf"))
        expanded_content[:, ratio:] = content[:, :, 1]
        expanded_gate[:, ratio:] = gate[:, :, 1]
        previous_valid = compressed_group_ids != 0
        if groups > 1:
            expanded_content[1:, :ratio] = torch.where(
                previous_valid[1:, None, None], content[:-1, :, 0], 0
            )
            expanded_gate[1:, :ratio] = torch.where(
                previous_valid[1:, None, None],
                gate[:-1, :, 0],
                torch.full_like(gate[:-1, :, 0], float("-inf")),
            )
        content, gate = expanded_content, expanded_gate
    else:
        content, gate = content.squeeze(2), gate.squeeze(2)
    weights = torch.softmax(gate.float(), dim=1).to(content.dtype)
    compressed = (content * weights).sum(dim=1).float()
    compressed = compressed * torch.rsqrt(
        compressed.square().mean(dim=-1, keepdim=True) + eps
    )
    compressed = compressed * norm_weight.float()
    positions = compressed_group_ids.clamp_min(0).long() * ratio
    return _rope_and_qnorm(
        compressed.to(kv_score.dtype),
        positions,
        cache,
        min(rope_dim, head_dim),
        eps,
        normalize=False,
    ).to(norm_weight.dtype)


def attach_indexer_aux_loss(
    output,
    q,
    index_q,
    index_kv_score,
    index_weights,
    main_kv_score,
    index_ape,
    index_norm,
    main_ape,
    main_norm,
    positions,
    cos_sin_cache,
    topk_indices,
    *,
    ratio,
    rope_dim,
    eps,
    softmax_scale,
    loss_coeff,
):
    """Attach the established sparse IndexShare loss to visible vLLM output.

    The visible indexer owns selection.  This graph replays exactly those
    indices and differentiates only the scores on that fixed active set.
    Attention query/KV targets are detached, matching Lite DSA semantics.
    """
    from megatron.lite.primitive.kernels.dsa_kernels import cp_indexer_loss
    from megatron.lite.primitive.modules.attention.dsa import (
        DSAIndexerLossAutoScaler,
    )

    index_k = _compressed_sequence_graph(
        index_kv_score,
        index_ape,
        index_norm,
        positions,
        cos_sin_cache,
        ratio=ratio,
        head_dim=index_q.shape[-1],
        rope_dim=rope_dim,
        eps=eps,
    )
    main_k = _compressed_sequence_graph(
        main_kv_score,
        main_ape,
        main_norm,
        positions,
        cos_sin_cache,
        ratio=ratio,
        head_dim=q.shape[-1],
        rope_dim=rope_dim,
        eps=eps,
    ).detach()
    index_q_visible = _rope_and_qnorm(
        index_q,
        positions,
        cos_sin_cache,
        rope_dim,
        eps,
        normalize=False,
    )
    q_target = _rope_and_qnorm(
        q,
        positions,
        cos_sin_cache,
        rope_dim,
        eps,
        normalize=True,
    ).detach()

    fixed_topk = topk_indices.detach()
    if fixed_topk.ndim == 2:
        fixed_topk = fixed_topk.unsqueeze(0)
    elif fixed_topk.ndim == 3 and fixed_topk.shape[1] == 1:
        fixed_topk = fixed_topk[:, 0].unsqueeze(0)
    if fixed_topk.ndim != 3 or fixed_topk.shape[0] != 1:
        raise ValueError(
            "DS4 vLLM indexer auxiliary loss currently requires one request; "
            f"got topk shape {tuple(topk_indices.shape)}"
        )
    if fixed_topk.shape[1] != q.shape[0]:
        raise ValueError("indexer top-k query rows do not match attention query rows")

    query_rows = torch.arange(q.shape[0], device=q.device)
    key_rows = torch.arange(index_k.shape[0], device=q.device)
    valid_counts = (query_rows + 1).div(ratio, rounding_mode="floor")
    mask = torch.where(
        key_rows.unsqueeze(0) < valid_counts.unsqueeze(1),
        torch.zeros((), device=q.device, dtype=torch.float32),
        torch.full((), float("-inf"), device=q.device, dtype=torch.float32),
    )
    # Lite's indexer loss expects weights pre-scaled by H^-1/2; it applies
    # D^-1/2 internally to the QK score.
    scaled_weights = index_weights * (index_q.shape[1] ** -0.5)
    indexer_loss = cp_indexer_loss(
        index_q_visible.unsqueeze(1),
        index_k.unsqueeze(1),
        scaled_weights.unsqueeze(1),
        fixed_topk,
        q_target.unsqueeze(1),
        main_k.unsqueeze(1),
        mask=mask,
        softmax_scale=softmax_scale,
        loss_coeff=loss_coeff,
        sparse_loss=True,
        calculate_per_token_loss=False,
    )
    return DSAIndexerLossAutoScaler.apply(output, indexer_loss)


class _VLLMAttentionCoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        visible_op,
        backward_op,
        scale,
        eps,
        rope_dim,
        q,
        kv,
        workspace,
        indices,
        topk_length,
        sink,
        slots,
        positions,
        cos_sin_cache,
        compressor_kv_score,
        compressor_ape,
        compressor_norm,
        compressor_ratio,
        compressor_workspace_slots,
        query_start_loc,
    ):
        (
            out,
            lse,
            q_visible,
            workspace_visible,
            indices_visible,
            topk_length_visible,
        ) = visible_op(q, kv)
        ctx.save_for_backward(
            q,
            kv,
            q_visible,
            workspace_visible,
            indices_visible,
            topk_length_visible,
            sink,
            slots,
            positions,
            cos_sin_cache,
            out,
            lse,
            compressor_kv_score,
            compressor_ape,
            compressor_norm,
            compressor_workspace_slots,
            query_start_loc,
        )
        ctx.backward_op = backward_op or _default_sparse_backward
        ctx.scale, ctx.eps, ctx.rope_dim = scale, eps, rope_dim
        ctx.compressor_ratio = compressor_ratio
        return own_visible_tensor(out)

    @staticmethod
    def backward(ctx: Any, grad_out):
        (
            q,
            kv,
            q_visible,
            workspace,
            indices,
            topk_length,
            sink,
            slots,
            positions,
            cache,
            out,
            lse,
            compressor_kv_score,
            compressor_ape,
            compressor_norm,
            compressor_workspace_slots,
            query_start_loc,
        ) = ctx.saved_tensors
        dq_visible, dworkspace = ctx.backward_op(
            q_visible,
            workspace,
            out,
            grad_out,
            lse,
            sink,
            indices,
            ctx.scale,
            topk_length,
        )
        dq_finite = bool(torch.isfinite(dq_visible).all())
        dworkspace_finite = bool(torch.isfinite(dworkspace).all())
        if not dq_finite or not dworkspace_finite:
            raise FloatingPointError(
                "FlashMLA sparse backward produced non-finite gradients: "
                f"compressor_ratio={ctx.compressor_ratio}, "
                f"dq_finite={dq_finite}, dkv_finite={dworkspace_finite}"
            )
        flat_dkv = dworkspace.reshape(-1, dworkspace.shape[-1])
        valid = slots >= 0
        compressed_len = compressor_workspace_slots.numel()
        safe_slots = slots.clamp_min(0).long()
        dkv_visible = flat_dkv.index_select(0, safe_slots)
        dkv_visible = dkv_visible * valid.unsqueeze(-1)
        with torch.enable_grad():
            q_replay = q.detach().requires_grad_(True)
            kv_replay = kv.detach().requires_grad_(True)
            q_functional = _rope_and_qnorm(
                q_replay,
                positions,
                cache,
                ctx.rope_dim,
                ctx.eps,
                normalize=True,
            )
            kv_functional = _rope_and_qnorm(
                kv_replay,
                positions,
                cache,
                ctx.rope_dim,
                ctx.eps,
                normalize=False,
            )
            dq, dkv = torch.autograd.grad(
                (q_functional, kv_functional),
                (q_replay, kv_replay),
                (dq_visible, dkv_visible),
            )
        dcompressor = dape = dcompressor_norm = None
        if ctx.compressor_ratio > 1 and compressed_len:
            with torch.enable_grad():
                replay_inputs = tuple(
                    value.detach().requires_grad_(True)
                    for value in (
                        compressor_kv_score,
                        compressor_ape,
                        compressor_norm,
                    )
                )
                compressed = _compressed_sequence_graph_packed(
                    *replay_inputs,
                    positions,
                    cache,
                    query_start_loc,
                    ratio=ctx.compressor_ratio,
                    head_dim=dworkspace.shape[-1],
                    rope_dim=ctx.rope_dim,
                    eps=ctx.eps,
                )
                dcompressor, dape, dcompressor_norm = torch.autograd.grad(
                    compressed,
                    replay_inputs,
                    dworkspace.reshape(-1, dworkspace.shape[-1]).index_select(
                        0, compressor_workspace_slots.long()
                    ),
                )
        return (
            (None,) * 5
            + (dq, dkv)
            + (None,) * 7
            + (dcompressor, dape, dcompressor_norm, None, None, None)
        )


def attention_core(
    visible_op,
    q,
    kv,
    workspace,
    indices,
    topk_length,
    sink,
    slots,
    positions,
    cos_sin_cache,
    *,
    softmax_scale: float,
    eps: float,
    rope_dim: int,
    backward_op=None,
    compressor_kv_score=None,
    compressor_ape=None,
    compressor_norm=None,
    compressor_ratio: int = 1,
    compressor_workspace_slots=None,
    query_start_loc=None,
):
    if compressor_kv_score is None:
        compressor_kv_score = q.new_empty((0, workspace.shape[-1]))
        compressor_ape = q.new_empty((0, workspace.shape[-1]))
        compressor_norm = q.new_empty((workspace.shape[-1],))
    if compressor_workspace_slots is None:
        compressed_len = (
            compressor_kv_score.shape[0] // compressor_ratio
            if compressor_ratio > 1
            else 0
        )
        # Backward compatibility for direct single-request callers: their
        # ``slots`` are SWA-local and the compressed prefix precedes them.
        # Production metadata always supplies absolute workspace coordinates.
        if compressed_len:
            slots = slots + compressed_len
        compressor_workspace_slots = torch.arange(
            compressed_len, dtype=torch.int64, device=q.device
        )
    if query_start_loc is None:
        query_start_loc = torch.tensor(
            [0, q.shape[0]], dtype=torch.int32, device=q.device
        )
    return _VLLMAttentionCoreFunction.apply(
        visible_op,
        backward_op,
        softmax_scale,
        eps,
        rope_dim,
        q,
        kv,
        workspace,
        indices,
        topk_length,
        sink,
        slots,
        positions,
        cos_sin_cache,
        compressor_kv_score,
        compressor_ape,
        compressor_norm,
        compressor_ratio,
        compressor_workspace_slots,
        query_start_loc,
    )


class _VisibleSparseAttentionFunction(torch.autograd.Function):
    """Differentiate an official sparse-attention visible invocation.

    Native CP constructs Q and the compact KV workspace as straight-through
    tensors whose forward values already match vLLM.  This owner therefore
    needs only the vendor sparse-attention VJP; communication, compaction and
    projections remain ordinary autograd edges outside the function.
    """

    @staticmethod
    def forward(ctx, visible_op, backward_op, scale, q, kv, indices, length, sink):
        result = visible_op(q.detach(), kv.detach())
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError("native CP FlashMLA must return output and lse")
        out = result[0]
        lse = result[2] if len(result) >= 3 else result[1]
        ctx.save_for_backward(q.detach(), kv.detach(), out, lse, indices, length, sink)
        ctx.backward_op = backward_op or _default_sparse_backward
        ctx.scale = scale
        return own_visible_tensor(out)

    @staticmethod
    def backward(ctx, grad_out):
        q, kv, out, lse, indices, length, sink = ctx.saved_tensors
        dq, dkv = ctx.backward_op(
            q, kv, out, grad_out, lse, sink, indices, ctx.scale, length
        )
        dkv = dkv.reshape_as(kv)
        if not bool(torch.isfinite(dq).all()) or not bool(torch.isfinite(dkv).all()):
            raise FloatingPointError("native CP sparse attention produced non-finite gradients")
        return None, None, None, dq, dkv, None, None, None


def visible_sparse_attention(
    visible_op,
    q,
    kv,
    indices,
    topk_length,
    sink,
    *,
    softmax_scale: float,
    backward_op=None,
):
    return _VisibleSparseAttentionFunction.apply(
        visible_op,
        backward_op,
        softmax_scale,
        q,
        kv,
        indices,
        topk_length,
        sink,
    )
