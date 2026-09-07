from __future__ import annotations

import os

import torch

_HOT_PATH_ASSERTS = os.getenv("MLITE_VLLM_HOT_PATH_ASSERTS") == "1"

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

    dsa_kernels._ensure_dsa_namespace()
    # cuDNN backward consumes flattened KV, unlike vLLM's request-major workspace.
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
    return result["dq"], result["dkv"], result["d_sink"]


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
    """Differentiate IndexShare scores on the visible fixed active set."""
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


class _VisibleSparseAttentionFunction(torch.autograd.Function):
    """Attach the vendor VJP to the official sparse-attention value."""

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
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, kv, out, lse, indices, length, sink = ctx.saved_tensors
        dq, dkv, d_sink = ctx.backward_op(
            q, kv, out, grad_out, lse, sink, indices, ctx.scale, length
        )
        dkv = dkv.reshape_as(kv)
        if _HOT_PATH_ASSERTS:
            torch._assert_async(
                torch.isfinite(dq).all(),
                "native CP sparse attention produced non-finite dq",
            )
            torch._assert_async(
                torch.isfinite(dkv).all(),
                "native CP sparse attention produced non-finite dkv",
            )
        return None, None, None, dq, dkv, None, None, d_sink


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
    if not torch.is_grad_enabled():
        result = visible_op(q, kv)
        if not isinstance(result, (tuple, list)) or not result:
            raise RuntimeError("visible sparse attention must return an output tensor")
        return result[0]
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
