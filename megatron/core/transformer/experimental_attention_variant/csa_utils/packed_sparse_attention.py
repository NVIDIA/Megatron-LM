# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Packed CSA attention, indexer loss and overlapped CP backward.

The caller supplies final physical indices and explicit process groups. SBHD
continues to use fused_sparse_attention; both paths share the FlashMLA adapter.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.tensor_parallel.mappings import async_reduce_scatter_along_first_dim
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

from . import csa_indexer_loss_kernels
from . import fused_sparse_attention as sbhd
from . import packed_layout
from .csa_teacher_lse import fused_csa_teacher_lse
from .fused_sparse_attention import (
    _compact_flat_topk_idxs,
    _csa_fwd_flash_mla,
    _csa_sparse_attention_backward,
)
from .fused_sparse_attention import _get_topk_alignment as get_flash_mla_topk_alignment
from .fused_sparse_attention import _kl_loss_from_dense_scores

_DSA: Any = None


def _ensure_dsa_namespace():
    global _DSA
    sbhd._ensure_dsa_namespace()
    _DSA = sbhd._DSA


class _DeferredReduceScatterState:
    """Carry one asynchronous reduce-scatter into its gradient consumer."""

    def __init__(self, wait_range: str):
        self.handle = None
        self.wait_range = wait_range


class _WaitForDeferredReduceScatter(torch.autograd.Function):
    """Wait only on the autograd branch that consumes the reduced gradient."""

    @staticmethod
    def forward(ctx, input_: Tensor, state: _DeferredReduceScatterState) -> Tensor:
        """Preserve the input while attaching the deferred collective state."""
        ctx.state = state
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, _grad_output: Tensor):
        """Wait for the collective and return its reduced gradient."""
        handle = ctx.state.handle
        if handle is None:
            raise RuntimeError("Deferred reduce-scatter was not launched before consumption")
        nvtx_range_push(ctx.state.wait_range)
        try:
            reduced_gradient = handle.wait()
        finally:
            ctx.state.handle = None
            nvtx_range_pop(ctx.state.wait_range)
        return reduced_gradient, None


def defer_reduce_scatter_wait(
    input_: Tensor, wait_range: str = "dsv4_cp_reduce_scatter_consumer_wait"
):
    """Return a gradient edge whose backward waits for an attached collective."""
    state = _DeferredReduceScatterState(wait_range)
    return _WaitForDeferredReduceScatter.apply(input_, state), state


def _validate_kv_reconstruction_parts(
    kv: Tensor, kv_reconstruction_parts: Tuple[Tensor, Tensor, Tensor]
) -> None:
    """Validate tensors used to rebuild a flat THD KV buffer in backward."""
    if len(kv_reconstruction_parts) != 3:
        raise ValueError(
            "kv_reconstruction_parts must contain boundary, local, and compressed KV tensors"
        )

    part_names = ("boundary", "local", "compressed")
    for name, part in zip(part_names, kv_reconstruction_parts):
        if not isinstance(part, Tensor):
            raise TypeError(f"{name} KV reconstruction part must be a torch.Tensor")
        if part.device != kv.device or part.dtype != kv.dtype:
            raise ValueError(
                f"{name} KV reconstruction part must match kv device and dtype; "
                f"got {part.device}/{part.dtype} and {kv.device}/{kv.dtype}"
            )
        if part.ndim != kv.ndim or part.shape[1:] != kv.shape[1:]:
            raise ValueError(
                f"{name} KV reconstruction part has incompatible shape {tuple(part.shape)} "
                f"for kv shape {tuple(kv.shape)}"
            )

    reconstructed_rows = sum(part.shape[0] for part in kv_reconstruction_parts)
    if reconstructed_rows != kv.shape[0]:
        raise ValueError(
            "KV reconstruction parts have an unexpected total row count: "
            f"got {reconstructed_rows}, expected {kv.shape[0]}"
        )


class CSASparseAttnFunc(torch.autograd.Function):
    """Sparse attention fwd + bwd on flat tensors.

    Forward uses :mod:`flash_mla`; backward uses cuDNN Frontend's
    :attr:`cudnn.DSA.sparse_attention_backward_wrapper`.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,  # (total_sq, H, D) bf16
        kv: Tensor,  # (total_skv, D) bf16
        attn_sink: Tensor,  # (H,) f32
        topk_idxs: Tensor,  # (total_sq, TopK) int32 global
        topk_length: Optional[Tensor],  # (total_sq,) int32 or None
        softmax_scale: float,
        indexer_topk: int,
        kv_reconstruction_parts: Tuple[Tensor, Tensor, Tensor] | None = None,
        q_padding_mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Run FlashMLA sparse-attention forward and save tensors for backward."""
        topk_idxs = torch.nn.functional.pad(
            topk_idxs, (0, -topk_idxs.shape[-1] % get_flash_mla_topk_alignment()), value=-1
        )
        out, lse, lse_indexer = _csa_fwd_flash_mla(
            q,
            kv,
            topk_idxs,
            softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            indexer_topk=indexer_topk,
        )

        if topk_length is not None:
            topk_idxs = topk_idxs.clamp_min(0)
            topk_length = topk_length.clone()
            if q_padding_mask is not None:
                topk_length.masked_fill_(q_padding_mask, 1)
        ctx.q_padding_mask = q_padding_mask
        ctx.reconstruct_kv_for_backward = kv_reconstruction_parts is not None
        if ctx.reconstruct_kv_for_backward:
            assert kv_reconstruction_parts is not None
            _validate_kv_reconstruction_parts(kv, kv_reconstruction_parts)
            ctx.save_for_backward(q, *kv_reconstruction_parts, attn_sink, topk_idxs, out, lse)
        else:
            ctx.save_for_backward(q, kv, attn_sink, topk_idxs, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.topk_length = topk_length
        return out, lse, lse_indexer

    @staticmethod
    def backward(ctx, dO, d_lse, d_lse_indexer):
        """Compute sparse-attention backward via cuDNN DSA wrapper."""
        _ensure_dsa_namespace()

        if ctx.reconstruct_kv_for_backward:
            q, boundary_kv, local_kv, compressed_kv, attn_sink, topk_idxs, out, lse = (
                ctx.saved_tensors
            )
            kv = torch.cat((boundary_kv, local_kv, compressed_kv), dim=0)
        else:
            q, kv, attn_sink, topk_idxs, out, lse = ctx.saved_tensors

        if ctx.q_padding_mask is not None:
            dO = dO.masked_fill(ctx.q_padding_mask[:, None, None], 0)
            lse = lse.masked_fill(ctx.q_padding_mask[:, None], 0)
        dq, dkv, d_sink = _csa_sparse_attention_backward(
            q,
            kv,
            out,
            dO,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=ctx.topk_length,
        )
        return dq, dkv, d_sink, None, None, None, None, None, None


def csa_sparse_attn(
    query,
    kv,
    attn_sink,
    topk_idxs,
    softmax_scale,
    topk_length=None,
    kv_reconstruction_parts=None,
    q_padding_mask=None,
):
    """Run fused attention for flat packed Q/KV and physical indices."""
    if query.ndim != 3 or kv.ndim != 2:
        raise ValueError("Packed CSA requires query [tokens, heads, dim] and KV [tokens, dim].")
    if topk_length is not None:
        # Short windows can leave holes before valid compressed keys. The backend
        # interprets topk_length as a valid prefix, so compact those holes first.
        topk_idxs, topk_length = _compact_flat_topk_idxs(topk_idxs)
    out, _, _ = CSASparseAttnFunc.apply(
        query,
        kv,
        attn_sink,
        topk_idxs,
        topk_length,
        softmax_scale,
        0,
        kv_reconstruction_parts,
        q_padding_mask,
    )
    return out.flatten(1)


def _thd_to_fake_bshd(*tensors: Tensor) -> Tuple[Tensor, ...]:
    """Prepend a B=1 dim to THD tensors for cuDNN wrappers that expect BSHD."""
    return tuple(t.unsqueeze(0) for t in tensors)


def _compute_attn_target(
    q_attn: Tensor,
    k_attn: Tensor,
    lse: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    qhead_per_kv_head: int,
) -> Tensor:
    """Compute packed teacher targets using global compressed-key indices."""
    _ensure_dsa_namespace()
    q_bshd, k_bsd, lse_bsh, topk_bst = _thd_to_fake_bshd(q_attn, k_attn, lse, topk_indices)
    return _DSA.sparse_attn_score_recompute_wrapper(
        q_bshd,
        k_bsd,
        lse_bsh,
        topk_bst,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        topk_indices_global=True,
    )["target"].squeeze(0)


def _scale_indexer_grads(grad_loss: Tensor, *grads: Tensor) -> Tuple[Tensor, ...]:
    """Scale independent indexer gradients with one foreach launch per dtype group."""
    if not grads:
        return ()

    grouped_grads: dict[
        tuple[torch.device, torch.dtype, torch.layout], list[tuple[int, Tensor]]
    ] = {}
    for index, grad in enumerate(grads):
        grouped_grads.setdefault((grad.device, grad.dtype, grad.layout), []).append((index, grad))

    scaled_by_index: dict[int, Tensor] = {}
    for indexed_grads in grouped_grads.values():
        scaled_group = torch._foreach_mul([grad for _, grad in indexed_grads], grad_loss)
        for (index, _), scaled_grad in zip(indexed_grads, scaled_group):
            scaled_by_index[index] = scaled_grad
    return tuple(scaled_by_index[index] for index in range(len(grads)))


class FusedCSAIndexerSparseAttnFromTopkFunc(torch.autograd.Function):
    """Sparse attention with caller-supplied indexer top-k.

    The caller owns top-k selection. Sparse attention and indexer-loss
    backward still use FlashMLA / cuDNN DSA wrappers.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        kv_full: Tensor,
        attn_sink: Tensor,
        topk_idxs: Tensor,
        q_indexer: Tensor,
        k_indexer: Tensor,
        weights: Tensor,
        indexer_topk_idxs: Tensor,
        compressed_kv: Tensor,
        softmax_scale: float,
        indexer_softmax_scale: float,
        loss_coeff: float,
        loss_divisor: float | Tensor,
        sparse_loss: bool,
        ratio: int,
        max_seqlen_q: int,
        indexer_layout: Tuple[Tensor, Tensor, Tensor],
        q_padding_mask: Tensor | None,
        local_k_indexer: Tensor,
        local_compressed_kv: Tensor,
        cp_group: torch.distributed.ProcessGroup,
        compressed_kv_start: int,
        indexer_rank_map: Tensor | None,
        indexer_k_reduce_scatter_state: _DeferredReduceScatterState,
        compressed_kv_reduce_scatter_state: _DeferredReduceScatterState,
        logical_window_width: int,
        kv_reconstruction_parts: Tuple[Tensor, Tensor, Tensor] | None,
    ) -> Tuple[Tensor, Tensor]:
        """Run packed attention with positive indexer loss and deferred CP reductions."""
        if loss_coeff <= 0:
            raise ValueError("Use csa_sparse_attn when indexer loss is disabled.")
        _ensure_dsa_namespace()

        total_q, np_ = query.shape[:2]
        idx_nh, idx_hd = q_indexer.shape[1], q_indexer.shape[2]
        total_comp = k_indexer.shape[0]
        indexer_topk = indexer_topk_idxs.shape[-1]

        # Preserve the fixed window suffix for the dense teacher before
        # compacting the complete attention index set.
        window_topk_idxs = topk_idxs[:, indexer_topk : indexer_topk + int(logical_window_width)]
        topk_idxs, topk_length = _compact_flat_topk_idxs(topk_idxs)

        # Do not request FlashMLA's partial indexer LSE: it omits both the
        # window and sink masses required by the CSA teacher.
        out_flat, lse, _ = _csa_fwd_flash_mla(
            query,
            kv_full,
            topk_idxs,
            softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            indexer_topk=0,
        )
        topk_idxs.clamp_min_(0)
        if q_padding_mask is not None:
            # Keep padded sink-only rows out of cuDNN DSA's zero-tile path.
            # Backward masks their dO and LSE before using this placeholder.
            topk_length.masked_fill_(q_padding_mask, 1)

        # cuDNN accepts a host scalar coefficient; normalize by real packed
        # rows on device after the kernel, without synchronizing padded lengths.
        real_row_scale = (
            total_q / loss_divisor.clamp_min(1)
            if torch.is_tensor(loss_divisor)
            else total_q / max(loss_divisor, 1)
        )
        unit_grad_loss = torch.ones((), device=query.device, dtype=torch.float32)

        if sparse_loss:
            indexer_topk_idxs_for_loss = indexer_topk_idxs
            if q_padding_mask is not None:
                indexer_topk_idxs_for_loss = indexer_topk_idxs.masked_fill(
                    q_padding_mask.unsqueeze(-1), -1
                )
            weights_scaled = weights
            if indexer_softmax_scale != 1.0:
                weights_scaled = (weights.float() * indexer_softmax_scale).to(weights.dtype)
            q_bshd, k_bsd, w_bsh, topk_bst = _thd_to_fake_bshd(
                q_indexer, k_indexer, weights_scaled, indexer_topk_idxs_for_loss
            )
            predict = _DSA.sparse_indexer_score_recompute_wrapper(
                q_bshd, k_bsd, w_bsh, topk_bst, qhead_per_kv_head=idx_nh, topk_indices_global=True
            )["predict"].squeeze(0)
            target = _compute_attn_target(
                query.detach(),
                compressed_kv.detach(),
                torch.logaddexp(lse.detach().float(), attn_sink.detach().float().view(1, np_)),
                indexer_topk_idxs_for_loss,
                softmax_scale,
                qhead_per_kv_head=np_,
            )
            indexer_loss = csa_indexer_loss_kernels.sparse_kl_loss(
                target,
                predict,
                indexer_topk_idxs_for_loss,
                loss_coeff,
                calculate_per_token_loss=True,
                loss_divisor=loss_divisor,
            )
            ig = _DSA.indexer_backward_wrapper(
                q_indexer.view(1, total_q, idx_nh, idx_hd),
                weights.view(1, total_q, idx_nh),
                k_indexer.view(1, total_comp, idx_hd),
                target.view(1, total_q, indexer_topk),
                predict.view(1, total_q, indexer_topk),
                indexer_topk_idxs_for_loss.view(1, total_q, indexer_topk),
                sm_scale=indexer_softmax_scale,
                loss_coeff=loss_coeff,
                grad_loss=unit_grad_loss,
                block_I=128,
            )
        else:
            cu_seqlens_q, cu_seqlens_k, q_causal_offsets = indexer_layout
            max_seqlen_k = max_seqlen_q // ratio
            torch._assert_async(
                (
                    ((cu_seqlens_q[1:] - cu_seqlens_q[:-1]) == 0)
                    | ((cu_seqlens_k[1:] - cu_seqlens_k[:-1]) > 0)
                ).all(),
                "cuDNN dense packed indexer loss requires a compressed key in each "
                "nonempty Q segment; use sparse loss for shorter documents.",
            )
            index_result = _DSA.dense_indexer_score_recompute_wrapper(
                q_indexer,
                k_indexer.unsqueeze(1),
                weights,
                qhead_per_kv_head=idx_nh,
                sm_scale=indexer_softmax_scale,
                ratio=ratio,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                q_causal_offsets=q_causal_offsets,
            )
            index_score, index_lse = index_result["out"], index_result["denom"]
            del index_result
            if q_padding_mask is not None:
                index_score = index_score.masked_fill(q_padding_mask.unsqueeze(-1), float("-inf"))
                index_lse = index_lse.masked_fill(q_padding_mask, float("-inf"))
            dense_teacher_lse = fused_csa_teacher_lse(
                query,
                kv_full,
                compressed_kv.detach(),
                attn_sink,
                window_topk_idxs,
                softmax_scale,
                ratio,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                q_causal_offsets=q_causal_offsets,
            )
            attn_result = _DSA.dense_attn_score_recompute_wrapper(
                query.detach(),
                compressed_kv.detach().unsqueeze(1),
                dense_teacher_lse,
                softmax_scale,
                qhead_per_kv_head=np_,
                ratio=ratio,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                q_causal_offsets=q_causal_offsets,
            )
            attn_score, attn_l1norm = attn_result["out"], attn_result["denom"]
            del attn_result
            if q_padding_mask is not None:
                attn_score = attn_score.masked_fill(q_padding_mask.unsqueeze(-1), 0)
                attn_l1norm = attn_l1norm.masked_fill(q_padding_mask, 0)
            raw_local_loss = _kl_loss_from_dense_scores(
                attn_score,
                attn_l1norm,
                index_score,
                index_lse,
                loss_coeff,
                calculate_per_token_loss=True,
            )
            indexer_loss = raw_local_loss / loss_divisor

            index_score_for_bwd = index_score.clone()
            index_lse_for_bwd = index_lse
            if q_padding_mask is not None:
                index_score_for_bwd[q_padding_mask] = 0
                index_lse_for_bwd = index_lse.masked_fill(q_padding_mask, 0)
            ig = _DSA.dense_indexer_backward_wrapper(
                q_indexer,
                weights,
                k_indexer,
                attn_score,
                attn_l1norm,
                index_score_for_bwd,
                index_lse_for_bwd,
                sm_scale=indexer_softmax_scale,
                loss_coeff=loss_coeff,
                grad_loss=unit_grad_loss,
                block_I=128,
                ratio=ratio,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                q_causal_offsets=q_causal_offsets,
            )
        saved_grad_q_indexer = ig["d_index_q"].view(total_q, idx_nh, idx_hd) * real_row_scale
        saved_grad_k_indexer = ig["d_index_k"].view(total_comp, idx_hd) * real_row_scale
        saved_grad_weights = ig["d_weights"].view(total_q, idx_nh) * real_row_scale
        if q_padding_mask is not None:
            saved_grad_q_indexer[q_padding_mask] = 0
            saved_grad_weights[q_padding_mask] = 0

        ctx.local_k_indexer_rows = local_k_indexer.shape[0]
        ctx.local_compressed_kv_rows = local_compressed_kv.shape[0]
        if indexer_rank_map is None:
            indexer_rank_map = torch.empty(0, dtype=torch.int32, device=query.device)

        ctx.cp_group = cp_group
        ctx.compressed_kv_start = int(compressed_kv_start)
        ctx.indexer_k_reduce_scatter_state = indexer_k_reduce_scatter_state
        ctx.compressed_kv_reduce_scatter_state = compressed_kv_reduce_scatter_state
        ctx.indexer_grad_is_sequence_major = not sparse_loss
        ctx.reconstruct_kv_for_backward = kv_reconstruction_parts is not None
        if ctx.reconstruct_kv_for_backward:
            assert kv_reconstruction_parts is not None
            _validate_kv_reconstruction_parts(kv_full, kv_reconstruction_parts)
            ctx.save_for_backward(
                query,
                *kv_reconstruction_parts,
                attn_sink,
                topk_idxs,
                topk_length,
                out_flat,
                lse,
                saved_grad_q_indexer,
                saved_grad_k_indexer,
                saved_grad_weights,
                indexer_rank_map,
            )
        else:
            ctx.save_for_backward(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                topk_length,
                out_flat,
                lse,
                saved_grad_q_indexer,
                saved_grad_k_indexer,
                saved_grad_weights,
                indexer_rank_map,
            )
        ctx.softmax_scale = softmax_scale
        ctx.q_padding_mask = q_padding_mask

        return out_flat.reshape(total_q, np_ * out_flat.shape[-1]), indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_loss):
        """Run sparse-attention and indexer-loss backward kernels."""
        _ensure_dsa_namespace()
        if ctx.reconstruct_kv_for_backward:
            (
                query,
                boundary_kv,
                local_kv,
                compressed_kv,
                attn_sink,
                topk_idxs,
                topk_length,
                out_flat,
                lse,
                saved_grad_q_indexer,
                saved_grad_k_indexer,
                saved_grad_weights,
                indexer_rank_map,
            ) = ctx.saved_tensors
            kv_full = torch.cat((boundary_kv, local_kv, compressed_kv), dim=0)
        else:
            (
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                topk_length,
                out_flat,
                lse,
                saved_grad_q_indexer,
                saved_grad_k_indexer,
                saved_grad_weights,
                indexer_rank_map,
            ) = ctx.saved_tensors

        cp_group = ctx.cp_group
        grad_k_indexer = saved_grad_k_indexer * grad_loss
        if ctx.indexer_grad_is_sequence_major:
            global_rows = ctx.local_k_indexer_rows * cp_group.size()
            grad_k_indexer_rank_major = grad_k_indexer.new_zeros(
                (global_rows, *grad_k_indexer.shape[1:])
            )
            valid_rows = indexer_rank_map >= 0
            rank_rows = indexer_rank_map.clamp_min(0).long()
            mask_shape = (valid_rows.shape[0],) + (1,) * (grad_k_indexer.ndim - 1)
            grad_k_indexer_rank_major.index_add_(
                0, rank_rows, grad_k_indexer * valid_rows.view(mask_shape)
            )
        else:
            grad_k_indexer_rank_major = grad_k_indexer

        dO_flat = grad_output.reshape(query.shape[0], query.shape[1], out_flat.shape[-1])
        if ctx.q_padding_mask is not None:
            dO_flat = dO_flat.masked_fill(ctx.q_padding_mask[:, None, None], 0)
            lse = lse.masked_fill(ctx.q_padding_mask[:, None], 0)
        nvtx_range_push("dsv4_cp_sparse_attention_backward")
        dq, dkv, d_sink = _csa_sparse_attention_backward(
            query,
            kv_full,
            out_flat,
            dO_flat,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=topk_length,
        )
        attn_bwd = {"dq": dq, "dkv": dkv, "d_sink": d_sink}
        nvtx_range_pop("dsv4_cp_sparse_attention_backward")

        expected_indexer_rows = ctx.local_k_indexer_rows * cp_group.size()
        if grad_k_indexer_rank_major.shape[0] != expected_indexer_rows:
            raise RuntimeError(
                "Indexer-K gradient has an unexpected CP-global shape: "
                f"got {grad_k_indexer_rank_major.shape[0]} rows, "
                f"expected {expected_indexer_rows}."
            )
        grad_compressed_kv = attn_bwd["dkv"][ctx.compressed_kv_start :]
        expected_rows = ctx.local_compressed_kv_rows * cp_group.size()
        if grad_compressed_kv.shape[0] != expected_rows:
            raise RuntimeError(
                "Compressed-KV gradient has an unexpected CP-global shape: "
                f"got {grad_compressed_kv.shape[0]} rows, expected {expected_rows}."
            )
        nvtx_range_push("dsv4_cp_attention_kv_reduce_scatter_launch")
        compressed_kv_reduce_scatter = async_reduce_scatter_along_first_dim(
            grad_compressed_kv, group=cp_group
        )
        nvtx_range_pop("dsv4_cp_attention_kv_reduce_scatter_launch")

        # Both reductions launch after the main sparse-attention backward,
        # avoiding its SM/L2 contention. Compressed-KV goes first because
        # its consumer branch is newer in autograd and runs first; Indexer-K
        # can then remain in flight during the attention compressor backward.
        nvtx_range_push("dsv4_cp_indexer_k_reduce_scatter_launch")
        indexer_reduce_scatter = async_reduce_scatter_along_first_dim(
            grad_k_indexer_rank_major, group=cp_group
        )
        nvtx_range_pop("dsv4_cp_indexer_k_reduce_scatter_launch")

        ctx.indexer_k_reduce_scatter_state.handle = indexer_reduce_scatter
        grad_local_k_indexer = indexer_reduce_scatter.tensor
        ctx.compressed_kv_reduce_scatter_state.handle = compressed_kv_reduce_scatter
        grad_local_compressed_kv = compressed_kv_reduce_scatter.tensor

        # These local branches do not consume either reduce-scatter result.
        # Queue them before either branch-local consumer wait. K stays in the
        # earlier scaling launch so its reduce-scatter is not delayed.
        nvtx_range_push("dsv4_cp_local_indexer_grads")
        grad_q_indexer, grad_weights = _scale_indexer_grads(
            grad_loss, saved_grad_q_indexer, saved_grad_weights
        )
        nvtx_range_pop("dsv4_cp_local_indexer_grads")

        return (
            attn_bwd["dq"],
            attn_bwd["dkv"],
            attn_bwd["d_sink"],
            None,
            grad_q_indexer,
            None,  # Global indexer-K gradients return through the local deferred edge.
            grad_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_local_k_indexer,
            grad_local_compressed_kv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def indexer_topk(
    q,
    k,
    weights,
    topk,
    ratio=4,
    indexer_softmax_scale=1.0,
    *,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    q_causal_offsets=None,
):
    """Select packed BF16 indexer keys with ratio-causal offsets and sanitized padding."""
    _ensure_dsa_namespace()
    w = (weights.float() * indexer_softmax_scale).to(weights.dtype)
    if max_seqlen_kv == 0 or k.shape[0] == 0:
        return torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32), torch.zeros(
            q.shape[0], device=q.device, dtype=torch.int32
        )
    # Score the CP-local query shard in one call, matching the CSA dense backend.
    # DSA's _indexer_topk_from_score_chunks / _indexer_top_k_wrapper_chunked
    # in dsa_cudnn_kernels.py (#5099) are references if profiling identifies this
    # workspace as an end-to-end peak and motivates ratio-aware query chunking.
    scores = _DSA.indexer_forward_wrapper(
        q,
        k.unsqueeze(1),
        w,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_kv,
        q_causal_offsets=q_causal_offsets,
    )["scores"].contiguous()
    lengths = packed_layout.build_seq_lens(
        cu_seqlens_q, cu_seqlens_kv, q.shape[0], ratio, q_causal_offsets
    )
    candidates = _DSA.indexer_top_k_wrapper(
        scores, lengths, top_k=min(topk, max_seqlen_kv), next_n=1, return_val=False
    )["indices"]
    return packed_layout.sanitize_topk(candidates, scores, lengths, output_width=topk)
