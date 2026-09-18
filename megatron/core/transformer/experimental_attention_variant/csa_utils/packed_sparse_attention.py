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


def batch_of_row(cu_seqlens_q: Tensor, total_q: Optional[int] = None) -> Tensor:
    """For a THD-packed query of length ``total_q``, return a ``(total_q,)``
    int64 tensor where entry ``i`` is the index of the segment that owns
    query row ``i`` (i.e. the unique ``b`` with
    ``cu_seqlens_q[b] <= i < cu_seqlens_q[b+1]``).

    When ``total_q`` exceeds ``cu_seqlens_q[-1]`` (e.g. after
    ``pad_thd_for_cuda_graph`` pads token tensors to a static capacity),
    orphan rows are clamped to the last segment so the returned indices
    are always in ``[0, B-1]`` and never cause OOB on per-segment arrays.

    Used by every helper that needs to translate between per-row indices
    and per-segment cumulative tensors.

    Args:
        cu_seqlens_q: ``(B+1,)`` int — cumulative Q lengths.
        total_q: optional row count override; defaults to
            ``int(cu_seqlens_q[-1].item())`` (forces a GPU→CPU sync).

    Returns:
        ``(total_q,)`` int64.
    """
    if total_q is None:
        total_q = int(cu_seqlens_q[-1].item())
    num_sequences = cu_seqlens_q.shape[0] - 1
    row_idx = torch.arange(total_q, device=cu_seqlens_q.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens_q[1:], right=True).clamp(
        max=max(num_sequences - 1, 0)
    )


@torch.no_grad()
def _compute_full_csa_teacher_lse(
    query: Tensor,
    query_flat: Tensor,
    full_kv_flat: Tensor,
    compressed_kv: Tensor,
    attn_sink: Tensor,
    window_indices: Tensor,
    softmax_scale: float,
    ratio: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    q_causal_offsets: Optional[Tensor] = None,
) -> Tensor:
    """Stream the full packed teacher denominator through the requested Triton backend."""
    return fused_csa_teacher_lse(
        query_flat,
        full_kv_flat,
        compressed_kv,
        attn_sink,
        window_indices,
        softmax_scale,
        ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_kv,
        q_causal_offsets=q_causal_offsets,
    )


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
    is_thd=True,
    kv_reconstruction_parts=None,
    q_padding_mask=None,
):
    """Run fused attention for flat packed Q/KV and physical indices."""
    if not is_thd or query.ndim != 3 or kv.ndim != 2:
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
    *,
    topk_indices_global: bool = False,
) -> Tensor:
    """Compute ``target`` distribution (L1-normalised head-sum softmax).

    Wraps :attr:`cudnn.DSA.sparse_attn_score_recompute_wrapper`. Same
    layout convention as :func:`_compute_indexer_predict`: 4-D q is
    BSHD; 3-D q is THD and gets fake-BSHD'd with ``B=1`` before the
    wrapper call (so the 4-D-Q shape check passes).
    """
    _ensure_dsa_namespace()
    is_thd = q_attn.ndim == 3
    if is_thd:
        if not topk_indices_global:
            raise ValueError(
                "THD ``_compute_attn_target`` requires "
                "``topk_indices_global=True`` so the kernel addresses K "
                "by flat ids over the packed ``(total_k, D)`` buffer."
            )
        q_bshd, k_bsd, lse_bsh, topk_bst = _thd_to_fake_bshd(q_attn, k_attn, lse, topk_indices)
    else:
        q_bshd, k_bsd, lse_bsh, topk_bst = q_attn, k_attn, lse, topk_indices

    result = _DSA.sparse_attn_score_recompute_wrapper(
        q_bshd,
        k_bsd,
        lse_bsh,
        topk_bst,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        topk_indices_global=topk_indices_global,
    )
    target = result["target"]
    if is_thd:
        target = target.squeeze(0)
    return target


def _kl_loss_from_target_predict(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
    loss_divisor: int | float | Tensor | None = None,
) -> Tensor:
    """KL(target || predict) reduced over ``(B, S_q)`` and scaled by loss_coeff.

    Rows with no valid top-K positions (early query rows with ratio causal
    masking) contribute 0 to the loss — the sparse score kernels produce
    garbage for those rows, mirroring ``compute_dsa_indexer_loss``'s
    ``row_valid`` handling. The default mean is taken over all ``(B, S_q)``
    positions. Per-token-loss mode returns a raw local sum unless
    ``loss_divisor`` is supplied, in which case the global normalization is
    folded into the same compiled reduction.
    """
    return csa_indexer_loss_kernels.sparse_kl_loss(
        target, predict, topk_indices, loss_coeff, calculate_per_token_loss, loss_divisor
    )


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


def _compute_dense_indexer_score(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    qhead_per_kv_head: int,
    indexer_softmax_scale: float,
    ratio: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    q_causal_offsets: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Dense indexer score forward over the full ``S_k`` axis (BSHD or THD).

    Wraps :attr:`cudnn.DSA.dense_indexer_score_recompute_wrapper`.
    Layout is selected by ``cu_seqlens_*`` kwargs:

    * **BSHD** (``cu_seqlens_*=None``): inputs are 4-D q ``(B, S_q, H, D)``,
      4-D k ``(B, S_k, H_kv, D)``, 3-D w ``(B, S_q, H)``. Outputs are
      ``out (B, S_q, S_k)`` + ``denom (B, S_q)``.
    * **THD** (``cu_seqlens_*`` supplied): inputs are 3-D q
      ``(total_q, H, D)``, 3-D k ``(total_k, H_kv, D)``, 2-D w
      ``(total_q, H)``. Outputs are ``out (total_q, max_seqlen_kv)`` +
      ``denom (total_q,)``.

    The ratio-causal limit is
    ``min(S_k, (q_causal_offset + q + 1) // ratio)``; omitted offsets are zero.
    """
    _ensure_dsa_namespace()
    kwargs = dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_kv,
    )
    if q_causal_offsets is not None:
        kwargs["q_causal_offsets"] = q_causal_offsets
    result = _DSA.dense_indexer_score_recompute_wrapper(
        q_indexer,
        k_indexer,
        weights,
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=indexer_softmax_scale,
        ratio=ratio,
        **kwargs,
    )
    return result["out"], result["denom"]


def _compute_dense_attn_score(
    q_attn: Tensor,
    k_attn: Tensor,
    lse: Tensor,
    qhead_per_kv_head: int,
    softmax_scale: float,
    ratio: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    q_causal_offsets: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Dense attention score forward over the full ``S_k`` axis (BSHD or THD).

    Wraps :attr:`cudnn.DSA.dense_attn_score_recompute_wrapper`. Same
    BSHD/THD layout convention as :func:`_compute_dense_indexer_score`.
    """
    _ensure_dsa_namespace()
    kwargs = dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_kv,
    )
    if q_causal_offsets is not None:
        kwargs["q_causal_offsets"] = q_causal_offsets
    result = _DSA.dense_attn_score_recompute_wrapper(
        q_attn,
        k_attn,
        lse,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        **kwargs,
    )
    return result["out"], result["denom"]


def _kl_loss_from_dense_scores(
    attn_score: Tensor,
    attn_l1norm: Tensor,
    index_score: Tensor,
    index_lse: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL(target || predict) over the **full** KV axis, averaged over rows.

    Derives ``target = attn_score / attn_l1norm`` (L1-normalised, matches
    ``compute_dsa_indexer_loss``'s ``attention_scores / sum`` step) and
    ``log_predict = index_score - index_lse`` (LSE-normalised log-softmax),
    then computes ``KL = sum_k target * (log target - log predict)`` and
    scales by ``loss_coeff``.

    Layout-agnostic: works for both BSHD inputs (shapes
    ``attn_score (B, S_q, S_k)``, ``attn_l1norm (B, S_q)``, …) and THD
    inputs (shapes ``attn_score (total_q, max_seqlen_kv)``,
    ``attn_l1norm (total_q,)``, …). The final ``.mean()`` averages over
    all rows in either case.

    Rows where the kernel's ``ratio`` causal mask leaves no valid KV
    position have ``attn_l1norm <= 0`` (L1) or ``index_lse == -inf``
    (LSE); those rows contribute 0 to the loss — the same ``row_valid``
    semantics as the reference ``compute_dsa_indexer_loss``.
    """
    return csa_indexer_loss_kernels.dense_kl_loss(
        attn_score, attn_l1norm, index_score, index_lse, loss_coeff, calculate_per_token_loss
    )


def _dense_indexer_loss_and_grads(
    query,
    kv_full,
    compressed_kv,
    attn_sink,
    window_indices,
    q_indexer,
    k_indexer,
    weights,
    softmax_scale,
    indexer_softmax_scale,
    loss_coeff,
    loss_divisor,
    ratio,
    max_seqlen_q,
    indexer_layout,
    q_padding_mask,
):
    """Compute full-key teacher loss in bounded query slabs using the existing cuDNN kernels.

    cuDNN normalizes each backward by its query count. Scaling its coefficient by
    chunk_rows / total_q preserves the original full-query gradient convention;
    the caller then applies the real-token/global-CP divisor once. Accumulate K
    gradients in FP32 across slabs and round only after the final accumulation.
    """
    total_q = q_indexer.shape[0]
    cu_q, cu_k, causal_offsets = indexer_layout
    max_k = max_seqlen_q // ratio
    torch._assert_async(
        (((cu_q[1:] - cu_q[:-1]) == 0) | ((cu_k[1:] - cu_k[:-1]) > 0)).all(),
        "cuDNN dense packed indexer loss requires a compressed key in each nonempty Q "
        "segment; use sparse loss or dsa_kernel_backend='none' for shorter documents.",
    )
    # Budget for score/teacher buffers and frontend temporaries; KL is fused on CUDA.
    chunk_rows = packed_layout.query_chunk_rows(total_q, max_k, live_buffers=4)
    grad_q = torch.empty_like(q_indexer)
    grad_weights = torch.empty_like(weights)
    grad_k = torch.zeros_like(k_indexer, dtype=torch.float32)
    chunk_grad_k = torch.empty_like(grad_k)
    loss = query.new_zeros((), dtype=torch.float32)
    unit_grad_loss = query.new_ones((), dtype=torch.float32)
    for start in range(0, total_q, chunk_rows):
        end = min(start + chunk_rows, total_q)
        if chunk_rows >= total_q:
            local_cu_q, offsets = cu_q, causal_offsets
        else:
            local_cu_q, offsets = packed_layout.slice_query_layout(cu_q, causal_offsets, start, end)
        metadata = dict(
            cu_seqlens_q=local_cu_q,
            cu_seqlens_kv=cu_k,
            max_seqlen_q=min(max_seqlen_q, end - start),
            max_seqlen_kv=max_k,
            q_causal_offsets=offsets,
        )
        index_score, index_lse = _compute_dense_indexer_score(
            q_indexer[start:end],
            k_indexer.unsqueeze(1),
            weights[start:end],
            qhead_per_kv_head=q_indexer.shape[1],
            indexer_softmax_scale=indexer_softmax_scale,
            ratio=ratio,
            **metadata,
        )
        mask = None if q_padding_mask is None else q_padding_mask[start:end]
        if mask is not None:
            index_score.masked_fill_(mask.unsqueeze(-1), float("-inf"))
            index_lse.masked_fill_(mask, float("-inf"))
        teacher_lse = _compute_full_csa_teacher_lse(
            query[start:end],
            query[start:end],
            kv_full,
            compressed_kv,
            attn_sink,
            window_indices[start:end],
            softmax_scale,
            ratio,
            **metadata,
        )
        attn_score, attn_l1norm = _compute_dense_attn_score(
            query[start:end],
            compressed_kv.unsqueeze(1),
            teacher_lse,
            qhead_per_kv_head=query.shape[1],
            softmax_scale=softmax_scale,
            ratio=ratio,
            **metadata,
        )
        if mask is not None:
            attn_score.masked_fill_(mask.unsqueeze(-1), 0)
            attn_l1norm.masked_fill_(mask, 0)
        loss.add_(
            _kl_loss_from_dense_scores(
                attn_score,
                attn_l1norm,
                index_score,
                index_lse,
                loss_coeff,
                calculate_per_token_loss=True,
            )
        )
        # Backward owns and overwrites both score buffers; the loss has consumed them.
        if mask is not None:
            index_score.masked_fill_(mask.unsqueeze(-1), 0)
            index_lse.masked_fill_(mask, 0)
        # The frontend overwrites (and zero-initializes) the supplied FP32 K buffer.
        _DSA.dense_indexer_backward_wrapper(
            q_indexer[start:end],
            weights[start:end],
            k_indexer,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            sm_scale=indexer_softmax_scale,
            loss_coeff=loss_coeff * ((end - start) / total_q),
            grad_loss=unit_grad_loss,
            block_I=128,
            ratio=ratio,
            cu_seqlens_q=local_cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=min(max_seqlen_q, end - start),
            max_seqlen_k=max_k,
            q_causal_offsets=offsets,
            d_index_q=grad_q[start:end],
            d_weights=grad_weights[start:end],
            d_index_k=chunk_grad_k,
        )
        grad_k.add_(chunk_grad_k)
        del index_score, index_lse, attn_score, attn_l1norm, teacher_lse
    return loss / loss_divisor, {
        "d_index_q": grad_q,
        "d_weights": grad_weights,
        "d_index_k": grad_k.to(k_indexer.dtype),
    }


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
        q_padding_mask: Optional[Tensor] = None,
        local_k_indexer: Optional[Tensor] = None,
        local_compressed_kv: Optional[Tensor] = None,
        cp_group=None,
        compressed_kv_start: int = 0,
        indexer_rank_map: Optional[Tensor] = None,
        indexer_k_reduce_scatter_state: Optional[_DeferredReduceScatterState] = None,
        compressed_kv_reduce_scatter_state: Optional[_DeferredReduceScatterState] = None,
        logical_window_width: int | None = None,
        kv_reconstruction_parts: Tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run fused sparse attention using caller-supplied top-k indices."""
        _ensure_dsa_namespace()

        total_q, np_ = query.shape[:2]
        idx_nh, idx_hd = q_indexer.shape[1], q_indexer.shape[2]
        total_comp = k_indexer.shape[0]
        indexer_topk = indexer_topk_idxs.shape[-1]

        # Preserve the fixed window suffix for the dense teacher before
        # compacting the complete attention index set.
        if logical_window_width is None:
            logical_window_width = topk_idxs.shape[-1] - indexer_topk
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
        bwd_loss_coeff = loss_coeff
        real_row_scale = (
            total_q / loss_divisor.clamp_min(1)
            if torch.is_tensor(loss_divisor)
            else total_q / max(loss_divisor, 1)
        )
        unit_grad_loss = torch.ones((), device=query.device, dtype=torch.float32)

        indexer_loss = query.new_zeros((), dtype=torch.float32)
        if loss_coeff > 0:
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
                    q_bshd,
                    k_bsd,
                    w_bsh,
                    topk_bst,
                    qhead_per_kv_head=idx_nh,
                    topk_indices_global=True,
                )["predict"].squeeze(0)
                target = _compute_attn_target(
                    query.detach(),
                    compressed_kv.detach(),
                    torch.logaddexp(lse.detach().float(), attn_sink.detach().float().view(1, np_)),
                    indexer_topk_idxs_for_loss,
                    softmax_scale,
                    qhead_per_kv_head=np_,
                    topk_indices_global=True,
                )
                indexer_loss = _kl_loss_from_target_predict(
                    target,
                    predict,
                    indexer_topk_idxs_for_loss,
                    loss_coeff,
                    calculate_per_token_loss=True,
                    loss_divisor=loss_divisor,
                )
                if loss_coeff > 0:
                    ig = _DSA.indexer_backward_wrapper(
                        q_indexer.view(1, total_q, idx_nh, idx_hd),
                        weights.view(1, total_q, idx_nh),
                        k_indexer.view(1, total_comp, idx_hd),
                        target.view(1, total_q, indexer_topk),
                        predict.view(1, total_q, indexer_topk),
                        indexer_topk_idxs_for_loss.view(1, total_q, indexer_topk),
                        sm_scale=indexer_softmax_scale,
                        loss_coeff=bwd_loss_coeff,
                        grad_loss=unit_grad_loss,
                        block_I=128,
                    )
            else:
                indexer_loss, ig = _dense_indexer_loss_and_grads(
                    query,
                    kv_full,
                    compressed_kv,
                    attn_sink,
                    window_topk_idxs,
                    q_indexer,
                    k_indexer,
                    weights,
                    softmax_scale,
                    indexer_softmax_scale,
                    loss_coeff,
                    loss_divisor,
                    ratio,
                    max_seqlen_q,
                    indexer_layout,
                    q_padding_mask,
                )
        if loss_coeff > 0:
            saved_grad_q_indexer = ig["d_index_q"].view(total_q, idx_nh, idx_hd) * real_row_scale
            saved_grad_k_indexer = ig["d_index_k"].view(total_comp, idx_hd) * real_row_scale
            saved_grad_weights = ig["d_weights"].view(total_q, idx_nh) * real_row_scale
            if q_padding_mask is not None:
                saved_grad_q_indexer[q_padding_mask] = 0
                saved_grad_weights[q_padding_mask] = 0
        else:
            saved_grad_q_indexer = torch.zeros_like(q_indexer)
            saved_grad_k_indexer = torch.zeros_like(k_indexer)
            saved_grad_weights = torch.zeros_like(weights)

        if cp_group is not None:
            if local_k_indexer is None or local_compressed_kv is None:
                raise RuntimeError("CP backward overlap requires both local compressed tensors.")
            ctx.local_k_indexer_rows = local_k_indexer.shape[0]
            ctx.local_compressed_kv_rows = local_compressed_kv.shape[0]
        if indexer_rank_map is None:
            indexer_rank_map = torch.empty(0, dtype=torch.int32, device=query.device)

        ctx.cp_group = cp_group
        ctx.compressed_kv_start = int(compressed_kv_start)
        ctx.indexer_k_reduce_scatter_state = indexer_k_reduce_scatter_state
        ctx.compressed_kv_reduce_scatter_state = compressed_kv_reduce_scatter_state
        ctx.indexer_grad_is_sequence_major = cp_group is not None and not sparse_loss
        ctx.num_forward_inputs = len(ctx.needs_input_grad)
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
        if getattr(ctx, "reconstruct_kv_for_backward", False):
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
        grad_k_indexer_rank_major = None
        if cp_group is not None:
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

        grad_local_k_indexer = None
        grad_local_compressed_kv = None
        if cp_group is not None:
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

            if ctx.indexer_k_reduce_scatter_state is not None:
                ctx.indexer_k_reduce_scatter_state.handle = indexer_reduce_scatter
                grad_local_k_indexer = indexer_reduce_scatter.tensor
            if ctx.compressed_kv_reduce_scatter_state is not None:
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

        if cp_group is not None:
            if ctx.indexer_k_reduce_scatter_state is None:
                grad_local_k_indexer = indexer_reduce_scatter.wait()
            if ctx.compressed_kv_reduce_scatter_state is None:
                grad_local_compressed_kv = compressed_kv_reduce_scatter.wait()
            grad_k_indexer = None

        gradients = (
            attn_bwd["dq"],
            attn_bwd["dkv"],
            attn_bwd["d_sink"],
            None,
            grad_q_indexer,
            grad_k_indexer,
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
        # Older call sites omit the optional CP-overlap inputs. PyTorch expects
        # exactly one backward result for every argument passed to ``apply``.
        return gradients[: ctx.num_forward_inputs]


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
    output = torch.empty((q.shape[0], topk), device=q.device, dtype=torch.int32)
    valid_counts = torch.empty(q.shape[0], device=q.device, dtype=torch.int32)
    chunk_rows = packed_layout.query_chunk_rows(q.shape[0], max_seqlen_kv)
    for start in range(0, q.shape[0], chunk_rows):
        end = min(start + chunk_rows, q.shape[0])
        if chunk_rows >= q.shape[0]:
            cu_q, offsets = cu_seqlens_q, q_causal_offsets
        else:
            cu_q, offsets = packed_layout.slice_query_layout(
                cu_seqlens_q, q_causal_offsets, start, end
            )
        scores = _DSA.indexer_forward_wrapper(
            q[start:end],
            k.unsqueeze(1),
            w[start:end],
            ratio=ratio,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=min(max_seqlen_q, end - start),
            max_seqlen_k=max_seqlen_kv,
            q_causal_offsets=offsets,
        )["scores"].contiguous()
        lengths = packed_layout.build_seq_lens(cu_q, cu_seqlens_kv, end - start, ratio, offsets)
        candidates = _DSA.indexer_top_k_wrapper(
            scores, lengths, top_k=min(topk, max_seqlen_kv), next_n=1, return_val=False
        )["indices"]
        ids, counts = packed_layout.sanitize_topk(candidates, scores, lengths, output_width=topk)
        output[start:end] = ids
        valid_counts[start:end] = counts
        # Do not retain the previous slab while allocating the next one.
        del scores, candidates, ids, counts
    return output, valid_counts
